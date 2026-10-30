from __future__ import annotations

import base64
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from studio.exporter import EncryptedJsonlExporter, NoApprovedPairsError
from studio.models import CapturedDocument, DistilledPair
from studio.paths import StudioPaths
from studio.storage import StudioVault, VaultAuthenticationError


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def decrypt_bundle(bundle, passphrase: str) -> bytes:
    chunks: list[bytes] = []
    for path in bundle.shard_paths:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        kdf = envelope["kdf"]
        key = Scrypt(
            salt=base64.b64decode(kdf["salt"]),
            length=kdf["length"],
            n=kdf["n"],
            r=kdf["r"],
            p=kdf["p"],
        ).derive(passphrase.encode("utf-8"))
        cipher = envelope["cipher"]
        chunks.append(
            AESGCM(key).decrypt(
                base64.b64decode(cipher["nonce"]),
                base64.b64decode(cipher["ciphertext"]),
                base64.b64decode(envelope["aad"]),
            )
        )
    return b"".join(chunks)


def add_pair(vault: StudioVault, capture: CapturedDocument, marker: str) -> DistilledPair:
    pair = DistilledPair.create(
        source_id=capture.id,
        instruction=f"instruction {marker}",
        response=f"response {marker}",
        source_provenance_hash=capture.source_provenance_hash,
        tags=(marker, "analysis"),
    )
    vault.add_pair(pair)
    return pair


class EncryptedJsonlExporterTests(unittest.TestCase):
    def test_export_contains_only_approved_pairs_and_no_plaintext(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            capture = CapturedDocument.create(
                blueprint_id="lab",
                url="https://lab.invalid/source",
                title="Authorized source",
                raw_text=("Technical source material for review. " * 12),
            )
            with StudioVault(paths, "vault passphrase") as vault:
                vault.add_capture(capture)
                approved = add_pair(vault, capture, "approved-secret")
                rejected = add_pair(vault, capture, "rejected-secret")
                add_pair(vault, capture, "pending-secret")
                vault.approve_pair(approved.id)
                vault.reject_pair(rejected.id)

                bundle = EncryptedJsonlExporter(vault, paths).export(
                    passphrase="export passphrase", bundle_name="reviewed"
                )

            try:
                self.assertEqual(bundle.record_count, 1)
                plaintext = decrypt_bundle(bundle, "export passphrase")
                lines = [
                    json.loads(line) for line in plaintext.decode("utf-8").splitlines()
                ]
                self.assertEqual(
                    lines,
                    [
                        {
                            "messages": [
                                {"role": "user", "content": "instruction approved-secret"},
                                {
                                    "role": "assistant",
                                    "content": "response approved-secret",
                                },
                            ],
                            "user_approved": True,
                            "source_provenance_hash": capture.source_provenance_hash,
                            "tags": ["approved-secret", "analysis"],
                        }
                    ],
                )
                persisted = b"".join(
                    path.read_bytes()
                    for path in (*bundle.shard_paths, bundle.manifest_path)
                )
                for marker in (
                    b"approved-secret",
                    b"rejected-secret",
                    b"pending-secret",
                ):
                    self.assertNotIn(marker, persisted)
            finally:
                bundle.cleanup()

    def test_export_uses_fresh_salt_and_authenticated_test_helper(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            capture = CapturedDocument.create(
                blueprint_id="lab",
                url="https://lab.invalid/source",
                title="Authorized source",
                raw_text=("Technical source material for review. " * 12),
            )
            with StudioVault(paths, "vault passphrase") as vault:
                vault.add_capture(capture)
                pair = add_pair(vault, capture, "approved")
                vault.approve_pair(pair.id)
                exporter = EncryptedJsonlExporter(vault, paths)
                with self.assertRaises(ValueError):
                    exporter.export(passphrase="short")
                first = exporter.export(passphrase="export passphrase")
                try:
                    first_envelope = json.loads(
                        first.shard_paths[0].read_text(encoding="utf-8")
                    )
                    first_plaintext = decrypt_bundle(first, "export passphrase")
                    with self.assertRaises(InvalidTag):
                        decrypt_bundle(first, "incorrect passphrase")
                finally:
                    first.cleanup()
                second = exporter.export(passphrase="export passphrase")
                try:
                    second_envelope = json.loads(
                        second.shard_paths[0].read_text(encoding="utf-8")
                    )
                    self.assertNotEqual(
                        first_envelope["kdf"]["salt"],
                        second_envelope["kdf"]["salt"],
                    )
                    self.assertIn(b"instruction approved", first_plaintext)
                finally:
                    second.cleanup()

    def test_large_jsonl_is_split_into_broker_safe_envelopes(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            capture = CapturedDocument.create(
                blueprint_id="lab",
                url="https://lab.invalid/source",
                title="Authorized source",
                raw_text=("Technical source material for review. " * 12),
            )
            with StudioVault(paths, "vault passphrase") as vault:
                vault.add_capture(capture)
                pair = DistilledPair.create(
                    source_id=capture.id,
                    instruction="Explain the reviewed trace.",
                    response="payload-block-" * 8_000,
                    source_provenance_hash=capture.source_provenance_hash,
                    tags=("network",),
                )
                vault.add_pair(pair)
                vault.approve_pair(pair.id)
                bundle = EncryptedJsonlExporter(vault, paths).export(
                    passphrase="export passphrase", bundle_name="large"
                )

            try:
                self.assertGreater(len(bundle.shard_paths), 1)
                for shard in bundle.shard_paths:
                    self.assertEqual(shard.suffix, ".json")
                    self.assertLessEqual(len(shard.read_bytes()), 60_000)
                manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["shard_count"], len(bundle.shard_paths))
                self.assertEqual(manifest["record_count"], 1)
                self.assertEqual(
                    len(decrypt_bundle(bundle, "export passphrase")),
                    bundle.plaintext_bytes,
                )
                self.assertFalse(any(paths.staging.glob("*.tmp-*")))
            finally:
                bundle.cleanup()

    def test_export_refuses_when_nothing_is_approved(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            with StudioVault(paths, "vault passphrase") as vault:
                with self.assertRaises(ValueError):
                    EncryptedJsonlExporter(vault, paths).export(passphrase="short")
                with self.assertRaises(NoApprovedPairsError):
                    EncryptedJsonlExporter(vault, paths).export(
                        passphrase="export passphrase"
                    )

    def test_export_fails_if_plaintext_approval_index_was_forged(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            capture = CapturedDocument.create(
                blueprint_id="lab",
                url="https://lab.invalid/source",
                title="Authorized source",
                raw_text=("Technical source material for review. " * 12),
            )
            with StudioVault(paths, "vault passphrase") as vault:
                vault.add_capture(capture)
                pending = add_pair(vault, capture, "pending")
            connection = sqlite3.connect(paths.database)
            connection.execute(
                "UPDATE pairs SET status = 'approved', user_approved = 1 WHERE id = ?",
                (pending.id,),
            )
            connection.commit()
            connection.close()

            with StudioVault(paths, "vault passphrase") as vault:
                with self.assertRaises(VaultAuthenticationError):
                    EncryptedJsonlExporter(vault, paths).export(
                        passphrase="export passphrase"
                    )

    def test_export_fails_if_plaintext_provenance_index_was_changed(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            capture = CapturedDocument.create(
                blueprint_id="lab",
                url="https://lab.invalid/source",
                title="Authorized source",
                raw_text=("Technical source material for review. " * 12),
            )
            with StudioVault(paths, "vault passphrase") as vault:
                vault.add_capture(capture)
                approved = add_pair(vault, capture, "approved")
                vault.approve_pair(approved.id)
            connection = sqlite3.connect(paths.database)
            connection.execute(
                "UPDATE pairs SET source_provenance_hash = ? WHERE id = ?",
                ("b" * 64, approved.id),
            )
            connection.commit()
            connection.close()

            with StudioVault(paths, "vault passphrase") as vault:
                with self.assertRaises(VaultAuthenticationError):
                    EncryptedJsonlExporter(vault, paths).export(
                        passphrase="export passphrase"
                    )


if __name__ == "__main__":
    unittest.main()
