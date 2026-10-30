"""Demonstrate the local review and encrypted export path with synthetic data.

This demo does not open a browser, use a real source, or start model training.
Its temporary workspace and fixed example passphrases contain no user data.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio.exporter import EncryptedJsonlExporter  # noqa: E402
from studio.models import CapturedDocument, DistilledPair  # noqa: E402
from studio.paths import StudioPaths  # noqa: E402
from studio.storage import StudioVault  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"demo check failed: {message}")


def _decrypt_demo_bundle(
    shard_paths: tuple[Path, ...], export_passphrase: str
) -> list[dict]:
    """Read back only the bundle created in this temporary demo workspace."""
    chunks = []
    for shard_path in shard_paths:
        envelope = json.loads(shard_path.read_text(encoding="utf-8"))
        kdf = envelope["kdf"]
        key = Scrypt(
            salt=base64.b64decode(kdf["salt"]),
            length=kdf["length"],
            n=kdf["n"],
            r=kdf["r"],
            p=kdf["p"],
        ).derive(export_passphrase.encode("utf-8"))
        cipher = envelope["cipher"]
        chunks.append(
            AESGCM(key).decrypt(
                base64.b64decode(cipher["nonce"]),
                base64.b64decode(cipher["ciphertext"]),
                base64.b64decode(envelope["aad"]),
            )
        )
    return [json.loads(line) for line in b"".join(chunks).splitlines()]


def _add_pair(vault: StudioVault, capture: CapturedDocument, label: str) -> DistilledPair:
    pair = DistilledPair.create(
        source_id=capture.id,
        instruction=f"Explain the {label} example.",
        response=f"This is the {label} sample response.",
        source_provenance_hash=capture.source_provenance_hash,
        tags=("synthetic-demo", label),
    )
    vault.add_pair(pair)
    return pair


def _run() -> None:
    with tempfile.TemporaryDirectory(prefix="lived-studio-demo-") as temporary:
        vault_passphrase = secrets.token_urlsafe(32)
        export_passphrase = secrets.token_urlsafe(32)
        previous_directory = Path.cwd()
        try:
            os.chdir(temporary)
            paths = StudioPaths(Path("workspace")).ensure()
            capture = CapturedDocument.create(
                blueprint_id="synthetic-demo",
                url="https://example.invalid/lived-studio-demo",
                title="Synthetic review example",
                raw_text="A locally authored example used only to demonstrate review and export.",
            )
            with StudioVault(paths, vault_passphrase) as vault:
                vault.add_capture(capture)
                approved = _add_pair(vault, capture, "approved")
                rejected = _add_pair(vault, capture, "rejected")
                _add_pair(vault, capture, "pending")
                vault.approve_pair(approved.id)
                vault.reject_pair(rejected.id)

                bundle = EncryptedJsonlExporter(vault, paths).export(
                    passphrase=export_passphrase,
                    bundle_name="synthetic-review",
                )
                try:
                    records = _decrypt_demo_bundle(bundle.shard_paths, export_passphrase)
                    persisted = b"".join(
                        path.read_bytes()
                        for path in (*bundle.shard_paths, bundle.manifest_path)
                    )
                    _require(bundle.record_count == 1, "one approved export record")
                    _require(len(records) == 1, "one decrypted export record")
                    _require(
                        records[0]["messages"] == [
                            {"role": "user", "content": "Explain the approved example."},
                            {"role": "assistant", "content": "This is the approved sample response."},
                        ],
                        "approved messages match the reviewed pair",
                    )
                    _require(records[0]["user_approved"] is True, "user approval is bound")
                    _require(
                        records[0]["tags"] == ["synthetic-demo", "approved"],
                        "approved tags match",
                    )
                    _require(
                        records[0]["source_provenance_hash"] == capture.source_provenance_hash,
                        "source provenance is bound",
                    )
                    for label in ("approved", "rejected", "pending"):
                        _require(
                            f"{label} sample response".encode() not in persisted,
                            f"{label} response is not plaintext in staged files",
                        )
                    for label in ("rejected", "pending"):
                        _require(
                            label not in json.dumps(records),
                            f"{label} pair is excluded from the export",
                        )
                finally:
                    bundle.cleanup()
        finally:
            os.chdir(previous_directory)

    preview = {key: records[0][key] for key in ("messages", "tags", "user_approved")}
    print("Synthetic review: 1 approved, 1 rejected, 1 pending")
    print("Encrypted export: 1 approved record; rejected and pending records excluded")
    print("Approved record preview (from decrypted demo bundle):")
    print(json.dumps(preview, indent=2))
    print("Checks passed: encrypted files contain no sample response plaintext")


if __name__ == "__main__":
    _run()
