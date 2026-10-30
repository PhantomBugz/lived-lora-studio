from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import threading
from typing import Iterator, Literal, Sequence
from urllib.parse import urlsplit
from uuid import uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .models import (
    CapturedDocument,
    DistilledPair,
    PairReviewRow,
    OPERATOR_APPROVED,
    OPERATOR_DELETED,
    OPERATOR_MODIFIED,
    OPERATOR_REJECTED,
    require_review_reason,
    thaw_json_value,
)
from .training_audit import (
    AuditAppendResult,
    ManifestAuditEvent,
    ManifestEventKind,
    ManifestGpuIdentity,
    ManifestPreparation,
    PreparedManifestResult,
    ReconciliationOutcome,
    TrainingRunIdentity,
)

from .training_models import (
    CaptureTrainingView,
    DecisionView,
    PairRevisionView,
    ReadinessSnapshot,
    SourceTargetView,
    TrainingMode,
)
from .paths import (StudioPaths, UnsafePathError, PathRaceError, _absolute_lexical, _checked_lstat, _entry_identity, _snapshot_chain, _path_chain)
from .proxy_settings import ProxySettings
from .capmonster import CapMonsterCredential
from .saved_access import SavedLogin, SavedSource
from .quality import QualityAssessment, assess_quality, hamming_distance, simhash64
from .source_artifacts import (
    ArtifactSegment, DedupeRecord, TECHNICAL_SIGNALS,
    canonicalize_text, choose_dedupe_winners, normalize_artifact_path,
    require_allowed_extension, reject_secret_material, winnowed_five_token_fingerprints,
)
from .dataset_lineage import _fsync_directory, _rename_no_replace, _write_new_private
from .github_ingestion import (
    GitHubCollectionPolicy, RepositoryIdentity, RepositoryInspection, RepositoryLocator, RepositoryRightsRecord,
    _canonical_json, _require_grant_scope, require_release_inspection,
)
from .history_anchor import HistoryAnchorProtocolError, advance_history_anchor, read_history_anchor
from .paths import _require_private_directory, _require_private_file, _file_identity


CAPTURE_READY = "ready"
CAPTURE_QUARANTINED = "quarantined"
CAPTURE_DISTILLED = "distilled"
PAIR_PENDING = "pending"
PAIR_APPROVED = "approved"
PAIR_REJECTED = "rejected"

_CAPTURE_STATUSES = frozenset(
    {CAPTURE_READY, CAPTURE_QUARANTINED, CAPTURE_DISTILLED}
)
_PAIR_STATUSES = frozenset({PAIR_PENDING, PAIR_APPROVED, PAIR_REJECTED})
_SHA256 = re.compile(r"[a-f0-9]{64}")
_SIMHASH_HEX = re.compile(r"[a-f0-9]{16}")
_POLICY_REASON_CODE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,255}")
_SALT_BYTES = 16
_NONCE_BYTES = 12
_CIPHER_VERSION = b"\x01"
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_KEY_BYTES = 32
_SCHEMA_VERSION = 6

_V4_SCHEMA_DIGESTS = {
    ("index", "capture_events_capture_idx", "capture_events"): "3c23c8955d78fdd0eef9098c83677fe869011ecc1f3dcc64851fc9c3a29648d8",
    ("index", "captures_content_sha256_idx", "captures"): "e32cb2690416dda0da07418c1971cfada66ab952615fcf5c9c8ea2cd72be69fe",
    ("index", "captures_status_idx", "captures"): "072c20c521951586ef71211340805cbad2dd6eee6d7a44ee4fca128c9c3de154",
    ("index", "pair_revisions_pair_idx", "pair_revisions"): "e64ce19abc217248d8ad5491c22904879563ee53381ea2352e40b9e515432738",
    ("index", "pairs_source_id_idx", "pairs"): "14b4defe5fdc7f00ce72c4cd9beae3dde4d89eff4c31611e4062c1d800bcc3ce",
    ("index", "pairs_status_idx", "pairs"): "8f1525db47de5faff45311ad621367822ab8d513fa17701d6301bcee7391d889",
    ("index", "review_events_pair_idx", "review_events"): "d78838b36f0210476de3a503edc2602e77869494b39077fd74a189beb2fd5648",
    ("index", "training_run_events_run_idx", "training_run_events"): "ca842530f51ee57b181c11abeb015ddae8db67886663e3babe0f6ea96500edfa",
    ("table", "audit_events", "audit_events"): "9970032321508b15fe3da8672e08eac59520a5df353570fe25f4ffcaf39da206",
    ("table", "audit_state", "audit_state"): "c73d61074299fd0fe9040266ca8c72cbc9cbc0f35903972ffb1dd16bf9efe6b5",
    ("table", "capture_events", "capture_events"): "f904f056856e9fdde3a54d7e3b0381d21689da5c58d29b6de2b5e40192df5d37",
    ("table", "captures", "captures"): "800b5b9c139bc71d87404753326fd72ecd9858aebb27957d43a0a4cb319c7a47",
    ("table", "pair_revisions", "pair_revisions"): "27f6ebcb9a67d916c463d58ec6cbda7edd897a851561c77c91d9563810887df0",
    ("table", "pairs", "pairs"): "203a08f1f41fb2e3a5da6c448dc9f28f370e2cce5712b573911227362a56d588",
    ("table", "rag_index_versions", "rag_index_versions"): "18347bc8d06db9ef0b667a565676c0df8b85aacae58620fbc9cdd19652afefdb",
    ("table", "review_events", "review_events"): "9c475086fbe4358e15a86dfab8a3ce4a1bff482f6fff0e82240d60ccff07a9fd",
    ("table", "source_targets", "source_targets"): "c8dcdc1c561c47af9620d6d5753e5ef8d2d6f8d10f3ca86fc7ea427b1c39484b",
    ("table", "tombstones", "tombstones"): "f3a14e9b96557587fab71cd8ba76b6d95e9ea325b89e02182a857cfda0847775",
    ("table", "training_root_events", "training_root_events"): "ad3bbbcbf4594e008f4723fefd609c2a9be0abcadfdf675f86fd5221aff3a02b",
    ("table", "training_run_events", "training_run_events"): "929a1071f7d0f80993443020592e897b1f49f45f0ad31e60f06032a8fad70896",
    ("table", "training_runs", "training_runs"): "e5f716ab22f72dda459663706389f4afed08ed853113f29fca534def4b48e6a4",
    ("table", "vault_meta", "vault_meta"): "3ffb64fe89e1521b60fa3768ae08b2918825d20057fe7f7f7ad26d9fd3957299",
}
_V5_SCHEMA_DIGESTS = _V4_SCHEMA_DIGESTS | {
    ("table", "scraper_access_records", "scraper_access_records"):
        "359d2e5e161261c0a816bd370720eeb3d482fb8fcc1f213aba28af8deb38aaf4",
}
_V6_SCHEMA_DIGESTS = _V4_SCHEMA_DIGESTS | {
    ("table", "scraper_access_records", "scraper_access_records"):
        "a96c420a456ff4564979e235d47b85a73b24ed3cd2bbb92cffbc83bce395610a",
}
_ACCESS_TABLE = "scraper_access_records"
_ACCESS_SOURCE = "SOURCE"
_ACCESS_LOGIN = "LOGIN"
_ACCESS_PROXY = "PROXY"
_ACCESS_CAPMONSTER = "CAPMONSTER"
_VERIFIER_NAME = "key_verifier"
_VERIFIER_VALUE = b"lived-lora-studio:vault-key-verifier:v2"
_LEGACY_QUARANTINE_REASON = "legacy_state_requires_review"
_LOCAL_STATE_ANCHOR_KIND = "LOCAL_STATE_ANCHOR_V1"
_LOCAL_STATE_SCOPE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,31}")
_MAX_AUDIT_SEQUENCE = 9_223_372_036_854_775_807
_ARTIFACT_KINDS = frozenset({"vbulletin_post", "github_blob", "forum_attachment"})
_ASSESSMENT_REASONS = frozenset({
    "insufficient_technical_signal", "marketing_dominant", "social_dominant",
    "overlong_line", "oversized_configuration_element", "minimum_length",
    "alphabetic_ratio", "repetition", "boilerplate", "control_characters",
    "vbulletin_quote_ratio_invalid", "vbulletin_quote_ratio_exceeds_0.30",
    "vbulletin_has_code_invalid", "vbulletin_social_only",
})
_DEDUPE_REASONS = frozenset({"canonical_text_match", "near_duplicate"})
_GITHUB_METADATA = frozenset({
    "source_kind", "repository_id", "repository_node_id", "owner", "repository",
    "repository_api_url", "license_api_url", "license_identity_sha256", "commit_sha",
    "tree_sha", "blob_sha", "raw_byte_sha256", "path", "license_spdx",
    "rights_record_sha256", "collector_policy_sha256",
})
_ATTACHMENT_METADATA = frozenset({
    "source_kind", "parent_post_id", "parent_post_provenance_hash", "attachment_id",
    "canonical_url", "filename", "raw_byte_sha256", "collector_policy_sha256",
})
_POST_METADATA = frozenset({
    "source_kind", "thread_id", "thread_page", "post_id", "post_index", "has_code",
    "has_attachment", "original_utf8_bytes", "retained_utf8_bytes", "quote_utf8_bytes",
})
_POST_OPTIONAL_METADATA = frozenset({
    "ingestion", "forum_id", "quote_ratio", "truncated", "source_text_truncated",
    "source_document_sha256", "chunk_index", "chunk_count",
})


@dataclass(frozen=True, slots=True)
class RagIndexVersion:
    version_id: str
    dataset_sha256: str
    corpus_sha256: str
    config_sha256: str
    embedding_model_sha256: str
    index_sha256: str
    index_bytes: int
    metadata_sha256: str
    metadata_bytes: int
    inventory_sha256: str
    dataset_bytes: int
    document_count: int
    chunk_count: int
    dimension: int = 384
    serialization_version: int = 1

    def __post_init__(self) -> None:
        if type(self.version_id) is not str or re.fullmatch(r"[a-f0-9]{32}", self.version_id) is None:
            raise ValueError("RAG version_id must be 32 lowercase hexadecimal characters")
        for name in (
            "dataset_sha256", "corpus_sha256", "config_sha256",
            "embedding_model_sha256", "index_sha256", "metadata_sha256",
            "inventory_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be lowercase SHA-256")
        for name, maximum in (
            ("index_bytes", 268_435_456),
            ("metadata_bytes", 67_108_928),
            ("dataset_bytes", 33_554_432),
            ("document_count", 50_000),
            ("chunk_count", 100_000),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} is outside the allowed bound")
        if type(self.dimension) is not int or self.dimension != 384:
            raise ValueError("RAG dimension must be 384")
        if type(self.serialization_version) is not int or self.serialization_version != 1:
            raise ValueError("RAG serialization version must be 1")

    @property
    def version_sha256(self) -> str:
        return hashlib.sha256(
            b"liveD:rag-index-version:v1\0"
            + StudioVault._canonical_ledger_bytes(self.to_state())
        ).hexdigest()

    def to_state(self) -> dict[str, object]:
        return {
            "chunk_count": self.chunk_count,
            "config_sha256": self.config_sha256,
            "corpus_sha256": self.corpus_sha256,
            "dataset_bytes": self.dataset_bytes,
            "dataset_sha256": self.dataset_sha256,
            "dimension": self.dimension,
            "document_count": self.document_count,
            "embedding_model_sha256": self.embedding_model_sha256,
            "index_bytes": self.index_bytes,
            "index_sha256": self.index_sha256,
            "inventory_sha256": self.inventory_sha256,
            "metadata_bytes": self.metadata_bytes,
            "metadata_sha256": self.metadata_sha256,
            "schema": "LIVED_RAG_INDEX_VERSION_V1",
            "serialization_version": self.serialization_version,
            "version_id": self.version_id,
        }


@dataclass(frozen=True, slots=True)
class RagIndexEvent:
    event_id: str
    sequence: int
    kind: str
    prepared_event_id: str | None
    previous_version_sha256: str | None
    version_sha256: str
    version: RagIndexVersion
    created_at: datetime

    def __post_init__(self) -> None:
        if type(self.event_id) is not str or re.fullmatch(r"[a-f0-9]{32}", self.event_id) is None:
            raise ValueError("RAG event_id is invalid")
        if type(self.sequence) is not int or not 1 <= self.sequence <= 9_223_372_036_854_775_807:
            raise ValueError("RAG event sequence is invalid")
        if self.kind not in {"PREPARED", "PUBLISHED", "SELECTED"}:
            raise ValueError("RAG event kind is invalid")
        if self.prepared_event_id is not None and (
            type(self.prepared_event_id) is not str
            or re.fullmatch(r"[a-f0-9]{32}", self.prepared_event_id) is None
        ):
            raise ValueError("RAG prepared event ID is invalid")
        if self.kind == "PREPARED" and self.prepared_event_id is not None:
            raise ValueError("prepared RAG event cannot reference itself")
        if self.kind != "PREPARED" and self.prepared_event_id is None:
            raise ValueError("terminal RAG event must reference its preparation")
        if self.previous_version_sha256 is not None and (
            type(self.previous_version_sha256) is not str
            or _SHA256.fullmatch(self.previous_version_sha256) is None
        ):
            raise ValueError("previous RAG version digest is invalid")
        if type(self.version_sha256) is not str or _SHA256.fullmatch(self.version_sha256) is None:
            raise ValueError("RAG version digest is invalid")
        if type(self.version) is not RagIndexVersion or self.version_sha256 != self.version.version_sha256:
            raise ValueError("RAG event does not bind its exact version")
        if (
            type(self.created_at) is not datetime
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("RAG event timestamp must be UTC-aware")


@dataclass(frozen=True, slots=True)
class RagIndexAppendResult:
    event: RagIndexEvent
    appended: bool

    def __post_init__(self) -> None:
        if type(self.event) is not RagIndexEvent or type(self.appended) is not bool:
            raise TypeError("invalid RAG index append result")


@dataclass(frozen=True, slots=True)
class LocalStateAnchor:
    """Authenticated pointer to a content-free history outside the vault."""

    scope: str
    count: int
    head_sha256: str
    audit_sequence: int
    audit_event_sha256: str

    def __post_init__(self) -> None:
        if type(self.scope) is not str or _LOCAL_STATE_SCOPE.fullmatch(self.scope) is None:
            raise ValueError("local-state anchor scope is invalid")
        if type(self.count) is not int or not 1 <= self.count <= _MAX_AUDIT_SEQUENCE:
            raise ValueError("local-state anchor count is invalid")
        if type(self.head_sha256) is not str or _SHA256.fullmatch(self.head_sha256) is None:
            raise ValueError("local-state anchor head must be lowercase SHA-256")
        if (
            type(self.audit_sequence) is not int
            or not 1 <= self.audit_sequence <= _MAX_AUDIT_SEQUENCE
        ):
            raise ValueError("local-state anchor audit sequence is invalid")
        if (
            type(self.audit_event_sha256) is not str
            or _SHA256.fullmatch(self.audit_event_sha256) is None
        ):
            raise ValueError("local-state anchor audit digest must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class LocalStateAnchorAppendResult:
    anchor: LocalStateAnchor
    appended: bool

    def __post_init__(self) -> None:
        if type(self.anchor) is not LocalStateAnchor or type(self.appended) is not bool:
            raise TypeError("invalid local-state anchor append result")


class VaultError(RuntimeError):
    pass


class VaultLockedError(VaultError):
    pass


class VaultAuthenticationError(VaultError):
    pass


class LocalStateAnchorConflictError(VaultError):
    pass


class _DurableAuditRejection(VaultAuthenticationError):
    """A rejected operation whose authenticated evidence must be committed."""


class VaultSchemaError(VaultError):
    pass


class VaultSaltError(VaultError):
    pass


class CaptureDeletionError(VaultError):
    pass


class RecordNotFoundError(KeyError):
    pass


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    capture_id: str
    distance: int


@dataclass(frozen=True, slots=True)
class CaptureAddResult:
    capture_id: str
    inserted: bool

    def __post_init__(self) -> None:
        if (
            type(self.capture_id) is not str
            or not self.capture_id
            or type(self.inserted) is not bool
        ):
            raise TypeError("invalid capture add result")


@dataclass(frozen=True, slots=True)
class AssessedCapture:
    document: CapturedDocument
    score: int
    signals: tuple[str, ...]
    reasons: tuple[str, ...]
    disposition: Literal["review_candidate", "quarantined"]
    raw_sha256: str
    canonical_text_sha256: str
    fingerprints: frozenset[int]


@dataclass(frozen=True, slots=True)
class _PreparedCaptureState:
    capture_id: str
    content_sha256: str
    source_provenance_hash: str
    simhash_hex: str
    quality_score: float
    quality_reasons_enc: bytes
    status: str
    quarantine_reason_enc: bytes
    state_enc: bytes


_PROCESS_LOCK_GUARD = threading.Lock()
_PROCESS_LOCKS: set[str] = set()


def _regular_single_file(metadata):
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or int(getattr(metadata, 'st_file_attributes', 0)) & 0x400):
        raise UnsafePathError('internal files must be regular, non-reparse files with one link')


def _fixed_descriptor(path, *, write=False, create=False, directory=False, share_delete=False, parent_descriptor=None):
    """Open without truncation or following a final link; validate before use."""
    if os.name != 'nt':
        flags = (os.O_RDWR if write else os.O_RDONLY) | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        return os.open(path.name if parent_descriptor is not None else path, flags, 0o600,
                       dir_fd=parent_descriptor)
    import ctypes
    from ctypes import wintypes
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    open_file = kernel.CreateFileW
    open_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                          wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    open_file.restype = wintypes.HANDLE
    handle = open_file(str(path), 0x80000000 | (0x40000000 if write else 0),
                       3 | (4 if share_delete else 0), None, 1 if create else 3,
                       0x00200000 | (0x02000000 if directory else 0x80), None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(handle, (os.O_RDWR if write else os.O_RDONLY) | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle(handle)
        raise


class _FixedFileSet:
    """Checked internal names, pinned descriptors, and persistence-boundary checks.

    Windows handles deny rename/delete while retained. POSIX checks complement
    the private workspace and exclusive vault lock; they do not isolate a
    concurrently hostile process running as the same OS user.
    """
    def __init__(self, parent, names, *, ephemeral=()):
        self.parent = _absolute_lexical(parent)
        self.names = tuple(names)
        self.ephemeral = frozenset(ephemeral)
        self._pins = {}
        self._expected = {name: None for name in self.names}
        self._directories = []
        self._chain = _snapshot_chain(self.parent, require_target=True)
        try:
            for component in _path_chain(self.parent):
                before = _checked_lstat(component)
                if not stat.S_ISDIR(before.st_mode):
                    raise UnsafePathError('internal file ancestry must contain only directories')
                descriptor = _fixed_descriptor(component, directory=True)
                self._directories.append(descriptor)
                if _entry_identity(os.fstat(descriptor)) != _entry_identity(before):
                    raise PathRaceError('internal directory changed while opening')
            self.validate()
        except BaseException:
            self.close()
            raise

    def path(self, name):
        if name not in self._expected:
            raise UnsafePathError('unknown internal filename')
        return self.parent / name

    def _ancestry(self):
        if _snapshot_chain(self.parent, require_target=True) != self._chain:
            raise PathRaceError('internal directory identity changed')

    def info(self, name):
        self.validate()
        descriptor = self._pins.get(name)
        return os.fstat(descriptor) if descriptor is not None else None

    def validate(self):
        self._ancestry()
        for name in self.names:
            try:
                actual = _checked_lstat(self.path(name))
            except FileNotFoundError:
                if self._expected[name] is not None and name not in self.ephemeral:
                    raise PathRaceError('internal file disappeared') from None
                descriptor = self._pins.pop(name, None)
                if descriptor is not None:
                    os.close(descriptor)
                self._expected[name] = None
                continue
            _regular_single_file(actual)
            identity = _entry_identity(actual)
            if self._expected[name] is not None and identity != self._expected[name]:
                raise PathRaceError('internal file identity changed')
            descriptor = self._pins.get(name)
            if descriptor is None:
                descriptor = _fixed_descriptor(self.path(name), share_delete=name in self.ephemeral,
                                               parent_descriptor=self._directories[-1] if os.name != "nt" else None)
                self._pins[name] = descriptor
            opened = os.fstat(descriptor)
            _regular_single_file(opened)
            if _entry_identity(opened) != identity:
                raise PathRaceError('internal file changed while opening')
            self._expected[name] = identity
        self._ancestry()

    def open(self, name, *, write=False, create=False):
        self.validate()
        absent = self._expected[name] is None
        if absent and not create:
            raise FileNotFoundError(self.path(name))
        descriptor = _fixed_descriptor(self.path(name), write=write, create=absent,
            share_delete=name in self.ephemeral,
            parent_descriptor=self._directories[-1] if os.name != "nt" else None)
        try:
            actual = os.fstat(descriptor)
            _regular_single_file(actual)
            after = _checked_lstat(self.path(name))
            if (_entry_identity(actual) != _entry_identity(after)
                    or self._expected[name] not in (None, _entry_identity(actual))):
                raise PathRaceError('internal file changed before use')
            self.validate()
            final = os.fstat(descriptor)
            _regular_single_file(final)
            if self._expected[name] != _entry_identity(final):
                raise PathRaceError("opened internal descriptor no longer matches its pathname")
            if write and os.name != 'nt':
                os.fchmod(descriptor, 0o600)
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def read(self, name, maximum):
        descriptor = self.open(name)
        try:
            data = os.read(descriptor, maximum + 1)
            self.validate()
            return data
        finally:
            os.close(descriptor)

    def private_modes(self):
        self.validate()
        if os.name != 'nt':
            for descriptor in self._pins.values():
                _regular_single_file(os.fstat(descriptor))
                os.fchmod(descriptor, 0o600)

    def remove(self, name):
        self.validate()
        if self._expected[name] is None:
            return
        descriptor = self._pins.pop(name, None)
        if descriptor is not None:
            os.close(descriptor)
        self._ancestry()
        actual = _checked_lstat(self.path(name))
        _regular_single_file(actual)
        if _entry_identity(actual) != self._expected[name]:
            raise PathRaceError('internal file changed before removal')
        if os.name == "nt":
            self.path(name).unlink()
        else:
            os.unlink(name, dir_fd=self._directories[-1])
        self._expected[name] = None

    def rename(self, source, destination):
        self.validate()
        identity = self._expected[source]
        if identity is None:
            return
        if self._expected[destination] is not None:
            raise PathRaceError('internal rotation destination already exists')
        descriptor = self._pins.pop(source, None)
        if descriptor is not None:
            os.close(descriptor)
        self._ancestry()
        before = _checked_lstat(self.path(source))
        _regular_single_file(before)
        if _entry_identity(before) != identity:
            raise PathRaceError('internal file changed before rotation')
        if os.name == "nt":
            self.path(source).rename(self.path(destination))
        else:
            os.rename(source, destination, src_dir_fd=self._directories[-1],
                      dst_dir_fd=self._directories[-1])
        self._expected[source] = None
        self._expected[destination] = identity
        self.validate()

    def close(self):
        for descriptor in (*self._pins.values(), *reversed(self._directories)):
            os.close(descriptor)
        self._pins.clear()
        self._directories.clear()


class _GuardedConnection(sqlite3.Connection):
    def __init__(self, fixed, *args, **kwargs):
        self._fixed = fixed
        fixed.validate()
        super().__init__(*args, **kwargs)
        # Avoid checkpoint writes during teardown after an identity violation.
        # Durable WAL is recovered by the next checked vault open.
        self.setconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True)
        self._guard_error = None
        self.set_authorizer(self._authorize)
        fixed.validate()

    def _authorize(self, *_arguments):
        try:
            self._fixed.validate()
        except (UnsafePathError, OSError) as error:
            self._guard_error = error
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    def execute(self, *args, **kwargs):
        self._fixed.validate()
        self._guard_error = None
        try:
            result = super().execute(*args, **kwargs)
        except sqlite3.DatabaseError:
            if self._guard_error is not None:
                raise self._guard_error
            raise
        self._fixed.validate()
        return result

    def executemany(self, *args, **kwargs):
        self._fixed.validate()
        self._guard_error = None
        try:
            result = super().executemany(*args, **kwargs)
        except sqlite3.DatabaseError:
            if self._guard_error is not None:
                raise self._guard_error
            raise
        self._fixed.validate()
        return result

    def executescript(self, *args, **kwargs):
        self._fixed.validate()
        self._guard_error = None
        try:
            result = super().executescript(*args, **kwargs)
        except sqlite3.DatabaseError:
            if self._guard_error is not None:
                raise self._guard_error
            raise
        self._fixed.validate()
        return result


class _ProcessFileLock:
    def __init__(self, path: Path, fixed: _FixedFileSet):
        self.path = path
        self._fixed = fixed
        self._file = None
        self._registry_key: str | None = None

    def acquire(self) -> None:
        registry_key = os.path.normcase(str(_absolute_lexical(self.path)))
        with _PROCESS_LOCK_GUARD:
            if registry_key in _PROCESS_LOCKS:
                raise VaultLockedError("another liveD Phantom Studio process owns the vault")
            _PROCESS_LOCKS.add(registry_key)
        try:
            self._file = os.fdopen(self._fixed.open(self.path.name, write=True, create=True), "r+b")
            self._fixed.validate()
            self._file.seek(0, os.SEEK_END)
            if self._file.tell() == 0:
                self._file.write(b"\0")
                self._file.flush()
                os.fsync(self._file.fileno())
            self._file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._registry_key = registry_key
        except BaseException as exc:
            if self._file is not None:
                self._file.close()
                self._file = None
            with _PROCESS_LOCK_GUARD:
                _PROCESS_LOCKS.discard(registry_key)
            if isinstance(exc, OSError):
                raise VaultLockedError("another liveD Phantom Studio process owns the vault") from exc
            raise

    def release(self) -> None:
        file_handle = self._file
        registry_key = self._registry_key
        self._file = None
        self._registry_key = None
        if file_handle is not None:
            try:
                file_handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(file_handle.fileno(), fcntl.LOCK_UN)
            finally:
                file_handle.close()
        if registry_key is not None:
            with _PROCESS_LOCK_GUARD:
                _PROCESS_LOCKS.discard(registry_key)


def _derive_key(
    passphrase: str,
    salt: bytes,
    *,
    enforce_minimum: bool,
) -> bytes:
    if not isinstance(passphrase, str) or not passphrase.strip():
        raise ValueError("passphrase must be a nonempty string")
    if enforce_minimum and len(passphrase) < 12:
        raise ValueError("passphrase must contain at least 12 characters")
    encoded = passphrase.encode("utf-8")
    if len(encoded) > 4096:
        raise ValueError("passphrase exceeds the UTF-8 size limit")
    return Scrypt(
        salt=salt,
        length=_KEY_BYTES,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    ).derive(encoded)


class StudioVault:
    """Exclusive encrypted persistence for captures and reviewed pairs."""

    def __init__(self, paths: StudioPaths, passphrase: str):
        if not isinstance(paths, StudioPaths):
            raise TypeError("paths must be a StudioPaths instance")
        self.paths = paths.ensure()
        self._mutex = threading.RLock()
        self._connection: sqlite3.Connection | None = None
        self._key: bytes | None = None
        self._fixed = _FixedFileSet(self.paths.data,
            ("studio.lock", "vault.salt", "studio.db", "studio.db-wal", "studio.db-shm", "studio.db-journal"),
            ephemeral=("studio.db-wal", "studio.db-shm", "studio.db-journal"))
        self._process_lock = _ProcessFileLock(self._fixed.path("studio.lock"), self._fixed)
        try:
            self._process_lock.acquire()
            _salt, self._key, had_existing_database = self._load_or_create_salt(
                passphrase
            )
            os.close(self._fixed.open("studio.db", write=True, create=True))
            self._connection = sqlite3.connect(
                self._fixed.path("studio.db"),
                factory=lambda *args, **kwargs: _GuardedConnection(self._fixed, *args, **kwargs),
                isolation_level=None,
                cached_statements=0,
                check_same_thread=False,
                timeout=5.0,
            )
            self._connection.row_factory = sqlite3.Row
            self._migrate(had_existing_database=had_existing_database)
            self._protect_database_files()
        except BaseException:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._key = None
            self._process_lock.release()
            self._fixed.close()
            raise

    def __enter__(self) -> "StudioVault":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        with self._mutex:
            connection = self._connection
            self._connection = None
            try:
                if connection is not None:
                    connection.close()
            finally:
                if self._key is not None:
                    self._key = b"\0" * len(self._key)
                    self._key = None
                self._process_lock.release()
                self._fixed.close()

    def _load_or_create_salt(
        self, passphrase: str
    ) -> tuple[bytes, bytes, bool]:
        has_existing_database = any(
            info is not None and info.st_size > 0
            for info in (self._fixed.info("studio.db"), self._fixed.info("studio.db-wal"))
        )
        if self._fixed.info("vault.salt") is not None:
            salt = self._fixed.read("vault.salt", _SALT_BYTES)
            if len(salt) != _SALT_BYTES:
                raise VaultSaltError("vault salt has an invalid length")
            key = _derive_key(passphrase, salt, enforce_minimum=not has_existing_database)
            return salt, key, has_existing_database
        if has_existing_database:
            raise VaultSaltError(
                "vault salt is missing for an existing encrypted database"
            )
        salt = os.urandom(_SALT_BYTES)
        key = _derive_key(passphrase, salt, enforce_minimum=True)
        descriptor = self._fixed.open("vault.salt", write=True, create=True)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(salt)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(descriptor)
        self._fixed.validate()
        return salt, key, False

    def _configure_connection(self) -> None:
        connection = self._require_connection()
        connection.execute("PRAGMA foreign_keys = ON")
        journal_mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(journal_mode).casefold() != "wal":
            raise VaultError("SQLite WAL mode could not be enabled")
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA busy_timeout = 5000")

    def _migrate(self, *, had_existing_database: bool) -> None:
        connection = self._require_connection()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("vault database header is invalid") from exc
        if version > _SCHEMA_VERSION:
            raise VaultSchemaError("vault schema is newer than this application")
        if version == 6:
            self._verify_key_verifier()
            self._validate_v6_schema()
            self._configure_connection()
            self._load_audit_state(connection)
            return
        if version == 5:
            self._verify_key_verifier()
            self._validate_v5_schema()
            self._configure_connection()
            self._migrate_v5_to_v6()
            self._load_audit_state(connection)
            return
        if version == 4:
            self._verify_key_verifier()
            self._validate_v4_schema()
            self._configure_connection()
            self._migrate_v4_to_v5()
            self._migrate_v5_to_v6()
            self._load_audit_state(connection)
            return
        if version == 3:
            self._verify_key_verifier()
            self._validate_v3_schema()
            self._configure_connection()
            self._migrate_v3_to_v4()
            self._migrate_v4_to_v5()
            self._migrate_v5_to_v6()
            self._load_audit_state(connection)
            return
        if version == 2:
            self._verify_key_verifier()
            self._validate_v2_schema()
            capture_states, verified_sources = self._prepare_legacy_capture_states(
                require_encrypted_record=False
            )
            pair_states = self._prepare_legacy_pair_states(
                verified_sources, authenticate_v2_state=True
            )
            self._configure_connection()
            self._migrate_v2_to_v3(capture_states, pair_states)
            self._migrate_v3_to_v4()
            self._migrate_v4_to_v5()
            self._migrate_v5_to_v6()
            self._load_audit_state(connection)
            return
        if version == 1:
            capture_states, verified_sources = self._prepare_legacy_capture_states(
                require_encrypted_record=True
            )
            pair_states = self._prepare_legacy_pair_states(
                verified_sources, authenticate_v2_state=False
            )
            self._configure_connection()
            self._migrate_v1_to_v3(capture_states, pair_states)
            self._migrate_v3_to_v4()
            self._migrate_v4_to_v5()
            self._migrate_v5_to_v6()
            self._load_audit_state(connection)
            return
        if version != 0 or had_existing_database:
            raise VaultSchemaError(
                "existing vault has no supported authenticated schema"
            )
        self._configure_connection()
        self._create_schema_v6()
        self._load_audit_state(connection)

    @staticmethod
    def _create_core_tables(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE captures (
                id TEXT PRIMARY KEY,
                blueprint_id TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                source_provenance_hash TEXT NOT NULL,
                url_enc BLOB NOT NULL,
                title_enc BLOB NOT NULL,
                raw_text_enc BLOB NOT NULL,
                metadata_enc BLOB NOT NULL,
                simhash_hex TEXT NOT NULL,
                quality_score REAL NOT NULL,
                quality_reasons_enc BLOB NOT NULL,
                status TEXT NOT NULL,
                quarantine_reason_enc BLOB,
                state_enc BLOB NOT NULL,
                CHECK (status IN ('ready', 'quarantined', 'distilled'))
            )
            """
        )
        connection.execute(
            "CREATE INDEX captures_content_sha256_idx ON captures(content_sha256)"
        )
        connection.execute("CREATE INDEX captures_status_idx ON captures(status)")
        connection.execute(
            """
            CREATE TABLE pairs (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES captures(id) ON DELETE CASCADE,
                source_provenance_hash TEXT NOT NULL,
                instruction_enc BLOB NOT NULL,
                response_enc BLOB NOT NULL,
                tags_enc BLOB NOT NULL,
                state_enc BLOB NOT NULL,
                user_approved INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                CHECK (user_approved IN (0, 1)),
                CHECK (status IN ('pending', 'approved', 'rejected')),
                CHECK ((status = 'approved' AND user_approved = 1) OR
                       (status != 'approved' AND user_approved = 0))
            )
            """
        )
        connection.execute("CREATE INDEX pairs_source_id_idx ON pairs(source_id)")
        connection.execute("CREATE INDEX pairs_status_idx ON pairs(status)")
        connection.execute(
            """
            CREATE TABLE vault_meta (
                name TEXT PRIMARY KEY,
                value_enc BLOB NOT NULL,
                CHECK (name = 'key_verifier')
            )
            """
        )

    @staticmethod
    def _create_v4_tables(connection: sqlite3.Connection) -> None:
        script = """
            CREATE TABLE audit_events (
                sequence INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL CHECK(length(kind) BETWEEN 1 AND 64),
                subject_id TEXT NOT NULL CHECK(length(subject_id) BETWEEN 1 AND 128),
                previous_event_sha256 TEXT,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE audit_state (
                singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
                sequence INTEGER NOT NULL CHECK(sequence >= 0),
                head_sha256 TEXT,
                review_head_sha256 TEXT,
                tombstone_head_sha256 TEXT,
                revision_head_sha256 TEXT,
                target_head_sha256 TEXT,
                capture_sequence INTEGER NOT NULL CHECK(capture_sequence >= 0),
                capture_head_sha256 TEXT,
                state_enc BLOB NOT NULL
            );
            CREATE TABLE capture_events (
                event_id TEXT PRIMARY KEY,
                capture_sequence INTEGER NOT NULL UNIQUE
                    CHECK(capture_sequence >= 1),
                capture_id TEXT NOT NULL,
                kind TEXT NOT NULL
                    CHECK(kind IN ('CREATED','MIGRATED','STATUS_CHANGED','QUARANTINED')),
                capture_state_sha256 TEXT NOT NULL,
                previous_event_sha256 TEXT,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX capture_events_capture_idx
                ON capture_events(capture_id, capture_sequence);
            CREATE TABLE pair_revisions (
                revision_id TEXT PRIMARY KEY,
                pair_id TEXT NOT NULL REFERENCES pairs(id),
                lineage_id TEXT NOT NULL,
                parent_revision_id TEXT REFERENCES pair_revisions(revision_id),
                revision INTEGER NOT NULL CHECK(revision >= 1),
                sequence INTEGER NOT NULL UNIQUE CHECK(sequence >= 1),
                payload_enc BLOB NOT NULL,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(pair_id, revision)
            );
            CREATE INDEX pair_revisions_pair_idx
                ON pair_revisions(pair_id, revision);
            CREATE TABLE review_events (
                event_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE CHECK(sequence >= 1),
                pair_id TEXT NOT NULL REFERENCES pairs(id),
                revision_id TEXT NOT NULL REFERENCES pair_revisions(revision_id),
                kind TEXT NOT NULL CHECK(kind IN ('APPROVED','DENIED','MODIFIED')),
                reason_code TEXT NOT NULL,
                previous_event_sha256 TEXT,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX review_events_pair_idx
                ON review_events(pair_id, sequence);
            CREATE TABLE source_targets (
                target_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE CHECK(sequence >= 1),
                blueprint_id TEXT NOT NULL,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE tombstones (
                tombstone_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE CHECK(sequence >= 1),
                record_kind TEXT NOT NULL CHECK(record_kind IN ('PAIR','CAPTURE')),
                record_id TEXT NOT NULL,
                reason_code TEXT NOT NULL CHECK(reason_code = 'OPERATOR_DELETED'),
                previous_tombstone_sha256 TEXT,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(record_kind, record_id)
            );
            CREATE TABLE training_runs (
                run_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL CHECK(mode IN ('lora','dpo','rag')),
                created_at TEXT NOT NULL,
                state_enc BLOB NOT NULL
            );
            CREATE TABLE training_run_events (
                event_id TEXT PRIMARY KEY,
                global_sequence INTEGER NOT NULL UNIQUE CHECK(global_sequence >= 1),
                run_id TEXT NOT NULL REFERENCES training_runs(run_id),
                event_ordinal INTEGER NOT NULL CHECK(event_ordinal >= 1),
                kind TEXT NOT NULL CHECK(kind IN (
                    'MANIFEST_PREPARED','MANIFEST_PUBLISHED',
                    'ABORTED_BEFORE_PUBLICATION','RECONCILIATION_BLOCKED',
                    'AUDIT_TAMPERED'
                )),
                prepared_event_id TEXT,
                manifest_sequence INTEGER,
                previous_event_sha256 TEXT,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(run_id, event_ordinal)
            );
            CREATE INDEX training_run_events_run_idx
                ON training_run_events(run_id, event_ordinal);
            CREATE TABLE training_root_events (
                event_id TEXT PRIMARY KEY,
                global_sequence INTEGER NOT NULL UNIQUE CHECK(global_sequence >= 1),
                kind TEXT NOT NULL CHECK(kind = 'ROOT_ANOMALY'),
                evidence_sha256 TEXT NOT NULL UNIQUE,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE rag_index_versions (
                index_id TEXT PRIMARY KEY,
                sequence INTEGER NOT NULL UNIQUE CHECK(sequence >= 1),
                previous_version_sha256 TEXT,
                state_enc BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(statement)

    def _create_schema_v3(self) -> None:
        with self._transaction() as connection:
            self._create_core_tables(connection)
            self._insert_key_verifier(connection)
            connection.execute("PRAGMA user_version = 3")

    def _create_schema_v4(self) -> None:
        with self._transaction() as connection:
            self._create_core_tables(connection)
            self._create_v4_tables(connection)
            self._insert_key_verifier(connection)
            self._insert_audit_genesis(connection)
            connection.execute("PRAGMA user_version = 4")

    @staticmethod
    def _create_v5_tables(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE scraper_access_records (
                record_id TEXT NOT NULL PRIMARY KEY
                    CHECK(
                        length(record_id) BETWEEN 1 AND 128
                        AND record_id NOT GLOB '*[^A-Za-z0-9._-]*'
                    ),
                kind TEXT NOT NULL
                    CHECK(kind IN ('SOURCE','LOGIN','PROXY')),
                state_enc BLOB NOT NULL
                    CHECK(typeof(state_enc) = 'blob' AND length(state_enc) >= 29)
            )
            """
        )

    @staticmethod
    def _create_v6_tables(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE scraper_access_records (
                record_id TEXT NOT NULL PRIMARY KEY
                    CHECK(
                        length(record_id) BETWEEN 1 AND 128
                        AND record_id NOT GLOB '*[^A-Za-z0-9._-]*'
                    ),
                kind TEXT NOT NULL
                    CHECK(kind IN ('SOURCE','LOGIN','PROXY','CAPMONSTER')),
                state_enc BLOB NOT NULL
                    CHECK(typeof(state_enc) = 'blob' AND length(state_enc) >= 29)
            )
            """
        )

    def _create_schema_v5(self) -> None:
        with self._transaction() as connection:
            self._create_core_tables(connection)
            self._create_v4_tables(connection)
            self._create_v5_tables(connection)
            self._insert_key_verifier(connection)
            self._insert_audit_genesis(connection)
            connection.execute("PRAGMA user_version = 5")
        self._validate_v5_schema()

    def _create_schema_v6(self) -> None:
        with self._transaction() as connection:
            self._create_core_tables(connection)
            self._create_v4_tables(connection)
            self._create_v6_tables(connection)
            self._insert_key_verifier(connection)
            self._insert_audit_genesis(connection)
            connection.execute("PRAGMA user_version = 6")
        self._validate_v6_schema()

    def _insert_key_verifier(self, connection: sqlite3.Connection) -> None:
        verifier = self._encrypt(
            "vault_meta",
            _VERIFIER_NAME,
            "value",
            _VERIFIER_VALUE,
        )
        connection.execute(
            "INSERT INTO vault_meta (name, value_enc) VALUES (?, ?)",
            (_VERIFIER_NAME, verifier),
        )

    def _verify_key_verifier(self) -> None:
        connection = self._require_connection()
        try:
            row = connection.execute(
                "SELECT value_enc FROM vault_meta WHERE name = ?",
                (_VERIFIER_NAME,),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("vault key verifier table is missing") from exc
        if row is None:
            raise VaultAuthenticationError("vault key verifier is missing")
        plaintext = self._decrypt(
            "vault_meta",
            _VERIFIER_NAME,
            "value",
            row["value_enc"],
        )
        if not hmac.compare_digest(plaintext, _VERIFIER_VALUE):
            raise VaultAuthenticationError("vault key verifier is invalid")

    def _validate_v3_schema(self) -> None:
        connection = self._require_connection()
        try:
            capture_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(captures)").fetchall()
            }
            pair_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(pairs)").fetchall()
            }
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("vault authenticated schema is invalid") from exc
        if "state_enc" not in capture_columns or "state_enc" not in pair_columns:
            raise VaultSchemaError("vault authenticated state is missing")

    def _validate_v4_schema(self) -> None:
        connection = self._require_connection()
        try:
            rows = connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError(
                "vault authenticated v4 schema is invalid"
            ) from exc
        actual: dict[tuple[str, str, str], str] = {}
        for row in rows:
            object_type = row["type"]
            name = row["name"]
            table = row["tbl_name"]
            sql = row["sql"]
            if (
                type(object_type) is not str
                or type(name) is not str
                or type(table) is not str
                or type(sql) is not str
            ):
                raise VaultSchemaError(
                    "vault v4 schema object is not canonical"
                )
            key = (object_type, name, table)
            if key in actual:
                raise VaultSchemaError(
                    "vault v4 schema contains a duplicate object"
                )
            normalized = " ".join(sql.split())
            actual[key] = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()
        if actual != _V4_SCHEMA_DIGESTS:
            raise VaultSchemaError(
                "vault v4 schema objects or constraints differ "
                "from the approved schema"
            )

    def _validate_v5_schema(self) -> None:
        connection = self._require_connection()
        try:
            rows = connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError(
                "vault authenticated v5 schema is invalid"
            ) from exc
        actual: dict[tuple[str, str, str], str] = {}
        for row in rows:
            object_type = row["type"]
            name = row["name"]
            table = row["tbl_name"]
            sql = row["sql"]
            if (
                type(object_type) is not str
                or type(name) is not str
                or type(table) is not str
                or type(sql) is not str
            ):
                raise VaultSchemaError(
                    "vault v5 schema object is not canonical"
                )
            key = (object_type, name, table)
            if key in actual:
                raise VaultSchemaError(
                    "vault v5 schema contains a duplicate object"
                )
            normalized = " ".join(sql.split())
            actual[key] = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()
        if actual != _V5_SCHEMA_DIGESTS:
            raise VaultSchemaError(
                "vault v5 schema objects or constraints differ "
                "from the approved schema"
            )

    def _validate_v6_schema(self) -> None:
        connection = self._require_connection()
        try:
            rows = connection.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError(
                "vault authenticated v6 schema is invalid"
            ) from exc
        actual: dict[tuple[str, str, str], str] = {}
        for row in rows:
            object_type = row["type"]
            name = row["name"]
            table = row["tbl_name"]
            sql = row["sql"]
            if (
                type(object_type) is not str
                or type(name) is not str
                or type(table) is not str
                or type(sql) is not str
            ):
                raise VaultSchemaError(
                    "vault v6 schema object is not canonical"
                )
            key = (object_type, name, table)
            if key in actual:
                raise VaultSchemaError(
                    "vault v6 schema contains a duplicate object"
                )
            normalized = " ".join(sql.split())
            actual[key] = hashlib.sha256(
                normalized.encode("utf-8")
            ).hexdigest()
        if actual != _V6_SCHEMA_DIGESTS:
            raise VaultSchemaError(
                "vault v6 schema objects or constraints differ "
                "from the approved schema"
            )

    def _validate_v2_schema(self) -> None:
        connection = self._require_connection()
        try:
            capture_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(captures)").fetchall()
            }
            pair_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(pairs)").fetchall()
            }
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("legacy v2 schema is invalid") from exc
        if "state_enc" in capture_columns or "state_enc" not in pair_columns:
            raise VaultSchemaError("vault pair state authentication is missing")

    def _prepare_legacy_capture_states(
        self, *, require_encrypted_record: bool
    ) -> tuple[list[_PreparedCaptureState], dict[str, str]]:
        connection = self._require_connection()
        try:
            rows = connection.execute(
                "SELECT * FROM captures ORDER BY captured_at, id"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("legacy vault capture schema is invalid") from exc
        if require_encrypted_record and not rows:
            raise VaultAuthenticationError(
                "legacy vault has no encrypted record that can verify its key"
            )
        prepared: list[_PreparedCaptureState] = []
        verified_sources: dict[str, str] = {}
        for row in rows:
            row_id = str(row["id"])
            metadata = self._decrypt_json(
                "captures", row_id, "metadata", row["metadata_enc"]
            )
            if not isinstance(metadata, dict):
                raise VaultAuthenticationError(
                    "legacy capture metadata is not an object"
                )
            try:
                rebuilt = CapturedDocument.create(
                    blueprint_id=row["blueprint_id"],
                    url=self._decrypt_text(
                        "captures", row_id, "url", row["url_enc"]
                    ),
                    title=self._decrypt_text(
                        "captures", row_id, "title", row["title_enc"]
                    ),
                    raw_text=self._decrypt_text(
                        "captures", row_id, "raw_text", row["raw_text_enc"]
                    ),
                    metadata=metadata,
                )
            except (TypeError, ValueError) as exc:
                raise VaultAuthenticationError(
                    "legacy capture failed authenticated validation"
                ) from exc
            captured_at = row["captured_at"]
            if not isinstance(captured_at, str) or not captured_at.strip():
                raise VaultAuthenticationError(
                    "legacy capture timestamp is invalid"
                )
            quality_reasons = self._decrypt_json(
                "captures",
                row_id,
                "quality_reasons",
                row["quality_reasons_enc"],
            )
            if not isinstance(quality_reasons, list) or not all(
                isinstance(reason, str) for reason in quality_reasons
            ):
                raise VaultAuthenticationError(
                    "legacy capture quality state is invalid"
                )
            if row["quarantine_reason_enc"] is not None:
                self._decrypt_text(
                    "captures",
                    row_id,
                    "quarantine_reason",
                    row["quarantine_reason_enc"],
                )
            quality = assess_quality(rebuilt.raw_text)
            simhash_hex = f"{simhash64(rebuilt.raw_text):016x}"
            quarantine_reason_enc = self._encrypt_text(
                "captures",
                row_id,
                "quarantine_reason",
                _LEGACY_QUARANTINE_REASON,
            )
            state_enc = self._encrypt_capture_state(
                row_id,
                blueprint_id=rebuilt.blueprint_id,
                captured_at=captured_at,
                content_sha256=rebuilt.content_sha256,
                source_provenance_hash=rebuilt.source_provenance_hash,
                source_provenance_version=rebuilt.source_provenance_version,
                simhash_hex=simhash_hex,
                quality_score=quality.score,
                quality_reasons=quality.reasons,
                status=CAPTURE_QUARANTINED,
                quarantine_reason=_LEGACY_QUARANTINE_REASON,
            )
            prepared.append(
                _PreparedCaptureState(
                    capture_id=row_id,
                    content_sha256=rebuilt.content_sha256,
                    source_provenance_hash=rebuilt.source_provenance_hash,
                    simhash_hex=simhash_hex,
                    quality_score=quality.score,
                    quality_reasons_enc=self._encrypt_json(
                        "captures", row_id, "quality_reasons", quality.reasons
                    ),
                    status=CAPTURE_QUARANTINED,
                    quarantine_reason_enc=quarantine_reason_enc,
                    state_enc=state_enc,
                )
            )
            verified_sources[row_id] = rebuilt.source_provenance_hash
        return prepared, verified_sources

    def _prepare_legacy_pair_states(
        self,
        verified_sources: dict[str, str],
        *,
        authenticate_v2_state: bool,
    ) -> list[tuple[str, str, str, str, bytes]]:
        connection = self._require_connection()
        try:
            rows = connection.execute(
                "SELECT * FROM pairs ORDER BY created_at, id"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("legacy vault pair schema is invalid") from exc
        prepared: list[tuple[str, str, str, str, bytes]] = []
        for row in rows:
            pair_id = str(row["id"])
            source_id = str(row["source_id"])
            expected_provenance = verified_sources.get(source_id)
            indexed_provenance = str(row["source_provenance_hash"])
            if authenticate_v2_state:
                legacy_state = self._authenticated_pair_state_v2(row)
                indexed_provenance = str(
                    legacy_state["source_provenance_hash"]
                )
            if (
                expected_provenance is None
                or indexed_provenance != expected_provenance
            ):
                raise VaultAuthenticationError(
                    "legacy pair provenance does not match an authenticated source"
                )
            instruction = self._decrypt_text(
                "pairs", pair_id, "instruction", row["instruction_enc"]
            )
            response = self._decrypt_text(
                "pairs", pair_id, "response", row["response_enc"]
            )
            tags = self._decrypt_json("pairs", pair_id, "tags", row["tags_enc"])
            if not isinstance(tags, list) or not all(
                isinstance(tag, str) for tag in tags
            ):
                raise VaultAuthenticationError("legacy pair tags are invalid")
            created_at = row["created_at"]
            if not isinstance(created_at, str) or not created_at.strip():
                raise VaultAuthenticationError("legacy pair timestamp is invalid")
            try:
                DistilledPair.create(
                    source_id=source_id,
                    instruction=instruction,
                    response=response,
                    source_provenance_hash=expected_provenance,
                    tags=tuple(tags),
                )
            except (TypeError, ValueError) as exc:
                raise VaultAuthenticationError(
                    "legacy pair failed authenticated validation"
                ) from exc
            prepared.append(
                (
                    pair_id,
                    source_id,
                    expected_provenance,
                    created_at,
                    self._encrypt_pair_state(
                        pair_id,
                        source_id=source_id,
                        source_provenance_hash=expected_provenance,
                        user_approved=False,
                        status=PAIR_PENDING,
                        created_at=created_at,
                        reviewed_at=None,
                    ),
                )
            )
        return prepared

    @staticmethod
    def _apply_capture_state_updates(
        connection: sqlite3.Connection,
        capture_states: list[_PreparedCaptureState],
    ) -> None:
        for state in capture_states:
            connection.execute(
                """
                UPDATE captures
                SET content_sha256 = ?, source_provenance_hash = ?,
                    simhash_hex = ?, quality_score = ?, quality_reasons_enc = ?,
                    status = ?, quarantine_reason_enc = ?, state_enc = ?
                WHERE id = ?
                """,
                (
                    state.content_sha256,
                    state.source_provenance_hash,
                    state.simhash_hex,
                    state.quality_score,
                    state.quality_reasons_enc,
                    state.status,
                    state.quarantine_reason_enc,
                    state.state_enc,
                    state.capture_id,
                ),
            )

    @staticmethod
    def _apply_pair_state_updates(
        connection: sqlite3.Connection,
        pair_states: list[tuple[str, str, str, str, bytes]],
    ) -> None:
        for pair_id, source_id, provenance, created_at, state_enc in pair_states:
            connection.execute(
                """
                UPDATE pairs
                SET source_id = ?, source_provenance_hash = ?,
                    user_approved = 0, status = ?, created_at = ?,
                    reviewed_at = NULL, state_enc = ?
                WHERE id = ?
                """,
                (
                    source_id,
                    provenance,
                    PAIR_PENDING,
                    created_at,
                    state_enc,
                    pair_id,
                ),
            )

    def _migrate_v1_to_v3(
        self,
        capture_states: list[_PreparedCaptureState],
        pair_states: list[tuple[str, str, str, str, bytes]],
    ) -> None:
        try:
            with self._transaction() as connection:
                connection.execute(
                    """
                    CREATE TABLE vault_meta (
                        name TEXT PRIMARY KEY,
                        value_enc BLOB NOT NULL,
                        CHECK (name = 'key_verifier')
                    )
                    """
                )
                connection.execute("ALTER TABLE captures ADD COLUMN state_enc BLOB")
                connection.execute("ALTER TABLE pairs ADD COLUMN state_enc BLOB")
                self._apply_capture_state_updates(connection, capture_states)
                self._apply_pair_state_updates(connection, pair_states)
                self._insert_key_verifier(connection)
                connection.execute("PRAGMA user_version = 3")
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("legacy vault migration failed") from exc

    def _migrate_v2_to_v3(
        self,
        capture_states: list[_PreparedCaptureState],
        pair_states: list[tuple[str, str, str, str, bytes]],
    ) -> None:
        try:
            with self._transaction() as connection:
                connection.execute("ALTER TABLE captures ADD COLUMN state_enc BLOB")
                self._apply_capture_state_updates(connection, capture_states)
                self._apply_pair_state_updates(connection, pair_states)
                connection.execute("PRAGMA user_version = 3")
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("legacy v2 vault migration failed") from exc

    def _protect_database_files(self) -> None:
        self._fixed.private_modes()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise VaultError("vault is closed")
        return self._connection

    def _require_key(self) -> bytes:
        if self._key is None:
            raise VaultError("vault is closed")
        return self._key

    @staticmethod
    def _aad(table: str, row_id: str, field: str) -> bytes:
        return f"lived-lora-studio:v1:{table}:{row_id}:{field}".encode("utf-8")

    def _encrypt(self, table: str, row_id: str, field: str, plaintext: bytes) -> bytes:
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(self._require_key()).encrypt(
            nonce,
            plaintext,
            self._aad(table, row_id, field),
        )
        return _CIPHER_VERSION + nonce + ciphertext

    def _decrypt(self, table: str, row_id: str, field: str, payload: bytes) -> bytes:
        try:
            encoded = bytes(payload)
            if len(encoded) < 1 + _NONCE_BYTES + 16:
                raise InvalidTag
            if encoded[:1] != _CIPHER_VERSION:
                raise InvalidTag
            nonce = encoded[1 : 1 + _NONCE_BYTES]
            ciphertext = encoded[1 + _NONCE_BYTES :]
            return AESGCM(self._require_key()).decrypt(
                nonce,
                ciphertext,
                self._aad(table, row_id, field),
            )
        except (InvalidTag, TypeError, ValueError) as exc:
            raise VaultAuthenticationError(
                f"encrypted {table}.{field} failed authentication"
            ) from exc

    def _encrypt_text(self, table: str, row_id: str, field: str, value: str) -> bytes:
        return self._encrypt(table, row_id, field, value.encode("utf-8"))

    def _decrypt_text(self, table: str, row_id: str, field: str, value: bytes) -> str:
        try:
            return self._decrypt(table, row_id, field, value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VaultAuthenticationError(
                f"encrypted {table}.{field} is not valid UTF-8"
            ) from exc

    def _encrypt_json(self, table: str, row_id: str, field: str, value) -> bytes:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return self._encrypt(table, row_id, field, encoded)

    def _decrypt_json(self, table: str, row_id: str, field: str, value: bytes):
        try:
            return json.loads(self._decrypt(table, row_id, field, value))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VaultAuthenticationError(
                f"encrypted {table}.{field} is not valid JSON"
            ) from exc

    @staticmethod
    def _canonical_ledger_bytes(value: object) -> bytes:
        try:
            return (
                json.dumps(
                    value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise VaultAuthenticationError(
                "ledger state is not canonical JSON"
            ) from exc

    @classmethod
    def _ledger_digest(cls, domain: str, value: object) -> str:
        return hashlib.sha256(
            f"liveD:vault-v4:{domain}\0".encode("ascii")
            + cls._canonical_ledger_bytes(value)
        ).hexdigest()

    @staticmethod
    def _require_sha256_or_none(value: object, name: str) -> str | None:
        if value is None:
            return None
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise VaultAuthenticationError(f"{name} must be lowercase SHA-256 or null")
        return value

    @staticmethod
    def _parse_utc_timestamp(value: object, name: str) -> datetime:
        if type(value) is not str or not value or len(value.encode("utf-8")) > 64:
            raise VaultAuthenticationError(f"{name} must be a bounded UTC timestamp")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise VaultAuthenticationError(f"{name} is not valid ISO-8601") from exc
        if (
            parsed.tzinfo is None
            or parsed.utcoffset() is None
            or parsed.utcoffset() != timedelta(0)
            or parsed.isoformat() != value
        ):
            raise VaultAuthenticationError(
                f"{name} must be canonical ISO-8601 with a zero UTC offset"
            )
        return parsed

    @staticmethod
    def _now_utc_text() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _audit_state_plain(
        *,
        sequence: int,
        head_sha256: str | None,
        review_head_sha256: str | None,
        tombstone_head_sha256: str | None,
        revision_head_sha256: str | None,
        target_head_sha256: str | None,
        capture_sequence: int,
        capture_head_sha256: str | None,
    ) -> dict[str, object]:
        return {
            "capture_head_sha256": capture_head_sha256,
            "capture_sequence": capture_sequence,
            "head_sha256": head_sha256,
            "review_head_sha256": review_head_sha256,
            "revision_head_sha256": revision_head_sha256,
            "schema": "LIVED_VAULT_AUDIT_STATE_V1",
            "sequence": sequence,
            "target_head_sha256": target_head_sha256,
            "tombstone_head_sha256": tombstone_head_sha256,
        }

    def _insert_audit_genesis(self, connection: sqlite3.Connection) -> None:
        state = self._audit_state_plain(
            sequence=0,
            head_sha256=None,
            review_head_sha256=None,
            tombstone_head_sha256=None,
            revision_head_sha256=None,
            target_head_sha256=None,
            capture_sequence=0,
            capture_head_sha256=None,
        )
        connection.execute(
            """
            INSERT INTO audit_state (
                singleton_id, sequence, head_sha256, review_head_sha256,
                tombstone_head_sha256, revision_head_sha256,
                target_head_sha256, capture_sequence,
                capture_head_sha256, state_enc
            ) VALUES (1, 0, NULL, NULL, NULL, NULL, NULL, 0, NULL, ?)
            """,
            (self._encrypt_json("audit_state", "singleton", "state", state),),
        )

    def _load_audit_state(self, connection: sqlite3.Connection) -> dict[str, object]:
        rows = connection.execute("SELECT * FROM audit_state").fetchall()
        if len(rows) != 1 or rows[0]["singleton_id"] != 1:
            raise VaultAuthenticationError(
                "vault audit anchor must contain exactly one singleton"
            )
        row = rows[0]
        state = self._decrypt_json(
            "audit_state", "singleton", "state", row["state_enc"]
        )
        expected_keys = {
            "capture_head_sha256",
            "capture_sequence",
            "head_sha256",
            "review_head_sha256",
            "revision_head_sha256",
            "schema",
            "sequence",
            "target_head_sha256",
            "tombstone_head_sha256",
        }
        if type(state) is not dict or set(state) != expected_keys:
            raise VaultAuthenticationError("vault audit anchor state is invalid")
        if state["schema"] != "LIVED_VAULT_AUDIT_STATE_V1":
            raise VaultAuthenticationError("vault audit anchor schema is invalid")
        sequence = state["sequence"]
        capture_sequence = state["capture_sequence"]
        if type(sequence) is not int or not 0 <= sequence <= 9_223_372_036_854_775_807:
            raise VaultAuthenticationError("vault audit sequence is invalid")
        if (
            type(capture_sequence) is not int
            or not 0 <= capture_sequence <= 9_223_372_036_854_775_807
        ):
            raise VaultAuthenticationError("vault capture sequence is invalid")
        for field in (
            "head_sha256",
            "review_head_sha256",
            "tombstone_head_sha256",
            "revision_head_sha256",
            "target_head_sha256",
            "capture_head_sha256",
        ):
            self._require_sha256_or_none(state[field], field)
        if (
            row["sequence"] != sequence
            or row["head_sha256"] != state["head_sha256"]
            or row["review_head_sha256"] != state["review_head_sha256"]
            or row["tombstone_head_sha256"] != state["tombstone_head_sha256"]
            or row["revision_head_sha256"] != state["revision_head_sha256"]
            or row["target_head_sha256"] != state["target_head_sha256"]
            or row["capture_sequence"] != capture_sequence
            or row["capture_head_sha256"] != state["capture_head_sha256"]
        ):
            raise VaultAuthenticationError(
                "vault audit anchor columns do not match authenticated state"
            )
        if sequence == 0 and any(
            state[field] is not None
            for field in (
                "head_sha256",
                "review_head_sha256",
                "tombstone_head_sha256",
                "revision_head_sha256",
                "target_head_sha256",
            )
        ):
            raise VaultAuthenticationError("vault audit genesis is invalid")
        if sequence > 0 and state["head_sha256"] is None:
            raise VaultAuthenticationError("vault audit head is missing")
        if (capture_sequence == 0) != (state["capture_head_sha256"] is None):
            raise VaultAuthenticationError("vault capture audit head is invalid")
        return state

    def _write_audit_state(
        self, connection: sqlite3.Connection, state: dict[str, object]
    ) -> None:
        expected = self._audit_state_plain(
            sequence=int(state["sequence"]),
            head_sha256=state["head_sha256"],
            review_head_sha256=state["review_head_sha256"],
            tombstone_head_sha256=state["tombstone_head_sha256"],
            revision_head_sha256=state["revision_head_sha256"],
            target_head_sha256=state["target_head_sha256"],
            capture_sequence=int(state["capture_sequence"]),
            capture_head_sha256=state["capture_head_sha256"],
        )
        if state != expected:
            raise VaultAuthenticationError("refusing to publish malformed audit state")
        cursor = connection.execute(
            """
            UPDATE audit_state
            SET sequence = ?, head_sha256 = ?, review_head_sha256 = ?,
                tombstone_head_sha256 = ?, revision_head_sha256 = ?,
                target_head_sha256 = ?, capture_sequence = ?,
                capture_head_sha256 = ?, state_enc = ?
            WHERE singleton_id = 1
            """,
            (
                state["sequence"],
                state["head_sha256"],
                state["review_head_sha256"],
                state["tombstone_head_sha256"],
                state["revision_head_sha256"],
                state["target_head_sha256"],
                state["capture_sequence"],
                state["capture_head_sha256"],
                self._encrypt_json("audit_state", "singleton", "state", state),
            ),
        )
        if cursor.rowcount != 1:
            raise VaultAuthenticationError("vault audit anchor disappeared")

    @staticmethod
    def _record_token(record_type: str, record_id: str, digest: str) -> tuple[str, str, str]:
        if type(record_type) is not str or not record_type:
            raise ValueError("record type must be nonempty")
        if type(record_id) is not str or not record_id:
            raise ValueError("record id must be nonempty")
        if type(digest) is not str or _SHA256.fullmatch(digest) is None:
            raise ValueError("record digest must be lowercase SHA-256")
        return record_type, record_id, digest

    @staticmethod
    def _normalized_record_tokens(
        values: Sequence[tuple[str, str, str]],
    ) -> tuple[tuple[str, str, str], ...]:
        normalized = tuple(sorted(values))
        if len(set(normalized)) != len(normalized):
            raise VaultAuthenticationError("audit operation contains duplicate records")
        return normalized

    def _audit_context(
        self, connection: sqlite3.Connection, *, verify: bool = True
    ) -> tuple[int, dict[str, object]]:
        if verify:
            self._verify_v4_ledger(connection)
        anchor = self._load_audit_state(connection)
        sequence = int(anchor["sequence"]) + 1
        if sequence > 9_223_372_036_854_775_807:
            raise VaultError("vault audit sequence is exhausted")
        return sequence, anchor

    def _publish_audit_operation(
        self,
        connection: sqlite3.Connection,
        *,
        sequence: int,
        kind: str,
        subject_id: str,
        created_at: str,
        record_digests: Sequence[tuple[str, str, str]],
        head_updates: dict[str, str],
    ) -> str:
        anchor = self._load_audit_state(connection)
        if sequence != int(anchor["sequence"]) + 1:
            raise VaultAuthenticationError("audit sequence changed during transaction")
        if (
            type(kind) is not str
            or not kind.isascii()
            or not 1 <= len(kind) <= 64
            or not kind.replace("_", "").isalnum()
        ):
            raise ValueError("audit kind must be a bounded ASCII identifier")
        if type(subject_id) is not str or not 1 <= len(subject_id.encode("utf-8")) <= 128:
            raise ValueError("audit subject id is invalid")
        self._parse_utc_timestamp(created_at, "audit event created_at")
        tokens = self._normalized_record_tokens(record_digests)
        allowed_heads = {
            "review_head_sha256",
            "tombstone_head_sha256",
            "revision_head_sha256",
            "target_head_sha256",
        }
        if not set(head_updates).issubset(allowed_heads):
            raise ValueError("unknown audit head update")
        for value in head_updates.values():
            if _SHA256.fullmatch(value) is None:
                raise ValueError("audit head update must be lowercase SHA-256")
        event_id = uuid4().hex
        state = {
            "created_at": created_at,
            "event_id": event_id,
            "kind": kind,
            "previous_event_sha256": anchor["head_sha256"],
            "record_digests": [list(token) for token in tokens],
            "sequence": sequence,
            "subject_id": subject_id,
        }
        event_digest = self._ledger_digest("audit-event", state)
        connection.execute(
            """
            INSERT INTO audit_events (
                sequence, event_id, kind, subject_id,
                previous_event_sha256, state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                event_id,
                kind,
                subject_id,
                anchor["head_sha256"],
                self._encrypt_json("audit_events", event_id, "state", state),
                created_at,
            ),
        )
        next_state = dict(anchor)
        next_state["sequence"] = sequence
        next_state["head_sha256"] = event_digest
        next_state.update(head_updates)
        self._write_audit_state(connection, next_state)
        return event_digest

    def _authenticate_audit_event(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], str]:
        event_id = str(row["event_id"])
        state = self._decrypt_json(
            "audit_events", event_id, "state", row["state_enc"]
        )
        if type(state) is not dict or set(state) != {
            "created_at",
            "event_id",
            "kind",
            "previous_event_sha256",
            "record_digests",
            "sequence",
            "subject_id",
        }:
            raise VaultAuthenticationError("audit event state is invalid")
        sequence = state["sequence"]
        kind = state["kind"]
        subject_id = state["subject_id"]
        if type(sequence) is not int or sequence < 1:
            raise VaultAuthenticationError("audit event sequence is invalid")
        if type(kind) is not str or not kind.isascii() or not 1 <= len(kind) <= 64:
            raise VaultAuthenticationError("audit event kind is invalid")
        if type(subject_id) is not str or not 1 <= len(subject_id.encode("utf-8")) <= 128:
            raise VaultAuthenticationError("audit event subject is invalid")
        previous = self._require_sha256_or_none(
            state["previous_event_sha256"], "audit previous head"
        )
        self._parse_utc_timestamp(state["created_at"], "audit event created_at")
        encoded_tokens = state["record_digests"]
        if type(encoded_tokens) is not list or len(encoded_tokens) > 16:
            raise VaultAuthenticationError("audit record digest list is invalid")
        tokens: list[tuple[str, str, str]] = []
        for token in encoded_tokens:
            if type(token) is not list or len(token) != 3:
                raise VaultAuthenticationError("audit record digest entry is invalid")
            try:
                tokens.append(self._record_token(token[0], token[1], token[2]))
            except (TypeError, ValueError) as exc:
                raise VaultAuthenticationError("audit record digest entry is invalid") from exc
        if tuple(tokens) != self._normalized_record_tokens(tokens):
            raise VaultAuthenticationError("audit record digests are not canonical")
        if (
            event_id != state["event_id"]
            or row["sequence"] != sequence
            or row["kind"] != kind
            or row["subject_id"] != subject_id
            or row["previous_event_sha256"] != previous
            or row["created_at"] != state["created_at"]
        ):
            raise VaultAuthenticationError(
                "audit event columns do not match authenticated state"
            )
        return state, self._ledger_digest("audit-event", state)

    @staticmethod
    def _require_local_state_scope(scope: object) -> str:
        if type(scope) is not str or _LOCAL_STATE_SCOPE.fullmatch(scope) is None:
            raise ValueError(
                "local-state anchor scope must be a bounded lowercase ASCII identifier"
            )
        return scope

    @staticmethod
    def _require_local_state_count(count: object, *, allow_zero: bool) -> int:
        minimum = 0 if allow_zero else 1
        if type(count) is not int or not minimum <= count <= _MAX_AUDIT_SEQUENCE:
            label = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"local-state anchor count must be a {label} integer")
        return count

    @staticmethod
    def _require_local_state_head(head_sha256: object, *, allow_none: bool) -> str | None:
        if head_sha256 is None and allow_none:
            return None
        if type(head_sha256) is not str or _SHA256.fullmatch(head_sha256) is None:
            raise ValueError("local-state anchor head must be lowercase SHA-256")
        return head_sha256

    @classmethod
    def _local_state_anchor_subject(
        cls, scope: object, count: object, head_sha256: object
    ) -> str:
        canonical_scope = cls._require_local_state_scope(scope)
        canonical_count = cls._require_local_state_count(count, allow_zero=False)
        canonical_head = cls._require_local_state_head(head_sha256, allow_none=False)
        subject = f"{canonical_scope}|{canonical_count}|{canonical_head}"
        if len(subject.encode("ascii")) > 128:
            raise ValueError("local-state anchor subject exceeds the audit bound")
        return subject

    @classmethod
    def _local_state_anchor_from_audit(
        cls, state: dict[str, object], event_sha256: str
    ) -> LocalStateAnchor:
        try:
            subject = state["subject_id"]
            if type(subject) is not str:
                raise ValueError("subject must be a string")
            fields = subject.split("|")
            if len(fields) != 3 or not fields[1].isascii() or not fields[1].isdigit():
                raise ValueError("subject is not canonical")
            count = int(fields[1], 10)
            anchor = LocalStateAnchor(
                scope=fields[0],
                count=count,
                head_sha256=fields[2],
                audit_sequence=int(state["sequence"]),
                audit_event_sha256=event_sha256,
            )
            if subject != cls._local_state_anchor_subject(
                anchor.scope, anchor.count, anchor.head_sha256
            ):
                raise ValueError("subject is not canonical")
            return anchor
        except (KeyError, TypeError, ValueError) as exc:
            raise VaultAuthenticationError(
                "authenticated local-state anchor event is invalid"
            ) from exc

    @classmethod
    def _local_state_anchor_histories(
        cls, ledger: dict[str, object]
    ) -> dict[str, tuple[LocalStateAnchor, ...]]:
        mutable: dict[str, list[LocalStateAnchor]] = {}
        for state, event_sha256 in ledger["audit_records"]:
            if state["kind"] != _LOCAL_STATE_ANCHOR_KIND:
                continue
            anchor = cls._local_state_anchor_from_audit(state, event_sha256)
            history = mutable.setdefault(anchor.scope, [])
            if history:
                previous = history[-1]
                if (
                    anchor.count != previous.count + 1
                    or anchor.head_sha256 == previous.head_sha256
                ):
                    raise VaultAuthenticationError(
                        "authenticated local-state anchor history is not contiguous"
                    )
            history.append(anchor)
        return {scope: tuple(history) for scope, history in mutable.items()}

    def _capture_record_digest(
        self,
        state: dict[str, object],
        document: CapturedDocument,
    ) -> str:
        material = {
            "document": {
                "blueprint_id": document.blueprint_id,
                "captured_at": document.captured_at,
                "content_sha256": document.content_sha256,
                "id": document.id,
                "metadata": thaw_json_value(document.metadata),
                "raw_text": document.raw_text,
                "source_provenance_hash": document.source_provenance_hash,
                "source_provenance_version": document.source_provenance_version,
                "title": document.title,
                "url": document.url,
            },
            "state": state,
        }
        return self._ledger_digest("capture-state", material)

    def _append_capture_event(
        self,
        connection: sqlite3.Connection,
        *,
        capture_id: str,
        kind: str,
        created_at: str,
    ) -> str:
        if kind not in {
            "CREATED",
            "MIGRATED",
            "STATUS_CHANGED",
            "QUARANTINED",
        }:
            raise ValueError("unknown capture event kind")
        self._parse_utc_timestamp(created_at, "capture event created_at")
        row = connection.execute(
            "SELECT * FROM captures WHERE id = ?", (capture_id,)
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"capture not found: {capture_id}")
        capture_state, document = self._authenticate_capture_row(row)
        capture_state_sha256 = self._capture_record_digest(
            capture_state, document
        )
        anchor = self._load_audit_state(connection)
        capture_sequence = int(anchor["capture_sequence"]) + 1
        if capture_sequence > 9_223_372_036_854_775_807:
            raise VaultError("vault capture sequence is exhausted")
        event_id = uuid4().hex
        state = {
            "capture_id": capture_id,
            "capture_sequence": capture_sequence,
            "capture_state_sha256": capture_state_sha256,
            "created_at": created_at,
            "event_id": event_id,
            "kind": kind,
            "previous_event_sha256": anchor["capture_head_sha256"],
        }
        digest = self._ledger_digest("capture-event", state)
        connection.execute(
            """
            INSERT INTO capture_events (
                event_id, capture_sequence, capture_id, kind,
                capture_state_sha256, previous_event_sha256,
                state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                capture_sequence,
                capture_id,
                kind,
                capture_state_sha256,
                anchor["capture_head_sha256"],
                self._encrypt_json("capture_events", event_id, "state", state),
                created_at,
            ),
        )
        next_anchor = dict(anchor)
        next_anchor["capture_sequence"] = capture_sequence
        next_anchor["capture_head_sha256"] = digest
        self._write_audit_state(connection, next_anchor)
        return digest

    def _authenticate_capture_event(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], str]:
        event_id = str(row["event_id"])
        state = self._decrypt_json(
            "capture_events", event_id, "state", row["state_enc"]
        )
        expected_keys = {
            "capture_id",
            "capture_sequence",
            "capture_state_sha256",
            "created_at",
            "event_id",
            "kind",
            "previous_event_sha256",
        }
        if type(state) is not dict or set(state) != expected_keys:
            raise VaultAuthenticationError("capture event state is invalid")
        if (
            type(state["capture_sequence"]) is not int
            or state["capture_sequence"] < 1
        ):
            raise VaultAuthenticationError("capture event sequence is invalid")
        if (
            type(state["event_id"]) is not str
            or not state["event_id"]
            or type(state["capture_id"]) is not str
            or not state["capture_id"]
        ):
            raise VaultAuthenticationError("capture event identifier is invalid")
        if state["kind"] not in {
            "CREATED",
            "MIGRATED",
            "STATUS_CHANGED",
            "QUARANTINED",
        }:
            raise VaultAuthenticationError("capture event kind is invalid")
        if (
            type(state["capture_state_sha256"]) is not str
            or _SHA256.fullmatch(state["capture_state_sha256"]) is None
        ):
            raise VaultAuthenticationError("capture state digest is invalid")
        previous = self._require_sha256_or_none(
            state["previous_event_sha256"], "previous capture event head"
        )
        self._parse_utc_timestamp(state["created_at"], "capture event created_at")
        if (
            row["event_id"] != state["event_id"]
            or row["capture_sequence"] != state["capture_sequence"]
            or row["capture_id"] != state["capture_id"]
            or row["kind"] != state["kind"]
            or row["capture_state_sha256"] != state["capture_state_sha256"]
            or row["previous_event_sha256"] != previous
            or row["created_at"] != state["created_at"]
        ):
            raise VaultAuthenticationError(
                "capture event columns do not match authenticated state"
            )
        return state, self._ledger_digest("capture-event", state)

    def _authenticate_revision_row(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], dict[str, object], str]:
        revision_id = str(row["revision_id"])
        payload = self._decrypt_json(
            "pair_revisions", revision_id, "payload", row["payload_enc"]
        )
        state = self._decrypt_json(
            "pair_revisions", revision_id, "state", row["state_enc"]
        )
        if type(payload) is not dict or set(payload) != {
            "instruction", "response", "tags"
        }:
            raise VaultAuthenticationError("pair revision payload is invalid")
        if (
            type(payload["instruction"]) is not str
            or type(payload["response"]) is not str
            or type(payload["tags"]) is not list
            or not all(type(tag) is str for tag in payload["tags"])
        ):
            raise VaultAuthenticationError("pair revision payload is invalid")
        expected_keys = {
            "created_at",
            "lineage_id",
            "pair_id",
            "parent_revision_id",
            "payload_sha256",
            "previous_revision_sha256",
            "revision",
            "revision_id",
            "sequence",
            "source_id",
            "source_provenance_hash",
        }
        if type(state) is not dict or set(state) != expected_keys:
            raise VaultAuthenticationError("pair revision state is invalid")
        for key in ("revision_id", "pair_id", "lineage_id", "source_id"):
            if type(state[key]) is not str or not state[key]:
                raise VaultAuthenticationError("pair revision identifier is invalid")
        if state["parent_revision_id"] is not None and (
            type(state["parent_revision_id"]) is not str
            or not state["parent_revision_id"]
        ):
            raise VaultAuthenticationError("pair revision parent is invalid")
        if type(state["revision"]) is not int or state["revision"] < 1:
            raise VaultAuthenticationError("pair revision number is invalid")
        if type(state["sequence"]) is not int or state["sequence"] < 1:
            raise VaultAuthenticationError("pair revision sequence is invalid")
        self._parse_utc_timestamp(state["created_at"], "pair revision created_at")
        if _SHA256.fullmatch(str(state["source_provenance_hash"])) is None:
            raise VaultAuthenticationError("pair revision provenance is invalid")
        self._require_sha256_or_none(
            state["previous_revision_sha256"], "previous revision head"
        )
        payload_sha256 = hashlib.sha256(
            self._canonical_ledger_bytes(payload)
        ).hexdigest()
        if state["payload_sha256"] != payload_sha256:
            raise VaultAuthenticationError("pair revision payload digest is invalid")
        if (
            row["revision_id"] != state["revision_id"]
            or row["pair_id"] != state["pair_id"]
            or row["lineage_id"] != state["lineage_id"]
            or row["parent_revision_id"] != state["parent_revision_id"]
            or row["revision"] != state["revision"]
            or row["sequence"] != state["sequence"]
            or row["created_at"] != state["created_at"]
        ):
            raise VaultAuthenticationError(
                "pair revision columns do not match authenticated state"
            )
        try:
            checked = DistilledPair.create(
                source_id=str(state["source_id"]),
                instruction=payload["instruction"],
                response=payload["response"],
                source_provenance_hash=str(state["source_provenance_hash"]),
                tags=tuple(payload["tags"]),
            )
        except (TypeError, ValueError) as exc:
            raise VaultAuthenticationError("pair revision failed validation") from exc
        if (
            checked.instruction != payload["instruction"]
            or checked.response != payload["response"]
            or checked.tags != tuple(payload["tags"])
        ):
            raise VaultAuthenticationError("pair revision payload is not canonical")
        digest = self._ledger_digest(
            "pair-revision", {"payload": payload, "state": state}
        )
        return state, payload, digest

    def _authenticate_review_event(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], str]:
        event_id = str(row["event_id"])
        state = self._decrypt_json(
            "review_events", event_id, "state", row["state_enc"]
        )
        expected_keys = {
            "created_at",
            "event_id",
            "kind",
            "pair_id",
            "previous_event_sha256",
            "reason_code",
            "revision_id",
            "sequence",
        }
        if type(state) is not dict or set(state) != expected_keys:
            raise VaultAuthenticationError("review event state is invalid")
        kind = state["kind"]
        reason = state["reason_code"]
        expected_reason = {
            "MODIFIED": OPERATOR_MODIFIED,
            "APPROVED": OPERATOR_APPROVED,
            "DENIED": OPERATOR_REJECTED,
        }.get(kind)
        try:
            if expected_reason is None:
                raise ValueError("unknown review event kind")
            require_review_reason(reason, expected_reason)
        except (TypeError, ValueError) as error:
            raise VaultAuthenticationError("review event kind and reason are invalid")
        if type(state["sequence"]) is not int or state["sequence"] < 1:
            raise VaultAuthenticationError("review event sequence is invalid")
        for key in ("event_id", "pair_id", "revision_id"):
            if type(state[key]) is not str or not state[key]:
                raise VaultAuthenticationError("review event identifier is invalid")
        previous = self._require_sha256_or_none(
            state["previous_event_sha256"], "previous review head"
        )
        self._parse_utc_timestamp(state["created_at"], "review event created_at")
        if (
            row["event_id"] != state["event_id"]
            or row["sequence"] != state["sequence"]
            or row["pair_id"] != state["pair_id"]
            or row["revision_id"] != state["revision_id"]
            or row["kind"] != kind
            or row["reason_code"] != reason
            or row["previous_event_sha256"] != previous
            or row["created_at"] != state["created_at"]
        ):
            raise VaultAuthenticationError(
                "review event columns do not match authenticated state"
            )
        return state, self._ledger_digest("review-event", state)

    def _authenticate_target_row(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], SourceTargetView, str]:
        target_id = str(row["target_id"])
        state = self._decrypt_json(
            "source_targets", target_id, "state", row["state_enc"]
        )
        expected_keys = {
            "blueprint_id",
            "categories",
            "created_at",
            "expected_yield",
            "label",
            "last_accepted_at",
            "operator_authorized",
            "previous_target_sha256",
            "sequence",
            "start_url",
            "target_id",
        }
        if type(state) is not dict or set(state) != expected_keys:
            raise VaultAuthenticationError("source target state is invalid")
        if type(state["sequence"]) is not int or state["sequence"] < 1:
            raise VaultAuthenticationError("source target sequence is invalid")
        self._parse_utc_timestamp(state["created_at"], "source target created_at")
        self._require_sha256_or_none(
            state["previous_target_sha256"], "previous target head"
        )
        try:
            last_accepted = (
                None
                if state["last_accepted_at"] is None
                else self._parse_utc_timestamp(
                    state["last_accepted_at"], "target last_accepted_at"
                )
            )
            view = SourceTargetView(
                target_id=state["target_id"],
                blueprint_id=state["blueprint_id"],
                label=state["label"],
                start_url=state["start_url"],
                categories=tuple(state["categories"]),
                expected_yield=state["expected_yield"],
                operator_authorized=state["operator_authorized"],
                last_accepted_at=last_accepted,
            )
        except (TypeError, ValueError) as exc:
            raise VaultAuthenticationError("source target failed validation") from exc
        if (
            row["target_id"] != state["target_id"]
            or row["sequence"] != state["sequence"]
            or row["blueprint_id"] != state["blueprint_id"]
            or row["created_at"] != state["created_at"]
        ):
            raise VaultAuthenticationError(
                "source target columns do not match authenticated state"
            )
        return state, view, self._ledger_digest("source-target", state)

    def _authenticate_tombstone_row(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], str]:
        tombstone_id = str(row["tombstone_id"])
        state = self._decrypt_json(
            "tombstones", tombstone_id, "state", row["state_enc"]
        )
        expected_keys = {
            "created_at",
            "previous_tombstone_sha256",
            "reason_code",
            "record_id",
            "record_kind",
            "sequence",
            "tombstone_id",
        }
        if type(state) is not dict or set(state) != expected_keys:
            raise VaultAuthenticationError("tombstone state is invalid")
        if state["record_kind"] not in {"PAIR", "CAPTURE"}:
            raise VaultAuthenticationError("tombstone record kind is invalid")
        if state["reason_code"] != OPERATOR_DELETED:
            raise VaultAuthenticationError("tombstone reason is invalid")
        if type(state["sequence"]) is not int or state["sequence"] < 1:
            raise VaultAuthenticationError("tombstone sequence is invalid")
        for key in ("tombstone_id", "record_id"):
            if type(state[key]) is not str or not state[key]:
                raise VaultAuthenticationError("tombstone identifier is invalid")
        previous = self._require_sha256_or_none(
            state["previous_tombstone_sha256"], "previous tombstone head"
        )
        self._parse_utc_timestamp(state["created_at"], "tombstone created_at")
        if (
            row["tombstone_id"] != state["tombstone_id"]
            or row["sequence"] != state["sequence"]
            or row["record_kind"] != state["record_kind"]
            or row["record_id"] != state["record_id"]
            or row["reason_code"] != state["reason_code"]
            or row["previous_tombstone_sha256"] != previous
            or row["created_at"] != state["created_at"]
        ):
            raise VaultAuthenticationError(
                "tombstone columns do not match authenticated state"
            )
        return state, self._ledger_digest("tombstone", state)
    def _insert_revision_record(
        self,
        connection: sqlite3.Connection,
        *,
        pair: DistilledPair,
        revision: int,
        parent_revision_id: str | None,
        sequence: int,
        created_at: str,
        previous_revision_sha256: str | None,
    ) -> tuple[str, str]:
        self._parse_utc_timestamp(created_at, "pair revision created_at")
        revision_id = uuid4().hex
        payload = {
            "instruction": pair.instruction,
            "response": pair.response,
            "tags": list(pair.tags),
        }
        state = {
            "created_at": created_at,
            "lineage_id": pair.id,
            "pair_id": pair.id,
            "parent_revision_id": parent_revision_id,
            "payload_sha256": hashlib.sha256(
                self._canonical_ledger_bytes(payload)
            ).hexdigest(),
            "previous_revision_sha256": previous_revision_sha256,
            "revision": revision,
            "revision_id": revision_id,
            "sequence": sequence,
            "source_id": pair.source_id,
            "source_provenance_hash": pair.source_provenance_hash,
        }
        digest = self._ledger_digest(
            "pair-revision", {"payload": payload, "state": state}
        )
        connection.execute(
            """
            INSERT INTO pair_revisions (
                revision_id, pair_id, lineage_id, parent_revision_id,
                revision, sequence, payload_enc, state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                pair.id,
                pair.id,
                parent_revision_id,
                revision,
                sequence,
                self._encrypt_json(
                    "pair_revisions", revision_id, "payload", payload
                ),
                self._encrypt_json(
                    "pair_revisions", revision_id, "state", state
                ),
                created_at,
            ),
        )
        return revision_id, digest

    def _insert_review_record(
        self,
        connection: sqlite3.Connection,
        *,
        pair_id: str,
        revision_id: str,
        sequence: int,
        kind: str,
        reason_code: str,
        created_at: str,
        previous_review_sha256: str | None,
    ) -> tuple[str, str]:
        expected = {
            "MODIFIED": OPERATOR_MODIFIED,
            "APPROVED": OPERATOR_APPROVED,
            "DENIED": OPERATOR_REJECTED,
        }.get(kind)
        if expected is None:
            raise ValueError("unknown review event kind")
        require_review_reason(reason_code, expected)
        self._parse_utc_timestamp(created_at, "review event created_at")
        event_id = uuid4().hex
        state = {
            "created_at": created_at,
            "event_id": event_id,
            "kind": kind,
            "pair_id": pair_id,
            "previous_event_sha256": previous_review_sha256,
            "reason_code": reason_code,
            "revision_id": revision_id,
            "sequence": sequence,
        }
        digest = self._ledger_digest("review-event", state)
        connection.execute(
            """
            INSERT INTO review_events (
                event_id, sequence, pair_id, revision_id, kind,
                reason_code, previous_event_sha256, state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                sequence,
                pair_id,
                revision_id,
                kind,
                reason_code,
                previous_review_sha256,
                self._encrypt_json("review_events", event_id, "state", state),
                created_at,
            ),
        )
        return event_id, digest

    def _insert_tombstone_record(
        self,
        connection: sqlite3.Connection,
        *,
        record_kind: str,
        record_id: str,
        sequence: int,
        created_at: str,
        previous_tombstone_sha256: str | None,
    ) -> tuple[str, str]:
        if record_kind not in {"PAIR", "CAPTURE"}:
            raise ValueError("unknown tombstone record kind")
        self._parse_utc_timestamp(created_at, "tombstone created_at")
        tombstone_id = uuid4().hex
        state = {
            "created_at": created_at,
            "previous_tombstone_sha256": previous_tombstone_sha256,
            "reason_code": OPERATOR_DELETED,
            "record_id": record_id,
            "record_kind": record_kind,
            "sequence": sequence,
            "tombstone_id": tombstone_id,
        }
        digest = self._ledger_digest("tombstone", state)
        connection.execute(
            """
            INSERT INTO tombstones (
                tombstone_id, sequence, record_kind, record_id, reason_code,
                previous_tombstone_sha256, state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tombstone_id,
                sequence,
                record_kind,
                record_id,
                OPERATOR_DELETED,
                previous_tombstone_sha256,
                self._encrypt_json("tombstones", tombstone_id, "state", state),
                created_at,
            ),
        )
        return tombstone_id, digest


    @staticmethod
    def _rag_version_from_state(state: object) -> RagIndexVersion:
        expected = {
            "chunk_count", "config_sha256", "corpus_sha256", "dataset_bytes",
            "dataset_sha256", "dimension", "document_count",
            "embedding_model_sha256", "index_bytes", "index_sha256",
            "inventory_sha256", "metadata_bytes", "metadata_sha256", "schema",
            "serialization_version", "version_id",
        }
        if type(state) is not dict or set(state) != expected:
            raise VaultAuthenticationError("RAG index version state is invalid")
        if state["schema"] != "LIVED_RAG_INDEX_VERSION_V1":
            raise VaultAuthenticationError("RAG index version schema is invalid")
        try:
            return RagIndexVersion(
                version_id=state["version_id"],
                dataset_sha256=state["dataset_sha256"],
                corpus_sha256=state["corpus_sha256"],
                config_sha256=state["config_sha256"],
                embedding_model_sha256=state["embedding_model_sha256"],
                index_sha256=state["index_sha256"],
                index_bytes=state["index_bytes"],
                metadata_sha256=state["metadata_sha256"],
                metadata_bytes=state["metadata_bytes"],
                inventory_sha256=state["inventory_sha256"],
                dataset_bytes=state["dataset_bytes"],
                document_count=state["document_count"],
                chunk_count=state["chunk_count"],
                dimension=state["dimension"],
                serialization_version=state["serialization_version"],
            )
        except (TypeError, ValueError) as error:
            raise VaultAuthenticationError("RAG index version values are invalid") from error

    def _authenticate_rag_index_row(
        self, row: sqlite3.Row
    ) -> tuple[RagIndexEvent, str]:
        event_id = str(row["index_id"])
        state = self._decrypt_json(
            "rag_index_versions", event_id, "state", row["state_enc"]
        )
        expected = {
            "created_at", "event_id", "kind", "prepared_event_id",
            "previous_version_sha256", "schema", "sequence", "version",
            "version_sha256",
        }
        if type(state) is not dict or set(state) != expected:
            raise VaultAuthenticationError("RAG index event state is invalid")
        if state["schema"] != "LIVED_RAG_INDEX_EVENT_V1":
            raise VaultAuthenticationError("RAG index event schema is invalid")
        version = self._rag_version_from_state(state["version"])
        try:
            event = RagIndexEvent(
                event_id=state["event_id"],
                sequence=state["sequence"],
                kind=state["kind"],
                prepared_event_id=state["prepared_event_id"],
                previous_version_sha256=state["previous_version_sha256"],
                version_sha256=state["version_sha256"],
                version=version,
                created_at=self._parse_utc_timestamp(
                    state["created_at"], "RAG index event created_at"
                ),
            )
        except (TypeError, ValueError) as error:
            raise VaultAuthenticationError("RAG index event values are invalid") from error
        if (
            row["index_id"] != event.event_id
            or row["sequence"] != event.sequence
            or row["previous_version_sha256"] != event.previous_version_sha256
            or row["created_at"] != event.created_at.isoformat()
        ):
            raise VaultAuthenticationError(
                "RAG index event columns do not match authenticated state"
            )
        return event, self._ledger_digest("rag-index-event", state)

    def _insert_rag_index_event(
        self,
        connection: sqlite3.Connection,
        *,
        sequence: int,
        kind: str,
        version: RagIndexVersion,
        prepared_event_id: str | None,
        previous_version_sha256: str | None,
        created_at: str,
    ) -> tuple[RagIndexEvent, str]:
        event_id = uuid4().hex
        created = self._parse_utc_timestamp(created_at, "RAG index event created_at")
        event = RagIndexEvent(
            event_id=event_id,
            sequence=sequence,
            kind=kind,
            prepared_event_id=prepared_event_id,
            previous_version_sha256=previous_version_sha256,
            version_sha256=version.version_sha256,
            version=version,
            created_at=created,
        )
        state = {
            "created_at": created_at,
            "event_id": event_id,
            "kind": kind,
            "prepared_event_id": prepared_event_id,
            "previous_version_sha256": previous_version_sha256,
            "schema": "LIVED_RAG_INDEX_EVENT_V1",
            "sequence": sequence,
            "version": version.to_state(),
            "version_sha256": version.version_sha256,
        }
        digest = self._ledger_digest("rag-index-event", state)
        connection.execute(
            """
            INSERT INTO rag_index_versions (
                index_id, sequence, previous_version_sha256, state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_id,
                sequence,
                previous_version_sha256,
                self._encrypt_json(
                    "rag_index_versions", event_id, "state", state
                ),
                created_at,
            ),
        )
        return event, digest

    @staticmethod
    def _verify_rag_lifecycle(
        records: Sequence[tuple[RagIndexEvent, str]],
    ) -> tuple[RagIndexVersion | None, RagIndexEvent | None]:
        preparations: dict[str, RagIndexEvent] = {}
        published: set[str] = set()
        version_ids: set[str] = set()
        selected: RagIndexVersion | None = None
        selected_digest: str | None = None
        pending: RagIndexEvent | None = None
        for event, _digest in records:
            if event.kind == "PREPARED":
                if event.version.version_id in version_ids:
                    raise VaultAuthenticationError("RAG version ID was reused")
                if pending is not None:
                    raise VaultAuthenticationError(
                        "a second RAG version was prepared before resolution"
                    )
                if event.previous_version_sha256 != selected_digest:
                    raise VaultAuthenticationError(
                        "RAG preparation does not bind the selected predecessor"
                    )
                preparations[event.event_id] = event
                version_ids.add(event.version.version_id)
                pending = event
                continue
            prepared = preparations.get(str(event.prepared_event_id))
            if (
                prepared is None
                or pending is None
                or prepared.event_id != pending.event_id
                or event.version != prepared.version
                or event.version_sha256 != prepared.version_sha256
                or event.previous_version_sha256 != prepared.previous_version_sha256
            ):
                raise VaultAuthenticationError(
                    "RAG terminal event does not bind its active preparation"
                )
            if event.kind == "PUBLISHED":
                if prepared.event_id in published:
                    raise VaultAuthenticationError("RAG version was published twice")
                published.add(prepared.event_id)
                continue
            if prepared.event_id not in published:
                raise VaultAuthenticationError(
                    "RAG version was selected before publication"
                )
            selected = prepared.version
            selected_digest = prepared.version_sha256
            pending = None
        return selected, pending

    def _insert_source_target_record(
        self,
        connection: sqlite3.Connection,
        *,
        view: SourceTargetView,
        sequence: int,
        created_at: str,
        previous_target_sha256: str | None,
    ) -> str:
        self._parse_utc_timestamp(created_at, "source target created_at")
        state = {
            "blueprint_id": view.blueprint_id,
            "categories": list(view.categories),
            "created_at": created_at,
            "expected_yield": view.expected_yield,
            "label": view.label,
            "last_accepted_at": None,
            "operator_authorized": view.operator_authorized,
            "previous_target_sha256": previous_target_sha256,
            "sequence": sequence,
            "start_url": view.start_url,
            "target_id": view.target_id,
        }
        digest = self._ledger_digest("source-target", state)
        connection.execute(
            """
            INSERT INTO source_targets (
                target_id, sequence, blueprint_id, state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                view.target_id,
                sequence,
                view.blueprint_id,
                self._encrypt_json(
                    "source_targets", view.target_id, "state", state
                ),
                created_at,
            ),
        )
        return digest

    @staticmethod
    def _training_identity_state(identity: TrainingRunIdentity) -> dict[str, object]:
        if identity.audit_schema_version == 2:
            schema = "LIVED_TRAINING_RUN_IDENTITY_V2"
        else:
            schema = "LIVED_TRAINING_RUN_IDENTITY_V3"
        state = {
            "config_sha256": identity.config_sha256,
            "created_at": identity.created_at.isoformat(),
            "cuda_version": identity.cuda_version,
            "dataset_bytes": identity.dataset_bytes,
            "dataset_count": identity.dataset_count,
            "dataset_sha256": identity.dataset_sha256,
            "dataset_manifest_sha256": identity.dataset_manifest_sha256,
            "driver_version": identity.driver_version,
            "engine_identity": identity.engine_identity,
            "gpus": [
                {
                    "compute_capability": gpu.compute_capability,
                    "memory_mib": gpu.memory_mib,
                    "name": gpu.name,
                }
                for gpu in identity.gpus
            ],
            "hardware_profile": identity.hardware_profile,
            "mode": identity.mode.value,
            "model_identity_sha256": identity.model_identity_sha256,
            "train_sha256": identity.train_sha256,
            "eval_sha256": identity.eval_sha256,
            "parent_kind": identity.parent_kind,
            "parent_identity_sha256": identity.parent_identity_sha256,
            "plan_expires_at": identity.plan_expires_at.isoformat(),
            "plan_id": identity.plan_id,
            "projection_schema": identity.projection_schema,
            "python_version": identity.python_version,
            "release_manifest_sha256": identity.release_manifest_sha256,
            "run_id": identity.run_id,
            "runtime_sha256": identity.runtime_sha256,
            "schema": schema,
            "snapshot_sequence": identity.snapshot_sequence,
        }
        if identity.audit_schema_version == 3:
            state.update(
                execution_backend=identity.execution_backend,
                release_scope=identity.release_scope,
            )
        return state

    def _insert_training_identity(
        self, connection: sqlite3.Connection, identity: TrainingRunIdentity
    ) -> str:
        state = self._training_identity_state(identity)
        digest = self._ledger_digest("training-run", state)
        connection.execute(
            """
            INSERT INTO training_runs (run_id, mode, created_at, state_enc)
            VALUES (?, ?, ?, ?)
            """,
            (
                identity.run_id,
                identity.mode.value,
                identity.created_at.isoformat(),
                self._encrypt_json(
                    "training_runs", identity.run_id, "state", state
                ),
            ),
        )
        return digest

    def _authenticate_training_identity(
        self, row: sqlite3.Row
    ) -> tuple[TrainingRunIdentity, str]:
        run_id = str(row["run_id"])
        state = self._decrypt_json(
            "training_runs", run_id, "state", row["state_enc"]
        )
        if type(state) is not dict:
            raise VaultAuthenticationError("training run identity state is invalid")
        schema = state.get("schema")
        if schema == "LIVED_TRAINING_RUN_IDENTITY_V2":
            version = 2
        elif schema == "LIVED_TRAINING_RUN_IDENTITY_V3":
            version = 3
        else:
            raise VaultAuthenticationError("training run identity schema is invalid")
        expected = {
            "config_sha256", "created_at", "cuda_version", "dataset_bytes",
            "dataset_count", "dataset_sha256", "driver_version",
            "dataset_manifest_sha256", "train_sha256", "eval_sha256",
            "engine_identity", "gpus", "hardware_profile", "mode",
            "parent_kind", "parent_identity_sha256",
            "model_identity_sha256", "plan_expires_at", "plan_id",
            "projection_schema", "python_version", "release_manifest_sha256",
            "run_id", "runtime_sha256", "schema", "snapshot_sequence",
        }
        if version == 3:
            expected |= {"execution_backend", "release_scope"}
        if set(state) != expected:
            raise VaultAuthenticationError("training run identity state is invalid")
        gpus = state["gpus"]
        if type(gpus) is not list or (
            len(gpus) != 0 if version == 3 else not 1 <= len(gpus) <= 8
        ):
            raise VaultAuthenticationError("training run GPU identity is invalid")
        try:
            identity = TrainingRunIdentity(
                run_id=state["run_id"],
                plan_id=state["plan_id"],
                mode=TrainingMode(state["mode"]),
                created_at=self._parse_utc_timestamp(
                    state["created_at"], "training run created_at"
                ),
                plan_expires_at=self._parse_utc_timestamp(
                    state["plan_expires_at"], "training run plan expiry"
                ),
                snapshot_sequence=state["snapshot_sequence"],
                dataset_count=state["dataset_count"],
                dataset_bytes=state["dataset_bytes"],
                dataset_sha256=state["dataset_sha256"],
                config_sha256=state["config_sha256"],
                model_identity_sha256=state["model_identity_sha256"],
                engine_identity=state["engine_identity"],
                hardware_profile=state["hardware_profile"],
                projection_schema=state["projection_schema"],
                release_manifest_sha256=state["release_manifest_sha256"],
                runtime_sha256=state["runtime_sha256"],
                python_version=state["python_version"],
                cuda_version=state["cuda_version"],
                driver_version=state["driver_version"],
                gpus=tuple(
                    ManifestGpuIdentity(
                        name=gpu["name"],
                        memory_mib=gpu["memory_mib"],
                        compute_capability=gpu["compute_capability"],
                    )
                    for gpu in gpus
                    if type(gpu) is dict
                    and set(gpu)
                    == {"compute_capability", "memory_mib", "name"}
                ),
                dataset_manifest_sha256=state["dataset_manifest_sha256"],
                train_sha256=state["train_sha256"],
                eval_sha256=state["eval_sha256"],
                parent_kind=state["parent_kind"],
                parent_identity_sha256=state["parent_identity_sha256"],
                audit_schema_version=version,
                execution_backend=state.get("execution_backend"),
                release_scope=state.get("release_scope"),
            )
        except (TypeError, ValueError) as exc:
            raise VaultAuthenticationError(
                "training run identity values are invalid"
            ) from exc
        if len(identity.gpus) != len(gpus):
            raise VaultAuthenticationError("training run GPU schema is invalid")
        if (
            row["run_id"] != identity.run_id
            or row["mode"] != identity.mode.value
            or row["created_at"] != identity.created_at.isoformat()
        ):
            raise VaultAuthenticationError(
                "training run identity columns do not match authenticated state"
            )
        return identity, self._ledger_digest("training-run", state)

    @staticmethod
    def _manifest_event_values(
        event: ManifestPreparation | ManifestAuditEvent,
    ) -> tuple[int | None, str | None, int | None, str | None, str | None]:
        if type(event) is ManifestPreparation:
            return (
                event.manifest_sequence,
                event.basename,
                event.byte_length,
                event.manifest_sha256,
                event.previous_manifest_sha256,
            )
        return (
            event.manifest_sequence,
            event.basename,
            event.byte_length,
            event.manifest_sha256,
            event.previous_manifest_sha256,
        )

    def _last_training_event(
        self, connection: sqlite3.Connection, run_id: str
    ) -> tuple[int, str | None]:
        row = connection.execute(
            """
            SELECT * FROM training_run_events
            WHERE run_id = ?
            ORDER BY event_ordinal DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return 0, None
        _event, digest = self._authenticate_training_event(row)
        return int(row["event_ordinal"]), digest

    def _insert_training_event(
        self,
        connection: sqlite3.Connection,
        *,
        global_sequence: int,
        anchor: dict[str, object],
        run_id: str,
        kind: ManifestEventKind,
        prepared_event_id: str | None,
        manifest: ManifestPreparation | ManifestAuditEvent | None,
        evidence_sha256: str | None,
        created_at: str,
        run_digest: str | None = None,
    ) -> tuple[ManifestAuditEvent, str]:
        ordinal, previous = self._last_training_event(connection, run_id)
        ordinal += 1
        event_id = uuid4().hex
        if manifest is None:
            manifest_sequence = None
            basename = None
            byte_length = None
            manifest_sha256 = None
            previous_manifest_sha256 = None
        else:
            (
                manifest_sequence,
                basename,
                byte_length,
                manifest_sha256,
                previous_manifest_sha256,
            ) = self._manifest_event_values(manifest)
        state = {
            "basename": basename,
            "byte_length": byte_length,
            "capture_sequence": int(anchor["capture_sequence"]),
            "created_at": created_at,
            "event_id": event_id,
            "event_ordinal": ordinal,
            "evidence_sha256": evidence_sha256,
            "global_sequence": global_sequence,
            "kind": kind.value,
            "manifest_sequence": manifest_sequence,
            "manifest_sha256": manifest_sha256,
            "prepared_event_id": prepared_event_id,
            "previous_event_sha256": previous,
            "previous_manifest_sha256": previous_manifest_sha256,
            "run_id": run_id,
            "schema": "LIVED_TRAINING_RUN_EVENT_V1",
        }
        digest = self._ledger_digest("training-run-event", state)
        connection.execute(
            """
            INSERT INTO training_run_events (
                event_id, global_sequence, run_id, event_ordinal, kind,
                prepared_event_id, manifest_sequence,
                previous_event_sha256, state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                global_sequence,
                run_id,
                ordinal,
                kind.value,
                prepared_event_id,
                manifest_sequence,
                previous,
                self._encrypt_json(
                    "training_run_events", event_id, "state", state
                ),
                created_at,
            ),
        )
        tokens = [
            self._record_token("TRAINING_RUN_EVENT", event_id, digest)
        ]
        if run_digest is not None:
            tokens.append(
                self._record_token("TRAINING_RUN", run_id, run_digest)
            )
        self._publish_audit_operation(
            connection,
            sequence=global_sequence,
            kind=kind.value,
            subject_id=run_id,
            created_at=created_at,
            record_digests=tokens,
            head_updates={},
        )
        return (
            ManifestAuditEvent(
                event_id=event_id,
                global_sequence=global_sequence,
                capture_sequence=int(anchor["capture_sequence"]),
                run_id=run_id,
                event_ordinal=ordinal,
                kind=kind,
                prepared_event_id=prepared_event_id,
                manifest_sequence=manifest_sequence,
                basename=basename,
                byte_length=byte_length,
                manifest_sha256=manifest_sha256,
                previous_manifest_sha256=previous_manifest_sha256,
                previous_event_sha256=previous,
                evidence_sha256=evidence_sha256,
                created_at=self._parse_utc_timestamp(
                    created_at, "training event created_at"
                ),
            ),
            digest,
        )

    def _authenticate_training_event(
        self, row: sqlite3.Row
    ) -> tuple[ManifestAuditEvent, str]:
        event_id = str(row["event_id"])
        state = self._decrypt_json(
            "training_run_events", event_id, "state", row["state_enc"]
        )
        expected = {
            "basename", "byte_length", "capture_sequence", "created_at",
            "event_id", "event_ordinal", "evidence_sha256",
            "global_sequence", "kind", "manifest_sequence",
            "manifest_sha256", "prepared_event_id",
            "previous_event_sha256", "previous_manifest_sha256",
            "run_id", "schema",
        }
        if type(state) is not dict or set(state) != expected:
            raise VaultAuthenticationError("training run event state is invalid")
        if state["schema"] != "LIVED_TRAINING_RUN_EVENT_V1":
            raise VaultAuthenticationError("training run event schema is invalid")
        try:
            event = ManifestAuditEvent(
                event_id=state["event_id"],
                global_sequence=state["global_sequence"],
                capture_sequence=state["capture_sequence"],
                run_id=state["run_id"],
                event_ordinal=state["event_ordinal"],
                kind=ManifestEventKind(state["kind"]),
                prepared_event_id=state["prepared_event_id"],
                manifest_sequence=state["manifest_sequence"],
                basename=state["basename"],
                byte_length=state["byte_length"],
                manifest_sha256=state["manifest_sha256"],
                previous_manifest_sha256=state["previous_manifest_sha256"],
                previous_event_sha256=state["previous_event_sha256"],
                evidence_sha256=state["evidence_sha256"],
                created_at=self._parse_utc_timestamp(
                    state["created_at"], "training event created_at"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise VaultAuthenticationError(
                "training run event values are invalid"
            ) from exc
        if (
            row["event_id"] != event.event_id
            or row["global_sequence"] != event.global_sequence
            or row["run_id"] != event.run_id
            or row["event_ordinal"] != event.event_ordinal
            or row["kind"] != event.kind.value
            or row["prepared_event_id"] != event.prepared_event_id
            or row["manifest_sequence"] != event.manifest_sequence
            or row["previous_event_sha256"] != event.previous_event_sha256
            or row["created_at"] != event.created_at.isoformat()
        ):
            raise VaultAuthenticationError(
                "training run event columns do not match authenticated state"
            )
        return event, self._ledger_digest("training-run-event", state)

    def _insert_root_event(
        self,
        connection: sqlite3.Connection,
        *,
        global_sequence: int,
        evidence_sha256: str,
        created_at: str,
    ) -> tuple[str, str]:
        event_id = uuid4().hex
        state = {
            "created_at": created_at,
            "event_id": event_id,
            "evidence_sha256": evidence_sha256,
            "global_sequence": global_sequence,
            "kind": ManifestEventKind.ROOT_ANOMALY.value,
            "schema": "LIVED_TRAINING_ROOT_EVENT_V1",
        }
        digest = self._ledger_digest("training-root-event", state)
        connection.execute(
            """
            INSERT INTO training_root_events (
                event_id, global_sequence, kind, evidence_sha256,
                state_enc, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                global_sequence,
                ManifestEventKind.ROOT_ANOMALY.value,
                evidence_sha256,
                self._encrypt_json(
                    "training_root_events", event_id, "state", state
                ),
                created_at,
            ),
        )
        self._publish_audit_operation(
            connection,
            sequence=global_sequence,
            kind=ManifestEventKind.ROOT_ANOMALY.value,
            subject_id="training-runs-root",
            created_at=created_at,
            record_digests=(
                self._record_token(
                    "TRAINING_ROOT_EVENT", event_id, digest
                ),
            ),
            head_updates={},
        )
        return event_id, digest

    def _authenticate_root_event(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], str]:
        event_id = str(row["event_id"])
        state = self._decrypt_json(
            "training_root_events", event_id, "state", row["state_enc"]
        )
        if type(state) is not dict or set(state) != {
            "created_at", "event_id", "evidence_sha256",
            "global_sequence", "kind", "schema",
        }:
            raise VaultAuthenticationError("training root event state is invalid")
        if (
            state["schema"] != "LIVED_TRAINING_ROOT_EVENT_V1"
            or state["kind"] != ManifestEventKind.ROOT_ANOMALY.value
            or type(state["event_id"]) is not str
            or re.fullmatch(r"[a-f0-9]{32}", state["event_id"]) is None
            or type(state["global_sequence"]) is not int
            or state["global_sequence"] < 1
            or type(state["evidence_sha256"]) is not str
            or _SHA256.fullmatch(state["evidence_sha256"]) is None
        ):
            raise VaultAuthenticationError("training root event values are invalid")
        self._parse_utc_timestamp(
            state["created_at"], "training root event created_at"
        )
        if (
            row["event_id"] != state["event_id"]
            or row["global_sequence"] != state["global_sequence"]
            or row["kind"] != state["kind"]
            or row["evidence_sha256"] != state["evidence_sha256"]
            or row["created_at"] != state["created_at"]
        ):
            raise VaultAuthenticationError(
                "training root columns do not match authenticated state"
            )
        return state, self._ledger_digest("training-root-event", state)

    def _verify_training_records(
        self,
        connection: sqlite3.Connection,
        records_by_sequence: dict[int, list[tuple[str, str, str]]],
        anchor: dict[str, object],
    ) -> tuple[
        dict[str, tuple[TrainingRunIdentity, str]],
        tuple[tuple[ManifestAuditEvent, str], ...],
        tuple[tuple[dict[str, object], str], ...],
    ]:
        identities: dict[str, tuple[TrainingRunIdentity, str]] = {}
        for row in connection.execute(
            "SELECT * FROM training_runs ORDER BY created_at, run_id"
        ).fetchall():
            identity, digest = self._authenticate_training_identity(row)
            if identity.run_id in identities:
                raise VaultAuthenticationError("duplicate training run identity")
            identities[identity.run_id] = (identity, digest)

        event_records: list[tuple[ManifestAuditEvent, str]] = []
        grouped: dict[str, list[tuple[ManifestAuditEvent, str]]] = {}
        capture_sequence_seen = 0
        for row in connection.execute(
            "SELECT * FROM training_run_events ORDER BY global_sequence"
        ).fetchall():
            event, digest = self._authenticate_training_event(row)
            if event.run_id not in identities:
                raise VaultAuthenticationError(
                    "training event lacks an authenticated run"
                )
            if (
                event.capture_sequence < capture_sequence_seen
                or event.capture_sequence > int(anchor["capture_sequence"])
            ):
                raise VaultAuthenticationError(
                    "training event capture sequence is invalid"
                )
            capture_sequence_seen = event.capture_sequence
            event_records.append((event, digest))
            grouped.setdefault(event.run_id, []).append((event, digest))
            records_by_sequence.setdefault(
                event.global_sequence, []
            ).append(
                self._record_token(
                    "TRAINING_RUN_EVENT", event.event_id, digest
                )
            )

        if set(grouped) != set(identities):
            raise VaultAuthenticationError(
                "training run identities and event chains differ"
            )
        for run_id, records in grouped.items():
            previous = None
            preparations: dict[str, ManifestAuditEvent] = {}
            preparation_by_sequence: dict[int, ManifestAuditEvent] = {}
            terminal_by_prepared: dict[str, list[ManifestAuditEvent]] = {}
            for expected_ordinal, (event, digest) in enumerate(records, start=1):
                if event.event_ordinal != expected_ordinal:
                    raise VaultAuthenticationError(
                        "training event ordinal is not contiguous"
                    )
                if event.previous_event_sha256 != previous:
                    raise VaultAuthenticationError(
                        "training run event chain is broken"
                    )
                previous = digest
                if event.kind is ManifestEventKind.MANIFEST_PREPARED:
                    if event.prepared_event_id is not None:
                        raise VaultAuthenticationError(
                            "prepared event references another preparation"
                        )
                    if event.manifest_sequence is None:
                        raise VaultAuthenticationError(
                            "prepared event lacks a manifest sequence"
                        )
                    existing = preparation_by_sequence.get(
                        event.manifest_sequence
                    )
                    if existing is not None and (
                        existing.basename,
                        existing.byte_length,
                        existing.manifest_sha256,
                        existing.previous_manifest_sha256,
                    ) != (
                        event.basename,
                        event.byte_length,
                        event.manifest_sha256,
                        event.previous_manifest_sha256,
                    ):
                        raise VaultAuthenticationError(
                            "manifest retry changed authenticated bytes"
                        )
                    preparation_by_sequence[event.manifest_sequence] = event
                    preparations[event.event_id] = event
                else:
                    if event.prepared_event_id is None:
                        if event.kind not in {
                            ManifestEventKind.RECONCILIATION_BLOCKED,
                            ManifestEventKind.AUDIT_TAMPERED,
                        }:
                            raise VaultAuthenticationError(
                                "training terminal event lacks preparation"
                            )
                    else:
                        prepared = preparations.get(event.prepared_event_id)
                        if prepared is None:
                            raise VaultAuthenticationError(
                                "training terminal references an unknown preparation"
                            )
                        if (
                            event.manifest_sequence,
                            event.basename,
                            event.byte_length,
                            event.manifest_sha256,
                            event.previous_manifest_sha256,
                        ) != (
                            prepared.manifest_sequence,
                            prepared.basename,
                            prepared.byte_length,
                            prepared.manifest_sha256,
                            prepared.previous_manifest_sha256,
                        ):
                            raise VaultAuthenticationError(
                                "training terminal does not bind its preparation"
                            )
                        terminal_by_prepared.setdefault(
                            prepared.event_id, []
                        ).append(event)
            initial = records[0][0]
            if (
                initial.kind is not ManifestEventKind.MANIFEST_PREPARED
                or initial.manifest_sequence != 0
            ):
                raise VaultAuthenticationError(
                    "training run does not start with manifest zero preparation"
                )
            published_sequences = sorted(
                prepared.manifest_sequence
                for prepared_id, prepared in preparations.items()
                if any(
                    item.kind is ManifestEventKind.MANIFEST_PUBLISHED
                    for item in terminal_by_prepared.get(prepared_id, ())
                )
            )
            if published_sequences != list(range(len(published_sequences))):
                raise VaultAuthenticationError(
                    "published training manifests are not contiguous"
                )
            identity, identity_digest = identities[run_id]
            records_by_sequence.setdefault(
                initial.global_sequence, []
            ).append(
                self._record_token(
                    "TRAINING_RUN", run_id, identity_digest
                )
            )

        root_records: list[tuple[dict[str, object], str]] = []
        for row in connection.execute(
            "SELECT * FROM training_root_events ORDER BY global_sequence"
        ).fetchall():
            state, digest = self._authenticate_root_event(row)
            root_records.append((state, digest))
            records_by_sequence.setdefault(
                int(state["global_sequence"]), []
            ).append(
                self._record_token(
                    "TRAINING_ROOT_EVENT", str(state["event_id"]), digest
                )
            )
        return identities, tuple(event_records), tuple(root_records)

    @staticmethod
    def _preparation_signature(
        event: ManifestPreparation | ManifestAuditEvent,
    ) -> tuple[object, ...]:
        return (
            event.manifest_sequence,
            event.basename,
            event.byte_length,
            event.manifest_sha256,
            event.previous_manifest_sha256,
        )

    def prepare_manifest(
        self, event: ManifestPreparation
    ) -> PreparedManifestResult:
        if type(event) is not ManifestPreparation:
            raise TypeError("event must be ManifestPreparation")
        with self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            identities = ledger["training_identities"]
            records = ledger["training_event_records"]
            current = identities.get(event.run_id)
            run_digest = None
            sequence, anchor = self._audit_context(
                connection, verify=False
            )
            if current is None:
                if event.run_identity is None or event.manifest_sequence != 0:
                    raise VaultAuthenticationError(
                        "new run requires its immutable identity and manifest zero"
                    )
                run_digest = self._insert_training_identity(
                    connection, event.run_identity
                )
            else:
                identity = current[0]
                if (
                    event.run_identity is not None
                    and event.run_identity != identity
                ):
                    raise VaultAuthenticationError(
                        "run identity differs from authenticated state"
                    )
            run_events = [
                item
                for item, _digest in records
                if item.run_id == event.run_id
            ]
            same_sequence = [
                item for item in run_events
                if item.kind is ManifestEventKind.MANIFEST_PREPARED
                and item.manifest_sequence == event.manifest_sequence
            ]
            for prepared in same_sequence:
                if (
                    self._preparation_signature(prepared)
                    != self._preparation_signature(event)
                ):
                    evidence = self._ledger_digest(
                        "manifest-retry-block",
                        {
                            "candidate_manifest_sha256": event.manifest_sha256,
                            "existing_manifest_sha256": prepared.manifest_sha256,
                            "manifest_sequence": event.manifest_sequence,
                            "run_id": event.run_id,
                        },
                    )
                    if any(
                        item.prepared_event_id is None
                        and item.kind
                        is ManifestEventKind.RECONCILIATION_BLOCKED
                        and item.evidence_sha256 == evidence
                        for item in run_events
                    ):
                        raise VaultAuthenticationError(
                            "manifest retry changed its canonical identity"
                        )
                    self._insert_training_event(
                        connection,
                        global_sequence=sequence,
                        anchor=anchor,
                        run_id=event.run_id,
                        kind=ManifestEventKind.RECONCILIATION_BLOCKED,
                        prepared_event_id=None,
                        manifest=None,
                        evidence_sha256=evidence,
                        created_at=self._now_utc_text(),
                    )
                    raise _DurableAuditRejection(
                        "manifest retry changed its canonical identity"
                    )
            terminal = {
                item.prepared_event_id: item
                for item in run_events
                if item.prepared_event_id is not None
            }
            for prepared in reversed(same_sequence):
                outcome = terminal.get(prepared.event_id)
                if outcome is None:
                    return PreparedManifestResult(
                        prepared.event_id, False
                    )
                if outcome.kind is ManifestEventKind.MANIFEST_PUBLISHED:
                    return PreparedManifestResult(
                        prepared.event_id, False
                    )
                if outcome.kind is not ManifestEventKind.ABORTED_BEFORE_PUBLICATION:
                    raise VaultAuthenticationError(
                        "manifest sequence is blocked by authenticated evidence"
                    )
            published = sorted(
                [
                    prepared
                    for prepared in run_events
                    if prepared.kind is ManifestEventKind.MANIFEST_PREPARED
                    and terminal.get(prepared.event_id) is not None
                    and terminal[prepared.event_id].kind
                    is ManifestEventKind.MANIFEST_PUBLISHED
                ],
                key=lambda item: int(item.manifest_sequence),
            )
            if event.manifest_sequence == 0:
                if published:
                    raise VaultAuthenticationError(
                        "manifest zero is already published"
                    )
            else:
                if (
                    len(published) != event.manifest_sequence
                    or published[-1].manifest_sequence
                    != event.manifest_sequence - 1
                    or published[-1].manifest_sha256
                    != event.previous_manifest_sha256
                ):
                    raise VaultAuthenticationError(
                        "manifest preparation has a gap or predecessor mismatch"
                    )
            inserted, _digest = self._insert_training_event(
                connection,
                global_sequence=sequence,
                anchor=anchor,
                run_id=event.run_id,
                kind=ManifestEventKind.MANIFEST_PREPARED,
                prepared_event_id=None,
                manifest=event,
                evidence_sha256=None,
                created_at=event.created_at.isoformat(),
                run_digest=run_digest,
            )
            return PreparedManifestResult(inserted.event_id, True)

    def _terminal_manifest_event(
        self,
        prepared_event_id: str,
        outcome: ReconciliationOutcome,
    ) -> AuditAppendResult:
        if (
            type(prepared_event_id) is not str
            or re.fullmatch(r"[a-f0-9]{32}", prepared_event_id) is None
        ):
            raise ValueError("prepared_event_id is invalid")
        if type(outcome) is not ReconciliationOutcome:
            raise TypeError("outcome must be ReconciliationOutcome")
        kind = ManifestEventKind(outcome.value)
        with self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            records = ledger["training_event_records"]
            prepared = next(
                (
                    event
                    for event, _digest in records
                    if event.event_id == prepared_event_id
                    and event.kind is ManifestEventKind.MANIFEST_PREPARED
                ),
                None,
            )
            if prepared is None:
                raise RecordNotFoundError(prepared_event_id)
            existing = [
                event for event, _digest in records
                if event.prepared_event_id == prepared_event_id
            ]
            if (
                kind is ManifestEventKind.AUDIT_TAMPERED
                and not any(
                    event.kind is ManifestEventKind.MANIFEST_PUBLISHED
                    for event in existing
                )
            ):
                raise VaultAuthenticationError(
                    "audit tamper evidence requires a published manifest"
                )
            for event in existing:
                if event.kind is kind:
                    return AuditAppendResult(event.event_id, False)
            if (
                kind in {
                    ManifestEventKind.MANIFEST_PUBLISHED,
                    ManifestEventKind.ABORTED_BEFORE_PUBLICATION,
                    ManifestEventKind.RECONCILIATION_BLOCKED,
                }
                and existing
            ):
                raise VaultAuthenticationError(
                    "manifest preparation already has a terminal outcome"
                )
            evidence = self._ledger_digest(
                "manifest-reconciliation",
                {
                    "kind": kind.value,
                    "manifest_sha256": prepared.manifest_sha256,
                    "prepared_event_id": prepared_event_id,
                    "run_id": prepared.run_id,
                },
            )
            sequence, anchor = self._audit_context(
                connection, verify=False
            )
            inserted, _digest = self._insert_training_event(
                connection,
                global_sequence=sequence,
                anchor=anchor,
                run_id=prepared.run_id,
                kind=kind,
                prepared_event_id=prepared_event_id,
                manifest=prepared,
                evidence_sha256=evidence,
                created_at=self._now_utc_text(),
            )
            return AuditAppendResult(inserted.event_id, True)

    def mark_manifest_published(
        self, prepared_event_id: str
    ) -> AuditAppendResult:
        return self._terminal_manifest_event(
            prepared_event_id,
            ReconciliationOutcome.MANIFEST_PUBLISHED,
        )

    def mark_manifest_reconciled(
        self,
        prepared_event_id: str,
        outcome: ReconciliationOutcome,
    ) -> AuditAppendResult:
        return self._terminal_manifest_event(
            prepared_event_id, outcome
        )

    def _training_events_public(
        self, *, run_id: str | None
    ) -> tuple[ManifestAuditEvent, ...]:
        connection = self._require_connection()
        ledger = self._verify_v4_ledger(connection)
        return tuple(
            event for event, _digest in ledger["training_event_records"]
            if run_id is None or event.run_id == run_id
        )

    def manifest_events(
        self, run_id: str
    ) -> tuple[ManifestAuditEvent, ...]:
        if type(run_id) is not str or re.fullmatch(r"[a-f0-9]{32}", run_id) is None:
            raise ValueError("run_id is invalid")
        return self._training_events_public(run_id=run_id)

    def all_manifest_events(self) -> tuple[ManifestAuditEvent, ...]:
        return self._training_events_public(run_id=None)

    def training_run_identity(
        self, run_id: str
    ) -> TrainingRunIdentity | None:
        if type(run_id) is not str or re.fullmatch(r"[a-f0-9]{32}", run_id) is None:
            raise ValueError("run_id is invalid")
        ledger = self._verify_v4_ledger(self._require_connection())
        record = ledger["training_identities"].get(run_id)
        return None if record is None else record[0]

    def mark_run_anomaly(
        self,
        run_id: str,
        outcome: ReconciliationOutcome,
        evidence_sha256: str,
    ) -> AuditAppendResult:
        if type(outcome) is not ReconciliationOutcome or outcome not in {
            ReconciliationOutcome.RECONCILIATION_BLOCKED,
            ReconciliationOutcome.AUDIT_TAMPERED,
        }:
            raise ValueError("run anomaly outcome is invalid")
        if type(evidence_sha256) is not str or _SHA256.fullmatch(evidence_sha256) is None:
            raise ValueError("run anomaly evidence must be lowercase SHA-256")
        kind = ManifestEventKind(outcome.value)
        with self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            if run_id not in ledger["training_identities"]:
                raise RecordNotFoundError(run_id)
            for event, _digest in ledger["training_event_records"]:
                if (
                    event.run_id == run_id
                    and event.prepared_event_id is None
                    and event.kind is kind
                    and event.evidence_sha256 == evidence_sha256
                ):
                    return AuditAppendResult(event.event_id, False)
            sequence, anchor = self._audit_context(
                connection, verify=False
            )
            inserted, _digest = self._insert_training_event(
                connection,
                global_sequence=sequence,
                anchor=anchor,
                run_id=run_id,
                kind=kind,
                prepared_event_id=None,
                manifest=None,
                evidence_sha256=evidence_sha256,
                created_at=self._now_utc_text(),
            )
            return AuditAppendResult(inserted.event_id, True)

    def mark_root_anomaly(
        self, evidence_sha256: str
    ) -> AuditAppendResult:
        if type(evidence_sha256) is not str or _SHA256.fullmatch(evidence_sha256) is None:
            raise ValueError("root anomaly evidence must be lowercase SHA-256")
        with self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            for state, _digest in ledger["training_root_records"]:
                if state["evidence_sha256"] == evidence_sha256:
                    return AuditAppendResult(str(state["event_id"]), False)
            sequence, _anchor = self._audit_context(
                connection, verify=False
            )
            event_id, _digest = self._insert_root_event(
                connection,
                global_sequence=sequence,
                evidence_sha256=evidence_sha256,
                created_at=self._now_utc_text(),
            )
            return AuditAppendResult(event_id, True)

    def _verify_v4_ledger(
        self, connection: sqlite3.Connection
    ) -> dict[str, object]:
        anchor = self._load_audit_state(connection)

        capture_rows = connection.execute(
            "SELECT * FROM captures ORDER BY captured_at, id"
        ).fetchall()
        captures: dict[str, tuple[dict[str, object], CapturedDocument]] = {}
        for row in capture_rows:
            state, document = self._authenticate_capture_row(row)
            self._parse_utc_timestamp(state["captured_at"], "capture captured_at")
            captures[str(row["id"])] = (state, document)

        capture_event_rows = connection.execute(
            "SELECT * FROM capture_events ORDER BY capture_sequence"
        ).fetchall()
        if len(capture_event_rows) != int(anchor["capture_sequence"]):
            raise VaultAuthenticationError(
                "capture event tail was removed or advanced"
            )
        previous_capture_event: str | None = None
        latest_capture_event: dict[str, dict[str, object]] = {}
        capture_event_records = []
        for expected_sequence, row in enumerate(capture_event_rows, start=1):
            state, digest = self._authenticate_capture_event(row)
            if state["capture_sequence"] != expected_sequence:
                raise VaultAuthenticationError(
                    "capture event sequence is not contiguous"
                )
            if state["previous_event_sha256"] != previous_capture_event:
                raise VaultAuthenticationError("capture event chain is broken")
            capture_id = str(state["capture_id"])
            first_for_capture = capture_id not in latest_capture_event
            if first_for_capture != (state["kind"] in {"CREATED", "MIGRATED"}):
                raise VaultAuthenticationError(
                    "capture event lifecycle is invalid"
                )
            latest_capture_event[capture_id] = state
            capture_event_records.append((state, digest))
            previous_capture_event = digest
        if anchor["capture_head_sha256"] != previous_capture_event:
            raise VaultAuthenticationError(
                "capture audit head does not match authenticated history"
            )
        if set(latest_capture_event) != set(captures):
            raise VaultAuthenticationError(
                "capture rows and authenticated state events do not match"
            )
        for capture_id, (state, document) in captures.items():
            if (
                latest_capture_event[capture_id]["capture_state_sha256"]
                != self._capture_record_digest(state, document)
            ):
                raise VaultAuthenticationError(
                    "capture state was replayed after its authenticated event"
                )

        pair_rows = connection.execute(
            "SELECT * FROM pairs ORDER BY created_at, id"
        ).fetchall()
        current_pairs: dict[str, tuple[sqlite3.Row, dict[str, object], DistilledPair]] = {}
        for row in pair_rows:
            state = self._authenticated_pair_state(row)
            self._parse_utc_timestamp(state["created_at"], "pair created_at")
            if state["reviewed_at"] is not None:
                self._parse_utc_timestamp(state["reviewed_at"], "pair reviewed_at")
            pair = self._pair_from_row(row, state)
            current_pairs[pair.id] = (row, state, pair)
            source = captures.get(pair.source_id)
            if (
                source is None
                or source[0]["source_provenance_hash"]
                != pair.source_provenance_hash
            ):
                raise VaultAuthenticationError(
                    "pair source does not match an authenticated capture"
                )

        revision_records = []
        records_by_sequence: dict[int, list[tuple[str, str, str]]] = {}
        previous_revision: str | None = None
        for row in connection.execute(
            "SELECT * FROM pair_revisions ORDER BY sequence, revision_id"
        ).fetchall():
            state, payload, digest = self._authenticate_revision_row(row)
            if state["previous_revision_sha256"] != previous_revision:
                raise VaultAuthenticationError("pair revision chain is broken")
            previous_revision = digest
            revision_records.append((state, payload, digest))
            records_by_sequence.setdefault(int(state["sequence"]), []).append(
                self._record_token("PAIR_REVISION", str(state["revision_id"]), digest)
            )

        review_records = []
        previous_review: str | None = None
        for row in connection.execute(
            "SELECT * FROM review_events ORDER BY sequence, event_id"
        ).fetchall():
            state, digest = self._authenticate_review_event(row)
            if state["previous_event_sha256"] != previous_review:
                raise VaultAuthenticationError("review event chain is broken")
            previous_review = digest
            review_records.append((state, digest))
            records_by_sequence.setdefault(int(state["sequence"]), []).append(
                self._record_token("REVIEW_EVENT", str(state["event_id"]), digest)
            )

        target_records = []
        previous_target: str | None = None
        for row in connection.execute(
            "SELECT * FROM source_targets ORDER BY sequence, target_id"
        ).fetchall():
            state, view, digest = self._authenticate_target_row(row)
            if state["previous_target_sha256"] != previous_target:
                raise VaultAuthenticationError("source target chain is broken")
            previous_target = digest
            target_records.append((state, view, digest))
            records_by_sequence.setdefault(int(state["sequence"]), []).append(
                self._record_token("SOURCE_TARGET", view.target_id, digest)
            )

        tombstone_records = []
        previous_tombstone: str | None = None
        for row in connection.execute(
            "SELECT * FROM tombstones ORDER BY sequence, tombstone_id"
        ).fetchall():
            state, digest = self._authenticate_tombstone_row(row)
            if state["previous_tombstone_sha256"] != previous_tombstone:
                raise VaultAuthenticationError("tombstone chain is broken")
            previous_tombstone = digest
            tombstone_records.append((state, digest))
            records_by_sequence.setdefault(int(state["sequence"]), []).append(
                self._record_token("TOMBSTONE", str(state["tombstone_id"]), digest)
            )

        rag_index_records = []
        for row in connection.execute(
            "SELECT * FROM rag_index_versions ORDER BY sequence, index_id"
        ).fetchall():
            event, digest = self._authenticate_rag_index_row(row)
            rag_index_records.append((event, digest))
            records_by_sequence.setdefault(event.sequence, []).append(
                self._record_token("RAG_INDEX_EVENT", event.event_id, digest)
            )
        selected_rag_index, pending_rag_index = self._verify_rag_lifecycle(
            rag_index_records
        )

        (
            training_identities,
            training_event_records,
            training_root_records,
        ) = self._verify_training_records(
            connection, records_by_sequence, anchor
        )

        audit_rows = connection.execute(
            "SELECT * FROM audit_events ORDER BY sequence"
        ).fetchall()
        sequence = int(anchor["sequence"])
        if len(audit_rows) != sequence:
            raise VaultAuthenticationError("vault audit event count was rolled back")
        previous_audit: str | None = None
        audit_records = []
        for expected_sequence, row in enumerate(audit_rows, start=1):
            state, digest = self._authenticate_audit_event(row)
            if state["sequence"] != expected_sequence:
                raise VaultAuthenticationError("vault audit sequence is not contiguous")
            if state["previous_event_sha256"] != previous_audit:
                raise VaultAuthenticationError("vault audit chain is broken")
            expected_tokens = self._normalized_record_tokens(
                records_by_sequence.get(expected_sequence, ())
            )
            actual_tokens = tuple(tuple(token) for token in state["record_digests"])
            if actual_tokens != expected_tokens:
                raise VaultAuthenticationError(
                    "vault audit event does not bind its complete record set"
                )
            audit_records.append((state, digest))
            previous_audit = digest
        if set(records_by_sequence) - set(range(1, sequence + 1)):
            raise VaultAuthenticationError("ledger record lies beyond the audit anchor")
        if (
            anchor["head_sha256"] != previous_audit
            or anchor["review_head_sha256"] != previous_review
            or anchor["tombstone_head_sha256"] != previous_tombstone
            or anchor["revision_head_sha256"] != previous_revision
            or anchor["target_head_sha256"] != previous_target
        ):
            raise VaultAuthenticationError(
                "vault audit anchor does not match authenticated history"
            )

        revisions_by_id = {
            str(state["revision_id"]): (state, payload, digest)
            for state, payload, digest in revision_records
        }
        revisions_by_pair: dict[str, list[tuple[dict[str, object], dict[str, object], str]]] = {}
        for record in revision_records:
            revisions_by_pair.setdefault(str(record[0]["pair_id"]), []).append(record)
        reviews_by_revision: dict[str, list[dict[str, object]]] = {}
        for state, _digest in review_records:
            revision = revisions_by_id.get(str(state["revision_id"]))
            if (
                revision is None
                or revision[0]["pair_id"] != state["pair_id"]
                or int(state["sequence"]) < int(revision[0]["sequence"])
                or (
                    state["kind"] == "MODIFIED"
                    and state["sequence"] != revision[0]["sequence"]
                )
            ):
                raise VaultAuthenticationError(
                    "review event does not bind its exact pair revision"
                )
            reviews_by_revision.setdefault(str(state["revision_id"]), []).append(state)

        if set(revisions_by_pair) != set(current_pairs):
            raise VaultAuthenticationError(
                "current pair rows and revision lineages do not match"
            )
        for pair_id, (_row, current_state, current_pair) in current_pairs.items():
            history = sorted(
                revisions_by_pair[pair_id], key=lambda item: int(item[0]["revision"])
            )
            if [int(item[0]["revision"]) for item in history] != list(
                range(1, len(history) + 1)
            ):
                raise VaultAuthenticationError("pair revision numbers are not contiguous")
            previous_id = None
            previous_time = None
            for state, _payload, _digest in history:
                created = self._parse_utc_timestamp(
                    state["created_at"], "pair revision created_at"
                )
                if (
                    state["lineage_id"] != pair_id
                    or state["source_id"] != current_pair.source_id
                    or state["source_provenance_hash"]
                    != current_pair.source_provenance_hash
                    or state["parent_revision_id"] != previous_id
                    or (previous_time is not None and created < previous_time)
                ):
                    raise VaultAuthenticationError(
                        "pair revision parentage or source binding is invalid"
                    )
                previous_id = str(state["revision_id"])
                previous_time = created
            first_state = history[0][0]
            if first_state["created_at"] != current_state["created_at"]:
                raise VaultAuthenticationError(
                    "current pair creation time does not match revision lineage"
                )
            latest_state, latest_payload, _digest = history[-1]
            if (
                latest_payload["instruction"] != current_pair.instruction
                or latest_payload["response"] != current_pair.response
                or tuple(latest_payload["tags"]) != current_pair.tags
            ):
                raise VaultAuthenticationError(
                    "current pair payload does not match its latest revision"
                )
            latest_events = reviews_by_revision.get(
                str(latest_state["revision_id"]), []
            )
            latest_event = max(
                latest_events, key=lambda item: int(item["sequence"]), default=None
            )
            expected_status = PAIR_PENDING
            expected_approved = False
            expected_reviewed = current_state["reviewed_at"]
            if latest_event is not None:
                expected_status = {
                    "MODIFIED": PAIR_PENDING,
                    "APPROVED": PAIR_APPROVED,
                    "DENIED": PAIR_REJECTED,
                }[str(latest_event["kind"])]
                expected_approved = expected_status == PAIR_APPROVED
                expected_reviewed = latest_event["created_at"]
            elif current_state["status"] != PAIR_PENDING:
                raise VaultAuthenticationError(
                    "current pair approval lacks an authenticated decision"
                )
            if (
                current_state["status"] != expected_status
                or bool(current_state["user_approved"]) != expected_approved
                or current_state["reviewed_at"] != expected_reviewed
            ):
                raise VaultAuthenticationError(
                    "current pair state does not match its latest decision"
                )

        pair_tombstones: set[str] = set()
        capture_tombstones: set[str] = set()
        for state, _digest in tombstone_records:
            if state["record_kind"] == "PAIR":
                if state["record_id"] not in current_pairs:
                    raise VaultAuthenticationError("pair tombstone target is missing")
                pair_tombstones.add(str(state["record_id"]))
            else:
                if state["record_id"] not in captures:
                    raise VaultAuthenticationError("capture tombstone target is missing")
                capture_tombstones.add(str(state["record_id"]))

        return {
            "anchor": anchor,
            "captures": captures,
            "capture_event_records": tuple(capture_event_records),
            "current_pairs": current_pairs,
            "revision_records": tuple(revision_records),
            "review_records": tuple(review_records),
            "audit_records": tuple(audit_records),
            "target_records": tuple(target_records),
            "tombstone_records": tuple(tombstone_records),
            "training_identities": training_identities,
            "training_event_records": training_event_records,
            "training_root_records": training_root_records,
            "rag_index_records": tuple(rag_index_records),
            "selected_rag_index": selected_rag_index,
            "pending_rag_index": pending_rag_index,
            "pair_tombstones": frozenset(pair_tombstones),
            "capture_tombstones": frozenset(capture_tombstones),
        }

    def _migrate_v3_to_v4(self) -> None:
        connection = self._require_connection()
        try:
            with self._transaction() as transaction:
                capture_rows = transaction.execute(
                    "SELECT * FROM captures ORDER BY captured_at, id"
                ).fetchall()
                authenticated_captures = [
                    self._authenticate_capture_row(row) for row in capture_rows
                ]
                pair_rows = transaction.execute(
                    "SELECT * FROM pairs ORDER BY created_at, id"
                ).fetchall()
                authenticated_pairs = []
                for row in pair_rows:
                    state = self._authenticated_pair_state(row)
                    pair = self._pair_from_row(row, state)
                    try:
                        self._validate_pair_record(pair)
                    except (TypeError, ValueError) as exc:
                        raise VaultAuthenticationError(
                            "v3 pair payload is not canonical"
                        ) from exc
                    authenticated_pairs.append((pair, state))

                capture_columns = (
                    "id", "blueprint_id", "captured_at", "content_sha256",
                    "source_provenance_hash", "url_enc", "title_enc",
                    "raw_text_enc", "metadata_enc", "simhash_hex",
                    "quality_score", "quality_reasons_enc", "status",
                    "quarantine_reason_enc", "state_enc",
                )
                pair_columns = (
                    "id", "source_id", "source_provenance_hash",
                    "instruction_enc", "response_enc", "tags_enc",
                    "state_enc", "user_approved", "status", "created_at",
                    "reviewed_at",
                )
                capture_values = [
                    tuple(row[column] for column in capture_columns)
                    for row in capture_rows
                ]
                pair_values = [
                    tuple(row[column] for column in pair_columns)
                    for row in pair_rows
                ]
                transaction.execute("DROP TABLE pairs")
                transaction.execute("DROP TABLE captures")
                transaction.execute("DROP TABLE vault_meta")
                self._create_core_tables(transaction)
                self._insert_key_verifier(transaction)
                transaction.executemany(
                    """
                    INSERT INTO captures (
                        id, blueprint_id, captured_at, content_sha256,
                        source_provenance_hash, url_enc, title_enc,
                        raw_text_enc, metadata_enc, simhash_hex,
                        quality_score, quality_reasons_enc, status,
                        quarantine_reason_enc, state_enc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    capture_values,
                )
                transaction.executemany(
                    """
                    INSERT INTO pairs (
                        id, source_id, source_provenance_hash,
                        instruction_enc, response_enc, tags_enc, state_enc,
                        user_approved, status, created_at, reviewed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    pair_values,
                )
                self._create_v4_tables(transaction)
                self._insert_audit_genesis(transaction)
                for _state, document in authenticated_captures:
                    self._append_capture_event(
                        transaction,
                        capture_id=document.id,
                        kind="MIGRATED",
                        created_at=document.captured_at,
                    )
                for pair, state in authenticated_pairs:
                    sequence, anchor = self._audit_context(
                        transaction, verify=False
                    )
                    revision_id, revision_digest = self._insert_revision_record(
                        transaction,
                        pair=pair,
                        revision=1,
                        parent_revision_id=None,
                        sequence=sequence,
                        created_at=pair.created_at,
                        previous_revision_sha256=anchor["revision_head_sha256"],
                    )
                    tokens = [
                        self._record_token(
                            "PAIR_REVISION", revision_id, revision_digest
                        )
                    ]
                    head_updates = {"revision_head_sha256": revision_digest}
                    status = str(state["status"])
                    review_created_at = state["reviewed_at"]
                    kind = None
                    reason = None
                    if status == PAIR_APPROVED:
                        kind, reason = "APPROVED", OPERATOR_APPROVED
                    elif status == PAIR_REJECTED:
                        kind, reason = "DENIED", OPERATOR_REJECTED
                    if kind is not None:
                        if review_created_at is None:
                            raise VaultAuthenticationError(
                                "reviewed legacy pair lacks a review timestamp"
                            )
                        event_id, review_digest = self._insert_review_record(
                            transaction,
                            pair_id=pair.id,
                            revision_id=revision_id,
                            sequence=sequence,
                            kind=kind,
                            reason_code=reason,
                            created_at=str(review_created_at),
                            previous_review_sha256=anchor["review_head_sha256"],
                        )
                        tokens.append(
                            self._record_token(
                                "REVIEW_EVENT", event_id, review_digest
                            )
                        )
                        head_updates["review_head_sha256"] = review_digest
                    operation_time = (
                        str(review_created_at)
                        if review_created_at is not None and kind is not None
                        else pair.created_at
                    )
                    self._publish_audit_operation(
                        transaction,
                        sequence=sequence,
                        kind="MIGRATED_PAIR",
                        subject_id=pair.id,
                        created_at=operation_time,
                        record_digests=tokens,
                        head_updates=head_updates,
                    )

                self._validate_v4_schema()
                ledger = self._verify_v4_ledger(transaction)
                try:
                    self._training_snapshot_from_ledger(ledger)
                except (TypeError, ValueError) as exc:
                    raise VaultAuthenticationError(
                        "migrated v4 snapshot is not canonical"
                    ) from exc
                transaction.execute("PRAGMA user_version = 4")
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("v3 to v4 migration failed") from exc
        self._validate_v4_schema()
        self._verify_v4_ledger(connection)

    def _migrate_v4_to_v5(self) -> None:
        connection = self._require_connection()
        try:
            with self._transaction() as transaction:
                self._create_v5_tables(transaction)
                self._validate_v5_schema()
                transaction.execute("PRAGMA user_version = 5")
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("v4 to v5 migration failed") from exc
        self._validate_v5_schema()
        self._verify_v4_ledger(connection)

    def _migrate_v5_to_v6(self) -> None:
        connection = self._require_connection()
        try:
            with self._transaction() as transaction:
                transaction.execute(
                    "ALTER TABLE scraper_access_records RENAME TO scraper_access_records_v5"
                )
                self._create_v6_tables(transaction)
                transaction.execute(
                    """
                    INSERT INTO scraper_access_records (record_id, kind, state_enc)
                    SELECT record_id, kind, state_enc
                    FROM scraper_access_records_v5
                    """
                )
                transaction.execute("DROP TABLE scraper_access_records_v5")
                self._validate_v6_schema()
                transaction.execute("PRAGMA user_version = 6")
        except sqlite3.DatabaseError as exc:
            raise VaultSchemaError("v5 to v6 migration failed") from exc
        self._validate_v6_schema()
        self._verify_v4_ledger(connection)

    def _encrypt_capture_state(
        self,
        capture_id: str,
        *,
        blueprint_id: str,
        captured_at: str,
        content_sha256: str,
        source_provenance_hash: str,
        source_provenance_version: int,
        simhash_hex: str,
        quality_score: float,
        quality_reasons: tuple[str, ...],
        status: str,
        quarantine_reason: str | None,
        artifact_assessment: dict[str, object] | None = None,
        artifact_automatic_state: dict[str, object] | None = None,
        artifact_manual_quarantine: bool | None = None,
    ) -> bytes:
        if not isinstance(blueprint_id, str) or not blueprint_id.strip():
            raise ValueError("capture blueprint id must be nonempty")
        if not isinstance(captured_at, str) or not captured_at.strip():
            raise ValueError("capture timestamp must be nonempty")
        if not _SHA256.fullmatch(content_sha256):
            raise ValueError("capture content hash must be lowercase SHA-256")
        if not _SHA256.fullmatch(source_provenance_hash):
            raise ValueError("capture provenance must be lowercase SHA-256")
        if type(source_provenance_version) is not int or source_provenance_version not in (1, 2):
            raise ValueError("capture provenance version must be 1 or 2")
        if not _SIMHASH_HEX.fullmatch(simhash_hex):
            raise ValueError("capture SimHash must be lowercase 64-bit hex")
        if (
            isinstance(quality_score, bool)
            or not isinstance(quality_score, (int, float))
            or not math.isfinite(float(quality_score))
            or not 0.0 <= float(quality_score) <= 1.0
        ):
            raise ValueError("capture quality score must be finite and normalized")
        if not isinstance(quality_reasons, tuple) or not all(
            isinstance(reason, str) for reason in quality_reasons
        ):
            raise TypeError("capture quality reasons must be a string tuple")
        if status not in _CAPTURE_STATUSES:
            raise ValueError("unknown capture status")
        if quarantine_reason is not None and not isinstance(quarantine_reason, str):
            raise TypeError("capture quarantine reason must be a string or None")
        return self._encrypt_json(
            "captures",
            capture_id,
            "state",
            {
                "blueprint_id": blueprint_id,
                "captured_at": captured_at,
                "content_sha256": content_sha256,
                "quality_reasons": list(quality_reasons),
                "quality_score": float(quality_score),
                "quarantine_reason": quarantine_reason,
                "simhash_hex": simhash_hex,
                "source_provenance_hash": source_provenance_hash,
                "source_provenance_version": source_provenance_version,
                "status": status,
                **({"artifact_assessment": artifact_assessment} if artifact_assessment is not None else {}),
                **({"artifact_automatic_state": artifact_automatic_state,
                    "artifact_manual_quarantine": artifact_manual_quarantine}
                   if artifact_automatic_state is not None else {}),
            },
        )

    def _authenticate_capture_row(
        self, row: sqlite3.Row
    ) -> tuple[dict[str, object], CapturedDocument]:
        capture_id = str(row["id"])
        try:
            encrypted_state = row["state_enc"]
        except (IndexError, KeyError) as exc:
            raise VaultAuthenticationError(
                "capture authenticated state is missing"
            ) from exc
        if encrypted_state is None:
            raise VaultAuthenticationError("capture authenticated state is missing")
        state = self._decrypt_json("captures", capture_id, "state", encrypted_state)
        required = {
            "blueprint_id",
            "captured_at",
            "content_sha256",
            "quality_reasons",
            "quality_score",
            "quarantine_reason",
            "simhash_hex",
            "source_provenance_hash",
            "status",
        }
        versioned = required | {"source_provenance_version"}
        if not isinstance(state, dict) or set(state) not in (
            required, versioned, versioned | {"artifact_assessment"},
            versioned | {"artifact_assessment", "artifact_automatic_state", "artifact_manual_quarantine"},
        ):
            raise VaultAuthenticationError("capture authenticated state is invalid")

        provenance_version = state.get("source_provenance_version", 1)
        if type(provenance_version) is not int or provenance_version not in (1, 2):
            raise VaultAuthenticationError("capture authenticated state is invalid")
        state["source_provenance_version"] = provenance_version

        blueprint_id = state["blueprint_id"]
        captured_at = state["captured_at"]
        content_sha256 = state["content_sha256"]
        provenance = state["source_provenance_hash"]
        simhash_hex = state["simhash_hex"]
        quality_score = state["quality_score"]
        quality_reasons = state["quality_reasons"]
        status = state["status"]
        quarantine_reason = state["quarantine_reason"]
        if (
            not isinstance(blueprint_id, str)
            or not blueprint_id.strip()
            or not isinstance(captured_at, str)
            or not captured_at.strip()
            or not isinstance(content_sha256, str)
            or not _SHA256.fullmatch(content_sha256)
            or not isinstance(provenance, str)
            or not _SHA256.fullmatch(provenance)
            or not isinstance(simhash_hex, str)
            or not _SIMHASH_HEX.fullmatch(simhash_hex)
            or isinstance(quality_score, bool)
            or not isinstance(quality_score, (int, float))
            or not math.isfinite(float(quality_score))
            or not 0.0 <= float(quality_score) <= 1.0
            or not isinstance(quality_reasons, list)
            or not all(isinstance(reason, str) for reason in quality_reasons)
            or not isinstance(status, str)
            or status not in _CAPTURE_STATUSES
            or (
                quarantine_reason is not None
                and not isinstance(quarantine_reason, str)
            )
        ):
            raise VaultAuthenticationError("capture authenticated state is invalid")

        row_quality_score = row["quality_score"]
        if (
            isinstance(row_quality_score, bool)
            or not isinstance(row_quality_score, (int, float))
            or blueprint_id != row["blueprint_id"]
            or captured_at != row["captured_at"]
            or content_sha256 != row["content_sha256"]
            or provenance != row["source_provenance_hash"]
            or simhash_hex != row["simhash_hex"]
            or float(quality_score) != float(row_quality_score)
            or status != row["status"]
        ):
            raise VaultAuthenticationError(
                "capture indexed state does not match its authenticated state"
            )

        encrypted_reasons = self._decrypt_json(
            "captures", capture_id, "quality_reasons", row["quality_reasons_enc"]
        )
        if (
            not isinstance(encrypted_reasons, list)
            or not all(isinstance(reason, str) for reason in encrypted_reasons)
            or encrypted_reasons != quality_reasons
        ):
            raise VaultAuthenticationError(
                "capture quality state does not match its authenticated state"
            )
        encrypted_reason = row["quarantine_reason_enc"]
        if quarantine_reason is None:
            if encrypted_reason is not None:
                raise VaultAuthenticationError(
                    "capture quarantine state does not match its authenticated state"
                )
        else:
            if encrypted_reason is None or self._decrypt_text(
                "captures", capture_id, "quarantine_reason", encrypted_reason
            ) != quarantine_reason:
                raise VaultAuthenticationError(
                    "capture quarantine state does not match its authenticated state"
                )

        metadata = self._decrypt_json(
            "captures", capture_id, "metadata", row["metadata_enc"]
        )
        if not isinstance(metadata, dict):
            raise VaultAuthenticationError("encrypted capture metadata is not an object")
        url = self._decrypt_text("captures", capture_id, "url", row["url_enc"])
        title = self._decrypt_text("captures", capture_id, "title", row["title_enc"])
        raw_text = self._decrypt_text(
            "captures", capture_id, "raw_text", row["raw_text_enc"]
        )
        try:
            rebuilt = CapturedDocument.create(
                blueprint_id=blueprint_id,
                url=url,
                title=title,
                raw_text=raw_text,
                metadata=metadata,
                source_provenance_version=provenance_version,
            )
        except (TypeError, ValueError) as exc:
            raise VaultAuthenticationError(
                "capture failed authenticated reconstruction"
            ) from exc
        recomputed_quality = assess_quality(raw_text)
        if (
            rebuilt.content_sha256 != content_sha256
            or rebuilt.source_provenance_hash != provenance
            or f"{simhash64(raw_text):016x}" != simhash_hex
            or recomputed_quality.score != float(quality_score)
            or list(recomputed_quality.reasons) != quality_reasons
        ):
            raise VaultAuthenticationError(
                "capture derived state failed authenticated validation"
            )
        document = CapturedDocument(
            id=capture_id,
            blueprint_id=blueprint_id,
            url=url,
            title=title,
            raw_text=raw_text,
            captured_at=captured_at,
            content_sha256=content_sha256,
            source_provenance_hash=provenance,
            source_provenance_version=provenance_version,
            metadata=rebuilt.metadata,
        )
        try:
            self._validate_capture_record(document)
            if "artifact_assessment" in state:
                assessed = self._assessed_from_state(document, state["artifact_assessment"])
                if "artifact_automatic_state" in state:
                    self._validate_artifact_automatic_state(assessed, state["artifact_automatic_state"])
                    if type(state["artifact_manual_quarantine"]) is not bool or (
                        state["artifact_manual_quarantine"] and status != CAPTURE_QUARANTINED
                    ):
                        raise ValueError("invalid artifact manual quarantine state")
        except (TypeError, ValueError) as exc:
            raise VaultAuthenticationError(
                "capture record is not canonical"
            ) from exc
        return state, document

    def _encrypt_pair_state(
        self,
        pair_id: str,
        *,
        source_id: str,
        source_provenance_hash: str,
        user_approved: bool,
        status: str,
        created_at: str,
        reviewed_at: str | None,
    ) -> bytes:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("pair source id must be nonempty")
        if not _SHA256.fullmatch(source_provenance_hash):
            raise ValueError("pair provenance must be lowercase SHA-256")
        if type(user_approved) is not bool:
            raise TypeError("user_approved must be a boolean")
        if status not in _PAIR_STATUSES:
            raise ValueError("unknown pair status")
        if user_approved != (status == PAIR_APPROVED):
            raise ValueError("pair approval flag and status are inconsistent")
        if not isinstance(created_at, str) or not created_at.strip():
            raise ValueError("pair creation timestamp must be nonempty")
        if reviewed_at is not None and (
            not isinstance(reviewed_at, str) or not reviewed_at.strip()
        ):
            raise ValueError("pair review timestamp must be a nonempty string or None")
        return self._encrypt_json(
            "pairs",
            pair_id,
            "state",
            {
                "created_at": created_at,
                "reviewed_at": reviewed_at,
                "source_id": source_id,
                "source_provenance_hash": source_provenance_hash,
                "status": status,
                "user_approved": user_approved,
            },
        )

    def _authenticated_pair_state_v2(self, row: sqlite3.Row) -> dict[str, object]:
        pair_id = str(row["id"])
        encrypted_state = row["state_enc"]
        if encrypted_state is None:
            raise VaultAuthenticationError("pair authenticated state is missing")
        state = self._decrypt_json("pairs", pair_id, "state", encrypted_state)
        if not isinstance(state, dict) or set(state) != {
            "source_provenance_hash",
            "status",
            "user_approved",
        }:
            raise VaultAuthenticationError("legacy v2 pair state is invalid")
        provenance = state["source_provenance_hash"]
        status = state["status"]
        approved = state["user_approved"]
        if (
            not isinstance(provenance, str)
            or not _SHA256.fullmatch(provenance)
            or not isinstance(status, str)
            or status not in _PAIR_STATUSES
            or type(approved) is not bool
            or approved != (status == PAIR_APPROVED)
            or provenance != row["source_provenance_hash"]
            or status != row["status"]
            or type(row["user_approved"]) is not int
            or row["user_approved"] not in (0, 1)
            or approved != bool(row["user_approved"])
        ):
            raise VaultAuthenticationError("legacy v2 pair state is invalid")
        return state

    def _authenticated_pair_state(self, row: sqlite3.Row) -> dict[str, object]:
        pair_id = str(row["id"])
        try:
            encrypted_state = row["state_enc"]
        except (IndexError, KeyError) as exc:
            raise VaultAuthenticationError(
                "pair authenticated state is missing"
            ) from exc
        if encrypted_state is None:
            raise VaultAuthenticationError("pair authenticated state is missing")
        state = self._decrypt_json("pairs", pair_id, "state", encrypted_state)
        if not isinstance(state, dict) or set(state) != {
            "created_at",
            "reviewed_at",
            "source_id",
            "source_provenance_hash",
            "status",
            "user_approved",
        }:
            raise VaultAuthenticationError("pair authenticated state is invalid")
        source_id = state["source_id"]
        provenance = state["source_provenance_hash"]
        status = state["status"]
        approved = state["user_approved"]
        created_at = state["created_at"]
        reviewed_at = state["reviewed_at"]
        if (
            not isinstance(source_id, str)
            or not source_id.strip()
            or not isinstance(provenance, str)
            or not _SHA256.fullmatch(provenance)
            or not isinstance(status, str)
            or status not in _PAIR_STATUSES
            or type(approved) is not bool
            or approved != (status == PAIR_APPROVED)
            or not isinstance(created_at, str)
            or not created_at.strip()
            or (
                reviewed_at is not None
                and (
                    not isinstance(reviewed_at, str)
                    or not reviewed_at.strip()
                )
            )
        ):
            raise VaultAuthenticationError("pair authenticated state is invalid")
        mirrored_approved = row["user_approved"]
        if (
            type(mirrored_approved) is not int
            or mirrored_approved not in (0, 1)
            or source_id != row["source_id"]
            or provenance != row["source_provenance_hash"]
            or status != row["status"]
            or approved != bool(mirrored_approved)
            or created_at != row["created_at"]
            or reviewed_at != row["reviewed_at"]
        ):
            raise VaultAuthenticationError(
                "pair indexed state does not match its authenticated state"
            )
        return state

    @contextmanager
    def corpus_mutation_guard(self) -> Iterator[None]:
        """Serialize a corpus operation without merging its atomic SQL writes.

        The vault already owns the exclusive process lock for its lifetime.
        All transactions share this reentrant mutex, including training writes.
        """
        with self._mutex:
            self._require_connection()
            yield

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._mutex:
            connection = self._require_connection()
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                self._protect_database_files()
                connection.execute("COMMIT")
            except _DurableAuditRejection:
                connection.execute("COMMIT")
                self._protect_database_files()
                raise
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise

    @staticmethod
    def _validate_limit(limit: int | None) -> int | None:
        if limit is None:
            return None
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer or None")
        if limit < 1:
            raise ValueError("limit must be positive")
        return limit

    def _validate_capture_record(
        self, document: CapturedDocument
    ) -> CapturedDocument:
        if type(document) is not CapturedDocument:
            raise TypeError("document must be a CapturedDocument")
        if (
            type(document.id) is not str
            or document.id != document.id.strip()
            or not document.id.isascii()
            or not 1 <= len(document.id.encode("ascii")) <= 128
            or any(
                ord(character) < 0x21 or ord(character) >= 0x7F
                for character in document.id
            )
        ):
            raise ValueError(
                "capture id must be a bounded visible ASCII identifier"
            )
        try:
            self._parse_utc_timestamp(
                document.captured_at, "capture captured_at"
            )
        except VaultAuthenticationError as exc:
            raise ValueError(
                "capture timestamp must be canonical UTC"
            ) from exc
        try:
            canonical_document = CapturedDocument.create(
                blueprint_id=document.blueprint_id,
                url=document.url,
                title=document.title,
                raw_text=document.raw_text,
                metadata=document.metadata,
                source_provenance_version=document.source_provenance_version,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("capture is not canonically valid") from exc
        if (
            canonical_document.blueprint_id != document.blueprint_id
            or canonical_document.url != document.url
            or canonical_document.title != document.title
            or canonical_document.raw_text != document.raw_text
            or canonical_document.content_sha256 != document.content_sha256
            or canonical_document.source_provenance_hash
            != document.source_provenance_hash
            or canonical_document.source_provenance_version
            != document.source_provenance_version
            or canonical_document.metadata != document.metadata
        ):
            raise ValueError("capture derived state is not canonical")
        source_kind = document.metadata.get("source_kind")
        if source_kind in _ARTIFACT_KINDS:
            if document.source_provenance_version != 2:
                raise ValueError("artifact captures require provenance version 2")
            self._validate_artifact_metadata(document)
        elif document.source_provenance_version != 1:
            raise ValueError("generic captures require provenance version 1")
        return document

    @staticmethod
    def _validate_artifact_metadata(document: CapturedDocument) -> None:
        metadata = document.metadata
        kind = metadata["source_kind"]

        def digest(name: str, size: int = 64, *, optional: bool = False) -> None:
            value = metadata[name]
            if optional and value is None:
                return
            if type(value) is not str or re.fullmatch(r"[a-f0-9]{" + str(size) + "}", value) is None:
                raise ValueError("invalid artifact provenance digest")

        def positive(name: str) -> None:
            if type(metadata[name]) is not int or not 1 <= metadata[name] <= _MAX_AUDIT_SEQUENCE:
                raise ValueError("invalid artifact provenance identifier")

        if kind == "vbulletin_post":
            if not _POST_METADATA <= metadata.keys() or metadata.keys() - (_POST_METADATA | _POST_OPTIONAL_METADATA):
                raise ValueError("invalid post provenance fields")
            for name in ("thread_id", "thread_page", "post_index"):
                positive(name)
            if "forum_id" in metadata:
                positive("forum_id")
            post_id = metadata["post_id"]
            if not ((type(post_id) is int and post_id > 0) or
                    (type(post_id) is str and re.fullmatch(r"post_message_[1-9][0-9]*", post_id))):
                raise ValueError("invalid post provenance identifier")
            for name in ("has_code", "has_attachment", "truncated", "source_text_truncated"):
                if name in metadata and type(metadata[name]) is not bool:
                    raise ValueError("invalid post provenance flag")
            for name in ("original_utf8_bytes", "retained_utf8_bytes", "quote_utf8_bytes"):
                if type(metadata[name]) is not int or not 0 <= metadata[name] <= 4_000_000:
                    raise ValueError("invalid post provenance byte count")
            if metadata["retained_utf8_bytes"] + metadata["quote_utf8_bytes"] > metadata["original_utf8_bytes"]:
                raise ValueError("invalid post provenance byte count")
            if "ingestion" in metadata and metadata["ingestion"] != "playwright":
                raise ValueError("invalid post provenance collector")
            if "quote_ratio" in metadata:
                ratio = metadata["quote_ratio"]
                if type(ratio) not in (float, int) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
                    raise ValueError("invalid post quote ratio")
            chunk_fields = {"source_document_sha256", "chunk_index", "chunk_count"}
            if metadata.keys() & chunk_fields:
                if not chunk_fields <= metadata.keys():
                    raise ValueError("incomplete post segment provenance")
                digest("source_document_sha256")
                positive("chunk_index")
                positive("chunk_count")
                if metadata["chunk_index"] > metadata["chunk_count"]:
                    raise ValueError("invalid post segment provenance")
            elif metadata["retained_utf8_bytes"] != len(document.raw_text.encode("utf-8")):
                raise ValueError("post provenance byte count does not match text")
            return
        if kind == "github_blob":
            if metadata.keys() != _GITHUB_METADATA:
                raise ValueError("invalid GitHub provenance fields")
            RepositoryInspection(
                RepositoryIdentity(metadata["repository_id"], metadata["repository_node_id"],
                                   RepositoryLocator(metadata["owner"], metadata["repository"])),
                metadata["commit_sha"], metadata["tree_sha"], metadata["license_spdx"],
                metadata["license_identity_sha256"], metadata["collector_policy_sha256"],
            )
            base = f"https://api.github.com/repos/{metadata['owner']}/{metadata['repository']}"
            if metadata["repository_api_url"] != base or metadata["license_api_url"] not in (None, base + "/license"):
                raise ValueError("invalid GitHub provenance URL")
            digest("blob_sha", 40)
            digest("rights_record_sha256", optional=True)
            path = metadata["path"]
        else:
            if metadata.keys() != _ATTACHMENT_METADATA:
                raise ValueError("invalid attachment provenance fields")
            parent_post_id = metadata["parent_post_id"]
            if type(parent_post_id) is not str or re.fullmatch(r"post_message_[1-9][0-9]*", parent_post_id) is None:
                raise ValueError("invalid attachment parent post identifier")
            positive("attachment_id")
            digest("parent_post_provenance_hash")
            url = metadata["canonical_url"]
            if not StudioVault._is_canonical_attachment_url(url, metadata["attachment_id"]):
                raise ValueError("invalid attachment provenance URL")
            path = metadata["filename"]
            if type(path) is not str or "/" in path:
                raise ValueError("invalid attachment filename")
        normalize_artifact_path(path)
        require_allowed_extension(path)
        digest("raw_byte_sha256")
        digest("collector_policy_sha256")

    @staticmethod
    def _is_canonical_attachment_url(value: object, attachment_id: int) -> bool:
        if type(value) is not str or not value.isascii() or len(value) > 4096:
            return False
        try:
            parsed = urlsplit(value)
            host = parsed.hostname
            if (parsed.scheme != "https" or not host or parsed.port is not None
                    or parsed.username is not None or parsed.password is not None
                    or parsed.fragment or "%" in host):
                return False
            if ":" in host:
                authority = f"[{ipaddress.IPv6Address(host).compressed}]"
            elif re.fullmatch(r"[0-9.]+", host):
                authority = str(ipaddress.IPv4Address(host))
            else:
                if len(host) > 253 or any(
                    re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) is None
                    for label in host.split(".")
                ):
                    return False
                authority = host
            return value == f"https://{authority}/attachment.php?attachmentid={attachment_id}"
        except (ValueError, UnicodeError):
            # Parse errors may contain userinfo; raise the fixed code only after
            # this exception context has been discarded by returning.
            return False

    def _validate_assessed_capture(self, item: AssessedCapture) -> AssessedCapture:
        if type(item) is not AssessedCapture:
            raise TypeError("batch items must be AssessedCapture values")
        document = self._validate_capture_record(item.document)
        if document.source_provenance_version != 2 or document.metadata.get("source_kind") not in _ARTIFACT_KINDS:
            raise ValueError("assessed captures require artifact provenance")
        if type(item.score) is not int or not 0 <= item.score <= 100:
            raise ValueError("invalid artifact assessment score")
        for values, allowed in ((item.signals, TECHNICAL_SIGNALS), (item.reasons, _ASSESSMENT_REASONS)):
            if type(values) is not tuple or any(type(value) is not str or value not in allowed for value in values):
                raise ValueError("invalid artifact assessment codes")
            if len(values) != len(set(values)):
                raise ValueError("duplicate artifact assessment codes")
        if item.disposition not in ("review_candidate", "quarantined"):
            raise ValueError("invalid artifact assessment disposition")
        if item.disposition == "review_candidate" and (item.reasons or item.score < 55 or len(item.signals) < 2):
            raise ValueError("inconsistent artifact assessment")
        if type(item.raw_sha256) is not str or item.raw_sha256 != document.content_sha256:
            raise ValueError("artifact raw digest does not match text")
        if type(item.canonical_text_sha256) is not str or item.canonical_text_sha256 != hashlib.sha256(canonicalize_text(document.raw_text).encode("utf-8")).hexdigest():
            raise ValueError("artifact canonical digest does not match text")
        if (type(item.fingerprints) is not frozenset or
                any(type(value) is not int or not 0 <= value < 2 ** 64 for value in item.fingerprints) or
                item.fingerprints != winnowed_five_token_fingerprints(document.raw_text)):
            raise ValueError("artifact fingerprints do not match text")
        path, _strength = self._artifact_dedupe_identity(document)
        reject_secret_material(document.raw_text, path)
        return item

    @staticmethod
    def _assessment_state(item: AssessedCapture) -> dict[str, object]:
        return {
            "score": item.score, "signals": list(item.signals), "reasons": list(item.reasons),
            "disposition": item.disposition, "raw_sha256": item.raw_sha256,
            "canonical_text_sha256": item.canonical_text_sha256,
            "fingerprints": sorted(item.fingerprints),
        }

    @staticmethod
    def _artifact_automatic_state(item: AssessedCapture, winner: str, duplicate_reason: str | None) -> dict[str, object]:
        reasons = set(item.reasons)
        if item.disposition == "quarantined" and not reasons:
            reasons.add("insufficient_technical_signal")
        if duplicate_reason is not None:
            reasons.add(duplicate_reason)
        return {
            "winner_provenance_hash": winner,
            "duplicate_reason": duplicate_reason,
            "status": CAPTURE_QUARANTINED if reasons else CAPTURE_READY,
            "quarantine_reason": ",".join(sorted(reasons)) or None,
        }

    def _validate_artifact_automatic_state(self, item: AssessedCapture, state: object) -> None:
        if type(state) is not dict or set(state) != {
            "winner_provenance_hash", "duplicate_reason", "status", "quarantine_reason",
        }:
            raise ValueError("invalid artifact automatic state")
        winner, reason = state["winner_provenance_hash"], state["duplicate_reason"]
        if type(winner) is not str or _SHA256.fullmatch(winner) is None:
            raise ValueError("invalid artifact automatic winner")
        if reason is not None and (type(reason) is not str or reason not in _DEDUPE_REASONS):
            raise ValueError("invalid artifact automatic reason")
        if (reason is None) != (winner == item.document.source_provenance_hash):
            raise ValueError("inconsistent artifact automatic winner")
        if state != self._artifact_automatic_state(item, winner, reason):
            raise ValueError("inconsistent artifact automatic state")

    def _artifact_controls(self, state: dict[str, object], document: CapturedDocument) -> dict[str, object]:
        if "artifact_assessment" not in state:
            return {}
        automatic = state.get("artifact_automatic_state")
        if automatic is None:
            item = self._assessed_from_state(document, state["artifact_assessment"])
            automatic = self._artifact_automatic_state(item, document.source_provenance_hash, None)
        return {
            "artifact_automatic_state": automatic,
            # Old assessed rows have no reliable origin marker. Preserve any
            # existing quarantine until an explicit operator status change.
            "artifact_manual_quarantine": state.get(
                "artifact_manual_quarantine", state["status"] == CAPTURE_QUARANTINED,
            ),
        }

    def _assessed_from_state(self, document: CapturedDocument, state: object) -> AssessedCapture:
        if type(state) is not dict or set(state) != {
            "score", "signals", "reasons", "disposition", "raw_sha256", "canonical_text_sha256", "fingerprints",
        }:
            raise ValueError("invalid stored artifact assessment")
        if any(type(state[name]) is not list for name in ("signals", "reasons", "fingerprints")):
            raise ValueError("invalid stored artifact assessment")
        fingerprints = state["fingerprints"]
        if any(type(value) is not int for value in fingerprints) or fingerprints != sorted(set(fingerprints)):
            raise ValueError("invalid stored artifact fingerprints")
        return self._validate_assessed_capture(AssessedCapture(
            document=document, score=state["score"], signals=tuple(state["signals"]),
            reasons=tuple(state["reasons"]), disposition=state["disposition"],
            raw_sha256=state["raw_sha256"], canonical_text_sha256=state["canonical_text_sha256"],
            fingerprints=frozenset(fingerprints),
        ))

    @staticmethod
    def _artifact_dedupe_identity(document: CapturedDocument) -> tuple[str, int]:
        metadata = document.metadata
        if metadata["source_kind"] == "github_blob":
            return metadata["path"], 3
        if metadata["source_kind"] == "forum_attachment":
            return metadata["filename"], 2
        return f"posts/{metadata['post_id']}.txt", 1

    def _artifact_dedupe_record(self, item: AssessedCapture) -> DedupeRecord:
        path, strength = self._artifact_dedupe_identity(item.document)
        return DedupeRecord(ArtifactSegment(
            text=item.document.raw_text, index=0, count=1, score=item.score,
            signals=item.signals, disposition=item.disposition, reasons=item.reasons,
            raw_sha256=item.raw_sha256, canonical_text_sha256=item.canonical_text_sha256,
            fingerprints=item.fingerprints,
        ), path, item.document.source_provenance_hash, strength)

    def add_assessed_capture_batch(self, items: Sequence[AssessedCapture]) -> tuple[CaptureAddResult, ...]:
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise TypeError("assessed captures must be a sequence")
        prepared = tuple(self._validate_assessed_capture(item) for item in items)
        if not prepared:
            return ()
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            retained: dict[str, AssessedCapture] = {}
            states: dict[str, dict[str, object]] = {}
            for capture_id, (state, document) in ledger["captures"].items():
                if capture_id in ledger["capture_tombstones"] or "artifact_assessment" not in state:
                    continue
                item = self._assessed_from_state(document, state["artifact_assessment"])
                provenance = document.source_provenance_hash
                if provenance in retained:
                    raise VaultAuthenticationError("ambiguous retained artifact provenance")
                retained[provenance] = item
                states[capture_id] = state
            by_provenance = dict(retained)
            new: dict[str, AssessedCapture] = {}
            all_ids = set(ledger["captures"])
            results = []
            for item in prepared:
                document = item.document
                provenance = document.source_provenance_hash
                prior = by_provenance.get(provenance)
                if prior is not None:
                    if self._assessment_state(prior) != self._assessment_state(item):
                        raise ValueError("conflicting artifact assessment")
                    if document.id in all_ids and document.id != prior.document.id:
                        raise ValueError("conflicting artifact capture identifier")
                    results.append(CaptureAddResult(prior.document.id, False))
                    continue
                if document.id in all_ids or any(
                    existing.source_provenance_hash == provenance
                    for capture_id, (_state, existing) in ledger["captures"].items()
                    if capture_id not in ledger["capture_tombstones"]
                ):
                    raise ValueError("conflicting artifact capture identity")
                by_provenance[provenance] = new[provenance] = item
                all_ids.add(document.id)
                results.append(CaptureAddResult(document.id, True))
            decisions = choose_dedupe_winners(
                tuple(self._artifact_dedupe_record(item) for item in new.values()),
                tuple(self._artifact_dedupe_record(item) for item in retained.values()),
            )
            for decision in decisions:
                item = by_provenance[decision.provenance_hash]
                document = item.document
                automatic = self._artifact_automatic_state(item, decision.winner_provenance_hash, decision.reason)
                status, reason = automatic["status"], automatic["quarantine_reason"]
                old_state = states.get(document.id)
                if old_state is None:
                    self._insert_assessed_capture(connection, item, automatic)
                else:
                    manual = self._artifact_controls(old_state, document)["artifact_manual_quarantine"]
                    old_reason = old_state["quarantine_reason"]
                    if manual:
                        status, reason = old_state["status"], old_reason
                    elif status == CAPTURE_READY and old_state["status"] == CAPTURE_DISTILLED:
                        status = CAPTURE_DISTILLED
                    if (old_state["status"] != status or old_reason != reason
                            or old_state.get("artifact_automatic_state") != automatic):
                        self._update_assessed_capture_status(
                            connection, document.id, old_state, status, reason, automatic, manual,
                        )
            return tuple(results)

    def _insert_assessed_capture(self, connection, item: AssessedCapture, automatic: dict[str, object]) -> None:
        document = item.document
        status, reason = automatic["status"], automatic["quarantine_reason"]
        quality = assess_quality(document.raw_text)
        simhash_hex = f"{simhash64(document.raw_text):016x}"
        state_enc = self._encrypt_capture_state(
            document.id, blueprint_id=document.blueprint_id, captured_at=document.captured_at,
            content_sha256=document.content_sha256, source_provenance_hash=document.source_provenance_hash,
            source_provenance_version=document.source_provenance_version, simhash_hex=simhash_hex,
            quality_score=quality.score, quality_reasons=quality.reasons, status=status,
            quarantine_reason=reason, artifact_assessment=self._assessment_state(item),
            artifact_automatic_state=automatic, artifact_manual_quarantine=False,
        )
        connection.execute(
            """INSERT INTO captures (
                id, blueprint_id, captured_at, content_sha256, source_provenance_hash,
                url_enc, title_enc, raw_text_enc, metadata_enc, simhash_hex, quality_score,
                quality_reasons_enc, status, quarantine_reason_enc, state_enc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (document.id, document.blueprint_id, document.captured_at, document.content_sha256,
             document.source_provenance_hash,
             self._encrypt_text("captures", document.id, "url", document.url),
             self._encrypt_text("captures", document.id, "title", document.title),
             self._encrypt_text("captures", document.id, "raw_text", document.raw_text),
             self._encrypt_json("captures", document.id, "metadata", thaw_json_value(document.metadata)),
             simhash_hex, quality.score,
             self._encrypt_json("captures", document.id, "quality_reasons", quality.reasons), status,
             self._encrypt_text("captures", document.id, "quarantine_reason", reason) if reason is not None else None,
             state_enc),
        )
        self._append_capture_event(connection, capture_id=document.id, kind="CREATED", created_at=document.captured_at)

    def _update_assessed_capture_status(self, connection, capture_id: str, state: dict[str, object], status: str, reason: str | None,
                                       automatic: dict[str, object], manual: bool) -> None:
        state_enc = self._encrypt_capture_state(
            capture_id, blueprint_id=state["blueprint_id"], captured_at=state["captured_at"],
            content_sha256=state["content_sha256"], source_provenance_hash=state["source_provenance_hash"],
            source_provenance_version=state["source_provenance_version"], simhash_hex=state["simhash_hex"],
            quality_score=state["quality_score"], quality_reasons=tuple(state["quality_reasons"]),
            status=status, quarantine_reason=reason, artifact_assessment=state.get("artifact_assessment"),
            artifact_automatic_state=automatic, artifact_manual_quarantine=manual,
        )
        cursor = connection.execute(
            "UPDATE captures SET status = ?, quarantine_reason_enc = ?, state_enc = ? WHERE id = ?",
            (status, self._encrypt_text("captures", capture_id, "quarantine_reason", reason) if reason is not None else None,
             state_enc, capture_id),
        )
        if cursor.rowcount != 1:
            raise VaultAuthenticationError("artifact capture disappeared during batch")
        self._append_capture_event(connection, capture_id=capture_id,
                                   kind="QUARANTINED" if status == CAPTURE_QUARANTINED else "STATUS_CHANGED",
                                   created_at=self._now_utc_text())

    def add_capture(
        self,
        document: CapturedDocument,
        *,
        near_duplicate_distance: int = 3,
        quality_assessment: QualityAssessment | None = None,
        additional_quarantine_reasons: Sequence[str] = (),
    ) -> str:
        return self.add_capture_result(
            document,
            near_duplicate_distance=near_duplicate_distance,
            quality_assessment=quality_assessment,
            additional_quarantine_reasons=additional_quarantine_reasons,
        ).capture_id

    def add_capture_result(
        self,
        document: CapturedDocument,
        *,
        near_duplicate_distance: int = 3,
        quality_assessment: QualityAssessment | None = None,
        additional_quarantine_reasons: Sequence[str] = (),
    ) -> CaptureAddResult:
        self._validate_capture_record(document)
        if document.metadata.get("source_kind") in {"github_blob", "forum_attachment"}:
            raise ValueError("source artifacts require an assessed capture batch")
        if (
            isinstance(near_duplicate_distance, bool)
            or not isinstance(near_duplicate_distance, int)
            or not 0 <= near_duplicate_distance <= 64
        ):
            raise ValueError("near_duplicate_distance must be from zero to 64")
        if quality_assessment is not None and not isinstance(
            quality_assessment, QualityAssessment
        ):
            raise TypeError("quality_assessment must be a QualityAssessment or None")
        if isinstance(additional_quarantine_reasons, (str, bytes)) or not isinstance(
            additional_quarantine_reasons, Sequence
        ):
            raise TypeError("additional quarantine reasons must be a string sequence")
        if len(additional_quarantine_reasons) > 32:
            raise ValueError("at most 32 additional quarantine reasons are permitted")
        normalized_additional: list[str] = []
        for reason in additional_quarantine_reasons:
            if not isinstance(reason, str):
                raise TypeError("each additional quarantine reason must be a string")
            normalized = reason.strip()
            if not normalized:
                raise ValueError("each additional quarantine reason must be nonempty")
            if len(normalized.encode("utf-8")) > 256:
                raise ValueError(
                    "each additional quarantine reason must be at most 256 UTF-8 bytes"
                )
            if _POLICY_REASON_CODE.fullmatch(normalized) is None:
                raise ValueError(
                    "each additional quarantine reason must be a bounded reason code"
                )
            if normalized not in normalized_additional:
                normalized_additional.append(normalized)

        document_quality = assess_quality(document.raw_text)
        policy_quality = quality_assessment or document_quality
        if not isinstance(policy_quality.reasons, tuple):
            raise TypeError(
                "quality assessment quarantine reasons must be a tuple"
            )
        for reason in policy_quality.reasons:
            if not isinstance(reason, str):
                raise TypeError(
                    "each quality quarantine reason must be a string"
                )
            if _POLICY_REASON_CODE.fullmatch(reason) is None:
                raise ValueError(
                    "each quality quarantine reason must be a bounded reason code"
                )
        policy_reasons = tuple(
            sorted(set((*policy_quality.reasons, *normalized_additional)))
        )
        merge_policy_reasons = (
            quality_assessment is not None or bool(normalized_additional)
        )
        fingerprint = simhash64(document.raw_text)
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            provenance_matches = [
                (capture_id, state)
                for capture_id, (state, existing_document) in ledger[
                    "captures"
                ].items()
                if capture_id not in ledger["capture_tombstones"]
                and state["source_provenance_hash"]
                == document.source_provenance_hash
                and existing_document.blueprint_id == document.blueprint_id
                and existing_document.url == document.url
                and existing_document.title == document.title
                and existing_document.raw_text == document.raw_text
                and existing_document.source_provenance_version
                == document.source_provenance_version
                and existing_document.metadata == document.metadata
            ]
            if provenance_matches:
                existing_id, existing_state = min(
                    provenance_matches,
                    key=lambda item: (
                        item[1]["status"] == CAPTURE_QUARANTINED,
                        str(item[1]["captured_at"]),
                        item[0],
                    ),
                )
                if policy_reasons:
                    existing_reason = existing_state["quarantine_reason"]
                    if existing_reason is None:
                        missing_policy_reasons = policy_reasons
                        opaque_existing_reason = ""
                    else:
                        opaque_existing_reason = str(existing_reason)
                        existing_policy_reasons: set[str] = set()
                        for component in opaque_existing_reason.split(";"):
                            if component.startswith("low_quality:"):
                                existing_policy_reasons.update(
                                    reason
                                    for reason in component.removeprefix(
                                        "low_quality:"
                                    ).split(",")
                                    if reason
                                )
                        missing_policy_reasons = tuple(
                            reason
                            for reason in policy_reasons
                            if reason not in existing_policy_reasons
                        )
                    if not missing_policy_reasons:
                        merged_reason = opaque_existing_reason
                    else:
                        policy_component = "low_quality:" + ",".join(
                            missing_policy_reasons
                        )
                        merged_reason = (
                            policy_component
                            if not opaque_existing_reason
                            else opaque_existing_reason + ";" + policy_component
                        )
                    if (
                        existing_state["status"] != CAPTURE_QUARANTINED
                        or merged_reason != existing_reason
                    ):
                        self.quarantine_capture(existing_id, merged_reason)
                return CaptureAddResult(existing_id, False)
            exact_id = next(
                (
                    capture_id
                    for capture_id, (state, _existing_document) in ledger[
                        "captures"
                    ].items()
                    if capture_id not in ledger["capture_tombstones"]
                    and state["status"] != CAPTURE_QUARANTINED
                    and state["content_sha256"] == document.content_sha256
                ),
                None,
            )
            near_match = None
            if exact_id is None:
                near_match = min(
                    (
                        DuplicateMatch(
                            capture_id=capture_id,
                            distance=hamming_distance(
                                fingerprint,
                                int(str(state["simhash_hex"]), 16),
                            ),
                        )
                        for capture_id, (state, _existing_document) in ledger[
                            "captures"
                        ].items()
                        if capture_id not in ledger["capture_tombstones"]
                        and state["status"] != CAPTURE_QUARANTINED
                    ),
                    key=lambda match: (match.distance, match.capture_id),
                    default=None,
                )
                if (
                    near_match is not None
                    and near_match.distance > near_duplicate_distance
                ):
                    near_match = None
            status = CAPTURE_READY
            quarantine_parts: list[str] = []
            if exact_id is not None:
                quarantine_parts.append(f"exact_duplicate:{exact_id}")
            elif near_match is not None:
                quarantine_parts.append(
                    f"near_duplicate:{near_match.capture_id}:{near_match.distance}"
                )
            if policy_reasons and (
                not quarantine_parts or merge_policy_reasons
            ):
                quarantine_parts.append(f"low_quality:{','.join(policy_reasons)}")
            if quarantine_parts:
                status = CAPTURE_QUARANTINED
            quarantine_reason = ";".join(quarantine_parts) or None
            simhash_hex = f"{fingerprint:016x}"
            state_enc = self._encrypt_capture_state(
                document.id,
                blueprint_id=document.blueprint_id,
                captured_at=document.captured_at,
                content_sha256=document.content_sha256,
                source_provenance_hash=document.source_provenance_hash,
                source_provenance_version=document.source_provenance_version,
                simhash_hex=simhash_hex,
                quality_score=document_quality.score,
                quality_reasons=document_quality.reasons,
                status=status,
                quarantine_reason=quarantine_reason,
            )

            with self._transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO captures (
                        id, blueprint_id, captured_at, content_sha256,
                        source_provenance_hash, url_enc, title_enc, raw_text_enc,
                        metadata_enc, simhash_hex, quality_score,
                        quality_reasons_enc, status, quarantine_reason_enc,
                        state_enc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.blueprint_id,
                        document.captured_at,
                        document.content_sha256,
                        document.source_provenance_hash,
                        self._encrypt_text("captures", document.id, "url", document.url),
                        self._encrypt_text(
                            "captures", document.id, "title", document.title
                        ),
                        self._encrypt_text(
                            "captures", document.id, "raw_text", document.raw_text
                        ),
                        self._encrypt_json(
                            "captures", document.id, "metadata", thaw_json_value(document.metadata)
                        ),
                        simhash_hex,
                        document_quality.score,
                        self._encrypt_json(
                            "captures",
                            document.id,
                            "quality_reasons",
                            document_quality.reasons,
                        ),
                        status,
                        self._encrypt_text(
                            "captures",
                            document.id,
                            "quarantine_reason",
                            quarantine_reason,
                        )
                        if quarantine_reason is not None
                        else None,
                        state_enc,
                    ),
                )
                self._append_capture_event(
                    connection,
                    capture_id=document.id,
                    kind="CREATED",
                    created_at=document.captured_at,
                )
        return CaptureAddResult(document.id, True)

    def _capture_from_row(self, row: sqlite3.Row) -> CapturedDocument:
        return self._authenticate_capture_row(row)[1]

    def _visible_capture_from_ledger(
        self, ledger: dict[str, object], capture_id: str
    ) -> tuple[dict[str, object], CapturedDocument]:
        record = ledger["captures"].get(capture_id)
        if record is None or capture_id in ledger["capture_tombstones"]:
            raise RecordNotFoundError(f"capture not found: {capture_id}")
        return record

    def get_capture(self, capture_id: str) -> CapturedDocument:
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            return self._visible_capture_from_ledger(ledger, capture_id)[1]

    def list_captures(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[CapturedDocument]:
        limit = self._validate_limit(limit)
        if status is not None and status not in _CAPTURE_STATUSES:
            raise ValueError("unknown capture status")
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            documents = [
                document
                for capture_id, (state, document) in ledger["captures"].items()
                if capture_id not in ledger["capture_tombstones"]
                and (status is None or state["status"] == status)
            ]
            documents.sort(key=lambda item: (item.captured_at, item.id))
            return documents if limit is None else documents[:limit]

    def get_capture_status(self, capture_id: str) -> str:
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            return str(
                self._visible_capture_from_ledger(ledger, capture_id)[0]["status"]
            )

    def set_capture_status(self, capture_id: str, status: str) -> None:
        if status not in _CAPTURE_STATUSES:
            raise ValueError("unknown capture status")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            self._set_capture_status_in_connection(connection, ledger, capture_id, status)

    def _set_capture_status_in_connection(
        self, connection: sqlite3.Connection, ledger: dict[str, object],
        capture_id: str, status: str,
    ) -> None:
        state, document = self._visible_capture_from_ledger(ledger, capture_id)
        controls = self._artifact_controls(state, document)
        if controls:
            controls["artifact_manual_quarantine"] = status == CAPTURE_QUARANTINED
        quarantine_reason = (
            state["quarantine_reason"] if status == CAPTURE_QUARANTINED else None
        )
        state_enc = self._encrypt_capture_state(
            capture_id,
            blueprint_id=str(state["blueprint_id"]),
            captured_at=str(state["captured_at"]),
            content_sha256=str(state["content_sha256"]),
            source_provenance_hash=str(state["source_provenance_hash"]),
            source_provenance_version=int(state["source_provenance_version"]),
            simhash_hex=str(state["simhash_hex"]),
            quality_score=float(state["quality_score"]),
            quality_reasons=tuple(state["quality_reasons"]),
            status=status,
            quarantine_reason=quarantine_reason,
            artifact_assessment=state.get("artifact_assessment"),
            **controls,
        )
        cursor = connection.execute(
            """
            UPDATE captures
            SET status = ?, quarantine_reason_enc = ?, state_enc = ?
            WHERE id = ?
            """,
            (
                status,
                self._encrypt_text(
                    "captures", capture_id, "quarantine_reason", str(quarantine_reason),
                ) if quarantine_reason is not None else None,
                state_enc,
                capture_id,
            ),
        )
        if cursor.rowcount != 1:
            raise RecordNotFoundError(f"capture not found: {capture_id}")
        self._append_capture_event(
            connection,
            capture_id=capture_id,
            kind="STATUS_CHANGED",
            created_at=self._now_utc_text(),
        )

    def quarantine_capture(self, capture_id: str, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("quarantine reason must be a nonempty string")
        reason = reason.strip()
        if len(reason.encode("utf-8")) > 2048:
            raise ValueError("quarantine reason exceeds the UTF-8 size limit")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            state, document = self._visible_capture_from_ledger(
                ledger, capture_id
            )
            controls = self._artifact_controls(state, document)
            if controls:
                controls["artifact_manual_quarantine"] = True
            encrypted_reason = self._encrypt_text(
                "captures", capture_id, "quarantine_reason", reason
            )
            state_enc = self._encrypt_capture_state(
                capture_id,
                blueprint_id=str(state["blueprint_id"]),
                captured_at=str(state["captured_at"]),
                content_sha256=str(state["content_sha256"]),
                source_provenance_hash=str(state["source_provenance_hash"]),
                source_provenance_version=int(state["source_provenance_version"]),
                simhash_hex=str(state["simhash_hex"]),
                quality_score=float(state["quality_score"]),
                quality_reasons=tuple(state["quality_reasons"]),
                status=CAPTURE_QUARANTINED,
                quarantine_reason=reason,
                artifact_assessment=state.get("artifact_assessment"),
                **controls,
            )
            cursor = connection.execute(
                """
                UPDATE captures
                SET status = ?, quarantine_reason_enc = ?, state_enc = ?
                WHERE id = ?
                """,
                (CAPTURE_QUARANTINED, encrypted_reason, state_enc, capture_id),
            )
            if cursor.rowcount != 1:
                raise RecordNotFoundError(f"capture not found: {capture_id}")
            self._append_capture_event(
                connection,
                capture_id=capture_id,
                kind="QUARANTINED",
                created_at=self._now_utc_text(),
            )

    def get_capture_quarantine_reason(self, capture_id: str) -> str | None:
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            reason = self._visible_capture_from_ledger(
                ledger, capture_id
            )[0]["quarantine_reason"]
            return None if reason is None else str(reason)

    def delete_capture(self, capture_id: str) -> None:
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            state, _document = self._visible_capture_from_ledger(
                ledger, capture_id
            )
            if state["status"] != CAPTURE_QUARANTINED:
                raise CaptureDeletionError(
                    "capture must be quarantined before logical deletion"
                )
            sequence, anchor = self._audit_context(connection, verify=False)
            created_at = self._now_utc_text()
            tombstone_id, tombstone_digest = self._insert_tombstone_record(
                connection,
                record_kind="CAPTURE",
                record_id=capture_id,
                sequence=sequence,
                created_at=created_at,
                previous_tombstone_sha256=anchor["tombstone_head_sha256"],
            )
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind="CAPTURE_DELETED",
                subject_id=capture_id,
                created_at=created_at,
                record_digests=(
                    self._record_token(
                        "TOMBSTONE", tombstone_id, tombstone_digest
                    ),
                ),
                head_updates={"tombstone_head_sha256": tombstone_digest},
            )

    def read_local_state_anchor(self, scope: str) -> LocalStateAnchor | None:
        """Return the latest authenticated external-history anchor for ``scope``."""

        canonical_scope = self._require_local_state_scope(scope)
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            history = self._local_state_anchor_histories(ledger).get(
                canonical_scope, ()
            )
            return history[-1] if history else None

    def advance_local_state_anchor(
        self,
        scope: str,
        *,
        expected_count: int,
        expected_head_sha256: str | None,
        new_count: int,
        new_head_sha256: str,
    ) -> LocalStateAnchorAppendResult:
        """Atomically advance a named external history using compare-and-swap."""

        canonical_scope = self._require_local_state_scope(scope)
        canonical_expected_count = self._require_local_state_count(
            expected_count, allow_zero=True
        )
        canonical_new_count = self._require_local_state_count(
            new_count, allow_zero=False
        )
        if canonical_expected_count == 0:
            if expected_head_sha256 is not None:
                raise ValueError(
                    "an absent local-state anchor must use count zero and no head"
                )
            canonical_expected_head = None
        else:
            canonical_expected_head = self._require_local_state_head(
                expected_head_sha256, allow_none=False
            )
        canonical_new_head = self._require_local_state_head(
            new_head_sha256, allow_none=False
        )

        def matches_expected(anchor: LocalStateAnchor | None) -> bool:
            if anchor is None:
                return (
                    canonical_expected_count == 0
                    and canonical_expected_head is None
                )
            return (
                anchor.count == canonical_expected_count
                and anchor.head_sha256 == canonical_expected_head
            )

        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            history = self._local_state_anchor_histories(ledger).get(
                canonical_scope, ()
            )
            current = history[-1] if history else None

            if (
                current is not None
                and current.count == canonical_new_count
                and current.head_sha256 == canonical_new_head
            ):
                predecessor = history[-2] if len(history) > 1 else None
                if matches_expected(current) or matches_expected(predecessor):
                    return LocalStateAnchorAppendResult(current, False)
                raise LocalStateAnchorConflictError(
                    "local-state anchor retry does not match its predecessor"
                )

            if not matches_expected(current):
                raise LocalStateAnchorConflictError(
                    "local-state anchor changed since the expected previous value"
                )
            if current is None:
                if canonical_new_count != 1:
                    raise ValueError(
                        "initial local-state anchor count must be exactly one"
                    )
            else:
                if canonical_new_count != current.count + 1:
                    raise ValueError(
                        "local-state anchor count must advance by exactly one"
                    )
                if canonical_new_head == current.head_sha256:
                    raise ValueError(
                        "local-state anchor head must change when count advances"
                    )

            sequence, _audit_state = self._audit_context(
                connection, verify=False
            )
            event_sha256 = self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind=_LOCAL_STATE_ANCHOR_KIND,
                subject_id=self._local_state_anchor_subject(
                    canonical_scope, canonical_new_count, canonical_new_head
                ),
                created_at=self._now_utc_text(),
                record_digests=(),
                head_updates={},
            )
            return LocalStateAnchorAppendResult(
                LocalStateAnchor(
                    scope=canonical_scope,
                    count=canonical_new_count,
                    head_sha256=canonical_new_head,
                    audit_sequence=sequence,
                    audit_event_sha256=event_sha256,
                ),
                True,
            )

    def find_exact_duplicate(self, content_sha256: str) -> str | None:
        if not isinstance(content_sha256, str) or not _SHA256.fullmatch(content_sha256):
            raise ValueError("content_sha256 must be lowercase SHA-256")
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            for capture_id, (state, _document) in ledger["captures"].items():
                if capture_id in ledger["capture_tombstones"]:
                    continue
                if (
                    state["status"] != CAPTURE_QUARANTINED
                    and state["content_sha256"] == content_sha256
                ):
                    return capture_id
            return None

    def find_near_duplicate(
        self,
        simhash_value: int,
        *,
        max_distance: int = 3,
    ) -> DuplicateMatch | None:
        if (
            isinstance(max_distance, bool)
            or not isinstance(max_distance, int)
            or not 0 <= max_distance <= 64
        ):
            raise ValueError("max_distance must be from zero to 64")
        hamming_distance(simhash_value, 0)
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            matches = []
            for capture_id, (state, _document) in ledger["captures"].items():
                if (
                    capture_id in ledger["capture_tombstones"]
                    or state["status"] == CAPTURE_QUARANTINED
                ):
                    continue
                matches.append(
                    DuplicateMatch(
                        capture_id=capture_id,
                        distance=hamming_distance(
                            simhash_value, int(str(state["simhash_hex"]), 16)
                        ),
                    )
                )
        matches = [match for match in matches if match.distance <= max_distance]
        return min(
            matches,
            key=lambda match: (match.distance, match.capture_id),
            default=None,
        )
    def find_near_duplicate_for_text(
        self,
        text: str,
        *,
        max_distance: int = 3,
    ) -> DuplicateMatch | None:
        return self.find_near_duplicate(
            simhash64(text), max_distance=max_distance
        )

    def _validate_pair_record(self, pair: DistilledPair) -> DistilledPair:
        if type(pair) is not DistilledPair:
            raise TypeError("pair must be a DistilledPair")
        if (
            type(pair.id) is not str
            or pair.id != pair.id.strip()
            or not pair.id.isascii()
            or not 1 <= len(pair.id.encode("ascii")) <= 128
            or any(ord(character) < 0x21 or ord(character) >= 0x7F for character in pair.id)
        ):
            raise ValueError("pair id must be a bounded visible ASCII identifier")
        if type(pair.user_approved) is not bool:
            raise TypeError("pair user_approved must be a boolean")
        self._parse_utc_timestamp(pair.created_at, "pair created_at")
        if pair.reviewed_at is not None:
            self._parse_utc_timestamp(pair.reviewed_at, "pair reviewed_at")
        try:
            canonical = DistilledPair.create(
                source_id=pair.source_id,
                instruction=pair.instruction,
                response=pair.response,
                source_provenance_hash=pair.source_provenance_hash,
                tags=pair.tags,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("pair is not canonically valid") from exc
        if (
            canonical.source_id != pair.source_id
            or canonical.instruction != pair.instruction
            or canonical.response != pair.response
            or canonical.tags != pair.tags
            or canonical.source_provenance_hash != pair.source_provenance_hash
        ):
            raise ValueError("pair payload is not canonical")
        return pair

    def add_pair(self, pair: DistilledPair) -> str:
        self._validate_pair_record(pair)
        if pair.user_approved or pair.reviewed_at is not None:
            raise ValueError("new pairs must begin pending operator review")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            self._add_pair_in_connection(connection, ledger, pair)
        return pair.id

    def commit_distilled_pair(self, capture_id: str, pair: DistilledPair) -> str:
        """Revalidate inference inputs and persist both outcomes atomically."""
        self._validate_pair_record(pair)
        if pair.user_approved or pair.reviewed_at is not None:
            raise ValueError("new pairs must begin pending operator review")
        if pair.source_id != capture_id:
            raise ValueError("distilled pair source does not match the requested capture")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            state, document = self._visible_capture_from_ledger(ledger, capture_id)
            if state["status"] != CAPTURE_READY:
                raise ValueError("distillation source is no longer ready")
            if pair.source_provenance_hash != document.source_provenance_hash:
                raise ValueError("pair provenance does not match its source capture")
            self._add_pair_in_connection(connection, ledger, pair)
            self._set_capture_status_in_connection(
                connection, ledger, capture_id, CAPTURE_DISTILLED,
            )
        return pair.id

    def _add_pair_in_connection(
        self, connection: sqlite3.Connection, ledger: dict[str, object],
        pair: DistilledPair,
    ) -> None:
        if pair.source_id in ledger["capture_tombstones"]:
            raise RecordNotFoundError(f"capture not found: {pair.source_id}")
        source = connection.execute(
            "SELECT * FROM captures WHERE id = ?", (pair.source_id,),
        ).fetchone()
        if source is None:
            raise RecordNotFoundError(f"capture not found: {pair.source_id}")
        source_state, _source_document = self._authenticate_capture_row(source)
        if source_state["source_provenance_hash"] != pair.source_provenance_hash:
            raise ValueError("pair provenance does not match its source capture")
        connection.execute(
            """
            INSERT INTO pairs (
                id, source_id, source_provenance_hash, instruction_enc,
                response_enc, tags_enc, state_enc, user_approved, status,
                created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)
            """,
            (
                pair.id,
                pair.source_id,
                pair.source_provenance_hash,
                self._encrypt_text("pairs", pair.id, "instruction", pair.instruction),
                self._encrypt_text("pairs", pair.id, "response", pair.response),
                self._encrypt_json("pairs", pair.id, "tags", pair.tags),
                self._encrypt_pair_state(
                    pair.id,
                    source_id=pair.source_id,
                    source_provenance_hash=pair.source_provenance_hash,
                    user_approved=False,
                    status=PAIR_PENDING,
                    created_at=pair.created_at,
                    reviewed_at=None,
                ),
                PAIR_PENDING,
                pair.created_at,
            ),
        )
        sequence, anchor = self._audit_context(connection, verify=False)
        revision_id, revision_digest = self._insert_revision_record(
            connection,
            pair=pair,
            revision=1,
            parent_revision_id=None,
            sequence=sequence,
            created_at=pair.created_at,
            previous_revision_sha256=anchor["revision_head_sha256"],
        )
        self._publish_audit_operation(
            connection,
            sequence=sequence,
            kind="PAIR_CREATED",
            subject_id=pair.id,
            created_at=pair.created_at,
            record_digests=(self._record_token("PAIR_REVISION", revision_id, revision_digest),),
            head_updates={"revision_head_sha256": revision_digest},
        )

    def _pair_from_row(
        self,
        row: sqlite3.Row,
        authenticated_state: dict[str, object] | None = None,
    ) -> DistilledPair:
        row_id = row["id"]
        state = authenticated_state or self._authenticated_pair_state(row)
        tags = self._decrypt_json("pairs", row_id, "tags", row["tags_enc"])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise VaultAuthenticationError("encrypted pair tags are not a string list")
        return DistilledPair(
            id=row_id,
            source_id=str(state["source_id"]),
            instruction=self._decrypt_text(
                "pairs", row_id, "instruction", row["instruction_enc"]
            ),
            response=self._decrypt_text(
                "pairs", row_id, "response", row["response_enc"]
            ),
            tags=tuple(tags),
            source_provenance_hash=str(state["source_provenance_hash"]),
            user_approved=bool(state["user_approved"]),
            created_at=str(state["created_at"]),
            reviewed_at=None
            if state["reviewed_at"] is None
            else str(state["reviewed_at"]),
        )

    def _visible_pair_from_ledger(
        self, ledger: dict[str, object], pair_id: str
    ) -> tuple[sqlite3.Row, dict[str, object], DistilledPair]:
        current_pairs = ledger["current_pairs"]
        record = current_pairs.get(pair_id)
        if (
            record is None
            or pair_id in ledger["pair_tombstones"]
            or record[2].source_id in ledger["capture_tombstones"]
        ):
            raise RecordNotFoundError(f"pair not found: {pair_id}")
        return record

    def get_pair(self, pair_id: str) -> DistilledPair:
        with self._mutex:
            connection = self._require_connection()
            ledger = self._verify_v4_ledger(connection)
            return self._visible_pair_from_ledger(ledger, pair_id)[2]

    def list_pairs(
        self,
        *,
        status: str | None = None,
        approved_only: bool = False,
        limit: int | None = None,
    ) -> list[DistilledPair]:
        limit = self._validate_limit(limit)
        if status is not None and status not in _PAIR_STATUSES:
            raise ValueError("unknown pair status")
        if approved_only and status not in (None, PAIR_APPROVED):
            raise ValueError("approved_only conflicts with the requested status")
        selected_status = PAIR_APPROVED if approved_only else status
        selected = [
            row.pair for row in self.list_pair_review_rows(status=selected_status)
        ]
        return selected if limit is None else selected[:limit]

    def list_pair_review_rows(
        self, *, status: str | None = None
    ) -> tuple[PairReviewRow, ...]:
        if status is not None and status not in _PAIR_STATUSES:
            raise ValueError("unknown pair status")
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            selected = [
                PairReviewRow(pair, str(state["status"]))
                for pair_id, (_row, state, pair) in ledger["current_pairs"].items()
                if pair_id not in ledger["pair_tombstones"]
                and pair.source_id not in ledger["capture_tombstones"]
                and (status is None or state["status"] == status)
            ]
            return tuple(
                sorted(selected, key=lambda row: (row.pair.created_at, row.pair.id))
            )

    def get_pair_status(self, pair_id: str) -> str:
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            _row, state, _pair = self._visible_pair_from_ledger(ledger, pair_id)
            return str(state["status"])

    def _update_current_pair(
        self,
        connection: sqlite3.Connection,
        *,
        pair: DistilledPair,
        status: str,
    ) -> None:
        if status not in _PAIR_STATUSES:
            raise ValueError("unknown pair status")
        approved = status == PAIR_APPROVED
        cursor = connection.execute(
            """
            UPDATE pairs
            SET instruction_enc = ?, response_enc = ?, tags_enc = ?,
                state_enc = ?, user_approved = ?, status = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (
                self._encrypt_text(
                    "pairs", pair.id, "instruction", pair.instruction
                ),
                self._encrypt_text("pairs", pair.id, "response", pair.response),
                self._encrypt_json("pairs", pair.id, "tags", pair.tags),
                self._encrypt_pair_state(
                    pair.id,
                    source_id=pair.source_id,
                    source_provenance_hash=pair.source_provenance_hash,
                    user_approved=approved,
                    status=status,
                    created_at=pair.created_at,
                    reviewed_at=pair.reviewed_at,
                ),
                int(approved),
                status,
                pair.reviewed_at,
                pair.id,
            ),
        )
        if cursor.rowcount != 1:
            raise RecordNotFoundError(f"pair not found: {pair.id}")

    def _review_current_pair(
        self,
        pair_id: str,
        *,
        kind: str,
        reason_code: str,
        plausible_rejection: bool = False,
    ) -> DistilledPair:
        expected_reason = {
            "APPROVED": OPERATOR_APPROVED,
            "DENIED": OPERATOR_REJECTED,
        }.get(kind)
        if expected_reason is None:
            raise ValueError("unknown review action")
        if type(plausible_rejection) is not bool:
            raise TypeError("plausible_rejection must be a boolean")
        if plausible_rejection and kind != "DENIED":
            raise ValueError("plausible_rejection applies only to a denial")
        require_review_reason(reason_code, expected_reason)
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            _row, _state, pair = self._visible_pair_from_ledger(ledger, pair_id)
            reviewed = pair.review(
                instruction=pair.instruction,
                response=pair.response,
                tags=pair.tags,
                approve=kind == "APPROVED",
            )
            if reviewed.reviewed_at is None:
                raise VaultError("review timestamp was not produced")
            sequence, anchor = self._audit_context(connection, verify=False)
            revisions = [
                state
                for state, _payload, _digest in ledger["revision_records"]
                if state["pair_id"] == pair_id
            ]
            latest_revision = max(
                revisions, key=lambda state: int(state["revision"])
            )
            event_id, review_digest = self._insert_review_record(
                connection,
                pair_id=pair_id,
                revision_id=str(latest_revision["revision_id"]),
                sequence=sequence,
                kind=kind,
                reason_code=reason_code,
                created_at=reviewed.reviewed_at,
                previous_review_sha256=anchor["review_head_sha256"],
            )
            audit_kind = "PAIR_APPROVED"
            if kind == "DENIED":
                denial_reason = (
                    "other" if reason_code == OPERATOR_REJECTED else reason_code
                )
                marker = "PLAUSIBLE_" if plausible_rejection else ""
                audit_kind = f"PAIR_DENIED_{marker}{denial_reason.upper()}"
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind=audit_kind,
                subject_id=pair_id,
                created_at=reviewed.reviewed_at,
                record_digests=(
                    self._record_token("REVIEW_EVENT", event_id, review_digest),
                ),
                head_updates={"review_head_sha256": review_digest},
            )
            self._update_current_pair(
                connection,
                pair=reviewed,
                status=PAIR_APPROVED if kind == "APPROVED" else PAIR_REJECTED,
            )
            return reviewed

    def approve_pair(
        self,
        pair_id: str,
        *,
        reason_code: str = OPERATOR_APPROVED,
    ) -> DistilledPair:
        return self._review_current_pair(
            pair_id, kind="APPROVED", reason_code=reason_code
        )

    def edit_pair(
        self,
        pair_id: str,
        instruction: str,
        response: str,
        tags: tuple[str, ...],
        *,
        reason_code: str = OPERATOR_MODIFIED,
    ) -> DistilledPair:
        require_review_reason(reason_code, OPERATOR_MODIFIED)
        if type(tags) is not tuple:
            raise TypeError("tags must be a tuple")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            _row, _state, pair = self._visible_pair_from_ledger(ledger, pair_id)
            if (
                instruction == pair.instruction
                and response == pair.response
                and tags == pair.tags
            ):
                raise ValueError("review edit does not change the pair")
            reviewed = pair.review(
                instruction=instruction,
                response=response,
                tags=tags,
                approve=False,
            )
            if reviewed.reviewed_at is None:
                raise VaultError("review timestamp was not produced")
            history = [
                (state, payload)
                for state, payload, _digest in ledger["revision_records"]
                if state["pair_id"] == pair_id
            ]
            latest, _latest_payload = max(
                history, key=lambda record: int(record[0]["revision"])
            )
            _original_state, original_payload = min(
                history, key=lambda record: int(record[0]["revision"])
            )
            response_differs = reviewed.response != original_payload["response"]
            sequence, anchor = self._audit_context(connection, verify=False)
            revision_id, revision_digest = self._insert_revision_record(
                connection,
                pair=reviewed,
                revision=int(latest["revision"]) + 1,
                parent_revision_id=str(latest["revision_id"]),
                sequence=sequence,
                created_at=reviewed.reviewed_at,
                previous_revision_sha256=anchor["revision_head_sha256"],
            )
            event_id, review_digest = self._insert_review_record(
                connection,
                pair_id=pair_id,
                revision_id=revision_id,
                sequence=sequence,
                kind="MODIFIED",
                reason_code=reason_code,
                created_at=reviewed.reviewed_at,
                previous_review_sha256=anchor["review_head_sha256"],
            )
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind=(
                    "PAIR_RESPONSE_DIFFERS"
                    if response_differs
                    else "PAIR_RESPONSE_MATCHES"
                ),
                subject_id=pair_id,
                created_at=reviewed.reviewed_at,
                record_digests=(
                    self._record_token(
                        "PAIR_REVISION", revision_id, revision_digest
                    ),
                    self._record_token("REVIEW_EVENT", event_id, review_digest),
                ),
                head_updates={
                    "revision_head_sha256": revision_digest,
                    "review_head_sha256": review_digest,
                },
            )
            self._update_current_pair(
                connection, pair=reviewed, status=PAIR_PENDING
            )
            return reviewed

    def reject_pair(
        self,
        pair_id: str,
        *,
        reason_code: str = "other",
        plausible_rejection: bool = False,
    ) -> DistilledPair:
        return self._review_current_pair(
            pair_id,
            kind="DENIED",
            reason_code=reason_code,
            plausible_rejection=plausible_rejection,
        )

    def record_pair_displayed(self, pair_id: str) -> str:
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            self._visible_pair_from_ledger(ledger, pair_id)
            sequence, _anchor = self._audit_context(connection, verify=False)
            displayed_at = self._now_utc_text()
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind="PAIR_DISPLAYED",
                subject_id=pair_id,
                created_at=displayed_at,
                record_digests=(),
                head_updates={},
            )
            return displayed_at

    def review_session_metrics(
        self, window_start: datetime, window_end: datetime
    ):
        from .review_metrics import derive_review_metrics

        with self._mutex:
            connection = self._require_connection()
            connection.execute("BEGIN")
            try:
                self._verify_v4_ledger(connection)
                metrics = derive_review_metrics(
                    connection, window_start, window_end
                )
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
                return metrics

    def dpo_readiness_telemetry(self):
        from .corpus_observability import derive_dpo_telemetry

        with self._mutex:
            connection = self._require_connection()
            connection.execute("BEGIN")
            try:
                ledger = self._verify_v4_ledger(connection)
                snapshot = self._training_snapshot_from_ledger(ledger)
                plausible = frozenset(
                    int(state["sequence"])
                    for state, _digest in ledger["audit_records"]
                    if str(state["kind"]).startswith("PAIR_DENIED_PLAUSIBLE_")
                )
                result = derive_dpo_telemetry(snapshot, plausible)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
                return result

    def _authenticated_selected_rag_sources(
        self, ledger: dict[str, object]
    ):
        from urllib.parse import urlsplit

        from .corpus_observability import RagCorpusSource
        from .rag import chunk_text

        selected = ledger["selected_rag_index"]
        if selected is None:
            return ()
        run_record = ledger["training_identities"].get(selected.version_id)
        if run_record is None:
            raise VaultAuthenticationError(
                "selected RAG index lacks an authenticated run identity"
            )
        identity = run_record[0]
        if (
            identity.mode is not TrainingMode.RAG
            or identity.projection_schema != "LIVED_RAG_V1"
            or identity.dataset_sha256 != selected.dataset_sha256
            or identity.dataset_bytes != selected.dataset_bytes
            or identity.dataset_count != selected.document_count
            or identity.config_sha256 != selected.config_sha256
            or identity.model_identity_sha256
            != selected.embedding_model_sha256
        ):
            raise VaultAuthenticationError(
                "selected RAG index differs from its authenticated run identity"
            )

        snapshot_sequence = identity.snapshot_sequence
        hidden_pairs = {
            str(state["record_id"])
            for state, _digest in ledger["tombstone_records"]
            if int(state["sequence"]) <= snapshot_sequence
            and state["record_kind"] == "PAIR"
        }
        hidden_captures = {
            str(state["record_id"])
            for state, _digest in ledger["tombstone_records"]
            if int(state["sequence"]) <= snapshot_sequence
            and state["record_kind"] == "CAPTURE"
        }
        latest_revisions: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
        for state, payload, _digest in ledger["revision_records"]:
            if int(state["sequence"]) > snapshot_sequence:
                continue
            pair_id = str(state["pair_id"])
            current = latest_revisions.get(pair_id)
            if current is None or int(state["revision"]) > int(current[0]["revision"]):
                latest_revisions[pair_id] = (state, payload)
        latest_decisions: dict[str, dict[str, object]] = {}
        for state, _digest in ledger["review_records"]:
            if int(state["sequence"]) > snapshot_sequence:
                continue
            revision_id = str(state["revision_id"])
            current = latest_decisions.get(revision_id)
            if current is None or int(state["sequence"]) > int(current["sequence"]):
                latest_decisions[revision_id] = state

        capture_ids = {
            str(state["source_id"])
            for pair_id, (state, _payload) in latest_revisions.items()
            if pair_id not in hidden_pairs
            and str(state["source_id"]) not in hidden_captures
            and (
                latest_decisions.get(str(state["revision_id"])) is not None
                and latest_decisions[str(state["revision_id"])]["kind"]
                == "APPROVED"
            )
        }
        try:
            documents = tuple(
                ledger["captures"][capture_id][1]
                for capture_id in capture_ids
            )
        except KeyError as error:
            raise VaultAuthenticationError(
                "selected RAG snapshot references a missing capture"
            ) from error
        ordered = tuple(
            sorted(documents, key=lambda item: (item.content_sha256, item.id))
        )
        if len(ordered) != selected.document_count:
            raise VaultAuthenticationError(
                "selected RAG document count differs from its authenticated snapshot"
            )

        dataset_digest = hashlib.sha256()
        corpus_digest = hashlib.sha256()
        dataset_prefix = b'{"documents":['
        dataset_suffix = b'],"schema":"LIVED_RAG_DATASET_V1"}\n'
        dataset_digest.update(dataset_prefix)
        dataset_bytes = len(dataset_prefix) + len(dataset_suffix)
        corpus_digest.update(b"[")
        chunk_count = 0
        for document_index, document in enumerate(ordered):
            raw_bytes = document.raw_text.encode("utf-8")
            if not 1 <= len(raw_bytes) <= 4_000_000:
                raise VaultAuthenticationError(
                    "selected RAG document size is invalid"
                )
            opaque = hashlib.sha256(
                b"liveD:rag-document:v1\0"
                + document.content_sha256.encode("ascii")
            ).hexdigest()[:32]
            chunk_records = []
            for ordinal, text in enumerate(chunk_text(document.raw_text)):
                record = {
                    "ordinal": ordinal,
                    "text": text,
                    "text_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                }
                chunk_records.append(record)
                corpus_record = self._canonical_ledger_bytes({
                    "opaque_document_id": opaque,
                    **record,
                })[:-1]
                if chunk_count:
                    corpus_digest.update(b",")
                corpus_digest.update(corpus_record)
                chunk_count += 1
            document_payload = self._canonical_ledger_bytes({
                "chunks": chunk_records,
                "content_sha256": document.content_sha256,
                "opaque_document_id": opaque,
            })[:-1]
            if document_index:
                dataset_digest.update(b",")
                dataset_bytes += 1
            dataset_digest.update(document_payload)
            dataset_bytes += len(document_payload)
            if dataset_bytes > selected.dataset_bytes:
                raise VaultAuthenticationError(
                    "selected RAG dataset exceeds its authenticated byte length"
                )
        dataset_digest.update(dataset_suffix)
        corpus_digest.update(b"]\n")
        if (
            dataset_bytes != selected.dataset_bytes
            or dataset_digest.hexdigest() != selected.dataset_sha256
            or chunk_count != selected.chunk_count
            or corpus_digest.hexdigest() != selected.corpus_sha256
        ):
            raise VaultAuthenticationError(
                "selected RAG corpus cannot be reproduced from its authenticated snapshot"
            )

        sources = []
        for document in ordered:
            host = urlsplit(document.url).hostname
            if host is None:
                raise VaultAuthenticationError(
                    "selected RAG source URL has no domain"
                )
            try:
                domain = host.encode("idna").decode("ascii").casefold()
            except UnicodeError as error:
                raise VaultAuthenticationError(
                    "selected RAG source domain is invalid"
                ) from error
            sources.append(RagCorpusSource(
                domain=domain,
                captured_at=self._parse_utc_timestamp(
                    document.captured_at, "capture captured_at"
                ),
            ))
        return tuple(sources)

    def rag_corpus_drift(self, reference_time: datetime):
        from .corpus_observability import derive_rag_corpus_drift

        with self._mutex:
            connection = self._require_connection()
            connection.execute("BEGIN")
            try:
                ledger = self._verify_v4_ledger(connection)
                sources = self._authenticated_selected_rag_sources(ledger)
                result = derive_rag_corpus_drift(sources, reference_time)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
                return result

    def delete_pair(self, pair_id: str) -> None:
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            self._visible_pair_from_ledger(ledger, pair_id)
            sequence, anchor = self._audit_context(connection, verify=False)
            created_at = self._now_utc_text()
            tombstone_id, tombstone_digest = self._insert_tombstone_record(
                connection,
                record_kind="PAIR",
                record_id=pair_id,
                sequence=sequence,
                created_at=created_at,
                previous_tombstone_sha256=anchor["tombstone_head_sha256"],
            )
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind="PAIR_DELETED",
                subject_id=pair_id,
                created_at=created_at,
                record_digests=(
                    self._record_token(
                        "TOMBSTONE", tombstone_id, tombstone_digest
                    ),
                ),
                head_updates={"tombstone_head_sha256": tombstone_digest},
            )


    def seal_local_blob(
        self, namespace: str, identifier: str, plaintext: bytes
    ) -> bytes:
        if (
            type(namespace) is not str
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", namespace) is None
        ):
            raise ValueError("local blob namespace is invalid")
        if (
            type(identifier) is not str
            or re.fullmatch(r"[a-f0-9]{64}", identifier) is None
        ):
            raise ValueError("local blob identifier must be lowercase SHA-256")
        if type(plaintext) is not bytes or len(plaintext) > 268_435_456:
            raise ValueError("local blob payload is invalid")
        with self._mutex:
            self._verify_v4_ledger(self._require_connection())
            return self._encrypt(
                "local_blob", namespace + ":" + identifier, "payload", plaintext
            )

    def open_local_blob(
        self, namespace: str, identifier: str, payload: bytes
    ) -> bytes:
        if (
            type(namespace) is not str
            or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", namespace) is None
        ):
            raise ValueError("local blob namespace is invalid")
        if (
            type(identifier) is not str
            or re.fullmatch(r"[a-f0-9]{64}", identifier) is None
        ):
            raise ValueError("local blob identifier must be lowercase SHA-256")
        if not isinstance(payload, bytes) or len(payload) > 268_435_512:
            raise ValueError("local blob payload is invalid")
        with self._mutex:
            self._verify_v4_ledger(self._require_connection())
            return self._decrypt(
                "local_blob", namespace + ":" + identifier, "payload", bytes(payload)
            )

    def rag_index_events(self) -> tuple[RagIndexEvent, ...]:
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            return tuple(event for event, _digest in ledger["rag_index_records"])

    def selected_rag_index_version(self) -> RagIndexVersion | None:
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            return ledger["selected_rag_index"]

    def pending_rag_index_event(self) -> RagIndexEvent | None:
        with self._mutex:
            ledger = self._verify_v4_ledger(self._require_connection())
            return ledger["pending_rag_index"]

    def prepare_rag_index_version(
        self, version: RagIndexVersion
    ) -> RagIndexAppendResult:
        if type(version) is not RagIndexVersion:
            raise TypeError("version must be RagIndexVersion")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            for event, _digest in ledger["rag_index_records"]:
                if event.kind == "PREPARED" and event.version.version_id == version.version_id:
                    if event.version != version:
                        raise ValueError("RAG version ID is already bound to different bytes")
                    return RagIndexAppendResult(event, False)
            if ledger["pending_rag_index"] is not None:
                raise ValueError("an unresolved RAG index preparation already exists")
            selected = ledger["selected_rag_index"]
            previous = None if selected is None else selected.version_sha256
            sequence, _anchor = self._audit_context(connection, verify=False)
            created_at = self._now_utc_text()
            event, digest = self._insert_rag_index_event(
                connection,
                sequence=sequence,
                kind="PREPARED",
                version=version,
                prepared_event_id=None,
                previous_version_sha256=previous,
                created_at=created_at,
            )
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind="RAG_INDEX_PREPARED",
                subject_id=version.version_id,
                created_at=created_at,
                record_digests=(
                    self._record_token("RAG_INDEX_EVENT", event.event_id, digest),
                ),
                head_updates={},
            )
            return RagIndexAppendResult(event, True)

    def mark_rag_index_published(
        self, prepared_event_id: str
    ) -> RagIndexAppendResult:
        if type(prepared_event_id) is not str or re.fullmatch(r"[a-f0-9]{32}", prepared_event_id) is None:
            raise ValueError("prepared RAG event ID is invalid")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            records = ledger["rag_index_records"]
            prepared = next(
                (
                    event for event, _digest in records
                    if event.event_id == prepared_event_id and event.kind == "PREPARED"
                ),
                None,
            )
            if prepared is None:
                raise KeyError(prepared_event_id)
            existing = next(
                (
                    event for event, _digest in records
                    if event.prepared_event_id == prepared_event_id
                    and event.kind == "PUBLISHED"
                ),
                None,
            )
            if existing is not None:
                return RagIndexAppendResult(existing, False)
            pending = ledger["pending_rag_index"]
            if pending is None or pending.event_id != prepared_event_id:
                raise ValueError("RAG preparation is no longer active")
            sequence, _anchor = self._audit_context(connection, verify=False)
            created_at = self._now_utc_text()
            event, digest = self._insert_rag_index_event(
                connection,
                sequence=sequence,
                kind="PUBLISHED",
                version=prepared.version,
                prepared_event_id=prepared_event_id,
                previous_version_sha256=prepared.previous_version_sha256,
                created_at=created_at,
            )
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind="RAG_INDEX_PUBLISHED",
                subject_id=prepared.version.version_id,
                created_at=created_at,
                record_digests=(
                    self._record_token("RAG_INDEX_EVENT", event.event_id, digest),
                ),
                head_updates={},
            )
            return RagIndexAppendResult(event, True)

    def select_rag_index_version(
        self, prepared_event_id: str
    ) -> RagIndexAppendResult:
        if type(prepared_event_id) is not str or re.fullmatch(r"[a-f0-9]{32}", prepared_event_id) is None:
            raise ValueError("prepared RAG event ID is invalid")
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            records = ledger["rag_index_records"]
            prepared = next(
                (
                    event for event, _digest in records
                    if event.event_id == prepared_event_id and event.kind == "PREPARED"
                ),
                None,
            )
            if prepared is None:
                raise KeyError(prepared_event_id)
            existing = next(
                (
                    event for event, _digest in records
                    if event.prepared_event_id == prepared_event_id
                    and event.kind == "SELECTED"
                ),
                None,
            )
            if existing is not None:
                return RagIndexAppendResult(existing, False)
            published = any(
                event.prepared_event_id == prepared_event_id
                and event.kind == "PUBLISHED"
                for event, _digest in records
            )
            if not published:
                raise ValueError("RAG index cannot be selected before publication")
            pending = ledger["pending_rag_index"]
            if pending is None or pending.event_id != prepared_event_id:
                raise ValueError("RAG preparation is no longer active")
            sequence, _anchor = self._audit_context(connection, verify=False)
            created_at = self._now_utc_text()
            event, digest = self._insert_rag_index_event(
                connection,
                sequence=sequence,
                kind="SELECTED",
                version=prepared.version,
                prepared_event_id=prepared_event_id,
                previous_version_sha256=prepared.previous_version_sha256,
                created_at=created_at,
            )
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind="RAG_INDEX_SELECTED",
                subject_id=prepared.version.version_id,
                created_at=created_at,
                record_digests=(
                    self._record_token("RAG_INDEX_EVENT", event.event_id, digest),
                ),
                head_updates={},
            )
            return RagIndexAppendResult(event, True)

    def register_source_target(
        self,
        *,
        target_id: str,
        blueprint_id: str,
        label: str,
        start_url: str,
        categories: tuple[str, ...],
        expected_yield: int,
        operator_authorized: bool,
    ) -> SourceTargetView:
        view = SourceTargetView(
            target_id=target_id,
            blueprint_id=blueprint_id,
            label=label,
            start_url=start_url,
            categories=categories,
            expected_yield=expected_yield,
            operator_authorized=operator_authorized,
            last_accepted_at=None,
        )
        with self._mutex, self._transaction() as connection:
            ledger = self._verify_v4_ledger(connection)
            if any(
                target.target_id == target_id
                for _state, target, _digest in ledger["target_records"]
            ):
                raise ValueError(f"source target already exists: {target_id}")
            sequence, anchor = self._audit_context(connection, verify=False)
            created_at = self._now_utc_text()
            target_digest = self._insert_source_target_record(
                connection,
                view=view,
                sequence=sequence,
                created_at=created_at,
                previous_target_sha256=anchor["target_head_sha256"],
            )
            self._publish_audit_operation(
                connection,
                sequence=sequence,
                kind="SOURCE_TARGET_REGISTERED",
                subject_id=target_id,
                created_at=created_at,
                record_digests=(
                    self._record_token("SOURCE_TARGET", target_id, target_digest),
                ),
                head_updates={"target_head_sha256": target_digest},
            )
        return view

    def _training_snapshot_from_ledger(
        self, ledger: dict[str, object]
    ) -> ReadinessSnapshot:
        hidden_pairs = set(ledger["pair_tombstones"])
        hidden_captures = set(ledger["capture_tombstones"])
        visible_pair_ids = {
            pair_id
            for pair_id, (_row, _state, pair) in ledger["current_pairs"].items()
            if pair_id not in hidden_pairs and pair.source_id not in hidden_captures
        }
        latest_decision: dict[str, dict[str, object]] = {}
        decision_views = []
        for state, _digest in ledger["review_records"]:
            pair_id = str(state["pair_id"])
            if pair_id not in visible_pair_ids:
                continue
            revision_id = str(state["revision_id"])
            current = latest_decision.get(revision_id)
            if current is None or int(state["sequence"]) > int(current["sequence"]):
                latest_decision[revision_id] = state
            decision_views.append(
                DecisionView(
                    event_id=str(state["event_id"]),
                    pair_id=pair_id,
                    revision_id=revision_id,
                    kind=str(state["kind"]),
                    reason_code=str(state["reason_code"]),
                    sequence=int(state["sequence"]),
                    created_at=self._parse_utc_timestamp(
                        state["created_at"], "review event created_at"
                    ),
                )
            )
        decision_views.sort(key=lambda item: (item.sequence, item.event_id))

        pair_views = []
        for state, payload, _digest in ledger["revision_records"]:
            pair_id = str(state["pair_id"])
            if pair_id not in visible_pair_ids:
                continue
            source_id = str(state["source_id"])
            capture_state = ledger["captures"][source_id][0]
            decision = latest_decision.get(str(state["revision_id"]))
            pair_views.append(
                PairRevisionView(
                    revision_id=str(state["revision_id"]),
                    pair_id=pair_id,
                    lineage_id=str(state["lineage_id"]),
                    parent_revision_id=(
                        None
                        if state["parent_revision_id"] is None
                        else str(state["parent_revision_id"])
                    ),
                    capture_id=source_id,
                    revision=int(state["revision"]),
                    instruction=str(payload["instruction"]),
                    response=str(payload["response"]),
                    tags=tuple(payload["tags"]),
                    approved=(
                        decision is not None and decision["kind"] == "APPROVED"
                    ),
                    quarantined=(
                        capture_state["status"] == CAPTURE_QUARANTINED
                    ),
                    created_at=self._parse_utc_timestamp(
                        state["created_at"], "pair revision created_at"
                    ),
                    event_sequence=int(state["sequence"]),
                )
            )
        pair_views.sort(key=lambda item: (item.event_sequence, item.revision_id))

        latest_pairs: dict[str, PairRevisionView] = {}
        for pair in pair_views:
            current = latest_pairs.get(pair.pair_id)
            if current is None or pair.revision > current.revision:
                latest_pairs[pair.pair_id] = pair
        capture_categories: dict[str, list[str]] = {}
        for pair in latest_pairs.values():
            if not pair.approved or pair.quarantined:
                continue
            bucket = capture_categories.setdefault(pair.capture_id, [])
            for tag in pair.tags:
                if tag not in bucket:
                    bucket.append(tag)

        capture_views = []
        for capture_id, (state, document) in ledger["captures"].items():
            if capture_id in hidden_captures:
                continue
            capture_views.append(
                CaptureTrainingView(
                    capture_id=capture_id,
                    content_sha256=str(state["content_sha256"]),
                    raw_text=document.raw_text,
                    quality=float(state["quality_score"]),
                    captured_at=self._parse_utc_timestamp(
                        state["captured_at"], "capture captured_at"
                    ),
                    authenticated=True,
                    quarantined=(state["status"] == CAPTURE_QUARANTINED),
                    categories=tuple(capture_categories.get(capture_id, ())),
                )
            )
        capture_views.sort(key=lambda item: (item.captured_at, item.capture_id))

        targets = tuple(
            view
            for _state, view, _digest in sorted(
                ledger["target_records"], key=lambda item: int(item[0]["sequence"])
            )
        )
        eligible_ids = []
        captures_by_id = {item.capture_id: item for item in capture_views}
        for pair in pair_views:
            latest = latest_pairs[pair.pair_id]
            capture = captures_by_id.get(pair.capture_id)
            if (
                latest.revision_id == pair.revision_id
                and pair.approved
                and not pair.quarantined
                and capture is not None
                and capture.authenticated
                and not capture.quarantined
            ):
                eligible_ids.append(pair.revision_id)
        eligible_identity = hashlib.sha256(
            (
                json.dumps(
                    eligible_ids,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        return ReadinessSnapshot(
            sequence=int(ledger["anchor"]["sequence"]),
            eligible_identity_sha256=eligible_identity,
            pairs=tuple(pair_views),
            decisions=tuple(decision_views),
            captures=tuple(capture_views),
            targets=targets,
        )

    @staticmethod
    def _require_access_record_id(record_id: object) -> str:
        if type(record_id) is not str:
            raise TypeError("record_id must be a string")
        try:
            encoded = record_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("record_id must be valid UTF-8") from exc
        if (
            len(encoded) > 128
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", record_id)
            is None
        ):
            raise ValueError("record_id must be a bounded opaque identifier")
        return record_id

    def _save_access_record(self, record: object, kind: str):
        if kind == _ACCESS_SOURCE:
            expected_type = SavedSource
            if type(record) is not expected_type:
                raise TypeError("source must be an exact SavedSource")
            state = record.to_state()
        elif kind == _ACCESS_LOGIN:
            expected_type = SavedLogin
            if type(record) is not expected_type:
                raise TypeError("login must be an exact SavedLogin")
            state = record.secret_state()
        elif kind == _ACCESS_PROXY:
            expected_type = ProxySettings
            if type(record) is not expected_type:
                raise TypeError("proxy settings must be exact ProxySettings")
            state = record.secret_state()
        elif kind == _ACCESS_CAPMONSTER:
            expected_type = CapMonsterCredential
            if type(record) is not expected_type:
                raise TypeError("CapMonster credential must be exact CapMonsterCredential")
            state = record.secret_state()
        else:
            raise ValueError("saved-access kind is invalid")
        record_id = self._require_access_record_id(record.record_id)
        with self._mutex, self._transaction() as connection:
            state_enc = self._encrypt_json(
                _ACCESS_TABLE, record_id, f"{kind}:state", state
            )
            existing = connection.execute(
                "SELECT 1 FROM scraper_access_records WHERE record_id=?",
                (record_id,),
            ).fetchone()
            if existing is not None:
                raise ValueError(f"saved-access record already exists: {record_id}")
            connection.execute(
                """
                INSERT INTO scraper_access_records (record_id, kind, state_enc)
                VALUES (?, ?, ?)
                """,
                (record_id, kind, state_enc),
            )
        return record

    def _list_access_records(self, kind: str, expected_type: type) -> list:
        with self._mutex:
            connection = self._require_connection()
            rows = connection.execute(
                """
                SELECT record_id, kind, state_enc
                FROM scraper_access_records
                WHERE kind=?
                ORDER BY record_id
                """,
                (kind,),
            ).fetchall()
            return self._authenticate_access_rows(rows, kind, expected_type)

    def _authenticate_access_rows(
        self, rows: Sequence[sqlite3.Row], kind: str, expected_type: type
    ) -> list:
        records = []
        for row in rows:
            record_id = row["record_id"]
            stored_kind = row["kind"]
            if type(record_id) is not str or stored_kind != kind:
                raise VaultAuthenticationError("saved-access indexed state is invalid")
            try:
                state = self._decrypt_json(
                    _ACCESS_TABLE,
                    record_id,
                    f"{kind}:state",
                    row["state_enc"],
                )
                if expected_type is SavedSource:
                    record = SavedSource.from_state(state)
                elif expected_type is SavedLogin:
                    record = SavedLogin.from_state(state)
                elif expected_type is ProxySettings:
                    record = ProxySettings.from_state(state)
                elif expected_type is CapMonsterCredential:
                    record = CapMonsterCredential.from_state(state)
                else:
                    raise TypeError("saved-access model type is invalid")
            except VaultAuthenticationError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise VaultAuthenticationError(
                    "saved-access encrypted state is invalid"
                ) from exc
            if type(record) is not expected_type or record.record_id != record_id:
                raise VaultAuthenticationError(
                    "saved-access record ID does not match encrypted state"
                )
            records.append(record)
        return records

    def _delete_access_record(self, record_id: object, kind: str) -> None:
        canonical_id = self._require_access_record_id(record_id)
        with self._mutex, self._transaction() as connection:
            cursor = connection.execute(
                "DELETE FROM scraper_access_records WHERE record_id=? AND kind=?",
                (canonical_id, kind),
            )
            if cursor.rowcount != 1:
                raise RecordNotFoundError(canonical_id)

    def save_saved_source(self, source: SavedSource) -> SavedSource:
        return self._save_access_record(source, _ACCESS_SOURCE)

    def list_saved_sources(self) -> list[SavedSource]:
        return self._list_access_records(_ACCESS_SOURCE, SavedSource)

    def delete_saved_source(self, record_id: str) -> None:
        self._delete_access_record(record_id, _ACCESS_SOURCE)

    def save_saved_login(self, login: SavedLogin) -> SavedLogin:
        return self._save_access_record(login, _ACCESS_LOGIN)

    def list_saved_logins(self) -> list[SavedLogin]:
        return self._list_access_records(_ACCESS_LOGIN, SavedLogin)

    def delete_saved_login(self, record_id: str) -> None:
        self._delete_access_record(record_id, _ACCESS_LOGIN)

    def save_proxy_settings(self, settings: ProxySettings) -> ProxySettings:
        return self._save_access_record(settings, _ACCESS_PROXY)

    def list_proxy_settings(self) -> list[ProxySettings]:
        return self._list_access_records(_ACCESS_PROXY, ProxySettings)

    def delete_proxy_settings(self, record_id: str) -> None:
        self._delete_access_record(record_id, _ACCESS_PROXY)

    def save_capmonster_credential(
        self, credential: CapMonsterCredential
    ) -> CapMonsterCredential:
        return self._save_access_record(credential, _ACCESS_CAPMONSTER)

    def replace_capmonster_credential(
        self, credential: CapMonsterCredential
    ) -> CapMonsterCredential:
        if type(credential) is not CapMonsterCredential:
            raise TypeError("CapMonster credential must be exact CapMonsterCredential")
        record_id = self._require_access_record_id(credential.record_id)
        state = credential.secret_state()
        with self._mutex, self._transaction() as connection:
            state_enc = self._encrypt_json(
                _ACCESS_TABLE, record_id, f"{_ACCESS_CAPMONSTER}:state", state
            )
            rows = connection.execute(
                """
                SELECT record_id, kind, state_enc
                FROM scraper_access_records
                WHERE kind=?
                ORDER BY record_id
                """,
                (_ACCESS_CAPMONSTER,),
            ).fetchall()
            self._authenticate_access_rows(
                rows, _ACCESS_CAPMONSTER, CapMonsterCredential
            )
            connection.execute(
                "DELETE FROM scraper_access_records WHERE kind=?",
                (_ACCESS_CAPMONSTER,),
            )
            connection.execute(
                """
                INSERT INTO scraper_access_records (record_id, kind, state_enc)
                VALUES (?, ?, ?)
                """,
                (record_id, _ACCESS_CAPMONSTER, state_enc),
            )
        return credential

    def list_capmonster_credentials(self) -> list[CapMonsterCredential]:
        return self._list_access_records(
            _ACCESS_CAPMONSTER, CapMonsterCredential
        )

    def delete_capmonster_credential(self, record_id: str) -> None:
        self._delete_access_record(record_id, _ACCESS_CAPMONSTER)

    def training_snapshot(self) -> ReadinessSnapshot:
        with self._mutex:
            connection = self._require_connection()
            connection.execute("BEGIN")
            try:
                ledger = self._verify_v4_ledger(connection)
                snapshot = self._training_snapshot_from_ledger(ledger)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
                return snapshot


class RepositoryRightsTamperError(VaultAuthenticationError):
    """Repository rights history does not match its authenticated vault head."""


class RepositoryRightsConflictError(VaultError):
    """A registration conflicts with an existing exact repository binding."""


_RIGHTS_SCOPE = "source:rights"
_RIGHTS_NAMESPACE = "source-rights"
_RIGHTS_MAX_BYTES = 32 * 1024
_RIGHTS_FINAL = re.compile(r"e([0-9]{20})-([a-f0-9]{64})[.]sealed")
_RIGHTS_PENDING = re.compile(r"[.]pending-e([0-9]{20})-([a-f0-9]{64})")
_RIGHTS_STAGING = re.compile(r"[.]staging-e([0-9]{20})-([a-f0-9]{64})")


def _rights_json_object(pairs):
    state = {}
    for name, value in pairs:
        if name in state:
            raise ValueError("duplicate rights record field")
        state[name] = value
    return state


class RepositoryRightsLedger:
    """Sealed append-only local rights, committed only by the vault anchor.

    The open vault owns the exclusive process lock. Its shared reentrant mutex
    serializes all ledger instances, reads, recovery, and compare-and-swap
    publication. The SQLite schema and training/source projections are untouched.
    """

    def __init__(self, vault: StudioVault):
        if not isinstance(vault, StudioVault):
            raise TypeError("repository rights require an open StudioVault")
        self.vault = vault
        self.root = _absolute_lexical(vault.paths.source_rights)
        with vault._mutex:
            vault._require_connection()
            _require_private_directory(self.root)
            self._root_chain = _snapshot_chain(self.root, require_target=True)
            with self._locked() as guard:
                self._history(guard)

    def _check_root(self) -> None:
        _require_private_directory(self.root)
        if _snapshot_chain(self.root, require_target=True) != self._root_chain:
            raise RepositoryRightsTamperError("repository rights directory changed")

    @contextmanager
    def _locked(self):
        with self.vault._mutex:
            self.vault._require_connection()
            try:
                self._check_root()
                # Retained Windows ancestor handles deny directory substitution;
                # POSIX ancestry checks complement the vault's process lock.
                guard = _FixedFileSet(self.root, ())
            except (OSError, UnsafePathError) as error:
                raise RepositoryRightsTamperError("unsafe repository rights directory") from error
            try:
                yield guard
                guard.validate()
                self._check_root()
            finally:
                guard.close()

    def _read_record(self, path: Path, digest: str, predecessor: str | None):
        self._check_root()
        before = _require_private_file(path)
        relative = path.relative_to(_absolute_lexical(self.vault.paths.root))
        sealed = self.vault.paths.secure_read_bytes(relative, maximum=_RIGHTS_MAX_BYTES)
        clear = self.vault.open_local_blob(_RIGHTS_NAMESPACE, digest, sealed)
        state = json.loads(clear.decode("ascii"), object_pairs_hook=_rights_json_object)
        if _canonical_json(state) != clear:
            raise ValueError("repository rights payload is not canonical")
        record = RepositoryRightsRecord.from_state(state)
        if not hmac.compare_digest(record.policy_sha256, GitHubCollectionPolicy().policy_sha256):
            raise RepositoryRightsTamperError("persisted repository rights policy mismatch")
        if record.record_sha256 != digest or record.predecessor_sha256 != predecessor:
            raise ValueError("repository rights chain mismatch")
        after = _require_private_file(path)
        if _file_identity(before) != _file_identity(after):
            raise RepositoryRightsTamperError("repository rights file changed while reading")
        return record, _file_identity(after)

    @staticmethod
    def _related(first: RepositoryIdentity, second: RepositoryIdentity) -> bool:
        return (
            first.repository_id == second.repository_id
            or first.node_id == second.node_id
            or (first.locator.owner.casefold(), first.locator.repository.casefold())
            == (second.locator.owner.casefold(), second.locator.repository.casefold())
        )

    @classmethod
    def _matching_identity(cls, records, identity: RepositoryIdentity):
        related = tuple(record for record in records if cls._related(record.repository, identity))
        if any(record.repository != identity for record in related):
            raise RepositoryRightsConflictError("repository identity changed")
        return related

    @classmethod
    def _validate_history_record(cls, records, record: RepositoryRightsRecord) -> None:
        try:
            related = cls._matching_identity(records, record.repository)
        except RepositoryRightsConflictError as error:
            raise RepositoryRightsTamperError("repository rights identity history conflicts") from error
        if any(previous.record_id == record.record_id for previous in records):
            raise RepositoryRightsTamperError("repository rights record ID is repeated")
        for previous in related:
            if previous.commit_sha != record.commit_sha:
                continue
            raise RepositoryRightsTamperError("repository rights history contains a conflicting duplicate")

    def _remove_owned(self, path: Path, identity, guard) -> None:
        guard.validate()
        self._check_root()
        if _file_identity(_require_private_file(path)) != identity:
            raise RepositoryRightsTamperError("repository rights cleanup object changed")
        path.unlink()
        _fsync_directory(self.root)
        guard.validate()

    def _publish_owned(self, source: Path, destination: Path, identity, guard) -> None:
        guard.validate()
        self._check_root()
        if _file_identity(_require_private_file(source)) != identity:
            raise RepositoryRightsTamperError("repository rights publication object changed")
        _rename_no_replace(source, destination)
        _fsync_directory(self.root)
        guard.validate()
        if _file_identity(_require_private_file(destination)) != identity:
            raise RepositoryRightsTamperError("published repository rights object changed")

    def _history(self, guard) -> tuple[RepositoryRightsRecord, ...]:
        try:
            return self._read_history(guard)
        except RepositoryRightsTamperError:
            raise
        except (OSError, ValueError, VaultAuthenticationError, HistoryAnchorProtocolError) as error:
            raise RepositoryRightsTamperError("repository rights history failed authentication") from error

    def _read_history(self, guard) -> tuple[RepositoryRightsRecord, ...]:
        guard.validate()
        self._check_root()
        anchor = read_history_anchor(self.vault, _RIGHTS_SCOPE)
        finals, pending, staging = [], [], []
        for path in self.root.iterdir():
            for pattern, group in ((_RIGHTS_FINAL, finals), (_RIGHTS_PENDING, pending), (_RIGHTS_STAGING, staging)):
                match = pattern.fullmatch(path.name)
                if match is not None:
                    metadata = _require_private_file(path)
                    if metadata.st_size > _RIGHTS_MAX_BYTES:
                        raise RepositoryRightsTamperError("repository rights object exceeds its bound")
                    group.append((int(match.group(1)), match.group(2), path, _file_identity(metadata)))
                    break
            else:
                raise RepositoryRightsTamperError("unknown repository rights object")
        if len(pending) > 1 or len(staging) > 1 or (pending and staging):
            raise RepositoryRightsTamperError("repository rights publication objects conflict")

        records: list[RepositoryRightsRecord] = []
        predecessor = None
        for ordinal, (sequence, digest, path, identity) in enumerate(sorted(finals), start=1):
            if sequence != ordinal:
                raise RepositoryRightsTamperError("repository rights history has a gap")
            record, observed = self._read_record(path, digest, predecessor)
            if observed != identity:
                raise RepositoryRightsTamperError("repository rights inventory changed")
            self._validate_history_record(records, record)
            records.append(record)
            predecessor = record.record_sha256

        pending_record = None
        if pending:
            sequence, digest, path, identity = pending[0]
            if sequence != len(records) + 1:
                raise RepositoryRightsTamperError("pending repository rights event is out of sequence")
            pending_record, observed = self._read_record(path, digest, predecessor)
            if observed != identity:
                raise RepositoryRightsTamperError("pending repository rights inventory changed")
            self._validate_history_record(records, pending_record)

        if anchor.count == len(records) and anchor.head_sha256 == predecessor:
            # Only this exact unchanged anchor proves transient objects did not
            # commit. Complete pending events were authenticated before cleanup.
            if staging:
                sequence, _digest, path, identity = staging[0]
                if sequence != len(records) + 1:
                    raise RepositoryRightsTamperError("staged repository rights event is out of sequence")
                self._remove_owned(path, identity, guard)
            elif pending:
                self._remove_owned(pending[0][2], pending[0][3], guard)
        elif (
            not staging and pending_record is not None
            and anchor.count == len(records) + 1
            and anchor.head_sha256 == pending_record.record_sha256
        ):
            sequence, digest, path, identity = pending[0]
            final = self.root / f"e{sequence:020d}-{digest}.sealed"
            self._publish_owned(path, final, identity, guard)
            replayed, _identity = self._read_record(final, digest, predecessor)
            if replayed != pending_record:
                raise RepositoryRightsTamperError("replayed repository rights event changed")
            records.append(replayed)
        else:
            raise RepositoryRightsTamperError("repository rights history does not match its anchor")
        guard.validate()
        return tuple(records)

    def find_exact(self, inspection: RepositoryInspection) -> RepositoryRightsRecord | None:
        inspection = require_release_inspection(inspection)
        with self._locked() as guard:
            records = self._matching_identity(self._history(guard), inspection.repository)
            return next((record for record in records if (
                record.commit_sha == inspection.commit_sha
                and record.tree_sha == inspection.tree_sha
                and record.policy_sha256 == inspection.policy_sha256
            )), None)

    def register(
        self, *, inspection: RepositoryInspection,
        grant: str, declared_scope: str, literal_confirmation: str,
    ) -> RepositoryRightsRecord:
        inspection = require_release_inspection(inspection)
        _require_grant_scope(grant, declared_scope)
        if type(literal_confirmation) is not str or literal_confirmation != "Y":
            raise ValueError("repository rights require literal Y confirmation")
        with self._locked() as guard:
            records = self._history(guard)
            for previous in self._matching_identity(records, inspection.repository):
                if previous.commit_sha != inspection.commit_sha:
                    continue
                if previous.tree_sha != inspection.tree_sha:
                    raise RepositoryRightsConflictError("repository commit tree changed")
                if previous.grant != grant or previous.declared_scope != declared_scope:
                    raise RepositoryRightsConflictError("exact repository rights already have a different grant or scope")
                return previous

            predecessor = records[-1].record_sha256 if records else None
            state = {
                "schema": "LIVED_REPOSITORY_RIGHTS_V1", "record_id": uuid4().hex,
                "repository": asdict(inspection.repository),
                "commit_sha": inspection.commit_sha, "tree_sha": inspection.tree_sha,
                "grant": grant, "declared_scope": declared_scope,
                "confirmed_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "policy_sha256": inspection.policy_sha256, "predecessor_sha256": predecessor,
            }
            state["record_sha256"] = RepositoryRightsRecord.digest_state(state)
            record = RepositoryRightsRecord.from_state(state)
            self._validate_history_record(records, record)
            payload = self.vault.seal_local_blob(_RIGHTS_NAMESPACE, record.record_sha256, _canonical_json(state))
            stem = f"e{len(records) + 1:020d}-{record.record_sha256}"
            staging, pending, final = (self.root / name for name in (
                f".staging-{stem}", f".pending-{stem}", f"{stem}.sealed",
            ))
            guard.validate()
            _write_new_private(staging, payload)
            identity = _file_identity(_require_private_file(staging))
            self._publish_owned(staging, pending, identity, guard)
            durable, observed = self._read_record(pending, record.record_sha256, predecessor)
            if durable != record or observed != identity:
                raise RepositoryRightsTamperError("pending repository rights event changed before anchoring")
            # Keep pending on every uncertain result, including an exception
            # after SQLite committed. Recovery decides solely from the anchor.
            advance_history_anchor(
                self.vault, _RIGHTS_SCOPE, expected_count=len(records),
                expected_head_sha256=predecessor, new_head_sha256=record.record_sha256,
            )
            self._publish_owned(pending, final, identity, guard)
            published, _identity = self._read_record(final, record.record_sha256, predecessor)
            if published != record:
                raise RepositoryRightsTamperError("published repository rights event changed")
            return published
