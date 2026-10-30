from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Protocol
from uuid import uuid4

from .paths import (
    PathRaceError,
    StudioPaths,
    UnsafePathError,
    _absolute_lexical,
    _checked_lstat,
    _entry_identity,
    _require_private_directory as _paths_require_private_directory,
    _require_private_file as _paths_require_private_file,
    _set_private_windows_acl,
    _snapshot_chain,
)
from .training_models import (
    CpuExecutionInventory,
    ExecutionHardwareInventory,
    HardwareInventory,
    RunState,
    TrainingEvent,
    TrainingMode,
    TrainingPlan,
)

_HEX32 = re.compile(r"[a-f0-9]{32}")
_HEX64 = re.compile(r"[a-f0-9]{64}")
_BASENAME = re.compile(r"manifest-([0-9]{4})[.]json")
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_MAX_RUNS = 2_048
_MAX_MANIFESTS_PER_RUN = 10_000
_MAX_CHILDREN = 10_100
_MAX_MANIFEST_BYTES = 131_072
_MAX_EVENTS = 100_000
_ALLOWED_STATE = frozenset(RunState)
_ALLOWED_MODE = frozenset(TrainingMode)


class AuditTamperError(RuntimeError):
    pass


class ManifestEventKind(StrEnum):
    MANIFEST_PREPARED = "MANIFEST_PREPARED"
    MANIFEST_PUBLISHED = "MANIFEST_PUBLISHED"
    ABORTED_BEFORE_PUBLICATION = "ABORTED_BEFORE_PUBLICATION"
    RECONCILIATION_BLOCKED = "RECONCILIATION_BLOCKED"
    AUDIT_TAMPERED = "AUDIT_TAMPERED"
    ROOT_ANOMALY = "ROOT_ANOMALY"


class ReconciliationOutcome(StrEnum):
    MANIFEST_PUBLISHED = "MANIFEST_PUBLISHED"
    ABORTED_BEFORE_PUBLICATION = "ABORTED_BEFORE_PUBLICATION"
    RECONCILIATION_BLOCKED = "RECONCILIATION_BLOCKED"
    AUDIT_TAMPERED = "AUDIT_TAMPERED"


def _integer(value: object, name: str, *, minimum: int = 0,
             maximum: int = _MAX_SEQUENCE) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _sha256(value: object, name: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _run_id(value: object) -> str:
    if type(value) is not str or _HEX32.fullmatch(value) is None:
        raise ValueError("run_id must be 32 lowercase hexadecimal characters")
    return value


def _event_id(value: object, name: str = "event_id") -> str:
    if type(value) is not str or _HEX32.fullmatch(value) is None:
        raise ValueError(f"{name} must be 32 lowercase hexadecimal characters")
    return value


def _utc(value: object, name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return value


def _canonical_utc_text(value: object, name: str) -> str:
    text = _text(value, name, maximum=64)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{name} must be valid ISO-8601") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset() != timedelta(0)
        or parsed.isoformat() != text
    ):
        raise ValueError(f"{name} must be canonical UTC")
    return text


def _text(value: object, name: str, *, maximum: int,
          ascii_only: bool = True) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a nonempty string")
    encoded = value.encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError(f"{name} is too long")
    if ascii_only and not value.isascii():
        raise ValueError(f"{name} must be ASCII")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{name} contains controls")
    return value


def _public_identity_text(value: object, name: str, *, maximum: int) -> str:
    """Validate bounded, content-free identity metadata for public manifests."""
    text = _text(value, name, maximum=maximum)
    if any(marker in text for marker in ("/", "\\", ":")):
        raise ValueError(f"{name} must not contain a path or URI")
    components = {
        component
        for component in re.split(r"[^a-z0-9]+", text.casefold())
        if component
    }
    sensitive = {
        "capture", "cookie", "credential", "pair", "path", "prompt",
        "provenance", "response", "secret", "session", "source", "token",
        "url",
    }
    if components & sensitive:
        raise ValueError(f"{name} contains a sensitive data label")
    return text


def _finite(value: object, name: str, *, optional: bool = False) -> float | None:
    if optional and value is None:
        return None
    if type(value) not in (int, float) or isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return converted


def _basename(value: object, sequence: int) -> str:
    if type(value) is not str or _BASENAME.fullmatch(value) is None:
        raise ValueError("manifest basename is invalid")
    if value != f"manifest-{sequence:04d}.json":
        raise ValueError("manifest basename does not match its sequence")
    return value


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("manifest document is not canonical JSON") from error


@dataclass(frozen=True, slots=True)
class ManifestGpuIdentity:
    name: str
    memory_mib: int
    compute_capability: str

    def __post_init__(self) -> None:
        _public_identity_text(self.name, "GPU name", maximum=128)
        _integer(self.memory_mib, "GPU memory", minimum=1, maximum=1_125_899_906_842_624)
        _public_identity_text(
            self.compute_capability, "compute capability", maximum=32
        )


@dataclass(frozen=True, slots=True)
class TrainingRunIdentity:
    run_id: str
    plan_id: str
    mode: TrainingMode
    created_at: datetime
    plan_expires_at: datetime
    snapshot_sequence: int
    dataset_count: int
    dataset_bytes: int
    dataset_sha256: str
    config_sha256: str
    model_identity_sha256: str
    engine_identity: str
    hardware_profile: str
    projection_schema: str
    release_manifest_sha256: str
    runtime_sha256: str
    python_version: str
    cuda_version: str | None
    driver_version: str | None
    gpus: tuple[ManifestGpuIdentity, ...]
    dataset_manifest_sha256: str | None
    train_sha256: str | None
    eval_sha256: str | None
    parent_kind: str
    parent_identity_sha256: str | None
    audit_schema_version: int = 2
    execution_backend: str | None = None
    release_scope: str | None = None

    def __post_init__(self) -> None:
        _run_id(self.run_id)
        _event_id(self.plan_id, "plan_id")
        if type(self.mode) is not TrainingMode:
            raise TypeError("mode must be TrainingMode")
        _utc(self.created_at, "created_at")
        _utc(self.plan_expires_at, "plan_expires_at")
        if self.plan_expires_at <= self.created_at:
            raise ValueError("plan expiry must follow creation")
        _integer(self.snapshot_sequence, "snapshot_sequence")
        _integer(self.dataset_count, "dataset_count", minimum=1, maximum=5_000_000)
        _integer(self.dataset_bytes, "dataset_bytes", minimum=1, maximum=8_589_934_592)
        for name in ("dataset_sha256", "config_sha256", "model_identity_sha256",
                     "release_manifest_sha256", "runtime_sha256"):
            _sha256(getattr(self, name), name)
        for name in ("engine_identity", "hardware_profile", "projection_schema",
                     "python_version"):
            _public_identity_text(getattr(self, name), name, maximum=128)
        _integer(
            self.audit_schema_version, "audit_schema_version", minimum=2, maximum=3
        )
        if self.audit_schema_version == 2:
            if self.execution_backend is not None or self.release_scope is not None:
                raise ValueError("V2 run identity cannot contain V3 labels")
            for name in ("cuda_version", "driver_version"):
                _public_identity_text(getattr(self, name), name, maximum=128)
            if type(self.gpus) is not tuple or not 1 <= len(self.gpus) <= 8:
                raise ValueError("gpus must be a bounded tuple")
            if not all(type(gpu) is ManifestGpuIdentity for gpu in self.gpus):
                raise TypeError("gpus must contain ManifestGpuIdentity records")
        else:
            if (
                self.mode is not TrainingMode.RAG
                or self.engine_identity != "local-faiss"
                or self.hardware_profile != "local"
                or type(self.execution_backend) is not str
                or self.execution_backend != "cpu"
                or type(self.release_scope) is not str
                or self.release_scope not in ("local-desktop", "canonical-training")
                or type(self.gpus) is not tuple
                or self.gpus
                or self.cuda_version is not None
                or self.driver_version is not None
            ):
                raise ValueError("V3 run identity must describe local CPU RAG execution")
        if self.mode is TrainingMode.RAG:
            if any(value is not None for value in (
                self.dataset_manifest_sha256, self.train_sha256, self.eval_sha256,
                self.parent_identity_sha256,
            )) or self.parent_kind != "NONE":
                raise ValueError("local RAG run identity has remote bindings")
        else:
            for name in ("dataset_manifest_sha256", "train_sha256", "eval_sha256"):
                _sha256(getattr(self, name), name)
            if self.mode is TrainingMode.LORA:
                if self.parent_kind != "NONE" or self.parent_identity_sha256 is not None:
                    raise ValueError("LoRA run identity has a parent")
            elif self.parent_kind in {"LORA", "BASE"}:
                _sha256(self.parent_identity_sha256, "parent_identity_sha256")
                if (
                    self.parent_kind == "BASE"
                    and self.parent_identity_sha256 != self.model_identity_sha256
                ):
                    raise ValueError(
                        "DPO raw-base parent differs from the base model identity"
                    )
            else:
                raise ValueError("DPO run identity parent is invalid")


@dataclass(frozen=True, slots=True)
class ManifestPreparation:
    run_id: str
    manifest_sequence: int
    basename: str
    byte_length: int
    manifest_sha256: str
    previous_manifest_sha256: str | None
    created_at: datetime
    run_identity: TrainingRunIdentity | None = None

    def __post_init__(self) -> None:
        _run_id(self.run_id)
        _integer(self.manifest_sequence, "manifest_sequence",
                 maximum=_MAX_MANIFESTS_PER_RUN - 1)
        _basename(self.basename, self.manifest_sequence)
        _integer(self.byte_length, "byte_length", minimum=2, maximum=_MAX_MANIFEST_BYTES)
        _sha256(self.manifest_sha256, "manifest_sha256")
        _sha256(self.previous_manifest_sha256, "previous_manifest_sha256", optional=True)
        _utc(self.created_at, "created_at")
        if (self.manifest_sequence == 0) != (self.previous_manifest_sha256 is None):
            raise ValueError("manifest predecessor does not match sequence")
        if self.run_identity is not None:
            if type(self.run_identity) is not TrainingRunIdentity:
                raise TypeError("run_identity must be TrainingRunIdentity or None")
            if self.manifest_sequence != 0 or self.run_identity.run_id != self.run_id:
                raise ValueError("run identity is only valid for manifest zero")


@dataclass(frozen=True, slots=True)
class PreparedManifestResult:
    prepared_event_id: str
    created: bool
    def __post_init__(self) -> None:
        _event_id(self.prepared_event_id, "prepared_event_id")
        if type(self.created) is not bool:
            raise TypeError("created must be boolean")


@dataclass(frozen=True, slots=True)
class AuditAppendResult:
    event_id: str
    created: bool
    def __post_init__(self) -> None:
        _event_id(self.event_id)
        if type(self.created) is not bool:
            raise TypeError("created must be boolean")


@dataclass(frozen=True, slots=True)
class ManifestAuditEvent:
    event_id: str
    global_sequence: int
    capture_sequence: int
    run_id: str
    event_ordinal: int
    kind: ManifestEventKind
    prepared_event_id: str | None
    manifest_sequence: int | None
    basename: str | None
    byte_length: int | None
    manifest_sha256: str | None
    previous_manifest_sha256: str | None
    previous_event_sha256: str | None
    evidence_sha256: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        _event_id(self.event_id)
        _integer(self.global_sequence, "global_sequence", minimum=1)
        _integer(self.capture_sequence, "capture_sequence")
        _run_id(self.run_id)
        _integer(self.event_ordinal, "event_ordinal", minimum=1)
        if type(self.kind) is not ManifestEventKind:
            raise TypeError("kind must be ManifestEventKind")
        if self.prepared_event_id is not None:
            _event_id(self.prepared_event_id, "prepared_event_id")
        if self.manifest_sequence is not None:
            _integer(self.manifest_sequence, "manifest_sequence",
                     maximum=_MAX_MANIFESTS_PER_RUN - 1)
            if self.basename is None:
                raise ValueError("manifest event basename is missing")
            _basename(self.basename, self.manifest_sequence)
        elif self.basename is not None:
            raise ValueError("manifest basename requires a sequence")
        if self.byte_length is not None:
            _integer(self.byte_length, "byte_length", minimum=2, maximum=_MAX_MANIFEST_BYTES)
        for name in ("manifest_sha256", "previous_manifest_sha256",
                     "previous_event_sha256", "evidence_sha256"):
            _sha256(getattr(self, name), name, optional=True)
        _utc(self.created_at, "created_at")
        has_manifest = all(
            value is not None
            for value in (
                self.manifest_sequence,
                self.basename,
                self.byte_length,
                self.manifest_sha256,
            )
        )
        has_any_manifest = any(
            value is not None
            for value in (
                self.manifest_sequence,
                self.basename,
                self.byte_length,
                self.manifest_sha256,
                self.previous_manifest_sha256,
            )
        )
        if self.kind is ManifestEventKind.ROOT_ANOMALY:
            raise ValueError("root anomalies use the global root-event schema")
        if self.kind is ManifestEventKind.MANIFEST_PREPARED:
            if (
                self.prepared_event_id is not None
                or self.evidence_sha256 is not None
                or not has_manifest
            ):
                raise ValueError("prepared manifest event fields are invalid")
        elif self.prepared_event_id is not None:
            if (
                self.kind
                not in {
                    ManifestEventKind.MANIFEST_PUBLISHED,
                    ManifestEventKind.ABORTED_BEFORE_PUBLICATION,
                    ManifestEventKind.RECONCILIATION_BLOCKED,
                    ManifestEventKind.AUDIT_TAMPERED,
                }
                or not has_manifest
                or self.evidence_sha256 is None
            ):
                raise ValueError("terminal manifest event fields are invalid")
        elif (
            self.kind
            not in {
                ManifestEventKind.RECONCILIATION_BLOCKED,
                ManifestEventKind.AUDIT_TAMPERED,
            }
            or has_any_manifest
            or self.evidence_sha256 is None
        ):
            raise ValueError("run-scoped anomaly event fields are invalid")


@dataclass(frozen=True, slots=True)
class RunManifestRecord:
    run_id: str
    sequence: int
    path: Path
    byte_length: int
    sha256: str
    previous_manifest_sha256: str | None
    def __post_init__(self) -> None:
        _run_id(self.run_id)
        _integer(self.sequence, "sequence", maximum=_MAX_MANIFESTS_PER_RUN - 1)
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        _integer(self.byte_length, "byte_length", minimum=2, maximum=_MAX_MANIFEST_BYTES)
        _sha256(self.sha256, "sha256")
        _sha256(self.previous_manifest_sha256, "previous_manifest_sha256", optional=True)
        if (self.sequence == 0) != (self.previous_manifest_sha256 is None):
            raise ValueError("record predecessor does not match sequence")


@dataclass(frozen=True, slots=True)
class TrainingRunRecord:
    run_id: str
    mode: TrainingMode
    created_at: datetime
    manifest: RunManifestRecord
    def __post_init__(self) -> None:
        _run_id(self.run_id)
        if type(self.mode) is not TrainingMode:
            raise TypeError("mode must be TrainingMode")
        _utc(self.created_at, "created_at")
        if type(self.manifest) is not RunManifestRecord:
            raise TypeError("manifest must be RunManifestRecord")
        if self.manifest.run_id != self.run_id or self.manifest.sequence != 0:
            raise ValueError("initial manifest does not match run")
    @property
    def path(self) -> Path:
        return self.manifest.path


@dataclass(frozen=True, slots=True)
class ManifestReconciliation:
    run_id: str | None
    prepared_event_id: str | None
    outcome: ReconciliationOutcome
    event_id: str
    evidence_sha256: str
    def __post_init__(self) -> None:
        if self.run_id is not None:
            _run_id(self.run_id)
        if self.prepared_event_id is not None:
            _event_id(self.prepared_event_id, "prepared_event_id")
        if type(self.outcome) is not ReconciliationOutcome:
            raise TypeError("outcome must be ReconciliationOutcome")
        _event_id(self.event_id)
        _sha256(self.evidence_sha256, "evidence_sha256")


class TrainingAuditStore(Protocol):
    def prepare_manifest(self, event: ManifestPreparation) -> PreparedManifestResult: ...
    def mark_manifest_published(self, prepared_event_id: str) -> AuditAppendResult: ...
    def mark_manifest_reconciled(self, prepared_event_id: str,
                                 outcome: ReconciliationOutcome) -> AuditAppendResult: ...
    def manifest_events(self, run_id: str) -> tuple[ManifestAuditEvent, ...]: ...
    def all_manifest_events(self) -> tuple[ManifestAuditEvent, ...]: ...
    def training_run_identity(self, run_id: str) -> TrainingRunIdentity | None: ...
    def mark_run_anomaly(self, run_id: str, outcome: ReconciliationOutcome,
                         evidence_sha256: str) -> AuditAppendResult: ...
    def mark_root_anomaly(self, evidence_sha256: str) -> AuditAppendResult: ...


class _DuplicateKey(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("manifest contains a duplicate JSON key")
        result[key] = value
    return result


def _validate_number(value: object, name: str) -> None:
    if value is not None:
        _finite(value, name)


class RunManifestWriter:
    def __init__(self, paths: StudioPaths, audit_store: TrainingAuditStore,
                 release_manifest_sha256: str, *,
                 release_scope: str = "canonical-training"):
        if type(paths) is not StudioPaths:
            raise TypeError("paths must be StudioPaths")
        _sha256(release_manifest_sha256, "release_manifest_sha256")
        if type(release_scope) is not str or release_scope not in (
            "local-desktop", "canonical-training"
        ):
            raise ValueError("release_scope must name a supported release binding")
        self.paths = paths
        self.audit_store = audit_store
        self.release_manifest_sha256 = release_manifest_sha256
        self.release_scope = release_scope
        workspace = _absolute_lexical(paths.root)
        workspace_state = _checked_lstat(workspace)
        if not stat.S_ISDIR(workspace_state.st_mode):
            raise UnsafePathError("workspace root is not a directory")
        self._device = int(workspace_state.st_dev)
        self._root = paths.secure_directory("studio_data/training_runs", create=False)
        _paths_require_private_directory(
            self._root, expected_device=self._device
        )
        self._root_chain = _snapshot_chain(self._root, require_target=True)

    def _require_stable_root(
        self,
    ) -> tuple[tuple[str, tuple[int, int, int]], ...]:
        before = _snapshot_chain(self._root, require_target=True)
        if before != self._root_chain:
            raise PathRaceError(
                "training-run root identity differs from the pinned root"
            )
        self._require_private_directory(self._root)
        after = _snapshot_chain(self._root, require_target=True)
        if after != before:
            raise PathRaceError(
                "training-run root changed during validation"
            )
        return after

    @staticmethod
    def _checkpoint(_name: str) -> None:
        return None

    @staticmethod
    def _bounded_entries(
        path: Path, maximum: int
    ) -> tuple[tuple[object, ...], bool]:
        _integer(maximum, "directory entry limit", minimum=1)
        iterator = os.scandir(path)
        entries: list[object] = []
        exceeded = False
        try:
            for index, entry in enumerate(iterator):
                if index == maximum:
                    exceeded = True
                    break
                entries.append(entry)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        return tuple(entries), exceeded

    @staticmethod
    def _identity(plan: TrainingPlan, hardware: ExecutionHardwareInventory,
                  release_sha256: str, run_id: str, *,
                  release_scope: str) -> TrainingRunIdentity:
        if type(plan) is not TrainingPlan:
            raise TypeError("plan must be TrainingPlan")
        if type(hardware) not in (HardwareInventory, CpuExecutionInventory):
            raise TypeError("hardware must be HardwareInventory or CpuExecutionInventory")
        cpu = type(hardware) is CpuExecutionInventory
        return TrainingRunIdentity(
            run_id=run_id, plan_id=plan.plan_id, mode=plan.mode,
            created_at=plan.created_at, plan_expires_at=plan.expires_at,
            snapshot_sequence=plan.snapshot_sequence,
            dataset_count=plan.dataset_count, dataset_bytes=plan.dataset_bytes,
            dataset_sha256=plan.dataset_sha256, config_sha256=plan.config_sha256,
            model_identity_sha256=plan.model_identity_sha256,
            engine_identity=plan.engine_identity,
            hardware_profile=plan.hardware_profile,
            projection_schema=plan.projection_schema,
            release_manifest_sha256=release_sha256,
            runtime_sha256=hardware.runtime_sha256,
            python_version=hardware.python_version,
            cuda_version=None if cpu else hardware.cuda_version,
            driver_version=None if cpu else hardware.driver_version,
            gpus=() if cpu else tuple(
                ManifestGpuIdentity(gpu.name, gpu.memory_mib, gpu.compute_capability)
                for gpu in hardware.gpus
            ),
            dataset_manifest_sha256=plan.dataset_manifest_sha256,
            train_sha256=plan.train_sha256,
            eval_sha256=plan.eval_sha256,
            parent_kind=plan.parent_kind,
            parent_identity_sha256=plan.parent_identity_sha256,
            audit_schema_version=3 if cpu else 2,
            execution_backend="cpu" if cpu else None,
            release_scope=release_scope if cpu else None,
        )

    @staticmethod
    def _manifest_schema(identity: TrainingRunIdentity) -> str:
        if identity.audit_schema_version == 2:
            schema = "LIVED_TRAINING_RUN_MANIFEST_V2"
        else:
            schema = "LIVED_TRAINING_RUN_MANIFEST_V3"
        return schema

    @staticmethod
    def _initial_document(identity: TrainingRunIdentity) -> dict[str, object]:
        document = {
            "config_sha256": identity.config_sha256,
            "created_at": identity.created_at.isoformat(),
            "dataset_bytes": identity.dataset_bytes,
            "dataset_count": identity.dataset_count,
            "dataset_sha256": identity.dataset_sha256,
            "dataset_manifest_sha256": identity.dataset_manifest_sha256,
            "engine_identity": identity.engine_identity,
            "hardware": {
                "cuda_version": identity.cuda_version,
                "driver_version": identity.driver_version,
                "gpus": [{"compute_capability": gpu.compute_capability,
                          "memory_mib": gpu.memory_mib, "name": gpu.name}
                         for gpu in identity.gpus],
                "python_version": identity.python_version,
                "runtime_sha256": identity.runtime_sha256,
            },
            "hardware_profile": identity.hardware_profile,
            "manifest_kind": "RUN_STARTED",
            "manifest_sequence": 0,
            "mode": identity.mode.value,
            "model_identity_sha256": identity.model_identity_sha256,
            "plan_expires_at": identity.plan_expires_at.isoformat(),
            "previous_manifest_sha256": None,
            "projection_schema": identity.projection_schema,
            "train_sha256": identity.train_sha256,
            "eval_sha256": identity.eval_sha256,
            "parent_kind": identity.parent_kind,
            "parent_identity_sha256": identity.parent_identity_sha256,
            "release_manifest_sha256": identity.release_manifest_sha256,
            "schema": RunManifestWriter._manifest_schema(identity),
            "snapshot_sequence": identity.snapshot_sequence,
        }
        if identity.audit_schema_version == 3:
            document.update(
                execution_backend=identity.execution_backend,
                release_scope=identity.release_scope,
            )
        return document

    @staticmethod
    def _event_document(sequence: int, previous: str,
                        event: TrainingEvent, *,
                        identity: TrainingRunIdentity) -> dict[str, object]:
        if type(event) is not TrainingEvent:
            raise TypeError("event must be TrainingEvent")
        if event.sequence != sequence:
            raise ValueError("training event sequence must match manifest sequence")
        if identity.audit_schema_version == 3 and event.gpu is not None:
            raise ValueError("CPU run events cannot contain GPU telemetry")
        gpu = None if event.gpu is None else {
            "total_bytes": event.gpu.total_bytes, "used_bytes": event.gpu.used_bytes
        }
        return {
            "checkpoint_sha256": event.checkpoint_sha256,
            "completed_units": event.completed_units,
            "elapsed_seconds": event.elapsed_seconds,
            "epoch": event.epoch,
            "eta_seconds": event.eta_seconds,
            "event_sequence": event.sequence,
            "gpu": gpu,
            "learning_rate": event.learning_rate,
            "loss": event.loss,
            "manifest_kind": "RUN_EVENT",
            "manifest_sequence": sequence,
            "phase_sha256": hashlib.sha256(
                b"liveD:training-phase:v1\0" + event.phase.encode("utf-8")
            ).hexdigest(),
            "previous_manifest_sha256": previous,
            "recorded_at": event.recorded_at.isoformat(),
            "schema": RunManifestWriter._manifest_schema(identity),
            "state": event.state.value,
            "throughput": event.throughput,
            "total_units": event.total_units,
        }

    def _run_directory(self, run_id: str) -> Path:
        _run_id(run_id)
        return self.paths.training_runs / run_id

    def _ensure_run_directory(self, run_id: str, *, known: bool) -> Path:
        self._require_stable_root()
        relative = f"studio_data/training_runs/{run_id}"
        candidate = _absolute_lexical(self._run_directory(run_id))
        try:
            metadata = _checked_lstat(candidate)
        except FileNotFoundError:
            if known:
                raise AuditTamperError("authenticated run directory is missing") from None
            before = _snapshot_chain(self._root, require_target=True)
            os.mkdir(candidate, 0o700)
            if os.name == "nt":
                _set_private_windows_acl(candidate, directory=True)
            else:
                os.chmod(candidate, 0o700, follow_symlinks=False)
            if _snapshot_chain(self._root, require_target=True) != before:
                raise PathRaceError("run root changed while creating a run directory")
        else:
            if not known:
                raise AuditTamperError("new run identifier already has a filesystem object")
        directory = self.paths.secure_directory(relative, create=False)
        self._require_private_directory(directory)
        self._require_stable_root()
        return directory

    def _require_private_directory(self, path: Path) -> None:
        _paths_require_private_directory(
            path, expected_device=self._device
        )

    def _require_private_file(self, path: Path) -> os.stat_result:
        return _paths_require_private_file(
            path, expected_device=self._device
        )

    def start_run(self, plan: TrainingPlan, hardware: ExecutionHardwareInventory,
                  *, run_id: str | None = None) -> TrainingRunRecord:
        reconciled = self.reconcile_pending()
        if any(
            result.outcome
            in {
                ReconciliationOutcome.RECONCILIATION_BLOCKED,
                ReconciliationOutcome.AUDIT_TAMPERED,
            }
            for result in reconciled
        ):
            raise AuditTamperError(
                "training audit reconciliation must be resolved before start"
            )
        known_runs = {
            event.run_id for event in self.audit_store.all_manifest_events()
        }
        self._require_stable_root()
        root_entries, root_exceeded = self._bounded_entries(
            self._root, _MAX_RUNS
        )
        self._require_stable_root()
        if (
            root_exceeded
            or any(entry.name not in known_runs for entry in root_entries)
        ):
            raise AuditTamperError(
                "unknown training-run root object must be resolved before start"
            )
        selected = uuid4().hex if run_id is None else _run_id(run_id)
        existing = self.audit_store.training_run_identity(selected)
        identity = self._identity(
            plan, hardware, self.release_manifest_sha256, selected,
            release_scope=self.release_scope,
        )
        if existing is not None and existing != identity:
            raise AuditTamperError("run identity does not match its authenticated record")
        directory = self._ensure_run_directory(selected, known=existing is not None)
        payload = _canonical_json(self._initial_document(identity))
        preparation = ManifestPreparation(
            selected, 0, "manifest-0000.json", len(payload),
            hashlib.sha256(payload).hexdigest(), None, plan.created_at, identity,
        )
        owned_directory_identity = (
            _entry_identity(_checked_lstat(_absolute_lexical(directory)))
            if existing is None
            else None
        )
        try:
            prepared = self.audit_store.prepare_manifest(preparation)
        except BaseException:
            if owned_directory_identity is not None:
                try:
                    current = _checked_lstat(_absolute_lexical(directory))
                    children, children_exceeded = self._bounded_entries(
                        directory, 1
                    )
                except (FileNotFoundError, OSError, UnsafePathError):
                    pass
                else:
                    if (
                        _entry_identity(current) == owned_directory_identity
                        and not children_exceeded
                        and not children
                    ):
                        os.rmdir(directory)
            raise
        self._checkpoint("after_prepared")
        path = self._publish(directory, preparation.basename, payload)
        self._checkpoint("before_published_event")
        self.audit_store.mark_manifest_published(prepared.prepared_event_id)
        record = self._record_from_file(
            selected, preparation, path,
            identity=self.audit_store.training_run_identity(selected),
        )
        return TrainingRunRecord(selected, plan.mode, plan.created_at, record)

    def append_event(self, run_id: str, event: TrainingEvent) -> RunManifestRecord:
        _run_id(run_id)
        identity = self.audit_store.training_run_identity(run_id)
        if identity is None:
            raise KeyError(run_id)
        chain = self._verify_run_chain(run_id, identity=identity)
        if not chain:
            raise AuditTamperError("run has no published initial manifest")
        sequence = len(chain)
        if sequence >= _MAX_MANIFESTS_PER_RUN:
            raise ValueError("run manifest limit reached")
        previous = chain[-1].sha256
        payload = _canonical_json(
            self._event_document(sequence, previous, event, identity=identity)
        )
        preparation = ManifestPreparation(
            run_id, sequence, f"manifest-{sequence:04d}.json", len(payload),
            hashlib.sha256(payload).hexdigest(), previous, event.recorded_at,
        )
        prepared = self.audit_store.prepare_manifest(preparation)
        self._checkpoint("after_prepared")
        path = self._publish(self._run_directory(run_id), preparation.basename, payload)
        self._checkpoint("before_published_event")
        self.audit_store.mark_manifest_published(prepared.prepared_event_id)
        return self._record_from_file(run_id, preparation, path, identity=identity)

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        position = 0
        while position < len(payload):
            written = os.write(descriptor, payload[position:])
            if written <= 0:
                raise OSError("manifest write made no progress")
            position += written

    @staticmethod
    def _rename_noreplace(source: Path, destination: Path) -> None:
        if os.name == "nt":
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            move = kernel.MoveFileExW
            move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
            move.restype = ctypes.c_int
            if not move(os.fspath(source), os.fspath(destination), 0x00000008):
                error = ctypes.get_last_error()
                if error in (80, 183):
                    raise FileExistsError(destination)
                raise ctypes.WinError(error)
            return
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError("renameat2(RENAME_NOREPLACE) is unavailable")
        renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                              ctypes.c_char_p, ctypes.c_uint)
        renameat2.restype = ctypes.c_int
        if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
            error = ctypes.get_errno()
            if error == 17:
                raise FileExistsError(destination)
            raise OSError(error, os.strerror(error), destination)

    @staticmethod
    def _open_exclusive_temporary(path: Path) -> int:
        if os.name != "nt":
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
            for option in ("O_CLOEXEC", "O_NOFOLLOW"):
                flags |= int(getattr(os, option, 0))
            return os.open(path, flags, 0o600)

        import msvcrt

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        create = kernel.CreateFileW
        create.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        create.restype = ctypes.c_void_p
        close = kernel.CloseHandle
        close.argtypes = (ctypes.c_void_p,)
        close.restype = ctypes.c_int
        handle = create(
            os.fspath(path),
            0x80000000 | 0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,
            None,
            1,
            0x00000080 | 0x00200000,
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            if error in (80, 183):
                raise FileExistsError(path)
            raise ctypes.WinError(error)
        flags = os.O_RDWR | int(getattr(os, "O_BINARY", 0))
        try:
            return msvcrt.open_osfhandle(handle, flags)
        except BaseException:
            close(handle)
            raise

    @staticmethod
    def _read_descriptor_bytes(
        descriptor: int, identity: tuple[int, int, int]
    ) -> bytes:
        before = os.fstat(descriptor)
        if (
            _entry_identity(before) != identity
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > _MAX_MANIFEST_BYTES
        ):
            raise UnsafePathError("manifest descriptor object is unsafe")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            _entry_identity(after) != identity
            or after.st_size != before.st_size
            or len(payload) != before.st_size
            or len(payload) > _MAX_MANIFEST_BYTES
        ):
            raise PathRaceError("manifest descriptor changed while reading")
        return payload

    def _publish(self, directory: Path, basename: str, payload: bytes) -> Path:
        if type(basename) is not str or _BASENAME.fullmatch(basename) is None:
            raise ValueError("manifest basename is invalid")
        if type(payload) is not bytes or not 2 <= len(payload) <= _MAX_MANIFEST_BYTES:
            raise ValueError("manifest payload is invalid")
        directory = _absolute_lexical(directory)
        if directory != _absolute_lexical(self._run_directory(directory.name)):
            raise UnsafePathError("manifest publication directory is not a run directory")
        self._require_stable_root()
        self._require_private_directory(directory)
        directory_chain = _snapshot_chain(directory, require_target=True)
        destination = directory / basename
        try:
            _checked_lstat(destination)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(destination)
        if (
            self._require_stable_root() != self._root_chain
            or _snapshot_chain(directory, require_target=True) != directory_chain
        ):
            raise PathRaceError("manifest directory changed before publication")
        self._checkpoint("before_write")
        self._require_stable_root()
        temporary = directory / f".manifest-{uuid4().hex}.tmp"
        descriptor = self._open_exclusive_temporary(temporary)
        identity: tuple[int, int, int] | None = None
        published = False
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                raise UnsafePathError("manifest temporary object is unsafe")
            identity = _entry_identity(opened)
            self._require_stable_root()
            path_created = _checked_lstat(temporary)
            if (
                _entry_identity(path_created) != identity
                or _snapshot_chain(directory, require_target=True)
                != directory_chain
            ):
                raise PathRaceError(
                    "manifest temporary path differs from its retained handle"
                )
            if os.name == "nt":
                _set_private_windows_acl(temporary, directory=False)
            else:
                os.fchmod(descriptor, 0o600)
            opened = os.fstat(descriptor)
            path_opened = self._require_private_file(temporary)
            if (
                _entry_identity(opened) != identity
                or _entry_identity(path_opened) != identity
                or opened.st_dev != os.lstat(directory).st_dev
            ):
                raise PathRaceError(
                    "manifest temporary identity changed during protection"
                )
            self._checkpoint("after_temp_create")
            if (
                self._require_stable_root() != self._root_chain
                or _snapshot_chain(directory, require_target=True)
                != directory_chain
            ):
                raise PathRaceError(
                    "manifest directory changed before payload write"
                )
            self._write_all(descriptor, payload)
            self._checkpoint("before_flush")
            os.fsync(descriptor)
            if self._read_descriptor_bytes(descriptor, identity) != payload:
                raise PathRaceError("manifest temporary bytes failed verification")
            path_before_rename = self._require_private_file(temporary)
            if (
                _entry_identity(path_before_rename) != identity
                or self._require_stable_root() != self._root_chain
                or _snapshot_chain(directory, require_target=True)
                != directory_chain
            ):
                raise PathRaceError(
                    "manifest temporary path changed before publication"
                )
            self._checkpoint("before_rename")
            if self._require_stable_root() != self._root_chain:
                raise PathRaceError("manifest root changed before rename")
            self._rename_noreplace(temporary, destination)
            published = True
            self._checkpoint("after_rename")
            if (
                self._require_stable_root() != self._root_chain
                or _snapshot_chain(directory, require_target=True)
                != directory_chain
            ):
                raise PathRaceError(
                    "manifest directory changed during publication"
                )
            destination_state = self._require_private_file(destination)
            retained_state = os.fstat(descriptor)
            if (
                _entry_identity(destination_state) != identity
                or _entry_identity(retained_state) != identity
                or destination_state.st_dev != os.lstat(directory).st_dev
            ):
                raise PathRaceError(
                    "published manifest identity differs from retained handle"
                )
            if os.name != "nt":
                parent_flags = os.O_RDONLY | os.O_DIRECTORY
                parent_flags |= int(getattr(os, "O_CLOEXEC", 0))
                parent = os.open(directory, parent_flags)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            self._checkpoint("before_verify")
            self._require_stable_root()
            if (
                self._read_descriptor_bytes(descriptor, identity) != payload
                or self._read_checked_bytes(destination) != payload
            ):
                raise PathRaceError("published manifest bytes failed verification")
            self._checkpoint("after_verify")
            return destination
        finally:
            os.close(descriptor)
            if not published and identity is not None:
                try:
                    actual = _checked_lstat(temporary)
                except FileNotFoundError:
                    pass
                else:
                    if _entry_identity(actual) == identity:
                        temporary.unlink()

    def _read_checked_bytes(self, path: Path) -> bytes:
        self._require_stable_root()
        before_chain = _snapshot_chain(path, require_target=True)
        before = self._require_private_file(path)
        if before.st_size > _MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds maximum size")
        flags = os.O_RDONLY
        for option in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= int(getattr(os, option, 0))
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if _entry_identity(opened) != _entry_identity(before):
                raise PathRaceError("manifest changed before descriptor open")
            chunks: list[bytes] = []
            remaining = before.st_size + 1
            while remaining:
                chunk = os.read(descriptor, min(65_536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            after = os.fstat(descriptor)
            path_after = self._require_private_file(path)
            if (_entry_identity(after) != _entry_identity(opened)
                    or _entry_identity(path_after) != _entry_identity(opened)
                    or _snapshot_chain(path, require_target=True) != before_chain):
                raise PathRaceError("manifest changed while reading")
            if len(payload) != before.st_size or len(payload) > _MAX_MANIFEST_BYTES:
                raise ValueError("manifest size changed or exceeded its bound")
            self._require_stable_root()
            return payload
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_document(document: object, sequence: int) -> None:
        if type(document) is not dict:
            raise AuditTamperError("manifest must be a JSON object")
        schema = document.get("schema")
        if schema not in (
            "LIVED_TRAINING_RUN_MANIFEST_V2", "LIVED_TRAINING_RUN_MANIFEST_V3"
        ):
            raise AuditTamperError("manifest schema identifier is invalid")
        cpu = schema == "LIVED_TRAINING_RUN_MANIFEST_V3"
        common = {"manifest_kind", "manifest_sequence",
                  "previous_manifest_sha256", "schema"}
        if document.get("manifest_kind") == "RUN_STARTED":
            expected = common | {
                "config_sha256", "created_at", "dataset_bytes", "dataset_count",
                "dataset_sha256", "engine_identity", "hardware", "hardware_profile",
                "mode", "model_identity_sha256", "plan_expires_at",
                "projection_schema", "release_manifest_sha256", "snapshot_sequence",
                "dataset_manifest_sha256", "train_sha256", "eval_sha256",
                "parent_kind", "parent_identity_sha256",
            }
            if cpu:
                expected |= {"execution_backend", "release_scope"}
            if set(document) != expected or sequence != 0:
                raise AuditTamperError("initial manifest schema is invalid")
            if document["mode"] not in {mode.value for mode in _ALLOWED_MODE}:
                raise AuditTamperError("manifest mode is invalid")
            for name in ("config_sha256", "dataset_sha256", "model_identity_sha256",
                         "release_manifest_sha256"):
                _sha256(document[name], name)
            mode = TrainingMode(document["mode"])
            if mode is TrainingMode.RAG:
                if any(document[name] is not None for name in (
                    "dataset_manifest_sha256", "train_sha256", "eval_sha256",
                    "parent_identity_sha256",
                )) or document["parent_kind"] != "NONE":
                    raise AuditTamperError("local RAG bindings are invalid")
            else:
                for name in ("dataset_manifest_sha256", "train_sha256", "eval_sha256"):
                    _sha256(document[name], name)
                if mode is TrainingMode.LORA:
                    if document["parent_kind"] != "NONE" or document["parent_identity_sha256"] is not None:
                        raise AuditTamperError("LoRA parent binding is invalid")
                elif document["parent_kind"] in {"LORA", "BASE"}:
                    _sha256(document["parent_identity_sha256"], "parent_identity_sha256")
                else:
                    raise AuditTamperError("DPO parent binding is invalid")
            for name in ("dataset_bytes", "dataset_count", "snapshot_sequence"):
                _integer(document[name], name)
            for name in ("created_at", "plan_expires_at"):
                _canonical_utc_text(document[name], name)
            for name in ("engine_identity", "hardware_profile", "projection_schema"):
                _public_identity_text(document[name], name, maximum=128)
            hardware = document["hardware"]
            if type(hardware) is not dict or set(hardware) != {
                "cuda_version", "driver_version", "gpus", "python_version",
                "runtime_sha256"}:
                raise AuditTamperError("manifest hardware schema is invalid")
            _sha256(hardware["runtime_sha256"], "runtime_sha256")
            version_fields = ("python_version",) if cpu else (
                "cuda_version", "driver_version", "python_version"
            )
            for name in version_fields:
                _text(hardware[name], name, maximum=128)
            gpus = hardware["gpus"]
            if cpu and (
                mode is not TrainingMode.RAG
                or document["engine_identity"] != "local-faiss"
                or document["hardware_profile"] != "local"
                or document["execution_backend"] != "cpu"
                or document["release_scope"] not in ("local-desktop", "canonical-training")
                or hardware["cuda_version"] is not None
                or hardware["driver_version"] is not None
            ):
                raise AuditTamperError("V3 manifest must describe local CPU RAG execution")
            if type(gpus) is not list or (
                len(gpus) != 0 if cpu else not 1 <= len(gpus) <= 8
            ):
                raise AuditTamperError("manifest GPU inventory is invalid")
            for gpu in gpus:
                if type(gpu) is not dict or set(gpu) != {
                    "compute_capability", "memory_mib", "name"}:
                    raise AuditTamperError("manifest GPU schema is invalid")
                ManifestGpuIdentity(gpu["name"], gpu["memory_mib"],
                                    gpu["compute_capability"])
        elif document.get("manifest_kind") == "RUN_EVENT":
            expected = common | {
                "checkpoint_sha256", "completed_units", "elapsed_seconds", "epoch",
                "eta_seconds", "event_sequence", "gpu", "learning_rate", "loss",
                "phase_sha256", "recorded_at", "state", "throughput", "total_units",
            }
            if set(document) != expected or sequence == 0:
                raise AuditTamperError("event manifest schema is invalid")
            if document["state"] not in {state.value for state in _ALLOWED_STATE}:
                raise AuditTamperError("event state is invalid")
            for name in ("manifest_sequence", "event_sequence", "completed_units",
                         "total_units"):
                _integer(document[name], name)
            if document["event_sequence"] != sequence:
                raise AuditTamperError("event sequence differs from manifest sequence")
            _sha256(document["phase_sha256"], "phase_sha256")
            _sha256(document["checkpoint_sha256"], "checkpoint_sha256", optional=True)
            _canonical_utc_text(document["recorded_at"], "recorded_at")
            for name in ("elapsed_seconds", "epoch", "eta_seconds",
                         "learning_rate", "loss", "throughput"):
                _validate_number(document[name], name)
            gpu = document["gpu"]
            if cpu and gpu is not None:
                raise AuditTamperError("CPU event manifest cannot contain GPU telemetry")
            if gpu is not None:
                if type(gpu) is not dict or set(gpu) != {"total_bytes", "used_bytes"}:
                    raise AuditTamperError("event GPU schema is invalid")
                _integer(gpu["used_bytes"], "used_bytes")
                _integer(gpu["total_bytes"], "total_bytes", minimum=1)
                if gpu["used_bytes"] > gpu["total_bytes"]:
                    raise AuditTamperError("GPU usage exceeds capacity")
        else:
            raise AuditTamperError("manifest kind is invalid")
        if document.get("manifest_sequence") != sequence:
            raise AuditTamperError("manifest sequence is invalid")
        previous = document.get("previous_manifest_sha256")
        _sha256(previous, "previous_manifest_sha256", optional=True)
        if (sequence == 0) != (previous is None):
            raise AuditTamperError("manifest predecessor is invalid")

    def _decode_canonical(self, payload: bytes, sequence: int) -> dict[str, object]:
        if not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
            raise AuditTamperError("manifest must have exactly one trailing LF")
        try:
            text = payload.decode("utf-8")
            document = json.loads(text, object_pairs_hook=_strict_object)
        except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateKey) as error:
            raise AuditTamperError("manifest is not strict UTF-8 JSON") from error
        self._validate_document(document, sequence)
        if _canonical_json(document) != payload:
            raise AuditTamperError("manifest bytes are not canonical")
        return document

    def _record_from_file(self, run_id: str, preparation: ManifestPreparation,
                          path: Path, *,
                          identity: TrainingRunIdentity | None) -> RunManifestRecord:
        payload = self._read_checked_bytes(path)
        document = self._decode_canonical(payload, preparation.manifest_sequence)
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != preparation.byte_length or digest != preparation.manifest_sha256:
            raise AuditTamperError("manifest bytes do not match authenticated preparation")
        if (
            identity is None
            or identity.run_id != run_id
            or document["schema"] != self._manifest_schema(identity)
        ):
            raise AuditTamperError("manifest schema differs from authenticated run identity")
        if (
            preparation.manifest_sequence == 0
            and document != self._initial_document(identity)
        ):
            raise AuditTamperError("initial manifest differs from authenticated run identity")
        if document["previous_manifest_sha256"] != preparation.previous_manifest_sha256:
            raise AuditTamperError("manifest predecessor differs from authenticated preparation")
        return RunManifestRecord(run_id, preparation.manifest_sequence, path,
                                 len(payload), digest,
                                 preparation.previous_manifest_sha256)

    @staticmethod
    def _preparations(events: tuple[ManifestAuditEvent, ...]):
        prepared: dict[str, ManifestAuditEvent] = {}
        terminal: dict[str, ManifestAuditEvent] = {}
        for event in events:
            if event.kind is ManifestEventKind.MANIFEST_PREPARED:
                prepared[event.event_id] = event
            elif event.prepared_event_id is not None:
                terminal.setdefault(event.prepared_event_id, event)
        return prepared, terminal

    @staticmethod
    def _preparation_from_event(event: ManifestAuditEvent) -> ManifestPreparation:
        if (event.kind is not ManifestEventKind.MANIFEST_PREPARED
                or event.manifest_sequence is None or event.basename is None
                or event.byte_length is None or event.manifest_sha256 is None):
            raise AuditTamperError("authenticated manifest preparation is incomplete")
        return ManifestPreparation(
            event.run_id, event.manifest_sequence, event.basename,
            event.byte_length, event.manifest_sha256,
            event.previous_manifest_sha256, event.created_at,
        )

    def verify_run_chain(self, run_id: str) -> tuple[RunManifestRecord, ...]:
        _run_id(run_id)
        identity = self.audit_store.training_run_identity(run_id)
        return self._verify_run_chain(run_id, identity=identity)

    def _verify_run_chain(
        self, run_id: str, *, identity: TrainingRunIdentity | None
    ) -> tuple[RunManifestRecord, ...]:
        # Reuse only the identity authenticated by this verification/append call.
        self._require_stable_root()
        events = self.audit_store.manifest_events(run_id)
        if len(events) > _MAX_EVENTS:
            raise AuditTamperError("run event count exceeds its bound")
        if any(event.kind in {ManifestEventKind.RECONCILIATION_BLOCKED,
                              ManifestEventKind.AUDIT_TAMPERED}
               for event in events):
            raise AuditTamperError("run has authenticated tamper evidence")
        prepared, terminal = self._preparations(events)
        published = [event for event_id, event in prepared.items()
                     if terminal.get(event_id) is not None
                     and terminal[event_id].kind is ManifestEventKind.MANIFEST_PUBLISHED]
        if any(event_id not in terminal for event_id in prepared):
            raise AuditTamperError("run has an unreconciled prepared manifest")
        published.sort(key=lambda item: int(item.manifest_sequence))
        if [item.manifest_sequence for item in published] != list(range(len(published))):
            raise AuditTamperError("published manifest sequence is not contiguous")
        directory = self._ensure_run_directory(run_id, known=True)
        directory_chain = _snapshot_chain(directory, require_target=True)
        entries, entries_exceeded = self._bounded_entries(
            directory, _MAX_CHILDREN
        )
        if (
            self._require_stable_root() != self._root_chain
            or _snapshot_chain(directory, require_target=True)
            != directory_chain
        ):
            raise PathRaceError("run directory changed during enumeration")
        if entries_exceeded:
            raise AuditTamperError("run directory child count exceeds its bound")
        allowed = {item.basename for item in published}
        if {entry.name for entry in entries} != allowed:
            raise AuditTamperError("run directory contains orphan or unknown children")
        records: list[RunManifestRecord] = []
        previous = None
        for event in published:
            preparation = self._preparation_from_event(event)
            if preparation.previous_manifest_sha256 != previous:
                raise AuditTamperError("authenticated manifest ancestry is broken")
            record = self._record_from_file(run_id, preparation,
                                            directory / preparation.basename,
                                            identity=identity)
            records.append(record)
            previous = record.sha256
        return tuple(records)

    @staticmethod
    def _evidence(*values: object) -> str:
        return hashlib.sha256(b"liveD:training-reconciliation:v1\0"
                              + _canonical_json(list(values))).hexdigest()

    @staticmethod
    def _add_result(results: list[ManifestReconciliation], *,
                    run_id: str | None, prepared_event_id: str | None,
                    outcome: ReconciliationOutcome, evidence: str,
                    appended: AuditAppendResult) -> None:
        if appended.created:
            results.append(ManifestReconciliation(
                run_id, prepared_event_id, outcome, appended.event_id, evidence
            ))

    def reconcile_pending(self) -> tuple[ManifestReconciliation, ...]:
        results: list[ManifestReconciliation] = []
        all_events = self.audit_store.all_manifest_events()
        if len(all_events) > _MAX_EVENTS:
            raise AuditTamperError("global training audit event count exceeds its bound")
        known_runs = {event.run_id for event in all_events}
        try:
            self._require_stable_root()
            entries, root_exceeded = self._bounded_entries(
                self._root, _MAX_RUNS
            )
            self._require_stable_root()
        except (FileNotFoundError, OSError, UnsafePathError, PathRaceError):
            evidence = self._evidence("training-root-unsafe")
            appended = self.audit_store.mark_root_anomaly(evidence)
            self._add_result(
                results,
                run_id=None,
                prepared_event_id=None,
                outcome=ReconciliationOutcome.RECONCILIATION_BLOCKED,
                evidence=evidence,
                appended=appended,
            )
            return tuple(results)
        if root_exceeded:
            evidence = self._evidence("root-entry-limit", _MAX_RUNS + 1)
            appended = self.audit_store.mark_root_anomaly(evidence)
            self._add_result(results, run_id=None, prepared_event_id=None,
                             outcome=ReconciliationOutcome.RECONCILIATION_BLOCKED,
                             evidence=evidence, appended=appended)
            return tuple(results)
        for entry in entries:
            if entry.name not in known_runs:
                evidence = self._evidence(
                    "unknown-root-child",
                    hashlib.sha256(entry.name.encode("utf-8")).hexdigest())
                appended = self.audit_store.mark_root_anomaly(evidence)
                self._add_result(results, run_id=None, prepared_event_id=None,
                                 outcome=ReconciliationOutcome.RECONCILIATION_BLOCKED,
                                 evidence=evidence, appended=appended)
        for run_id in sorted(known_runs):
            events = self.audit_store.manifest_events(run_id)
            identity = self.audit_store.training_run_identity(run_id)
            prepared, terminal = self._preparations(events)
            directory = self._run_directory(run_id)
            try:
                self._require_stable_root()
                absolute_directory = _absolute_lexical(directory)
                self._require_private_directory(absolute_directory)
                directory_chain = _snapshot_chain(
                    absolute_directory, require_target=True
                )
                children, children_exceeded = self._bounded_entries(
                    directory, _MAX_CHILDREN
                )
                if children_exceeded:
                    raise UnsafePathError("run directory child count exceeds bound")
                if (
                    self._require_stable_root() != self._root_chain
                    or _snapshot_chain(
                        absolute_directory, require_target=True
                    )
                    != directory_chain
                ):
                    raise PathRaceError(
                        "run directory changed during reconciliation"
                    )
            except (FileNotFoundError, OSError, UnsafePathError, PathRaceError):
                children = []
                directory_safe = False
            else:
                directory_safe = True
            expected = {event.basename for event in prepared.values()}
            if not directory_safe:
                directory_evidence = self._evidence(
                    "unsafe-run-directory", run_id
                )
                appended = self.audit_store.mark_run_anomaly(
                    run_id,
                    ReconciliationOutcome.RECONCILIATION_BLOCKED,
                    directory_evidence,
                )
                self._add_result(
                    results,
                    run_id=run_id,
                    prepared_event_id=None,
                    outcome=ReconciliationOutcome.RECONCILIATION_BLOCKED,
                    evidence=directory_evidence,
                    appended=appended,
                )
            if directory_safe:
                for child in children:
                    if child.name not in expected:
                        evidence = self._evidence(
                            "orphan-run-child", run_id,
                            hashlib.sha256(child.name.encode("utf-8")).hexdigest())
                        appended = self.audit_store.mark_run_anomaly(
                            run_id, ReconciliationOutcome.RECONCILIATION_BLOCKED,
                            evidence)
                        self._add_result(
                            results, run_id=run_id, prepared_event_id=None,
                            outcome=ReconciliationOutcome.RECONCILIATION_BLOCKED,
                            evidence=evidence, appended=appended)
            for prepared_id, event in prepared.items():
                outcome_event = terminal.get(prepared_id)
                path = directory / str(event.basename)
                if directory_safe:
                    try:
                        _checked_lstat(_absolute_lexical(path))
                    except FileNotFoundError:
                        exists = False
                    except (OSError, UnsafePathError, PathRaceError):
                        exists = True
                    else:
                        exists = True
                else:
                    exists = False
                evidence = self._evidence(
                    run_id, prepared_id, event.manifest_sha256,
                    event.byte_length, event.previous_manifest_sha256)
                if outcome_event is None:
                    if not directory_safe:
                        outcome = ReconciliationOutcome.RECONCILIATION_BLOCKED
                    elif not exists:
                        outcome = ReconciliationOutcome.ABORTED_BEFORE_PUBLICATION
                    else:
                        try:
                            self._record_from_file(
                                run_id, self._preparation_from_event(event), path,
                                identity=identity)
                        except (OSError, ValueError, UnsafePathError, PathRaceError, AuditTamperError):
                            outcome = ReconciliationOutcome.RECONCILIATION_BLOCKED
                        else:
                            outcome = ReconciliationOutcome.MANIFEST_PUBLISHED
                    appended = self.audit_store.mark_manifest_reconciled(
                        prepared_id, outcome)
                    self._add_result(
                        results, run_id=run_id, prepared_event_id=prepared_id,
                        outcome=outcome, evidence=evidence, appended=appended)
                elif outcome_event.kind is ManifestEventKind.MANIFEST_PUBLISHED:
                    try:
                        self._record_from_file(
                            run_id, self._preparation_from_event(event), path,
                            identity=identity)
                    except (OSError, ValueError, UnsafePathError, PathRaceError, AuditTamperError):
                        outcome = ReconciliationOutcome.AUDIT_TAMPERED
                        appended = self.audit_store.mark_manifest_reconciled(
                            prepared_id, outcome)
                        self._add_result(
                            results, run_id=run_id,
                            prepared_event_id=prepared_id, outcome=outcome,
                            evidence=evidence, appended=appended)
                elif (
                    exists
                    and outcome_event.kind
                    is ManifestEventKind.ABORTED_BEFORE_PUBLICATION
                    and not any(
                        other.event_id != prepared_id
                        and other.basename == event.basename
                        and terminal.get(other.event_id) is not None
                        and terminal[other.event_id].kind
                        is ManifestEventKind.MANIFEST_PUBLISHED
                        for other in prepared.values()
                    )
                ):
                    orphan_evidence = self._evidence(
                        "file-after-terminal", run_id, prepared_id)
                    appended = self.audit_store.mark_run_anomaly(
                        run_id, ReconciliationOutcome.RECONCILIATION_BLOCKED,
                        orphan_evidence)
                    self._add_result(
                        results, run_id=run_id, prepared_event_id=None,
                        outcome=ReconciliationOutcome.RECONCILIATION_BLOCKED,
                        evidence=orphan_evidence, appended=appended)
        return tuple(results)


__all__ = [
    "AuditAppendResult", "AuditTamperError", "ManifestAuditEvent",
    "ManifestEventKind", "ManifestGpuIdentity", "ManifestPreparation",
    "ManifestReconciliation", "PreparedManifestResult",
    "ReconciliationOutcome", "RunManifestRecord", "RunManifestWriter",
    "TrainingAuditStore", "TrainingRunIdentity", "TrainingRunRecord",
]
