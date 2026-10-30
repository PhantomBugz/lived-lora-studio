from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import hashlib
import json
import math
import re
from urllib.parse import unquote, urlsplit


_HEX32 = re.compile(r"[a-f0-9]{32}")
_HEX64 = re.compile(r"[a-f0-9]{64}")
_MAX_SEQUENCE = 9_223_372_036_854_775_807
_MAX_COUNT = 5_000_000
_MAX_DATASET_BYTES = 8_589_934_592
_MAX_GPU_BYTES = 1_125_899_906_842_624
_BIDI_CONTROL_RANGES = (
    (0x202A, 0x202E),
    (0x2066, 0x2069),
)
_BIDI_CONTROL_SINGLETONS = frozenset((0x061C, 0x200E, 0x200F))

# This digest identifies the one screening contract accepted by readiness and,
# in Task 5, by the outbound training screen. Hashing a domain-separated wire
# label keeps the identity stable without treating an arbitrary 64-hex value as
# proof that the current policy ran.
TRAINING_SCREENING_POLICY_SHA256 = hashlib.sha256(
    b"liveD:training-screening-policy:v1\x00screened-projections"
).hexdigest()


class TrainingMode(StrEnum):
    LORA = "lora"
    DPO = "dpo"
    RAG = "rag"


class RunState(StrEnum):
    PLANNED = "planned"
    STAGED = "staged"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class ReadinessState(StrEnum):
    DATA_BLOCKED = "DATA_BLOCKED"
    RUNTIME_BLOCKED = "RUNTIME_BLOCKED"
    ASSESSMENT_PENDING = "ASSESSMENT_PENDING"
    APPROACHING = "APPROACHING"
    READY = "READY"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingAction(StrEnum):
    START = "start"
    PAUSE = "pause"
    CANCEL = "cancel"


def _integer(value: int, name: str, *, minimum: int = 0,
             maximum: int = _MAX_SEQUENCE) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be from {minimum} through {maximum}")
    return value


def _boolean(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _finite(value: float | int, name: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} is below its minimum")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{name} exceeds its maximum")
    return numeric


def _text(value: str, name: str, *, maximum: int, strip: bool = True,
          ascii_only: bool = False) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    normalized = value.strip() if strip else value
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    if strip and normalized != value:
        raise ValueError(f"{name} must already be normalized")
    for character in normalized:
        codepoint = ord(character)
        bidi_control = (
            codepoint in _BIDI_CONTROL_SINGLETONS
            or any(
                lower <= codepoint <= upper
                for lower, upper in _BIDI_CONTROL_RANGES
            )
        )
        unsafe_control = (
            (codepoint < 0x20 and (strip or codepoint not in (0x09, 0x0A, 0x0D)))
            or 0x7F <= codepoint <= 0x9F
            or bidi_control
        )
        if unsafe_control:
            raise ValueError(
                f"{name} contains an unsafe control or bidi character"
            )
    try:
        encoded = normalized.encode("ascii" if ascii_only else "utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain ASCII characters only") from error
    if len(encoded) > maximum:
        raise ValueError(f"{name} exceeds its size limit")
    return normalized


def _https_url(value: str, name: str) -> str:
    try:
        normalized = _text(value, name, maximum=4096)
        decoded = unquote(normalized, errors="strict")
        if any(character.isspace() for character in decoded):
            raise ValueError("URL contains whitespace")
        for character in decoded:
            codepoint = ord(character)
            if (
                codepoint < 0x20
                or 0x7F <= codepoint <= 0x9F
                or codepoint in _BIDI_CONTROL_SINGLETONS
                or any(
                    lower <= codepoint <= upper
                    for lower, upper in _BIDI_CONTROL_RANGES
                )
            ):
                raise ValueError("URL contains unsafe control text")
        parsed = urlsplit(normalized)
        port = parsed.port
        host_port = parsed.netloc.rsplit("@", 1)[-1]
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or host_port.endswith(":")
            or (port is not None and not 1 <= port <= 65_535)
        ):
            raise ValueError("URL boundary is invalid")
    except TypeError:
        raise
    except (UnicodeError, ValueError) as error:
        raise ValueError(
            f"{name} must be a credential-free absolute HTTPS URL "
            "without unsafe text or an invalid port"
        ) from error
    return normalized


def _optional_text(value: str | None, name: str, *, maximum: int,
                   ascii_only: bool = False) -> str | None:
    if value is None:
        return None
    return _text(value, name, maximum=maximum, ascii_only=ascii_only)


def _sha256(value: str, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if _HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _opaque_id(value: str, name: str, *, maximum: int = 128) -> str:
    return _text(value, name, maximum=maximum, ascii_only=True)


def _utc(value: datetime, name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be UTC-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC-aware with a zero offset")
    return value


def _tuple(value: tuple[object, ...], name: str, *, maximum: int) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an immutable tuple")
    if len(value) > maximum:
        raise ValueError(f"{name} contains too many values")
    return value


def _string_tuple(value: tuple[str, ...], name: str, *, maximum_items: int,
                  maximum_bytes: int, allow_empty: bool = True,
                  ascii_only: bool = False) -> tuple[str, ...]:
    _tuple(value, name, maximum=maximum_items)
    if not allow_empty and not value:
        raise ValueError(f"{name} must not be empty")
    normalized = tuple(_text(item, f"{name} item", maximum=maximum_bytes,
                             ascii_only=ascii_only) for item in value)
    if normalized != value:
        raise ValueError(f"{name} values must already be normalized")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate identifiers")
    return value


def _identity_sha256(values: tuple[str, ...]) -> str:
    payload = (json.dumps(list(values), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _eligible_latest_revision_ids(
    pairs: tuple["PairRevisionView", ...],
    captures: tuple["CaptureTrainingView", ...],
) -> tuple[str, ...]:
    captures_by_id = {capture.capture_id: capture for capture in captures}
    latest: dict[str, PairRevisionView] = {}
    for pair in pairs:
        current = latest.get(pair.pair_id)
        if current is None or pair.revision > current.revision:
            latest[pair.pair_id] = pair
    return tuple(
        pair.revision_id
        for pair in pairs
        if latest[pair.pair_id].revision_id == pair.revision_id
        and pair.approved
        and not pair.quarantined
        and pair.capture_id in captures_by_id
        and captures_by_id[pair.capture_id].authenticated
        and not captures_by_id[pair.capture_id].quarantined
    )


@dataclass(frozen=True, slots=True)
class GpuInventory:
    uuid: str
    name: str
    memory_mib: int
    compute_capability: str

    def __post_init__(self) -> None:
        _text(self.uuid, "gpu uuid", maximum=128, ascii_only=True)
        _text(self.name, "gpu name", maximum=128)
        _integer(self.memory_mib, "gpu memory_mib", minimum=1,
                 maximum=_MAX_GPU_BYTES)
        _text(self.compute_capability, "compute capability", maximum=32,
              ascii_only=True)


@dataclass(frozen=True, slots=True)
class HardwareInventory:
    gpus: tuple[GpuInventory, ...]
    python_version: str
    cuda_version: str
    driver_version: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        _tuple(self.gpus, "gpus", maximum=8)
        if not self.gpus:
            raise ValueError("gpus must contain at least one GPU")
        if not all(type(gpu) is GpuInventory for gpu in self.gpus):
            raise TypeError("gpus must contain only GpuInventory records")
        if len({gpu.uuid for gpu in self.gpus}) != len(self.gpus):
            raise ValueError("gpus contains duplicate identifiers")
        _text(self.python_version, "python version", maximum=64, ascii_only=True)
        _text(self.cuda_version, "cuda version", maximum=64, ascii_only=True)
        _text(self.driver_version, "driver version", maximum=64, ascii_only=True)
        _sha256(self.runtime_sha256, "runtime_sha256")


@dataclass(frozen=True, slots=True)
class CpuExecutionInventory:
    python_version: str
    runtime_sha256: str

    def __post_init__(self) -> None:
        _text(self.python_version, "python version", maximum=64, ascii_only=True)
        _sha256(self.runtime_sha256, "runtime_sha256")


ExecutionHardwareInventory = HardwareInventory | CpuExecutionInventory


class RemoteReadinessState(StrEnum):
    UNCONFIGURED = "UNCONFIGURED"
    NOT_CHECKED = "NOT_CHECKED"
    VERIFIED = "VERIFIED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class RuntimeProfileReadiness:
    profile_id: str
    mode: TrainingMode

    def __post_init__(self) -> None:
        _opaque_id(self.profile_id, "profile_id", maximum=64)
        if type(self.mode) is not TrainingMode:
            raise TypeError("profile mode must be TrainingMode")
        if self.mode not in (TrainingMode.LORA, TrainingMode.DPO):
            raise ValueError("remote profiles support only LoRA and DPO")


@dataclass(frozen=True, slots=True)
class TrainingRecipeDescriptor:
    """Capabilities bound to one exact workload configuration, not an estimate."""

    profile_id: str
    mode: TrainingMode
    model_identity_sha256: str
    config_sha256: str
    backend: str
    platform: str
    minimum_devices: int
    maximum_devices: int
    parallelism: str
    precision_features: tuple[str, ...]
    minimum_device_memory_bytes: int | None
    reserved_device_memory_bytes: int | None

    def __post_init__(self) -> None:
        RuntimeProfileReadiness(self.profile_id, self.mode)
        _sha256(self.model_identity_sha256, "model_identity_sha256")
        _sha256(self.config_sha256, "config_sha256")
        for name in ("backend", "platform", "parallelism"):
            _text(getattr(self, name), name, maximum=128, ascii_only=True)
        _integer(self.minimum_devices, "minimum_devices", minimum=1, maximum=1024)
        _integer(self.maximum_devices, "maximum_devices", minimum=1, maximum=1024)
        if self.minimum_devices > self.maximum_devices:
            raise ValueError("device count bounds are inverted")
        _string_tuple(self.precision_features, "precision_features", maximum_items=32,
                      maximum_bytes=64, ascii_only=True)
        for name in ("minimum_device_memory_bytes", "reserved_device_memory_bytes"):
            value = getattr(self, name)
            if value is not None:
                _integer(value, name, maximum=_MAX_GPU_BYTES)

    @property
    def identity_sha256(self) -> str:
        document = {
            "schema": "LIVED_TRAINING_RECIPE_DESCRIPTOR_V1",
            "profile_id": self.profile_id,
            "mode": self.mode.value,
            "model_identity_sha256": self.model_identity_sha256,
            "config_sha256": self.config_sha256,
            "backend": self.backend,
            "platform": self.platform,
            "minimum_devices": self.minimum_devices,
            "maximum_devices": self.maximum_devices,
            "parallelism": self.parallelism,
            "precision_features": sorted(self.precision_features),
            "minimum_device_memory_bytes": self.minimum_device_memory_bytes,
            "reserved_device_memory_bytes": self.reserved_device_memory_bytes,
        }
        encoded = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedRemoteRuntimeReadiness:
    remote_runtime_sha256: str
    wheelhouse_manifest_sha256: str
    base_model_identity_sha256: str
    profiles: tuple[RuntimeProfileReadiness, ...]
    target_binding_sha256: str | None = None
    recipes: tuple[TrainingRecipeDescriptor, ...] = ()
    recipe_smoke_sha256s: tuple[tuple[str, str], ...] = ()
    # Authenticated sealed qualification snapshot, never a fresh observation.
    qualified_hardware: HardwareInventory | None = None

    def __post_init__(self) -> None:
        _sha256(self.remote_runtime_sha256, "remote_runtime_sha256")
        _sha256(self.wheelhouse_manifest_sha256, "wheelhouse_manifest_sha256")
        _sha256(self.base_model_identity_sha256, "base_model_identity_sha256")
        _tuple(self.profiles, "profiles", maximum=16)
        if not all(type(profile) is RuntimeProfileReadiness for profile in self.profiles):
            raise TypeError("profiles must contain RuntimeProfileReadiness records")
        profiles = {profile.profile_id: profile.mode for profile in self.profiles}
        if len(profiles) != len(self.profiles):
            raise ValueError("profiles contain duplicate identifiers")
        _tuple(self.recipes, "recipes", maximum=16)
        if not all(type(recipe) is TrainingRecipeDescriptor for recipe in self.recipes):
            raise TypeError("recipes must contain TrainingRecipeDescriptor records")
        _tuple(self.recipe_smoke_sha256s, "recipe_smoke_sha256s", maximum=16)
        smoke: dict[str, str] = {}
        for pair in self.recipe_smoke_sha256s:
            _tuple(pair, "recipe smoke pair", maximum=2)
            if len(pair) != 2:
                raise ValueError("recipe smoke pair requires profile ID and digest")
            profile_id, digest = pair
            _opaque_id(profile_id, "recipe smoke profile_id", maximum=64)
            _sha256(digest, "recipe smoke digest")
            if profile_id in smoke:
                raise ValueError("recipe smoke contains duplicate identifiers")
            smoke[profile_id] = digest
        if self.target_binding_sha256 is not None:
            _sha256(self.target_binding_sha256, "target_binding_sha256")
        if self.qualified_hardware is not None:
            if type(self.qualified_hardware) is not HardwareInventory:
                raise TypeError("qualified_hardware must be HardwareInventory")
            if self.qualified_hardware.runtime_sha256 != self.remote_runtime_sha256:
                raise ValueError("qualified hardware runtime identity differs")
        if (self.target_binding_sha256 is not None or self.recipes or smoke
                or self.qualified_hardware is not None):
            recipe_profiles = {recipe.profile_id: recipe.mode for recipe in self.recipes}
            if (
                self.target_binding_sha256 is None
                or not self.recipes
                or len(recipe_profiles) != len(self.recipes)
                or recipe_profiles != profiles
                or set(smoke) != set(recipe_profiles)
                or any(recipe.model_identity_sha256 != self.base_model_identity_sha256
                       for recipe in self.recipes)
            ):
                raise ValueError("recipe qualification requires matching target, profiles, model and smoke")


@dataclass(frozen=True, slots=True)
class RemoteRuntimeReadiness:
    state: RemoteReadinessState
    evidence: VerifiedRemoteRuntimeReadiness | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.state) is not RemoteReadinessState:
            raise TypeError("state must be RemoteReadinessState")
        _string_tuple(self.reasons, "remote reasons", maximum_items=32, maximum_bytes=512)
        if self.state is RemoteReadinessState.VERIFIED:
            if type(self.evidence) is not VerifiedRemoteRuntimeReadiness or self.reasons:
                raise ValueError("VERIFIED requires evidence and no blocking reasons")
        elif self.evidence is not None or not self.reasons:
            raise ValueError("unverified remote state requires absent evidence and nonempty reasons")


@dataclass(frozen=True, slots=True)
class RuntimeReadiness:
    local_embedding_sha256: str
    local_blocked_reasons: tuple[str, ...]
    remote: RemoteRuntimeReadiness

    def __post_init__(self) -> None:
        _sha256(self.local_embedding_sha256, "local_embedding_sha256")
        _string_tuple(
            self.local_blocked_reasons,
            "local_blocked_reasons",
            maximum_items=32,
            maximum_bytes=512,
        )
        if type(self.remote) is not RemoteRuntimeReadiness:
            raise TypeError("remote must be RemoteRuntimeReadiness")

    @property
    def remote_blocked_reasons(self) -> tuple[str, ...]:
        return self.remote.reasons

    @property
    def available_profile_ids(self) -> tuple[str, ...]:
        return (() if self.remote.evidence is None else
                tuple(profile.profile_id for profile in self.remote.evidence.profiles))

    def profiles_for(self, mode: TrainingMode) -> tuple[RuntimeProfileReadiness, ...]:
        if type(mode) is not TrainingMode:
            raise TypeError("mode must be TrainingMode")
        if self.remote.evidence is None:
            return ()
        return tuple(profile for profile in self.remote.evidence.profiles if profile.mode is mode)

    @property
    def blocked_reasons(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*self.local_blocked_reasons, *self.remote_blocked_reasons)
            )
        )


@dataclass(frozen=True, slots=True)
class TrainingThresholds:
    lora_pairs: int = 5_000
    dpo_samples: int = 2_000
    rag_documents: int = 500

    def __post_init__(self) -> None:
        _integer(self.lora_pairs, "lora_pairs", minimum=1, maximum=_MAX_COUNT)
        _integer(self.dpo_samples, "dpo_samples", minimum=1, maximum=_MAX_COUNT)
        _integer(self.rag_documents, "rag_documents", minimum=1, maximum=_MAX_COUNT)


@dataclass(frozen=True, slots=True)
class PairRevisionView:
    revision_id: str
    pair_id: str
    lineage_id: str
    parent_revision_id: str | None
    capture_id: str
    revision: int
    instruction: str
    response: str
    tags: tuple[str, ...]
    approved: bool
    quarantined: bool
    created_at: datetime
    event_sequence: int

    def __post_init__(self) -> None:
        _opaque_id(self.revision_id, "revision_id")
        _opaque_id(self.pair_id, "pair_id")
        _opaque_id(self.lineage_id, "lineage_id")
        _optional_text(self.parent_revision_id, "parent_revision_id", maximum=128,
                       ascii_only=True)
        _opaque_id(self.capture_id, "capture_id")
        _integer(self.revision, "revision", minimum=1, maximum=_MAX_COUNT)
        _text(self.instruction, "instruction", maximum=65_536, strip=False)
        _text(self.response, "response", maximum=262_144, strip=False)
        _string_tuple(self.tags, "tags", maximum_items=32, maximum_bytes=64)
        _boolean(self.approved, "approved")
        _boolean(self.quarantined, "quarantined")
        _utc(self.created_at, "created_at")
        _integer(self.event_sequence, "event_sequence", minimum=0)
        if self.parent_revision_id == self.revision_id:
            raise ValueError("parent_revision_id must not reference itself")


@dataclass(frozen=True, slots=True)
class DecisionView:
    event_id: str
    pair_id: str
    revision_id: str
    kind: str
    reason_code: str
    sequence: int
    created_at: datetime

    def __post_init__(self) -> None:
        _opaque_id(self.event_id, "event_id")
        _opaque_id(self.pair_id, "pair_id")
        _opaque_id(self.revision_id, "revision_id")
        if self.kind not in {"APPROVED", "DENIED", "MODIFIED"}:
            raise ValueError("decision kind must be APPROVED, DENIED, or MODIFIED")
        _text(self.reason_code, "reason_code", maximum=128, ascii_only=True)
        _integer(self.sequence, "sequence", minimum=0)
        _utc(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class CaptureTrainingView:
    capture_id: str
    content_sha256: str
    raw_text: str
    quality: float
    captured_at: datetime
    authenticated: bool
    quarantined: bool
    categories: tuple[str, ...]

    def __post_init__(self) -> None:
        _opaque_id(self.capture_id, "capture_id")
        _sha256(self.content_sha256, "content_sha256")
        _text(self.raw_text, "raw_text", maximum=4_000_000, strip=False)
        _finite(self.quality, "quality", minimum=0.0, maximum=1.0)
        _utc(self.captured_at, "captured_at")
        _boolean(self.authenticated, "authenticated")
        _boolean(self.quarantined, "quarantined")
        _string_tuple(self.categories, "categories", maximum_items=32,
                      maximum_bytes=64)
        actual = hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()
        if actual != self.content_sha256:
            raise ValueError("content_sha256 does not authenticate raw_text")


@dataclass(frozen=True, slots=True)
class SourceTargetView:
    target_id: str
    blueprint_id: str
    label: str
    start_url: str
    categories: tuple[str, ...]
    expected_yield: int
    operator_authorized: bool
    last_accepted_at: datetime | None

    def __post_init__(self) -> None:
        _opaque_id(self.target_id, "target_id")
        _text(self.blueprint_id, "blueprint_id", maximum=128, ascii_only=True)
        _text(self.label, "label", maximum=256)
        _https_url(self.start_url, "start_url")
        _string_tuple(self.categories, "categories", maximum_items=32,
                      maximum_bytes=64, allow_empty=False)
        _integer(self.expected_yield, "expected_yield", maximum=_MAX_COUNT)
        _boolean(self.operator_authorized, "operator_authorized")
        if self.last_accepted_at is not None:
            _utc(self.last_accepted_at, "last_accepted_at")


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    sequence: int
    eligible_identity_sha256: str
    pairs: tuple[PairRevisionView, ...]
    decisions: tuple[DecisionView, ...]
    captures: tuple[CaptureTrainingView, ...]
    targets: tuple[SourceTargetView, ...]

    def __post_init__(self) -> None:
        _integer(self.sequence, "snapshot sequence", minimum=0)
        _sha256(self.eligible_identity_sha256, "eligible_identity_sha256")
        _tuple(self.pairs, "pairs", maximum=_MAX_COUNT)
        _tuple(self.decisions, "decisions", maximum=_MAX_COUNT)
        _tuple(self.captures, "captures", maximum=_MAX_COUNT)
        _tuple(self.targets, "targets", maximum=10_000)
        if not all(type(value) is PairRevisionView for value in self.pairs):
            raise TypeError("pairs must contain only PairRevisionView records")
        if not all(type(value) is DecisionView for value in self.decisions):
            raise TypeError("decisions must contain only DecisionView records")
        if not all(type(value) is CaptureTrainingView for value in self.captures):
            raise TypeError("captures must contain only CaptureTrainingView records")
        if not all(type(value) is SourceTargetView for value in self.targets):
            raise TypeError("targets must contain only SourceTargetView records")
        for name, identities in (
            ("pair revision IDs", tuple(pair.revision_id for pair in self.pairs)),
            ("decision event IDs", tuple(event.event_id for event in self.decisions)),
            ("capture IDs", tuple(capture.capture_id for capture in self.captures)),
            ("target IDs", tuple(target.target_id for target in self.targets)),
        ):
            if len(set(identities)) != len(identities):
                raise ValueError(f"{name} contain duplicate identifiers")
        revisions = tuple((pair.pair_id, pair.revision) for pair in self.pairs)
        if len(set(revisions)) != len(revisions):
            raise ValueError("pair revisions contain duplicate revision numbers")
        capture_ids = {capture.capture_id for capture in self.captures}
        unresolved_captures = {
            pair.capture_id for pair in self.pairs
            if pair.capture_id not in capture_ids
        }
        if unresolved_captures:
            raise ValueError("pair revisions reference unresolved captures")
        revisions_by_id = {pair.revision_id: pair for pair in self.pairs}
        pairs_by_id: dict[str, list[PairRevisionView]] = {}
        for pair in self.pairs:
            pairs_by_id.setdefault(pair.pair_id, []).append(pair)
        for pair_id, history in pairs_by_id.items():
            ordered = sorted(history, key=lambda pair: pair.revision)
            if len({pair.lineage_id for pair in ordered}) != 1:
                raise ValueError(
                    f"pair {pair_id} contains mixed revision lineages"
                )
            for earlier, later in zip(ordered, ordered[1:]):
                if earlier.event_sequence >= later.event_sequence:
                    raise ValueError(
                        f"pair {pair_id} revision event order is inverted"
                    )
            for pair in ordered:
                if pair.parent_revision_id is None:
                    if pair.revision != ordered[0].revision:
                        raise ValueError(
                            f"pair {pair_id} revision parent is missing"
                        )
                    continue
                parent = revisions_by_id.get(pair.parent_revision_id)
                if (
                    parent is None
                    or parent.pair_id != pair.pair_id
                    or parent.lineage_id != pair.lineage_id
                    or parent.revision >= pair.revision
                    or parent.event_sequence >= pair.event_sequence
                ):
                    raise ValueError(
                        f"pair {pair_id} revision parent or lineage is incoherent"
                    )
        pair_sequences = tuple(pair.event_sequence for pair in self.pairs)
        decision_sequences = tuple(event.sequence for event in self.decisions)
        if any(left >= right for left, right in zip(pair_sequences,
                                                    pair_sequences[1:])):
            raise ValueError("pair snapshot sequences must increase monotonically")
        if any(left >= right for left, right in zip(decision_sequences,
                                                    decision_sequences[1:])):
            raise ValueError("decision snapshot sequences must increase monotonically")
        maximum_sequence = max((*pair_sequences, *decision_sequences, 0))
        if self.sequence < maximum_sequence:
            raise ValueError("snapshot sequence does not cover every record sequence")
        expected_identity = _identity_sha256(
            _eligible_latest_revision_ids(self.pairs, self.captures)
        )
        if self.eligible_identity_sha256 != expected_identity:
            raise ValueError(
                "eligible_identity_sha256 does not match eligible latest approvals"
            )


@dataclass(frozen=True, slots=True)
class ScreeningAssessment:
    mode: TrainingMode
    projection_schema: str
    snapshot_sequence: int
    input_record_ids: tuple[str, ...]
    accepted_record_ids: tuple[str, ...]
    input_identity_sha256: str
    accepted_identity_sha256: str
    input_count: int
    accepted_count: int
    rejected_count: int
    rejection_counts: tuple[tuple[str, int], ...]
    policy_sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        if type(self.mode) is not TrainingMode:
            raise TypeError("mode must be a TrainingMode")
        if self.projection_schema != f"LIVED_{self.mode.value.upper()}_V1":
            raise ValueError("projection schema does not match the training mode")
        _integer(self.snapshot_sequence, "snapshot_sequence", minimum=0)
        _string_tuple(self.input_record_ids, "input_record_ids",
                      maximum_items=_MAX_COUNT, maximum_bytes=128,
                      ascii_only=True)
        _string_tuple(self.accepted_record_ids, "accepted_record_ids",
                      maximum_items=_MAX_COUNT, maximum_bytes=128,
                      ascii_only=True)
        _sha256(self.input_identity_sha256, "input_identity_sha256")
        _sha256(self.accepted_identity_sha256, "accepted_identity_sha256")
        _integer(self.input_count, "input_count", maximum=_MAX_COUNT)
        _integer(self.accepted_count, "accepted_count", maximum=_MAX_COUNT)
        _integer(self.rejected_count, "rejected_count", maximum=_MAX_COUNT)
        _tuple(self.rejection_counts, "rejection_counts", maximum=64)
        reason_codes: list[str] = []
        for bucket in self.rejection_counts:
            if type(bucket) is not tuple or len(bucket) != 2:
                raise TypeError("rejection_counts entries must be immutable pairs")
            reason, count = bucket
            reason_codes.append(_text(reason, "rejection reason", maximum=128,
                                      ascii_only=True))
            _integer(count, "rejection count", maximum=_MAX_COUNT)
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("rejection_counts contains duplicate reason codes")
        _sha256(self.policy_sha256, "policy_sha256")
        _utc(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class EmbeddingAssessment:
    snapshot_sequence: int
    input_record_ids: tuple[str, ...]
    sampled_record_ids: tuple[str, ...]
    input_identity_sha256: str
    sampled_identity_sha256: str
    model_sha256: str
    input_count: int
    sampled_records: int
    diversity: float | None
    cache_sha256: str
    created_at: datetime

    def __post_init__(self) -> None:
        _integer(self.snapshot_sequence, "snapshot_sequence", minimum=0)
        _string_tuple(self.input_record_ids, "input_record_ids",
                      maximum_items=_MAX_COUNT, maximum_bytes=128,
                      ascii_only=True)
        _string_tuple(self.sampled_record_ids, "sampled_record_ids",
                      maximum_items=2_000, maximum_bytes=128,
                      ascii_only=True)
        _sha256(self.input_identity_sha256, "input_identity_sha256")
        _sha256(self.sampled_identity_sha256, "sampled_identity_sha256")
        _sha256(self.model_sha256, "model_sha256")
        _integer(self.input_count, "input_count", maximum=_MAX_COUNT)
        _integer(self.sampled_records, "sampled_records", maximum=2_000)
        if self.diversity is not None:
            _finite(self.diversity, "diversity", minimum=0.0, maximum=1.0)
        _sha256(self.cache_sha256, "cache_sha256")
        _utc(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class ModeReadiness:
    eligible: int
    required: int
    percentage: float
    visual_fraction: float
    quality: float | None = None
    diversity: float | None = None
    stale_records: int = 0
    temporal_decay_records: int = 0
    category_distribution: tuple[tuple[str, int], ...] = ()
    data_status: str = "READY"
    runtime_status: str = "READY"
    state: ReadinessState = ReadinessState.READY
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.eligible, "eligible", maximum=_MAX_COUNT)
        _integer(self.required, "required", minimum=1, maximum=_MAX_COUNT)
        expected_percentage = round((self.eligible * 100.0) / self.required, 1)
        if _finite(self.percentage, "percentage", minimum=0.0) != expected_percentage:
            raise ValueError("percentage does not match eligible and required counts")
        expected_fraction = min(1.0, self.eligible / self.required)
        if _finite(self.visual_fraction, "visual_fraction", minimum=0.0,
                   maximum=1.0) != expected_fraction:
            raise ValueError("visual_fraction does not match eligible and required counts")
        if self.quality is not None:
            _finite(self.quality, "quality", minimum=0.0, maximum=1.0)
        if self.diversity is not None:
            _finite(self.diversity, "diversity", minimum=0.0, maximum=1.0)
        _integer(self.stale_records, "stale_records", maximum=self.eligible)
        _integer(self.temporal_decay_records, "temporal_decay_records",
                 maximum=self.stale_records)
        _tuple(self.category_distribution, "category_distribution", maximum=256)
        categories: list[str] = []
        for bucket in self.category_distribution:
            if type(bucket) is not tuple or len(bucket) != 2:
                raise TypeError("category_distribution entries must be immutable pairs")
            category, count = bucket
            categories.append(_text(category, "category", maximum=64))
            _integer(count, "category count", maximum=self.eligible)
        if len(set(categories)) != len(categories):
            raise ValueError("category_distribution contains duplicate categories")
        _text(self.data_status, "data_status", maximum=64, ascii_only=True)
        _text(self.runtime_status, "runtime_status", maximum=64, ascii_only=True)
        if type(self.state) is not ReadinessState:
            raise TypeError("state must be a ReadinessState")
        _string_tuple(self.reasons, "reasons", maximum_items=32,
                      maximum_bytes=512)

    @classmethod
    def from_counts(cls, *, eligible: int, required: int) -> "ModeReadiness":
        _integer(eligible, "eligible", maximum=_MAX_COUNT)
        _integer(required, "required", minimum=1, maximum=_MAX_COUNT)
        raw = round((eligible * 100.0) / required, 1)
        state = (ReadinessState.DATA_BLOCKED if eligible == 0
                 else ReadinessState.READY if eligible >= required
                 else ReadinessState.APPROACHING)
        return cls(
            eligible=eligible,
            required=required,
            percentage=raw,
            visual_fraction=min(1.0, eligible / required),
            data_status=state.value,
            state=state,
            reasons=("No eligible records",) if eligible == 0 else (),
        )


@dataclass(frozen=True, slots=True)
class ActionableRecommendation:
    category: str
    deficit: int
    rationale: str
    target_id: str | None
    blueprint_id: str | None
    label: str
    start_url: str | None
    expected_yield: int

    def __post_init__(self) -> None:
        _text(self.category, "category", maximum=64)
        _integer(self.deficit, "deficit", minimum=1, maximum=_MAX_COUNT)
        _text(self.rationale, "rationale", maximum=2_048)
        _optional_text(self.target_id, "target_id", maximum=128, ascii_only=True)
        _optional_text(self.blueprint_id, "blueprint_id", maximum=128,
                       ascii_only=True)
        _text(self.label, "label", maximum=256)
        boundary = (
            self.target_id is not None,
            self.blueprint_id is not None,
            self.start_url is not None,
        )
        _integer(self.expected_yield, "expected_yield", maximum=_MAX_COUNT)
        if any(boundary) and not all(boundary):
            raise ValueError(
                "recommendation target boundary requires target_id, "
                "blueprint_id, and start_url together"
            )
        if not any(boundary) and self.expected_yield != 0:
            raise ValueError(
                "recommendation without a target must have zero expected_yield"
            )
        if self.start_url is not None:
            _https_url(self.start_url, "start_url")


@dataclass(frozen=True, slots=True)
class TrainingReadinessReport:
    generated_at: datetime
    lora: ModeReadiness
    dpo: ModeReadiness
    rag: ModeReadiness
    hardware_identity_sha256: str
    recommendations: tuple[ActionableRecommendation, ...] = ()

    def __post_init__(self) -> None:
        _utc(self.generated_at, "generated_at")
        if not all(type(value) is ModeReadiness
                   for value in (self.lora, self.dpo, self.rag)):
            raise TypeError("readiness modes must be ModeReadiness records")
        _sha256(self.hardware_identity_sha256, "hardware_identity_sha256")
        _tuple(self.recommendations, "recommendations", maximum=128)
        if not all(type(value) is ActionableRecommendation
                   for value in self.recommendations):
            raise TypeError("recommendations must contain only ActionableRecommendation records")


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    plan_id: str
    mode: TrainingMode
    snapshot_sequence: int
    dataset_count: int
    dataset_bytes: int
    dataset_sha256: str
    config_sha256: str
    model_identity_sha256: str
    engine_identity: str
    hardware_profile: str
    projection_schema: str
    created_at: datetime
    expires_at: datetime
    disclosure_summary: str
    dataset_manifest_sha256: str | None
    train_sha256: str | None
    eval_sha256: str | None
    parent_kind: str
    parent_identity_sha256: str | None

    def __post_init__(self) -> None:
        if _HEX32.fullmatch(self.plan_id) is None:
            raise ValueError("plan_id must be 32 lowercase hexadecimal characters")
        if type(self.mode) is not TrainingMode:
            raise TypeError("mode must be a TrainingMode")
        _integer(self.snapshot_sequence, "snapshot_sequence", minimum=0)
        _integer(self.dataset_count, "dataset_count", minimum=1,
                 maximum=_MAX_COUNT)
        _integer(self.dataset_bytes, "dataset_bytes", minimum=1,
                 maximum=_MAX_DATASET_BYTES)
        _sha256(self.dataset_sha256, "dataset_sha256")
        _sha256(self.config_sha256, "config_sha256")
        _sha256(self.model_identity_sha256, "model_identity_sha256")
        _text(self.engine_identity, "engine_identity", maximum=128,
              ascii_only=True)
        _text(self.hardware_profile, "hardware_profile", maximum=128,
              ascii_only=True)
        _text(self.projection_schema, "projection_schema", maximum=128,
              ascii_only=True)
        expected_schema = f"LIVED_{self.mode.value.upper()}_V1"
        if self.projection_schema != expected_schema:
            raise ValueError(
                "projection_schema does not match the training mode"
            )
        if self.mode is TrainingMode.RAG:
            if (
                self.engine_identity != "local-faiss"
                or self.hardware_profile != "local"
            ):
                raise ValueError(
                    "RAG engine and hardware profile must match its local mode"
                )
        else:
            mode_marker = f"_{self.mode.value}_"
            if (
                self.engine_identity != "pinned-remote"
                or mode_marker not in f"_{self.hardware_profile}_"
            ):
                raise ValueError(
                    "remote engine and hardware profile do not match the mode"
                )
        _utc(self.created_at, "created_at")
        _utc(self.expires_at, "expires_at")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        _text(self.disclosure_summary, "disclosure_summary", maximum=2_048)
        if self.mode is TrainingMode.RAG:
            if any(
                value is not None
                for value in (
                    self.dataset_manifest_sha256,
                    self.train_sha256,
                    self.eval_sha256,
                )
            ):
                raise ValueError("local RAG plan cannot bind remote dataset splits")
            if self.parent_kind != "NONE" or self.parent_identity_sha256 is not None:
                raise ValueError("local RAG plan cannot bind a training parent")
        else:
            for value, name in (
                (self.dataset_manifest_sha256, "dataset_manifest_sha256"),
                (self.train_sha256, "train_sha256"),
                (self.eval_sha256, "eval_sha256"),
            ):
                try:
                    _sha256(value, name)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"remote plan {name} is invalid") from error
            if self.mode is TrainingMode.LORA:
                if self.parent_kind != "NONE" or self.parent_identity_sha256 is not None:
                    raise ValueError("LoRA plan cannot bind a parent")
            elif self.parent_kind == "UNBOUND":
                if self.parent_identity_sha256 is not None:
                    raise ValueError("unbound DPO plan cannot name a parent")
            elif self.parent_kind in {"LORA", "BASE"}:
                try:
                    _sha256(self.parent_identity_sha256, "parent_identity_sha256")
                except (TypeError, ValueError) as error:
                    raise ValueError("DPO plan parent identity is invalid") from error
                if (
                    self.parent_kind == "BASE"
                    and self.parent_identity_sha256 != self.model_identity_sha256
                ):
                    raise ValueError(
                        "DPO raw-base parent must match the exact base model identity"
                    )
            else:
                raise ValueError("DPO plan parent kind is invalid")


@dataclass(frozen=True, slots=True)
class ApprovalGrant:
    action: TrainingAction
    subject_id: str
    dataset_sha256: str
    config_sha256: str
    nonce: str
    issued_at: datetime
    expires_at: datetime
    hmac_sha256: str
    parent_kind: str
    parent_identity_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.action) is not TrainingAction:
            raise TypeError("action must be a TrainingAction")
        _opaque_id(self.subject_id, "subject_id")
        _sha256(self.dataset_sha256, "dataset_sha256")
        _sha256(self.config_sha256, "config_sha256")
        _opaque_id(self.nonce, "nonce")
        _utc(self.issued_at, "issued_at")
        _utc(self.expires_at, "expires_at")
        _sha256(self.hmac_sha256, "hmac_sha256")
        if self.parent_kind in {"NONE", "UNBOUND"}:
            if self.parent_identity_sha256 is not None:
                raise ValueError("unbound approval parent cannot have an identity")
        elif self.parent_kind in {"LORA", "BASE"}:
            _sha256(self.parent_identity_sha256, "parent_identity_sha256")
        else:
            raise ValueError("approval parent kind is invalid")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    used_bytes: int
    total_bytes: int

    def __post_init__(self) -> None:
        _integer(self.used_bytes, "used_bytes", maximum=_MAX_GPU_BYTES)
        _integer(self.total_bytes, "total_bytes", minimum=1,
                 maximum=_MAX_GPU_BYTES)
        if self.used_bytes > self.total_bytes:
            raise ValueError("used_bytes must not exceed total_bytes")


def _validate_progress(*, sequence: int, state: RunState, phase: str,
                       completed_units: int, total_units: int,
                       epoch: float | None, loss: float | None,
                       learning_rate: float | None,
                       throughput: float | None, elapsed_seconds: float,
                       eta_seconds: float | None, gpu: GpuSnapshot | None,
                       checkpoint_sha256: str | None) -> None:
    _integer(sequence, "sequence", minimum=0)
    if type(state) is not RunState:
        raise TypeError("state must be a RunState")
    _text(phase, "phase", maximum=128, ascii_only=True)
    _integer(completed_units, "completed_units", maximum=_MAX_SEQUENCE)
    _integer(total_units, "total_units", maximum=_MAX_SEQUENCE)
    if completed_units > total_units:
        raise ValueError("completed_units must not exceed total_units")
    for value, name in ((epoch, "epoch"), (loss, "loss"),
                        (learning_rate, "learning_rate"),
                        (throughput, "throughput"),
                        (eta_seconds, "eta_seconds")):
        if value is not None:
            _finite(value, name, minimum=0.0)
    _finite(elapsed_seconds, "elapsed_seconds", minimum=0.0)
    if gpu is not None and type(gpu) is not GpuSnapshot:
        raise TypeError("gpu must be a GpuSnapshot or None")
    if checkpoint_sha256 is not None:
        _sha256(checkpoint_sha256, "checkpoint_sha256")


@dataclass(frozen=True, slots=True)
class TrainingRunSnapshot:
    run_id: str
    mode: TrainingMode
    state: RunState
    sequence: int
    phase: str
    completed_units: int
    total_units: int
    epoch: float | None
    loss: float | None
    learning_rate: float | None
    throughput: float | None
    elapsed_seconds: float
    eta_seconds: float | None
    gpu: GpuSnapshot | None
    last_checkpoint_sha256: str | None
    updated_at: datetime
    dataset_sha256: str
    config_sha256: str
    parent_kind: str
    parent_identity_sha256: str | None

    def __post_init__(self) -> None:
        _opaque_id(self.run_id, "run_id")
        if type(self.mode) is not TrainingMode:
            raise TypeError("mode must be a TrainingMode")
        _validate_progress(
            sequence=self.sequence, state=self.state, phase=self.phase,
            completed_units=self.completed_units, total_units=self.total_units,
            epoch=self.epoch, loss=self.loss,
            learning_rate=self.learning_rate, throughput=self.throughput,
            elapsed_seconds=self.elapsed_seconds, eta_seconds=self.eta_seconds,
            gpu=self.gpu, checkpoint_sha256=self.last_checkpoint_sha256,
        )
        _utc(self.updated_at, "updated_at")
        _sha256(self.dataset_sha256, "dataset_sha256")
        _sha256(self.config_sha256, "config_sha256")
        if self.mode in {TrainingMode.LORA, TrainingMode.RAG}:
            if self.parent_kind != "NONE" or self.parent_identity_sha256 is not None:
                raise ValueError("LoRA and RAG runs cannot bind a parent")
        elif self.parent_kind in {"LORA", "BASE"}:
            _sha256(self.parent_identity_sha256, "parent_identity_sha256")
        else:
            raise ValueError("DPO run parent kind is invalid")


@dataclass(frozen=True, slots=True)
class TrainingEvent:
    sequence: int
    state: RunState
    phase: str
    completed_units: int
    total_units: int
    recorded_at: datetime
    epoch: float | None = None
    loss: float | None = None
    learning_rate: float | None = None
    throughput: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    gpu: GpuSnapshot | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        _validate_progress(
            sequence=self.sequence, state=self.state, phase=self.phase,
            completed_units=self.completed_units, total_units=self.total_units,
            epoch=self.epoch, loss=self.loss,
            learning_rate=self.learning_rate, throughput=self.throughput,
            elapsed_seconds=self.elapsed_seconds, eta_seconds=self.eta_seconds,
            gpu=self.gpu, checkpoint_sha256=self.checkpoint_sha256,
        )
        _utc(self.recorded_at, "recorded_at")


__all__ = [
    "ActionableRecommendation",
    "ApprovalGrant",
    "CaptureTrainingView",
    "DecisionView",
    "EmbeddingAssessment",
    "GpuInventory",
    "GpuSnapshot",
    "HardwareInventory",
    "ModeReadiness",
    "PairRevisionView",
    "ReadinessSnapshot",
    "ReadinessState",
    "RunState",
    "RemoteReadinessState",
    "RemoteRuntimeReadiness",
    "RuntimeProfileReadiness",
    "RuntimeReadiness",
    "TrainingRecipeDescriptor",
    "VerifiedRemoteRuntimeReadiness",
    "ScreeningAssessment",
    "SourceTargetView",
    "TRAINING_SCREENING_POLICY_SHA256",
    "TrainingAction",
    "TrainingEvent",
    "TrainingMode",
    "TrainingPlan",
    "TrainingReadinessReport",
    "TrainingRunSnapshot",
    "TrainingThresholds",
]
