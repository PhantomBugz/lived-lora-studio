from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import base64
import binascii
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import stat
import threading
import unicodedata
from urllib.parse import unquote
from uuid import uuid4

from .models import OPERATOR_APPROVED, OPERATOR_REJECTED
from .paths import _windows_current_user_sid, _windows_sid_text
from .training_models import (
    DecisionView,
    PairRevisionView,
    ReadinessSnapshot,
    ScreeningAssessment,
    TRAINING_SCREENING_POLICY_SHA256,
    TrainingMode,
)


_MAX_TEXT_BYTES = 8_192
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_LOCAL_STATE_MARKERS = (
    "LIVED_LOCAL_SESSION_STATE_V1",
    "LIVED_LOCAL_ONLY_DPAPI_V1",
    "LIVED_LOCAL_ONLY_AEAD_V1",
    ".lived_state",
)
_BIDI_SINGLETONS = frozenset((0x061C, 0x200E, 0x200F))
_BIDI_RANGES = ((0x202A, 0x202E), (0x2066, 0x2069))
_CREDENTIAL_PATTERNS = (
    re.compile(
        r"(?ix)(?<![A-Za-z0-9_])(?P<key_quote>[\"'])?"
        r"[A-Za-z0-9_-]*(?:password|passwd|pwd|secret|token|"
        r"api[_-]?(?:key|token)|authorization|cookie|credential|session(?:[_-]?id)?)"
        r"[A-Za-z0-9_-]*"
        r"(?(key_quote)(?P=key_quote))\s*[:=]\s*"
        r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,}\]]+)"
    ),
    re.compile(r"(?i)\bBearer\s+\S+"),
    re.compile(r"(?i)-----BEGIN [^-]*(?:PRIVATE|SECRET) KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
)
_DRIVE_PATH = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]")
_UNC_PATH = re.compile(
    r"(?<![A-Za-z0-9_\\/])(?:\\{2,}|//)[^\\/\s]+(?:[\\/][^\\/\s]+)*"
)
_NT_DEVICE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_\\])\\(?:Device|GLOBALROOT|DosDevices|\?\?)\\"
)
_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9_.~/])/{1,}(?:[^\s/]+(?:/{1,}[^\s/]*)*)"
)
_ROOTED_WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9_\\])\\"
    r"(?!\\|[abfnrtv](?![A-Za-z0-9_\\/])|"
    r"u[0-9A-Fa-f]{4}(?![A-Za-z0-9_\\/])|"
    r"x[0-9A-Fa-f]{2}(?![A-Za-z0-9_\\/])|0(?![A-Za-z0-9_\\/]))"
    r"[^\\/\s]+(?:[\\/][^\\/\s]+)*"
)
_HOME_PATH = re.compile(r"(?<![A-Za-z0-9_])~(?:[A-Za-z0-9_.-]+)?[\\/]")
_URL = re.compile(r"(?i)(?<![A-Za-z0-9+.-])[A-Z][A-Z0-9+.-]*://\S+")
_ESCAPE = re.compile(r"\\(?:u([0-9a-fA-F]{4})|x([0-9a-fA-F]{2})|0(?![0-9]))")
_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{4,}={0,2}(?![A-Za-z0-9+/_-])")
_HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){2,}(?![0-9A-Fa-f])")


class TrainingDataError(RuntimeError):
    """A screened training dataset could not be safely produced or verified."""


class EgressRejectedError(TrainingDataError):
    def __init__(self, reason: str, message: str, *, fatal: bool = False):
        super().__init__(message)
        self.reason = reason
        self.fatal = fatal


class DatasetChangedError(TrainingDataError):
    """The owned dataset or its authenticated source snapshot changed."""


@dataclass(frozen=True, slots=True)
class DpoCandidate:
    sample_id: str
    parent: PairRevisionView
    child: PairRevisionView
    parent_decision: DecisionView
    child_decision: DecisionView


@dataclass(frozen=True, slots=True)
class PreparedTrainingDataset:
    mode: TrainingMode
    path: Path
    record_count: int
    byte_length: int
    sha256: str
    snapshot_sequence: int
    screening: ScreeningAssessment
    _capability: _OwnedDatasetCapability = field(repr=False)
    _source_binding_sha256: str = field(repr=False)
    _input_record_ids: tuple[str, ...] = field(repr=False)

    @property
    def _descriptor(self) -> int:
        return self._capability.fileno()

    @property
    def _cleaned(self) -> bool:
        return self._capability.closed

    def __enter__(self) -> PreparedTrainingDataset:
        if self._capability.closed:
            raise DatasetChangedError("prepared dataset was already cleaned")
        return self

    def __exit__(self, _kind: object, _error: object, _traceback: object) -> None:
        self.cleanup()

    def iter_transport_chunks(self, chunk_size: int = 65_536):
        """Stream verified bytes through the retained capability only."""
        return self._capability.iter_verified_chunks(
            self.byte_length, self.sha256, chunk_size=chunk_size
        )

    def cleanup(self) -> None:
        self._capability.cleanup()


class TrainingEgressScreen:
    """Apply the authenticated liveD scan and an independent residual scan."""

    __slots__ = ("__scrubber", "__memo")

    def __init__(self, scrubber: Callable[[str], str]):
        if not callable(scrubber):
            raise TypeError("scrubber must be callable")
        self.__scrubber = scrubber
        self.__memo: dict[str, str] = {}

    def screen_text(self, value: str) -> str:
        if type(value) is not str:
            raise EgressRejectedError("NON_STRING", "outbound text must be a string")
        if not value.strip():
            raise EgressRejectedError("EMPTY_TEXT", "outbound text must be nonempty")
        try:
            if len(value.encode("utf-8")) > _MAX_TEXT_BYTES:
                raise EgressRejectedError("TEXT_TOO_LARGE", "outbound text exceeds 8192 UTF-8 bytes")
        except UnicodeEncodeError as error:
            raise EgressRejectedError("INVALID_UTF8", "outbound text is not valid UTF-8") from error
        try:
            first = self.__scrubber(value)
            second = self.__scrubber(value)
        except Exception as error:
            raise EgressRejectedError(
                "SCRUBBER_EXCEPTION",
                "authenticated scrubber raised an exception",
                fatal=True,
            ) from error
        if type(first) is not str or type(second) is not str:
            raise EgressRejectedError(
                "SCRUBBER_INVALID_RESULT",
                "authenticated scrubber returned a non-string result",
                fatal=True,
            )
        if first != second:
            raise EgressRejectedError(
                "SCRUBBER_NONDETERMINISTIC",
                "authenticated scrubber must be deterministic",
                fatal=True,
            )
        previous = self.__memo.get(value)
        if previous is not None and previous != first:
            raise EgressRejectedError(
                "SCRUBBER_NONDETERMINISTIC",
                "authenticated scrubber must be deterministic for every repeated input",
                fatal=True,
            )
        if not first.strip():
            raise EgressRejectedError("SCRUBBED_EMPTY", "scrubbed text must be nonempty")
        try:
            if len(first.encode("utf-8")) > _MAX_TEXT_BYTES:
                raise EgressRejectedError(
                    "SCRUBBED_TOO_LARGE", "scrubbed text exceeds 8192 UTF-8 bytes"
                )
        except UnicodeEncodeError as error:
            raise EgressRejectedError("INVALID_UTF8", "scrubbed text is not valid UTF-8") from error
        _residual_scan(first)
        self.__memo.setdefault(value, first)
        return first


def _identity(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(list(values))).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise TrainingDataError("training record is not canonical JSON") from error


def _unsafe_control(value: str) -> bool:
    for character in value:
        point = ord(character)
        if (
            point == 0
            or (point < 0x20 and character not in "\n\r\t")
            or 0x7F <= point <= 0x9F
            or point in _BIDI_SINGLETONS
            or any(lower <= point <= upper for lower, upper in _BIDI_RANGES)
        ):
            return True
    return False


def _decode_escapes(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        number = match.group(1) or match.group(2)
        if number is None:
            return "\x00"
        return chr(int(number, 16))

    return _ESCAPE.sub(replace, value)


def _base64_variants(value: str) -> tuple[str, ...]:
    decoded: list[str] = []
    for match in _BASE64_TOKEN.finditer(value):
        token = match.group(0)
        if len(token) > 16_384 or len(token) % 4 == 1:
            continue
        unpadded = token.rstrip("=")
        padded = unpadded + "=" * ((4 - len(unpadded) % 4) % 4)
        for altchars in (None, b"-_"):
            try:
                raw = base64.b64decode(
                    padded.encode("ascii"), altchars=altchars, validate=True
                )
                canonical = (
                    base64.b64encode(raw, altchars=altchars)
                    .decode("ascii")
                    .rstrip("=")
                )
                if canonical != unpadded:
                    continue
                decoded_text = raw.decode("utf-8")
            except (ValueError, UnicodeError, binascii.Error):
                continue
            if decoded_text and decoded_text not in decoded:
                decoded.append(decoded_text)
    return tuple(decoded)


def _hex_variants(value: str) -> tuple[str, ...]:
    decoded: list[str] = []
    for match in _HEX_TOKEN.finditer(value):
        token = match.group(0)
        if len(token) > 16_384:
            continue
        try:
            decoded_text = bytes.fromhex(token).decode("utf-8")
        except (ValueError, UnicodeError):
            continue
        if decoded_text and decoded_text not in decoded:
            decoded.append(decoded_text)
    return tuple(decoded)


def _direct_decoded_variants(value: str) -> tuple[str, ...]:
    return (
        unquote(value),
        html.unescape(value),
        _decode_escapes(value),
        *_base64_variants(value),
        *_hex_variants(value),
    )


def _decoded_variants(value: str) -> tuple[str, ...]:
    queue = [value]
    ordered = [value]
    seen = {value}
    for _round in range(3):
        current = tuple(queue)
        queue.clear()
        for item in current:
            for candidate in _direct_decoded_variants(item):
                if candidate not in seen:
                    if len(ordered) >= 64:
                        raise EgressRejectedError(
                            "ENCODED_VARIANT_LIMIT",
                            "outbound text expands into too many encoded variants",
                        )
                    seen.add(candidate)
                    ordered.append(candidate)
                    queue.append(candidate)
    for frontier in queue:
        if any(
            candidate not in seen
            for candidate in _direct_decoded_variants(frontier)
        ):
            raise EgressRejectedError(
                "ENCODED_NESTING_LIMIT",
                "outbound text exceeds the encoded nesting limit",
            )
    return tuple(ordered)


def _reject_identifiers(value: str, identifiers: tuple[str, ...]) -> None:
    normalized = tuple(
        identifier.casefold() for identifier in identifiers if identifier
    )
    for variant in _decoded_variants(value):
        folded = variant.casefold()
        if any(identifier in folded for identifier in normalized):
            raise EgressRejectedError(
                "LOCAL_IDENTIFIER", "outbound text contains a local identifier"
            )


def _reject_pair_identifiers(value: str, pair: PairRevisionView) -> None:
    _reject_identifiers(
        value,
        tuple(
            identifier
            for identifier in (
                pair.revision_id,
                pair.pair_id,
                pair.lineage_id,
                pair.capture_id,
                pair.parent_revision_id,
            )
            if identifier is not None
        ),
    )


def _residual_scan(value: str) -> None:
    for variant in _decoded_variants(value):
        folded = variant.casefold()
        if _unsafe_control(variant):
            raise EgressRejectedError("RESIDUAL_CONTROL", "outbound text contains unsafe controls")
        if any(marker.casefold() in folded for marker in _LOCAL_STATE_MARKERS):
            raise EgressRejectedError("LOCAL_STATE_MARKER", "local state must remain on this host")
        if any(pattern.search(variant) for pattern in _CREDENTIAL_PATTERNS):
            raise EgressRejectedError("RESIDUAL_CREDENTIAL", "outbound text contains credential material")
        if _URL.search(variant):
            raise EgressRejectedError("RESIDUAL_URL", "outbound text contains a source URL")
        if (
            _DRIVE_PATH.search(variant)
            or _UNC_PATH.search(variant)
            or _NT_DEVICE_PATH.search(variant)
            or _POSIX_PATH.search(variant)
            or _ROOTED_WINDOWS_PATH.search(variant)
            or _HOME_PATH.search(variant)
        ):
            raise EgressRejectedError("RESIDUAL_PATH", "outbound text contains an absolute local path")


def _screen_tags(tags: tuple[str, ...], screen: TrainingEgressScreen) -> tuple[str, ...]:
    if type(tags) is not tuple:
        raise EgressRejectedError("TAG_INVALID", "tags must be an immutable tuple")
    result: list[str] = []
    for tag in tags:
        if type(tag) is not str or not tag:
            raise EgressRejectedError("TAG_INVALID", "tags must be nonempty strings")
        if (
            tag != tag.strip()
            or tag != tag.casefold()
            or tag != unicodedata.normalize("NFKC", tag)
            or tag in result
        ):
            raise EgressRejectedError("TAG_INVALID", "tags must be normalized, casefolded, trimmed, and unique")
        screened = screen.screen_text(tag)
        if screened != tag:
            raise EgressRejectedError("TAG_CHANGED", "screening must not rewrite operator-approved tags")
        result.append(tag)
    return tuple(result)


def serialize_lora(pair: PairRevisionView, screen: TrainingEgressScreen) -> bytes:
    if type(pair) is not PairRevisionView:
        raise TypeError("pair must be a PairRevisionView")
    if type(screen) is not TrainingEgressScreen:
        raise TypeError("screen must be a TrainingEgressScreen")
    for raw_value in (pair.instruction, pair.response, *pair.tags):
        _reject_pair_identifiers(raw_value, pair)
    instruction = screen.screen_text(pair.instruction)
    response = screen.screen_text(pair.response)
    tags = _screen_tags(pair.tags, screen)
    line = _canonical_json(
        {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "tags": list(tags),
        }
    )
    _reject_pair_identifiers(line.decode("utf-8"), pair)
    return line


def select_lora_pairs(snapshot: ReadinessSnapshot) -> tuple[PairRevisionView, ...]:
    if type(snapshot) is not ReadinessSnapshot:
        raise TypeError("snapshot must be a ReadinessSnapshot")
    captures = {capture.capture_id: capture for capture in snapshot.captures}
    latest: dict[str, PairRevisionView] = {}
    for pair in snapshot.pairs:
        previous = latest.get(pair.pair_id)
        if previous is None or pair.revision > previous.revision:
            latest[pair.pair_id] = pair
    selected: list[PairRevisionView] = []
    for pair in snapshot.pairs:
        capture = captures.get(pair.capture_id)
        if (
            latest[pair.pair_id].revision_id == pair.revision_id
            and pair.approved
            and not pair.quarantined
            and capture is not None
            and capture.authenticated
            and not capture.quarantined
        ):
            selected.append(pair)
    return tuple(selected)


def dpo_sample_id(parent: PairRevisionView, child: PairRevisionView) -> str:
    return _identity((parent.revision_id, child.revision_id))


def select_dpo_candidates(snapshot: ReadinessSnapshot) -> tuple[DpoCandidate, ...]:
    if type(snapshot) is not ReadinessSnapshot:
        raise TypeError("snapshot must be a ReadinessSnapshot")
    by_revision = {pair.revision_id: pair for pair in snapshot.pairs}
    latest_decision: dict[str, DecisionView] = {}
    for decision in snapshot.decisions:
        current = latest_decision.get(decision.revision_id)
        if current is None or decision.sequence > current.sequence:
            latest_decision[decision.revision_id] = decision
    selected: list[DpoCandidate] = []
    for child in select_lora_pairs(snapshot):
        if child.parent_revision_id is None:
            continue
        parent = by_revision.get(child.parent_revision_id)
        parent_decision = latest_decision.get(child.parent_revision_id)
        child_decision = latest_decision.get(child.revision_id)
        if (
            parent is None
            or parent.approved
            or parent.quarantined
            or parent.pair_id != child.pair_id
            or parent.lineage_id != child.lineage_id
            or parent.capture_id != child.capture_id
            or parent_decision is None
            or child_decision is None
            or parent_decision.pair_id != parent.pair_id
            or parent_decision.revision_id != parent.revision_id
            or parent_decision.kind != "DENIED"
            or parent_decision.reason_code not in {
                OPERATOR_REJECTED,
                "hallucinated_claim",
                "wrong_api",
                "stale_technique",
                "social_noise",
                "other",
            }
            or child_decision.pair_id != child.pair_id
            or child_decision.revision_id != child.revision_id
            or child_decision.kind != "APPROVED"
            or child_decision.reason_code != OPERATOR_APPROVED
            or not (
                parent.event_sequence
                < parent_decision.sequence
                < child.event_sequence
                < child_decision.sequence
            )
        ):
            continue
        selected.append(
            DpoCandidate(
                sample_id=dpo_sample_id(parent, child),
                parent=parent,
                child=child,
                parent_decision=parent_decision,
                child_decision=child_decision,
            )
        )
    return tuple(selected)


def _pair_binding(pair: PairRevisionView) -> dict[str, object]:
    return {
        "revision_id": pair.revision_id,
        "pair_id": pair.pair_id,
        "lineage_id": pair.lineage_id,
        "parent_revision_id": pair.parent_revision_id,
        "capture_id": pair.capture_id,
        "revision": pair.revision,
        "instruction": pair.instruction,
        "response": pair.response,
        "tags": list(pair.tags),
        "approved": pair.approved,
        "quarantined": pair.quarantined,
        "created_at": pair.created_at.isoformat(),
        "event_sequence": pair.event_sequence,
    }


def _decision_binding(decision: DecisionView) -> dict[str, object]:
    return {
        "event_id": decision.event_id,
        "pair_id": decision.pair_id,
        "revision_id": decision.revision_id,
        "kind": decision.kind,
        "reason_code": decision.reason_code,
        "sequence": decision.sequence,
        "created_at": decision.created_at.isoformat(),
    }


def _capture_binding(snapshot: ReadinessSnapshot, capture_id: str) -> dict[str, object]:
    capture = next(item for item in snapshot.captures if item.capture_id == capture_id)
    return {
        "capture_id": capture.capture_id,
        "content_sha256": capture.content_sha256,
        "quality": capture.quality,
        "captured_at": capture.captured_at.isoformat(),
        "authenticated": capture.authenticated,
        "quarantined": capture.quarantined,
        "categories": list(capture.categories),
    }


def _selection(snapshot: ReadinessSnapshot, mode: TrainingMode):
    if mode is TrainingMode.LORA:
        pairs = select_lora_pairs(snapshot)
        ids = tuple(pair.revision_id for pair in pairs)
        raw = [
            {"pair": _pair_binding(pair), "capture": _capture_binding(snapshot, pair.capture_id)}
            for pair in pairs
        ]
        return pairs, ids, hashlib.sha256(_canonical_json(raw)).hexdigest()
    if mode is TrainingMode.DPO:
        candidates = select_dpo_candidates(snapshot)
        ids = tuple(candidate.sample_id for candidate in candidates)
        raw = [
            {
                "parent": _pair_binding(candidate.parent),
                "child": _pair_binding(candidate.child),
                "parent_decision": _decision_binding(candidate.parent_decision),
                "child_decision": _decision_binding(candidate.child_decision),
                "capture": _capture_binding(snapshot, candidate.child.capture_id),
            }
            for candidate in candidates
        ]
        return candidates, ids, hashlib.sha256(_canonical_json(raw)).hexdigest()
    raise ValueError("training projections support only lora or dpo")


def _reject_local_identifiers(value: str, snapshot: ReadinessSnapshot) -> None:
    identifiers = {
        *(pair.revision_id for pair in snapshot.pairs),
        *(pair.pair_id for pair in snapshot.pairs),
        *(pair.lineage_id for pair in snapshot.pairs),
        *(pair.capture_id for pair in snapshot.pairs),
        *(decision.event_id for decision in snapshot.decisions),
        *(capture.content_sha256 for capture in snapshot.captures),
        *(target.target_id for target in snapshot.targets),
        snapshot.eligible_identity_sha256,
    }
    _reject_identifiers(value, tuple(identifiers))


def _lora_line(pair: PairRevisionView, screen: TrainingEgressScreen, snapshot: ReadinessSnapshot) -> bytes:
    for raw_value in (pair.instruction, pair.response, *pair.tags):
        _reject_local_identifiers(raw_value, snapshot)
    line = serialize_lora(pair, screen)
    value = line.decode("utf-8")
    _reject_local_identifiers(value, snapshot)
    return line


def _dpo_line(candidate: DpoCandidate, screen: TrainingEgressScreen, snapshot: ReadinessSnapshot) -> bytes:
    chosen_prompt = screen.screen_text(candidate.child.instruction)
    rejected_prompt = screen.screen_text(candidate.parent.instruction)
    chosen = screen.screen_text(candidate.child.response)
    rejected = screen.screen_text(candidate.parent.response)
    reason_code = screen.screen_text(candidate.parent_decision.reason_code)
    if reason_code != candidate.parent_decision.reason_code:
        raise EgressRejectedError(
            "DPO_REASON_CHANGED",
            "screening must not rewrite the authenticated DPO decision reason",
        )

    if candidate.child.instruction != candidate.parent.instruction:
        changed = (
            chosen_prompt != candidate.child.instruction
            or rejected_prompt != candidate.parent.instruction
        )
        if changed and chosen_prompt != rejected_prompt:
            raise EgressRejectedError(
                "DPO_PROMPT_SCRUB_MISMATCH",
                "DPO prompts remain unequal after asymmetric screening",
            )
        raise EgressRejectedError("DPO_PROMPT_CHANGED", "DPO revision changed its prompt")
    if chosen_prompt != rejected_prompt:
        raise EgressRejectedError(
            "DPO_PROMPT_SCRUB_MISMATCH", "DPO prompts differ after screening"
        )
    if chosen == rejected:
        raise EgressRejectedError(
            "DPO_RESPONSE_SCRUB_COLLISION", "DPO responses collide after screening"
        )
    line = _canonical_json(
        {
            "prompt": chosen_prompt,
            "chosen": chosen,
            "rejected": rejected,
            "reason_code": reason_code,
        }
    )
    _reject_local_identifiers(line.decode("utf-8"), snapshot)
    return line


@dataclass(frozen=True, slots=True)
class _HandleState:
    identity: tuple[int, object]
    size: int
    links: int
    delete_pending: bool
    directory: bool
    reparse: bool


def _close_owned_stream(stream: io.RawIOBase) -> None:
    stream.close()


def _close_owned_guard(guard: int, *, windows: bool) -> None:
    if windows:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        if not kernel.CloseHandle(wintypes.HANDLE(guard)):
            raise OSError(ctypes.get_last_error(), "owned directory handle close failed")
    else:
        os.close(guard)


def _windows_handle_state(handle: int) -> _HandleState:
    import ctypes
    from ctypes import wintypes

    class FileId128(ctypes.Structure):
        _fields_ = (("identifier", ctypes.c_ubyte * 16),)

    class FileIdInfo(ctypes.Structure):
        _fields_ = (
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", FileId128),
        )

    class FileStandardInfo(ctypes.Structure):
        _fields_ = (
            ("allocation_size", ctypes.c_longlong),
            ("end_of_file", ctypes.c_longlong),
            ("number_of_links", wintypes.DWORD),
            ("delete_pending", wintypes.BOOLEAN),
            ("directory", wintypes.BOOLEAN),
        )

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = (
            ("file_attributes", wintypes.DWORD),
            ("reparse_tag", wintypes.DWORD),
        )

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetFileInformationByHandleEx.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL

    def query(info_class: int, structure):
        value = structure()
        if not kernel.GetFileInformationByHandleEx(
            wintypes.HANDLE(handle),
            info_class,
            ctypes.byref(value),
            ctypes.sizeof(value),
        ):
            raise OSError(
                ctypes.get_last_error(),
                "owned Windows handle metadata query failed",
            )
        return value

    file_id = query(18, FileIdInfo)
    standard = query(1, FileStandardInfo)
    attributes = query(9, FileAttributeTagInfo)
    return _HandleState(
        identity=(
            int(file_id.volume_serial_number),
            bytes(file_id.file_id.identifier),
        ),
        size=int(standard.end_of_file),
        links=int(standard.number_of_links),
        delete_pending=bool(standard.delete_pending),
        directory=bool(standard.directory),
        reparse=bool(int(attributes.file_attributes) & 0x400),
    )


def _mark_windows_delete_pending(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    class FileDispositionInfo(ctypes.Structure):
        _fields_ = (("delete_file", wintypes.BOOLEAN),)

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.SetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel.SetFileInformationByHandle.restype = wintypes.BOOL
    disposition = FileDispositionInfo(True)
    if not kernel.SetFileInformationByHandle(
        wintypes.HANDLE(handle),
        4,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise OSError(
            ctypes.get_last_error(),
            "private training handle could not be marked delete-pending",
        )


def _set_private_windows_handle_acl(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    class Trustee(ctypes.Structure):
        _fields_ = (
            ("multiple", ctypes.c_void_p),
            ("operation", ctypes.c_int),
            ("form", ctypes.c_int),
            ("kind", ctypes.c_int),
            ("name", ctypes.c_void_p),
        )

    class ExplicitAccess(ctypes.Structure):
        _fields_ = (
            ("permissions", wintypes.DWORD),
            ("access_mode", ctypes.c_int),
            ("inheritance", wintypes.DWORD),
            ("trustee", Trustee),
        )

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    advapi.ConvertStringSidToSidW.argtypes = (
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.ConvertStringSidToSidW.restype = wintypes.BOOL
    advapi.SetEntriesInAclW.argtypes = (
        wintypes.ULONG,
        ctypes.POINTER(ExplicitAccess),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.SetEntriesInAclW.restype = wintypes.DWORD
    advapi.SetSecurityInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    advapi.SetSecurityInfo.restype = wintypes.DWORD

    allocated: list[ctypes.c_void_p] = []
    acl = ctypes.c_void_p()
    try:
        entries = (ExplicitAccess * 2)()
        for index, sid_text in enumerate(
            (_windows_current_user_sid(), "S-1-5-18")
        ):
            sid = ctypes.c_void_p()
            if not advapi.ConvertStringSidToSidW(sid_text, ctypes.byref(sid)):
                raise TrainingDataError("private Windows SID could not be created")
            allocated.append(sid)
            entries[index] = ExplicitAccess(
                0x001F01FF,
                2,
                0,
                Trustee(None, 0, 0, 1, sid),
            )
        result = advapi.SetEntriesInAclW(2, entries, None, ctypes.byref(acl))
        if result != 0:
            raise TrainingDataError("private Windows handle ACL could not be built")
        result = advapi.SetSecurityInfo(
            wintypes.HANDLE(handle),
            1,
            0x00000004 | 0x80000000,
            None,
            None,
            acl,
            None,
        )
        if result != 0:
            raise TrainingDataError("private Windows handle ACL could not be applied")
    finally:
        if acl:
            kernel.LocalFree(acl)
        for sid in allocated:
            kernel.LocalFree(sid)


def _require_private_windows_handle_acl(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    class AclSizeInformation(ctypes.Structure):
        _fields_ = (
            ("ace_count", wintypes.DWORD),
            ("bytes_in_use", wintypes.DWORD),
            ("bytes_free", wintypes.DWORD),
        )

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    advapi.GetSecurityInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.GetSecurityInfo.restype = wintypes.DWORD
    advapi.GetSecurityDescriptorControl.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL
    advapi.GetAclInformation.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_int,
    )
    advapi.GetAclInformation.restype = wintypes.BOOL
    advapi.GetAce.argtypes = (
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.GetAce.restype = wintypes.BOOL

    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    result = advapi.GetSecurityInfo(
        wintypes.HANDLE(handle),
        1,
        0x00000001 | 0x00000004,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if result != 0 or not owner or not dacl or not descriptor:
        raise TrainingDataError("private Windows handle ACL could not be read")
    try:
        current_sid = _windows_current_user_sid()
        if _windows_sid_text(int(owner.value)) != current_sid:
            raise TrainingDataError("private Windows handle owner is unsafe")
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if (
            not advapi.GetSecurityDescriptorControl(
                descriptor, ctypes.byref(control), ctypes.byref(revision)
            )
            or not (control.value & 0x1000)
        ):
            raise TrainingDataError("private Windows handle ACL is not protected")
        info = AclSizeInformation()
        if not advapi.GetAclInformation(
            dacl, ctypes.byref(info), ctypes.sizeof(info), 2
        ):
            raise TrainingDataError("private Windows handle ACL could not be enumerated")
        allowed = {current_sid, "S-1-5-18"}
        seen: set[str] = set()
        if info.ace_count != 2:
            raise TrainingDataError("private Windows handle ACL has unexpected identities")
        for index in range(info.ace_count):
            ace = ctypes.c_void_p()
            if not advapi.GetAce(dacl, index, ctypes.byref(ace)):
                raise TrainingDataError("private Windows handle ACL entry could not be read")
            address = int(ace.value)
            ace_type = ctypes.c_ubyte.from_address(address).value
            ace_size = ctypes.c_ushort.from_address(address + 2).value
            mask = ctypes.c_uint32.from_address(address + 4).value
            sid = _windows_sid_text(address + 8)
            if (
                ace_type != 0
                or ace_size < 12
                or sid not in allowed
                or mask & 0x001F01FF != 0x001F01FF
                or sid in seen
            ):
                raise TrainingDataError("private Windows handle ACL entry is unsafe")
            seen.add(sid)
        if seen != allowed:
            raise TrainingDataError("private Windows handle ACL is incomplete")
    finally:
        kernel.LocalFree(descriptor)


def _open_windows_root_guards(root: Path) -> tuple[int, ...]:
    import ctypes
    from ctypes import wintypes

    if not root.is_absolute() or not re.fullmatch(r"[A-Za-z]:", root.drive):
        raise TrainingDataError("training staging root must be on a local Windows drive")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    invalid = ctypes.c_void_p(-1).value
    chain: list[Path] = [Path(root.anchor)]
    current = Path(root.anchor)
    for component in root.parts[1:]:
        current = current / component
        chain.append(current)

    guards: list[int] = []
    try:
        for component in chain:
            handle = kernel.CreateFileW(
                os.fspath(component),
                0x00000080 | 0x00020000,
                0x00000001 | 0x00000002,
                None,
                3,
                0x02000000 | 0x00200000,
                None,
            )
            if int(handle) == invalid:
                raise OSError(
                    ctypes.get_last_error(),
                    "training staging ancestry could not be locked",
                )
            numeric = int(handle)
            state = _windows_handle_state(numeric)
            if not state.directory or state.reparse:
                kernel.CloseHandle(wintypes.HANDLE(numeric))
                raise TrainingDataError("training staging ancestry is not a real directory")
            guards.append(numeric)
        _require_private_windows_handle_acl(guards[-1])
        return tuple(guards)
    except BaseException:
        for guard in reversed(guards):
            try:
                _close_owned_guard(guard, windows=True)
            except OSError:
                pass
        raise


class _OwnedDatasetCapability:
    __slots__ = (
        "_stream",
        "_guards",
        "_windows",
        "_root_identity",
        "_file_identity",
        "_expected_size",
        "_closed",
        "_broken",
        "_lock",
    )

    def __init__(
        self,
        *,
        stream: io.RawIOBase,
        guards: tuple[int, ...],
        windows: bool,
        root_identity: tuple[int, object],
        file_identity: tuple[int, object],
    ):
        self._stream = stream
        self._guards = list(guards)
        self._windows = windows
        self._root_identity = root_identity
        self._file_identity = file_identity
        self._expected_size: int | None = None
        self._closed = False
        self._broken = False
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        return self._closed

    def fileno(self) -> int:
        if self._closed or self._broken:
            return -1
        return self._stream.fileno()

    def _state(self) -> _HandleState:
        if self._windows:
            import msvcrt

            return _windows_handle_state(msvcrt.get_osfhandle(self._stream.fileno()))
        metadata = os.fstat(self._stream.fileno())
        return _HandleState(
            identity=(int(metadata.st_dev), int(metadata.st_ino)),
            size=int(metadata.st_size),
            links=int(metadata.st_nlink),
            delete_pending=True,
            directory=not stat.S_ISREG(metadata.st_mode),
            reparse=stat.S_ISLNK(metadata.st_mode),
        )

    def _verify_metadata(self) -> None:
        if self._closed or self._broken:
            raise DatasetChangedError("prepared dataset capability is closed")
        state = self._state()
        if (
            state.identity != self._file_identity
            or state.directory
            or state.reparse
            or state.links != 0
            or not state.delete_pending
            or (
                self._expected_size is not None
                and state.size != self._expected_size
            )
        ):
            raise DatasetChangedError("prepared dataset handle identity changed")
        if self._windows:
            import msvcrt

            _require_private_windows_handle_acl(
                msvcrt.get_osfhandle(self._stream.fileno())
            )
            root = _windows_handle_state(self._guards[-1])
            if root.identity != self._root_identity or not root.directory or root.reparse:
                raise DatasetChangedError("training staging root handle changed")
            _require_private_windows_handle_acl(self._guards[-1])
        else:
            file_metadata = os.fstat(self._stream.fileno())
            root_metadata = os.fstat(self._guards[-1])
            getuid = getattr(os, "getuid", None)
            if (
                (callable(getuid) and file_metadata.st_uid != getuid())
                or stat.S_IMODE(file_metadata.st_mode) != _PRIVATE_FILE_MODE
                or (int(root_metadata.st_dev), int(root_metadata.st_ino))
                != self._root_identity
                or not stat.S_ISDIR(root_metadata.st_mode)
                or (callable(getuid) and root_metadata.st_uid != getuid())
                or stat.S_IMODE(root_metadata.st_mode) != _PRIVATE_DIRECTORY_MODE
            ):
                raise DatasetChangedError("private POSIX capability permissions changed")

    def write_all(self, payload: bytes) -> None:
        with self._lock:
            self._verify_metadata()
            view = memoryview(payload)
            written = 0
            self._stream.seek(0)
            while written < len(view):
                count = self._stream.write(view[written:])
                if count is None or count <= 0:
                    raise TrainingDataError("training dataset write made no progress")
                written += count
            self._stream.truncate(len(payload))
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._expected_size = len(payload)
            self._verify_metadata()

    def write_chunks(self, chunks) -> tuple[int, str]:
        with self._lock:
            self._verify_metadata()
            self._stream.seek(0)
            digest = hashlib.sha256()
            written = 0
            for chunk in chunks:
                if type(chunk) is not bytes or not chunk:
                    raise TrainingDataError(
                        "training dataset chunks must be nonempty bytes"
                    )
                digest.update(chunk)
                view = memoryview(chunk)
                offset = 0
                while offset < len(view):
                    count = self._stream.write(view[offset:])
                    if count is None or count <= 0:
                        raise TrainingDataError(
                            "training dataset write made no progress"
                        )
                    offset += count
                    written += count
            self._stream.truncate(written)
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._expected_size = written
            self._verify_metadata()
            return written, digest.hexdigest()

    def iter_verified_chunks(
        self,
        expected_size: int,
        expected_sha256: str,
        *,
        chunk_size: int = 65_536,
    ):
        if (
            type(chunk_size) is not int
            or isinstance(chunk_size, bool)
            or not 1 <= chunk_size <= 65_536
        ):
            raise ValueError("chunk_size must be an integer from 1 through 65536")
        with self._lock:
            self._verify_metadata()
            self._stream.seek(0)
            digest = hashlib.sha256()
            remaining = expected_size
            while remaining:
                chunk = self._stream.read(min(chunk_size, remaining))
                if type(chunk) is not bytes or not chunk:
                    raise DatasetChangedError(
                        "prepared dataset ended before its declared length"
                    )
                remaining -= len(chunk)
                digest.update(chunk)
                yield chunk
            if self._stream.read(1) != b"":
                raise DatasetChangedError(
                    "prepared dataset exceeds its declared length"
                )
            if digest.hexdigest() != expected_sha256:
                raise DatasetChangedError("prepared dataset SHA-256 changed")
            self._verify_metadata()

    def read_exact(self, expected_size: int) -> bytes:
        with self._lock:
            self._verify_metadata()
            self._stream.seek(0)
            payload = self._stream.read(expected_size + 1)
            if type(payload) is not bytes or len(payload) != expected_size:
                raise DatasetChangedError("prepared dataset size changed")
            self._verify_metadata()
            return payload

    def cleanup(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._broken:
                raise TrainingDataError("prepared dataset capability has an ambiguous close state")
            try:
                _close_owned_stream(self._stream)
            except BaseException as error:
                if self._stream.closed:
                    self._broken = True
                raise TrainingDataError("prepared dataset cleanup failed while closing its owned file") from error
            while self._guards:
                guard = self._guards[-1]
                try:
                    _close_owned_guard(guard, windows=self._windows)
                except BaseException as error:
                    raise TrainingDataError(
                        "prepared dataset cleanup failed while closing its root guards"
                    ) from error
                else:
                    self._guards.pop()
            self._closed = True


def _open_windows_capability(
    root: Path, mode: TrainingMode
) -> tuple[_OwnedDatasetCapability, Path]:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    guards = _open_windows_root_guards(root)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value
    display = root / f"training-{mode.value}-{uuid4().hex}.jsonl"
    handle = kernel.CreateFileW(
        os.fspath(display),
        0x80000000 | 0x40000000 | 0x00010000 | 0x00020000 | 0x00040000,
        0,
        None,
        1,
        0x00000100 | 0x00200000 | 0x04000000,
        None,
    )
    if int(handle) == invalid:
        for guard in reversed(guards):
            _close_owned_guard(guard, windows=True)
        raise OSError(
            ctypes.get_last_error(),
            "private delete-on-close training file could not be created",
        )
    raw_handle = int(handle)
    descriptor = -1
    stream: io.RawIOBase | None = None
    try:
        _set_private_windows_handle_acl(raw_handle)
        _require_private_windows_handle_acl(raw_handle)
        linked_state = _windows_handle_state(raw_handle)
        root_state = _windows_handle_state(guards[-1])
        if (
            linked_state.directory
            or linked_state.reparse
            or linked_state.links != 1
            or linked_state.delete_pending
            or linked_state.identity[0] != root_state.identity[0]
        ):
            raise TrainingDataError("private Windows training handle is unsafe before sealing")
        _mark_windows_delete_pending(raw_handle)
        state = _windows_handle_state(raw_handle)
        if (
            state.identity != linked_state.identity
            or state.links != 0
            or not state.delete_pending
        ):
            raise TrainingDataError("private Windows training handle is not delete-pending")
        descriptor = msvcrt.open_osfhandle(
            raw_handle, os.O_RDWR | int(getattr(os, "O_BINARY", 0))
        )
        raw_handle = -1
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        descriptor = -1
        capability = _OwnedDatasetCapability(
            stream=stream,
            guards=guards,
            windows=True,
            root_identity=root_state.identity,
            file_identity=state.identity,
        )
        stream = None
        return capability, display
    except BaseException:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)
        elif raw_handle >= 0:
            kernel.CloseHandle(wintypes.HANDLE(raw_handle))
        for guard in reversed(guards):
            try:
                _close_owned_guard(guard, windows=True)
            except OSError:
                pass
        raise


def _open_posix_root_guards(root: Path) -> tuple[int, ...]:
    if not root.is_absolute():
        raise TrainingDataError("training staging root must be absolute")
    flags = os.O_RDONLY
    for option in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
        flags |= int(getattr(os, option, 0))
    guards: list[int] = []
    try:
        guards.append(os.open(os.path.sep, flags))
        for component in root.parts[1:]:
            if component in ("", os.path.sep):
                continue
            descriptor = os.open(component, flags, dir_fd=guards[-1])
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(descriptor)
                raise TrainingDataError("training staging ancestry is not a directory")
            guards.append(descriptor)
        metadata = os.fstat(guards[-1])
        getuid = getattr(os, "getuid", None)
        if callable(getuid) and metadata.st_uid != getuid():
            raise TrainingDataError("training staging root owner is unsafe")
        if stat.S_IMODE(metadata.st_mode) != _PRIVATE_DIRECTORY_MODE:
            raise TrainingDataError("training staging root mode must be 0700")
        return tuple(guards)
    except BaseException:
        for guard in reversed(guards):
            try:
                os.close(guard)
            except OSError:
                pass
        raise


def _open_posix_capability(
    root: Path, mode: TrainingMode
) -> tuple[_OwnedDatasetCapability, Path]:
    guards = _open_posix_root_guards(root)
    temporary_flag = int(getattr(os, "O_TMPFILE", 0))
    if not temporary_flag:
        for guard in reversed(guards):
            os.close(guard)
        raise TrainingDataError("this host lacks anonymous O_TMPFILE staging")
    flags = os.O_RDWR | temporary_flag
    for option in ("O_CLOEXEC", "O_EXCL"):
        flags |= int(getattr(os, option, 0))
    descriptor = -1
    stream: io.RawIOBase | None = None
    try:
        descriptor = os.open(".", flags, _PRIVATE_FILE_MODE, dir_fd=guards[-1])
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        metadata = os.fstat(descriptor)
        getuid = getattr(os, "getuid", None)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 0
            or stat.S_IMODE(metadata.st_mode) != _PRIVATE_FILE_MODE
            or (callable(getuid) and metadata.st_uid != getuid())
            or metadata.st_dev != os.fstat(guards[-1]).st_dev
        ):
            raise TrainingDataError("anonymous POSIX training handle is unsafe")
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        descriptor = -1
        root_metadata = os.fstat(guards[-1])
        capability = _OwnedDatasetCapability(
            stream=stream,
            guards=guards,
            windows=False,
            root_identity=(int(root_metadata.st_dev), int(root_metadata.st_ino)),
            file_identity=(int(metadata.st_dev), int(metadata.st_ino)),
        )
        stream = None
        display = root / f"training-{mode.value}-{uuid4().hex}.anonymous.jsonl"
        return capability, display
    except BaseException:
        if stream is not None:
            stream.close()
        elif descriptor >= 0:
            os.close(descriptor)
        for guard in reversed(guards):
            try:
                os.close(guard)
            except OSError:
                pass
        raise


def _open_owned_capability(
    root: Path, mode: TrainingMode
) -> tuple[_OwnedDatasetCapability, Path]:
    root = root.absolute()
    if os.name == "nt":
        return _open_windows_capability(root, mode)
    return _open_posix_capability(root, mode)


def _write_owned_dataset(
    *,
    mode: TrainingMode,
    lines,
    root: Path,
    record_count: int,
    snapshot_sequence: int,
    screening: ScreeningAssessment,
    source_binding_sha256: str,
    input_record_ids: tuple[str, ...],
) -> PreparedTrainingDataset:
    if not isinstance(root, Path):
        raise TypeError("staging must be a Path")
    capability: _OwnedDatasetCapability | None = None
    try:
        capability, display_path = _open_owned_capability(root, mode)
        byte_length, digest = capability.write_chunks(lines)
        for _chunk in capability.iter_verified_chunks(byte_length, digest):
            pass
        return PreparedTrainingDataset(
            mode=mode,
            path=display_path,
            record_count=record_count,
            byte_length=byte_length,
            sha256=digest,
            snapshot_sequence=snapshot_sequence,
            screening=screening,
            _capability=capability,
            _source_binding_sha256=source_binding_sha256,
            _input_record_ids=input_record_ids,
        )
    except BaseException:
        if capability is not None:
            try:
                capability.cleanup()
            except TrainingDataError:
                pass
        raise


def _assessment(
    *,
    mode: TrainingMode,
    snapshot: ReadinessSnapshot,
    input_ids: tuple[str, ...],
    accepted_ids: tuple[str, ...],
    rejections: Counter[str],
) -> ScreeningAssessment:
    rejected = sum(rejections.values())
    if len(accepted_ids) + rejected != len(input_ids):
        raise TrainingDataError("screening accounting is inconsistent")
    return ScreeningAssessment(
        mode=mode,
        projection_schema=f"LIVED_{mode.value.upper()}_V1",
        snapshot_sequence=snapshot.sequence,
        input_record_ids=input_ids,
        accepted_record_ids=accepted_ids,
        input_identity_sha256=_identity(input_ids),
        accepted_identity_sha256=_identity(accepted_ids),
        input_count=len(input_ids),
        accepted_count=len(accepted_ids),
        rejected_count=rejected,
        rejection_counts=tuple(sorted(rejections.items())),
        policy_sha256=TRAINING_SCREENING_POLICY_SHA256,
        created_at=datetime.now(timezone.utc),
    )


def _prepare(
    snapshot: ReadinessSnapshot,
    screen: TrainingEgressScreen,
    staging: Path,
    mode: TrainingMode,
) -> PreparedTrainingDataset:
    if type(snapshot) is not ReadinessSnapshot:
        raise TypeError("snapshot must be a ReadinessSnapshot")
    if type(screen) is not TrainingEgressScreen:
        raise TypeError("screen must be a TrainingEgressScreen")
    selected, input_ids, binding = _selection(snapshot, mode)

    def projection_records(accepted: list[str], rejected: Counter[str]):
        for record, identity in zip(selected, input_ids, strict=True):
            try:
                if mode is TrainingMode.LORA:
                    line = _lora_line(record, screen, snapshot)
                else:
                    line = _dpo_line(record, screen, snapshot)
            except EgressRejectedError as error:
                if error.fatal:
                    raise
                rejected[error.reason] += 1
                continue
            accepted.append(identity)
            yield line

    first_ids: list[str] = []
    first_rejections: Counter[str] = Counter()
    first_digest = hashlib.sha256()
    first_bytes = 0
    for line in projection_records(first_ids, first_rejections):
        first_digest.update(line)
        first_bytes += len(line)
    accepted_ids = tuple(first_ids)
    assessment = _assessment(
        mode=mode,
        snapshot=snapshot,
        input_ids=input_ids,
        accepted_ids=accepted_ids,
        rejections=first_rejections,
    )

    repeated_ids: list[str] = []
    repeated_rejections: Counter[str] = Counter()
    dataset = _write_owned_dataset(
        mode=mode,
        lines=projection_records(repeated_ids, repeated_rejections),
        root=staging,
        record_count=len(accepted_ids),
        snapshot_sequence=snapshot.sequence,
        screening=assessment,
        source_binding_sha256=binding,
        input_record_ids=input_ids,
    )
    if (
        dataset.sha256 != first_digest.hexdigest()
        or dataset.byte_length != first_bytes
        or tuple(repeated_ids) != accepted_ids
        or repeated_rejections != first_rejections
    ):
        dataset.cleanup()
        raise EgressRejectedError(
            "PROJECTION_NONDETERMINISTIC",
            "complete screened projection must be byte-identical when repeated",
            fatal=True,
        )
    return dataset


def prepare_lora_dataset(
    snapshot: ReadinessSnapshot,
    screen: TrainingEgressScreen,
    staging: Path,
) -> PreparedTrainingDataset:
    return _prepare(snapshot, screen, staging, TrainingMode.LORA)


def prepare_dpo_dataset(
    snapshot: ReadinessSnapshot,
    screen: TrainingEgressScreen,
    staging: Path,
) -> PreparedTrainingDataset:
    return _prepare(snapshot, screen, staging, TrainingMode.DPO)


def verify_prepared_dataset(
    dataset: PreparedTrainingDataset,
    current: ReadinessSnapshot,
) -> None:
    if type(dataset) is not PreparedTrainingDataset:
        raise TypeError("dataset must be a PreparedTrainingDataset")
    if type(current) is not ReadinessSnapshot:
        raise TypeError("current must be a ReadinessSnapshot")
    if dataset._cleaned or dataset._descriptor < 0:
        raise DatasetChangedError("prepared dataset capability is closed")
    if current.sequence != dataset.snapshot_sequence:
        raise DatasetChangedError("new approval or state event changed the snapshot sequence")
    _selected, input_ids, binding = _selection(current, dataset.mode)
    if input_ids != dataset._input_record_ids or binding != dataset._source_binding_sha256:
        raise DatasetChangedError("authenticated dataset source changed after screening")
    for _chunk in dataset.iter_transport_chunks():
        pass


__all__ = [
    "DatasetChangedError",
    "DpoCandidate",
    "EgressRejectedError",
    "PreparedTrainingDataset",
    "TrainingDataError",
    "TrainingEgressScreen",
    "dpo_sample_id",
    "prepare_dpo_dataset",
    "prepare_lora_dataset",
    "select_dpo_candidates",
    "select_lora_pairs",
    "serialize_lora",
    "verify_prepared_dataset",
]
