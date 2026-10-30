from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from typing import Any
from uuid import uuid4


_HEX64 = re.compile(r"[a-f0-9]{64}")

OPERATOR_MODIFIED = "OPERATOR_MODIFIED"
OPERATOR_APPROVED = "OPERATOR_APPROVED"
OPERATOR_REJECTED = "OPERATOR_REJECTED"
OPERATOR_DELETED = "OPERATOR_DELETED"
DENIAL_REASON_CODES = (
    "hallucinated_claim",
    "wrong_api",
    "stale_technique",
    "social_noise",
    "other",
)
REVIEW_REASON_CODES = frozenset(
    {
        OPERATOR_MODIFIED,
        OPERATOR_APPROVED,
        OPERATOR_REJECTED,
        OPERATOR_DELETED,
        *DENIAL_REASON_CODES,
    }
)


def require_review_reason(value: str, expected: str) -> str:
    if type(value) is not str or value not in REVIEW_REASON_CODES:
        raise ValueError("reason_code must be a fixed operator reason code")
    if expected == OPERATOR_REJECTED:
        if value not in {*DENIAL_REASON_CODES, OPERATOR_REJECTED}:
            raise ValueError(
                "reason_code must be a fixed denial reason category"
            )
    elif value != expected:
        raise ValueError(f"reason_code must be {expected}")
    return value
_PROVENANCE_V2_DOMAIN = b"liveD:capture-provenance:v2\x00"
_MAX_METADATA_DEPTH = 16


class FrozenJsonDict(Mapping[str, Any]):
    """A strict JSON object backed only by immutable private tuples."""

    __slots__ = ("__items",)

    def __init__(self, values: Mapping[str, Any]):
        if hasattr(self, "_FrozenJsonDict__items"):
            raise TypeError("capture metadata is immutable")
        if not isinstance(values, Mapping):
            raise TypeError("capture metadata must be a mapping")
        object.__setattr__(self, "_FrozenJsonDict__items", tuple(values.items()))

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self.__items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self.__items)

    def __len__(self) -> int:
        return len(self.__items)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise TypeError("capture metadata is immutable")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("capture metadata is immutable")

    def __repr__(self) -> str:
        return repr(dict(self.items()))


def _freeze_json(value: object, *, depth: int, active: set[int]) -> object:
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError("capture metadata exceeds the nesting limit")
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("capture metadata numbers must be finite")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("capture metadata contains a cycle")
        active.add(identity)
        try:
            frozen_values: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("capture metadata object keys must be strings")
                frozen_values[key] = _freeze_json(
                    item,
                    depth=depth + 1,
                    active=active,
                )
            return FrozenJsonDict(frozen_values)
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in active:
            raise ValueError("capture metadata contains a cycle")
        active.add(identity)
        try:
            return tuple(
                _freeze_json(item, depth=depth + 1, active=active)
                for item in value
            )
        finally:
            active.remove(identity)
    raise TypeError("capture metadata must contain only strict JSON values")


def freeze_json_metadata(value: Mapping[str, Any] | None) -> FrozenJsonDict:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a JSON object")
    frozen = _freeze_json(value, depth=0, active=set())
    if not isinstance(frozen, FrozenJsonDict):
        raise TypeError("metadata must be a JSON object")
    return frozen


def thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json_value(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        thaw_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")



def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _required(
    value: str, name: str, *, maximum: int, strip: bool = True
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if strip:
        value = value.strip()
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds the UTF-8 size limit")
    return value


@dataclass(frozen=True, slots=True)
class CapturedDocument:
    id: str
    blueprint_id: str
    url: str
    title: str
    raw_text: str
    captured_at: str
    content_sha256: str
    source_provenance_hash: str
    source_provenance_version: int
    metadata: FrozenJsonDict

    def __post_init__(self) -> None:
        if type(self.source_provenance_version) is not int or (
            self.source_provenance_version not in (1, 2)
        ):
            raise ValueError("source_provenance_version must be 1 or 2")
        if not isinstance(self.metadata, FrozenJsonDict):
            raise TypeError("capture metadata must be deeply immutable")

    @classmethod
    def create(
        cls,
        *,
        blueprint_id: str,
        url: str,
        title: str,
        raw_text: str,
        metadata: Mapping[str, Any] | None = None,
        source_provenance_version: int = 1,
    ) -> "CapturedDocument":
        if type(source_provenance_version) is not int or source_provenance_version not in (1, 2):
            raise ValueError("source_provenance_version must be 1 or 2")
        blueprint_id = _required(blueprint_id, "blueprint_id", maximum=128)
        url = _required(url, "url", maximum=4096)
        title = _required(title or "Untitled source", "title", maximum=2048)
        raw_text = _required(
            raw_text, "raw_text", maximum=4_000_000, strip=False
        )
        frozen_metadata = freeze_json_metadata(metadata)
        content_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        canonical_fields = {
            "blueprint_id": blueprint_id,
            "content_sha256": content_sha256,
            "title": title,
            "url": url,
        }
        canonical = _canonical_json(canonical_fields)
        if source_provenance_version == 2:
            canonical = _PROVENANCE_V2_DOMAIN + _canonical_json(
                {**canonical_fields, "metadata": frozen_metadata}
            )
        return cls(
            id=uuid4().hex,
            blueprint_id=blueprint_id,
            url=url,
            title=title,
            raw_text=raw_text,
            captured_at=_now(),
            content_sha256=content_sha256,
            source_provenance_hash=hashlib.sha256(canonical).hexdigest(),
            source_provenance_version=source_provenance_version,
            metadata=frozen_metadata,
        )

    def public_metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "blueprint_id": self.blueprint_id,
            "url": self.url,
            "title": self.title,
            "captured_at": self.captured_at,
            "content_sha256": self.content_sha256,
            "source_provenance_hash": self.source_provenance_hash,
            "source_provenance_version": self.source_provenance_version,
            "metadata": thaw_json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DistilledPair:
    id: str
    source_id: str
    instruction: str
    response: str
    tags: tuple[str, ...]
    source_provenance_hash: str
    user_approved: bool
    created_at: str
    reviewed_at: str | None

    @classmethod
    def create(
        cls,
        *,
        source_id: str,
        instruction: str,
        response: str,
        source_provenance_hash: str,
        tags: tuple[str, ...] = (),
    ) -> "DistilledPair":
        source_id = _required(source_id, "source_id", maximum=128)
        instruction = _required(instruction, "instruction", maximum=65_536)
        response = _required(response, "response", maximum=262_144)
        if not isinstance(source_provenance_hash, str) or not _HEX64.fullmatch(source_provenance_hash):
            raise ValueError("source_provenance_hash must be lowercase SHA-256")
        normalized_tags = tuple(
            dict.fromkeys(_required(tag, "tag", maximum=64).casefold() for tag in tags)
        )
        if len(normalized_tags) > 32:
            raise ValueError("at most 32 tags are permitted")
        return cls(
            id=uuid4().hex,
            source_id=source_id,
            instruction=instruction,
            response=response,
            tags=normalized_tags,
            source_provenance_hash=source_provenance_hash,
            user_approved=False,
            created_at=_now(),
            reviewed_at=None,
        )

    def review(
        self,
        *,
        instruction: str,
        response: str,
        tags: tuple[str, ...],
        approve: bool,
    ) -> "DistilledPair":
        updated = DistilledPair.create(
            source_id=self.source_id,
            instruction=instruction,
            response=response,
            source_provenance_hash=self.source_provenance_hash,
            tags=tags,
        )
        return replace(
            updated,
            id=self.id,
            created_at=self.created_at,
            user_approved=bool(approve),
            reviewed_at=_now(),
        )


@dataclass(frozen=True, slots=True)
class PairReviewRow:
    pair: DistilledPair
    status: str
