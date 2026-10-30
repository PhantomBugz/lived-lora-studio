from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import heapq
import importlib
import math
import statistics
import struct

from .blueprints import (
    BUILTIN_BLUEPRINTS,
    SourceBlueprint,
    require_registered_blueprint_url,
)
from .training_models import (
    ActionableRecommendation,
    ReadinessSnapshot,
    SourceTargetView,
)


MAX_SEMANTIC_SAMPLE = 2_000
MAX_VECTOR_DIMENSION = 16_384
MAX_ELIGIBLE_TEXTS = 5_000_000
MAX_TEXT_BYTES = 4_000_000
MAX_LOSS_POINTS = 5_000_000
MAX_PURE_PYTHON_DOT_COMPONENTS = 1_000_000
COSINE_BLOCK_ROWS = 64
MAX_CATEGORIES = 256
MAX_DEFICITS = 128
MAX_COUNT = 5_000_000
LOW_DIVERSITY_THRESHOLD = 0.12
DOMINANT_CATEGORY_FRACTION = 0.60
REQUIRED_CATEGORY_FRACTION = 0.10
_BIDI_RANGES = ((0x202A, 0x202E), (0x2066, 0x2069))
_BIDI_SINGLETONS = frozenset((0x061C, 0x200E, 0x200F))


def _bounded_sequence(value: object, name: str, *, maximum: int) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a bounded sequence")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds its item limit")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_COUNT,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
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


def _safe_text(
    value: object,
    name: str,
    *,
    maximum_bytes: int,
    allow_formatting: bool = False,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    if not value or (not allow_formatting and value != value.strip()):
        raise ValueError(f"{name} must be nonempty and normalized")
    for character in value:
        codepoint = ord(character)
        if (
            (
                codepoint < 0x20
                and (not allow_formatting or codepoint not in (0x09, 0x0A, 0x0D))
            )
            or 0x7F <= codepoint <= 0x9F
            or codepoint in _BIDI_SINGLETONS
            or any(lower <= codepoint <= upper for lower, upper in _BIDI_RANGES)
        ):
            raise ValueError(f"{name} contains unsafe control or bidi text")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise ValueError(f"{name} exceeds its UTF-8 size limit")
    return value


def _category(value: object, name: str = "category") -> str:
    return _safe_text(value, name, maximum_bytes=64).casefold()


@dataclass(frozen=True, slots=True)
class LossPoint:
    step: int
    loss: float

    def __post_init__(self) -> None:
        _integer(self.step, "step", minimum=1, maximum=9_223_372_036_854_775_807)
        object.__setattr__(self, "loss", _finite(self.loss, "loss"))


def detect_loss_plateau(
    points: Sequence[LossPoint],
    *,
    patience_steps: int = 200,
    minimum_delta: float = 0.001,
) -> bool:
    values = _bounded_sequence(points, "points", maximum=MAX_LOSS_POINTS)
    _integer(
        patience_steps,
        "patience_steps",
        minimum=1,
        maximum=MAX_LOSS_POINTS,
    )
    delta = _finite(minimum_delta, "minimum_delta", minimum=0.0)
    if delta == 0.0:
        raise ValueError("minimum_delta must be greater than zero")
    if not all(type(point) is LossPoint for point in values):
        raise TypeError("points must contain only LossPoint records")
    ordered = sorted(values, key=lambda point: point.step)
    if len({point.step for point in ordered}) != len(ordered):
        raise ValueError("points contain duplicate completed steps")
    if not ordered:
        return False

    best = math.inf
    last_improvement_step = ordered[0].step
    for point in ordered:
        if point.loss <= best - delta:
            best = point.loss
            last_improvement_step = point.step
    return ordered[-1].step - last_improvement_step >= patience_steps


def hash_ordered_text_sample(texts: Sequence[str]) -> tuple[str, ...]:
    values = _bounded_sequence(
        texts,
        "texts",
        maximum=MAX_ELIGIBLE_TEXTS,
    )
    def keyed_values():
        for index, value in enumerate(values):
            text = _safe_text(
                value,
                "text",
                maximum_bytes=MAX_TEXT_BYTES,
                allow_formatting=True,
            )
            encoded = text.encode("utf-8")
            yield hashlib.sha256(encoded).digest(), encoded, index, text

    return tuple(
        item[3]
        for item in heapq.nsmallest(MAX_SEMANTIC_SAMPLE, keyed_values())
    )


def _finite_unit_vectors(
    vectors: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    values = _bounded_sequence(
        vectors,
        "vectors",
        maximum=MAX_SEMANTIC_SAMPLE,
    )
    if not values:
        return ()
    normalized: list[tuple[float, ...]] = []
    width: int | None = None
    for index, vector in enumerate(values):
        components = _bounded_sequence(
            vector,
            f"vector {index}",
            maximum=MAX_VECTOR_DIMENSION,
        )
        if not components:
            raise ValueError("vectors must not be empty")
        if width is None:
            width = len(components)
        elif len(components) != width:
            raise ValueError("vectors must have one consistent dimension")
        finite = tuple(
            _finite(component, f"vector {index} component")
            for component in components
        )
        norm = math.sqrt(math.fsum(component * component for component in finite))
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("vectors must have a finite nonzero norm")
        normalized.append(tuple(component / norm for component in finite))

    def vector_key(vector: tuple[float, ...]) -> tuple[bytes, bytes]:
        packed = struct.pack(f"!{len(vector)}d", *vector)
        return hashlib.sha256(packed).digest(), packed

    return tuple(sorted(normalized, key=vector_key))


def _vector_shape(
    vectors: Sequence[Sequence[float]],
) -> tuple[Sequence[object], int]:
    values = _bounded_sequence(
        vectors,
        "vectors",
        maximum=MAX_SEMANTIC_SAMPLE,
    )
    if not values:
        return values, 0
    width: int | None = None
    for index, vector in enumerate(values):
        components = _bounded_sequence(
            vector,
            f"vector {index}",
            maximum=MAX_VECTOR_DIMENSION,
        )
        if not components:
            raise ValueError("vectors must not be empty")
        if width is None:
            width = len(components)
        elif len(components) != width:
            raise ValueError("vectors must have one consistent dimension")
        for component in components:
            _finite(component, f"vector {index} component")
    return values, width or 0


def _semantic_diversity_accelerated(
    vectors: Sequence[object],
    *,
    width: int,
) -> float:
    try:
        numpy = importlib.import_module("numpy")
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for bounded semantic diversity at this sample size"
        ) from error

    matrix = numpy.empty((len(vectors), width), dtype=numpy.float64)
    for row_index, vector in enumerate(vectors):
        components = vector
        matrix[row_index, :] = [
            _finite(component, f"vector {row_index} component")
            for component in components
        ]
    norms = numpy.linalg.norm(matrix, axis=1)
    if not bool(numpy.all(numpy.isfinite(norms))) or bool(numpy.any(norms == 0.0)):
        raise ValueError("vectors must have a finite nonzero norm")
    matrix = matrix / norms[:, None]

    def row_key(index: int) -> tuple[bytes, bytes]:
        encoded = matrix[index].astype(">f8", copy=False).tobytes(order="C")
        return hashlib.sha256(encoded).digest(), encoded

    order = sorted(range(len(vectors)), key=row_key)
    matrix = numpy.ascontiguousarray(matrix[order], dtype=numpy.float64)
    nearest_distances: list[float] = []
    for start in range(0, len(vectors), COSINE_BLOCK_ROWS):
        stop = min(start + COSINE_BLOCK_ROWS, len(vectors))
        similarities = matrix[start:stop] @ matrix.T
        rows = numpy.arange(stop - start)
        similarities[rows, numpy.arange(start, stop)] = -numpy.inf
        nearest = numpy.max(similarities, axis=1)
        distances = 1.0 - nearest
        nearest_distances.extend(float(value) for value in distances)
    return min(1.0, max(0.0, math.fsum(nearest_distances) / len(nearest_distances)))


def semantic_diversity(vectors: Sequence[Sequence[float]]) -> float:
    values, width = _vector_shape(vectors)
    if len(values) >= 2:
        dot_components = len(values) * len(values) * width
        if dot_components > MAX_PURE_PYTHON_DOT_COMPONENTS:
            return _semantic_diversity_accelerated(values, width=width)
    normalized = _finite_unit_vectors(values)
    if len(normalized) < 2:
        return 0.0
    nearest_distances = []
    for index, left in enumerate(normalized):
        nearest = math.inf
        for other_index, right in enumerate(normalized):
            if index == other_index:
                continue
            cosine = math.fsum(
                left_component * right_component
                for left_component, right_component in zip(left, right, strict=True)
            )
            nearest = min(nearest, 1.0 - cosine)
        nearest_distances.append(nearest)
    return min(1.0, max(0.0, statistics.fmean(nearest_distances)))


def category_feedback(
    category_distribution: Sequence[tuple[str, int]],
    *,
    eligible: int,
    diversity: float | None,
    required_categories: Sequence[str] = (),
) -> tuple[str, ...]:
    distribution = _bounded_sequence(
        category_distribution,
        "category_distribution",
        maximum=MAX_CATEGORIES,
    )
    total = _integer(eligible, "eligible")
    score = (
        None
        if diversity is None
        else _finite(diversity, "diversity", minimum=0.0, maximum=1.0)
    )
    required = _bounded_sequence(
        required_categories,
        "required_categories",
        maximum=32,
    )

    counts: dict[str, int] = {}
    for bucket in distribution:
        if type(bucket) is not tuple or len(bucket) != 2:
            raise TypeError("category_distribution entries must be immutable pairs")
        category, count = bucket
        normalized = _category(category)
        if normalized in counts:
            raise ValueError("category_distribution contains duplicate categories")
        counts[normalized] = _integer(
            count,
            "category count",
            maximum=total,
        )

    normalized_required = tuple(
        _category(value, "required category")
        for value in required
    )
    if len(set(normalized_required)) != len(normalized_required):
        raise ValueError("required_categories contains duplicate categories")

    flags: list[str] = []
    if score is not None and score < LOW_DIVERSITY_THRESHOLD:
        flags.append("LOW_SEMANTIC_DIVERSITY")
    if total:
        flags.extend(
            f"DOMINANT_CATEGORY:{category}"
            for category, count in sorted(counts.items())
            if count / total > DOMINANT_CATEGORY_FRACTION
        )
        flags.extend(
            f"REQUIRED_CATEGORY_BELOW_MINIMUM:{category}"
            for category in sorted(normalized_required)
            if counts.get(category, 0) / total < REQUIRED_CATEGORY_FRACTION
        )
    return tuple(flags[:32])


def _validated_deficits(deficits: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    if not isinstance(deficits, Mapping):
        raise TypeError("deficits must be a category-keyed mapping")
    entries = tuple(deficits.items())
    if len(entries) > MAX_DEFICITS:
        raise ValueError("deficits exceeds its category limit")
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for category, count in entries:
        name = _category(category, "deficit category")
        if name in seen:
            raise ValueError("deficits contains duplicate normalized categories")
        seen.add(name)
        normalized.append(
            (name, _integer(count, "deficit", minimum=1))
        )
    return tuple(sorted(normalized, key=lambda item: (-item[1], item[0])))


def _freshness_key(target: SourceTargetView) -> tuple[bool, float]:
    if target.last_accepted_at is None:
        return True, 0.0
    timestamp = target.last_accepted_at.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError("target freshness timestamp must be finite")
    return False, -timestamp


def _require_utc(value: datetime, name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be UTC-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use a zero UTC offset")
    return value


def _has_recent_accepted_history(
    target: SourceTargetView,
    *,
    now: datetime,
) -> bool:
    accepted_at = target.last_accepted_at
    if accepted_at is None or target.expected_yield == 0:
        return False
    if accepted_at > now:
        raise ValueError("target accepted history cannot be in the future")
    return (now - accepted_at).days <= 30


def recommend_sources(
    snapshot: ReadinessSnapshot,
    deficits: Mapping[str, int],
    *,
    now: datetime | None = None,
    blueprints: Mapping[str, SourceBlueprint] = BUILTIN_BLUEPRINTS,
) -> tuple[ActionableRecommendation, ...]:
    if type(snapshot) is not ReadinessSnapshot:
        raise TypeError("snapshot must be a ReadinessSnapshot")
    required = _validated_deficits(deficits)
    deficit_map = dict(required)
    report_time = (
        datetime.now(timezone.utc)
        if now is None
        else _require_utc(now, "now")
    )

    validated_urls: dict[str, str] = {}
    eligible: list[SourceTargetView] = []
    for target in snapshot.targets:
        if (
            not target.operator_authorized
            or not _has_recent_accepted_history(target, now=report_time)
        ):
            continue
        validated_urls[target.target_id] = require_registered_blueprint_url(
            target.blueprint_id,
            target.start_url,
            blueprints=blueprints,
        )
        if any(
            category.casefold() in deficit_map
            for category in target.categories
        ):
            eligible.append(target)

    def ranking_key(target: SourceTargetView) -> tuple[object, ...]:
        covered = sum(
            deficit_map.get(category.casefold(), 0)
            for category in target.categories
        )
        no_history, freshness = _freshness_key(target)
        return (
            -covered,
            -target.expected_yield,
            no_history,
            freshness,
            target.label.casefold(),
            target.target_id,
        )

    eligible.sort(key=ranking_key)
    recommendations: list[ActionableRecommendation] = []
    for category, deficit in required:
        target = next(
            (
                candidate
                for candidate in eligible
                if category in {
                    candidate_category.casefold()
                    for candidate_category in candidate.categories
                }
            ),
            None,
        )
        if target is None:
            recommendations.append(
                ActionableRecommendation(
                    category=category,
                    deficit=deficit,
                    rationale=(
                        "No operator-authorized exact source target covers "
                        f"{category}; map and approve one before collection."
                    ),
                    target_id=None,
                    blueprint_id=None,
                    label="MAP_SOURCE_TARGET",
                    start_url=None,
                    expected_yield=0,
                )
            )
            continue
        recommendations.append(
            ActionableRecommendation(
                category=category,
                deficit=deficit,
                rationale=(
                    "Collect additional approved records from this exact "
                    f"operator-authorized {category} target."
                ),
                target_id=target.target_id,
                blueprint_id=target.blueprint_id,
                label=target.label,
                start_url=validated_urls[target.target_id],
                expected_yield=target.expected_yield,
            )
        )
    return tuple(recommendations)


__all__ = [
    "DOMINANT_CATEGORY_FRACTION",
    "LOW_DIVERSITY_THRESHOLD",
    "LossPoint",
    "MAX_SEMANTIC_SAMPLE",
    "REQUIRED_CATEGORY_FRACTION",
    "category_feedback",
    "detect_loss_plateau",
    "hash_ordered_text_sample",
    "recommend_sources",
    "semantic_diversity",
]
