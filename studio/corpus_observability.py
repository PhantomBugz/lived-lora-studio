from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable

from .models import DENIAL_REASON_CODES, OPERATOR_REJECTED
from .training_data import select_dpo_candidates
from .training_models import ReadinessSnapshot


_DOMAIN = re.compile(
    r"(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?[.])*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


def _utc(value: datetime, name: str) -> datetime:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset().total_seconds() != 0
    ):
        raise ValueError(f"{name} must be an aware UTC datetime")
    return value


@dataclass(frozen=True, slots=True)
class DpoReadinessTelemetry:
    decision_count: int
    approved_count: int
    denied_count: int
    modified_count: int
    action_ratios: tuple[tuple[str, float | None], ...]
    denial_reasons: tuple[tuple[str, int], ...]
    denial_reason_coverage: int
    preference_pair_count: int
    plausible_preference_pair_count: int
    unmarked_preference_pair_count: int
    plausible_rejection_share: float | None


@dataclass(frozen=True, slots=True)
class RagCorpusSource:
    domain: str
    captured_at: datetime

    def __post_init__(self) -> None:
        if (
            type(self.domain) is not str
            or self.domain != self.domain.casefold()
            or _DOMAIN.fullmatch(self.domain) is None
        ):
            raise ValueError("RAG source domain must be normalized ASCII DNS text")
        _utc(self.captured_at, "captured_at")


@dataclass(frozen=True, slots=True)
class RagDomainDrift:
    domain: str
    document_count: int
    oldest_capture: datetime
    newest_capture: datetime
    older_than_six_months_count: int
    older_than_six_months_share: float
    over_thirty_percent_older_than_six_months: bool


@dataclass(frozen=True, slots=True)
class RagCorpusDriftReport:
    reference_time: datetime
    six_month_cutoff: datetime
    document_count: int
    domains: tuple[RagDomainDrift, ...]


def derive_dpo_telemetry(
    snapshot: ReadinessSnapshot,
    plausible_denial_sequences: frozenset[int],
) -> DpoReadinessTelemetry:
    if type(snapshot) is not ReadinessSnapshot:
        raise TypeError("snapshot must be an exact ReadinessSnapshot")
    if type(plausible_denial_sequences) is not frozenset or any(
        type(sequence) is not int or sequence < 1
        for sequence in plausible_denial_sequences
    ):
        raise TypeError("plausible denial sequences must be an integer frozenset")

    counts = {"APPROVED": 0, "DENIED": 0, "MODIFIED": 0}
    reasons = {reason: 0 for reason in DENIAL_REASON_CODES}
    denied_sequences: set[int] = set()
    for decision in snapshot.decisions:
        counts[decision.kind] += 1
        if decision.kind == "DENIED":
            denied_sequences.add(decision.sequence)
            reason = (
                "other"
                if decision.reason_code == OPERATOR_REJECTED
                else decision.reason_code
            )
            if reason not in reasons:
                raise ValueError("DPO denial has an unknown reason code")
            reasons[reason] += 1
    if not plausible_denial_sequences.issubset(denied_sequences):
        raise ValueError("plausible signal is not bound to a denied decision")

    total = sum(counts.values())
    ratios = tuple(
        (
            label,
            None if total == 0 else round(counts[kind] / total, 12),
        )
        for label, kind in (
            ("approved", "APPROVED"),
            ("denied", "DENIED"),
            ("modified", "MODIFIED"),
        )
    )
    candidates = select_dpo_candidates(snapshot)
    plausible = sum(
        candidate.parent_decision.sequence in plausible_denial_sequences
        for candidate in candidates
    )
    preference_pairs = len(candidates)
    return DpoReadinessTelemetry(
        decision_count=total,
        approved_count=counts["APPROVED"],
        denied_count=counts["DENIED"],
        modified_count=counts["MODIFIED"],
        action_ratios=ratios,
        denial_reasons=tuple((reason, reasons[reason]) for reason in DENIAL_REASON_CODES),
        denial_reason_coverage=sum(count > 0 for count in reasons.values()),
        preference_pair_count=preference_pairs,
        plausible_preference_pair_count=plausible,
        unmarked_preference_pair_count=preference_pairs - plausible,
        plausible_rejection_share=(
            None
            if preference_pairs == 0
            else round(plausible / preference_pairs, 12)
        ),
    )


def _six_calendar_months_before(value: datetime) -> datetime:
    month_index = value.year * 12 + (value.month - 1) - 6
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def derive_rag_corpus_drift(
    sources: Iterable[RagCorpusSource],
    reference_time: datetime,
) -> RagCorpusDriftReport:
    now = _utc(reference_time, "reference_time")
    copied = tuple(sources)
    if len(copied) > 5_000_000 or any(
        type(source) is not RagCorpusSource for source in copied
    ):
        raise TypeError("sources must contain bounded RagCorpusSource records")
    cutoff = _six_calendar_months_before(now)
    grouped: dict[str, list[datetime]] = {}
    for source in copied:
        if source.captured_at > now:
            raise ValueError("RAG capture timestamp is in the future")
        grouped.setdefault(source.domain, []).append(source.captured_at)
    domains: list[RagDomainDrift] = []
    for domain in sorted(grouped):
        values = sorted(grouped[domain])
        old = sum(value < cutoff for value in values)
        count = len(values)
        domains.append(RagDomainDrift(
            domain=domain,
            document_count=count,
            oldest_capture=values[0],
            newest_capture=values[-1],
            older_than_six_months_count=old,
            older_than_six_months_share=round(old / count, 12),
            over_thirty_percent_older_than_six_months=(old * 10 > count * 3),
        ))
    return RagCorpusDriftReport(
        reference_time=now,
        six_month_cutoff=cutoff,
        document_count=len(copied),
        domains=tuple(domains),
    )


__all__ = [
    "DpoReadinessTelemetry",
    "RagCorpusDriftReport",
    "RagCorpusSource",
    "RagDomainDrift",
    "derive_dpo_telemetry",
    "derive_rag_corpus_drift",
]
