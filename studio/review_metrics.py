from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import sqlite3
import statistics

from .models import DENIAL_REASON_CODES

_DENIAL_EVENT_TO_REASON = {
    **{
        f"PAIR_DENIED_{reason.upper()}": reason
        for reason in DENIAL_REASON_CODES
    },
    **{
        f"PAIR_DENIED_PLAUSIBLE_{reason.upper()}": reason
        for reason in DENIAL_REASON_CODES
    },
    "PAIR_DENIED": "other",
}
_RESPONSE_STATE_KINDS = frozenset(
    {"PAIR_RESPONSE_DIFFERS", "PAIR_RESPONSE_MATCHES"}
)
_ACTION_KINDS = frozenset(
    {
        "PAIR_APPROVED",
        "PAIR_MODIFIED",
        *_RESPONSE_STATE_KINDS,
        *_DENIAL_EVENT_TO_REASON,
    }
)
_QUERY_KINDS = tuple(sorted({"PAIR_DISPLAYED", *_ACTION_KINDS}))


@dataclass(frozen=True, slots=True)
class ReviewSessionMetrics:
    window_start: datetime
    window_end: datetime
    reviewed_pairs: int
    average_decision_seconds: float | None
    median_decision_seconds: float | None
    edited_pairs: int
    edit_rate: float | None
    denial_reasons: tuple[tuple[str, int], ...]


def _utc(value: datetime, name: str) -> datetime:
    if type(value) is not datetime:
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    if value.utcoffset().total_seconds() != 0:
        raise ValueError(f"{name} must use UTC")
    return value


def _timestamp(value: object) -> datetime:
    if type(value) is not str:
        raise ValueError("audit timestamp must be text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("audit timestamp is invalid") from error
    return _utc(parsed, "audit timestamp")


def derive_review_metrics(
    connection: sqlite3.Connection,
    window_start: datetime,
    window_end: datetime,
) -> ReviewSessionMetrics:
    """Derive content-free review aggregates from one authenticated audit table."""
    if not isinstance(connection, sqlite3.Connection):
        raise TypeError("connection must be a sqlite3 connection")
    start = _utc(window_start, "window_start")
    end = _utc(window_end, "window_end")
    if end < start:
        raise ValueError("window_end must not precede window_start")

    placeholders = ",".join("?" for _ in _QUERY_KINDS)
    rows = connection.execute(
        "SELECT sequence, kind, subject_id, created_at "
        "FROM audit_events "
        "WHERE created_at >= ? AND created_at <= ? "
        f"AND kind IN ({placeholders}) "
        "ORDER BY sequence",
        (start.isoformat(), end.isoformat(), *_QUERY_KINDS),
    ).fetchall()

    displayed: dict[str, datetime] = {}
    actions: dict[str, list[tuple[int, str, datetime]]] = {}
    denial_counts = {reason: 0 for reason in DENIAL_REASON_CODES}
    for sequence_value, kind_value, subject_value, created_value in rows:
        if type(sequence_value) is not int or sequence_value < 1:
            raise ValueError("audit sequence is invalid")
        if type(kind_value) is not str or type(subject_value) is not str:
            raise ValueError("audit event identity is invalid")
        created_at = _timestamp(created_value)
        if kind_value == "PAIR_DISPLAYED":
            displayed.setdefault(subject_value, created_at)
            continue
        if kind_value in _ACTION_KINDS:
            actions.setdefault(subject_value, []).append(
                (sequence_value, kind_value, created_at)
            )
            reason = _DENIAL_EVENT_TO_REASON.get(kind_value)
            if reason is not None:
                denial_counts[reason] += 1

    decision_seconds: list[float] = []
    reviewed_subjects: set[str] = set()
    edited_subjects: set[str] = set()
    for subject_id, displayed_at in displayed.items():
        after_display = tuple(
            event for event in actions.get(subject_id, ()) if event[2] >= displayed_at
        )
        if not after_display:
            continue
        reviewed_subjects.add(subject_id)
        first_action = min(after_display, key=lambda event: (event[0], event[2]))
        decision_seconds.append((first_action[2] - displayed_at).total_seconds())
        response_states = tuple(
            event for event in after_display if event[1] in _RESPONSE_STATE_KINDS
        )
        if response_states:
            latest = max(response_states, key=lambda event: (event[0], event[2]))
            if latest[1] == "PAIR_RESPONSE_DIFFERS":
                edited_subjects.add(subject_id)

    reviewed = len(reviewed_subjects)
    average = (
        round(sum(decision_seconds) / len(decision_seconds), 6)
        if decision_seconds
        else None
    )
    median = (
        round(float(statistics.median(decision_seconds)), 6)
        if decision_seconds
        else None
    )
    edited = len(edited_subjects)
    edit_rate = round(edited / reviewed, 12) if reviewed else None
    return ReviewSessionMetrics(
        window_start=start,
        window_end=end,
        reviewed_pairs=reviewed,
        average_decision_seconds=average,
        median_decision_seconds=median,
        edited_pairs=edited,
        edit_rate=edit_rate,
        denial_reasons=tuple(
            (reason, denial_counts[reason]) for reason in DENIAL_REASON_CODES
        ),
    )


__all__ = [
    "DENIAL_REASON_CODES",
    "ReviewSessionMetrics",
    "derive_review_metrics",
]
