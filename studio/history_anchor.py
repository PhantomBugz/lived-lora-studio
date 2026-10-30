from __future__ import annotations

from dataclasses import dataclass
import re


_SHA256 = re.compile(r"[0-9a-f]{64}")


class HistoryAnchorProtocolError(RuntimeError):
    """The authenticated anchor provider returned an invalid result."""


@dataclass(frozen=True, slots=True)
class HistoryAnchorHead:
    count: int
    head_sha256: str | None


def require_history_anchor_store(store: object) -> object:
    if not callable(getattr(store, "read_local_state_anchor", None)):
        raise TypeError("anchor_store must expose read_local_state_anchor")
    if not callable(getattr(store, "advance_local_state_anchor", None)):
        raise TypeError("anchor_store must expose advance_local_state_anchor")
    return store


def read_history_anchor(store: object, scope: str) -> HistoryAnchorHead:
    require_history_anchor_store(store)
    anchor = store.read_local_state_anchor(scope)
    if anchor is None:
        return HistoryAnchorHead(0, None)
    observed_scope = getattr(anchor, "scope", None)
    count = getattr(anchor, "count", None)
    head = getattr(anchor, "head_sha256", None)
    if (
        type(observed_scope) is not str
        or observed_scope != scope
        or type(count) is not int
        or count < 1
        or type(head) is not str
        or _SHA256.fullmatch(head) is None
    ):
        raise HistoryAnchorProtocolError(
            "authenticated history anchor result is invalid"
        )
    return HistoryAnchorHead(count, head)


def advance_history_anchor(
    store: object,
    scope: str,
    *,
    expected_count: int,
    expected_head_sha256: str | None,
    new_head_sha256: str,
) -> bool:
    require_history_anchor_store(store)
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("expected history count must be nonnegative")
    if expected_count == 0:
        if expected_head_sha256 is not None:
            raise ValueError("an absent history must not have a head")
    elif (
        type(expected_head_sha256) is not str
        or _SHA256.fullmatch(expected_head_sha256) is None
    ):
        raise ValueError("expected history head must be lowercase SHA-256")
    if (
        type(new_head_sha256) is not str
        or _SHA256.fullmatch(new_head_sha256) is None
        or new_head_sha256 == expected_head_sha256
    ):
        raise ValueError("new history head must be a changed lowercase SHA-256")
    new_count = expected_count + 1
    result = store.advance_local_state_anchor(
        scope,
        expected_count=expected_count,
        expected_head_sha256=expected_head_sha256,
        new_count=new_count,
        new_head_sha256=new_head_sha256,
    )
    anchor = getattr(result, "anchor", None)
    appended = getattr(result, "appended", None)
    if (
        type(appended) is not bool
        or getattr(anchor, "scope", None) != scope
        or getattr(anchor, "count", None) != new_count
        or getattr(anchor, "head_sha256", None) != new_head_sha256
    ):
        raise HistoryAnchorProtocolError(
            "authenticated history anchor append result is invalid"
        )
    return appended


__all__ = [
    "HistoryAnchorHead",
    "HistoryAnchorProtocolError",
    "advance_history_anchor",
    "read_history_anchor",
    "require_history_anchor_store",
]
