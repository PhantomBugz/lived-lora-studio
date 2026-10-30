from __future__ import annotations

from dataclasses import dataclass
import math
import posixpath
import re
from typing import Literal
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from .forum_attachments import (
    AttachmentRejected,
    canonical_attachment_url,
    validate_attachment_filename,
)


class VbulletinValidationError(ValueError):
    """A vBulletin value is malformed or exceeds a fixed local bound."""


class VbulletinRouteError(VbulletinValidationError):
    """A URL is not an allowed vBulletin forum or thread route."""


_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")
_POST_ID = re.compile(r"post_message_[1-9][0-9]*")
_ROUTE_ENDPOINTS: dict[str, tuple[Literal["forum", "thread"], str]] = {
    "/forum/forumdisplay.php": ("forum", "f"),
    "/forum/showthread.php": ("thread", "t"),
}

_MAX_FORUM_PAGES = 100
_MAX_THREADS = 100
_MAX_PAGES_PER_THREAD = 100
_MAX_POSTS_PER_PAGE = 100
_MAX_POSTS = 10_000


def _normalized_host(value: object) -> str:
    if type(value) is not str or not value:
        raise VbulletinRouteError("allowed hosts must be exact DNS hostnames")
    if any(character in value for character in "/:@*?#\\") or value.endswith("."):
        raise VbulletinRouteError("allowed hosts must be exact DNS hostnames")
    try:
        if len(value.encode("utf-8")) > 253:
            raise VbulletinRouteError("allowed hosts must be exact DNS hostnames")
        host = value.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise VbulletinRouteError("allowed host is invalid") from exc
    labels = host.split(".")
    if len(labels) < 2 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise VbulletinRouteError("allowed host is invalid")
    return host


def _normalize_https_url(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise VbulletinRouteError("URL must be a nonempty string")
    value = value.strip()
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise VbulletinRouteError("URL contains invalid characters")
    try:
        split = urlsplit(value)
        port = split.port
    except ValueError as exc:
        raise VbulletinRouteError("URL is malformed") from exc
    if split.scheme.casefold() != "https" or not split.netloc:
        raise VbulletinRouteError("only absolute HTTPS URLs are accepted")
    if split.username is not None or split.password is not None:
        raise VbulletinRouteError("URL credentials are not accepted")
    if port not in (None, 443):
        raise VbulletinRouteError("only the default HTTPS port is accepted")
    if split.hostname is None:
        raise VbulletinRouteError("URL requires a host")
    host = _normalized_host(split.hostname)
    path = split.path or "/"
    trailing_slash = path.endswith("/")
    path = posixpath.normpath("/" + path.lstrip("/"))
    if trailing_slash and path != "/":
        path += "/"
    return urlunsplit(("https", host, path, split.query, ""))


def _raw_route_path(url: object) -> str:
    if type(url) is not str or not url.strip():
        raise VbulletinRouteError("URL must be a nonempty string")
    try:
        path = urlsplit(url.strip()).path
    except ValueError as exc:
        raise VbulletinRouteError("URL is malformed") from exc
    if path not in _ROUTE_ENDPOINTS:
        raise VbulletinRouteError("URL path is not an allowed vBulletin endpoint")
    return path


def _unique_query(pairs: list[tuple[str, str]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for name, value in pairs:
        if name in values:
            raise VbulletinRouteError(f"duplicate query field: {name}")
        values[name] = value
    return values


def _raw_query_pairs(query: str) -> list[tuple[str, str]]:
    """Reject escaped names or values before URL form decoding occurs."""

    if not query:
        return []
    pairs: list[tuple[str, str]] = []
    for field in query.split("&"):
        if field.count("=") != 1:
            raise VbulletinRouteError("URL query is malformed")
        name, value = field.split("=", 1)
        if name not in {"f", "t", "page"} or not _POSITIVE_DECIMAL.fullmatch(value):
            raise VbulletinRouteError("URL query must use exact raw keys and decimals")
        pairs.append((name, value))
    return pairs


def _positive_decimal(value: str, name: str) -> int:
    if not _POSITIVE_DECIMAL.fullmatch(value):
        raise VbulletinRouteError(f"{name} must be a positive decimal value")
    try:
        return int(value)
    except ValueError as exc:
        raise VbulletinRouteError(f"{name} must be a positive decimal value") from exc


@dataclass(frozen=True, slots=True)
class VbulletinRoute:
    kind: Literal["forum", "thread"]
    entity_id: int
    page: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("forum", "thread"):
            raise VbulletinValidationError("kind must be forum or thread")
        for name, value in (("entity_id", self.entity_id), ("page", self.page)):
            if type(value) is not int or value < 1:
                raise VbulletinValidationError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class VbulletinLinkHint:
    url: str
    is_sticky: bool = False
    has_code: bool = False
    has_attachment: bool = False

    def __post_init__(self) -> None:
        if type(self.url) is not str or not self.url:
            raise VbulletinValidationError("url must be a nonempty string")
        for name, value in (
            ("is_sticky", self.is_sticky),
            ("has_code", self.has_code),
            ("has_attachment", self.has_attachment),
        ):
            if type(value) is not bool:
                raise VbulletinValidationError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class VbulletinPost:
    post_id: str
    text: str
    original_utf8_bytes: int
    retained_utf8_bytes: int
    quote_utf8_bytes: int
    has_code: bool
    has_attachment: bool
    truncated: bool
    attachment_urls: tuple[str, ...] = ()
    attachment_filenames: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.post_id) is not str or not _POST_ID.fullmatch(self.post_id):
            raise VbulletinValidationError("post_id must be a vBulletin post message ID")
        if type(self.text) is not str or not self.text:
            raise VbulletinValidationError("text must be a nonempty string")
        try:
            actual_retained_bytes = len(self.text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise VbulletinValidationError("text must be valid UTF-8") from exc
        counts = (
            ("original_utf8_bytes", self.original_utf8_bytes),
            ("retained_utf8_bytes", self.retained_utf8_bytes),
            ("quote_utf8_bytes", self.quote_utf8_bytes),
        )
        for name, value in counts:
            if type(value) is not int or value < 0:
                raise VbulletinValidationError(f"{name} must be a nonnegative integer")
        if self.retained_utf8_bytes != actual_retained_bytes:
            raise VbulletinValidationError("retained_utf8_bytes must match text")
        if self.retained_utf8_bytes + self.quote_utf8_bytes > self.original_utf8_bytes:
            raise VbulletinValidationError("retained and quote bytes exceed original bytes")
        for name, value in (
            ("has_code", self.has_code),
            ("has_attachment", self.has_attachment),
            ("truncated", self.truncated),
        ):
            if type(value) is not bool:
                raise VbulletinValidationError(f"{name} must be a boolean")
        if type(self.attachment_urls) is not tuple or len(self.attachment_urls) > 8:
            raise VbulletinValidationError(
                "attachment_urls must be a tuple with at most eight URLs"
            )
        if (
            type(self.attachment_filenames) is not tuple
            or len(self.attachment_filenames) != len(self.attachment_urls)
        ):
            raise VbulletinValidationError(
                "attachment filenames must align exactly with attachment URLs"
            )
        canonical_urls: list[str] = []
        for value in self.attachment_urls:
            try:
                _attachment_id, canonical = canonical_attachment_url(
                    value,
                    allowed_host="www.unknowncheats.me",
                )
            except AttachmentRejected as exc:
                raise VbulletinValidationError(
                    "attachment_urls contains a noncanonical URL"
                ) from exc
            canonical_urls.append(canonical)
        for filename in self.attachment_filenames:
            try:
                validate_attachment_filename(filename)
            except AttachmentRejected as exc:
                raise VbulletinValidationError(
                    "attachment filenames contains an unsafe filename"
                ) from exc
        if len(set(canonical_urls)) != len(canonical_urls):
            raise VbulletinValidationError("attachment_urls contains a duplicate URL")
        if canonical_urls and not self.has_attachment:
            raise VbulletinValidationError(
                "attachment URLs require the attachment marker"
            )


@dataclass(frozen=True, slots=True)
class VbulletinCrawlLimits:
    max_forum_pages: int
    max_threads: int
    max_pages_per_thread: int
    max_posts_per_page: int
    max_posts: int

    def __post_init__(self) -> None:
        bounded_values = (
            ("max_forum_pages", self.max_forum_pages, _MAX_FORUM_PAGES),
            ("max_threads", self.max_threads, _MAX_THREADS),
            ("max_pages_per_thread", self.max_pages_per_thread, _MAX_PAGES_PER_THREAD),
            ("max_posts_per_page", self.max_posts_per_page, _MAX_POSTS_PER_PAGE),
            ("max_posts", self.max_posts, _MAX_POSTS),
        )
        for name, value, maximum in bounded_values:
            if type(value) is not int or not 1 <= value <= maximum:
                raise VbulletinValidationError(
                    f"{name} must be an integer from 1 through {maximum}"
                )


def clamp_vbulletin_limits(
    requested: VbulletinCrawlLimits,
    canonical: VbulletinCrawlLimits,
) -> VbulletinCrawlLimits:
    """Apply route-budget reductions without weakening hidden hard caps."""

    if not isinstance(requested, VbulletinCrawlLimits):
        raise TypeError("requested limits must be VbulletinCrawlLimits")
    if not isinstance(canonical, VbulletinCrawlLimits):
        raise TypeError("canonical limits must be VbulletinCrawlLimits")
    return VbulletinCrawlLimits(
        max_forum_pages=min(
            requested.max_forum_pages,
            canonical.max_forum_pages,
        ),
        max_threads=min(requested.max_threads, canonical.max_threads),
        max_pages_per_thread=min(
            requested.max_pages_per_thread,
            canonical.max_pages_per_thread,
        ),
        max_posts_per_page=canonical.max_posts_per_page,
        max_posts=canonical.max_posts,
    )


class VbulletinScheduler:
    """Select only bounded routes that remain tied to the starting route."""

    def __init__(
        self,
        start: VbulletinRoute,
        limits: VbulletinCrawlLimits,
    ) -> None:
        if not isinstance(start, VbulletinRoute):
            raise TypeError("start must be a VbulletinRoute")
        if not isinstance(limits, VbulletinCrawlLimits):
            raise TypeError("limits must be VbulletinCrawlLimits")
        self._start = start
        self._limits = limits
        self._host: str | None = None
        self._forum_pages: set[int] = (
            {start.page} if start.kind == "forum" else set()
        )
        self._thread_pages: dict[int, set[int]] = (
            {start.entity_id: {start.page}} if start.kind == "thread" else {}
        )

    def _current_is_scheduled(self, route: VbulletinRoute) -> bool:
        if route.kind == "forum":
            return (
                self._start.kind == "forum"
                and route.entity_id == self._start.entity_id
                and route.page in self._forum_pages
            )
        pages = self._thread_pages.get(route.entity_id)
        return pages is not None and route.page in pages

    def consider(
        self,
        current: str,
        links: tuple[str | VbulletinLinkHint, ...]
        | list[str | VbulletinLinkHint],
    ) -> tuple[str, ...]:
        """Return newly accepted routes after bounded filtering and ranking."""

        normalized_current = _normalize_https_url(current)
        current_host = urlsplit(normalized_current).hostname
        if current_host is None:
            raise VbulletinRouteError("current URL requires a host")
        if self._host is not None and current_host != self._host:
            return ()
        current_route = parse_vbulletin_route(
            current,
            allowed_hosts=(current_host,),
        )
        if not self._current_is_scheduled(current_route):
            return ()
        if not isinstance(links, (tuple, list)):
            raise TypeError("links must be a tuple or list of URL strings or hints")

        # Validate and rank against local candidates first. Scheduler state is
        # changed only after every bounded occurrence has been considered.
        occurrences: list[
            tuple[int, str, VbulletinRoute, bool, bool, bool]
        ] = []
        for index, value in enumerate(links[:512]):
            if type(value) is str:
                raw_url = value
                is_sticky = False
                has_code = False
                has_attachment = False
            elif type(value) is VbulletinLinkHint:
                raw_url = value.url
                is_sticky = value.is_sticky
                has_code = value.has_code
                has_attachment = value.has_attachment
            else:
                continue
            try:
                raw_parts = urlsplit(raw_url)
            except ValueError:
                continue
            if raw_parts.fragment:
                continue
            try:
                candidate_url = _normalize_https_url(raw_url)
                if urlsplit(candidate_url).hostname != current_host:
                    continue
                candidate = parse_vbulletin_route(
                    raw_url,
                    allowed_hosts=(current_host,),
                )
            except (TypeError, VbulletinRouteError):
                continue
            occurrences.append(
                (
                    index,
                    candidate_url,
                    candidate,
                    is_sticky,
                    has_code,
                    has_attachment,
                )
            )

        if self._host is None:
            self._host = current_host

        if current_route.kind == "forum":
            listing: tuple[int, str, VbulletinRoute, bool, bool, bool] | None = None
            threads: dict[
                int, tuple[int, str, VbulletinRoute, bool, bool, bool]
            ] = {}
            for occurrence in occurrences:
                index, _url, candidate, sticky, code, attachment = occurrence
                if candidate.kind == "forum":
                    if (
                        candidate.entity_id == self._start.entity_id
                        and candidate.page == current_route.page + 1
                        and candidate.page not in self._forum_pages
                        and (
                            listing is None
                            or index < listing[0]
                        )
                    ):
                        listing = occurrence
                    continue
                if candidate.entity_id in self._thread_pages:
                    continue
                previous = threads.get(candidate.entity_id)
                priority = (not sticky, not (code or attachment), index)
                if previous is None:
                    threads[candidate.entity_id] = occurrence
                    continue
                previous_priority = (
                    not previous[3],
                    not (previous[4] or previous[5]),
                    previous[0],
                )
                if priority < previous_priority:
                    threads[candidate.entity_id] = occurrence

            ranked_threads = sorted(
                threads.values(),
                key=lambda occurrence: (
                    not occurrence[3],
                    not (occurrence[4] or occurrence[5]),
                    occurrence[0],
                ),
            )
            available_threads = max(
                0,
                self._limits.max_threads - len(self._thread_pages),
            )
            ranked_threads = ranked_threads[:available_threads]
            if len(self._forum_pages) >= self._limits.max_forum_pages:
                listing = None

            # Preserve the legacy source position of listing pagination while
            # using the ranked thread candidates to fill thread positions.
            slots = sorted(
                ([listing] if listing is not None else []) + ranked_threads,
                key=lambda occurrence: occurrence[0],
            )
            ranked_iterator = iter(ranked_threads)
            selected_occurrences = [
                occurrence if occurrence[2].kind == "forum" else next(ranked_iterator)
                for occurrence in slots
            ]
            for occurrence in selected_occurrences:
                candidate = occurrence[2]
                if candidate.kind == "forum":
                    self._forum_pages.add(candidate.page)
                else:
                    self._thread_pages[candidate.entity_id] = {candidate.page}
            return tuple(occurrence[1] for occurrence in selected_occurrences)

        pages = self._thread_pages[current_route.entity_id]
        pagination: dict[
            int, tuple[int, str, VbulletinRoute, bool, bool, bool]
        ] = {}
        for occurrence in occurrences:
            index, _url, candidate, _sticky, code, attachment = occurrence
            if (
                candidate.kind != "thread"
                or candidate.entity_id != current_route.entity_id
                or candidate.page <= current_route.page
                or candidate.page in pages
            ):
                continue
            previous = pagination.get(candidate.page)
            priority = (not (code or attachment), index)
            if previous is None or priority < (
                not (previous[4] or previous[5]),
                previous[0],
            ):
                pagination[candidate.page] = occurrence
        ranked_pages = sorted(
            pagination.values(),
            key=lambda occurrence: (
                not (occurrence[4] or occurrence[5]),
                occurrence[0],
            ),
        )
        available_pages = max(
            0,
            self._limits.max_pages_per_thread - len(pages),
        )
        selected_occurrences = ranked_pages[:available_pages]
        for occurrence in selected_occurrences:
            pages.add(occurrence[2].page)
        return tuple(occurrence[1] for occurrence in selected_occurrences)


def parse_vbulletin_route(url: str, *, allowed_hosts: tuple[str, ...]) -> VbulletinRoute:
    """Parse one configured UnknownCheats forum or thread endpoint strictly."""

    if type(allowed_hosts) is not tuple or not allowed_hosts:
        raise VbulletinRouteError("allowed_hosts must be a nonempty tuple")
    hosts = tuple(dict.fromkeys(_normalized_host(value) for value in allowed_hosts))
    raw_path = _raw_route_path(url)
    normalized = _normalize_https_url(url)
    split = urlsplit(normalized)
    if split.hostname not in hosts:
        raise VbulletinRouteError("URL host is outside the allowed host set")
    route = _ROUTE_ENDPOINTS[raw_path]
    _raw_query_pairs(split.query)
    try:
        values = _unique_query(
            parse_qsl(split.query, keep_blank_values=True, strict_parsing=True)
        )
    except ValueError as exc:
        raise VbulletinRouteError("URL query is malformed") from exc
    kind, entity_name = route
    allowed_names = {entity_name, "page"}
    if set(values) - allowed_names or entity_name not in values:
        raise VbulletinRouteError("URL query does not match the endpoint")
    entity_id = _positive_decimal(values[entity_name], entity_name)
    page = _positive_decimal(values.get("page", "1"), "page")
    return VbulletinRoute(kind, entity_id, page)


__all__ = [
    "VbulletinCrawlLimits",
    "VbulletinLinkHint",
    "VbulletinPost",
    "VbulletinRoute",
    "VbulletinRouteError",
    "VbulletinScheduler",
    "VbulletinValidationError",
    "clamp_vbulletin_limits",
    "parse_vbulletin_route",
]
