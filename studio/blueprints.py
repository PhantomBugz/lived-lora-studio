from __future__ import annotations

from dataclasses import dataclass
import json
import math
import posixpath
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Self
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from .vbulletin import VbulletinCrawlLimits


class BlueprintValidationError(ValueError):
    """A source blueprint does not match the supported schema."""


class UrlScopeError(ValueError):
    """A URL is insecure, malformed, or outside a blueprint's host set."""


_SCHEMA_FIELDS = frozenset(
    {
        "schema_version",
        "blueprint_id",
        "name",
        "allowed_hosts",
        "login_hosts",
        "content_selector",
        "title_selector",
        "link_selector",
        "max_pages",
        "crawl_delay_seconds",
    }
)
_BLUEPRINT_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9a-fA-F]{2})")
_CSS_MAX_BYTES = 512
_BIDI_CONTROL_RANGES = ((0x202A, 0x202E), (0x2066, 0x2069))
_BIDI_CONTROL_SINGLETONS = frozenset((0x061C, 0x200E, 0x200F))


def _required_string(value: object, name: str, *, maximum: int) -> str:
    if type(value) is not str or not value.strip():
        raise BlueprintValidationError(f"{name} must be a nonempty string")
    normalized = value.strip()
    if len(normalized.encode("utf-8")) > maximum:
        raise BlueprintValidationError(f"{name} exceeds its UTF-8 size limit")
    return normalized


def _normalize_host(value: object) -> str:
    host = _required_string(value, "allowed host", maximum=253)
    if any(character in host for character in "/:@*?#\\") or host.endswith("."):
        raise BlueprintValidationError("allowed hosts must be exact DNS hostnames")
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise BlueprintValidationError("allowed host is not a valid DNS hostname") from exc
    labels = ascii_host.split(".")
    if len(labels) < 2 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise BlueprintValidationError("allowed host is not a valid DNS hostname")
    return ascii_host


def _validate_css_selector(value: object, name: str) -> str:
    selector = _required_string(value, name, maximum=_CSS_MAX_BYTES)
    lowered = selector.casefold()
    if lowered.startswith(("xpath=", "text=", "javascript:")):
        raise BlueprintValidationError(f"{name} must be a CSS selector")
    if any(ord(character) < 32 or ord(character) == 127 for character in selector):
        raise BlueprintValidationError(f"{name} contains control characters")
    if any(character in selector for character in "{};`"):
        raise BlueprintValidationError(f"{name} contains non-selector syntax")
    if selector[0] in ">+~," or selector[-1] in ">+~,":
        raise BlueprintValidationError(f"{name} has an incomplete selector group")

    closing = {"]": "[", ")": "("}
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    for character in selector:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in "\"'":
            quote = character
        elif character in "[(":
            stack.append(character)
        elif character in "])":
            if not stack or stack.pop() != closing[character]:
                raise BlueprintValidationError(f"{name} has unbalanced delimiters")
    if escaped or quote is not None or stack:
        raise BlueprintValidationError(f"{name} has unbalanced delimiters")
    return selector


def normalize_https_url(url: str, *, base_url: str | None = None) -> str:
    """Return a fragment-free canonical HTTPS URL without changing its query."""

    if type(url) is not str or not url.strip():
        raise UrlScopeError("URL must be a nonempty string")
    value = url.strip()
    if base_url is not None:
        normalized_base = normalize_https_url(base_url)
        value = urljoin(normalized_base, value)
    if _INVALID_PERCENT_ESCAPE.search(value):
        raise UrlScopeError("URL contains an invalid percent escape")
    try:
        decoded = unquote(value, errors="strict")
    except UnicodeError as exc:
        raise UrlScopeError("URL contains invalid UTF-8 escapes") from exc
    for character in decoded:
        codepoint = ord(character)
        if (
            character == "\\"
            or character.isspace()
            or codepoint < 0x20
            or 0x7F <= codepoint <= 0x9F
            or codepoint in _BIDI_CONTROL_SINGLETONS
            or any(
                lower <= codepoint <= upper
                for lower, upper in _BIDI_CONTROL_RANGES
            )
        ):
            raise UrlScopeError("URL contains unsafe control or bidi text")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise UrlScopeError("URL is malformed") from exc
    if parts.scheme.casefold() != "https" or not parts.netloc:
        raise UrlScopeError("only absolute HTTPS URLs are accepted")
    if parts.username is not None or parts.password is not None:
        raise UrlScopeError("URL credentials are not accepted")
    if port not in (None, 443):
        raise UrlScopeError("only the default HTTPS port is accepted")
    if parts.hostname is None:
        raise UrlScopeError("URL requires a host")
    try:
        host = _normalize_host(parts.hostname)
    except BlueprintValidationError as exc:
        raise UrlScopeError("URL host is invalid") from exc

    path = parts.path or "/"
    trailing_slash = path.endswith("/")
    path = "/" + path.lstrip("/")
    path = posixpath.normpath(path)
    if trailing_slash and path != "/":
        path += "/"
    netloc = host
    return urlunsplit(("https", netloc, path, parts.query, ""))


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BlueprintValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class SourceBlueprint:
    schema_version: int
    blueprint_id: str
    name: str
    allowed_hosts: tuple[str, ...]
    login_hosts: tuple[str, ...]
    content_selector: str
    title_selector: str
    link_selector: str
    max_pages: int
    crawl_delay_seconds: float

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise BlueprintValidationError("schema_version must be the integer 1")
        blueprint_id = _required_string(self.blueprint_id, "blueprint_id", maximum=128)
        if not _BLUEPRINT_ID.fullmatch(blueprint_id):
            raise BlueprintValidationError("blueprint_id must use lowercase URL-safe characters")
        name = _required_string(self.name, "name", maximum=256)
        if not isinstance(self.allowed_hosts, tuple) or not self.allowed_hosts:
            raise BlueprintValidationError("allowed_hosts must contain at least one host")
        hosts = tuple(dict.fromkeys(_normalize_host(host) for host in self.allowed_hosts))
        if len(hosts) > 32:
            raise BlueprintValidationError("at most 32 allowed hosts are permitted")
        if not isinstance(self.login_hosts, tuple) or not self.login_hosts:
            raise BlueprintValidationError("login_hosts must contain at least one host")
        login_hosts = tuple(
            dict.fromkeys(_normalize_host(host) for host in self.login_hosts)
        )
        if len(login_hosts) > 32:
            raise BlueprintValidationError("at most 32 login hosts are permitted")
        if not set(hosts).issubset(login_hosts):
            raise BlueprintValidationError(
                "login_hosts must include every allowed content host"
            )
        content_selector = _validate_css_selector(self.content_selector, "content_selector")
        title_selector = _validate_css_selector(self.title_selector, "title_selector")
        link_selector = _validate_css_selector(self.link_selector, "link_selector")
        if type(self.max_pages) is not int or not 1 <= self.max_pages <= 100:
            raise BlueprintValidationError("max_pages must be an integer from 1 through 100")
        if type(self.crawl_delay_seconds) not in (int, float):
            raise BlueprintValidationError("crawl_delay_seconds must be a number")
        delay = float(self.crawl_delay_seconds)
        if not math.isfinite(delay) or not 0.0 <= delay <= 60.0:
            raise BlueprintValidationError("crawl_delay_seconds must be from 0 through 60")

        object.__setattr__(self, "blueprint_id", blueprint_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "allowed_hosts", hosts)
        object.__setattr__(self, "login_hosts", login_hosts)
        object.__setattr__(self, "content_selector", content_selector)
        object.__setattr__(self, "title_selector", title_selector)
        object.__setattr__(self, "link_selector", link_selector)
        object.__setattr__(self, "crawl_delay_seconds", delay)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        if not isinstance(value, Mapping):
            raise BlueprintValidationError("blueprint JSON must contain an object")
        fields = set(value)
        missing = sorted(_SCHEMA_FIELDS - fields)
        unknown = sorted(fields - _SCHEMA_FIELDS)
        if missing or unknown:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if unknown:
                details.append(f"unknown fields: {', '.join(unknown)}")
            raise BlueprintValidationError("; ".join(details))

        hosts = value["allowed_hosts"]
        if type(hosts) is not list or not all(type(host) is str for host in hosts):
            raise BlueprintValidationError("allowed_hosts must be an array of strings")
        login_hosts = value["login_hosts"]
        if type(login_hosts) is not list or not all(
            type(host) is str for host in login_hosts
        ):
            raise BlueprintValidationError("login_hosts must be an array of strings")
        if type(value["schema_version"]) is not int:
            raise BlueprintValidationError("schema_version must be an integer")
        if type(value["max_pages"]) is not int:
            raise BlueprintValidationError("max_pages must be an integer")
        if type(value["crawl_delay_seconds"]) not in (int, float):
            raise BlueprintValidationError("crawl_delay_seconds must be a number")
        for field in (
            "blueprint_id",
            "name",
            "content_selector",
            "title_selector",
            "link_selector",
        ):
            if type(value[field]) is not str:
                raise BlueprintValidationError(f"{field} must be a string")

        return cls(
            schema_version=value["schema_version"],
            blueprint_id=value["blueprint_id"],
            name=value["name"],
            allowed_hosts=tuple(hosts),
            login_hosts=tuple(login_hosts),
            content_selector=value["content_selector"],
            title_selector=value["title_selector"],
            link_selector=value["link_selector"],
            max_pages=value["max_pages"],
            crawl_delay_seconds=value["crawl_delay_seconds"],
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> Self:
        if not isinstance(payload, (str, bytes, bytearray)):
            raise BlueprintValidationError("blueprint JSON must be text or bytes")
        try:
            value = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
        except BlueprintValidationError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BlueprintValidationError("blueprint JSON is invalid") from exc
        return cls.from_mapping(value)

    def require_allowed_url(self, url: str, *, base_url: str | None = None) -> str:
        normalized = normalize_https_url(url, base_url=base_url)
        host = urlsplit(normalized).hostname
        if host not in self.allowed_hosts:
            raise UrlScopeError("URL host is outside the blueprint's exact allowed host set")
        return normalized

    def allows_url(self, url: str, *, base_url: str | None = None) -> bool:
        try:
            self.require_allowed_url(url, base_url=base_url)
        except (TypeError, ValueError):
            return False
        return True

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "blueprint_id": self.blueprint_id,
            "name": self.name,
            "allowed_hosts": list(self.allowed_hosts),
            "login_hosts": list(self.login_hosts),
            "content_selector": self.content_selector,
            "title_selector": self.title_selector,
            "link_selector": self.link_selector,
            "max_pages": self.max_pages,
            "crawl_delay_seconds": self.crawl_delay_seconds,
        }


@dataclass(frozen=True, slots=True)
class UnknownCheatsBlueprint(SourceBlueprint):
    """The only built-in source permitted to use vBulletin crawl behavior."""

    strategy: Literal["vbulletin"] = "vbulletin"
    vbulletin_limits: VbulletinCrawlLimits = VbulletinCrawlLimits(25, 100, 25, 50, 1_000)
    endpoint_paths: tuple[str, str] = (
        "/forum/forumdisplay.php",
        "/forum/showthread.php",
    )

    def __post_init__(self) -> None:
        SourceBlueprint.__post_init__(self)
        expected = (
            1,
            "unknowncheats",
            "UnknownCheats",
            ("unknowncheats.me", "www.unknowncheats.me"),
            ("unknowncheats.me", "www.unknowncheats.me"),
            "#posts [id^='post_message_'], #posts .post_message",
            "h1, .threadtitle",
            "a[href]",
            25,
            1.5,
            "vbulletin",
            VbulletinCrawlLimits(25, 100, 25, 50, 1_000),
            ("/forum/forumdisplay.php", "/forum/showthread.php"),
        )
        actual = (
            self.schema_version,
            self.blueprint_id,
            self.name,
            self.allowed_hosts,
            self.login_hosts,
            self.content_selector,
            self.title_selector,
            self.link_selector,
            self.max_pages,
            self.crawl_delay_seconds,
            self.strategy,
            self.vbulletin_limits,
            self.endpoint_paths,
        )
        if actual != expected:
            raise BlueprintValidationError("UnknownCheats blueprint fields are fixed")

    def require_allowed_url(self, url: str, *, base_url: str | None = None) -> str:
        normalized = SourceBlueprint.require_allowed_url(
            self,
            url,
            base_url=base_url,
        )
        parts = urlsplit(normalized)
        if parts.hostname == "unknowncheats.me":
            return urlunsplit(
                (
                    parts.scheme,
                    "www.unknowncheats.me",
                    parts.path,
                    parts.query,
                    parts.fragment,
                )
            )
        return normalized


def _builtin(
    blueprint_id: str,
    name: str,
    allowed_hosts: tuple[str, ...],
    content_selector: str,
    title_selector: str,
    *,
    max_pages: int,
    crawl_delay_seconds: float,
) -> SourceBlueprint:
    return SourceBlueprint(
        schema_version=1,
        blueprint_id=blueprint_id,
        name=name,
        allowed_hosts=allowed_hosts,
        login_hosts=allowed_hosts,
        content_selector=content_selector,
        title_selector=title_selector,
        link_selector="a[href]",
        max_pages=max_pages,
        crawl_delay_seconds=crawl_delay_seconds,
    )


def _unknowncheats_blueprint() -> UnknownCheatsBlueprint:
    return UnknownCheatsBlueprint(
        schema_version=1,
        blueprint_id="unknowncheats",
        name="UnknownCheats",
        allowed_hosts=("unknowncheats.me", "www.unknowncheats.me"),
        login_hosts=("unknowncheats.me", "www.unknowncheats.me"),
        content_selector="#posts [id^='post_message_'], #posts .post_message",
        title_selector="h1, .threadtitle",
        link_selector="a[href]",
        max_pages=25,
        crawl_delay_seconds=1.5,
    )


BUILTIN_BLUEPRINT_IDS = ("unknowncheats", "hackerone", "tryhackme", "generic")
BUILTIN_BLUEPRINTS: Mapping[str, SourceBlueprint] = MappingProxyType(
    {
        "unknowncheats": _unknowncheats_blueprint(),
        "hackerone": _builtin(
            "hackerone",
            "HackerOne",
            ("hackerone.com", "www.hackerone.com"),
            "main article, main .report, main",
            "main h1, h1",
            max_pages=25,
            crawl_delay_seconds=1.0,
        ),
        "tryhackme": _builtin(
            "tryhackme",
            "TryHackMe",
            ("tryhackme.com", "www.tryhackme.com"),
            "main article, main .markdown-body, main",
            "main h1, h1",
            max_pages=25,
            crawl_delay_seconds=1.0,
        ),
    }
)


def require_registered_blueprint_url(
    blueprint_id: str,
    url: str,
    *,
    blueprints: Mapping[str, SourceBlueprint] = BUILTIN_BLUEPRINTS,
) -> str:
    identifier = _required_string(
        blueprint_id,
        "blueprint_id",
        maximum=128,
    )
    if identifier != blueprint_id or not _BLUEPRINT_ID.fullmatch(identifier):
        raise BlueprintValidationError(
            "blueprint_id must be normalized lowercase URL-safe text"
        )
    if not isinstance(blueprints, Mapping):
        raise TypeError("blueprints must be an exact blueprint registry mapping")
    entries = tuple(blueprints.items())
    if len(entries) > 256:
        raise BlueprintValidationError("blueprint registry exceeds its item limit")
    registry: dict[str, SourceBlueprint] = {}
    for key, candidate in entries:
        if type(key) is not str or not isinstance(candidate, SourceBlueprint):
            raise BlueprintValidationError(
                "blueprint registry entries must map IDs to SourceBlueprint records"
            )
        if key != candidate.blueprint_id:
            raise BlueprintValidationError(
                "blueprint registry key does not match its blueprint identity"
            )
        if (
            key == "unknowncheats"
            and candidate is not BUILTIN_BLUEPRINTS["unknowncheats"]
        ):
            raise BlueprintValidationError(
                "only the canonical UnknownCheats blueprint may use that identity"
            )
        registry[key] = candidate
    blueprint = registry.get(identifier)
    if blueprint is None:
        raise BlueprintValidationError(
            "source target references an unknown registered blueprint"
        )
    return blueprint.require_allowed_url(url)


def generic_blueprint(
    *,
    host: str,
    login_hosts: tuple[str, ...] = (),
    content_selector: str,
    title_selector: str = "h1",
    link_selector: str = "a[href]",
    max_pages: int = 25,
    crawl_delay_seconds: float = 1.0,
    name: str = "Generic source",
) -> SourceBlueprint:
    normalized_host = _normalize_host(host)
    if not isinstance(login_hosts, tuple):
        raise BlueprintValidationError("login_hosts must be a tuple of exact host names")
    normalized_login_hosts = tuple(
        dict.fromkeys(
            (normalized_host, *(_normalize_host(value) for value in login_hosts))
        )
    )
    identifier = re.sub(r"[^a-z0-9]+", "-", normalized_host).strip("-")
    return SourceBlueprint(
        schema_version=1,
        blueprint_id=f"generic-{identifier}"[:128].rstrip("-"),
        name=name,
        allowed_hosts=(normalized_host,),
        login_hosts=normalized_login_hosts,
        content_selector=content_selector,
        title_selector=title_selector,
        link_selector=link_selector,
        max_pages=max_pages,
        crawl_delay_seconds=crawl_delay_seconds,
    )


load_blueprint_json = SourceBlueprint.from_json


__all__ = [
    "BUILTIN_BLUEPRINT_IDS",
    "BUILTIN_BLUEPRINTS",
    "BlueprintValidationError",
    "SourceBlueprint",
    "UnknownCheatsBlueprint",
    "UrlScopeError",
    "generic_blueprint",
    "load_blueprint_json",
    "normalize_https_url",
    "require_registered_blueprint_url",
]
