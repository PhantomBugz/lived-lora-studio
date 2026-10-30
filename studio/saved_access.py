from __future__ import annotations

from collections.abc import Mapping
from ipaddress import ip_address
import re
from typing import Any
from unicodedata import category
from urllib.parse import urlsplit
from uuid import uuid4

from .blueprints import (
    BlueprintValidationError,
    UrlScopeError,
    _validate_css_selector,
    normalize_https_url,
)


_RECORD_ID = re.compile(r"[a-f0-9]{32}")
_ABSOLUTE_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_WHATWG_IPV4_TOKEN = re.compile(r"(?:[0-9]+|0[xX][0-9A-Fa-f]*)")
_PLAYWRIGHT_ENGINE_PREFIX = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(?::[A-Za-z0-9_-]+)?=")
_MAX_RECORD_ID = 32
_MAX_LABEL = 256
_MAX_BLUEPRINT_ID = 128
_MAX_URL = 4_096
_MAX_HOSTNAME = 253
_MAX_USERNAME = 512
_MAX_PASSWORD = 4_096
_MAX_SELECTOR = 1_024


class SavedAccessValidationError(ValueError):
    """Raised when a local saved-access record is malformed or unsafe."""


def _has_control(value: str) -> bool:
    return any(category(character) in {"Cc", "Cf"} for character in value)


def _required_text(
    value: object,
    name: str,
    *,
    maximum: int,
    strip: bool = True,
) -> str:
    if type(value) is not str:
        raise SavedAccessValidationError(f"{name} must be a string")
    if _has_control(value):
        raise SavedAccessValidationError(
            f"{name} must not contain control characters"
        )
    normalized = value.strip() if strip else value
    if not normalized.strip():
        raise SavedAccessValidationError(f"{name} must be nonempty")
    try:
        size = len(normalized.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise SavedAccessValidationError(f"{name} must be valid UTF-8") from error
    if size > maximum:
        raise SavedAccessValidationError(
            f"{name} exceeds the UTF-8 size limit"
        )
    return normalized


def _reject_browser_divergent_host(value: str) -> None:
    if any(ord(character) > 127 for character in value):
        raise SavedAccessValidationError(
            "hostname must be an ASCII DNS name"
        )
    if any(character in value for character in "\\/%:@*?#") or value.isspace():
        raise SavedAccessValidationError("hostname is invalid")
    if any(label.casefold().startswith("xn--") for label in value.split(".")):
        raise SavedAccessValidationError(
            "internationalized hostname labels are not supported"
        )
    try:
        ip_address(value)
    except ValueError:
        labels = value.split(".")
        if (
            labels
            and _WHATWG_IPV4_TOKEN.fullmatch(labels[-1]) is not None
        ):
            raise SavedAccessValidationError(
                "hostname must not use a legacy numeric address"
            )
    else:
        raise SavedAccessValidationError(
            "hostname must be an exact DNS name, not an IP address"
        )


def _canonical_hostname(value: object) -> str:
    raw_hostname = _required_text(
        value,
        "hostname",
        maximum=_MAX_HOSTNAME,
    )
    _reject_browser_divergent_host(raw_hostname)
    hostname = raw_hostname.casefold()
    try:
        normalized = normalize_https_url(f"https://{hostname}/")
    except (BlueprintValidationError, UrlScopeError, TypeError, ValueError) as error:
        raise SavedAccessValidationError("hostname is invalid") from error
    canonical = urlsplit(normalized).hostname
    if canonical is None:
        raise SavedAccessValidationError("hostname is invalid")
    return canonical


def _canonical_https_url(value: object, *, name: str) -> tuple[str, str]:
    url = _required_text(value, name, maximum=_MAX_URL)
    try:
        raw_parts = urlsplit(url)
    except ValueError as error:
        raise SavedAccessValidationError(f"{name} is invalid") from error
    if raw_parts.fragment:
        raise SavedAccessValidationError(
            f"{name} must not contain a fragment"
        )
    raw_hostname = raw_parts.hostname
    if raw_hostname is None:
        raise SavedAccessValidationError(f"{name} requires a host")
    _reject_browser_divergent_host(raw_hostname)
    try:
        canonical = normalize_https_url(url)
    except (BlueprintValidationError, UrlScopeError, TypeError, ValueError) as error:
        raise SavedAccessValidationError(f"{name} is invalid") from error
    hostname = urlsplit(canonical).hostname
    if hostname is None:
        raise SavedAccessValidationError(f"{name} requires a host")
    return canonical, hostname


def _has_unquoted_playwright_chain(selector: str) -> bool:
    quote_character: str | None = None
    escaped = False
    for index, character in enumerate(selector):
        if escaped:
            escaped = False
            continue
        if character == chr(92):
            escaped = True
            continue
        if quote_character is not None:
            if character == quote_character:
                quote_character = None
            continue
        if character in {"'", '"'}:
            quote_character = character
            continue
        if character == ">" and selector[index : index + 2] == ">>":
            return True
    return False


def _canonical_selector(value: object, name: str) -> str:
    selector = _required_text(value, name, maximum=_MAX_SELECTOR)
    if (
        _ABSOLUTE_URL.match(selector)
        or selector.startswith(("//", ".//", ".."))
        or _has_unquoted_playwright_chain(selector)
        or _PLAYWRIGHT_ENGINE_PREFIX.match(selector)
    ):
        raise SavedAccessValidationError(
            f"{name} must be a local CSS selector"
        )
    try:
        return _validate_css_selector(selector, name)
    except BlueprintValidationError as error:
        raise SavedAccessValidationError(str(error)) from error


def _require_record_id(value: object) -> str:
    record_id = _required_text(
        value,
        "record_id",
        maximum=_MAX_RECORD_ID,
    )
    if not _RECORD_ID.fullmatch(record_id):
        raise SavedAccessValidationError(
            "record_id must be a lowercase opaque identifier"
        )
    return record_id


def _strict_state(
    value: object,
    required_keys: frozenset[str],
) -> dict[str, object]:
    if type(value) is not dict or set(value) != required_keys:
        raise SavedAccessValidationError(
            "saved-access state has an invalid schema"
        )
    return value


class SavedSource:
    """Immutable saved source; serialization requires explicit to_state()."""

    __slots__ = (
        "record_id",
        "label",
        "blueprint_id",
        "url",
        "hostname",
        "__sealed",
    )

    def __init__(
        self,
        record_id: str,
        label: str,
        blueprint_id: str,
        url: str,
        hostname: str,
    ) -> None:
        canonical_record_id = _require_record_id(record_id)
        canonical_label = _required_text(label, "label", maximum=_MAX_LABEL)
        canonical_blueprint_id = _required_text(
            blueprint_id,
            "blueprint_id",
            maximum=_MAX_BLUEPRINT_ID,
        )
        canonical_url, url_hostname = _canonical_https_url(url, name="url")
        canonical_hostname = _canonical_hostname(hostname)
        if canonical_url != url:
            raise SavedAccessValidationError("url must be canonical")
        if canonical_hostname != hostname or url_hostname != canonical_hostname:
            raise SavedAccessValidationError("hostname must match url")
        object.__setattr__(self, "record_id", canonical_record_id)
        object.__setattr__(self, "label", canonical_label)
        object.__setattr__(self, "blueprint_id", canonical_blueprint_id)
        object.__setattr__(self, "url", canonical_url)
        object.__setattr__(self, "hostname", canonical_hostname)
        object.__setattr__(self, "_SavedSource__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_SavedSource__sealed", False):
            raise AttributeError("SavedSource is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SavedSource is immutable")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("SavedSource requires explicit to_state() serialization")

    def __getstate__(self) -> object:
        raise TypeError("SavedSource requires explicit to_state() serialization")

    def __copy__(self) -> "SavedSource":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "SavedSource":
        return self

    @classmethod
    def create(
        cls,
        *,
        blueprint_id: str,
        url: str,
        label: str | None = None,
    ) -> "SavedSource":
        canonical_url, hostname = _canonical_https_url(url, name="url")
        return cls(
            record_id=uuid4().hex,
            label=(
                hostname
                if label is None
                else _required_text(label, "label", maximum=_MAX_LABEL)
            ),
            blueprint_id=_required_text(
                blueprint_id,
                "blueprint_id",
                maximum=_MAX_BLUEPRINT_ID,
            ),
            url=canonical_url,
            hostname=hostname,
        )

    def to_state(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "label": self.label,
            "blueprint_id": self.blueprint_id,
            "url": self.url,
            "hostname": self.hostname,
        }

    @classmethod
    def from_state(cls, state: object) -> "SavedSource":
        values = _strict_state(
            state,
            frozenset(
                {
                    "record_id",
                    "label",
                    "blueprint_id",
                    "url",
                    "hostname",
                }
            ),
        )
        return cls(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        return self is other or (
            type(other) is SavedSource
            and self.record_id == other.record_id
            and self.label == other.label
            and self.blueprint_id == other.blueprint_id
            and self.url == other.url
            and self.hostname == other.hostname
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.record_id,
                self.label,
                self.blueprint_id,
                self.url,
                self.hostname,
            )
        )

    def __repr__(self) -> str:
        return (
            "SavedSource("
            f"record_id={self.record_id!r}, "
            f"label={self.label!r}, "
            f"blueprint_id={self.blueprint_id!r}, "
            f"hostname={self.hostname!r})"
        )


class SavedLogin:
    """Immutable login record with explicit password access."""

    __slots__ = (
        "record_id",
        "label",
        "hostname",
        "login_url",
        "__username",
        "username_selector",
        "password_selector",
        "__password",
        "__sealed",
    )

    def __init__(
        self,
        record_id: str,
        label: str,
        hostname: str,
        login_url: str,
        username: str,
        password: str,
        username_selector: str,
        password_selector: str,
    ) -> None:
        canonical_record_id = _require_record_id(record_id)
        canonical_label = _required_text(label, "label", maximum=_MAX_LABEL)
        canonical_hostname = _canonical_hostname(hostname)
        canonical_url, url_hostname = _canonical_https_url(
            login_url,
            name="login_url",
        )
        if canonical_url != login_url:
            raise SavedAccessValidationError(
                "login_url must be canonical"
            )
        if canonical_hostname != hostname or url_hostname != canonical_hostname:
            raise SavedAccessValidationError(
                "login_url hostname must match hostname"
            )
        canonical_username = _required_text(
            username,
            "username",
            maximum=_MAX_USERNAME,
            strip=False,
        )
        canonical_password = _required_text(
            password,
            "password",
            maximum=_MAX_PASSWORD,
            strip=False,
        )
        canonical_username_selector = _canonical_selector(
            username_selector,
            "username_selector",
        )
        canonical_password_selector = _canonical_selector(
            password_selector,
            "password_selector",
        )
        if canonical_username_selector != username_selector:
            raise SavedAccessValidationError(
                "username_selector must be canonical"
            )
        if canonical_password_selector != password_selector:
            raise SavedAccessValidationError(
                "password_selector must be canonical"
            )

        object.__setattr__(self, "record_id", canonical_record_id)
        object.__setattr__(self, "label", canonical_label)
        object.__setattr__(self, "hostname", canonical_hostname)
        object.__setattr__(self, "login_url", canonical_url)
        object.__setattr__(
            self,
            "_SavedLogin__username",
            canonical_username,
        )
        object.__setattr__(
            self,
            "username_selector",
            canonical_username_selector,
        )
        object.__setattr__(
            self,
            "password_selector",
            canonical_password_selector,
        )
        object.__setattr__(
            self,
            "_SavedLogin__password",
            canonical_password,
        )
        object.__setattr__(self, "_SavedLogin__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_SavedLogin__sealed", False):
            raise AttributeError("SavedLogin is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("SavedLogin is immutable")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("SavedLogin secrets cannot be pickled")

    def __getstate__(self) -> object:
        raise TypeError("SavedLogin secrets cannot be pickled")

    def __copy__(self) -> "SavedLogin":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "SavedLogin":
        return self

    @classmethod
    def create(
        cls,
        *,
        hostname: str,
        login_url: str,
        username: str,
        password: str,
        username_selector: str,
        password_selector: str,
        label: str | None = None,
    ) -> "SavedLogin":
        canonical_hostname = _canonical_hostname(hostname)
        canonical_url, url_hostname = _canonical_https_url(
            login_url,
            name="login_url",
        )
        if url_hostname != canonical_hostname:
            raise SavedAccessValidationError(
                "login_url hostname must match hostname"
            )
        return cls(
            record_id=uuid4().hex,
            label=(
                canonical_hostname
                if label is None
                else _required_text(label, "label", maximum=_MAX_LABEL)
            ),
            hostname=canonical_hostname,
            login_url=canonical_url,
            username=_required_text(
                username,
                "username",
                maximum=_MAX_USERNAME,
                strip=False,
            ),
            password=_required_text(
                password,
                "password",
                maximum=_MAX_PASSWORD,
                strip=False,
            ),
            username_selector=_canonical_selector(
                username_selector,
                "username_selector",
            ),
            password_selector=_canonical_selector(
                password_selector,
                "password_selector",
            ),
        )

    def secret_username(self) -> str:
        """Return the username only for the local browser adapter."""

        return self.__username

    def secret_password(self) -> str:
        """Return the password only for the local browser adapter."""

        return self.__password

    def secret_state(self) -> dict[str, str]:
        """Return secret-bearing state only for the encrypted local vault."""

        return {
            "record_id": self.record_id,
            "label": self.label,
            "hostname": self.hostname,
            "login_url": self.login_url,
            "username": self.secret_username(),
            "password": self.secret_password(),
            "username_selector": self.username_selector,
            "password_selector": self.password_selector,
        }

    def masked_display(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "label": self.label,
            "hostname": self.hostname,
            "login_url": self.login_url,
            "username": "********",
            "password": "********",
            "username_selector": self.username_selector,
            "password_selector": self.password_selector,
        }

    @classmethod
    def from_state(cls, state: object) -> "SavedLogin":
        values = _strict_state(
            state,
            frozenset(
                {
                    "record_id",
                    "label",
                    "hostname",
                    "login_url",
                    "username",
                    "password",
                    "username_selector",
                    "password_selector",
                }
            ),
        )
        return cls(**values)  # type: ignore[arg-type]

    def __eq__(self, other: object) -> bool:
        return self is other or (
            type(other) is SavedLogin
            and self.record_id == other.record_id
            and self.label == other.label
            and self.hostname == other.hostname
            and self.login_url == other.login_url
            and self.__username == other.__username
            and self.__password == other.__password
            and self.username_selector == other.username_selector
            and self.password_selector == other.password_selector
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.record_id,
                self.label,
                self.hostname,
                self.login_url,
                self.__username,
                self.__password,
                self.username_selector,
                self.password_selector,
            )
        )

    def __repr__(self) -> str:
        return (
            "SavedLogin("
            f"record_id={self.record_id!r}, "
            f"label={self.label!r}, "
            f"hostname={self.hostname!r})"
        )


__all__ = ["SavedAccessValidationError", "SavedLogin", "SavedSource"]
