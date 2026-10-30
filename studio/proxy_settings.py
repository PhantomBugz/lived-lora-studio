from __future__ import annotations

from ipaddress import ip_address
import re
from unicodedata import category
from urllib.parse import quote, unquote_to_bytes, urlsplit


_MAX_RECORD_ID = 128
_MAX_LABEL = 256
_MAX_PROXY_URI = 2_048
_MAX_CREDENTIAL = 255
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_WHATWG_IPV4_TOKEN = re.compile(r"(?:[0-9]+|0[xX][0-9A-Fa-f]*)")
_PORT = re.compile(r"[1-9][0-9]{0,4}")
_USERINFO = re.compile(r"(?:[A-Za-z0-9._~-]|%[0-9A-Fa-f]{2})+")


class ProxySettingsValidationError(ValueError):
    """Raised when a local proxy record is malformed or ambiguous."""


def _has_control(value: str) -> bool:
    return any(category(character) in {"Cc", "Cf"} for character in value)


def _required_text(value: object, name: str, maximum: int) -> str:
    if type(value) is not str:
        raise ProxySettingsValidationError(f"{name} must be a string")
    if not value or value != value.strip() or _has_control(value):
        raise ProxySettingsValidationError(f"{name} must be nonempty and canonical")
    try:
        if len(value.encode("utf-8")) > maximum:
            raise ProxySettingsValidationError(f"{name} exceeds the size limit")
    except UnicodeEncodeError as error:
        raise ProxySettingsValidationError(f"{name} must be valid UTF-8") from error
    return value


def _canonical_hostname(value: str) -> str:
    try:
        address = ip_address(value)
    except ValueError:
        if any(ord(character) > 127 for character in value):
            raise ProxySettingsValidationError(
                "proxy host must use an ASCII DNS name or numeric IP address"
            )
        labels = value.split(".")
        if any(label.casefold().startswith("xn--") for label in labels):
            raise ProxySettingsValidationError(
                "internationalized proxy host labels are not supported"
            )
        if (
            labels
            and _WHATWG_IPV4_TOKEN.fullmatch(labels[-1]) is not None
        ):
            raise ProxySettingsValidationError(
                "proxy host is an ambiguous numeric address"
            )
        hostname = value.lower()
        labels = hostname.split(".")
        if (
            not hostname
            or hostname.endswith(".")
            or len(hostname) > 253
            or len(labels) < 2
            or any(not _DNS_LABEL.fullmatch(label) for label in labels)
        ):
            raise ProxySettingsValidationError("proxy host is invalid")
        return hostname
    return address.compressed


def _decoded_credential(value: str, name: str) -> str:
    if not _USERINFO.fullmatch(value):
        raise ProxySettingsValidationError(
            f"{name} must use URL encoding for reserved characters"
        )
    try:
        decoded = unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ProxySettingsValidationError(f"{name} must be valid UTF-8") from error
    if not decoded or _has_control(decoded):
        raise ProxySettingsValidationError(f"{name} is invalid")
    try:
        if len(decoded.encode("utf-8")) > _MAX_CREDENTIAL:
            raise ProxySettingsValidationError(f"{name} exceeds the size limit")
    except UnicodeEncodeError as error:
        raise ProxySettingsValidationError(f"{name} must be valid UTF-8") from error
    return decoded


def _split_authority(netloc: str) -> tuple[str | None, str | None, str, int]:
    if netloc.count("@") > 1:
        raise ProxySettingsValidationError("proxy_uri userinfo is ambiguous")
    userinfo, separator, hostport = netloc.rpartition("@")
    username: str | None = None
    password: str | None = None
    if separator:
        if userinfo.count(":") != 1:
            raise ProxySettingsValidationError(
                "proxy_uri requires both username and password"
            )
        raw_username, raw_password = userinfo.split(":", 1)
        username = _decoded_credential(raw_username, "proxy username")
        password = _decoded_credential(raw_password, "proxy password")
    else:
        hostport = netloc

    if hostport.startswith("["):
        close = hostport.find("]")
        if close < 0 or close + 1 >= len(hostport) or hostport[close + 1] != ":":
            raise ProxySettingsValidationError(
                "proxy_uri requires an explicit host and port"
            )
        raw_host = hostport[1:close]
        raw_port = hostport[close + 2 :]
        if "]" in raw_host or hostport.count("]") != 1:
            raise ProxySettingsValidationError("proxy_uri host is invalid")
        if "%" in raw_host:
            raise ProxySettingsValidationError(
                "scoped IPv6 proxy hosts are not supported"
            )
        try:
            if ip_address(raw_host).version != 6:
                raise ProxySettingsValidationError(
                    "bracketed proxy host must be an IPv6 address"
                )
        except ValueError as error:
            raise ProxySettingsValidationError(
                "bracketed proxy host must be an IPv6 address"
            ) from error
    else:
        if hostport.count(":") != 1:
            raise ProxySettingsValidationError(
                "proxy_uri requires an explicit host and port"
            )
        raw_host, raw_port = hostport.rsplit(":", 1)

    if (
        not raw_host
        or not _PORT.fullmatch(raw_port)
        or not 1 <= int(raw_port) <= 65_535
    ):
        raise ProxySettingsValidationError(
            "proxy_uri requires a port from 1 through 65535"
        )
    return username, password, _canonical_hostname(raw_host), int(raw_port)


def _validate_proxy_uri(value: object) -> str:
    uri = _required_text(value, "proxy_uri", _MAX_PROXY_URI)
    if "\\" in uri or "?" in uri or "#" in uri:
        raise ProxySettingsValidationError("proxy_uri contains invalid characters")
    try:
        parsed = urlsplit(uri)
    except ValueError as error:
        raise ProxySettingsValidationError("proxy_uri is invalid") from error
    if parsed.scheme not in {"http", "https", "socks5"} or not parsed.netloc:
        raise ProxySettingsValidationError(
            "proxy_uri must use http, https, or socks5"
        )
    if parsed.path or parsed.query or parsed.fragment:
        raise ProxySettingsValidationError(
            "proxy_uri must not contain a path, query, or fragment"
        )

    username, password, expected_host, expected_port = _split_authority(
        parsed.netloc
    )
    try:
        parsed_port = parsed.port
    except ValueError as error:
        raise ProxySettingsValidationError("proxy_uri has an invalid port") from error
    if (
        parsed.hostname is None
        or parsed_port != expected_port
        or _canonical_hostname(parsed.hostname) != expected_host
    ):
        raise ProxySettingsValidationError(
            "proxy_uri requires an explicit host and port"
        )

    host_for_uri = (
        f"[{expected_host}]" if ":" in expected_host else expected_host
    )
    authentication = ""
    if username is not None:
        authentication = (
            quote(username, safe="-._~")
            + ":"
            + quote(password or "", safe="-._~")
            + "@"
        )
    return (
        f"{parsed.scheme.casefold()}://{authentication}"
        f"{host_for_uri}:{expected_port}"
    )


class ProxySettings:
    """Immutable scraper-only proxy configuration with explicit secret access."""

    __slots__ = ("record_id", "label", "__proxy_uri", "__sealed")

    def __init__(self, record_id: str, label: str, proxy_uri: str) -> None:
        object.__setattr__(
            self,
            "record_id",
            _required_text(record_id, "record_id", _MAX_RECORD_ID),
        )
        object.__setattr__(
            self,
            "label",
            _required_text(label, "label", _MAX_LABEL),
        )
        object.__setattr__(
            self,
            "_ProxySettings__proxy_uri",
            _validate_proxy_uri(proxy_uri),
        )
        object.__setattr__(self, "_ProxySettings__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_ProxySettings__sealed", False):
            raise AttributeError("ProxySettings is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("ProxySettings is immutable")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("ProxySettings secrets cannot be pickled")

    def __getstate__(self) -> object:
        raise TypeError("ProxySettings secrets cannot be pickled")

    def __copy__(self) -> "ProxySettings":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "ProxySettings":
        return self

    def _parts(self):
        return urlsplit(self.__proxy_uri)

    @property
    def endpoint(self) -> str:
        parts = self._parts()
        host = _canonical_hostname(parts.hostname or "")
        authority = f"[{host}]" if ":" in host else host
        return f"{parts.scheme}://{authority}:{parts.port}"

    def transport_fields(self) -> dict[str, str]:
        parts = self._parts()
        return {
            "scheme": parts.scheme,
            "hostname": _canonical_hostname(parts.hostname or ""),
            "port": str(parts.port),
        }

    def secret_authentication(self) -> tuple[str | None, str | None]:
        """Return decoded credentials only for the local upstream adapter."""

        parts = self._parts()
        if parts.username is None:
            return None, None
        return (
            _decoded_credential(parts.username, "proxy username"),
            _decoded_credential(parts.password or "", "proxy password"),
        )

    def masked_display(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "label": self.label,
            "server": self.endpoint,
        }

    def secret_proxy_uri(self) -> str:
        """Return the full URI only for the encrypted vault."""

        return self.__proxy_uri

    def secret_state(self) -> dict[str, str]:
        """Return secret-bearing state only for the encrypted local vault."""

        return {
            "record_id": self.record_id,
            "label": self.label,
            "proxy_uri": self.secret_proxy_uri(),
        }

    @classmethod
    def from_state(cls, state: object) -> "ProxySettings":
        if type(state) is not dict or set(state) != {
            "record_id",
            "label",
            "proxy_uri",
        }:
            raise ProxySettingsValidationError(
                "proxy settings state has an invalid schema"
            )
        return cls(
            record_id=state["record_id"],  # type: ignore[arg-type]
            label=state["label"],  # type: ignore[arg-type]
            proxy_uri=state["proxy_uri"],  # type: ignore[arg-type]
        )

    def __eq__(self, other: object) -> bool:
        return self is other or (
            type(other) is ProxySettings
            and self.record_id == other.record_id
            and self.label == other.label
            and self.__proxy_uri == other.__proxy_uri
        )

    def __hash__(self) -> int:
        return hash((self.record_id, self.label, self.__proxy_uri))

    def __repr__(self) -> str:
        display = self.masked_display()
        return (
            "ProxySettings("
            f"record_id={display['record_id']!r}, "
            f"label={display['label']!r}, "
            f"server={display['server']!r})"
        )


__all__ = ["ProxySettings", "ProxySettingsValidationError"]
