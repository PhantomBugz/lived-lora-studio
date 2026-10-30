"""Bounded retrieval for explicitly enabled vBulletin source attachments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import hmac
import http.client
import json
import re
import ssl
from threading import Event, Lock
from types import MappingProxyType
from typing import Protocol
from urllib.parse import urlsplit

from .source_artifacts import (
    ALLOWED_EXTENSIONS,
    ArtifactInput,
    ArtifactRejected,
    normalize_artifact_path,
    require_allowed_extension,
    validate_artifact,
)


_ATTACHMENT_PATH = "/forum/attachment.php"
_AUTHORIZED_HOST = "www.unknowncheats.me"
_POSITIVE_DECIMAL = re.compile(r"[1-9][0-9]*")
_POST_ID = re.compile(r"post_message_[1-9][0-9]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DNS_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_COOKIE_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_SAFE_CONTENT_TYPES = frozenset(
    {
        "application/javascript",
        "application/json",
        "application/octet-stream",
        "application/toml",
        "application/typescript",
        "application/xml",
        "application/x-yaml",
        "text/csv",
        "text/markdown",
        "text/plain",
        "text/x-c",
        "text/x-c++",
        "text/x-java-source",
        "text/x-python",
        "text/x-rust",
        "text/xml",
        "text/yaml",
    }
)
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')
_MAX_COOKIE_HEADER_BYTES = 8192
_MAX_HEADER_VALUE_BYTES = 4096
_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 20.0
_READ_CHUNK_BYTES = 64 * 1024
_USER_AGENT = "liveD-LoRA-Studio/1.0"

_POLICY_DOCUMENT = json.dumps(
    {
        "allowed_extensions": sorted(ALLOWED_EXTENSIONS),
        "canonical_path": _ATTACHMENT_PATH,
        "chunk_bytes": _READ_CHUNK_BYTES,
        "connect_timeout_seconds": _CONNECT_TIMEOUT_SECONDS,
        "max_file_bytes": 512 * 1024,
        "max_per_post": 8,
        "max_per_run": 100,
        "max_run_bytes": 32 * 1024 * 1024,
        "read_timeout_seconds": _READ_TIMEOUT_SECONDS,
        "redirects": "disabled",
        "schema": "lived-forum-attachment-policy-v1",
    },
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
ATTACHMENT_COLLECTOR_POLICY_SHA256 = hashlib.sha256(_POLICY_DOCUMENT).hexdigest()


class AttachmentRejected(ValueError):
    """An attachment failed one fixed, non-sensitive policy rule."""

    def __init__(self, reason_code: str):
        self.reason_code = reason_code
        super().__init__(reason_code)


class AttachmentCollectionCancelled(AttachmentRejected):
    def __init__(self):
        super().__init__("attachment_collection_cancelled")


def _reject(reason_code: str) -> None:
    raise AttachmentRejected(reason_code) from None


def _discard_exception_text(error: BaseException) -> None:
    """Best-effort removal of untrusted dependency text before fixed failure."""

    try:
        error.args = ()
    except BaseException:
        pass


def _canonical_host(value: object) -> str:
    if type(value) is not str or not value or value != value.lower():
        _reject("invalid_attachment_host")
    if value.endswith(".") or len(value) > 253 or not value.isascii():
        _reject("invalid_attachment_host")
    labels = value.split(".")
    if len(labels) < 2 or any(_DNS_LABEL.fullmatch(label) is None for label in labels):
        _reject("invalid_attachment_host")
    return value


def validate_attachment_filename(value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        _reject("unsafe_attachment_filename")
    if (
        value.startswith(".")
        or value.endswith((" ", "."))
        or len(value.encode("ascii")) > 255
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        _reject("unsafe_attachment_filename")
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        _reject("unsafe_attachment_filename")
    try:
        normalized = normalize_artifact_path(value)
        require_allowed_extension(normalized)
    except ArtifactRejected:
        _reject("unsafe_attachment_filename")
    if normalized != value:
        _reject("unsafe_attachment_filename")
    return value


def canonical_attachment_url(url: object, *, allowed_host: str) -> tuple[int, str]:
    """Return the ID and URL only when every route byte is canonical."""

    host = _canonical_host(allowed_host)
    if host != _AUTHORIZED_HOST:
        _reject("invalid_attachment_host")
    if (
        type(url) is not str
        or not url
        or not url.isascii()
        or len(url) > 4096
        or "\\" in url
        or "%" in url
        or any(ord(character) < 32 or ord(character) == 127 for character in url)
    ):
        _reject("invalid_attachment_url")
    try:
        parsed = urlsplit(url)
        parsed_port = parsed.port
    except (TypeError, ValueError):
        _reject("invalid_attachment_url")
    if (
        parsed.scheme != "https"
        or parsed.netloc != host
        or parsed.hostname != host
        or parsed_port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != _ATTACHMENT_PATH
        or parsed.fragment
    ):
        _reject("invalid_attachment_url")
    prefix = "attachmentid="
    if not parsed.query.startswith(prefix) or parsed.query.count("=") != 1:
        _reject("invalid_attachment_url")
    raw_id = parsed.query[len(prefix):]
    if _POSITIVE_DECIMAL.fullmatch(raw_id) is None:
        _reject("invalid_attachment_url")
    attachment_id = int(raw_id)
    canonical = f"https://{host}{_ATTACHMENT_PATH}?attachmentid={attachment_id}"
    if url != canonical:
        _reject("invalid_attachment_url")
    return attachment_id, canonical


@dataclass(frozen=True, slots=True)
class AttachmentDescriptor:
    attachment_id: int
    url: str
    filename: str
    parent_post_id: str
    parent_provenance_hash: str

    def __post_init__(self) -> None:
        if type(self.attachment_id) is not int or self.attachment_id < 1:
            _reject("invalid_attachment_id")
        parsed_id, canonical = canonical_attachment_url(
            self.url,
            allowed_host=_AUTHORIZED_HOST,
        )
        if parsed_id != self.attachment_id or canonical != self.url:
            _reject("invalid_attachment_url")
        validate_attachment_filename(self.filename)
        if type(self.parent_post_id) is not str or _POST_ID.fullmatch(
            self.parent_post_id
        ) is None:
            _reject("invalid_parent_post_id")
        if type(self.parent_provenance_hash) is not str or _SHA256.fullmatch(
            self.parent_provenance_hash
        ) is None:
            _reject("invalid_parent_provenance_hash")


def parse_attachment_descriptor(
    url: str,
    *,
    filename: str,
    parent_post_id: str,
    parent_provenance_hash: str,
    allowed_host: str,
) -> AttachmentDescriptor:
    attachment_id, canonical = canonical_attachment_url(
        url,
        allowed_host=allowed_host,
    )
    return AttachmentDescriptor(
        attachment_id=attachment_id,
        url=canonical,
        filename=validate_attachment_filename(filename),
        parent_post_id=parent_post_id,
        parent_provenance_hash=parent_provenance_hash,
    )


@dataclass(frozen=True, slots=True)
class AttachmentLimits:
    max_per_post: int = 8
    max_per_run: int = 100
    max_file_bytes: int = 512 * 1024
    max_run_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        values = (
            (self.max_per_post, 8),
            (self.max_per_run, 100),
            (self.max_file_bytes, 512 * 1024),
            (self.max_run_bytes, 32 * 1024 * 1024),
        )
        if any(type(value) is not int or not 1 <= value <= maximum for value, maximum in values):
            raise ValueError("attachment limits exceed fixed policy")


@dataclass(frozen=True, slots=True, repr=False)
class EphemeralCookieCapsule:
    host: str
    request_path: str
    cookie_header: bytearray
    _consume_lock: Lock = field(
        default_factory=Lock,
        init=False,
        repr=False,
        compare=False,
    )
    _consumed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _canonical_host(self.host)
        if self.request_path != _ATTACHMENT_PATH:
            _reject("invalid_cookie_capsule")
        if type(self.cookie_header) is not bytearray:
            _reject("invalid_cookie_capsule")
        if len(self.cookie_header) > _MAX_COOKIE_HEADER_BYTES:
            _reject("invalid_cookie_capsule")
        try:
            text = bytes(self.cookie_header).decode("ascii")
        except UnicodeDecodeError:
            _reject("invalid_cookie_capsule")
        if any(ord(character) < 32 or ord(character) == 127 for character in text):
            _reject("invalid_cookie_capsule")

    def __repr__(self) -> str:
        return "EphemeralCookieCapsule()"

    def _claim_once(self) -> None:
        with self._consume_lock:
            if self._consumed:
                _reject("cookie_capsule_reused")
            object.__setattr__(self, "_consumed", True)

    def zero(self) -> None:
        for index in range(len(self.cookie_header)):
            self.cookie_header[index] = 0


@dataclass(frozen=True, slots=True)
class AttachmentResponse:
    status: int
    headers: Mapping[str, str]
    chunks: Iterable[bytes]

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 100 <= self.status <= 599:
            _reject("invalid_attachment_response")
        if not isinstance(self.headers, Mapping):
            _reject("invalid_attachment_response")
        normalized: dict[str, str] = {}
        for raw_name, raw_value in self.headers.items():
            if type(raw_name) is not str or type(raw_value) is not str:
                _reject("invalid_attachment_response")
            name = raw_name.strip().lower()
            if (
                not name
                or name in normalized
                or len(raw_value.encode("utf-8")) > _MAX_HEADER_VALUE_BYTES
                or "\r" in raw_value
                or "\n" in raw_value
            ):
                _reject("invalid_attachment_response")
            normalized[name] = raw_value.strip()
        if isinstance(self.chunks, (bytes, bytearray, str)) or not isinstance(
            self.chunks, Iterable
        ):
            _reject("invalid_attachment_response")
        object.__setattr__(self, "headers", MappingProxyType(normalized))


class AttachmentTransport(Protocol):
    def fetch(
        self,
        descriptor: AttachmentDescriptor,
        *,
        cookie_capsule: EphemeralCookieCapsule,
        cancel: Event,
    ) -> AttachmentResponse: ...


class _ConnectionChunks:
    __slots__ = ("_cancel", "_connection", "_response", "_closed")

    def __init__(self, connection: object, response: object, cancel: Event) -> None:
        self._connection = connection
        self._response = response
        self._cancel = cancel
        self._closed = False

    def __iter__(self) -> _ConnectionChunks:
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        if self._cancel.is_set():
            self.close()
            raise AttachmentCollectionCancelled()
        try:
            chunk = self._response.read(_READ_CHUNK_BYTES)
        except AttachmentCollectionCancelled:
            raise
        except BaseException:
            self.close()
            _reject("attachment_stream_failed")
        if not chunk:
            self.close()
            raise StopIteration
        if type(chunk) is not bytes:
            self.close()
            _reject("invalid_attachment_chunk")
        return chunk

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._response.close()
        except BaseException:
            pass
        try:
            self._connection.close()
        except BaseException:
            pass


class HttpsAttachmentTransport:
    """One fresh, non-redirecting HTTPS connection per descriptor."""

    def __init__(self, *, ssl_context: ssl.SSLContext | None = None) -> None:
        self._ssl_context = ssl_context or ssl.create_default_context()

    def fetch(
        self,
        descriptor: AttachmentDescriptor,
        *,
        cookie_capsule: EphemeralCookieCapsule,
        cancel: Event,
    ) -> AttachmentResponse:
        if type(descriptor) is not AttachmentDescriptor:
            _reject("invalid_attachment_descriptor")
        if type(cookie_capsule) is not EphemeralCookieCapsule:
            _reject("invalid_cookie_capsule")
        if not isinstance(cancel, Event):
            raise TypeError("cancel must be a threading.Event")
        if cancel.is_set():
            raise AttachmentCollectionCancelled()
        parsed = urlsplit(descriptor.url)
        if cookie_capsule.host != parsed.hostname or cookie_capsule.request_path != parsed.path:
            _reject("cookie_scope_mismatch")
        try:
            cookie = bytes(cookie_capsule.cookie_header).decode("ascii")
        except UnicodeDecodeError:
            _reject("invalid_cookie_capsule")
        connection = None
        try:
            connection = http.client.HTTPSConnection(
                parsed.hostname,
                443,
                timeout=_CONNECT_TIMEOUT_SECONDS,
                context=self._ssl_context,
            )
            headers = {
                "Accept": "text/plain, application/octet-stream;q=0.8, */*;q=0.1",
                "Connection": "close",
                "User-Agent": _USER_AGENT,
            }
            if cookie:
                headers["Cookie"] = cookie
            target = parsed.path + "?" + parsed.query
            connection.request("GET", target, headers=headers)
            sock = getattr(connection, "sock", None)
            if sock is None or not callable(getattr(sock, "settimeout", None)):
                _reject("attachment_transport_failed")
            sock.settimeout(_READ_TIMEOUT_SECONDS)
            raw_response = connection.getresponse()
            response_headers: dict[str, str] = {}
            for raw_name, raw_value in raw_response.getheaders():
                name = str(raw_name).strip().lower()
                if name in response_headers:
                    _reject("duplicate_attachment_header")
                response_headers[name] = str(raw_value)
            return AttachmentResponse(
                raw_response.status,
                response_headers,
                _ConnectionChunks(connection, raw_response, cancel),
            )
        except AttachmentRejected:
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass
            raise
        except BaseException:
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass
            _reject("attachment_transport_failed")


def _content_disposition_filename(value: str) -> str:
    parts = [part.strip() for part in value.split(";")]
    if not parts or parts[0].lower() != "attachment" or len(parts) != 2:
        _reject("invalid_content_disposition")
    name, separator, raw = parts[1].partition("=")
    if separator != "=" or name.strip().lower() != "filename":
        _reject("invalid_content_disposition")
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] == '"':
        raw = raw[1:-1]
    elif not raw or any(character in raw for character in " \t\""):
        _reject("invalid_content_disposition")
    return validate_attachment_filename(raw)


def _validate_response_headers(
    response: AttachmentResponse,
    descriptor: AttachmentDescriptor,
    limits: AttachmentLimits,
    consumed: int,
) -> int | None:
    if 300 <= response.status <= 399:
        _reject("attachment_redirect_rejected")
    if response.status != 200:
        _reject("attachment_http_status")
    headers = response.headers
    transfer_encoding = headers.get("transfer-encoding")
    if transfer_encoding is not None and transfer_encoding.lower() != "chunked":
        _reject("attachment_transfer_encoding_rejected")
    if transfer_encoding is not None and "content-length" in headers:
        _reject("attachment_length_conflict")
    if headers.get("content-encoding", "identity").lower() != "identity":
        _reject("attachment_content_encoding_rejected")
    content_type = headers.get("content-type")
    if content_type is None:
        _reject("attachment_content_type_rejected")
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type not in _SAFE_CONTENT_TYPES:
        _reject("attachment_content_type_rejected")
    declared: int | None = None
    if "content-length" in headers:
        raw_length = headers["content-length"]
        if not re.fullmatch(r"0|[1-9][0-9]*", raw_length):
            _reject("invalid_content_length")
        declared = int(raw_length)
        if declared > limits.max_file_bytes:
            _reject("attachment_file_limit")
        if consumed + declared > limits.max_run_bytes:
            _reject("attachment_run_byte_limit")
    disposition = headers.get("content-disposition")
    if disposition is not None:
        try:
            header_name = _content_disposition_filename(disposition)
        except AttachmentRejected as error:
            if error.reason_code == "unsafe_attachment_filename":
                _reject("unsafe_attachment_filename")
            raise
        if not hmac.compare_digest(header_name, descriptor.filename):
            _reject("content_disposition_filename_mismatch")
    return declared


def _close_chunks(chunks: Iterable[bytes]) -> None:
    close = getattr(chunks, "close", None)
    if callable(close):
        try:
            close()
        except BaseException:
            pass


def _reject_binary_control_bytes(raw_bytes: bytes) -> None:
    if any(
        (value < 32 and value not in (9, 10, 13)) or value == 127
        for value in raw_bytes
    ):
        _reject("binary_control_byte")


def fetch_attachments(
    descriptors: Sequence[AttachmentDescriptor],
    *,
    capsule_factory: Callable[[AttachmentDescriptor], EphemeralCookieCapsule],
    transport: AttachmentTransport,
    limits: AttachmentLimits,
    cancel: Event,
) -> tuple[ArtifactInput, ...]:
    """Retrieve and validate a complete bounded run without partial output."""

    if not isinstance(descriptors, Sequence) or isinstance(descriptors, (str, bytes)):
        raise TypeError("descriptors must be a sequence")
    if not callable(capsule_factory):
        raise TypeError("capsule_factory must be callable")
    if not isinstance(limits, AttachmentLimits):
        raise TypeError("limits must be AttachmentLimits")
    if not isinstance(cancel, Event):
        raise TypeError("cancel must be a threading.Event")
    values = tuple(descriptors)
    if len(values) > limits.max_per_run:
        _reject("attachment_run_limit")
    counts: dict[str, int] = {}
    seen_keys: set[tuple[int, str]] = set()
    for value in values:
        if type(value) is not AttachmentDescriptor:
            _reject("invalid_attachment_descriptor")
        counts[value.parent_post_id] = counts.get(value.parent_post_id, 0) + 1
        if counts[value.parent_post_id] > limits.max_per_post:
            _reject("attachment_post_limit")
        key = (value.attachment_id, value.url)
        if key in seen_keys:
            _reject("duplicate_attachment_descriptor")
        seen_keys.add(key)

    artifacts: list[ArtifactInput] = []
    consumed = 0
    for value in values:
        if cancel.is_set():
            raise AttachmentCollectionCancelled()
        capsule: EphemeralCookieCapsule | None = None
        chunks: Iterable[bytes] | None = None
        try:
            capsule_failed = False
            try:
                capsule = capsule_factory(value)
            except BaseException as error:
                _discard_exception_text(error)
                capsule_failed = True
            if capsule_failed:
                _reject("cookie_capsule_unavailable")
            if type(capsule) is not EphemeralCookieCapsule:
                _reject("invalid_cookie_capsule")
            capsule._claim_once()
            parsed = urlsplit(value.url)
            if capsule.host != parsed.hostname or capsule.request_path != parsed.path:
                _reject("cookie_scope_mismatch")
            response: AttachmentResponse | object | None = None
            transport_failed = False
            try:
                response = transport.fetch(
                    value,
                    cookie_capsule=capsule,
                    cancel=cancel,
                )
            except AttachmentRejected:
                raise
            except BaseException as error:
                _discard_exception_text(error)
                transport_failed = True
            if transport_failed:
                _reject("attachment_transport_failed")
            if type(response) is not AttachmentResponse:
                _reject("invalid_attachment_response")
            chunks = response.chunks
            declared = _validate_response_headers(response, value, limits, consumed)
            buffer = bytearray()
            try:
                for chunk in chunks:
                    if cancel.is_set():
                        raise AttachmentCollectionCancelled()
                    if type(chunk) is not bytes:
                        _reject("invalid_attachment_chunk")
                    if len(buffer) + len(chunk) > limits.max_file_bytes:
                        _reject("attachment_file_limit")
                    if consumed + len(buffer) + len(chunk) > limits.max_run_bytes:
                        _reject("attachment_run_byte_limit")
                    buffer.extend(chunk)
            except AttachmentRejected:
                raise
            except BaseException:
                _reject("attachment_stream_failed")
            raw_bytes = bytes(buffer)
            if declared is not None and declared != len(raw_bytes):
                _reject("content_length_mismatch")
            _reject_binary_control_bytes(raw_bytes)
            raw_hash = hashlib.sha256(raw_bytes).hexdigest()
            provenance = {
                "source_kind": "forum_attachment",
                "parent_post_id": value.parent_post_id,
                "parent_post_provenance_hash": value.parent_provenance_hash,
                "attachment_id": value.attachment_id,
                "canonical_url": value.url,
                "filename": value.filename,
                "raw_byte_sha256": raw_hash,
                "collector_policy_sha256": ATTACHMENT_COLLECTOR_POLICY_SHA256,
            }
            artifact = ArtifactInput(
                "forum_attachment",
                value.filename,
                raw_bytes,
                provenance,
                3,
            )
            validated = validate_artifact(artifact)
            if not hmac.compare_digest(validated.raw_sha256, raw_hash):
                _reject("attachment_digest_mismatch")
            artifacts.append(artifact)
            consumed += len(raw_bytes)
        finally:
            if chunks is not None:
                _close_chunks(chunks)
            if capsule is not None:
                capsule.zero()
    if cancel.is_set():
        raise AttachmentCollectionCancelled()
    return tuple(artifacts)


__all__ = [
    "ATTACHMENT_COLLECTOR_POLICY_SHA256",
    "AttachmentCollectionCancelled",
    "AttachmentDescriptor",
    "AttachmentLimits",
    "AttachmentRejected",
    "AttachmentResponse",
    "AttachmentTransport",
    "EphemeralCookieCapsule",
    "HttpsAttachmentTransport",
    "canonical_attachment_url",
    "fetch_attachments",
    "parse_attachment_descriptor",
    "validate_attachment_filename",
]
