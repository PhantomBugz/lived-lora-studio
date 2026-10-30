"""Immutable metadata and exact-commit rights for GitHub source inspection.

These values contain metadata only. Network inspection and collection are owned
by the collector; constructing a value does not acquire repository content.
"""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import hmac
import http.client
import json
import re
import socket
import ssl
from threading import Event
from types import MappingProxyType
from typing import Callable, Literal, Mapping
from urllib.parse import quote, urlencode, urlsplit

from .source_artifacts import (
    ArtifactInput, ArtifactRejected, normalize_artifact_path,
    require_allowed_extension, validate_artifact,
)


_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]{1,100}")
_NODE_ID = re.compile(r"[A-Za-z0-9_=-]{1,128}")
_SPDX = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,127}")
_SHA1 = re.compile(r"[a-f0-9]{40}")
_SHA256 = re.compile(r"[a-f0-9]{64}")
_RECORD_ID = re.compile(r"[a-f0-9]{32}")
_RIGHTS_SCHEMA = "LIVED_REPOSITORY_RIGHTS_V1"
_RIGHTS_DOMAIN = b"liveD:repository-rights:v1\x00"
API_HOST = "api.github.com"
FIXED_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "liveD-LoRA-Studio/1.0",
}
MAX_JSON_BYTES = 8 * 1024 * 1024
_RETRY_STATUSES = frozenset({429, 502, 503, 504})


def _require_pattern(value: object, pattern: re.Pattern, name: str) -> None:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {name}")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def strict_json(payload: bytes) -> object:
    if type(payload) is not bytes:
        raise TypeError("GitHub JSON payload must be bytes")
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("GitHub response exceeds byte limit")
    try:
        text = payload.decode("utf-8", errors="strict")
        return json.loads(
            text, object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid GitHub JSON response") from None


@dataclass(frozen=True, slots=True)
class GitHubRequest:
    method: Literal["GET"]
    path: str
    query: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.method != "GET":
            raise ValueError("GitHub requests must use GET")
        if (
            type(self.path) is not str or not self.path.startswith("/")
            or self.path.startswith("//") or "?" in self.path or "#" in self.path
            or "\\" in self.path or any(ord(char) < 33 or ord(char) == 127 for char in self.path)
        ):
            raise ValueError("invalid GitHub origin path")
        if type(self.query) is not tuple:
            raise TypeError("GitHub query must be an immutable tuple")
        for item in self.query:
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("invalid GitHub query pair")
            for value in item:
                if type(value) is not str or any(ord(char) < 32 or ord(char) == 127 for char in value):
                    raise ValueError("invalid GitHub query value")


@dataclass(frozen=True, slots=True)
class GitHubResponse:
    status: int
    headers: Mapping[str, str]
    payload: bytes

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 100 <= self.status <= 599:
            raise ValueError("invalid GitHub response status")
        if type(self.payload) is not bytes:
            raise TypeError("invalid GitHub response payload")
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))


@dataclass(frozen=True, slots=True)
class GitHubSearchQuery:
    text: str

    def __post_init__(self) -> None:
        if (
            type(self.text) is not str or not self.text or self.text.strip() != self.text
            or len(self.text.encode("utf-8", errors="strict")) > 512
            or any(ord(char) < 32 or ord(char) == 127 for char in self.text)
        ):
            raise ValueError("invalid GitHub search query")


@dataclass(frozen=True, slots=True)
class CollectionPaused:
    resume_at_utc: str
    reason: Literal["github_rate_limit"]

    def __post_init__(self) -> None:
        if self.reason != "github_rate_limit":
            raise ValueError("invalid pause reason")
        try:
            timestamp = datetime.fromisoformat(self.resume_at_utc)
        except (TypeError, ValueError):
            raise ValueError("invalid pause time") from None
        if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
            raise ValueError("pause time must be UTC")


class CollectionCancelled(RuntimeError):
    """Collection was stopped through its request-local cancellation event."""


class GitHubRateLimitPaused(RuntimeError):
    """A server rate-limit delay exceeded the bounded in-process wait."""

    def __init__(self, paused: CollectionPaused):
        self.paused = paused
        super().__init__("GitHub collection paused for rate limit")


class GitHubTransport:
    """Direct, fixed-host GitHub HTTPS transport with bounded retries."""

    def __init__(self, *, connection_factory: Callable[..., object] | None = None):
        self._connection_factory = connection_factory or http.client.HTTPSConnection

    @staticmethod
    def _headers(response: object) -> dict[str, str]:
        values: dict[str, str] = {}
        for name, value in response.getheaders():
            lowered = str(name).lower()
            if lowered in values:
                raise ValueError("duplicate GitHub response header")
            values[lowered] = str(value)
        return values

    @staticmethod
    def _rate_limit_delay(headers: Mapping[str, str]) -> tuple[float, datetime] | None:
        now = datetime.now(timezone.utc)
        candidates: list[datetime] = []
        retry_after = headers.get("retry-after")
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except ValueError:
                try:
                    retry_time = parsedate_to_datetime(retry_after)
                except (TypeError, ValueError):
                    raise ValueError("invalid GitHub rate-limit response") from None
                if retry_time.tzinfo is None:
                    raise ValueError("invalid GitHub rate-limit response")
                candidates.append(retry_time.astimezone(timezone.utc))
            else:
                if not 0 <= seconds <= 86_400:
                    raise ValueError("invalid GitHub rate-limit response")
                candidates.append(now + timedelta(seconds=seconds))
        reset = headers.get("x-ratelimit-reset")
        if reset is not None:
            try:
                resume = datetime.fromtimestamp(int(reset), timezone.utc)
            except (OverflowError, ValueError):
                raise ValueError("invalid GitHub rate-limit response") from None
            candidates.append(resume)
        if not candidates:
            return None
        resume = max(candidates)
        return max(0.0, (resume - now).total_seconds()), resume

    @staticmethod
    def _is_rate_limited(status: int, headers: Mapping[str, str]) -> bool:
        if status == 429:
            return True
        if status != 403:
            return False
        remaining = headers.get("x-ratelimit-remaining")
        if remaining is not None and (not remaining.isascii() or not remaining.isdigit()):
            raise ValueError("invalid GitHub rate-limit response")
        if "retry-after" in headers or remaining == "0":
            return True
        return "x-ratelimit-reset" in headers and remaining is None

    def _once(self, request: GitHubRequest, *, token: str | None) -> GitHubResponse:
        if token is not None and (
            type(token) is not str or not token
            or any(ord(char) < 33 or ord(char) == 127 for char in token)
        ):
            raise ValueError("invalid GitHub token")
        target = request.path
        if request.query:
            target += "?" + urlencode(request.query)
        headers = {**FIXED_HEADERS, "Connection": "close"}
        connection = None
        payload = bytearray()
        try:
            if token is not None:
                headers["Authorization"] = f"Bearer {token}"
            connection = self._connection_factory(
                API_HOST, 443, timeout=5.0, context=ssl.create_default_context(),
            )
            connection.connect()
            sock = getattr(connection, "sock", None)
            if sock is None:
                raise RuntimeError("GitHub HTTPS connection has no socket")
            sock.settimeout(20.0)
            connection.request("GET", target, body=None, headers=headers)
            response = connection.getresponse()
            response_headers = self._headers(response)
            if 300 <= response.status <= 399 or "location" in response_headers:
                raise ValueError("GitHub redirects are not accepted")
            while True:
                chunk = response.read(min(64 * 1024, MAX_JSON_BYTES + 1 - len(payload)))
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > MAX_JSON_BYTES:
                    raise ValueError("GitHub response exceeds byte limit")
            return GitHubResponse(response.status, response_headers, bytes(payload))
        finally:
            headers.pop("Authorization", None)
            payload.clear()
            if connection is not None:
                connection.close()

    def request(self, request: GitHubRequest, *, token: str | None, cancel: Event) -> GitHubResponse:
        if type(request) is not GitHubRequest:
            raise TypeError("invalid GitHub request")
        if not hasattr(cancel, "is_set") or not hasattr(cancel, "wait"):
            raise TypeError("invalid GitHub cancellation event")
        for attempt in range(3):
            if cancel.is_set():
                raise CollectionCancelled("GitHub collection cancelled")
            try:
                response = self._once(request, token=token)
            except (TimeoutError, socket.timeout):
                if attempt == 2:
                    raise RuntimeError("GitHub transport timed out") from None
                if cancel.wait(0):
                    raise CollectionCancelled("GitHub collection cancelled")
                continue
            except Exception as error:
                if token is not None and token in str(error):
                    raise RuntimeError("GitHub transport failed") from None
                raise
            rate_limited = self._is_rate_limited(response.status, response.headers)
            if rate_limited:
                wait = self._rate_limit_delay(response.headers)
                if wait is not None:
                    seconds, resume = wait
                    if seconds > 60:
                        raise GitHubRateLimitPaused(CollectionPaused(
                            resume.isoformat(timespec="seconds"), "github_rate_limit",
                        ))
                    if cancel.wait(seconds):
                        raise CollectionCancelled("GitHub collection cancelled")
                    if attempt == 2:
                        return response
                    continue
            if response.status not in _RETRY_STATUSES and not rate_limited:
                return response
            if attempt == 2:
                return response
            if cancel.wait(0):
                raise CollectionCancelled("GitHub collection cancelled")
        raise RuntimeError("unreachable GitHub transport state")


class GitHubRestClient:
    """Strict JSON facade over the fixed-host transport."""

    def __init__(
        self, transport: GitHubTransport, *, token: str | None = None,
        cancel: Event | None = None,
    ):
        if type(transport) is not GitHubTransport:
            raise TypeError("invalid GitHub transport")
        self._transport = transport
        self._token = token
        self._cancel = cancel or Event()

    def bound(self, *, token: str | None, cancel: Event) -> GitHubRestClient:
        return GitHubRestClient(self._transport, token=token, cancel=cancel)

    def _json(self, request: GitHubRequest) -> tuple[int, object]:
        response = self._transport.request(request, token=self._token, cancel=self._cancel)
        return response.status, strict_json(response.payload)

    def search(self, query: GitHubSearchQuery, page: int) -> tuple[RepositoryLocator, ...]:
        if type(query) is not GitHubSearchQuery:
            raise TypeError("invalid GitHub search")
        if type(page) is not int or not 1 <= page <= 10:
            raise ValueError("GitHub search page exceeds its bound")
        status, payload = self._json(GitHubRequest(
            "GET", "/search/repositories",
            (("q", query.text), ("per_page", "100"), ("page", str(page))),
        ))
        if status != 200 or type(payload) is not dict or type(payload.get("items")) is not list:
            raise ValueError("invalid GitHub search response")
        items = payload["items"]
        if len(items) > 100:
            raise ValueError("GitHub search response exceeds item limit")
        result = []
        for item in items:
            if type(item) is not dict or type(item.get("owner")) is not dict:
                raise ValueError("invalid GitHub search item")
            locator = RepositoryLocator(item["owner"].get("login"), item.get("name"))
            if item.get("full_name") != f"{locator.owner}/{locator.repository}":
                raise ValueError("GitHub search identity mismatch")
            expected_url = f"https://{API_HOST}/repos/{locator.owner}/{locator.repository}"
            if item.get("url") != expected_url:
                raise ValueError("GitHub search route mismatch")
            result.append(locator)
        return tuple(result)

    @staticmethod
    def _base(identity: RepositoryIdentity) -> str:
        locator = identity.locator
        return f"https://{API_HOST}/repos/{locator.owner}/{locator.repository}"

    def repository(self, locator: RepositoryLocator) -> RepositoryMetadata:
        if type(locator) is not RepositoryLocator:
            raise TypeError("invalid repository locator")
        path = f"/repos/{quote(locator.owner, safe='')}/{quote(locator.repository, safe='')}"
        status, payload = self._json(GitHubRequest("GET", path))
        if status != 200 or type(payload) is not dict:
            raise ValueError("invalid GitHub repository response")
        owner = payload.get("owner")
        if type(owner) is not dict:
            raise ValueError("invalid GitHub repository owner")
        returned = RepositoryLocator(owner.get("login"), payload.get("name"))
        if returned != locator or payload.get("full_name") != f"{locator.owner}/{locator.repository}":
            raise ValueError("GitHub repository identity mismatch")
        identity = RepositoryIdentity(payload.get("id"), payload.get("node_id"), returned)
        base = self._base(identity)
        return RepositoryMetadata(
            identity, payload.get("default_branch"), payload.get("url"),
            payload.get("license_url"),
        )

    def license(self, identity: RepositoryIdentity, commit_sha: str) -> LicenseMetadata | None:
        if type(identity) is not RepositoryIdentity:
            raise TypeError("invalid repository identity")
        _require_pattern(commit_sha, _SHA1, "license commit SHA")
        locator = identity.locator
        path = f"/repos/{quote(locator.owner, safe='')}/{quote(locator.repository, safe='')}/license"
        status, payload = self._json(GitHubRequest("GET", path, (("ref", commit_sha),)))
        if status == 404:
            return None
        if status != 200 or type(payload) is not dict or type(payload.get("license")) is not dict:
            raise ValueError("invalid GitHub license response")
        if payload.get("url") != self._base(identity) + "/license":
            raise ValueError("GitHub license route mismatch")
        return LicenseMetadata.create(
            payload["license"].get("spdx_id"), payload.get("url"), payload.get("sha"),
        )

    def commit(self, identity: RepositoryIdentity, ref: str) -> CommitMetadata:
        if type(identity) is not RepositoryIdentity:
            raise TypeError("invalid repository identity")
        if type(ref) is not str or not ref or len(ref.encode("utf-8")) > 1024 or any(
            ord(char) < 32 or ord(char) == 127 for char in ref
        ):
            raise ValueError("invalid GitHub commit reference")
        locator = identity.locator
        path = (f"/repos/{quote(locator.owner, safe='')}/{quote(locator.repository, safe='')}"
                f"/commits/{quote(ref, safe='')}")
        status, payload = self._json(GitHubRequest("GET", path))
        if status != 200 or type(payload) is not dict or type(payload.get("commit")) is not dict:
            raise ValueError("invalid GitHub commit response")
        tree = payload["commit"].get("tree")
        if type(tree) is not dict:
            raise ValueError("invalid GitHub commit tree")
        result = CommitMetadata(payload.get("sha"), tree.get("sha"))
        expected_url = self._base(identity) + "/commits/" + result.commit_sha
        if payload.get("url") != expected_url:
            raise ValueError("GitHub commit route mismatch")
        return result

    def tree(self, identity: RepositoryIdentity, tree_sha: str) -> TreeResult:
        _require_pattern(tree_sha, _SHA1, "tree SHA")
        locator = identity.locator
        path = (f"/repos/{quote(locator.owner, safe='')}/{quote(locator.repository, safe='')}"
                f"/git/trees/{tree_sha}")
        status, payload = self._json(GitHubRequest("GET", path, (("recursive", "1"),)))
        if (
            status != 200 or type(payload) is not dict or payload.get("sha") != tree_sha
            or type(payload.get("truncated")) is not bool or type(payload.get("tree")) is not list
        ):
            raise ValueError("invalid GitHub tree response")
        base = self._base(identity)
        entries = []
        for state in payload["tree"]:
            if type(state) is not dict:
                raise ValueError("invalid GitHub tree entry")
            entry = TreeEntry(
                state.get("path"), state.get("mode"), state.get("type"),
                state.get("sha"), state.get("size", 0),
            )
            routes = {
                "blob": base + "/git/blobs/" + entry.sha,
                "tree": base + "/git/trees/" + entry.sha,
                "commit": base + "/git/commits/" + entry.sha,
            }
            if state.get("url") != routes.get(entry.kind):
                raise ValueError("GitHub tree entry route mismatch")
            entries.append(entry)
        return TreeResult(tuple(entries), payload["truncated"])

    def blob(self, identity: RepositoryIdentity, blob_sha: str) -> BlobResult:
        _require_pattern(blob_sha, _SHA1, "blob SHA")
        locator = identity.locator
        path = (f"/repos/{quote(locator.owner, safe='')}/{quote(locator.repository, safe='')}"
                f"/git/blobs/{blob_sha}")
        status, payload = self._json(GitHubRequest("GET", path))
        expected_url = self._base(identity) + "/git/blobs/" + blob_sha
        if (
            status != 200 or type(payload) is not dict or payload.get("sha") != blob_sha
            or payload.get("encoding") != "base64" or payload.get("url") != expected_url
            or type(payload.get("content")) is not str or type(payload.get("size")) is not int
            or payload["size"] < 0
        ):
            raise ValueError("invalid GitHub blob response")
        try:
            encoded = b"".join(payload["content"].encode("ascii").split())
            content = base64.b64decode(encoded, validate=True)
        except (UnicodeError, ValueError):
            raise ValueError("invalid GitHub blob encoding") from None
        if len(content) != payload["size"]:
            raise ValueError("GitHub blob size mismatch")
        return BlobResult(blob_sha, content)


@dataclass(frozen=True, slots=True)
class RepositoryLocator:
    owner: str
    repository: str

    def __post_init__(self) -> None:
        _require_pattern(self.owner, _OWNER, "repository owner")
        _require_pattern(self.repository, _REPOSITORY, "repository name")
        if self.repository in {".", ".."}:
            raise ValueError("invalid repository name")


def parse_repository_input(value: str) -> RepositoryLocator | GitHubSearchQuery:
    if type(value) is not str or not value or value.strip() != value or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise ValueError("invalid GitHub repository input")
    if "://" in value or value.lower().startswith("github.com"):
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https" or parsed.netloc != "github.com"
            or parsed.username is not None or parsed.password is not None
            or parsed.port is not None or parsed.query or parsed.fragment
        ):
            raise ValueError("invalid GitHub repository URL")
        parts = parsed.path.split("/")
        if len(parts) != 3 or not parts[1] or not parts[2] or parts[2].endswith(".git"):
            raise ValueError("invalid GitHub repository URL")
        return RepositoryLocator(parts[1], parts[2])
    return GitHubSearchQuery(value)


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    repository_id: int
    node_id: str
    locator: RepositoryLocator

    def __post_init__(self) -> None:
        if type(self.repository_id) is not int or not 0 < self.repository_id < 2**63:
            raise ValueError("invalid repository ID")
        _require_pattern(self.node_id, _NODE_ID, "repository node ID")
        if type(self.locator) is not RepositoryLocator:
            raise ValueError("invalid repository locator")


@dataclass(frozen=True, slots=True)
class GitHubCollectionPolicy:
    """Canonical release-policy values, shared by inspection and registration."""

    release_licenses: frozenset[str] = frozenset({
        "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MIT", "MS-PL",
    })
    max_repositories: int = 25
    max_tree_entries: int = 20_000
    max_files_per_repository: int = 1_000
    max_repository_bytes: int = 32 * 1024 * 1024
    max_run_bytes: int = 100 * 1024 * 1024
    policy_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.release_licenses) is not frozenset:
            raise ValueError("release licenses must be immutable")
        for license_id in self.release_licenses:
            _require_pattern(license_id, _SPDX, "release license")
        state = {"release_licenses": sorted(self.release_licenses)}
        for name in ("max_repositories", "max_tree_entries", "max_files_per_repository",
                     "max_repository_bytes", "max_run_bytes"):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value < 2**63:
                raise ValueError("invalid collection budget")
            state[name] = value
        object.__setattr__(self, "policy_sha256", hashlib.sha256(_canonical_json(state)).hexdigest())


@dataclass(frozen=True, slots=True)
class RepositoryInspection:
    repository: RepositoryIdentity
    commit_sha: str
    tree_sha: str
    license_spdx: str | None
    license_identity_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        if type(self.repository) is not RepositoryIdentity:
            raise ValueError("invalid inspected repository identity")
        _require_pattern(self.commit_sha, _SHA1, "commit SHA")
        _require_pattern(self.tree_sha, _SHA1, "tree SHA")
        if self.license_spdx is not None:
            _require_pattern(self.license_spdx, _SPDX, "license SPDX")
        _require_pattern(self.license_identity_sha256, _SHA256, "license identity SHA-256")
        _require_pattern(self.policy_sha256, _SHA256, "policy SHA-256")


@dataclass(frozen=True, slots=True)
class RepositoryMetadata:
    identity: RepositoryIdentity
    default_branch: str
    api_url: str
    license_url: str

    def __post_init__(self) -> None:
        if type(self.identity) is not RepositoryIdentity:
            raise ValueError("invalid repository metadata identity")
        if (
            type(self.default_branch) is not str or not self.default_branch
            or len(self.default_branch.encode("utf-8")) > 1024
            or any(ord(char) < 32 or ord(char) == 127 for char in self.default_branch)
        ):
            raise ValueError("invalid repository default branch")
        base = (f"https://{API_HOST}/repos/{self.identity.locator.owner}/"
                f"{self.identity.locator.repository}")
        if self.api_url != base or self.license_url != base + "/license":
            raise ValueError("GitHub repository route mismatch")


def _license_identity(spdx_id: str | None, api_url: str, content_sha: str | None) -> str:
    return hashlib.sha256(_canonical_json({
        "api_url": api_url, "content_sha": content_sha, "spdx_id": spdx_id,
    })).hexdigest()


@dataclass(frozen=True, slots=True)
class LicenseMetadata:
    spdx_id: str
    api_url: str
    content_sha: str | None
    identity_sha256: str

    def __post_init__(self) -> None:
        _require_pattern(self.spdx_id, _SPDX, "license SPDX")
        if type(self.api_url) is not str or not self.api_url.startswith(f"https://{API_HOST}/repos/"):
            raise ValueError("invalid GitHub license route")
        if self.content_sha is not None:
            _require_pattern(self.content_sha, _SHA1, "license content SHA")
        _require_pattern(self.identity_sha256, _SHA256, "license identity SHA-256")
        if not hmac.compare_digest(
            self.identity_sha256, _license_identity(self.spdx_id, self.api_url, self.content_sha),
        ):
            raise ValueError("license identity digest mismatch")

    @classmethod
    def create(cls, spdx_id: str, api_url: str, content_sha: str | None) -> LicenseMetadata:
        return cls(spdx_id, api_url, content_sha, _license_identity(spdx_id, api_url, content_sha))


@dataclass(frozen=True, slots=True)
class CommitMetadata:
    commit_sha: str
    tree_sha: str

    def __post_init__(self) -> None:
        _require_pattern(self.commit_sha, _SHA1, "commit SHA")
        _require_pattern(self.tree_sha, _SHA1, "tree SHA")


@dataclass(frozen=True, slots=True)
class TreeEntry:
    path: str
    mode: str
    kind: str
    sha: str
    size: int

    def __post_init__(self) -> None:
        normalize_artifact_path(self.path)
        if type(self.mode) is not str or re.fullmatch(r"[0-7]{6}", self.mode) is None:
            raise ValueError("invalid GitHub tree mode")
        if self.kind not in {"blob", "tree", "commit"}:
            raise ValueError("invalid GitHub tree kind")
        _require_pattern(self.sha, _SHA1, "tree entry SHA")
        if type(self.size) is not int or not 0 <= self.size <= MAX_JSON_BYTES:
            raise ValueError("invalid GitHub tree entry size")


@dataclass(frozen=True, slots=True)
class TreeResult:
    entries: tuple[TreeEntry, ...]
    truncated: bool

    def __post_init__(self) -> None:
        if type(self.entries) is not tuple or any(type(item) is not TreeEntry for item in self.entries):
            raise ValueError("invalid GitHub tree entries")
        if type(self.truncated) is not bool:
            raise ValueError("invalid GitHub tree truncation flag")


@dataclass(frozen=True, slots=True)
class BlobResult:
    sha: str
    content: bytes

    def __post_init__(self) -> None:
        _require_pattern(self.sha, _SHA1, "blob SHA")
        if type(self.content) is not bytes:
            raise TypeError("invalid GitHub blob content")


@dataclass(frozen=True, slots=True)
class GitHubProgress:
    phase: str
    repository: str
    license_spdx: str | None
    commit_sha: str | None
    tree_sha: str | None
    admitted_files: int
    rejected_files: int
    consumed_bytes: int
    byte_limit: int

    def __post_init__(self) -> None:
        if type(self.phase) is not str or not self.phase:
            raise ValueError("invalid GitHub progress phase")
        if type(self.repository) is not str or not self.repository:
            raise ValueError("invalid GitHub progress repository")
        if self.license_spdx is not None:
            _require_pattern(self.license_spdx, _SPDX, "progress license SPDX")
        for value in (self.admitted_files, self.rejected_files, self.consumed_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("invalid GitHub progress count")
        if type(self.byte_limit) is not int or self.byte_limit <= 0 or self.consumed_bytes > self.byte_limit:
            raise ValueError("invalid GitHub progress byte limit")


@dataclass(frozen=True, slots=True)
class CollectionResult:
    artifacts: tuple[ArtifactInput, ...]
    paused: CollectionPaused | None
    progress: GitHubProgress

    def __post_init__(self) -> None:
        if type(self.artifacts) is not tuple or any(type(item) is not ArtifactInput for item in self.artifacts):
            raise ValueError("invalid GitHub collection artifacts")
        if self.paused is not None and type(self.paused) is not CollectionPaused:
            raise ValueError("invalid GitHub collection pause")
        if type(self.progress) is not GitHubProgress:
            raise ValueError("invalid GitHub collection progress")


def require_release_inspection(inspection: RepositoryInspection) -> RepositoryInspection:
    """Verify the inspection against locally computed release policy bytes.

    The service facade must resolve this metadata before presenting the rights
    confirmation. Neither the ledger nor that facade accepts a policy hash
    override from the registration caller.
    """
    if type(inspection) is not RepositoryInspection:
        raise ValueError("invalid repository inspection")
    if not hmac.compare_digest(inspection.policy_sha256, GitHubCollectionPolicy().policy_sha256):
        raise ValueError("repository inspection policy mismatch")
    return inspection


def _require_grant_scope(grant: object, declared_scope: object) -> None:
    if type(grant) is not str or grant not in {"operator_owned", "written_permission"}:
        raise ValueError("invalid repository rights grant")
    if (
        type(declared_scope) is not str
        or not declared_scope
        or declared_scope.strip() != declared_scope
        or any(ord(character) < 32 or ord(character) == 127 for character in declared_scope)
    ):
        raise ValueError("invalid repository rights scope")
    try:
        size = len(declared_scope.encode("utf-8"))
    except UnicodeError:
        raise ValueError("invalid repository rights scope") from None
    if size > 2048:
        raise ValueError("repository rights scope exceeds its bound")


@dataclass(frozen=True, slots=True)
class RepositoryRightsRecord:
    record_id: str
    repository: RepositoryIdentity
    commit_sha: str
    tree_sha: str
    grant: Literal["operator_owned", "written_permission"]
    declared_scope: str
    confirmed_at_utc: str
    policy_sha256: str
    predecessor_sha256: str | None
    record_sha256: str

    def __post_init__(self) -> None:
        _require_pattern(self.record_id, _RECORD_ID, "rights record ID")
        if type(self.repository) is not RepositoryIdentity:
            raise ValueError("invalid rights repository identity")
        _require_pattern(self.commit_sha, _SHA1, "commit SHA")
        _require_pattern(self.tree_sha, _SHA1, "tree SHA")
        _require_grant_scope(self.grant, self.declared_scope)
        if type(self.confirmed_at_utc) is not str:
            raise ValueError("invalid rights confirmation time")
        try:
            timestamp = datetime.fromisoformat(self.confirmed_at_utc)
        except ValueError:
            raise ValueError("invalid rights confirmation time") from None
        if (
            timestamp.tzinfo is None
            or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp)
            or timestamp.isoformat(timespec="microseconds") != self.confirmed_at_utc
        ):
            raise ValueError("rights confirmation time must be canonical UTC")
        _require_pattern(self.policy_sha256, _SHA256, "policy SHA-256")
        if self.predecessor_sha256 is not None:
            _require_pattern(self.predecessor_sha256, _SHA256, "predecessor SHA-256")
        _require_pattern(self.record_sha256, _SHA256, "rights record SHA-256")
        if not hmac.compare_digest(self.record_sha256, self.digest_state(self.to_state())):
            raise ValueError("rights record digest mismatch")

    def to_state(self) -> dict[str, object]:
        return {"schema": _RIGHTS_SCHEMA, **asdict(self)}

    @staticmethod
    def digest_state(state: dict[str, object]) -> str:
        body = {name: value for name, value in state.items() if name != "record_sha256"}
        return hashlib.sha256(_RIGHTS_DOMAIN + _canonical_json(body)).hexdigest()

    @classmethod
    def from_state(cls, state: object) -> RepositoryRightsRecord:
        if type(state) is not dict or set(state) != {"schema", *cls.__dataclass_fields__}:
            raise ValueError("invalid rights record fields")
        if state["schema"] != _RIGHTS_SCHEMA:
            raise ValueError("invalid rights record schema")
        repository = state["repository"]
        if type(repository) is not dict or set(repository) != {"repository_id", "node_id", "locator"}:
            raise ValueError("invalid rights repository fields")
        locator = repository["locator"]
        if type(locator) is not dict or set(locator) != {"owner", "repository"}:
            raise ValueError("invalid rights locator fields")
        values = {name: value for name, value in state.items() if name != "schema"}
        values["repository"] = RepositoryIdentity(
            repository["repository_id"], repository["node_id"], RepositoryLocator(**locator),
        )
        return cls(**values)


def _request_client(
    client: GitHubRestClient, *, token: str | None, cancel: Event,
) -> GitHubRestClient:
    if isinstance(client, GitHubRestClient):
        return client.bound(token=token, cancel=cancel)
    return client


def _missing_license_identity(identity: RepositoryIdentity, commit_sha: str) -> str:
    base = f"https://{API_HOST}/repos/{identity.locator.owner}/{identity.locator.repository}"
    return hashlib.sha256(_canonical_json({
        "api_url": base + "/license", "commit_sha": commit_sha,
        "content_sha": None, "spdx_id": None,
    })).hexdigest()


def inspect_repository(
    locator: RepositoryLocator, *, client: GitHubRestClient,
    policy: GitHubCollectionPolicy, token: str | None, cancel: Event,
) -> RepositoryInspection:
    """Resolve only immutable repository, license, commit, and tree identity metadata."""
    if type(locator) is not RepositoryLocator or type(policy) is not GitHubCollectionPolicy:
        raise TypeError("invalid GitHub inspection input")
    if cancel.is_set():
        raise CollectionCancelled("GitHub collection cancelled")
    active = _request_client(client, token=token, cancel=cancel)
    try:
        repository = active.repository(locator)
        if repository.identity.locator != locator:
            raise ValueError("GitHub repository identity drift")
        if cancel.is_set():
            raise CollectionCancelled("GitHub collection cancelled")
        commit = active.commit(repository.identity, repository.default_branch)
        if cancel.is_set():
            raise CollectionCancelled("GitHub collection cancelled")
        license_metadata = active.license(repository.identity, commit.commit_sha)
        if license_metadata is not None and license_metadata.api_url != repository.license_url:
            raise ValueError("GitHub license repository identity mismatch")
        return RepositoryInspection(
            repository.identity, commit.commit_sha, commit.tree_sha,
            None if license_metadata is None else license_metadata.spdx_id,
            (_missing_license_identity(repository.identity, commit.commit_sha) if license_metadata is None
             else license_metadata.identity_sha256),
            policy.policy_sha256,
        )
    finally:
        active = None


def _matching_rights(
    inspection: RepositoryInspection, record: RepositoryRightsRecord | None,
) -> RepositoryRightsRecord | None:
    if record is None:
        return None
    if type(record) is not RepositoryRightsRecord:
        raise PermissionError("repository rights record is invalid")
    if (
        record.repository != inspection.repository
        or record.commit_sha != inspection.commit_sha
        or record.tree_sha != inspection.tree_sha
        or record.policy_sha256 != inspection.policy_sha256
    ):
        raise PermissionError("repository rights do not match immutable inspection")
    return record


def _git_blob_identity(content: bytes) -> str:
    header = b"blob " + str(len(content)).encode("ascii") + b"\0"
    return hashlib.sha1(header + content).hexdigest()


def _progress(
    inspection: RepositoryInspection, policy: GitHubCollectionPolicy, *, phase: str,
    admitted: int = 0, rejected: int = 0, consumed: int = 0,
) -> GitHubProgress:
    locator = inspection.repository.locator
    return GitHubProgress(
        phase, f"{locator.owner}/{locator.repository}", inspection.license_spdx,
        inspection.commit_sha, inspection.tree_sha, admitted, rejected, consumed,
        min(policy.max_repository_bytes, policy.max_run_bytes),
    )


def collect_github(
    inspection: RepositoryInspection, *, client: GitHubRestClient,
    policy: GitHubCollectionPolicy,
    rights_lookup: Callable[[RepositoryInspection], RepositoryRightsRecord | None],
    token: str | None, cancel: Event, progress: Callable[[GitHubProgress], None],
) -> CollectionResult:
    """Collect validated immutable blobs after license or exact-rights admission."""
    if type(inspection) is not RepositoryInspection or type(policy) is not GitHubCollectionPolicy:
        raise TypeError("invalid GitHub collection input")
    if not hmac.compare_digest(inspection.policy_sha256, policy.policy_sha256):
        raise ValueError("repository inspection policy mismatch")
    if not callable(rights_lookup) or not callable(progress):
        raise TypeError("invalid GitHub collection callback")
    if cancel.is_set():
        raise CollectionCancelled("GitHub collection cancelled")

    rights = None
    if inspection.license_spdx not in policy.release_licenses:
        rights = _matching_rights(inspection, rights_lookup(inspection))
        if rights is None:
            raise PermissionError("repository is not licensed or locally authorized")

    active = _request_client(client, token=token, cancel=cancel)
    current = _progress(inspection, policy, phase="enumerating")
    progress(current)
    try:
        try:
            tree = active.tree(inspection.repository, inspection.tree_sha)
        except GitHubRateLimitPaused as paused:
            current = _progress(inspection, policy, phase="paused")
            progress(current)
            return CollectionResult((), paused.paused, current)
        if tree.truncated:
            raise ValueError("GitHub tree response is truncated")
        if len(tree.entries) > policy.max_tree_entries:
            raise ValueError("GitHub tree exceeds entry limit")
        if cancel.is_set():
            raise CollectionCancelled("GitHub collection cancelled")

        rejected = 0
        candidates: list[TreeEntry] = []
        paths: set[str] = set()
        for entry in tree.entries:
            if entry.path in paths:
                raise ValueError("GitHub tree repeats a path")
            paths.add(entry.path)
            if entry.kind != "blob" or entry.mode not in {"100644", "100755"}:
                rejected += 1
                continue
            try:
                normalize_artifact_path(entry.path)
                require_allowed_extension(entry.path)
            except ArtifactRejected:
                rejected += 1
                continue
            candidates.append(entry)

        if len(candidates) > policy.max_files_per_repository:
            raise ValueError("GitHub repository exceeds file limit")
        declared_bytes = sum(entry.size for entry in candidates)
        if declared_bytes > policy.max_repository_bytes:
            raise ValueError("GitHub repository exceeds byte limit")
        if declared_bytes > policy.max_run_bytes:
            raise ValueError("GitHub run exceeds byte limit")

        artifacts: list[ArtifactInput] = []
        consumed = 0
        base = (f"https://{API_HOST}/repos/{inspection.repository.locator.owner}/"
                f"{inspection.repository.locator.repository}")
        license_url = None if inspection.license_spdx is None else base + "/license"
        for entry in candidates:
            if cancel.is_set():
                raise CollectionCancelled("GitHub collection cancelled")
            try:
                blob = active.blob(inspection.repository, entry.sha)
            except GitHubRateLimitPaused as paused:
                current = _progress(
                    inspection, policy, phase="paused", rejected=rejected,
                    consumed=consumed,
                )
                progress(current)
                return CollectionResult((), paused.paused, current)
            if blob.sha != entry.sha or _git_blob_identity(blob.content) != entry.sha:
                raise ValueError("GitHub blob identity mismatch")
            if len(blob.content) != entry.size:
                raise ValueError("GitHub blob size differs from immutable tree")
            consumed += len(blob.content)
            if consumed > policy.max_repository_bytes:
                raise ValueError("GitHub repository exceeds byte limit")
            if consumed > policy.max_run_bytes:
                raise ValueError("GitHub run exceeds byte limit")
            raw_sha256 = hashlib.sha256(blob.content).hexdigest()
            provenance = {
                "source_kind": "github_blob",
                "repository_id": inspection.repository.repository_id,
                "repository_node_id": inspection.repository.node_id,
                "owner": inspection.repository.locator.owner,
                "repository": inspection.repository.locator.repository,
                "repository_api_url": base,
                "license_api_url": license_url,
                "license_identity_sha256": inspection.license_identity_sha256,
                "commit_sha": inspection.commit_sha,
                "tree_sha": inspection.tree_sha,
                "blob_sha": entry.sha,
                "raw_byte_sha256": raw_sha256,
                "path": entry.path,
                "license_spdx": inspection.license_spdx,
                "rights_record_sha256": None if rights is None else rights.record_sha256,
                "collector_policy_sha256": inspection.policy_sha256,
            }
            artifact = ArtifactInput("github_blob", entry.path, blob.content, provenance, 3)
            try:
                validated = validate_artifact(artifact)
            except ArtifactRejected:
                rejected += 1
                continue
            if not hmac.compare_digest(validated.raw_sha256, raw_sha256):
                raise ValueError("GitHub raw-byte digest mismatch")
            artifacts.append(artifact)
            current = _progress(
                inspection, policy, phase="collecting", admitted=len(artifacts),
                rejected=rejected, consumed=consumed,
            )
            progress(current)

        if cancel.is_set():
            raise CollectionCancelled("GitHub collection cancelled")
        try:
            immutable = active.commit(inspection.repository, inspection.commit_sha)
        except GitHubRateLimitPaused as paused:
            current = _progress(
                inspection, policy, phase="paused", rejected=rejected, consumed=consumed,
            )
            progress(current)
            return CollectionResult((), paused.paused, current)
        if immutable.commit_sha != inspection.commit_sha or immutable.tree_sha != inspection.tree_sha:
            raise ValueError("GitHub immutable revision changed")
        current = _progress(
            inspection, policy, phase="complete", admitted=len(artifacts),
            rejected=rejected, consumed=consumed,
        )
        progress(current)
        return CollectionResult(tuple(artifacts), None, current)
    finally:
        active = None
