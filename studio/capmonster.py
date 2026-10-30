from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import ssl
import subprocess
import sys
import time
from typing import Any, Callable
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import (
    HTTPSHandler,
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)
from uuid import uuid4


_RECORD_ID = re.compile(r"[a-f0-9]{32}")
_MAX_API_KEY_BYTES = 4_096
_MAX_RESPONSE_BYTES = 65_536
_TIMEOUT_SECONDS = 10.0
_BALANCE_URL = "https://api.capmonster.cloud/getBalance"
_CREATE_TASK_URL = "https://api.capmonster.cloud/createTask"
_TASK_RESULT_URL = "https://api.capmonster.cloud/getTaskResult"
_TURNSTILE_POLL_SECONDS = 2.0
_TURNSTILE_MAX_POLLS = 30
_TURNSTILE_DEADLINE_SECONDS = 60.0
_MAX_TURNSTILE_VALUE_BYTES = 8_192
_MAX_TURNSTILE_TOKEN_BYTES = 16_384
_CHILD_CLEANUP_TIMEOUT_SECONDS = 1.0
_PROVIDER_URLS = frozenset((_BALANCE_URL, _CREATE_TASK_URL, _TASK_RESULT_URL))


class CapMonsterValidationError(ValueError):
    """A local CapMonster credential is malformed."""


class CapMonsterProviderError(RuntimeError):
    """A balance check failed without disclosing provider details."""


def _bounded_provider_text(value: object, name: str, *, maximum: int) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    if len(encoded) > maximum or any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} is outside its bounded text schema")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class TurnstileChallenge:
    """Validated, ephemeral render data for a token-only provider request."""

    website_url: str
    website_key: str
    action: str | None = None
    c_data: str | None = None
    page_data: str | None = None
    user_agent: str | None = None

    def __post_init__(self) -> None:
        website_url = _bounded_provider_text(
            self.website_url, "website_url", maximum=_MAX_TURNSTILE_VALUE_BYTES
        )
        parts = urlsplit(website_url)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError("website_url must be an HTTPS document URL")
        object.__setattr__(self, "website_url", website_url)
        object.__setattr__(
            self,
            "website_key",
            _bounded_provider_text(
                self.website_key,
                "website_key",
                maximum=_MAX_TURNSTILE_VALUE_BYTES,
            ),
        )
        for name in ("action", "c_data", "page_data", "user_agent"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _bounded_provider_text(
                        value, name, maximum=_MAX_TURNSTILE_VALUE_BYTES
                    ),
                )

    def __repr__(self) -> str:
        return "TurnstileChallenge()"


def _required_record_id(value: object) -> str:
    if type(value) is not str or _RECORD_ID.fullmatch(value) is None:
        raise CapMonsterValidationError(
            "record_id must be a lowercase opaque identifier"
        )
    return value


def _required_api_key(value: object) -> str:
    if type(value) is not str or not value:
        raise CapMonsterValidationError("api_key must be a nonempty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CapMonsterValidationError("api_key must be valid UTF-8") from exc
    if len(encoded) > _MAX_API_KEY_BYTES:
        raise CapMonsterValidationError("api_key exceeds the UTF-8 size limit")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise CapMonsterValidationError("api_key must not contain control characters")
    return value


class CapMonsterCredential:
    """Immutable local provider credential with explicit secret access only."""

    __slots__ = ("record_id", "__api_key", "__sealed")

    def __init__(self, record_id: str, api_key: str) -> None:
        object.__setattr__(self, "record_id", _required_record_id(record_id))
        object.__setattr__(self, "_CapMonsterCredential__api_key", _required_api_key(api_key))
        object.__setattr__(self, "_CapMonsterCredential__sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_CapMonsterCredential__sealed", False):
            raise AttributeError("CapMonsterCredential is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("CapMonsterCredential is immutable")

    def __repr__(self) -> str:
        return f"CapMonsterCredential(record_id={self.record_id!r})"

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("CapMonsterCredential secrets cannot be pickled")

    def __getstate__(self) -> object:
        raise TypeError("CapMonsterCredential requires explicit secret_state() serialization")

    def __copy__(self) -> object:
        raise TypeError("CapMonsterCredential secrets cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> object:
        raise TypeError("CapMonsterCredential secrets cannot be copied")

    @classmethod
    def create(cls, *, api_key: str) -> "CapMonsterCredential":
        return cls(uuid4().hex, api_key)

    def secret_key(self) -> str:
        """Return the key only to the local encrypted vault or balance client."""

        return self.__api_key

    def secret_state(self) -> dict[str, str]:
        """Return secret-bearing state only for the encrypted local vault."""

        return {"record_id": self.record_id, "api_key": self.secret_key()}

    def masked_display(self) -> dict[str, str]:
        return {"record_id": self.record_id, "api_key": "********"}

    @classmethod
    def from_state(cls, state: object) -> "CapMonsterCredential":
        if type(state) is not dict or set(state) != {"record_id", "api_key"}:
            raise CapMonsterValidationError("CapMonster credential state has an invalid schema")
        return cls(state["record_id"], state["api_key"])

    def __eq__(self, other: object) -> bool:
        return self is other or (
            type(other) is CapMonsterCredential
            and self.record_id == other.record_id
            and self.__api_key == other.__api_key
        )

    def __hash__(self) -> int:
        return hash((self.record_id, self.__api_key))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> Request | None:
        return None


def _strict_json(payload: bytes) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate provider response field")
            result[key] = value
        return result

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        parse_float=Decimal,
    )
    if type(value) is not dict:
        raise ValueError("provider response must be an object")
    return value


class CapMonsterClient:
    """Small local-only client that performs the documented balance request."""

    def __init__(
        self,
        *,
        opener: object | None = None,
        process_factory: Callable[..., object] = subprocess.Popen,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._opener = opener
        self._process_factory = process_factory
        self._unreaped_children: list[object] = []
        self._closed = False
        self._monotonic = monotonic
        self._sleeper = sleeper

    @staticmethod
    def _response_status(response: object) -> int | None:
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else None
        return status if type(status) is int else None

    def _post_json(self, url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
        if self._closed:
            raise ValueError("provider client is closed")
        if self._opener is None:
            return self._post_json_in_child(url, payload, timeout)
        response = None
        try:
            request = Request(
                url,
                data=json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            response = self._opener.open(request, timeout=timeout)
            if self._response_status(response) != 200:
                raise ValueError("provider response status is invalid")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
            if type(raw) is not bytes or len(raw) > _MAX_RESPONSE_BYTES:
                raise ValueError("provider response length is invalid")
            return _strict_json(raw)
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    def _post_json_in_child(
        self,
        url: str,
        payload: dict[str, object],
        timeout: float,
    ) -> dict[str, object]:
        if url not in _PROVIDER_URLS:
            raise ValueError("provider URL is invalid")
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("provider timeout is invalid")
        request = json.dumps(
            payload, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        if len(request) > _MAX_RESPONSE_BYTES:
            raise ValueError("provider request length is invalid")
        command = (
            sys.executable,
            "-I",
            "-S",
            str(Path(__file__).with_name("capmonster_transport.py")),
            url,
        )
        self._reap_owned_children()
        child = self._process_factory(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            shell=False,
            close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            stdout, _stderr = child.communicate(
                input=request,
                timeout=min(_TIMEOUT_SECONDS, float(timeout)),
            )
        except subprocess.TimeoutExpired:
            self._terminate_child(child)
            raise ValueError("provider deadline exceeded") from None
        except Exception:
            self._terminate_child(child)
            raise ValueError("provider transport failed") from None
        if getattr(child, "returncode", None) != 0:
            raise ValueError("provider transport failed")
        if type(stdout) is not bytes or len(stdout) > _MAX_RESPONSE_BYTES:
            raise ValueError("provider response length is invalid")
        return _strict_json(stdout)

    def _terminate_child(self, child: object) -> None:
        terminate = getattr(child, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:
                pass
        try:
            child.communicate(timeout=_CHILD_CLEANUP_TIMEOUT_SECONDS)
            if getattr(child, "returncode", None) is not None:
                return
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        kill = getattr(child, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass
        try:
            child.communicate(timeout=_CHILD_CLEANUP_TIMEOUT_SECONDS)
        except Exception:
            pass
        if getattr(child, "returncode", None) is None:
            if child not in self._unreaped_children:
                self._unreaped_children.append(child)
            raise ValueError("provider child exit could not be confirmed")

    def _reap_owned_children(self) -> None:
        for child in tuple(self._unreaped_children):
            try:
                self._terminate_child(child)
            except Exception:
                raise ValueError("provider child exit could not be confirmed") from None
            self._unreaped_children.remove(child)

    def close(self) -> None:
        self._closed = True
        self._reap_owned_children()
        if self._unreaped_children:
            raise CapMonsterProviderError("CapMonster child cleanup is incomplete")

    @staticmethod
    def _provider_error(values: dict[str, object]) -> None:
        error_id = values.get("errorId")
        if type(error_id) is not int:
            raise ValueError("provider error id is invalid")
        if error_id != 0:
            if (
                set(values) != {"errorId", "errorCode", "errorDescription"}
                or type(values["errorCode"]) is not str
                or type(values["errorDescription"]) is not str
            ):
                raise ValueError("provider error schema is invalid")
            raise ValueError("provider rejected request")

    def _turnstile_remaining(self, started: float) -> float:
        now = self._monotonic()
        if (
            type(now) not in (int, float)
            or not math.isfinite(float(now))
            or not math.isfinite(float(started))
        ):
            raise ValueError("monotonic clock is invalid")
        remaining = _TURNSTILE_DEADLINE_SECONDS - (float(now) - float(started))
        if not math.isfinite(remaining) or remaining <= 0:
            raise ValueError("provider deadline exceeded")
        return remaining

    def solve_turnstile(self, api_key: str, challenge: TurnstileChallenge) -> str:
        """Create one proxyless Turnstile task and return one bounded token."""

        key = None
        try:
            key = _required_api_key(api_key)
            if type(challenge) is not TurnstileChallenge:
                raise ValueError("Turnstile challenge is invalid")
            task: dict[str, object] = {
                "type": "TurnstileTask",
                "websiteURL": challenge.website_url,
                "websiteKey": challenge.website_key,
            }
            optional = (
                ("action", "pageAction"),
                ("c_data", "data"),
                ("page_data", "pageData"),
                ("user_agent", "userAgent"),
            )
            if any(
                getattr(challenge, source) is not None
                for source, _target in optional
            ):
                task["cloudflareTaskType"] = "token"
            for source, target in optional:
                value = getattr(challenge, source)
                if value is not None:
                    task[target] = value
            started = self._monotonic()
            if type(started) not in (int, float) or not math.isfinite(float(started)):
                raise ValueError("monotonic clock is invalid")
            create = self._post_json(
                _CREATE_TASK_URL,
                {"clientKey": key, "task": task},
                min(_TIMEOUT_SECONDS, _TURNSTILE_DEADLINE_SECONDS),
            )
            self._turnstile_remaining(float(started))
            self._provider_error(create)
            if set(create) != {"errorId", "taskId"}:
                raise ValueError("create response schema is invalid")
            task_id = create["taskId"]
            if type(task_id) is not int or task_id < 1:
                raise ValueError("provider task id is invalid")
            for _poll in range(_TURNSTILE_MAX_POLLS):
                remaining = self._turnstile_remaining(float(started))
                if remaining < _TURNSTILE_POLL_SECONDS:
                    raise ValueError("provider deadline exceeded")
                self._sleeper(_TURNSTILE_POLL_SECONDS)
                remaining = self._turnstile_remaining(float(started))
                result = self._post_json(
                    _TASK_RESULT_URL,
                    {"clientKey": key, "taskId": task_id},
                    min(_TIMEOUT_SECONDS, float(remaining)),
                )
                self._turnstile_remaining(float(started))
                self._provider_error(result)
                status = result.get("status")
                if status == "processing":
                    if set(result) == {"errorId", "status"}:
                        continue
                    if (
                        set(result) == {"errorCode", "errorDescription", "errorId", "status"}
                        and result["errorCode"] is None
                        and result["errorDescription"] is None
                    ):
                        continue
                if status == "ready" and set(result) == {"errorId", "status", "solution"}:
                    solution = result["solution"]
                    if (
                        type(solution) is not dict
                        or set(solution) not in ({"token"}, {"userAgent", "token"})
                    ):
                        raise ValueError("ready response solution schema is invalid")
                    if "userAgent" in solution:
                        user_agent = _bounded_provider_text(
                            solution["userAgent"],
                            "userAgent",
                            maximum=_MAX_TURNSTILE_VALUE_BYTES,
                        )
                        if (
                            challenge.user_agent is not None
                            and user_agent != challenge.user_agent
                        ):
                            raise ValueError("ready response user agent changed")
                    return _bounded_provider_text(
                        solution["token"],
                        "token",
                        maximum=_MAX_TURNSTILE_TOKEN_BYTES,
                    )
                raise ValueError("provider task state is invalid")
            raise ValueError("provider result poll limit exceeded")
        except Exception:
            raise CapMonsterProviderError("CapMonster provider unavailable") from None
        finally:
            key = None

    def balance(self, api_key: str) -> Decimal:
        key = _required_api_key(api_key)
        try:
            values = self._post_json(
                _BALANCE_URL, {"clientKey": key}, _TIMEOUT_SECONDS
            )
            error_id = values.get("errorId")
            if type(error_id) is not int:
                raise ValueError("provider error id is invalid")
            if error_id != 0:
                if set(values) != {"errorId", "errorCode", "errorDescription"}:
                    raise ValueError("provider error schema is invalid")
                if type(values["errorCode"]) is not str or type(values["errorDescription"]) is not str:
                    raise ValueError("provider error schema is invalid")
                raise ValueError("provider rejected balance request")
            if set(values) != {"errorId", "balance"}:
                raise ValueError("provider success schema is invalid")
            raw_balance = values["balance"]
            if type(raw_balance) not in (int, Decimal):
                raise ValueError("provider balance is invalid")
            balance = Decimal(raw_balance)
            if not balance.is_finite() or balance < 0:
                raise ValueError("provider balance is invalid")
        except Exception:
            raise CapMonsterProviderError("CapMonster balance unavailable") from None
        return balance


__all__ = [
    "CapMonsterClient",
    "CapMonsterCredential",
    "CapMonsterProviderError",
    "CapMonsterValidationError",
    "TurnstileChallenge",
]
