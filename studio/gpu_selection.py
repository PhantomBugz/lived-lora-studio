"""Read-only NVIDIA GPU discovery and explicit device selection.

An observed device is not proof that a CUDA training runtime is installed.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Callable, Literal

from .hardware_inventory import ProcessOutput, _resolve_tool, _run_process, _windows_directory


_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
_COMPUTE_CAPABILITY = re.compile(r"[0-9]{1,2}\.[0-9]{1,2}\Z")
_SSH_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_QUERY = "--query-gpu=uuid,name,memory.total,memory.free,compute_cap,driver_version"
_FORMAT = "--format=csv,noheader,nounits"
_SELECTION_FIELDS = frozenset({
    "schema", "target", "gpu_uuid", "observed_at_utc",
    "binding_sha256", "training_qualified",
})


class GpuProbeError(RuntimeError):
    """A bounded device probe could not establish a complete inventory."""


class GpuSelectionError(ValueError):
    """An exact, recent GPU choice cannot be made from this inventory."""


@dataclass(frozen=True, slots=True)
class GpuTarget:
    kind: Literal["local", "ssh"]
    ssh_alias: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("local", "ssh"):
            raise ValueError("GPU target must be local or ssh")
        if self.kind == "local" and self.ssh_alias is not None:
            raise ValueError("local target cannot have an SSH alias")
        if self.kind == "ssh" and not self.ssh_alias:
            raise ValueError("remote target requires an SSH alias")
        if self.kind == "ssh" and (
            type(self.ssh_alias) is not str or _SSH_ALIAS.fullmatch(self.ssh_alias) is None
        ):
            raise ValueError("SSH alias contains unsupported characters")


@dataclass(frozen=True, slots=True)
class GpuDevice:
    uuid: str
    name: str
    total_mib: int
    free_mib: int
    compute_capability: str
    driver_version: str
    device_kind: str = "physical"

    def __post_init__(self) -> None:
        if type(self.uuid) is not str or _GPU_UUID.fullmatch(self.uuid) is None:
            raise ValueError("GPU UUID is invalid")
        if (type(self.name) is not str or not self.name or len(self.name) > 128
                or not self.name.isprintable()):
            raise ValueError("GPU name is invalid")
        if (type(self.total_mib) is not int or type(self.free_mib) is not int
                or not 1 <= self.total_mib <= 1_048_576
                or not 0 <= self.free_mib <= self.total_mib):
            raise ValueError("GPU memory fields are invalid")
        if (type(self.compute_capability) is not str
                or _COMPUTE_CAPABILITY.fullmatch(self.compute_capability) is None):
            raise ValueError("GPU compute capability is invalid")
        if (type(self.driver_version) is not str or not self.driver_version
                or len(self.driver_version) > 64 or not self.driver_version.isprintable()):
            raise ValueError("GPU driver version is invalid")
        if (type(self.device_kind) is not str or not self.device_kind
                or len(self.device_kind) > 32 or not self.device_kind.isprintable()):
            raise ValueError("GPU device kind is invalid")


@dataclass(frozen=True, slots=True)
class GpuInventory:
    target: GpuTarget
    observed_at: datetime
    devices: tuple[GpuDevice, ...]
    training_qualified: bool = False

    def __post_init__(self) -> None:
        if type(self.target) is not GpuTarget:
            raise ValueError("GPU inventory target is invalid")
        if (type(self.observed_at) is not datetime or self.observed_at.tzinfo is None
                or self.observed_at.utcoffset() is None):
            raise ValueError("GPU inventory observation time must be timezone-aware")
        if (type(self.devices) is not tuple or len(self.devices) > 128
                or not all(type(device) is GpuDevice for device in self.devices)):
            raise ValueError("GPU inventory devices are invalid")
        if self.training_qualified is not False:
            raise ValueError("GPU observation cannot claim training qualification")


@dataclass(frozen=True, slots=True)
class GpuSelection:
    target: GpuTarget
    gpu_uuid: str
    observed_at: datetime
    binding_sha256: str
    training_qualified: bool = False


def _parse_devices(payload: bytes) -> tuple[GpuDevice, ...]:
    try:
        reader = csv.reader(io.StringIO(payload.decode("utf-8")), strict=True, skipinitialspace=True)
        devices = []
        for row in reader:
            if len(row) != 6:
                raise GpuProbeError("NVIDIA inventory has an invalid CSV row")
            uuid, name, total, free, compute, driver = (part.strip() for part in row)
            if (_GPU_UUID.fullmatch(uuid) is None or not name
                    or len(name) > 128 or not name.isprintable()):
                raise GpuProbeError("NVIDIA inventory has an invalid device identity")
            if not total.isdecimal() or not free.isdecimal():
                raise GpuProbeError("NVIDIA inventory has invalid memory fields")
            total_mib, free_mib = int(total), int(free)
            if not 1 <= total_mib <= 1_048_576 or not 0 <= free_mib <= total_mib:
                raise GpuProbeError("NVIDIA inventory has invalid memory fields")
            if (_COMPUTE_CAPABILITY.fullmatch(compute) is None or not driver
                    or len(driver) > 64 or not driver.isprintable()):
                raise GpuProbeError("NVIDIA inventory lacks compute capability or driver")
            devices.append(GpuDevice(uuid, name, total_mib, free_mib, compute, driver))
            if len(devices) > 128:
                raise GpuProbeError("NVIDIA inventory exceeds the device limit")
        if len({device.uuid.casefold() for device in devices}) != len(devices):
            raise GpuProbeError("NVIDIA inventory contains duplicate GPU UUIDs")
        return tuple(devices)
    except (UnicodeError, csv.Error) as error:
        raise GpuProbeError("NVIDIA inventory is malformed") from error


def _resolve_ssh() -> Path | None:
    if os.name == "nt":
        windows = _windows_directory()
        candidates = () if windows is None else (windows / "System32/OpenSSH/ssh.exe",)
    else:
        candidates = (Path("/usr/bin/ssh"), Path("/bin/ssh"))
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            if os.name != "nt" and any(
                path.stat().st_uid != 0 or path.stat().st_mode & 0o022
                for path in (candidate, *candidate.parents)
            ):
                continue
            if os.access(candidate, os.X_OK):
                return candidate
        except OSError:
            continue
    return None


def discover_gpus(
    target: GpuTarget,
    *,
    process_runner: Callable[..., ProcessOutput] | None = None,
    tool_resolver: Callable[[str], Path | None] | None = None,
    clock: Callable[[], datetime] | None = None,
    timeout_seconds: float = 8.0,
) -> GpuInventory:
    """Query the selected machine without changing its runtime or data."""
    if type(target) is not GpuTarget:
        raise TypeError("target must be GpuTarget")
    if (type(timeout_seconds) not in (int, float)
            or not 0 < timeout_seconds <= 8.0):
        raise ValueError("GPU probe timeout must be greater than zero and at most 8 seconds")
    resolver = (
        lambda name: _resolve_ssh() if name == "ssh" else _resolve_tool(name)
    ) if tool_resolver is None else tool_resolver
    executable = resolver("nvidia-smi" if target.kind == "local" else "ssh")
    if executable is None:
        raise GpuProbeError("trusted GPU probe tool is unavailable")
    if target.kind == "local":
        argv = (str(executable), _QUERY, _FORMAT)
    else:
        argv = (
            str(executable), "-T",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes",
            "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no",
            "-o", "NumberOfPasswordPrompts=0",
            "-o", "ForwardAgent=no",
            "-o", "ClearAllForwardings=yes",
            "-o", "PermitLocalCommand=no",
            "-o", "ControlMaster=no",
            "-o", "ConnectTimeout=5",
            "--", target.ssh_alias, "/usr/bin/nvidia-smi", _QUERY, _FORMAT,
        )
    runner = _run_process if process_runner is None else process_runner
    result = runner(argv,
                    deadline=time.monotonic() + timeout_seconds,
                    max_output_bytes=64 * 1024)
    if result.returncode != 0 or result.timed_out or result.truncated:
        raise GpuProbeError("NVIDIA inventory query did not complete")
    observed_at = datetime.now(timezone.utc) if clock is None else clock()
    return GpuInventory(target, observed_at, _parse_devices(result.output))


def select_gpu(
    inventory: GpuInventory,
    gpu_uuid: str,
    *,
    now: datetime | None = None,
    max_age_seconds: float = 60.0,
) -> GpuSelection:
    """Select one recent, observed CUDA-generation device by stable UUID."""
    if type(inventory) is not GpuInventory:
        raise TypeError("inventory must be GpuInventory")
    if (not all(type(item) is GpuDevice for item in inventory.devices)
            or len({item.uuid.casefold() for item in inventory.devices}) != len(inventory.devices)):
        raise GpuSelectionError("GPU inventory contains ambiguous device identities")
    if type(gpu_uuid) is not str or _GPU_UUID.fullmatch(gpu_uuid) is None:
        raise GpuSelectionError("selected GPU UUID is invalid")
    current = datetime.now(timezone.utc) if now is None else now
    if (type(current) is not datetime or current.tzinfo is None
            or current.utcoffset() is None):
        raise GpuSelectionError("selection time must be timezone-aware")
    if not 0 <= (current - inventory.observed_at).total_seconds() <= max_age_seconds:
        raise GpuSelectionError("GPU inventory is stale")
    matches = [item for item in inventory.devices if item.uuid.casefold() == gpu_uuid.casefold()]
    if len(matches) != 1:
        raise GpuSelectionError("selected GPU UUID is not unique and present")
    device = matches[0]
    major, minor = (int(part) for part in device.compute_capability.split("."))
    if (major, minor) < (7, 0):
        raise GpuSelectionError("selected GPU compute capability is unsupported")
    return GpuSelection(inventory.target, device.uuid, inventory.observed_at,
                        _binding_digest(inventory.target, device.uuid))


def _binding_digest(target: GpuTarget, gpu_uuid: str) -> str:
    binding = json.dumps({"kind": target.kind, "ssh_alias": target.ssh_alias,
                          "gpu_uuid": gpu_uuid},
                         sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(b"lived-gpu-selection-v1\0" + binding).hexdigest()


def selection_document(chosen: GpuSelection) -> dict[str, object]:
    """Return a stable, credential-free selection record for explicit export."""
    if type(chosen) is not GpuSelection:
        raise TypeError("chosen must be GpuSelection")
    if (type(chosen.target) is not GpuTarget
            or type(chosen.gpu_uuid) is not str
            or _GPU_UUID.fullmatch(chosen.gpu_uuid) is None
            or chosen.binding_sha256 != _binding_digest(chosen.target, chosen.gpu_uuid)
            or chosen.training_qualified is not False
            or type(chosen.observed_at) is not datetime
            or chosen.observed_at.tzinfo is None
            or chosen.observed_at.utcoffset() is None):
        raise GpuSelectionError("selection target, GPU, or observation binding is invalid")
    return {
        "schema": "LIVED_GPU_SELECTION_V1",
        "target": {"kind": chosen.target.kind, "ssh_alias": chosen.target.ssh_alias},
        "gpu_uuid": chosen.gpu_uuid,
        "observed_at_utc": chosen.observed_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "binding_sha256": chosen.binding_sha256,
        "training_qualified": False,
    }


def save_selection(chosen: GpuSelection, path: Path | str) -> Path:
    """Write an explicitly chosen GPU to a new JSON file without overwrite."""
    destination = Path(path)
    encoded = (json.dumps(selection_document(chosen), sort_keys=True,
                          separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return destination


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for name, value in pairs:
        if name in document:
            raise GpuSelectionError("saved selection contains duplicate JSON keys")
        document[name] = value
    return document


def load_selection(path: Path | str) -> GpuSelection:
    """Read a bounded saved choice and verify its target/device binding."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(Path(path), flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise GpuSelectionError("saved selection must be a regular file")
        payload = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    if len(payload) > 4096:
        raise GpuSelectionError("saved selection exceeds the size limit")
    try:
        document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
        if type(document) is not dict or set(document) != _SELECTION_FIELDS:
            raise GpuSelectionError("saved selection schema differs")
        target_doc = document["target"]
        if type(target_doc) is not dict or set(target_doc) != {"kind", "ssh_alias"}:
            raise GpuSelectionError("saved target schema differs")
        if (document["schema"] != "LIVED_GPU_SELECTION_V1"
                or document["training_qualified"] is not False):
            raise GpuSelectionError("saved selection claim is unsupported")
        target = GpuTarget(target_doc["kind"], target_doc["ssh_alias"])
        gpu_uuid = document["gpu_uuid"]
        if type(gpu_uuid) is not str or _GPU_UUID.fullmatch(gpu_uuid) is None:
            raise GpuSelectionError("saved GPU UUID is invalid")
        timestamp = document["observed_at_utc"]
        if type(timestamp) is not str or not timestamp.endswith("Z"):
            raise GpuSelectionError("saved observation time is invalid")
        observed_at = datetime.fromisoformat(timestamp)
        if observed_at.tzinfo is None or observed_at.utcoffset().total_seconds() != 0:
            raise GpuSelectionError("saved observation time is not UTC")
        if document["binding_sha256"] != _binding_digest(target, gpu_uuid):
            raise GpuSelectionError("saved target and GPU binding differs")
        return GpuSelection(target, gpu_uuid, observed_at, document["binding_sha256"])
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, GpuSelectionError):
            raise
        raise GpuSelectionError("saved selection is invalid") from error


__all__ = ["GpuDevice", "GpuInventory", "GpuProbeError", "GpuSelection",
           "GpuSelectionError", "GpuTarget", "discover_gpus", "load_selection", "save_selection",
           "select_gpu", "selection_document"]
