"""Bounded, read-only local adapter observations; never training qualification.

NVIDIA CSV reports driver-visible GPU/partition UUIDs and MiB at query time.
Windows CIM reports display adapters, host RAM and logical CPUs; AdapterRAM is
deliberately ignored. Device count is the count of selected observations, not
a promise of physical training GPU count. NVIDIA UUID and CIM PNP identities are
different namespaces; unresolved overlapping observations are retained with a
count-uncertainty warning. CIM identifiers carry a windows-cim source prefix.
Explicit virtual adapters are labeled; other CIM physical/partition topology
stays unknown, including PCI adapters. Non-NVIDIA VRAM is unknown.
Torch distribution metadata is read without importing Torch or testing a device;
it supplies neither an authenticated runtime identity nor runtime verification.
No remote policy/target binding is inferred from local observations.
An independent native-memory completeness witness survives ancillary host,
display and runtime gaps; it never establishes topology or training support.
"""
from __future__ import annotations

import csv
import ctypes
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Callable, Literal


_MAX_SECONDS = 8.0
_MAX_OUTPUT_BYTES = 256 * 1024
_MAX_DEVICES = 128
_MIB = 1024**2
_MAX_MEMORY = 1024**5
_NVIDIA_FIELDS = "uuid,name,memory.total,memory.free,driver_version"
_CIM_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$adapters = @(); $cpu = $null; $total = $null; $free = $null; $errors = @()
try {
  $adapters = @(Get-CimInstance Win32_VideoController | ForEach-Object {
    @{name=$_.Name; vendor=$_.AdapterCompatibility; pnp_id=$_.PNPDeviceID;
      driver_version=$_.DriverVersion}
  })
} catch { $errors += 'video_unavailable' }
try {
  $hostOs = Get-CimInstance Win32_OperatingSystem
  $total = [long]$hostOs.TotalVisibleMemorySize * 1024
  $free = [long]$hostOs.FreePhysicalMemory * 1024
} catch { $errors += 'ram_unavailable' }
try {
  $cpu = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
} catch { $errors += 'cpu_unavailable' }
@{adapters=$adapters; cpu_count=$cpu; ram_total_bytes=$total;
  ram_free_bytes=$free; errors=$errors} | ConvertTo-Json -Depth 4 -Compress
""".strip()


@dataclass(frozen=True, slots=True)
class ObservedGpu:
    identifier: str
    name: str
    vendor: str
    total_bytes: int | None
    free_bytes: int | None
    compute_capability: str | None
    driver_version: str | None
    device_kind: str
    runtime_visible: bool | None


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    backend: str
    version: str | None
    accelerator_build: str | None
    verified: bool
    reasons: tuple[str, ...]
    platform: str | None = None
    runtime_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class HardwareObservation:
    target_id: str
    observed_at: datetime
    status: str
    devices: tuple[ObservedGpu, ...]
    cpu_count: int | None
    ram_total_bytes: int | None
    ram_free_bytes: int | None
    runtime: RuntimeObservation
    sources: tuple[str, ...]
    warnings: tuple[str, ...]
    target_binding_sha256: str | None = None
    memory_inventory_complete: bool = False
    # Adapter-source success is independent of host/runtime warnings and the
    # native-memory witness. Failed bounded queries may still retain partial rows.
    adapter_enumeration: Literal["unknown", "observed", "empty", "failed"] = "unknown"


@dataclass(frozen=True, slots=True)
class ProcessOutput:
    """Internal bounded combined stdout/stderr result; raw text is never logged."""
    output: bytes
    returncode: int | None
    timed_out: bool = False
    truncated: bool = False


def _platform_label() -> str:
    machine = platform.machine().casefold()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return f"{platform.system().casefold()}-{machine or 'unknown'}"


def _windows_directory() -> Path | None:
    # Query the OS rather than trusting caller-controlled SystemRoot/PATH.
    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetWindowsDirectoryW(buffer, len(buffer))
    return Path(buffer.value) if 0 < length < len(buffer) else None


def _trusted_tool_candidates(name: str) -> tuple[Path, ...]:
    if os.name == "nt":
        windows = _windows_directory()
        if windows is None:
            return ()
        if name == "powershell":
            return (windows / "System32/WindowsPowerShell/v1.0/powershell.exe",)
        if name == "nvidia-smi":
            return (windows / "System32/nvidia-smi.exe",
                    windows.parent / "Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe")
    elif name == "nvidia-smi":
        return (Path("/usr/bin/nvidia-smi"), Path("/usr/local/cuda/bin/nvidia-smi"))
    return ()


def _resolve_tool(name: str) -> Path | None:
    """Resolve only known OS/vendor paths; never search cwd or PATH."""
    for candidate in _trusted_tool_candidates(name):
        try:
            if not candidate.is_absolute() or not candidate.is_file():
                continue
            if os.name != "nt":
                # A user-writable tool or ancestor is not an OS/vendor installation.
                if any(p.stat().st_uid != 0 or p.stat().st_mode & 0o022
                       for p in (candidate, *candidate.parents)):
                    continue
                if not os.access(candidate, os.X_OK):
                    continue
            return candidate
        except OSError:
            continue
    return None


def _run_process(argv: tuple[str, ...], *, deadline: float, max_output_bytes: int) -> ProcessOutput:
    """Drain one owned child's nonblocking pipe, then kill/reap on bounds.

    Each command shares the caller's absolute deadline and remaining output cap.
    Up to 0.5 seconds of that deadline is reserved for owned child/pipe cleanup.
    CPython 3.12 supports nonblocking pipes on Windows and POSIX. No buffered
    reader/thread holds a close lock when a descendant inherits an output handle.
    Helpers have no stdin, shell, profile, or visible Windows console.
    """
    if deadline - time.monotonic() <= 0.5 or max_output_bytes <= 0:
        return ProcessOutput(b"", None, timed_out=True)
    kwargs = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE,
              "stderr": subprocess.STDOUT, "shell": False, "bufsize": 0}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        child = subprocess.Popen(argv, **kwargs)
    except OSError:
        return ProcessOutput(b"", None)
    output = bytearray()
    exceeded = False
    done = False
    timed_out = False
    unavailable = False
    try:
        assert child.stdout is not None
        descriptor = child.stdout.fileno()
        os.set_blocking(descriptor, False)
        while not done or child.poll() is None:
            remaining = deadline - time.monotonic() - 0.5
            if remaining <= 0:
                timed_out = True
                break
            if not done:
                try:
                    chunk = os.read(descriptor, 4096)
                except BlockingIOError:
                    chunk = None
                if chunk == b"":
                    done = True
                elif chunk:
                    available = max_output_bytes - len(output)
                    output.extend(chunk[:available])
                    if len(chunk) > available:
                        exceeded = True
                        break
                    # Drain available data without wasting the shared deadline.
                    continue
            time.sleep(min(remaining, 0.01))
    except OSError:
        unavailable = True
    finally:
        if child.poll() is None:
            child.kill()
        # Kill precedes wait; unlike communicate(), this never buffers uncapped output.
        child.wait(timeout=max(0.01, deadline - time.monotonic()))
        if child.stdout is not None:
            child.stdout.close()
    return ProcessOutput(bytes(output), None if unavailable else child.returncode, timed_out, exceeded)


def _torch_version() -> str | None:
    try:
        version = metadata.version("torch")
        return _text(version)
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return None


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > 512 or any(ord(char) < 32 or char == "\ufffd" for char in value):
        return None
    return None if value.casefold() in {"n/a", "[n/a]", "not supported", "unknown"} else value


def _positive_int(value: object, maximum: int = _MAX_MEMORY, *, zero: bool = False) -> int | None:
    if type(value) is int and (0 if zero else 1) <= value <= maximum:
        return value
    return None


def _mib(value: str, *, zero: bool = False) -> int | None:
    value = value.strip()
    if len(value) > 15 or re.fullmatch(r"[0-9]+", value) is None:
        return None
    return _positive_int(int(value) * _MIB, zero=zero)


def _coalesce(devices: list[ObservedGpu], warnings: list[str], source: str) -> list[ObservedGpu]:
    selected: dict[str, ObservedGpu] = {}
    for device in devices:
        key = device.identifier.casefold()
        previous = selected.get(key)
        if previous is None:
            selected[key] = device
        else:
            warnings.append(f"{source}: duplicate stable adapter identity; count retained once")
            if previous != device:
                warnings.append(f"{source}: conflicting duplicate fields are unknown")
                values = {field: getattr(previous, field) if getattr(previous, field) == getattr(device, field)
                          else None for field in ("total_bytes", "free_bytes", "compute_capability", "driver_version")}
                selected[key] = replace(previous, **values,
                    name=previous.name if previous.name == device.name else "Conflicting adapter names",
                    device_kind=previous.device_kind if previous.device_kind == device.device_kind else "unknown")
    return list(selected.values())


def _parse_nvidia(output: bytes, *, compute_field: bool,
                  warnings: list[str]) -> tuple[list[ObservedGpu], bool, bool]:
    """Return devices, complete memory and complete adapter enumeration separately."""
    devices = []
    memory_complete = True
    enumeration_complete = True
    try:
        text = output.decode("utf-8", errors="strict")
    except UnicodeError:
        memory_complete = False
        enumeration_complete = False
        warnings.append("nvidia-smi: malformed CSV encoding; invalid fields omitted")
        text = output.decode("utf-8", errors="replace")
    try:
        rows = csv.reader(io.StringIO(text), strict=True)
        for row in rows:
            if not row or all(not part.strip() for part in row):
                memory_complete = False
                enumeration_complete = False
                warnings.append("nvidia-smi: empty native row skipped")
                continue
            if len(row) != (6 if compute_field else 5) or len(devices) >= _MAX_DEVICES:
                memory_complete = False
                enumeration_complete = False
                warnings.append("nvidia-smi: malformed row or adapter limit reached")
                continue
            identifier, name = _text(row[0]), _text(row[1])
            if identifier is None or name is None:
                memory_complete = False
                enumeration_complete = False
                warnings.append("nvidia-smi: missing or invalid adapter identity/name")
                continue
            total, free = _mib(row[2]), _mib(row[3], zero=True)
            if total is None or (free is not None and free > total):
                free = None
            driver = _text(row[4])
            compute = _text(row[5]) if compute_field else None
            if compute is not None and re.fullmatch(r"[0-9]{1,2}\.[0-9]{1,2}", compute) is None:
                compute = None
            kind = "partition" if identifier.startswith("MIG-") else (
                "physical" if identifier.startswith("GPU-") else "unknown")
            if None in (total, free, driver, compute) or kind == "unknown":
                warnings.append("nvidia-smi: adapter fields or device kind are unknown")
            if total is None or free is None or kind == "unknown":
                memory_complete = False
            devices.append(ObservedGpu(identifier, name, "NVIDIA", total, free, compute, driver, kind, None))
    except (UnicodeError, csv.Error):
        memory_complete = False
        enumeration_complete = False
        warnings.append("nvidia-smi: malformed CSV encoding or structure")
    coalesced = _coalesce(devices, warnings, "nvidia-smi")
    return (coalesced, memory_complete and len(coalesced) == len(devices),
            enumeration_complete and len(coalesced) == len(devices))


def _json_pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate native JSON field")
        result[name] = value
    return result


def _parse_cim(output: bytes, warnings: list[str]):
    try:
        value = json.loads(output.decode("utf-8-sig"), object_pairs_hook=_json_pairs)
        if type(value) is not dict or type(value.get("adapters")) is not list:
            raise ValueError("invalid CIM structure")
    except (UnicodeError, ValueError, RecursionError):
        warnings.append("windows-cim: malformed adapter/host observation")
        return [], None, None, None, False, False
    devices = []
    errors = value.get("errors", [])
    enumeration_complete = (type(errors) is list and all(type(error) is str and
        error in {"ram_unavailable", "cpu_unavailable"} for error in errors))
    for raw in value["adapters"][:_MAX_DEVICES]:
        if type(raw) is not dict:
            enumeration_complete = False
            warnings.append("windows-cim: malformed adapter")
            continue
        identifier, name = _text(raw.get("pnp_id")), _text(raw.get("name"))
        vendor = _text(raw.get("vendor")) or "Unknown"
        if identifier is None or name is None:
            enumeration_complete = False
            warnings.append("windows-cim: missing stable adapter identity/name")
            continue
        upper_id = identifier.upper()
        virtual = any(label in name.casefold() for label in (
            "virtual display", "indirect display", "remote display", "virtualbox",
            "vmware", "virtio", "hyper-v video", "qxl",
        ))
        # PCI is a bus identity inside this OS; guests can expose PCI devices too.
        # CIM alone cannot establish physical/pass-through/partition topology.
        kind = "virtual" if virtual else "unknown"
        if "VEN_10DE" in upper_id:
            vendor = "NVIDIA"
        elif "VEN_1002" in upper_id:
            vendor = "AMD"
        elif "nvidia" in vendor.casefold():
            vendor = "NVIDIA"
        elif "advanced micro devices" in vendor.casefold() or re.search(r"\b(?:amd|ati)\b", vendor, re.I):
            vendor = "AMD"
        devices.append(ObservedGpu(f"windows-cim:{identifier}", name, vendor, None, None, None,
                                   _text(raw.get("driver_version")), kind, False if virtual else None))
    if devices:
        warnings.append("windows-cim: display adapters do not establish physical topology or compute support; VRAM is unknown")
    if len(value["adapters"]) > _MAX_DEVICES:
        enumeration_complete = False
        warnings.append("windows-cim: adapter limit reached")
    cpu = _positive_int(value.get("cpu_count"), 65536)
    total = _positive_int(value.get("ram_total_bytes"))
    free = _positive_int(value.get("ram_free_bytes"), zero=True)
    if total is None or (free is not None and free > total):
        free = None
    if None in (cpu, total, free) or value.get("errors"):
        warnings.append("windows-cim: host fields are unavailable or invalid")
    coalesced = _coalesce(devices, warnings, "windows-cim")
    enumeration_complete = enumeration_complete and len(coalesced) == len(devices)
    if not enumeration_complete:
        warnings.append("windows-cim: adapter enumeration failed or incomplete")
    return coalesced, cpu, total, free, True, enumeration_complete


def _linux_ram() -> tuple[int | None, int | None]:
    try:
        with Path("/proc/meminfo").open("rb") as stream:
            output = stream.read(65537)
        if len(output) > 65536:
            return None, None
        values = dict(re.findall(rb"^(MemTotal|MemAvailable):\s+([0-9]+) kB$", output, re.M))
        total = _positive_int(int(values[b"MemTotal"]) * 1024)
        free = _positive_int(int(values[b"MemAvailable"]) * 1024, zero=True)
        return total, free if total is not None and free is not None and free <= total else None
    except (OSError, ValueError, KeyError):
        return None, None


def _detect_local_hardware(*, timeout_seconds: float, process_runner: Callable,
                           tool_resolver: Callable) -> HardwareObservation:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be a finite positive number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite positive number")
    deadline = time.monotonic() + min(timeout_seconds, _MAX_SECONDS)
    remaining_output = _MAX_OUTPUT_BYTES
    observed_at = datetime.now(timezone.utc)
    platform_label = _platform_label()
    warnings: list[str] = []
    sources: list[str] = []
    unavailable_tools: set[str] = set()
    cpu = _positive_int(os.cpu_count(), 65536)
    if cpu is not None:
        sources.append("os.cpu_count")
    total = free = None

    def query(name: str, args: tuple[str, ...]) -> ProcessOutput | None:
        nonlocal remaining_output
        if remaining_output <= 0 or time.monotonic() >= deadline:
            warnings.append(f"{name}: shared deadline/output budget exhausted")
            return None
        tool = tool_resolver(name)
        if tool is None:
            unavailable_tools.add(name)
            warnings.append(f"{name}: trusted native tool unavailable")
            return None
        try:
            result = process_runner((str(tool), *args), deadline=deadline, max_output_bytes=remaining_output)
        except (OSError, subprocess.SubprocessError):
            warnings.append(f"{name}: native process unavailable")
            return None
        cap = remaining_output
        bounded = result.output[:cap]
        remaining_output -= min(len(result.output), cap)
        result = replace(result, output=bounded, truncated=result.truncated or len(result.output) > cap)
        if result.timed_out:
            warnings.append(f"{name}: native process timeout")
        if result.truncated:
            warnings.append(f"{name}: native output cap reached")
        if result.truncated or result.timed_out:
            result = replace(result, output=bounded[:bounded.rfind(b"\n") + 1])
        if result.returncode != 0 and not (result.timed_out or result.truncated):
            warnings.append(f"{name}: native query failed")
        return result

    compute_field = True
    native = query("nvidia-smi", (f"--query-gpu={_NVIDIA_FIELDS},compute_cap", "--format=csv,noheader,nounits"))
    if (native is not None and native.returncode not in (0, None) and not native.timed_out
            and not native.truncated and b"compute_cap" in native.output.lower()
            and any(text in native.output.lower() for text in (b"not a valid", b"unsupported", b"invalid field"))):
        compute_field = False
        native = query("nvidia-smi", (f"--query-gpu={_NVIDIA_FIELDS}", "--format=csv,noheader,nounits"))
    devices = []
    memory_inventory_complete = False
    native_enumerated = False
    if native is not None and (native.returncode == 0 or native.timed_out or native.truncated):
        devices, parsed_memory_complete, parsed_enumeration = _parse_nvidia(
            native.output, compute_field=compute_field, warnings=warnings)
        memory_inventory_complete = (native.returncode == 0 and not native.timed_out
                                     and not native.truncated and parsed_memory_complete)
        native_enumerated = (native.returncode == 0 and not native.timed_out
                             and not native.truncated and parsed_enumeration)
        if native.returncode == 0 or devices:
            sources.append("nvidia-smi")
    if devices:
        warnings.append("nvidia-smi: topology and partition availability were not probed")

    enumeration_succeeded = native_enumerated
    if platform_label.startswith("windows-"):
        cim_enumerated = False
        cim = query("powershell", ("-NoLogo", "-NoProfile", "-NonInteractive", "-Command", _CIM_SCRIPT))
        if cim is not None and cim.returncode == 0 and not cim.truncated and not cim.timed_out:
            adapters, host_cpu, total, free, parsed, cim_enumerated = _parse_cim(cim.output, warnings)
            if parsed:
                sources.append("windows-cim")
            cpu = host_cpu or cpu
            if devices and any(device.vendor == "NVIDIA" and device.device_kind != "virtual" for device in adapters):
                warnings.append("NVIDIA native/CIM identity overlap is uncertain; unmatched source observations retained; physical count unverified")
            devices.extend(adapters)
        # NVIDIA's tool is optional on AMD/other Windows systems. An attempted
        # failed query is not equivalent to an absent optional tool. CIM must
        # establish successful video enumeration separately from host facts.
        enumeration_succeeded = cim_enumerated and (native_enumerated or "nvidia-smi" in unavailable_tools)
    elif platform_label.startswith("linux-"):
        total, free = _linux_ram()
        if total is not None:
            sources.append("linux-meminfo")
        warnings.append("Non-NVIDIA/virtual adapter inventory is unavailable on this platform")
    else:
        warnings.append("Native host/adapter inventory is unavailable on this platform")

    version = _torch_version()
    build = version.partition("+")[2] or None if version else None
    reasons = ["Package metadata only; runtime execution and accelerator availability were not tested"]
    if version is not None:
        sources.append("python-package-metadata:torch")
    else:
        warnings.append("Torch package metadata unavailable")
        reasons.append("Torch distribution version is unavailable")
    if build == "cpu":
        reasons.append("CPU Torch build cannot execute CUDA workloads")
        devices = [replace(device, runtime_visible=False) for device in devices]
    runtime = RuntimeObservation("pytorch", version, build, False, tuple(reasons), platform_label)
    if cpu is None or total is None or free is None:
        warnings.append("Host CPU/RAM observation is incomplete")
    # Topology uncertainty alone does not erase an otherwise complete native query.
    incomplete = [warning for warning in warnings if "topology and partition" not in warning]
    has_hardware = cpu is not None or total is not None or bool(devices) or "windows-cim" in sources or "nvidia-smi" in sources
    status = "unavailable" if not has_hardware else ("partial" if incomplete else "complete")
    adapter_enumeration = "failed"
    if enumeration_succeeded:
        # Only Windows CIM currently covers non-NVIDIA adapters. An empty
        # NVIDIA-only platform query cannot establish an all-vendor zero count.
        adapter_enumeration = "observed" if devices else (
            "empty" if platform_label.startswith("windows-") else "unknown")
    return HardwareObservation("local", observed_at, status, tuple(devices), cpu, total, free,
                               runtime, tuple(sources), tuple(dict.fromkeys(warnings)),
                               memory_inventory_complete=memory_inventory_complete,
                               adapter_enumeration=adapter_enumeration)


def detect_local_hardware(*, timeout_seconds: float = 8.0) -> HardwareObservation:
    """Observe this host with a shared native deadline (max 8s) and 256 KiB cap.

    This read-only API never loads models, imports Torch, touches a corpus, runs
    training/readiness or opens a remote transport. Missing tools stay unknown.
    Runtime verification/hash and target binding remain absent; use authenticated
    evidence and a matching recipe for qualification separately.
    """
    return _detect_local_hardware(timeout_seconds=timeout_seconds,
                                  process_runner=_run_process, tool_resolver=_resolve_tool)


__all__ = ["ObservedGpu", "RuntimeObservation", "HardwareObservation", "detect_local_hardware"]
