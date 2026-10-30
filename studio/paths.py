from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
import stat
from typing import Final


_REPARSE_ATTRIBUTE: Final[int] = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400
)
_EMBEDDING_MODEL_LAYOUT: Final[tuple[str, str, str]] = (
    "resources",
    "embedding-model",
    "all-MiniLM-L6-v2",
)


class UnsafePathError(ValueError):
    """A path was outside the workspace or crossed an unsafe filesystem object."""


class PathRaceError(UnsafePathError):
    """A checked filesystem object changed while it was being used."""


class PathReadLimitError(ValueError):
    """A guarded read exceeded its caller-supplied byte limit."""


def _relative_parts(
    value: str | os.PathLike[str], *, allow_dot: bool = False
) -> tuple[str, ...]:
    text = os.fspath(value)
    if not isinstance(text, str) or not text or "\x00" in text:
        raise UnsafePathError("path must be a nonempty relative path")
    windows = PureWindowsPath(text)
    posix = Path(text.replace("\\", "/"))
    if windows.drive or windows.root or posix.is_absolute():
        raise UnsafePathError("absolute paths are not accepted")
    parts = tuple(part for part in text.replace("\\", "/").split("/") if part)
    if (not parts and not allow_dot) or any(part in {".", ".."} for part in parts):
        raise UnsafePathError("path traversal is not accepted")
    return parts


def _absolute_lexical(path: Path) -> Path:
    """Return an absolute path without resolving links or reparse points."""
    return Path(os.path.abspath(os.fspath(path)))


def _path_chain(path: Path) -> tuple[Path, ...]:
    absolute = _absolute_lexical(path)
    return (*reversed(absolute.parents), absolute)


def _entry_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        *_entry_identity(metadata),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _is_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & _REPARSE_ATTRIBUTE
    )


def _checked_lstat(path: Path) -> os.stat_result:
    metadata = os.lstat(path)
    if _is_reparse(metadata):
        raise UnsafePathError(
            f"symbolic links, junctions, and reparse points are not accepted: {path}"
        )
    return metadata


def _snapshot_chain(
    path: Path, *, require_target: bool
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    snapshot: list[tuple[str, tuple[int, int, int]]] = []
    missing = False
    for component in _path_chain(path):
        try:
            metadata = _checked_lstat(component)
        except FileNotFoundError:
            missing = True
            continue
        if missing:
            raise PathRaceError("filesystem ancestry changed during path validation")
        snapshot.append((os.path.normcase(os.fspath(component)), _entry_identity(metadata)))
    target_key = os.path.normcase(os.fspath(_absolute_lexical(path)))
    if require_target and (not snapshot or snapshot[-1][0] != target_key):
        raise UnsafePathError(f"path does not exist: {path}")
    return tuple(snapshot)


def _snapshot_directory_chain(
    path: Path,
) -> tuple[tuple[str, tuple[int, int, int]], ...]:
    """Snapshot an existing directory and every lexical ancestor twice."""
    absolute = _absolute_lexical(path)
    before = _snapshot_chain(absolute, require_target=True)
    for component in _path_chain(absolute):
        metadata = _checked_lstat(component)
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnsafePathError("directory ancestry contains a non-directory")
    after = _snapshot_chain(absolute, require_target=True)
    if after != before:
        raise PathRaceError("directory ancestry changed during validation")
    return after


def _require_private_embedding_release_ancestry(path: Path) -> None:
    """Reject writable or foreign-owned fixed-layout ancestors on POSIX."""
    if os.name == "nt":
        return
    absolute = _absolute_lexical(path)
    release_root = absolute.parents[2]
    expected_device = int(_checked_lstat(absolute).st_dev)
    for component in (
        release_root,
        release_root / _EMBEDDING_MODEL_LAYOUT[0],
        release_root / _EMBEDDING_MODEL_LAYOUT[0] / _EMBEDDING_MODEL_LAYOUT[1],
    ):
        try:
            metadata = _checked_lstat(component)
        except (OSError, UnsafePathError) as error:
            raise UnsafePathError(
                "installed embedding release ancestor is not private"
            ) from error
        if (
            int(metadata.st_dev) != expected_device
            or not stat.S_ISDIR(metadata.st_mode)
            or (hasattr(os, "getuid") and metadata.st_uid != os.getuid())
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise UnsafePathError(
                "installed embedding release ancestor is writable or untrusted"
            )


def _require_contained(root: Path, candidate: Path) -> None:
    root_absolute = _absolute_lexical(root)
    candidate_absolute = _absolute_lexical(candidate)
    try:
        common = Path(os.path.commonpath((root_absolute, candidate_absolute)))
    except ValueError:
        raise UnsafePathError("path escapes the selected workspace") from None
    if os.path.normcase(os.fspath(common)) != os.path.normcase(os.fspath(root_absolute)):
        raise UnsafePathError("path escapes the selected workspace")


def _windows_api():
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = (ctypes.c_void_p,)
    kernel.LocalFree.restype = ctypes.c_void_p
    advapi.OpenProcessToken.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    advapi.OpenProcessToken.restype = wintypes.BOOL
    advapi.GetTokenInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi.GetTokenInformation.restype = wintypes.BOOL
    advapi.ConvertSidToStringSidW.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    )
    advapi.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi.ConvertStringSidToSidW.argtypes = (
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.ConvertStringSidToSidW.restype = wintypes.BOOL
    return ctypes, wintypes, advapi, kernel


def _windows_sid_text(pointer: int) -> str:
    ctypes, wintypes, advapi, kernel = _windows_api()
    rendered = wintypes.LPWSTR()
    if not advapi.ConvertSidToStringSidW(
        ctypes.c_void_p(pointer), ctypes.byref(rendered)
    ):
        raise UnsafePathError("Windows SID could not be rendered")
    try:
        return str(rendered.value)
    finally:
        kernel.LocalFree(rendered)


def _windows_current_user_sid() -> str:
    ctypes, wintypes, advapi, kernel = _windows_api()
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(
        kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)
    ):
        raise UnsafePathError("current Windows identity could not be opened")
    try:
        needed = wintypes.DWORD()
        advapi.GetTokenInformation(
            token, 1, None, 0, ctypes.byref(needed)
        )
        if needed.value == 0:
            raise UnsafePathError("current Windows identity could not be sized")
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi.GetTokenInformation(
            token, 1, buffer, needed.value, ctypes.byref(needed)
        ):
            raise UnsafePathError("current Windows identity could not be read")
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        if not sid:
            raise UnsafePathError("current Windows identity has no SID")
        return _windows_sid_text(int(sid))
    finally:
        kernel.CloseHandle(token)


def _create_private_windows_directory(path: Path) -> tuple[int, int, int]:
    """Create with the exact owner and protected DACL in one Windows operation."""
    ctypes, wintypes, advapi, kernel = _windows_api()

    class SecurityAttributes(ctypes.Structure):
        _fields_ = (
            ("length", wintypes.DWORD),
            ("descriptor", ctypes.c_void_p),
            ("inherit_handle", wintypes.BOOL),
        )

    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
    )
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    kernel.CreateDirectoryW.argtypes = (
        wintypes.LPCWSTR,
        ctypes.POINTER(SecurityAttributes),
    )
    kernel.CreateDirectoryW.restype = wintypes.BOOL

    current_sid = _windows_current_user_sid()
    descriptor_text = (
        f"O:{current_sid}D:P"
        f"(A;OICI;0x001F01FF;;;{current_sid})"
        "(A;OICI;0x001F01FF;;;SY)"
    )
    descriptor = ctypes.c_void_p()
    converted = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        descriptor_text, 1, ctypes.byref(descriptor), None
    )
    if not converted or not descriptor:
        if descriptor:
            kernel.LocalFree(descriptor)
        raise UnsafePathError(
            "private Windows directory descriptor could not be built"
        )
    try:
        attributes = SecurityAttributes(
            ctypes.sizeof(SecurityAttributes), descriptor, False
        )
        if not kernel.CreateDirectoryW(os.fspath(path), ctypes.byref(attributes)):
            if ctypes.get_last_error() in (80, 183):
                raise FileExistsError("directory already exists")
            raise UnsafePathError("private Windows directory could not be created")
    finally:
        kernel.LocalFree(descriptor)
    metadata = _checked_lstat(path)
    if not stat.S_ISDIR(metadata.st_mode):
        raise PathRaceError("created directory changed")
    return _entry_identity(metadata)


def _create_private_windows_file(path: Path, *, read_write: bool = False) -> int:
    """Exclusively create a file with its private owner and DACL in place."""
    import msvcrt

    ctypes, wintypes, advapi, kernel = _windows_api()

    class SecurityAttributes(ctypes.Structure):
        _fields_ = (
            ("length", wintypes.DWORD),
            ("descriptor", ctypes.c_void_p),
            ("inherit_handle", wintypes.BOOL),
        )

    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
    )
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(SecurityAttributes),
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE

    current_sid = _windows_current_user_sid()
    descriptor_text = (
        f"O:{current_sid}D:P"
        f"(A;;0x001F01FF;;;{current_sid})"
        "(A;;0x001F01FF;;;SY)"
    )
    security_descriptor = ctypes.c_void_p()
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        descriptor_text, 1, ctypes.byref(security_descriptor), None
    ) or not security_descriptor:
        if security_descriptor:
            kernel.LocalFree(security_descriptor)
        raise UnsafePathError("private Windows file descriptor could not be built")
    try:
        attributes = SecurityAttributes(
            ctypes.sizeof(SecurityAttributes), security_descriptor, False
        )
        handle = kernel.CreateFileW(
            os.fspath(path),
            0xC0000000 if read_write else 0x40000000,
            0x00000001 | 0x00000002 | 0x00000004,  # preserve normal sharing
            ctypes.byref(attributes),
            1,  # CREATE_NEW
            0x00000080 | 0x00200000,  # normal file, open reparse point itself
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            failure = ctypes.WinError(error)
            failure.filename = os.fspath(path)
            raise failure
        try:
            return msvcrt.open_osfhandle(
                handle,
                (os.O_RDWR if read_write else os.O_WRONLY)
                | int(getattr(os, "O_BINARY", 0)),
            )
        except BaseException:
            kernel.CloseHandle(handle)
            raise
    finally:
        kernel.LocalFree(security_descriptor)


def _set_private_windows_acl(
    path: Path, *, directory: bool, read_only: bool = False
) -> None:
    ctypes, wintypes, advapi, kernel = _windows_api()

    class Trustee(ctypes.Structure):
        _fields_ = (
            ("multiple", ctypes.c_void_p),
            ("operation", ctypes.c_int),
            ("form", ctypes.c_int),
            ("kind", ctypes.c_int),
            ("name", ctypes.c_void_p),
        )

    class ExplicitAccess(ctypes.Structure):
        _fields_ = (
            ("permissions", wintypes.DWORD),
            ("access_mode", ctypes.c_int),
            ("inheritance", wintypes.DWORD),
            ("trustee", Trustee),
        )

    advapi.SetEntriesInAclW.argtypes = (
        wintypes.ULONG,
        ctypes.POINTER(ExplicitAccess),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.SetEntriesInAclW.restype = wintypes.DWORD
    advapi.SetNamedSecurityInfoW.argtypes = (
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    advapi.SetNamedSecurityInfoW.restype = wintypes.DWORD

    allocated: list[ctypes.c_void_p] = []
    acl = ctypes.c_void_p()
    try:
        entries = (ExplicitAccess * 2)()
        for index, sid_text in enumerate(
            (_windows_current_user_sid(), "S-1-5-18")
        ):
            sid = ctypes.c_void_p()
            if not advapi.ConvertStringSidToSidW(
                sid_text, ctypes.byref(sid)
            ) or not sid:
                raise UnsafePathError("private Windows SID could not be created")
            allocated.append(sid)
            entries[index] = ExplicitAccess(
                0x00120089 if read_only and index == 0 else 0x001F01FF,
                2,
                0x3 if directory else 0,
                Trustee(None, 0, 0, 1, sid),
            )
        result = advapi.SetEntriesInAclW(
            2, entries, None, ctypes.byref(acl)
        )
        if result != 0:
            raise UnsafePathError("private Windows ACL could not be built")
        result = advapi.SetNamedSecurityInfoW(
            os.fspath(path),
            1,
            0x00000004 | 0x80000000,
            None,
            None,
            acl,
            None,
        )
        if result != 0:
            raise UnsafePathError("private Windows ACL could not be applied")
    finally:
        if acl:
            kernel.LocalFree(acl)
        for sid in allocated:
            kernel.LocalFree(sid)


def _require_private_windows_acl(
    path: Path, *, read_only: bool = False
) -> None:
    ctypes, wintypes, advapi, kernel = _windows_api()

    class AclSizeInformation(ctypes.Structure):
        _fields_ = (
            ("ace_count", wintypes.DWORD),
            ("bytes_in_use", wintypes.DWORD),
            ("bytes_free", wintypes.DWORD),
        )

    advapi.GetNamedSecurityInfoW.argtypes = (
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi.GetSecurityDescriptorControl.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL
    advapi.GetAclInformation.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_int,
    )
    advapi.GetAclInformation.restype = wintypes.BOOL
    advapi.GetAce.argtypes = (
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    )
    advapi.GetAce.restype = wintypes.BOOL

    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    result = advapi.GetNamedSecurityInfoW(
        os.fspath(path),
        1,
        0x00000001 | 0x00000004,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if result != 0 or not owner or not dacl or not descriptor:
        raise UnsafePathError("private Windows ACL could not be read")
    try:
        current_sid = _windows_current_user_sid()
        if _windows_sid_text(int(owner.value)) != current_sid:
            raise UnsafePathError(
                "private Windows path owner is not the current user"
            )
        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if (
            not advapi.GetSecurityDescriptorControl(
                descriptor, ctypes.byref(control), ctypes.byref(revision)
            )
            or not (control.value & 0x1000)
        ):
            raise UnsafePathError(
                "private Windows ACL inheritance is not protected"
            )
        info = AclSizeInformation()
        if not advapi.GetAclInformation(
            dacl, ctypes.byref(info), ctypes.sizeof(info), 2
        ):
            raise UnsafePathError("private Windows ACL could not be enumerated")
        expected_masks = {
            current_sid: 0x00120089 if read_only else 0x001F01FF,
            "S-1-5-18": 0x001F01FF,
        }
        seen: set[str] = set()
        if info.ace_count != 2:
            raise UnsafePathError(
                "private Windows ACL contains extra or missing identities"
            )
        for index in range(info.ace_count):
            ace = ctypes.c_void_p()
            if not advapi.GetAce(dacl, index, ctypes.byref(ace)):
                raise UnsafePathError(
                    "private Windows ACL entry could not be read"
                )
            address = int(ace.value)
            ace_type = ctypes.c_ubyte.from_address(address).value
            ace_size = ctypes.c_ushort.from_address(address + 2).value
            mask = ctypes.c_uint32.from_address(address + 4).value
            if ace_type != 0 or ace_size < 12:
                raise UnsafePathError(
                    "private Windows ACL contains an unexpected entry"
                )
            sid = _windows_sid_text(address + 8)
            if (
                sid not in expected_masks
                or mask != expected_masks[sid]
                or sid in seen
            ):
                raise UnsafePathError(
                    "private Windows ACL grants an unexpected identity"
                )
            seen.add(sid)
        if seen != set(expected_masks):
            raise UnsafePathError(
                "private Windows ACL lacks its exact owner and SYSTEM grants"
            )
    finally:
        kernel.LocalFree(descriptor)


def _require_private_directory(
    path: Path,
    *,
    expected_device: int | None = None,
    windows_read_only: bool = False,
) -> None:
    metadata = _checked_lstat(path)
    if (
        expected_device is not None
        and (
            type(expected_device) is not int
            or int(metadata.st_dev) != expected_device
        )
    ):
        raise UnsafePathError("private directory is on an unexpected device")
    if not stat.S_ISDIR(metadata.st_mode):
        raise UnsafePathError("private path must be a directory")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise UnsafePathError("private directory owner is unsafe")
    if os.name == "nt":
        _require_private_windows_acl(path, read_only=windows_read_only)
    elif stat.S_IMODE(metadata.st_mode) != 0o700:
        raise UnsafePathError("private directory mode must be 0700")


def _require_private_file(
    path: Path,
    *,
    expected_device: int | None = None,
    windows_read_only: bool = False,
) -> os.stat_result:
    metadata = _checked_lstat(path)
    if (
        expected_device is not None
        and (
            type(expected_device) is not int
            or int(metadata.st_dev) != expected_device
        )
    ):
        raise UnsafePathError("private file is on an unexpected device")
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise UnsafePathError("private file must be a regular one-link file")
    if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise UnsafePathError("private file owner is unsafe")
    if os.name == "nt":
        _require_private_windows_acl(path, read_only=windows_read_only)
    elif stat.S_IMODE(metadata.st_mode) != 0o600:
        raise UnsafePathError("private file mode must be 0600")
    return metadata


def _make_checked_directory(path: Path) -> None:
    absolute = _absolute_lexical(path)
    chain = _path_chain(absolute)
    created: dict[Path, tuple[int, int, int]] = {}
    for index, component in enumerate(chain):
        try:
            metadata = _checked_lstat(component)
        except FileNotFoundError:
            if index == 0:
                raise UnsafePathError("filesystem anchor does not exist") from None
            parent = chain[index - 1]
            before = _snapshot_chain(parent, require_target=True)
            try:
                if os.name == "nt":
                    created[component] = _create_private_windows_directory(component)
                else:
                    os.mkdir(component, 0o700)
            except FileExistsError:
                if os.name == "nt":
                    raise PathRaceError("directory appeared during creation") from None
                pass
            metadata = _checked_lstat(component)
            if _snapshot_chain(parent, require_target=True) != before:
                raise PathRaceError("directory ancestry changed while it was created")
            if component in created and _entry_identity(metadata) != created[component]:
                raise PathRaceError("created directory changed")
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnsafePathError(f"workspace directory is not a directory: {component}")
    before = _snapshot_chain(absolute, require_target=True)
    if os.name != "nt":
        os.chmod(absolute, 0o700, follow_symlinks=False)
    # New Windows directories are private at creation. Existing ones are
    # validated without rewriting ACLs on their possibly populated children.
    after = _snapshot_chain(absolute, require_target=True)
    if after != before:
        raise PathRaceError("directory changed while securing it")
    for component, identity in created.items():
        if _entry_identity(_checked_lstat(component)) != identity:
            raise PathRaceError("created directory changed")
    _require_private_directory(absolute)
    if _snapshot_chain(absolute, require_target=True) != after:
        raise PathRaceError("directory changed while validating it")


def _read_descriptor(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class StudioPaths:
    """Workspace-relative paths with checked, non-reparse filesystem access."""

    def __init__(
        self,
        root: Path = Path("."),
        *,
        installed_embedding_model_root: Path | None = None,
    ):
        root_text = os.fspath(root)
        if root_text == ".":
            self.root = Path(".")
        else:
            self.root = Path(*_relative_parts(root_text))
        self.data = self.root / "studio_data"
        self.database = self.data / "studio.db"
        self.salt = self.data / "vault.salt"
        self.lock = self.data / "studio.lock"
        self.log = self.data / "studio.log"
        self.browser_profile = self.data / "browser_profile"
        self.crawl_receipts = self.data / "crawl_receipts"
        self.source_rights = self.data / "source_rights"
        self.source_reset_state = self.data / "source_reset_state"
        self.source_reset_receipts = self.data / "source_reset_receipts"
        self.staging = self.data / "staging"
        self.training_runs = self.data / "training_runs"
        self.dataset_snapshots = self.data / "dataset_snapshots"
        self.artifact_registry = self.data / "artifact_registry"
        self.rag_staging = self.data / "rag_staging"
        self.rag_indexes = self.data / "rag_indexes"
        self.rag_cache = self.data / "rag_cache"
        self._installed_embedding_model_root: Path | None = None
        self._installed_embedding_root_snapshot: (
            tuple[tuple[str, tuple[int, int, int]], ...] | None
        ) = None
        if installed_embedding_model_root is not None:
            supplied = os.fspath(installed_embedding_model_root)
            if not isinstance(supplied, str) or not supplied or "\x00" in supplied:
                raise UnsafePathError(
                    "installed embedding model root must be an absolute path"
                )
            supplied_path = Path(supplied)
            if not supplied_path.is_absolute():
                raise UnsafePathError(
                    "installed embedding model root must be an absolute path"
                )
            installed = _absolute_lexical(supplied_path)
            expected_suffix = tuple(
                os.path.normcase(part) for part in _EMBEDDING_MODEL_LAYOUT
            )
            actual_suffix = tuple(
                os.path.normcase(part)
                for part in installed.parts[-len(_EMBEDDING_MODEL_LAYOUT) :]
            )
            if actual_suffix != expected_suffix:
                raise UnsafePathError(
                    "installed embedding model root does not have the fixed release layout"
                )
            snapshot = _snapshot_directory_chain(installed)
            _require_private_embedding_release_ancestry(installed)
            _require_private_directory(installed)
            if _snapshot_directory_chain(installed) != snapshot:
                raise PathRaceError(
                    "installed embedding model root changed during validation"
                )
            self._installed_embedding_model_root = installed
            self._installed_embedding_root_snapshot = snapshot
        self.embedding_model = (
            self._installed_embedding_model_root
            if self._installed_embedding_model_root is not None
            else self.root.joinpath(*_EMBEDDING_MODEL_LAYOUT)
        )
        self.exports = self.root / "exports"
        self.captures = self.root / "captures"
        self.analysis = self.root / "analysis"

    def ensure(self) -> "StudioPaths":
        root_absolute = _absolute_lexical(self.root)
        for directory in (
            self.root,
            self.data,
            self.browser_profile,
            self.crawl_receipts,
            self.source_rights,
            self.source_reset_state,
            self.source_reset_receipts,
            self.staging,
            self.training_runs,
            self.dataset_snapshots,
            self.artifact_registry,
            self.rag_staging,
            self.rag_indexes,
            self.rag_cache,
            self.exports,
            self.captures,
            self.analysis,
        ):
            absolute = _absolute_lexical(directory)
            _require_contained(root_absolute, absolute)
            _make_checked_directory(absolute)
        return self

    def _candidate(
        self, value: str | os.PathLike[str], *, must_exist: bool
    ) -> tuple[Path, Path]:
        parts = _relative_parts(value)
        candidate = self.root.joinpath(*parts)
        root_absolute = _absolute_lexical(self.root)
        candidate_absolute = _absolute_lexical(candidate)
        _require_contained(root_absolute, candidate_absolute)
        _snapshot_chain(root_absolute, require_target=True)
        if must_exist:
            _snapshot_chain(candidate_absolute, require_target=True)
        else:
            _snapshot_chain(candidate_absolute, require_target=False)
            try:
                _checked_lstat(candidate_absolute)
            except FileNotFoundError:
                pass
        root_resolved = root_absolute.resolve(strict=True)
        resolved = candidate_absolute.resolve(strict=must_exist)
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise UnsafePathError("path escapes the selected workspace")
        return candidate, candidate_absolute

    def anchor(self, value: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
        """Validate a relative path for naming; use secure_read_bytes for reading."""
        candidate, _absolute = self._candidate(value, must_exist=must_exist)
        return candidate

    def secure_directory(
        self, value: str | os.PathLike[str], *, create: bool = False
    ) -> Path:
        """Return a reparse-free absolute directory contained by this workspace."""
        parts = _relative_parts(value)
        root_absolute = _absolute_lexical(self.root)
        candidate = root_absolute.joinpath(*parts)
        _require_contained(root_absolute, candidate)
        _snapshot_chain(root_absolute, require_target=True)
        if create:
            _make_checked_directory(candidate)
        metadata = _checked_lstat(candidate)
        if not stat.S_ISDIR(metadata.st_mode):
            raise UnsafePathError("secured path is not a directory")
        before = _snapshot_chain(candidate, require_target=True)
        resolved = candidate.resolve(strict=True)
        root_resolved = root_absolute.resolve(strict=True)
        if resolved != root_resolved and root_resolved not in resolved.parents:
            raise UnsafePathError("path escapes the selected workspace")
        if _snapshot_chain(candidate, require_target=True) != before:
            raise PathRaceError("directory changed during path validation")
        return resolved

    def secure_read_bytes(
        self,
        value: str | os.PathLike[str],
        *,
        maximum: int,
    ) -> bytes:
        """Read one stable regular file through a checked descriptor."""
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError("maximum must be a nonnegative integer")
        _candidate, absolute = self._candidate(value, must_exist=True)
        before_chain = _snapshot_chain(absolute, require_target=True)
        before_path = _checked_lstat(absolute)
        if not stat.S_ISREG(before_path.st_mode):
            raise UnsafePathError("secured input must be a regular file")
        if before_path.st_nlink != 1:
            raise UnsafePathError("secured input must not be a hard link")
        if before_path.st_size > maximum:
            raise PathReadLimitError("secured input exceeds the read maximum")

        flags = os.O_RDONLY
        for option in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= int(getattr(os, option, 0))
        try:
            descriptor = os.open(absolute, flags)
        except OSError as error:
            raise PathRaceError(f"secured input could not be opened: {error}") from error
        try:
            opened_before = os.fstat(descriptor)
            if not stat.S_ISREG(opened_before.st_mode):
                raise UnsafePathError("secured input must be a regular file")
            if _file_identity(opened_before) != _file_identity(before_path):
                raise PathRaceError("secured input changed before it was opened")
            if _snapshot_chain(absolute, require_target=True) != before_chain:
                raise PathRaceError("secured input ancestry changed before reading")

            data = _read_descriptor(descriptor, maximum)

            opened_after = os.fstat(descriptor)
            after_path = _checked_lstat(absolute)
            after_chain = _snapshot_chain(absolute, require_target=True)
            if (
                _file_identity(opened_after) != _file_identity(opened_before)
                or _file_identity(after_path) != _file_identity(opened_before)
                or after_chain != before_chain
            ):
                raise PathRaceError("secured input changed while it was read")
            if len(data) > maximum:
                raise PathReadLimitError("secured input exceeds the read maximum")
            return data
        finally:
            os.close(descriptor)

    def _revalidate_installed_embedding_root(self) -> Path:
        root = self._installed_embedding_model_root
        expected = self._installed_embedding_root_snapshot
        if root is None or expected is None:
            raise UnsafePathError("installed embedding model root is unavailable")
        current = _snapshot_directory_chain(root)
        _require_private_embedding_release_ancestry(root)
        _require_private_directory(root)
        if _snapshot_directory_chain(root) != current or current != expected:
            raise PathRaceError(
                "installed embedding model root identity changed after startup"
            )
        return root

    def secure_embedding_directory(self) -> Path:
        """Return only the fixed, checked all-MiniLM-L6-v2 directory."""
        if self._installed_embedding_model_root is not None:
            return self._revalidate_installed_embedding_root()
        return self.secure_directory("/".join(_EMBEDDING_MODEL_LAYOUT), create=False)

    def secure_embedding_read_bytes(
        self,
        value: str | os.PathLike[str],
        *,
        maximum: int,
    ) -> bytes:
        """Read one relative file only from the fixed embedding-model tree."""
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError("maximum must be a nonnegative integer")
        parts = _relative_parts(value)
        if self._installed_embedding_model_root is None:
            relative = "/".join((*_EMBEDDING_MODEL_LAYOUT, *parts))
            return self.secure_read_bytes(relative, maximum=maximum)

        root = self._revalidate_installed_embedding_root()
        root_metadata = _checked_lstat(root)
        absolute = root.joinpath(*parts)
        _require_contained(root, absolute)
        before_chain = _snapshot_chain(absolute, require_target=True)
        before_path = _checked_lstat(absolute)
        if not stat.S_ISREG(before_path.st_mode):
            raise UnsafePathError("secured embedding input must be a regular file")
        if before_path.st_nlink != 1:
            raise UnsafePathError("secured embedding input must not be a hard link")
        _require_private_file(absolute, expected_device=int(root_metadata.st_dev))
        if before_path.st_size > maximum:
            raise PathReadLimitError(
                "secured embedding input exceeds the read maximum"
            )

        flags = os.O_RDONLY
        for option in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= int(getattr(os, option, 0))
        try:
            descriptor = os.open(absolute, flags)
        except OSError as error:
            raise PathRaceError(
                f"secured embedding input could not be opened: {error}"
            ) from error
        try:
            opened_before = os.fstat(descriptor)
            if not stat.S_ISREG(opened_before.st_mode):
                raise UnsafePathError(
                    "secured embedding input must be a regular file"
                )
            if _file_identity(opened_before) != _file_identity(before_path):
                raise PathRaceError(
                    "secured embedding input changed before it was opened"
                )
            if _snapshot_chain(absolute, require_target=True) != before_chain:
                raise PathRaceError(
                    "secured embedding input ancestry changed before reading"
                )

            data = _read_descriptor(descriptor, maximum)

            opened_after = os.fstat(descriptor)
            after_path = _require_private_file(
                absolute, expected_device=int(root_metadata.st_dev)
            )
            after_chain = _snapshot_chain(absolute, require_target=True)
            if (
                _file_identity(opened_after) != _file_identity(opened_before)
                or _file_identity(after_path) != _file_identity(opened_before)
                or after_chain != before_chain
            ):
                raise PathRaceError(
                    "secured embedding input changed while it was read"
                )
            if len(data) > maximum:
                raise PathReadLimitError(
                    "secured embedding input exceeds the read maximum"
                )
            self._revalidate_installed_embedding_root()
            return data
        finally:
            os.close(descriptor)


__all__ = [
    "PathRaceError",
    "PathReadLimitError",
    "StudioPaths",
    "UnsafePathError",
]
