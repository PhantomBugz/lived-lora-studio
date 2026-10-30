from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterator
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

from .paths import PathRaceError, StudioPaths, UnsafePathError
from .storage import StudioVault


MAX_ENVELOPE_BYTES = 60_000
_PLAINTEXT_CHUNK_BYTES = 40_000
_SALT_BYTES = 16
_NONCE_BYTES = 12
_KEY_BYTES = 32
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_BUNDLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ExportError(RuntimeError):
    pass


class NoApprovedPairsError(ExportError):
    pass


@dataclass(frozen=True, slots=True)
class _StagedFile:
    name: str
    identity: tuple[int, int, int, int, int, int]
    sha256: str


def _file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        stat.S_IFMT(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
    )


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0)) & 0x0400
    )


def _require_regular_single_link(
    metadata: os.stat_result, *, message: str
) -> None:
    if (
        _is_link_or_reparse(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or int(metadata.st_nlink) != 1
    ):
        raise PathRaceError(message)


def _read_all(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _FILE_READ_ATTRIBUTES = 0x0080
    _DELETE_ACCESS = 0x00010000
    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x00000001
    _FILE_SHARE_WRITE = 0x00000002
    _FILE_SHARE_DELETE = 0x00000004
    _CREATE_NEW = 1
    _OPEN_EXISTING = 3
    _FILE_ATTRIBUTE_NORMAL = 0x00000080
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
    _FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    _CreateFileW.restype = wintypes.HANDLE
    _GetFileInformationByHandle = _kernel32.GetFileInformationByHandle
    _GetFileInformationByHandle.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    )
    _GetFileInformationByHandle.restype = wintypes.BOOL
    _GetFinalPathNameByHandleW = _kernel32.GetFinalPathNameByHandleW
    _GetFinalPathNameByHandleW.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    _GetFinalPathNameByHandleW.restype = wintypes.DWORD
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = (wintypes.HANDLE,)
    _CloseHandle.restype = wintypes.BOOL


def _windows_open_directory(path: Path, *, block_delete: bool) -> int:
    share = _FILE_SHARE_READ | _FILE_SHARE_WRITE
    if not block_delete:
        share |= _FILE_SHARE_DELETE
    handle = _CreateFileW(
        os.fspath(path),
        _FILE_READ_ATTRIBUTES | (_DELETE_ACCESS if block_delete else 0),
        share,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        error = ctypes.get_last_error()
        raise PathRaceError(
            f"staging directory could not be pinned: {ctypes.WinError(error)}"
        )
    return int(handle)


def _windows_open_file(path: Path, *, create: bool, share_delete: bool) -> int:
    import msvcrt

    desired_access = _GENERIC_READ | (_GENERIC_WRITE if create else 0)
    share = _FILE_SHARE_READ | (_FILE_SHARE_DELETE if share_delete else 0)
    handle = _CreateFileW(
        os.fspath(path),
        desired_access,
        share,
        None,
        _CREATE_NEW if create else _OPEN_EXISTING,
        _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
        None,
    )
    if handle == _INVALID_HANDLE_VALUE:
        error = ctypes.get_last_error()
        raise PathRaceError(
            f"staged file could not be opened securely: {ctypes.WinError(error)}"
        )
    flags = (os.O_RDWR if create else os.O_RDONLY) | int(
        getattr(os, "O_BINARY", 0)
    )
    try:
        return msvcrt.open_osfhandle(int(handle), flags)
    except BaseException:
        _CloseHandle(handle)
        raise


def _windows_handle_identity(handle: int) -> tuple[int, int, int]:
    information = _ByHandleFileInformation()
    if not _GetFileInformationByHandle(handle, ctypes.byref(information)):
        error = ctypes.get_last_error()
        raise PathRaceError(
            f"staging directory identity could not be read: {ctypes.WinError(error)}"
        )
    if int(information.dwFileAttributes) & 0x0400:
        raise PathRaceError("staging directory became a reparse point")
    return (
        int(information.dwVolumeSerialNumber),
        (int(information.nFileIndexHigh) << 32) | int(information.nFileIndexLow),
        int(information.dwFileAttributes) & 0x10,
    )


def _windows_final_path(handle: int) -> Path:
    capacity = 512
    while True:
        buffer = ctypes.create_unicode_buffer(capacity)
        length = int(_GetFinalPathNameByHandleW(handle, buffer, capacity, 0))
        if length == 0:
            error = ctypes.get_last_error()
            raise PathRaceError(
                f"staging directory path could not be read: {ctypes.WinError(error)}"
            )
        if length < capacity:
            return Path(buffer.value)
        capacity = length + 1


class _PinnedStagingDirectory:
    """Capability for one staging directory object, independent of its pathname."""

    def __init__(self, paths: StudioPaths):
        self._paths = paths
        try:
            configured = paths.secure_directory("studio_data/staging", create=False)
        except UnsafePathError as error:
            raise PathRaceError("staging directory is not securely bound") from error
        self.configured_path = Path(os.path.abspath(configured))
        metadata = os.lstat(self.configured_path)
        if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise PathRaceError("staging directory is not a regular directory")
        self._path_identity = (
            int(metadata.st_dev),
            int(metadata.st_ino),
            stat.S_IFMT(metadata.st_mode),
        )
        self._closed = False
        if os.name == "nt":
            self._handle = _windows_open_directory(
                self.configured_path, block_delete=True
            )
            try:
                self._handle_identity = _windows_handle_identity(self._handle)
            except BaseException:
                _CloseHandle(self._handle)
                raise
            self._descriptor = -1
        else:
            flags = os.O_RDONLY
            for option in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
                flags |= int(getattr(os, option, 0))
            try:
                self._descriptor = os.open(self.configured_path, flags)
            except OSError as error:
                raise PathRaceError("staging directory could not be pinned") from error
            self._handle = -1
            opened = os.fstat(self._descriptor)
            if (
                int(opened.st_dev),
                int(opened.st_ino),
                stat.S_IFMT(opened.st_mode),
            ) != self._path_identity:
                self.close()
                raise PathRaceError("staging directory changed while it was pinned")

    def _ensure_open(self) -> None:
        if self._closed:
            raise PathRaceError("staging directory capability is closed")

    def assert_bound(self) -> None:
        """Require the configured path to still name the pinned directory."""
        self._ensure_open()
        try:
            current = self._paths.secure_directory(
                "studio_data/staging", create=False
            )
            metadata = os.lstat(current)
        except (OSError, UnsafePathError) as error:
            raise PathRaceError("staging directory binding changed") from error
        current_identity = (
            int(metadata.st_dev),
            int(metadata.st_ino),
            stat.S_IFMT(metadata.st_mode),
        )
        if current_identity != self._path_identity:
            raise PathRaceError("staging directory identity changed")
        if os.name == "nt":
            probe = _windows_open_directory(Path(current), block_delete=False)
            try:
                if _windows_handle_identity(probe) != self._handle_identity:
                    raise PathRaceError("staging directory identity changed")
            finally:
                _CloseHandle(probe)
        else:
            opened = os.fstat(self._descriptor)
            opened_identity = (
                int(opened.st_dev),
                int(opened.st_ino),
                stat.S_IFMT(opened.st_mode),
            )
            if opened_identity != self._path_identity:
                raise PathRaceError("staging directory identity changed")

    @contextmanager
    def _operation_root(self, *, require_bound: bool) -> Iterator[Path | None]:
        self._ensure_open()
        if require_bound:
            self.assert_bound()
        if os.name != "nt":
            # All child operations are descriptor-relative. POSIX permits a
            # same-owner rename while this descriptor is open, but it cannot
            # redirect an operation into the replacement pathname; the binding
            # checks before and after handoff then fail closed.
            yield None
            return

        actual = _windows_final_path(self._handle)
        if _windows_handle_identity(self._handle) != self._handle_identity:
            raise PathRaceError("pinned staging directory identity changed")
        if require_bound:
            self.assert_bound()
        yield actual
        if _windows_handle_identity(self._handle) != self._handle_identity:
            raise PathRaceError("pinned staging directory changed during use")
        if require_bound:
            self.assert_bound()

    @staticmethod
    def _validate_name(name: str) -> None:
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or chr(92) in name
            or "\x00" in name
        ):
            raise PathRaceError("invalid staged filename")

    def _stat_child(self, name: str, root: Path | None) -> os.stat_result:
        if os.name == "nt":
            assert root is not None
            return os.lstat(root / name)
        return os.stat(name, dir_fd=self._descriptor, follow_symlinks=False)

    def _open_child(
        self,
        name: str,
        root: Path | None,
        *,
        create: bool,
        share_delete: bool = False,
    ) -> int:
        flags = os.O_RDWR if create else os.O_RDONLY
        for option in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= int(getattr(os, option, 0))
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        try:
            if os.name == "nt":
                assert root is not None
                return _windows_open_file(
                    root / name,
                    create=create,
                    share_delete=share_delete,
                )
            return os.open(
                name,
                flags,
                0o600,
                dir_fd=self._descriptor,
            )
        except OSError as error:
            raise PathRaceError("staged file could not be opened securely") from error

    def _unlink_child(self, name: str, root: Path | None) -> None:
        if os.name == "nt":
            assert root is not None
            os.unlink(root / name)
        else:
            os.unlink(name, dir_fd=self._descriptor)

    def create(self, name: str, content: bytes) -> _StagedFile:
        self._validate_name(name)
        descriptor = -1
        created_identity: tuple[int, int, int, int, int, int] | None = None
        try:
            with self._operation_root(require_bound=True) as root:
                descriptor = self._open_child(name, root, create=True)
                view = memoryview(content)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write to staging file")
                    view = view[written:]
                os.fsync(descriptor)
                opened = os.fstat(descriptor)
                _require_regular_single_link(
                    opened, message="new staged file is not a private regular file"
                )
                created_identity = _file_identity(opened)
                on_path = self._stat_child(name, root)
                _require_regular_single_link(
                    on_path, message="new staged file was substituted"
                )
                if _file_identity(on_path) != created_identity:
                    raise PathRaceError("new staged file identity changed")
            self.assert_bound()
            return _StagedFile(
                name=name,
                identity=created_identity,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        except BaseException:
            if descriptor >= 0 and created_identity is None:
                try:
                    created_identity = _file_identity(os.fstat(descriptor))
                except OSError:
                    pass
            if descriptor >= 0:
                os.close(descriptor)
                descriptor = -1
            if created_identity is not None:
                try:
                    self._remove_one(
                        _StagedFile(
                            name=name,
                            identity=created_identity,
                            sha256=hashlib.sha256(content).hexdigest(),
                        ),
                        missing_ok=True,
                    )
                except Exception:
                    pass
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _open_verified(
        self,
        record: _StagedFile,
        root: Path | None,
        *,
        share_delete: bool = False,
    ) -> tuple[int, bytes]:
        before = self._stat_child(record.name, root)
        _require_regular_single_link(
            before, message="staged export file was replaced before broker handoff"
        )
        if _file_identity(before) != record.identity:
            raise PathRaceError("staged export file identity changed")
        descriptor = self._open_child(
            record.name,
            root,
            create=False,
            share_delete=share_delete,
        )
        try:
            opened = os.fstat(descriptor)
            _require_regular_single_link(
                opened, message="staged export descriptor is not a private regular file"
            )
            if _file_identity(opened) != record.identity:
                raise PathRaceError("staged export changed before secure open")
            data = _read_all(descriptor, record.identity[4])
            if (
                len(data) != record.identity[4]
                or hashlib.sha256(data).hexdigest() != record.sha256
            ):
                raise PathRaceError("staged export bytes do not match the sealed bundle")
            return descriptor, data
        except BaseException:
            os.close(descriptor)
            raise

    def _reverify_open(
        self,
        record: _StagedFile,
        descriptor: int,
        root: Path | None,
    ) -> None:
        opened = os.fstat(descriptor)
        _require_regular_single_link(
            opened, message="staged export descriptor changed during broker handoff"
        )
        if _file_identity(opened) != record.identity:
            raise PathRaceError("staged export descriptor changed during broker handoff")
        os.lseek(descriptor, 0, os.SEEK_SET)
        data = _read_all(descriptor, record.identity[4])
        if (
            len(data) != record.identity[4]
            or hashlib.sha256(data).hexdigest() != record.sha256
        ):
            raise PathRaceError("staged export bytes changed during broker handoff")
        on_path = self._stat_child(record.name, root)
        _require_regular_single_link(
            on_path, message="staged export file was replaced during broker handoff"
        )
        if _file_identity(on_path) != record.identity:
            raise PathRaceError("staged export file identity changed during broker handoff")

    @contextmanager
    def verified_payloads(
        self, records: tuple[_StagedFile, ...]
    ) -> Iterator[dict[str, str]]:
        descriptors: list[tuple[_StagedFile, int]] = []
        with self._operation_root(require_bound=True) as root:
            try:
                payloads: dict[str, str] = {}
                for record in records:
                    descriptor, data = self._open_verified(record, root)
                    descriptors.append((record, descriptor))
                    try:
                        payloads[record.name] = data.decode("utf-8")
                    except UnicodeDecodeError as error:
                        raise PathRaceError(
                            "staged export is not valid UTF-8"
                        ) from error
                try:
                    yield payloads
                finally:
                    for record, descriptor in descriptors:
                        self._reverify_open(record, descriptor, root)
            finally:
                for _record, descriptor in descriptors:
                    os.close(descriptor)

    def _remove_one(self, record: _StagedFile, *, missing_ok: bool) -> None:
        with self._operation_root(require_bound=False) as root:
            try:
                metadata = self._stat_child(record.name, root)
            except FileNotFoundError:
                if missing_ok:
                    return
                raise PathRaceError("staged export disappeared before cleanup") from None
            _require_regular_single_link(
                metadata, message="staged export was replaced before cleanup"
            )
            if _file_identity(metadata) != record.identity:
                raise PathRaceError("staged export identity changed before cleanup")
            descriptor, data = self._open_verified(
                record, root, share_delete=(os.name == "nt")
            )
            try:
                if hashlib.sha256(data).hexdigest() != record.sha256:
                    raise PathRaceError("staged export bytes changed before cleanup")
                self._unlink_child(record.name, root)
                after = os.fstat(descriptor)
                if int(after.st_nlink) != 0 and os.name != "nt":
                    raise PathRaceError("staged export cleanup removed another file")
            finally:
                os.close(descriptor)

    def cleanup(self, records: tuple[_StagedFile, ...]) -> None:
        first_error: BaseException | None = None
        try:
            for record in reversed(records):
                try:
                    self._remove_one(record, missing_ok=True)
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        finally:
            self.close()
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if os.name == "nt":
            if self._handle >= 0:
                _CloseHandle(self._handle)
                self._handle = -1
        elif self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(frozen=True, slots=True)
class ExportBundle:
    bundle_id: str
    manifest_path: Path
    shard_paths: tuple[Path, ...]
    record_count: int
    plaintext_bytes: int
    _staging: _PinnedStagingDirectory = field(repr=False, compare=False)
    _records: tuple[_StagedFile, ...] = field(repr=False, compare=False)

    @contextmanager
    def verified_payloads(self) -> Iterator[dict[str, str]]:
        with self._staging.verified_payloads(self._records) as payloads:
            yield payloads

    def cleanup(self) -> None:
        self._staging.cleanup(self._records)


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _derive_export_key(passphrase: str, salt: bytes) -> bytes:
    if not isinstance(passphrase, str) or not passphrase.strip():
        raise ValueError("passphrase must be a nonempty string")
    if len(passphrase) < 12:
        raise ValueError("passphrase must contain at least 12 characters")
    encoded = passphrase.encode("utf-8")
    if len(encoded) > 4096:
        raise ValueError("passphrase exceeds the UTF-8 size limit")
    return Scrypt(
        salt=salt,
        length=_KEY_BYTES,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
    ).derive(encoded)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class EncryptedJsonlExporter:
    """Writes approved pairs as one-way encrypted broker envelope shards."""

    def __init__(self, vault: StudioVault, paths: StudioPaths):
        if not callable(getattr(vault, "list_pairs", None)):
            raise TypeError("vault must provide list_pairs")
        if not isinstance(paths, StudioPaths):
            raise TypeError("paths must be a StudioPaths instance")
        self.vault = vault
        self.paths = paths.ensure()

    def export(
        self,
        *,
        passphrase: str,
        bundle_name: str = "training-set",
    ) -> ExportBundle:
        if not isinstance(bundle_name, str) or not _BUNDLE_NAME.fullmatch(bundle_name):
            raise ValueError(
                "bundle_name must contain only letters, digits, dots, underscores, or hyphens"
            )
        salt = os.urandom(_SALT_BYTES)
        key = _derive_export_key(passphrase, salt)
        approved_pairs = self.vault.list_pairs(approved_only=True)
        approved_pairs = [pair for pair in approved_pairs if pair.user_approved]
        if not approved_pairs:
            raise NoApprovedPairsError("no user-approved pairs are available for export")

        lines = []
        for pair in approved_pairs:
            lines.append(
                _json_bytes(
                    {
                        "messages": [
                            {"role": "user", "content": pair.instruction},
                            {"role": "assistant", "content": pair.response},
                        ],
                        "user_approved": True,
                        "source_provenance_hash": pair.source_provenance_hash,
                        "tags": list(pair.tags),
                    }
                )
            )
        plaintext = b"\n".join(lines) + b"\n"
        chunks = tuple(
            plaintext[offset : offset + _PLAINTEXT_CHUNK_BYTES]
            for offset in range(0, len(plaintext), _PLAINTEXT_CHUNK_BYTES)
        )

        bundle_id = uuid4().hex
        prefix = f"{bundle_name}-{bundle_id}"
        shard_names: list[str] = []
        shard_contents: list[bytes] = []
        cipher = AESGCM(key)
        for index, chunk in enumerate(chunks):
            aad = (
                f"lived-lora-studio:export:v1:{bundle_id}:{index}:{len(chunks)}"
            ).encode("utf-8")
            nonce = os.urandom(_NONCE_BYTES)
            ciphertext = cipher.encrypt(nonce, chunk, aad)
            envelope = {
                "format": "lived-lora-studio.encrypted-jsonl.v1",
                "bundle_id": bundle_id,
                "index": index,
                "count": len(chunks),
                "kdf": {
                    "name": "scrypt",
                    "salt": _b64(salt),
                    "length": _KEY_BYTES,
                    "n": _SCRYPT_N,
                    "r": _SCRYPT_R,
                    "p": _SCRYPT_P,
                },
                "cipher": {
                    "name": "aes-256-gcm",
                    "nonce": _b64(nonce),
                    "ciphertext": _b64(ciphertext),
                },
                "aad": _b64(aad),
            }
            encoded_envelope = _json_bytes(envelope)
            if len(encoded_envelope) > MAX_ENVELOPE_BYTES:
                raise ExportError("encrypted shard exceeds the broker size limit")
            shard_names.append(f"{prefix}.{index:04d}.json")
            shard_contents.append(encoded_envelope)

        ordered_digest = hashlib.sha256()
        for content in shard_contents:
            ordered_digest.update(hashlib.sha256(content).digest())
        manifest = {
            "format": "lived-lora-studio.export-manifest.v1",
            "bundle_id": bundle_id,
            "record_count": len(approved_pairs),
            "plaintext_bytes": len(plaintext),
            "shard_count": len(shard_names),
            "shards": {
                "first_index": 0,
                "filename_template": f"{prefix}.{{index:04d}}.json",
                "ordered_sha256": ordered_digest.hexdigest(),
            },
        }
        manifest_name = f"{prefix}.manifest.json"
        encoded_manifest = _json_bytes(manifest)
        if len(encoded_manifest) > MAX_ENVELOPE_BYTES:
            raise ExportError("export manifest exceeds the broker size limit")

        staging = _PinnedStagingDirectory(self.paths)
        shard_records: list[_StagedFile] = []
        records: list[_StagedFile] = []
        try:
            for name, content in zip(shard_names, shard_contents, strict=True):
                record = staging.create(name, content)
                shard_records.append(record)
                records.append(record)
            manifest_record = staging.create(manifest_name, encoded_manifest)
            records.append(manifest_record)
            staging.assert_bound()
            return ExportBundle(
                bundle_id=bundle_id,
                manifest_path=staging.configured_path / manifest_name,
                shard_paths=tuple(
                    staging.configured_path / record.name
                    for record in shard_records
                ),
                record_count=len(approved_pairs),
                plaintext_bytes=len(plaintext),
                _staging=staging,
                _records=(manifest_record, *shard_records),
            )
        except BaseException:
            staging.cleanup(tuple(records))
            raise
