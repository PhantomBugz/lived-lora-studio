from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from threading import Lock, RLock
import time
from typing import Callable, Iterable
from uuid import uuid4

from .history_anchor import (
    HistoryAnchorProtocolError,
    advance_history_anchor,
    read_history_anchor,
    require_history_anchor_store,
)

from .paths import (
    PathRaceError,
    UnsafePathError,
    _checked_lstat,
    _create_private_windows_file,
    _entry_identity,
    _make_checked_directory,
    _require_private_directory,
    _require_private_file,
)
from .training_models import TrainingMode
from .training_limits import DATASET_MAX_BYTES, DATASET_MAX_RECORDS


_SCHEMA = "LIVED_DATASET_SNAPSHOT_V1"
_REMOTE_MODES = frozenset({TrainingMode.LORA, TrainingMode.DPO})
_VERSION = re.compile(r"v([0-9]{6})")
_PENDING = re.compile(r"\.pending-([0-9a-f]{32})")
DATASET_RECORD_MAX_BYTES = 16 * 1024 * 1024
_EVAL_DIVISOR = 20
_LOCK_NAME = ".lineage.lock"
_PROCESS_LOCKS_GUARD = Lock()
_PROCESS_LOCKS: dict[str, RLock] = {}
_ACTIVE_LOCK_DESCRIPTORS_GUARD = Lock()
_ACTIVE_LOCK_DESCRIPTORS: set[int] = set()


class DatasetLineageError(RuntimeError):
    pass


class DatasetLineageTamperError(DatasetLineageError):
    pass


@dataclass(frozen=True, slots=True)
class JsonlPartition:
    train_records: tuple[bytes, ...]
    eval_records: tuple[bytes, ...]
    eval_indices: tuple[int, ...]
    train_count: int
    eval_count: int
    train_bytes: int
    eval_bytes: int
    train_sha256: str
    eval_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    mode: TrainingMode
    version: int
    directory: Path
    train_path: Path
    eval_path: Path
    source_sha256: str
    source_bytes: int
    source_count: int
    snapshot_sequence: int
    train_sha256: str
    train_bytes: int
    train_count: int
    eval_sha256: str
    eval_bytes: int
    eval_count: int
    predecessor_manifest_sha256: str | None
    manifest_sha256: str
    created_at: datetime


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetLineageTamperError("dataset manifest has a duplicate key")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _require_canonical_record(line: bytes) -> None:
    if type(line) is not bytes or not line.endswith(b"\n") or line == b"\n":
        raise ValueError("dataset records must be nonempty newline-terminated bytes")
    if len(line) > DATASET_RECORD_MAX_BYTES:
        raise ValueError("dataset record exceeds the released bound")
    try:
        decoded = json.loads(
            line[:-1].decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("dataset record is not valid canonical JSON") from error
    if type(decoded) is not dict or _canonical_json(decoded) != line:
        raise ValueError("dataset record is not canonical JSON")


def partition_jsonl_records(records: Iterable[bytes]) -> JsonlPartition:
    copied = tuple(records)
    if len(copied) > DATASET_MAX_RECORDS:
        raise ValueError("dataset record count exceeds the released bound")
    for line in copied:
        _require_canonical_record(line)
    total_bytes = sum(len(line) for line in copied)
    if total_bytes > DATASET_MAX_BYTES:
        raise ValueError("dataset byte length exceeds the released bound")

    # Canonical content identities provide an order-independent exact-size
    # partition for each immutable dataset version.
    digests = tuple(hashlib.sha256(line).digest() for line in copied)
    eval_indexes = frozenset(select_eval_indices(digests))
    train = tuple(line for index, line in enumerate(copied) if index not in eval_indexes)
    evaluation = tuple(line for index, line in enumerate(copied) if index in eval_indexes)
    train_payload = b"".join(train)
    eval_payload = b"".join(evaluation)
    return JsonlPartition(
        train_records=train,
        eval_records=evaluation,
        eval_indices=tuple(sorted(eval_indexes)),
        train_count=len(train),
        eval_count=len(evaluation),
        train_bytes=len(train_payload),
        eval_bytes=len(eval_payload),
        train_sha256=hashlib.sha256(train_payload).hexdigest(),
        eval_sha256=hashlib.sha256(eval_payload).hexdigest(),
    )


def select_eval_indices(record_sha256: Iterable[bytes]) -> tuple[int, ...]:
    digests = tuple(record_sha256)
    if len(digests) > DATASET_MAX_RECORDS:
        raise ValueError("dataset record count exceeds the released bound")
    if any(type(digest) is not bytes or len(digest) != 32 for digest in digests):
        raise ValueError("record identities must be SHA-256 digest bytes")
    if len(set(digests)) != len(digests):
        raise ValueError("dataset contains a duplicate record identity")

    if not digests:
        return ()
    held_out_count = max(1, (len(digests) + _EVAL_DIVISOR - 1) // _EVAL_DIVISOR)
    selected = sorted(range(len(digests)), key=lambda index: digests[index])[
        :held_out_count
    ]
    return tuple(sorted(selected))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc(value: datetime) -> datetime:
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() is None
        or value.utcoffset().total_seconds() != 0
    ):
        raise ValueError("dataset snapshot clock must return aware UTC")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_private(path: Path, payload: bytes) -> None:
    if os.name == "nt":
        descriptor = _create_private_windows_file(path)
    else:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise UnsafePathError("dataset snapshot file handle is unsafe")
        identity = _entry_identity(opened)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written < 1:
                raise OSError("dataset snapshot write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        if _entry_identity(os.fstat(descriptor)) != identity:
            raise PathRaceError("dataset snapshot file handle changed")
    finally:
        os.close(descriptor)
    if _entry_identity(_require_private_file(path)) != identity:
        raise PathRaceError("dataset snapshot file path changed")


def _process_lock_for(path: Path) -> RLock:
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, RLock())


def _reset_process_locks_after_fork() -> None:
    global _PROCESS_LOCKS_GUARD, _PROCESS_LOCKS
    global _ACTIVE_LOCK_DESCRIPTORS_GUARD, _ACTIVE_LOCK_DESCRIPTORS
    # A POSIX child must not retain the parent's open-file-description flock.
    # Do not acquire an inherited guard here: its owner may have vanished at
    # fork. The callback runs with Python's GIL held, so copying the set is
    # sufficient before replacing all process-local synchronization state.
    inherited = tuple(_ACTIVE_LOCK_DESCRIPTORS)
    _ACTIVE_LOCK_DESCRIPTORS = set()
    _ACTIVE_LOCK_DESCRIPTORS_GUARD = Lock()
    for descriptor in inherited:
        try:
            os.close(descriptor)
        except OSError:
            pass
    _PROCESS_LOCKS_GUARD = Lock()
    _PROCESS_LOCKS = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_locks_after_fork)


def _ensure_lineage_lock_file(path: Path) -> None:
    try:
        _write_new_private(path, b"\0")
    except FileExistsError:
        pass
    last_error: Exception | None = None
    for _ in range(100):
        try:
            if _require_private_file(path).st_size == 1:
                return
            last_error = DatasetLineageTamperError(
                "dataset lineage lock size changed"
            )
        except Exception as error:
            last_error = error
        time.sleep(0.01)
    raise DatasetLineageTamperError(
        "dataset lineage lock is not a private single-byte capability"
    ) from last_error


def _register_lock_descriptor(descriptor: int) -> None:
    with _ACTIVE_LOCK_DESCRIPTORS_GUARD:
        _ACTIVE_LOCK_DESCRIPTORS.add(descriptor)


def _unregister_lock_descriptor(descriptor: int) -> None:
    with _ACTIVE_LOCK_DESCRIPTORS_GUARD:
        _ACTIVE_LOCK_DESCRIPTORS.discard(descriptor)


def _windows_lock_descriptor(descriptor: int) -> None:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    lock = kernel.LockFileEx
    lock.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    lock.restype = wintypes.BOOL
    overlapped = _Overlapped()
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
    if not lock(handle, 0x00000002, 0, 1, 0, ctypes.byref(overlapped)):
        raise OSError(ctypes.get_last_error(), "LockFileEx lineage lock failed")


def _windows_unlock_descriptor(descriptor: int) -> None:
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = (
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        )

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    unlock = kernel.UnlockFileEx
    unlock.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    )
    unlock.restype = wintypes.BOOL
    overlapped = _Overlapped()
    handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
    if not unlock(handle, 0, 1, 0, ctypes.byref(overlapped)):
        raise OSError(ctypes.get_last_error(), "UnlockFileEx lineage lock failed")


class _LineageFileLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor = -1

    def __enter__(self) -> "_LineageFileLock":
        before = _require_private_file(self._path)
        flags = os.O_RDWR
        for name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
            flags |= int(getattr(os, name, 0))
        descriptor = os.open(self._path, flags)
        _register_lock_descriptor(descriptor)
        try:
            opened = os.fstat(descriptor)
            after = _require_private_file(self._path)
            if (
                opened.st_size != 1
                or _entry_identity(before) != _entry_identity(opened)
                or _entry_identity(opened) != _entry_identity(after)
            ):
                raise DatasetLineageTamperError(
                    "dataset lineage lock changed while opened"
                )
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                _windows_lock_descriptor(descriptor)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.read(descriptor, 2) != b"\0":
                raise DatasetLineageTamperError(
                    "dataset lineage lock content changed"
                )
            locked = _require_private_file(self._path)
            if _entry_identity(opened) != _entry_identity(locked):
                raise DatasetLineageTamperError(
                    "dataset lineage lock path changed while held"
                )
        except BaseException:
            _unregister_lock_descriptor(descriptor)
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        descriptor = self._descriptor
        self._descriptor = -1
        if descriptor < 0:
            return
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                _windows_unlock_descriptor(descriptor)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            _unregister_lock_descriptor(descriptor)
            os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel.MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        if not move(os.fspath(source), os.fspath(destination), 0x00000008):
            code = ctypes.get_last_error()
            if code in {80, 183}:
                raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), destination)
            raise OSError(code, "MoveFileExW no-replace publication failed")
        return
    if os.name != "posix":
        raise OSError(errno.ENOSYS, "atomic no-replace publication is unavailable")
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(code, os.strerror(code), destination)
        raise OSError(code, os.strerror(code), destination)


def _read_checked(path: Path, expected_maximum: int) -> bytes:
    before = _require_private_file(path)
    if before.st_size > expected_maximum:
        raise DatasetLineageTamperError("dataset snapshot file exceeds its bound")
    with path.open("rb") as stream:
        payload = stream.read(expected_maximum + 1)
    after = _require_private_file(path)
    if (
        len(payload) > expected_maximum
        or _entry_identity(before) != _entry_identity(after)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise DatasetLineageTamperError("dataset snapshot file changed while read")
    return payload


class DatasetLineageStore:
    def __init__(
        self,
        root: Path,
        *,
        anchor_store: object,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("dataset lineage root must be a Path")
        if not callable(clock):
            raise TypeError("dataset lineage clock must be callable")
        self.root = root
        self._anchor_store = require_history_anchor_store(anchor_store)
        self._clock = clock
        _make_checked_directory(root)
        _require_private_directory(root)
        self._lock_path = root / _LOCK_NAME
        with _process_lock_for(root):
            _ensure_lineage_lock_file(self._lock_path)
            with _LineageFileLock(self._lock_path):
                _require_private_directory(root)
                for mode in _REMOTE_MODES:
                    _make_checked_directory(self._mode_root(mode))
                    _require_private_directory(self._mode_root(mode))

    @contextmanager
    def _transaction(self):
        # Resolve the process lock on every entry. A forked child receives a
        # fresh lock map and cannot inherit a locked RLock from a dead thread.
        with _process_lock_for(self.root):
            _require_private_directory(self.root)
            _require_private_file(self._lock_path)
            with _LineageFileLock(self._lock_path):
                _require_private_directory(self.root)
                yield

    def _mode_root(self, mode: TrainingMode) -> Path:
        if mode not in _REMOTE_MODES:
            raise ValueError("dataset snapshots support only LoRA and DPO")
        return self.root / mode.value

    @staticmethod
    def _anchor_scope(mode: TrainingMode) -> str:
        if mode not in _REMOTE_MODES:
            raise ValueError("dataset snapshots support only LoRA and DPO")
        return f"dataset:{mode.value}"

    def _authenticated_head(self, mode: TrainingMode):
        try:
            return read_history_anchor(
                self._anchor_store, self._anchor_scope(mode)
            )
        except HistoryAnchorProtocolError as error:
            raise DatasetLineageTamperError(
                "authenticated dataset history anchor is invalid"
            ) from error

    @staticmethod
    def _file_names(mode: TrainingMode, version: int) -> tuple[str, str]:
        stem = f"lived-{mode.value}-v{version}"
        return f"{stem}-train.jsonl", f"{stem}-eval.jsonl"

    def _load_snapshot(
        self,
        mode: TrainingMode,
        directory: Path,
        *,
        expected_version: int,
        predecessor: str | None,
    ) -> DatasetSnapshot:
        _require_private_directory(directory)
        manifest_path = directory / "manifest.json"
        manifest_payload = _read_checked(manifest_path, 64 * 1024)
        try:
            document = json.loads(
                manifest_payload.decode("utf-8"),
                object_pairs_hook=_strict_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value: {value}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise DatasetLineageTamperError("dataset manifest is malformed") from error
        if type(document) is not dict or _canonical_json(document) != manifest_payload:
            raise DatasetLineageTamperError("dataset manifest is not canonical")
        expected_top = {
            "schema", "mode", "version", "predecessor_manifest_sha256",
            "created_at_utc", "source", "train", "eval",
        }
        if set(document) != expected_top:
            raise DatasetLineageTamperError("dataset manifest fields changed")
        if (
            document["schema"] != _SCHEMA
            or document["mode"] != mode.value
            or document["version"] != expected_version
            or document["predecessor_manifest_sha256"] != predecessor
        ):
            raise DatasetLineageTamperError("dataset manifest lineage changed")
        source = document["source"]
        train = document["train"]
        evaluation = document["eval"]
        if type(source) is not dict or set(source) != {
            "sha256", "bytes", "records", "snapshot_sequence"
        }:
            raise DatasetLineageTamperError("dataset source identity changed")
        expected_train = {"file", "sha256", "bytes", "records"}
        expected_eval = {*expected_train, "source_indices"}
        if (
            type(train) is not dict
            or type(evaluation) is not dict
            or set(train) != expected_train
            or set(evaluation) != expected_eval
        ):
            raise DatasetLineageTamperError("dataset split identity changed")
        train_name, eval_name = self._file_names(mode, expected_version)
        if train["file"] != train_name or evaluation["file"] != eval_name:
            raise DatasetLineageTamperError("dataset split filename changed")
        for record in (source, train, evaluation):
            if (
                type(record["sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                or type(record["bytes"]) is not int
                or record["bytes"] < 0
                or type(record["records"]) is not int
                or record["records"] < 0
            ):
                raise DatasetLineageTamperError("dataset hash or count is invalid")
        if (
            type(source["snapshot_sequence"]) is not int
            or source["snapshot_sequence"] < 0
            or train["records"] + evaluation["records"] != source["records"]
            or train["bytes"] + evaluation["bytes"] != source["bytes"]
            or type(evaluation["source_indices"]) is not list
            or any(type(index) is not int for index in evaluation["source_indices"])
            or evaluation["source_indices"]
            != sorted(set(evaluation["source_indices"]))
            or len(evaluation["source_indices"]) != evaluation["records"]
            or any(
                index < 0 or index >= source["records"]
                for index in evaluation["source_indices"]
            )
        ):
            raise DatasetLineageTamperError("dataset split totals changed")
        train_path = directory / train_name
        eval_path = directory / eval_name
        train_payload = _read_checked(train_path, DATASET_MAX_BYTES)
        eval_payload = _read_checked(eval_path, DATASET_MAX_BYTES)
        if (
            len(train_payload) != train["bytes"]
            or len(eval_payload) != evaluation["bytes"]
            or hashlib.sha256(train_payload).hexdigest() != train["sha256"]
            or hashlib.sha256(eval_payload).hexdigest() != evaluation["sha256"]
        ):
            raise DatasetLineageTamperError("dataset split hash changed")
        train_records = tuple(train_payload.splitlines(keepends=True))
        eval_records = tuple(eval_payload.splitlines(keepends=True))
        if (
            len(train_records) != train["records"]
            or len(eval_records) != evaluation["records"]
        ):
            raise DatasetLineageTamperError("dataset split record count changed")
        for line in (*train_records, *eval_records):
            _require_canonical_record(line)
        eval_indices = frozenset(evaluation["source_indices"])
        train_iterator = iter(train_records)
        eval_iterator = iter(eval_records)
        source_records = tuple(
            next(eval_iterator) if index in eval_indices else next(train_iterator)
            for index in range(source["records"])
        )
        try:
            next(train_iterator)
            raise DatasetLineageTamperError("dataset train split has trailing records")
        except StopIteration:
            pass
        try:
            next(eval_iterator)
            raise DatasetLineageTamperError("dataset eval split has trailing records")
        except StopIteration:
            pass
        source_payload = b"".join(source_records)
        expected_partition = partition_jsonl_records(source_records)
        if (
            len(source_payload) != source["bytes"]
            or hashlib.sha256(source_payload).hexdigest() != source["sha256"]
            or expected_partition.eval_indices
            != tuple(evaluation["source_indices"])
        ):
            raise DatasetLineageTamperError("dataset source identity changed")
        try:
            created_at = datetime.fromisoformat(document["created_at_utc"])
        except (TypeError, ValueError) as error:
            raise DatasetLineageTamperError("dataset timestamp is invalid") from error
        _require_utc(created_at)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        return DatasetSnapshot(
            mode=mode,
            version=expected_version,
            directory=directory,
            train_path=train_path,
            eval_path=eval_path,
            source_sha256=source["sha256"],
            source_bytes=source["bytes"],
            source_count=source["records"],
            snapshot_sequence=source["snapshot_sequence"],
            train_sha256=train["sha256"],
            train_bytes=train["bytes"],
            train_count=train["records"],
            eval_sha256=evaluation["sha256"],
            eval_bytes=evaluation["bytes"],
            eval_count=evaluation["records"],
            predecessor_manifest_sha256=predecessor,
            manifest_sha256=manifest_sha256,
            created_at=created_at,
        )

    def _discard_unanchored_pending(
        self,
        directory: Path,
        mode: TrainingMode,
        expected_version: int,
    ) -> None:
        match = _PENDING.fullmatch(directory.name)
        if directory.parent != self._mode_root(mode) or match is None:
            raise DatasetLineageTamperError(
                "dataset pending path is outside its authenticated lineage"
            )
        _require_private_directory(directory)
        train_name, eval_name = self._file_names(mode, expected_version)
        allowed = {train_name, eval_name, "manifest.json"}
        entries = tuple(directory.iterdir())
        if any(entry.name not in allowed for entry in entries):
            raise DatasetLineageTamperError(
                "unanchored dataset pending object contains unknown content"
            )
        for entry in entries:
            _require_private_file(entry)
        shutil.rmtree(directory)

    def snapshots(self, mode: TrainingMode) -> tuple[DatasetSnapshot, ...]:
        with self._transaction():
            return self._snapshots_locked(mode)

    def _snapshots_locked(self, mode: TrainingMode) -> tuple[DatasetSnapshot, ...]:
        root = self._mode_root(mode)
        _require_private_directory(root)
        entries = tuple(root.iterdir())
        unexpected = tuple(
            entry.name
            for entry in entries
            if _VERSION.fullmatch(entry.name) is None
            and _PENDING.fullmatch(entry.name) is None
        )
        if unexpected:
            raise DatasetLineageTamperError(
                "unknown dataset lineage object must be resolved"
            )
        pending = tuple(
            entry for entry in entries if _PENDING.fullmatch(entry.name) is not None
        )
        versions = sorted(
            (
                (_VERSION.fullmatch(entry.name), entry)
                for entry in entries
                if _VERSION.fullmatch(entry.name) is not None
            ),
            key=lambda item: int(item[0].group(1)),
        )
        snapshots: list[DatasetSnapshot] = []
        predecessor: str | None = None
        for ordinal, (match, directory) in enumerate(versions, start=1):
            assert match is not None
            if int(match.group(1)) != ordinal:
                raise DatasetLineageTamperError("dataset versions are not contiguous")
            snapshot = self._load_snapshot(
                mode,
                directory,
                expected_version=ordinal,
                predecessor=predecessor,
            )
            snapshots.append(snapshot)
            predecessor = snapshot.manifest_sha256
        try:
            anchor = self._authenticated_head(mode)
        except Exception as error:
            if isinstance(error, DatasetLineageTamperError):
                raise
            raise DatasetLineageTamperError(
                "authenticated dataset history anchor cannot be verified"
            ) from error
        if anchor.count == len(snapshots):
            if anchor.head_sha256 != predecessor:
                raise DatasetLineageTamperError(
                    "authenticated dataset history head changed"
                )
            for directory in pending:
                self._discard_unanchored_pending(
                    directory,
                    mode,
                    len(snapshots) + 1,
                )
            if pending:
                _fsync_directory(root)
            return tuple(snapshots)
        if (
            anchor.count != len(snapshots) + 1
            or len(pending) != 1
        ):
            raise DatasetLineageTamperError(
                "authenticated dataset history was truncated or extended"
            )
        ordinal = len(snapshots) + 1
        pending_snapshot = self._load_snapshot(
                mode,
                pending[0],
                expected_version=ordinal,
                predecessor=predecessor,
            )
        if pending_snapshot.manifest_sha256 != anchor.head_sha256:
            raise DatasetLineageTamperError(
                "pending dataset version does not match its authenticated anchor"
            )
        final = root / f"v{ordinal:06d}"
        try:
            _checked_lstat(final)
        except FileNotFoundError:
            pass
        else:
            raise DatasetLineageTamperError(
                "pending dataset version conflicts with a final version"
            )
        _rename_no_replace(pending[0], final)
        _fsync_directory(root)
        replayed = self._load_snapshot(
            mode,
            final,
            expected_version=ordinal,
            predecessor=predecessor,
        )
        if replayed.manifest_sha256 != anchor.head_sha256:
            raise DatasetLineageTamperError(
                "replayed dataset version changed after publication"
            )
        snapshots.append(replayed)
        return tuple(snapshots)

    def publish(self, dataset: object) -> DatasetSnapshot:
        with self._transaction():
            return self._publish_locked(dataset)

    def _publish_locked(self, dataset: object) -> DatasetSnapshot:
        mode = getattr(dataset, "mode", None)
        if mode not in _REMOTE_MODES:
            raise ValueError("dataset snapshot requires a LoRA or DPO dataset")
        chunks = getattr(dataset, "iter_transport_chunks", None)
        if not callable(chunks):
            raise TypeError("dataset must expose verified transport chunks")
        payload_parts: list[bytes] = []
        total = 0
        for chunk in chunks():
            if type(chunk) is not bytes:
                raise TypeError("dataset transport chunks must be bytes")
            total += len(chunk)
            if total > DATASET_MAX_BYTES:
                raise ValueError("dataset exceeds the snapshot byte bound")
            payload_parts.append(chunk)
        payload = b"".join(payload_parts)
        record_count = getattr(dataset, "record_count", None)
        byte_length = getattr(dataset, "byte_length", None)
        source_sha256 = getattr(dataset, "sha256", None)
        snapshot_sequence = getattr(dataset, "snapshot_sequence", None)
        if (
            type(record_count) is not int
            or type(byte_length) is not int
            or type(source_sha256) is not str
            or type(snapshot_sequence) is not int
            or record_count < 1
            or snapshot_sequence < 0
            or byte_length != len(payload)
            or hashlib.sha256(payload).hexdigest() != source_sha256
        ):
            raise DatasetLineageError("prepared dataset identity changed")
        records = tuple(payload.splitlines(keepends=True))
        if len(records) != record_count:
            raise DatasetLineageError("prepared dataset record count changed")
        partition = partition_jsonl_records(records)

        existing = self._snapshots_locked(mode)
        if existing:
            latest = existing[-1]
            if (
                latest.source_sha256 == source_sha256
                and latest.source_bytes == byte_length
                and latest.source_count == record_count
                and latest.snapshot_sequence == snapshot_sequence
            ):
                return latest
        version = len(existing) + 1
        predecessor = existing[-1].manifest_sha256 if existing else None
        train_name, eval_name = self._file_names(mode, version)
        now = _require_utc(self._clock())
        document = {
            "schema": _SCHEMA,
            "mode": mode.value,
            "version": version,
            "predecessor_manifest_sha256": predecessor,
            "created_at_utc": now.isoformat(),
            "source": {
                "sha256": source_sha256,
                "bytes": byte_length,
                "records": record_count,
                "snapshot_sequence": snapshot_sequence,
            },
            "train": {
                "file": train_name,
                "sha256": partition.train_sha256,
                "bytes": partition.train_bytes,
                "records": partition.train_count,
            },
            "eval": {
                "file": eval_name,
                "sha256": partition.eval_sha256,
                "bytes": partition.eval_bytes,
                "records": partition.eval_count,
                "source_indices": list(partition.eval_indices),
            },
        }
        manifest_payload = _canonical_json(document)
        mode_root = self._mode_root(mode)
        pending = mode_root / f".pending-{uuid4().hex}"
        final = mode_root / f"v{version:06d}"
        _make_checked_directory(pending)
        anchor_attempted = False
        try:
            _write_new_private(pending / train_name, b"".join(partition.train_records))
            _write_new_private(pending / eval_name, b"".join(partition.eval_records))
            _write_new_private(pending / "manifest.json", manifest_payload)
            _fsync_directory(pending)
            try:
                _checked_lstat(final)
            except FileNotFoundError:
                pass
            else:
                raise DatasetLineageError("dataset version already exists")
            anchor_attempted = True
            advance_history_anchor(
                self._anchor_store,
                self._anchor_scope(mode),
                expected_count=len(existing),
                expected_head_sha256=predecessor,
                new_head_sha256=hashlib.sha256(manifest_payload).hexdigest(),
            )
            _rename_no_replace(pending, final)
            _fsync_directory(mode_root)
        except BaseException:
            try:
                if (
                    not anchor_attempted
                    and pending.exists()
                    and _PENDING.fullmatch(pending.name)
                ):
                    self._discard_unanchored_pending(
                        pending,
                        mode,
                        version,
                    )
            except OSError:
                pass
            raise
        snapshots = self._snapshots_locked(mode)
        published = snapshots[-1]
        if (
            published.version != version
            or published.source_sha256 != source_sha256
            or published.manifest_sha256
            != hashlib.sha256(manifest_payload).hexdigest()
        ):
            raise DatasetLineageTamperError("published dataset identity changed")
        return published


__all__ = [
    "DATASET_MAX_BYTES",
    "DATASET_MAX_RECORDS",
    "DATASET_RECORD_MAX_BYTES",
    "DatasetLineageError",
    "DatasetLineageStore",
    "DatasetLineageTamperError",
    "DatasetSnapshot",
    "JsonlPartition",
    "partition_jsonl_records",
    "select_eval_indices",
]
