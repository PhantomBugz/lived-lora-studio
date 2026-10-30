from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import ctypes
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from tempfile import mkdtemp
from typing import Protocol
from uuid import uuid4

from .paths import (
    PathRaceError,
    StudioPaths,
    UnsafePathError,
    _absolute_lexical,
    _checked_lstat,
    _entry_identity,
    _file_identity,
    _read_descriptor,
    _require_private_directory,
    _require_private_file,
    _set_private_windows_acl,
    _snapshot_chain,
)
from .storage import RagIndexVersion, StudioVault, VaultAuthenticationError
from .training_audit import RunManifestWriter
from .training_data import select_lora_pairs
from .training_intelligence import semantic_diversity
from .training_models import (
    ApprovalGrant,
    CpuExecutionInventory,
    EmbeddingAssessment,
    ReadinessSnapshot,
    RunState,
    TrainingEvent,
    TrainingMode,
    TrainingPlan,
    TrainingRunSnapshot,
)


_DIMENSION = 384
_CHUNK_SIZE = 1_200
_CHUNK_OVERLAP = 120
_MAX_DOCUMENTS = 50_000
_MAX_DOCUMENT_BYTES = 4_000_000
_MAX_DATASET_BYTES = 33_554_432
_MAX_CHUNKS = 100_000
_MAX_QUERY_BYTES = 8_192
_MAX_RESULTS = 64
_MAX_INDEX_BYTES = 268_435_456
_MAX_METADATA_CLEAR_BYTES = 67_108_864
_MAX_METADATA_BYTES = _MAX_METADATA_CLEAR_BYTES + 64
_MAX_CACHE_BYTES = 67_108_864
_MAX_CACHE_FILES = 128
_MAX_CACHE_AGGREGATE_BYTES = 134_217_728
_MAX_MODEL_FILE_BYTES = 1_073_741_824
_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_REQUIRED_MODEL_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
_MAX_MODEL_TREE_ENTRIES = len(_REQUIRED_MODEL_FILES) + 2
_TOKENIZER_MODEL_FILES = frozenset((
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
))
_RUNTIME_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+-]{0,31}")

_SHA256 = re.compile(r"[a-f0-9]{64}")
_HEX32 = re.compile(r"[a-f0-9]{32}")
_SECRET = re.compile(
    r"(?i)(?:password|passwd|secret|token|api[_-]?key|authorization|cookie)\s*[:=]"
)
_BIDI_SINGLETONS = frozenset((0x061C, 0x200E, 0x200F))
_BIDI_RANGES = ((0x202A, 0x202E), (0x2066, 0x2069))


def _training_phase_sha256(phase: str) -> str:
    return hashlib.sha256(
        b"liveD:training-phase:v1" + bytes((0,)) + phase.encode("utf-8")
    ).hexdigest()


class RagError(RuntimeError):
    pass


class RagValidationError(RagError):
    pass


class RagCacheError(RagValidationError):
    pass


class EmbeddingRuntimeBlocked(RagError):
    pass


class RagReconciliationBlocked(RagError):
    pass


class EmbeddingBackend(Protocol):
    identity_sha256: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class IndexBackend(Protocol):
    identity_sha256: str

    def build_batches(
        self,
        batches: Iterable[Sequence[Sequence[float]]],
        *,
        dimension: int,
        expected_count: int,
    ) -> bytes: ...
    def load(self, payload: bytes, *, dimension: int) -> object: ...
    def search(
        self, index: object, vector: Sequence[float], *, top_k: int
    ) -> Sequence[tuple[float, int]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingModelInventory:
    model_root: Path
    manifest_sha256: str
    tokenizer_identity_sha256: str
    sentence_transformers_version: str
    file_count: int
    file_manifest: tuple[tuple[str, int, str], ...]


@dataclass(frozen=True, slots=True)
class RagChunk:
    opaque_document_id: str
    ordinal: int
    text: str
    text_sha256: str


@dataclass(frozen=True, slots=True)
class RagSearchResult:
    opaque_document_id: str
    chunk_ordinal: int
    score: float
    text: str

    def __post_init__(self) -> None:
        if _HEX32.fullmatch(self.opaque_document_id) is None:
            raise ValueError("opaque document identifier is invalid")
        if type(self.chunk_ordinal) is not int or self.chunk_ordinal < 0:
            raise ValueError("chunk ordinal is invalid")
        if type(self.score) not in (int, float) or not math.isfinite(float(self.score)):
            raise ValueError("search score must be finite")
        if type(self.text) is not str or not self.text:
            raise ValueError("search text must be nonempty")


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RagValidationError("RAG state is not canonical JSON") from error


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _strict_json(payload: bytes, *, maximum: int) -> object:
    if type(payload) is not bytes or not 1 <= len(payload) <= maximum:
        raise RagValidationError("JSON payload is outside its size bound")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("non-finite JSON")),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise RagValidationError("JSON payload failed strict parsing") from error
    if _canonical_json(value) != payload:
        raise RagValidationError("JSON payload is not canonical")
    return value


def _unsafe_control(value: str) -> bool:
    for character in value:
        point = ord(character)
        if (
            point == 0
            or (point < 0x20 and character not in "\n\r\t")
            or 0x7F <= point <= 0x9F
            or point in _BIDI_SINGLETONS
            or any(lower <= point <= upper for lower, upper in _BIDI_RANGES)
        ):
            return True
    return False


def chunk_text(
    text: str, *, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP
) -> tuple[str, ...]:
    if type(text) is not str:
        raise TypeError("text must be a string")
    if type(size) is not int or type(overlap) is not int:
        raise TypeError("size and overlap must be integers")
    if not 1 <= size <= 65_536:
        raise ValueError("chunk size is outside the allowed bound")
    if not 0 <= overlap < size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    if len(text.encode("utf-8")) > _MAX_DOCUMENT_BYTES:
        raise ValueError("document exceeds the RAG input bound")
    if not text:
        return ()
    step = size - overlap
    chunks = []
    for offset in range(0, len(text), step):
        chunk = text[offset : offset + size]
        if not chunk:
            break
        chunks.append(chunk)
        if offset + size >= len(text):
            break
    return tuple(chunks)


def _expected_model_file_type(relative: str) -> str:
    suffix = Path(relative).suffix.casefold()
    if suffix == ".json":
        return "json"
    if suffix == ".safetensors":
        return "safetensors"
    if suffix == ".txt":
        return "text"
    raise EmbeddingRuntimeBlocked("embedding model file type is unsupported")


def _model_entry_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        *_entry_identity(metadata),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _model_tree_state_once(
    root: Path,
    *,
    windows_read_only: bool = False,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[tuple[str, tuple[int, int, int]], ...],
]:
    try:
        _require_private_directory(
            root, windows_read_only=windows_read_only
        )
        root_metadata = _checked_lstat(root)
        ancestry = _snapshot_chain(root, require_target=True)
    except (OSError, UnsafePathError) as error:
        raise EmbeddingRuntimeBlocked(
            "embedding model root is not private and stable"
        ) from error
    expected_device = int(root_metadata.st_dev)
    pending = [root]
    files: list[str] = []
    identities: list[tuple[str, tuple[int, ...]]] = [
        (".", _model_entry_identity(root_metadata))
    ]
    inspected = 0
    while pending:
        current = pending.pop()
        try:
            iterator = os.scandir(current)
        except OSError as error:
            raise EmbeddingRuntimeBlocked(
                "embedding model tree cannot be enumerated"
            ) from error
        entries = []
        try:
            for entry in iterator:
                inspected += 1
                if inspected > _MAX_MODEL_TREE_ENTRIES:
                    raise EmbeddingRuntimeBlocked(
                        "embedding model tree has too many entries"
                    )
                entries.append(entry)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        for entry in sorted(entries, key=lambda item: item.name, reverse=True):
            path = Path(entry.path)
            try:
                metadata = _checked_lstat(path)
            except (OSError, UnsafePathError) as error:
                raise EmbeddingRuntimeBlocked(
                    "embedding model entry cannot be inspected"
                ) from error
            relative = path.relative_to(root).as_posix()
            identities.append((relative, _model_entry_identity(metadata)))
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    _require_private_directory(
                        path,
                        expected_device=expected_device,
                        windows_read_only=windows_read_only,
                    )
                except UnsafePathError as error:
                    raise EmbeddingRuntimeBlocked(
                        "embedding model directory is not private"
                    ) from error
                pending.append(path)
            elif stat.S_ISREG(metadata.st_mode):
                try:
                    _require_private_file(
                        path,
                        expected_device=expected_device,
                        windows_read_only=windows_read_only,
                    )
                except UnsafePathError as error:
                    raise EmbeddingRuntimeBlocked(
                        "embedding model file is not private"
                    ) from error
                if metadata.st_nlink != 1:
                    raise EmbeddingRuntimeBlocked(
                        "embedding model tree contains a hard link"
                    )
                if not 1 <= metadata.st_size <= _MAX_MODEL_FILE_BYTES:
                    raise EmbeddingRuntimeBlocked(
                        "embedding model file size is outside its bound"
                    )
                files.append(relative)
            else:
                raise EmbeddingRuntimeBlocked(
                    "embedding model tree contains an unsafe object"
                )
    return tuple(sorted(files)), tuple(sorted(identities)), ancestry


def _stable_model_tree_state(
    root: Path,
    *,
    windows_read_only: bool = False,
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    before_files, before_identities, before_ancestry = _model_tree_state_once(
        root, windows_read_only=windows_read_only
    )
    after_files, after_identities, after_ancestry = _model_tree_state_once(
        root, windows_read_only=windows_read_only
    )
    if (
        after_files != before_files
        or after_identities != before_identities
        or after_ancestry != before_ancestry
    ):
        raise EmbeddingRuntimeBlocked(
            "embedding model tree changed during verification"
        )
    return after_files, after_identities


def _model_tree_files(root: Path) -> tuple[str, ...]:
    files, _identity = _stable_model_tree_state(root)
    return files


def _model_tree_identity(root: Path) -> tuple[tuple[str, tuple[int, ...]], ...]:
    _files, identity = _stable_model_tree_state(root)
    return identity


def _expected_model_tree_entries() -> tuple[str, ...]:
    directories = {
        Path(relative).parent.as_posix()
        for relative in _REQUIRED_MODEL_FILES
        if Path(relative).parent != Path(".")
    }
    return tuple(sorted((".", *directories, *_REQUIRED_MODEL_FILES, "MODEL-MANIFEST.json")))


def _model_file_digest(path: Path, *, maximum: int) -> tuple[int, str]:
    if type(maximum) is not int or not 1 <= maximum <= _MAX_MODEL_FILE_BYTES:
        raise EmbeddingRuntimeBlocked("embedding model file bound is invalid")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(min(1_048_576, maximum - size + 1))
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum:
                    raise EmbeddingRuntimeBlocked(
                        "embedding model file exceeds its authenticated size"
                    )
                digest.update(chunk)
    except OSError as error:
        raise EmbeddingRuntimeBlocked(
            "embedding model file could not be authenticated"
        ) from error
    return size, digest.hexdigest()


def _windows_model_guard_identity(handle: int) -> tuple[int, bytes]:
    from ctypes import wintypes

    class FileId128(ctypes.Structure):
        _fields_ = (("identifier", ctypes.c_ubyte * 16),)

    class FileIdInfo(ctypes.Structure):
        _fields_ = (
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", FileId128),
        )

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    query.restype = wintypes.BOOL
    information = FileIdInfo()
    if not query(
        wintypes.HANDLE(handle),
        18,
        ctypes.byref(information),
        ctypes.sizeof(information),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return (
        int(information.volume_serial_number),
        bytes(information.file_id.identifier),
    )


def _open_windows_model_guard(path: Path, *, directory: bool) -> int:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create.restype = wintypes.HANDLE
    desired = (0x00000080 if directory else 0x80000000) | 0x00020000
    flags = 0x00200000 | (0x02000000 if directory else 0x00000080)
    handle = create(
        os.fspath(path),
        desired,
        0x00000001,
        None,
        3,
        flags,
        None,
    )
    numeric = int(handle)
    if numeric == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return numeric


def _close_windows_model_guard(handle: int) -> None:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    close = kernel.CloseHandle
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    if not close(wintypes.HANDLE(handle)):
        raise ctypes.WinError(ctypes.get_last_error())


def _copy_windows_model_guard(
    handle: int,
    destination: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    read_file = kernel.ReadFile
    read_file.argtypes = (
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    )
    read_file.restype = wintypes.BOOL
    digest = hashlib.sha256()
    total = 0
    try:
        with destination.open("xb") as target:
            while total < expected_size:
                requested = min(1_048_576, expected_size - total)
                buffer = ctypes.create_string_buffer(requested)
                received = wintypes.DWORD()
                if not read_file(
                    wintypes.HANDLE(handle),
                    buffer,
                    requested,
                    ctypes.byref(received),
                    None,
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                count = int(received.value)
                if count == 0:
                    raise EmbeddingRuntimeBlocked(
                        "embedding model guard ended before its authenticated size"
                    )
                payload = buffer.raw[:count]
                target.write(payload)
                digest.update(payload)
                total += count
            probe = ctypes.create_string_buffer(1)
            received = wintypes.DWORD()
            if not read_file(
                wintypes.HANDLE(handle),
                probe,
                1,
                ctypes.byref(received),
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if received.value:
                raise EmbeddingRuntimeBlocked(
                    "embedding model guard exceeds its authenticated size"
                )
            target.flush()
            os.fsync(target.fileno())
    except OSError as error:
        raise EmbeddingRuntimeBlocked(
            "embedding model guard could not be copied"
        ) from error
    if total != expected_size or digest.hexdigest() != expected_sha256:
        raise EmbeddingRuntimeBlocked(
            "embedding model guard differs from its authenticated manifest"
        )


class _WindowsSnapshotModelView:
    def __init__(
        self,
        files: tuple[str, ...],
        source_guards: tuple[tuple[str, int, object], ...],
        file_manifest: tuple[tuple[str, int, str], ...],
    ) -> None:
        self._files = files
        self._directories = tuple(
            sorted(
                {
                    Path(relative).parent.as_posix()
                    for relative in files
                    if Path(relative).parent != Path(".")
                },
                key=lambda value: (len(Path(value).parts), value),
            )
        )
        self._root = Path(mkdtemp(prefix="lived-embedding-view-")).resolve()
        self._identity: tuple[tuple[str, tuple[int, ...]], ...] = ()
        self._guards: list[tuple[str, int, object]] = []
        self._closed = False
        self._sealed = False
        manifest = {
            relative: (size, sha256)
            for relative, size, sha256 in file_manifest
        }
        source_files = {
            relative: guard
            for relative, guard, _expected in source_guards
            if relative in files
        }
        if set(manifest) != set(files) or set(source_files) != set(files):
            raise EmbeddingRuntimeBlocked(
                "embedding model snapshot inputs are incomplete"
            )
        try:
            _set_private_windows_acl(self._root, directory=True)
            for relative in self._directories:
                directory = self._root / Path(relative)
                directory.mkdir()
                _set_private_windows_acl(directory, directory=True)
            for relative in self._files:
                destination = self._root / Path(relative)
                expected_size, expected_sha256 = manifest[relative]
                _copy_windows_model_guard(
                    source_files[relative],
                    destination,
                    expected_size=expected_size,
                    expected_sha256=expected_sha256,
                )
                _set_private_windows_acl(destination, directory=False)
            for relative in self._files:
                _set_private_windows_acl(
                    self._root / Path(relative),
                    directory=False,
                    read_only=True,
                )
            for relative in reversed(self._directories):
                _set_private_windows_acl(
                    self._root / Path(relative),
                    directory=True,
                    read_only=True,
                )
            _set_private_windows_acl(
                self._root, directory=True, read_only=True
            )
            self._sealed = True
            snapshot_files, snapshot_identity = _stable_model_tree_state(
                self._root, windows_read_only=True
            )
            if (
                snapshot_files != files
                or tuple(path for path, _state in snapshot_identity)
                != _expected_model_tree_entries()
            ):
                raise EmbeddingRuntimeBlocked(
                    "embedding model snapshot contains unexpected objects"
                )
            identity_by_path = dict(snapshot_identity)
            for relative in _expected_model_tree_entries():
                candidate = (
                    self._root
                    if relative == "."
                    else self._root / Path(relative)
                )
                guard = _open_windows_model_guard(
                    candidate,
                    directory=relative == "." or relative not in files,
                )
                try:
                    expected = _windows_model_guard_identity(guard)
                    if (
                        _model_entry_identity(_checked_lstat(candidate))
                        != identity_by_path[relative]
                    ):
                        raise EmbeddingRuntimeBlocked(
                            "embedding model snapshot changed while guarded"
                        )
                except BaseException:
                    _close_windows_model_guard(guard)
                    raise
                self._guards.append((relative, guard, expected))
            self._identity = snapshot_identity
            self.reverify()
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def load_root(self) -> Path:
        self.reverify()
        return self._root

    def reverify(self) -> None:
        if self._closed:
            raise EmbeddingRuntimeBlocked(
                "embedding model snapshot is closed"
            )
        files, identity = _stable_model_tree_state(
            self._root, windows_read_only=True
        )
        if files != self._files or identity != self._identity:
            raise EmbeddingRuntimeBlocked(
                "embedding model snapshot changed while leased"
            )
        for _relative, guard, expected in self._guards:
            try:
                current = _windows_model_guard_identity(guard)
            except OSError as error:
                raise EmbeddingRuntimeBlocked(
                    "embedding model snapshot guard could not be verified"
                ) from error
            if current != expected:
                raise EmbeddingRuntimeBlocked(
                    "embedding model snapshot guard identity changed"
                )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        root_guard: int | None = None
        for relative, guard, _expected in reversed(self._guards):
            if relative == ".":
                root_guard = guard
                continue
            try:
                _close_windows_model_guard(guard)
            except OSError as error:
                errors.append(error)
        self._guards.clear()
        if self._sealed:
            try:
                _set_private_windows_acl(self._root, directory=True)
                for relative in self._directories:
                    _set_private_windows_acl(
                        self._root / Path(relative), directory=True
                    )
                for relative in self._files:
                    _set_private_windows_acl(
                        self._root / Path(relative), directory=False
                    )
            except (OSError, UnsafePathError) as error:
                errors.append(error)
            self._sealed = False
        for relative in reversed(self._files):
            try:
                (self._root / Path(relative)).unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                errors.append(error)
        for relative in reversed(self._directories):
            try:
                (self._root / Path(relative)).rmdir()
            except FileNotFoundError:
                pass
            except OSError as error:
                errors.append(error)
        if root_guard is not None:
            try:
                _close_windows_model_guard(root_guard)
            except OSError as error:
                errors.append(error)
        try:
            self._root.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            errors.append(error)
        if errors:
            raise EmbeddingRuntimeBlocked(
                "embedding model snapshot could not be released"
            ) from errors[0]


class _LinuxDescriptorModelView:
    def __init__(
        self,
        files: tuple[str, ...],
        guards: tuple[tuple[str, int, object], ...],
        file_manifest: tuple[tuple[str, int, str], ...],
    ) -> None:
        self._files = files
        self._guards = {
            relative: (guard, expected)
            for relative, guard, expected in guards
            if relative in files
        }
        if set(self._guards) != set(files):
            raise EmbeddingRuntimeBlocked(
                "embedding model file descriptors are incomplete"
            )
        self._manifest = {
            relative: (size, sha256)
            for relative, size, sha256 in file_manifest
        }
        if set(self._manifest) != set(files):
            raise EmbeddingRuntimeBlocked(
                "embedding model descriptor manifest is incomplete"
            )
        self._directories = tuple(
            sorted(
                {
                    Path(relative).parent.as_posix()
                    for relative in files
                    if Path(relative).parent != Path(".")
                },
                key=lambda value: (len(Path(value).parts), value),
            )
        )
        self._root = Path(mkdtemp(prefix="lived-embedding-view-"))
        self._descriptor: int | None = None
        self._identity: tuple[int, ...] | None = None
        self._sealed = False
        try:
            os.chmod(self._root, 0o700, follow_symlinks=False)
            for relative in self._directories:
                os.mkdir(self._root / Path(relative), 0o700)
            for relative in self._files:
                guard, _expected = self._guards[relative]
                os.symlink(
                    f"/proc/self/fd/{guard}",
                    self._root / Path(relative),
                    target_is_directory=False,
                )
            for relative in reversed(self._directories):
                os.chmod(
                    self._root / Path(relative),
                    0o500,
                    follow_symlinks=False,
                )
            os.chmod(self._root, 0o500, follow_symlinks=False)
            self._sealed = True
            flags = os.O_RDONLY
            for name in ("O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY"):
                flags |= int(getattr(os, name, 0))
            descriptor = os.open(self._root, flags)
            try:
                os.set_inheritable(descriptor, False)
                identity = _model_entry_identity(os.fstat(descriptor))
            except BaseException:
                os.close(descriptor)
                raise
            self._descriptor = descriptor
            self._identity = identity
            self.reverify()
            descriptor_root = Path("/proc/self/fd") / str(descriptor)
            for relative in self._files:
                expected_size, expected_sha256 = self._manifest[relative]
                if _model_file_digest(
                    descriptor_root / Path(relative), maximum=expected_size
                ) != (expected_size, expected_sha256):
                    raise EmbeddingRuntimeBlocked(
                        "embedding descriptor model file differs from its manifest"
                    )
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def load_root(self) -> Path:
        self.reverify()
        descriptor = self._descriptor
        if descriptor is None:
            raise EmbeddingRuntimeBlocked(
                "embedding descriptor model view is closed"
            )
        return Path("/proc/self/fd") / str(descriptor)

    def reverify(self) -> None:
        descriptor = self._descriptor
        expected_identity = self._identity
        if descriptor is None or expected_identity is None:
            raise EmbeddingRuntimeBlocked(
                "embedding descriptor model view is closed"
            )
        descriptor_root = Path("/proc/self/fd") / str(descriptor)
        try:
            descriptor_identity = _model_entry_identity(os.fstat(descriptor))
            path_identity = _model_entry_identity(
                os.stat(descriptor_root, follow_symlinks=True)
            )
        except OSError as error:
            raise EmbeddingRuntimeBlocked(
                "embedding descriptor model view is unavailable"
            ) from error
        if descriptor_identity != expected_identity or path_identity != expected_identity:
            raise EmbeddingRuntimeBlocked(
                "embedding descriptor model view identity changed"
            )
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o500:
            raise EmbeddingRuntimeBlocked(
                "embedding descriptor model view became writable"
            )
        for relative in self._directories:
            try:
                directory_metadata = os.stat(
                    descriptor_root / Path(relative), follow_symlinks=False
                )
            except OSError as error:
                raise EmbeddingRuntimeBlocked(
                    "embedding descriptor model directory is unavailable"
                ) from error
            if (
                not stat.S_ISDIR(directory_metadata.st_mode)
                or stat.S_IMODE(directory_metadata.st_mode) != 0o500
            ):
                raise EmbeddingRuntimeBlocked(
                    "embedding descriptor model directory became writable"
                )
        for relative in self._files:
            guard, expected = self._guards[relative]
            candidate = descriptor_root / Path(relative)
            expected_target = f"/proc/self/fd/{guard}"
            try:
                link_metadata = os.lstat(candidate)
                target = os.readlink(candidate)
                candidate_identity = _model_entry_identity(
                    os.stat(candidate, follow_symlinks=True)
                )
                guard_identity = _model_entry_identity(os.fstat(guard))
            except OSError as error:
                raise EmbeddingRuntimeBlocked(
                    "embedding descriptor model file is unavailable"
                ) from error
            if (
                not stat.S_ISLNK(link_metadata.st_mode)
                or target != expected_target
                or candidate_identity != expected
                or guard_identity != expected
            ):
                raise EmbeddingRuntimeBlocked(
                    "embedding descriptor model file identity changed"
                )

    def close(self) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        self._identity = None
        errors: list[BaseException] = []
        if descriptor is not None:
            if self._sealed:
                try:
                    os.fchmod(descriptor, 0o700)
                    for relative in self._directories:
                        os.chmod(
                            relative,
                            0o700,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                except OSError as error:
                    errors.append(error)
                self._sealed = False
            for relative in reversed(self._files):
                try:
                    os.unlink(relative, dir_fd=descriptor)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    errors.append(error)
            for relative in reversed(self._directories):
                try:
                    os.rmdir(relative, dir_fd=descriptor)
                except FileNotFoundError:
                    pass
                except OSError as error:
                    errors.append(error)
            try:
                os.close(descriptor)
            except OSError as error:
                errors.append(error)
        else:
            if self._sealed:
                try:
                    os.chmod(self._root, 0o700, follow_symlinks=False)
                    for relative in self._directories:
                        os.chmod(
                            self._root / Path(relative),
                            0o700,
                            follow_symlinks=False,
                        )
                except OSError as error:
                    errors.append(error)
                self._sealed = False
            for relative in reversed(self._files):
                try:
                    (self._root / Path(relative)).unlink()
                except FileNotFoundError:
                    pass
                except OSError as error:
                    errors.append(error)
            for relative in reversed(self._directories):
                try:
                    (self._root / Path(relative)).rmdir()
                except FileNotFoundError:
                    pass
                except OSError as error:
                    errors.append(error)
        try:
            self._root.rmdir()
        except FileNotFoundError:
            pass
        except OSError as error:
            errors.append(error)
        if errors:
            raise EmbeddingRuntimeBlocked(
                "embedding descriptor model view could not be released"
            ) from errors[0]


class _ModelTreeLease:
    def __init__(
        self,
        root: Path,
        files: tuple[str, ...],
        identity: tuple[tuple[str, tuple[int, ...]], ...],
        guards: tuple[tuple[str, int, object], ...],
        file_manifest: tuple[tuple[str, int, str], ...],
        *,
        windows: bool,
    ) -> None:
        self.root = root
        self._files = files
        self._identity = identity
        self._guards = list(guards)
        self._windows = windows
        self._closed = False
        roots = [guard for relative, guard, _expected in guards if relative == "."]
        if len(roots) != 1:
            raise EmbeddingRuntimeBlocked(
                "embedding model root descriptor is unavailable"
            )
        self._root_guard = roots[0]
        self._model_view: _WindowsSnapshotModelView | _LinuxDescriptorModelView = (
            _WindowsSnapshotModelView(files, guards, file_manifest)
            if windows
            else _LinuxDescriptorModelView(files, guards, file_manifest)
        )

    def load_root(self) -> Path:
        """Return a path bound to the leased root descriptor on Linux."""
        if self._closed:
            raise EmbeddingRuntimeBlocked(
                "embedding model tree lease is closed"
            )
        return self._model_view.load_root()

    def reverify(self) -> None:
        if self._closed:
            raise EmbeddingRuntimeBlocked(
                "embedding model tree lease is closed"
            )
        files, identity = _stable_model_tree_state(self.root)
        if files != self._files or identity != self._identity:
            raise EmbeddingRuntimeBlocked(
                "embedding model tree changed while leased"
            )
        for _relative, guard, expected in self._guards:
            try:
                current = (
                    _windows_model_guard_identity(guard)
                    if self._windows
                    else _model_entry_identity(os.fstat(guard))
                )
            except OSError as error:
                raise EmbeddingRuntimeBlocked(
                    "embedding model guard could not be verified"
                ) from error
            if current != expected:
                raise EmbeddingRuntimeBlocked(
                    "embedding model guard identity changed"
                )
        self._model_view.reverify()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        try:
            self._model_view.close()
        except BaseException as error:
            errors.append(error)
        for _relative, guard, _expected in reversed(self._guards):
            try:
                if self._windows:
                    _close_windows_model_guard(guard)
                else:
                    os.close(guard)
            except OSError as error:
                errors.append(error)
        self._guards.clear()
        if errors:
            raise EmbeddingRuntimeBlocked(
                "embedding model guards could not be released"
            ) from errors[0]


def _open_model_tree_lease(
    root: Path,
    file_manifest: tuple[tuple[str, int, str], ...],
) -> _ModelTreeLease:
    files, identity = _stable_model_tree_state(root)
    if tuple(relative for relative, _state in identity) != _expected_model_tree_entries():
        raise EmbeddingRuntimeBlocked(
            "embedding model tree contains unexpected objects"
        )
    identity_by_path = dict(identity)
    guards: list[tuple[str, int, object]] = []
    windows = os.name == "nt"
    try:
        for relative in _expected_model_tree_entries():
            candidate = root if relative == "." else root / Path(relative)
            directory = relative == "." or relative not in files
            metadata = _checked_lstat(candidate)
            expected_path_identity = identity_by_path[relative]
            if _model_entry_identity(metadata) != expected_path_identity:
                raise EmbeddingRuntimeBlocked(
                    "embedding model entry changed before leasing"
                )
            if windows:
                guard = _open_windows_model_guard(
                    candidate, directory=directory
                )
                expected_guard_identity: object = (
                    _windows_model_guard_identity(guard)
                )
            else:
                flags = os.O_RDONLY
                for name in ("O_CLOEXEC", "O_NOFOLLOW"):
                    flags |= int(getattr(os, name, 0))
                if directory:
                    flags |= int(getattr(os, "O_DIRECTORY", 0))
                guard = os.open(candidate, flags)
                try:
                    os.set_inheritable(guard, False)
                    import fcntl
                    fcntl.flock(
                        guard, fcntl.LOCK_SH | fcntl.LOCK_NB
                    )
                    expected_guard_identity = _model_entry_identity(
                        os.fstat(guard)
                    )
                    if expected_guard_identity != expected_path_identity:
                        raise EmbeddingRuntimeBlocked(
                            "embedding model descriptor differs from its path"
                        )
                except BaseException as error:
                    try:
                        os.close(guard)
                    except OSError:
                        pass
                    if isinstance(error, EmbeddingRuntimeBlocked):
                        raise
                    raise EmbeddingRuntimeBlocked(
                        "embedding model descriptor lock is unavailable"
                    ) from error
            guards.append((relative, guard, expected_guard_identity))
        lease = _ModelTreeLease(
            root,
            files,
            identity,
            tuple(guards),
            file_manifest,
            windows=windows,
        )
        lease.reverify()
        return lease
    except BaseException:
        for _relative, guard, _expected in reversed(guards):
            try:
                if windows:
                    _close_windows_model_guard(guard)
                else:
                    os.close(guard)
            except OSError:
                pass
        raise


def verify_embedding_model_inventory(
    paths: StudioPaths, expected_manifest_sha256: str
) -> EmbeddingModelInventory:
    if type(paths) is not StudioPaths:
        raise TypeError("paths must be StudioPaths")
    if (
        type(expected_manifest_sha256) is not str
        or _SHA256.fullmatch(expected_manifest_sha256) is None
    ):
        raise EmbeddingRuntimeBlocked(
            "release-pinned embedding manifest identity is unavailable"
        )
    try:
        manifest = paths.secure_embedding_read_bytes(
            "MODEL-MANIFEST.json", maximum=1_048_576
        )
    except (FileNotFoundError, OSError, UnsafePathError, ValueError) as error:
        raise EmbeddingRuntimeBlocked("pinned embedding MODEL-MANIFEST.json is unavailable") from error
    manifest_sha256 = hashlib.sha256(manifest).hexdigest()
    if manifest_sha256 != expected_manifest_sha256:
        raise EmbeddingRuntimeBlocked(
            "embedding model manifest differs from the release trust root"
        )
    try:
        document = _strict_json(manifest, maximum=1_048_576)
    except RagValidationError as error:
        raise EmbeddingRuntimeBlocked("embedding model manifest failed authentication") from error
    expected = {
        "dimension", "files", "model_id", "schema",
        "sentence_transformers_version", "tokenizer_identity_sha256",
    }
    if type(document) is not dict or set(document) != expected:
        raise EmbeddingRuntimeBlocked("embedding model manifest schema is invalid")
    if (
        document["schema"] != "LIVED_EMBEDDING_MODEL_INVENTORY_V1"
        or document["model_id"] != _MODEL_ID
        or type(document["dimension"]) is not int
        or document["dimension"] != _DIMENSION
        or type(document["sentence_transformers_version"]) is not str
        or _RUNTIME_VERSION.fullmatch(document["sentence_transformers_version"])
        is None
        or type(document["tokenizer_identity_sha256"]) is not str
        or _SHA256.fullmatch(document["tokenizer_identity_sha256"]) is None
    ):
        raise EmbeddingRuntimeBlocked("embedding model manifest identity is invalid")
    files = document["files"]
    if type(files) is not list or len(files) != len(_REQUIRED_MODEL_FILES):
        raise EmbeddingRuntimeBlocked("embedding model manifest file count is invalid")
    seen = []
    for entry in files:
        if type(entry) is not dict or set(entry) != {"bytes", "path", "sha256", "type"}:
            raise EmbeddingRuntimeBlocked("embedding model file entry is invalid")
        relative = entry["path"]
        if type(relative) is not str or relative not in _REQUIRED_MODEL_FILES or relative in seen:
            raise EmbeddingRuntimeBlocked("embedding model file path is invalid")
        if type(entry["bytes"]) is not int or not 1 <= entry["bytes"] <= _MAX_MODEL_FILE_BYTES:
            raise EmbeddingRuntimeBlocked("embedding model file size is invalid")
        if type(entry["sha256"]) is not str or _SHA256.fullmatch(entry["sha256"]) is None:
            raise EmbeddingRuntimeBlocked("embedding model file hash is invalid")
        if entry["type"] != _expected_model_file_type(relative):
            raise EmbeddingRuntimeBlocked("embedding model file type is invalid")
        try:
            payload = paths.secure_embedding_read_bytes(
                relative, maximum=entry["bytes"]
            )
        except (FileNotFoundError, OSError, UnsafePathError, ValueError) as error:
            raise EmbeddingRuntimeBlocked("embedding model file is unavailable") from error
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise EmbeddingRuntimeBlocked("embedding model file differs from its manifest")
        seen.append(relative)
    if tuple(seen) != _REQUIRED_MODEL_FILES:
        raise EmbeddingRuntimeBlocked("embedding model manifest ordering is invalid")
    tokenizer_material = [
        [entry["path"], entry["bytes"], entry["sha256"]]
        for entry in files
        if entry["path"] in _TOKENIZER_MODEL_FILES
    ]
    tokenizer_identity = hashlib.sha256(
        _canonical_json(tokenizer_material)
    ).hexdigest()
    if tokenizer_identity != document["tokenizer_identity_sha256"]:
        raise EmbeddingRuntimeBlocked(
            "embedding tokenizer identity is inconsistent with the manifest"
        )
    try:
        root = paths.secure_embedding_directory()
    except (OSError, UnsafePathError) as error:
        raise EmbeddingRuntimeBlocked("embedding model root is unavailable") from error
    model_files, model_identity = _stable_model_tree_state(root)
    if (
        model_files
        != tuple(sorted((*_REQUIRED_MODEL_FILES, "MODEL-MANIFEST.json")))
        or tuple(relative for relative, _state in model_identity)
        != _expected_model_tree_entries()
    ):
        raise EmbeddingRuntimeBlocked(
            "embedding model tree contains unexpected bytes"
        )
    return EmbeddingModelInventory(
        model_root=root,
        manifest_sha256=manifest_sha256,
        tokenizer_identity_sha256=document["tokenizer_identity_sha256"],
        sentence_transformers_version=document["sentence_transformers_version"],
        file_count=len(files),
        file_manifest=tuple(
            (entry["path"], entry["bytes"], entry["sha256"])
            for entry in files
        )
        + (("MODEL-MANIFEST.json", len(manifest), manifest_sha256),),
    )


class PinnedLocalEmbedder:
    dimension = _DIMENSION

    def __init__(
        self,
        inventory: EmbeddingModelInventory,
        model: object,
        lease: _ModelTreeLease,
    ) -> None:
        self.inventory = inventory
        self.identity_sha256 = inventory.manifest_sha256
        self._model = model
        self._lease: _ModelTreeLease | None = lease

    @classmethod
    def from_paths(
        cls, paths: StudioPaths, expected_manifest_sha256: str
    ) -> "PinnedLocalEmbedder":
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        inventory = verify_embedding_model_inventory(
            paths, expected_manifest_sha256
        )
        lease = _open_model_tree_lease(
            inventory.model_root, inventory.file_manifest
        )
        try:
            locked = verify_embedding_model_inventory(
                paths, expected_manifest_sha256
            )
            lease.reverify()
            if locked != inventory:
                raise EmbeddingRuntimeBlocked(
                    "embedding model changed before it was loaded"
                )
            load_root = lease.load_root()
            try:
                module = importlib.import_module("sentence_transformers")
            except (ImportError, OSError) as error:
                raise EmbeddingRuntimeBlocked(
                    "pinned sentence-transformers runtime is unavailable"
                ) from error
            if (
                getattr(module, "__version__", None)
                != inventory.sentence_transformers_version
            ):
                raise EmbeddingRuntimeBlocked(
                    "sentence-transformers version differs from the model manifest"
                )
            try:
                model = module.SentenceTransformer(
                    os.fspath(load_root),
                    device="cpu",
                    local_files_only=True,
                    trust_remote_code=False,
                )
                model.eval()
            except Exception as error:
                raise EmbeddingRuntimeBlocked(
                    "pinned embedding model could not be loaded offline"
                ) from error
            after = verify_embedding_model_inventory(
                paths, expected_manifest_sha256
            )
            lease.reverify()
            if after != inventory:
                raise EmbeddingRuntimeBlocked(
                    "embedding model changed while it was loaded"
                )
            return cls(inventory, model, lease)
        except BaseException:
            lease.close()
            raise

    def close(self) -> None:
        lease = self._lease
        if lease is None:
            return
        self._lease = None
        lease.close()

    def __enter__(self) -> "PinnedLocalEmbedder":
        if self._lease is None:
            raise EmbeddingRuntimeBlocked(
                "embedding model tree lease is closed"
            )
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings")
        if len(texts) > 256 or not all(type(value) is str for value in texts):
            raise ValueError("embedding batch is invalid")
        lease = self._lease
        if lease is None:
            raise RagValidationError("local embedding model is closed")
        try:
            lease.reverify()
            encoded = self._model.encode(
                list(texts),
                batch_size=min(32, max(1, len(texts))),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            lease.reverify()
            return encoded.tolist()
        except Exception as error:
            raise RagValidationError("local embedding operation failed") from error


class NativeFaissBackend:
    def __init__(self) -> None:
        numpy, faiss = self._modules()
        if getattr(numpy, "__version__", None) != "2.5.3":
            raise EmbeddingRuntimeBlocked("pinned NumPy 2.5.3 runtime is unavailable")
        if getattr(faiss, "__version__", None) != "1.15.0":
            raise EmbeddingRuntimeBlocked("pinned FAISS 1.15.0 runtime is unavailable")
        self.identity_sha256 = hashlib.sha256(
            b"liveD:faiss-index-flat-ip:v1\0numpy=2.5.3\0faiss=1.15.0"
        ).hexdigest()

    @staticmethod
    def _modules():
        try:
            numpy = importlib.import_module("numpy")
            faiss = importlib.import_module("faiss")
        except (ImportError, OSError) as error:
            raise EmbeddingRuntimeBlocked("pinned NumPy/FAISS runtime is unavailable") from error
        return numpy, faiss

    def build_batches(
        self,
        batches: Iterable[Sequence[Sequence[float]]],
        *,
        dimension: int,
        expected_count: int,
    ) -> bytes:
        if type(dimension) is not int or dimension != _DIMENSION:
            raise RagValidationError("FAISS dimension must be exactly 384")
        if (
            type(expected_count) is not int
            or not 1 <= expected_count <= _MAX_CHUNKS
        ):
            raise RagValidationError("FAISS vector count is outside its bound")
        numpy, faiss = self._modules()
        index = faiss.IndexFlatIP(dimension)
        count = 0
        for batch in batches:
            if not batch:
                continue
            array = numpy.asarray(batch, dtype="float32", order="C")
            if (
                array.ndim != 2
                or array.shape[1] != dimension
                or not bool(numpy.all(numpy.isfinite(array)))
            ):
                raise RagValidationError("FAISS vector batch is invalid")
            norms = numpy.linalg.norm(array, axis=1)
            if not bool(
                numpy.all(numpy.isclose(norms, 1.0, rtol=1e-5, atol=1e-5))
            ):
                raise RagValidationError("FAISS vector batch is not normalized")
            count += int(array.shape[0])
            if count > expected_count:
                raise RagValidationError("FAISS received too many vectors")
            index.add(array)
        if count != expected_count or int(index.ntotal) != expected_count:
            raise RagValidationError("FAISS vector count is incomplete")
        payload = faiss.serialize_index(index)
        return bytes(memoryview(payload))

    def build(self, vectors: Sequence[Sequence[float]], *, dimension: int) -> bytes:
        return self.build_batches(
            (vectors,),
            dimension=dimension,
            expected_count=len(vectors),
        )

    def load(self, payload: bytes, *, dimension: int) -> object:
        numpy, faiss = self._modules()
        index = faiss.deserialize_index(numpy.frombuffer(payload, dtype="uint8").copy())
        if type(index).__name__ != "IndexFlatIP" or index.d != dimension:
            raise RagValidationError("serialized FAISS index has the wrong type or dimension")
        return index

    def search(
        self, index: object, vector: Sequence[float], *, top_k: int
    ) -> Sequence[tuple[float, int]]:
        numpy, _faiss = self._modules()
        scores, ordinals = index.search(
            numpy.asarray([vector], dtype="float32", order="C"), top_k
        )
        return tuple(
            (float(score), int(ordinal))
            for score, ordinal in zip(scores[0], ordinals[0], strict=True)
            if int(ordinal) >= 0
        )


def _rag_documents(snapshot: ReadinessSnapshot):
    captures = {capture.capture_id: capture for capture in snapshot.captures}
    selected = []
    seen = set()
    for pair in select_lora_pairs(snapshot):
        if pair.capture_id not in seen:
            seen.add(pair.capture_id)
            selected.append(captures[pair.capture_id])
    return tuple(selected)


def _opaque_id(content_sha256: str) -> str:
    return hashlib.sha256(
        b"liveD:rag-document:v1\0" + content_sha256.encode("ascii")
    ).hexdigest()[:32]


def _metadata_blob_id(version_id: str) -> str:
    if type(version_id) is not str or _HEX32.fullmatch(version_id) is None:
        raise RagValidationError("RAG metadata version identifier is invalid")
    return hashlib.sha256(
        b"liveD:rag-index-metadata:v1|" + version_id.encode("ascii")
    ).hexdigest()


def _dataset(snapshot: ReadinessSnapshot):
    selected = _rag_documents(snapshot)
    if not selected:
        raise RagValidationError("RAG requires at least one eligible document")
    if len(selected) > _MAX_DOCUMENTS:
        raise RagValidationError("RAG document count exceeds the local bound")
    ordered = tuple(
        sorted(selected, key=lambda item: (item.content_sha256, item.capture_id))
    )
    chunks: list[RagChunk] = []
    document_payloads: list[bytes] = []
    fixed_bytes = len(
        b'{"documents":[],"schema":"LIVED_RAG_DATASET_V1"}\n'
    )
    dataset_bytes = fixed_bytes
    for document in ordered:
        encoded = document.raw_text.encode("utf-8")
        if not encoded or len(encoded) > _MAX_DOCUMENT_BYTES:
            raise RagValidationError("RAG document size is invalid")
        document_chunks = chunk_text(document.raw_text)
        opaque = _opaque_id(document.content_sha256)
        records = []
        for ordinal, text in enumerate(document_chunks):
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk = RagChunk(opaque, ordinal, text, digest)
            chunks.append(chunk)
            records.append(
                {"ordinal": ordinal, "text": text, "text_sha256": digest}
            )
            if len(chunks) > _MAX_CHUNKS:
                raise RagValidationError(
                    "RAG chunk count exceeds the local bound"
                )
        document_payload = _canonical_json(
            {
                "chunks": records,
                "content_sha256": document.content_sha256,
                "opaque_document_id": opaque,
            }
        )[:-1]
        dataset_bytes += len(document_payload)
        if document_payloads:
            dataset_bytes += 1
        if dataset_bytes > _MAX_DATASET_BYTES:
            raise RagValidationError(
                "RAG dataset bytes exceed the local memory budget"
            )
        document_payloads.append(document_payload)
    payload = (
        b'{"documents":['
        + b",".join(document_payloads)
        + b'],"schema":"LIVED_RAG_DATASET_V1"}\n'
    )
    if len(payload) != dataset_bytes:
        raise RagValidationError("RAG dataset byte accounting failed")
    return ordered, tuple(chunks), payload

def _configuration(model_sha256: str, backend_sha256: str) -> tuple[bytes, str]:
    for value in (model_sha256, backend_sha256):
        if type(value) is not str or _SHA256.fullmatch(value) is None:
            raise EmbeddingRuntimeBlocked("local runtime identity is invalid")
    payload = _canonical_json({
        "chunk_overlap": _CHUNK_OVERLAP,
        "chunk_size": _CHUNK_SIZE,
        "dimension": _DIMENSION,
        "embedding_model_sha256": model_sha256,
        "index_backend_sha256": backend_sha256,
        "index_type": "IndexFlatIP",
        "normalization": "l2-unit",
        "schema": "LIVED_RAG_CONFIGURATION_V1",
        "serialization_version": 1,
    })
    return payload, hashlib.sha256(payload).hexdigest()


def _validate_vectors(
    vectors: Sequence[Sequence[float]], expected: int
) -> tuple[tuple[float, ...], ...]:
    if len(vectors) != expected:
        raise RagValidationError("embedding count does not match the input count")
    result = []
    for vector in vectors:
        if len(vector) != _DIMENSION:
            raise RagValidationError("embedding dimension must be exactly 384")
        values = tuple(float(value) for value in vector)
        if not all(math.isfinite(value) for value in values):
            raise RagValidationError("embedding contains a non-finite component")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise RagValidationError("embedding is not unit normalized")
        result.append(values)
    return tuple(result)


def _embed_bounded(embedder: EmbeddingBackend, texts: Sequence[str]):
    result = []
    for offset in range(0, len(texts), 64):
        batch = tuple(texts[offset : offset + 64])
        result.extend(_validate_vectors(embedder.embed(batch), len(batch)))
    return tuple(result)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"):
        flags |= int(getattr(os, name, 0))
    descriptor = os.open(path, flags, 0o600)
    try:
        position = 0
        while position < len(payload):
            written = os.write(descriptor, payload[position:])
            if written <= 0:
                raise OSError("RAG file write made no progress")
            position += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if os.name != "nt":
        os.chmod(path, 0o600, follow_symlinks=False)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel.MoveFileExW
        move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
        move.restype = ctypes.c_int
        if not move(os.fspath(source), os.fspath(destination), 0x00000008):
            error = ctypes.get_last_error()
            if error in (80, 183):
                raise FileExistsError(destination)
            raise ctypes.WinError(error)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == 17:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), destination)


class RagEngine:
    def __init__(
        self,
        paths: StudioPaths,
        vault: StudioVault,
        embedder: EmbeddingBackend,
        *,
        manifest_writer: RunManifestWriter,
        approval_verifier: Callable[[TrainingPlan, ApprovalGrant], None],
        audit_hardware: CpuExecutionInventory,
        index_backend: IndexBackend | None = None,
        clock: Callable[[], datetime] | None = None,
        local_text_screen: Callable[[str], str] | None = None,
    ) -> None:
        if type(paths) is not StudioPaths or type(vault) is not StudioVault:
            raise TypeError("paths and vault must use their exact production types")
        if not callable(approval_verifier):
            raise TypeError("approval_verifier must be callable")
        if type(audit_hardware) is not CpuExecutionInventory:
            raise TypeError("audit_hardware must be CpuExecutionInventory")
        if getattr(embedder, "dimension", None) != _DIMENSION:
            raise EmbeddingRuntimeBlocked("local embedding dimension must be 384")
        identity = getattr(embedder, "identity_sha256", None)
        if type(identity) is not str or _SHA256.fullmatch(identity) is None:
            raise EmbeddingRuntimeBlocked("local embedding identity is invalid")
        self.paths = paths
        self.vault = vault
        self.embedder = embedder
        self._roots: dict[
            str, tuple[Path, tuple[tuple[str, tuple[int, int, int]], ...], int]
        ] = {}
        for name, candidate in (
            ("staging", paths.rag_staging),
            ("indexes", paths.rag_indexes),
            ("cache", paths.rag_cache),
        ):
            absolute = _absolute_lexical(candidate)
            try:
                _require_private_directory(absolute)
                metadata = _checked_lstat(absolute)
                chain = _snapshot_chain(absolute, require_target=True)
            except (OSError, UnsafePathError) as error:
                raise RagValidationError(
                    f"private RAG {name} root is unavailable"
                ) from error
            self._roots[name] = (absolute, chain, int(metadata.st_dev))
        devices = {
            self._roots["staging"][2],
            self._roots["indexes"][2],
            self._roots["cache"][2],
        }
        if len(devices) != 1:
            raise RagValidationError(
                "RAG roots must use one filesystem for atomic publication"
            )
        self.manifest_writer = manifest_writer
        self.approval_verifier = approval_verifier
        self.audit_hardware = audit_hardware
        self.index_backend = NativeFaissBackend() if index_backend is None else index_backend
        backend_identity = getattr(self.index_backend, "identity_sha256", None)
        if type(backend_identity) is not str or _SHA256.fullmatch(backend_identity) is None:
            raise EmbeddingRuntimeBlocked("local index runtime identity is invalid")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.local_text_screen = local_text_screen or (lambda value: value)

    def _require_stable_root(self, name: str) -> Path:
        try:
            path, expected_chain, device = self._roots[name]
        except KeyError:
            raise RagValidationError("unknown private RAG root") from None
        try:
            _require_private_directory(path, expected_device=device)
            current = _snapshot_chain(path, require_target=True)
        except (OSError, UnsafePathError) as error:
            raise RagValidationError(
                f"private RAG {name} root failed verification"
            ) from error
        if current != expected_chain:
            raise RagValidationError(
                f"private RAG {name} root identity changed"
            )
        return path

    def _require_stable_roots(self) -> None:
        for name in ("staging", "indexes", "cache"):
            self._require_stable_root(name)

    def _read_private_bytes(
        self, path: Path, *, root_name: str, maximum: int
    ) -> bytes:
        root = self._require_stable_root(root_name)
        try:
            if Path(os.path.commonpath((root, path))) != root:
                raise RagValidationError("RAG file escaped its private root")
            before_chain = _snapshot_chain(path, require_target=True)
            before = _require_private_file(
                path, expected_device=self._roots[root_name][2]
            )
            if before.st_size > maximum:
                raise RagValidationError("RAG file exceeds its read bound")
            flags = os.O_RDONLY
            for flag_name in (
                "O_BINARY", "O_CLOEXEC", "O_NOINHERIT", "O_NOFOLLOW"
            ):
                flags |= int(getattr(os, flag_name, 0))
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if _file_identity(opened) != _file_identity(before):
                    raise RagValidationError(
                        "RAG file changed before descriptor verification"
                    )
                payload = _read_descriptor(descriptor, maximum)
                after_open = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            after = _require_private_file(
                path, expected_device=self._roots[root_name][2]
            )
            if (
                _file_identity(opened) != _file_identity(after_open)
                or _file_identity(before) != _file_identity(after)
                or before_chain
                != _snapshot_chain(path, require_target=True)
            ):
                raise RagValidationError("RAG file changed while it was read")
            self._require_stable_root(root_name)
            return payload
        except RagValidationError:
            raise
        except (OSError, UnsafePathError) as error:
            raise RagValidationError("RAG file cannot be read safely") from error

    def _write_private_exclusive(
        self, path: Path, payload: bytes, *, root_name: str
    ) -> None:
        root = self._require_stable_root(root_name)
        parent = path.parent
        try:
            if Path(os.path.commonpath((root, path))) != root:
                raise RagValidationError("RAG output escaped its private root")
            _require_private_directory(
                parent, expected_device=self._roots[root_name][2]
            )
            parent_chain = _snapshot_chain(parent, require_target=True)
            _write_exclusive(path, payload)
            if os.name == "nt":
                _set_private_windows_acl(path, directory=False)
            _require_private_file(
                path, expected_device=self._roots[root_name][2]
            )
            if (
                _snapshot_chain(parent, require_target=True) != parent_chain
                or self._require_stable_root(root_name) != root
            ):
                raise RagValidationError(
                    "RAG output parent changed while the file was written"
                )
        except RagValidationError:
            raise
        except (OSError, UnsafePathError) as error:
            raise RagValidationError(
                "RAG output could not be published safely"
            ) from error

    def _now(self) -> datetime:
        value = self.clock()
        if (
            type(value) is not datetime
            or value.tzinfo is None
            or value.utcoffset() != timedelta(0)
        ):
            raise RagValidationError("RAG clock must return a UTC-aware datetime")
        return value

    def _cache_inventory(self) -> tuple[tuple[str, int], ...]:
        root = self._require_stable_root("cache")
        try:
            entries = self._bounded_directory_entries(
                root, maximum=_MAX_CACHE_FILES
            )
            inventory = []
            aggregate = 0
            for entry in entries:
                if re.fullmatch(r"[a-f0-9]{64}[.]bin", entry.name) is None:
                    raise RagCacheError(
                        "RAG cache contains an unexpected object"
                    )
                metadata = _require_private_file(
                    Path(entry.path), expected_device=self._roots["cache"][2]
                )
                if not 1 <= metadata.st_size <= _MAX_CACHE_BYTES:
                    raise RagCacheError(
                        "RAG cache file size is outside its bound"
                    )
                aggregate += int(metadata.st_size)
                if aggregate > _MAX_CACHE_AGGREGATE_BYTES:
                    raise RagCacheError(
                        "RAG cache aggregate byte bound was exceeded"
                    )
                inventory.append((entry.name, int(metadata.st_size)))
            self._require_stable_root("cache")
            return tuple(sorted(inventory))
        except RagCacheError:
            raise
        except (OSError, UnsafePathError, RagValidationError) as error:
            raise RagCacheError("RAG cache root failed verification") from error

    def _cache_key(self, snapshot: ReadinessSnapshot, ids: tuple[str, ...]) -> str:
        return hashlib.sha256(_canonical_json({
            "input_identity_sha256": hashlib.sha256(_canonical_json(list(ids))).hexdigest(),
            "model_sha256": self.embedder.identity_sha256,
            "schema": "LIVED_RAG_ASSESSMENT_CACHE_KEY_V1",
            "snapshot_sequence": snapshot.sequence,
        })).hexdigest()

    def assess(self, snapshot: ReadinessSnapshot) -> EmbeddingAssessment:
        if type(snapshot) is not ReadinessSnapshot:
            raise TypeError("snapshot must be ReadinessSnapshot")
        documents = _rag_documents(snapshot)
        input_ids = tuple(document.capture_id for document in documents)
        sampled_ids = tuple(sorted(
            input_ids,
            key=lambda value: hashlib.sha256(value.encode("ascii")).digest(),
        )[:2_000])
        key = self._cache_key(snapshot, input_ids)
        cache_name = f"{key}.bin"
        inventory = self._cache_inventory()
        cache_path = self._require_stable_root("cache") / cache_name
        if any(name == cache_name for name, _size in inventory):
            try:
                sealed = self._read_private_bytes(
                    cache_path, root_name="cache", maximum=_MAX_CACHE_BYTES
                )
                clear = self.vault.open_local_blob("rag-assessment", key, sealed)
                document = _strict_json(clear, maximum=268_435_456)
                assessment = replace(
                    self._assessment_from_cache(document),
                    cache_sha256=hashlib.sha256(sealed).hexdigest(),
                )
            except (OSError, ValueError, VaultAuthenticationError, RagValidationError) as error:
                raise RagCacheError("authenticated RAG assessment cache is invalid") from error
            expected_input_digest = hashlib.sha256(_canonical_json(list(input_ids))).hexdigest()
            expected_sample_digest = hashlib.sha256(_canonical_json(list(sampled_ids))).hexdigest()
            if (
                assessment.snapshot_sequence != snapshot.sequence
                or assessment.input_record_ids != input_ids
                or assessment.sampled_record_ids != sampled_ids
                or assessment.input_identity_sha256 != expected_input_digest
                or assessment.sampled_identity_sha256 != expected_sample_digest
                or assessment.model_sha256 != self.embedder.identity_sha256
                or assessment.cache_sha256 != hashlib.sha256(sealed).hexdigest()
            ):
                raise RagCacheError("RAG assessment cache does not bind the current snapshot")
            return assessment

        by_id = {document.capture_id: document for document in documents}
        sample_texts = tuple(by_id[value].raw_text for value in sampled_ids)
        vectors = _embed_bounded(self.embedder, sample_texts)
        diversity = (
            None
            if not vectors
            else round(semantic_diversity(vectors), 12)
        )
        created = self._now()
        base = {
            "created_at": created.isoformat(),
            "diversity": diversity,
            "input_identity_sha256": hashlib.sha256(_canonical_json(list(input_ids))).hexdigest(),
            "input_record_ids": list(input_ids),
            "model_sha256": self.embedder.identity_sha256,
            "sampled_identity_sha256": hashlib.sha256(_canonical_json(list(sampled_ids))).hexdigest(),
            "sampled_record_ids": list(sampled_ids),
            "schema": "LIVED_RAG_ASSESSMENT_CACHE_V1",
            "snapshot_sequence": snapshot.sequence,
        }
        clear = _canonical_json(base)
        sealed = self.vault.seal_local_blob("rag-assessment", key, clear)
        inventory = self._cache_inventory()
        if (
            len(inventory) >= _MAX_CACHE_FILES
            or sum(size for _name, size in inventory) + len(sealed)
            > _MAX_CACHE_AGGREGATE_BYTES
        ):
            raise RagCacheError("RAG cache capacity is exhausted")
        self._write_private_exclusive(
            cache_path, sealed, root_name="cache"
        )
        _fsync_directory(self._require_stable_root("cache"))
        self._cache_inventory()
        return EmbeddingAssessment(
            snapshot_sequence=snapshot.sequence,
            input_record_ids=input_ids,
            sampled_record_ids=sampled_ids,
            input_identity_sha256=base["input_identity_sha256"],
            sampled_identity_sha256=base["sampled_identity_sha256"],
            model_sha256=self.embedder.identity_sha256,
            input_count=len(input_ids),
            sampled_records=len(sampled_ids),
            diversity=diversity,
            cache_sha256=hashlib.sha256(sealed).hexdigest(),
            created_at=created,
        )

    @staticmethod
    def _assessment_from_cache(document: object) -> EmbeddingAssessment:
        expected = {
            "created_at", "diversity", "input_identity_sha256",
            "input_record_ids", "model_sha256", "sampled_identity_sha256",
            "sampled_record_ids", "schema", "snapshot_sequence",
        }
        if type(document) is not dict or set(document) != expected:
            raise RagCacheError("RAG assessment cache schema is invalid")
        if document["schema"] != "LIVED_RAG_ASSESSMENT_CACHE_V1":
            raise RagCacheError("RAG assessment cache version is invalid")
        try:
            created = datetime.fromisoformat(document["created_at"])
            input_ids = tuple(document["input_record_ids"])
            sample_ids = tuple(document["sampled_record_ids"])
            return EmbeddingAssessment(
                snapshot_sequence=document["snapshot_sequence"],
                input_record_ids=input_ids,
                sampled_record_ids=sample_ids,
                input_identity_sha256=document["input_identity_sha256"],
                sampled_identity_sha256=document["sampled_identity_sha256"],
                model_sha256=document["model_sha256"],
                input_count=len(input_ids),
                sampled_records=len(sample_ids),
                diversity=document["diversity"],
                cache_sha256="0" * 64,
                created_at=created,
            )
        except (TypeError, ValueError) as error:
            raise RagCacheError("RAG assessment cache values are invalid") from error

    def prepare_plan(self, snapshot: ReadinessSnapshot) -> TrainingPlan:
        if type(snapshot) is not ReadinessSnapshot:
            raise TypeError("snapshot must be ReadinessSnapshot")
        documents, _chunks, dataset = _dataset(snapshot)
        _config, config_sha256 = _configuration(
            self.embedder.identity_sha256, self.index_backend.identity_sha256
        )
        now = self._now()
        return TrainingPlan(
            plan_id=uuid4().hex,
            mode=TrainingMode.RAG,
            snapshot_sequence=snapshot.sequence,
            dataset_count=len(documents),
            dataset_bytes=len(dataset),
            dataset_sha256=hashlib.sha256(dataset).hexdigest(),
            config_sha256=config_sha256,
            model_identity_sha256=self.embedder.identity_sha256,
            engine_identity="local-faiss",
            hardware_profile="local",
            projection_schema="LIVED_RAG_V1",
            created_at=now,
            expires_at=now + timedelta(minutes=5),
            disclosure_summary="Wholly local authenticated RAG index; zero remote disclosure.",
            dataset_manifest_sha256=None,
            train_sha256=None,
            eval_sha256=None,
            parent_kind="NONE",
            parent_identity_sha256=None,
        )

    def _revalidate_plan(self, plan: TrainingPlan):
        if type(plan) is not TrainingPlan or plan.mode is not TrainingMode.RAG:
            raise RagValidationError("RAG start requires a RAG TrainingPlan")
        if self._now() > plan.expires_at:
            raise RagValidationError("RAG training plan has expired")
        snapshot = self.vault.training_snapshot()
        documents, chunks, dataset = _dataset(snapshot)
        _config, config_sha256 = _configuration(
            self.embedder.identity_sha256, self.index_backend.identity_sha256
        )
        if (
            snapshot.sequence != plan.snapshot_sequence
            or len(documents) != plan.dataset_count
            or len(dataset) != plan.dataset_bytes
            or hashlib.sha256(dataset).hexdigest() != plan.dataset_sha256
            or config_sha256 != plan.config_sha256
            or plan.model_identity_sha256 != self.embedder.identity_sha256
            or plan.engine_identity != "local-faiss"
            or plan.hardware_profile != "local"
            or plan.projection_schema != "LIVED_RAG_V1"
        ):
            raise RagValidationError("RAG plan no longer matches the authenticated vault snapshot")
        return documents, chunks, dataset

    @staticmethod
    def _checkpoint(_name: str) -> None:
        return None

    def _emit(
        self,
        run_id: str,
        sequence: int,
        state: RunState,
        phase: str,
        completed: int,
        total: int,
        started_at: datetime,
        event_sink: Callable[[TrainingEvent], None],
    ) -> TrainingEvent:
        now = self._now()
        event = TrainingEvent(
            sequence=sequence,
            state=state,
            phase=phase,
            completed_units=completed,
            total_units=total,
            recorded_at=now,
            elapsed_seconds=max(0.0, (now - started_at).total_seconds()),
        )
        self.manifest_writer.append_event(run_id, event)
        event_sink(event)
        return event


    @staticmethod
    def _bounded_directory_entries(path: Path, maximum: int = 1_024):
        entries = []
        iterator = os.scandir(path)
        try:
            for index, entry in enumerate(iterator):
                if index >= maximum:
                    raise RagValidationError("RAG directory entry bound was exceeded")
                entries.append(entry)
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        return tuple(entries)

    def _verify_root_inventory(self) -> None:
        self._require_stable_roots()
        events = self.vault.rag_index_events()
        known_versions = {event.version.version_id for event in events}
        pending = self.vault.pending_rag_index_event()
        for root_name, allow_staging in (
            ("indexes", False),
            ("staging", True),
        ):
            root = self._require_stable_root(root_name)
            for entry in self._bounded_directory_entries(root):
                path = Path(entry.path)
                try:
                    metadata = _checked_lstat(path)
                    _require_private_directory(
                        path, expected_device=self._roots[root_name][2]
                    )
                except (OSError, UnsafePathError) as error:
                    raise RagValidationError(
                        "RAG root contains an unsafe object"
                    ) from error
                if (
                    entry.name not in known_versions
                    or not stat.S_ISDIR(metadata.st_mode)
                ):
                    raise RagValidationError(
                        "RAG root contains an unexpected filesystem object"
                    )
                if allow_staging and (
                    pending is None
                    or entry.name != pending.version.version_id
                ):
                    raise RagValidationError(
                        "RAG staging contains an unauthenticated object"
                    )
            self._require_stable_root(root_name)

    def _verified_manifest_state(
        self, version: RagIndexVersion
    ) -> tuple[tuple[object, ...], RunState]:
        try:
            chain = self.manifest_writer.verify_run_chain(
                version.version_id
            )
        except Exception as error:
            raise RagReconciliationBlocked(
                "RAG run manifest chain failed authentication"
            ) from error
        if len(chain) < 2:
            raise RagReconciliationBlocked(
                "RAG run manifest is missing its accepted event"
            )

        def document(record):
            try:
                payload = self.manifest_writer._read_checked_bytes(
                    record.path
                )
                if (
                    len(payload) != record.byte_length
                    or hashlib.sha256(payload).hexdigest()
                    != record.sha256
                ):
                    raise RagReconciliationBlocked(
                        "RAG run manifest bytes changed"
                    )
                return _strict_json(payload, maximum=131_072)
            except RagReconciliationBlocked:
                raise
            except Exception as error:
                raise RagReconciliationBlocked(
                    "RAG run manifest cannot be reopened safely"
                ) from error

        accepted = document(chain[1])
        if (
            type(accepted) is not dict
            or accepted.get("manifest_kind") != "RUN_EVENT"
            or accepted.get("state") != RunState.RUNNING.value
            or accepted.get("phase_sha256")
            != _training_phase_sha256("rag-accepted")
            or accepted.get("completed_units") != 0
            or accepted.get("total_units") != version.document_count
        ):
            raise RagReconciliationBlocked(
                "RAG run manifest lacks the exact accepted event"
            )
        latest = document(chain[-1])
        if latest.get("manifest_kind") == "RUN_STARTED":
            state = RunState.RUNNING
        elif latest.get("manifest_kind") == "RUN_EVENT":
            try:
                state = RunState(latest["state"])
            except (KeyError, TypeError, ValueError) as error:
                raise RagReconciliationBlocked(
                    "RAG run manifest state is invalid"
                ) from error
        else:
            raise RagReconciliationBlocked(
                "RAG run manifest kind is invalid"
            )
        return chain, state

    @staticmethod
    def _path_exists_no_follow(path: Path) -> bool:
        try:
            _checked_lstat(path)
        except FileNotFoundError:
            return False
        except (OSError, UnsafePathError) as error:
            raise RagReconciliationBlocked(
                "RAG reconciliation path is unsafe"
            ) from error
        return True

    def _append_reconciled_terminal(
        self, version: RagIndexVersion, chain: tuple[object, ...]
    ) -> None:
        event = TrainingEvent(
            sequence=len(chain),
            state=RunState.COMPLETED,
            phase="rag-reconciled",
            completed_units=version.document_count,
            total_units=version.document_count,
            recorded_at=self._now(),
            elapsed_seconds=0.0,
        )
        self.manifest_writer.append_event(version.version_id, event)

    def reconcile(self) -> RagIndexVersion:
        self._require_stable_roots()
        self._verify_root_inventory()
        pending = self.vault.pending_rag_index_event()
        if pending is None:
            selected = self.vault.selected_rag_index_version()
            if selected is None:
                raise RagReconciliationBlocked(
                    "there is no RAG publication to reconcile"
                )
            self._verify_version_files(selected)
            chain, state = self._verified_manifest_state(selected)
            if state is RunState.COMPLETED:
                return selected
            if state in {
                RunState.CANCELLED,
                RunState.FAILED,
            }:
                raise RagReconciliationBlocked(
                    "selected RAG index has a non-complete terminal manifest"
                )
            self._append_reconciled_terminal(selected, chain)
            verified = self.current_version()
            if verified != selected:
                raise RagReconciliationBlocked(
                    "RAG terminal reconciliation changed selection"
                )
            return verified

        version = pending.version
        staging = (
            self._require_stable_root("staging") / version.version_id
        )
        destination = (
            self._require_stable_root("indexes") / version.version_id
        )
        staging_exists = self._path_exists_no_follow(staging)
        destination_exists = self._path_exists_no_follow(destination)
        if staging_exists and destination_exists:
            raise RagReconciliationBlocked(
                "RAG reconciliation found both staging and published objects"
            )
        if staging_exists:
            self._verify_version_files(version, staging=True)
        chain, state = self._verified_manifest_state(version)
        if state in {
            RunState.CANCELLED,
            RunState.COMPLETED,
            RunState.FAILED,
        }:
            raise RagReconciliationBlocked(
                "pending RAG publication has a terminal manifest"
            )
        if staging_exists:
            staging_root = self._require_stable_root("staging")
            indexes_root = self._require_stable_root("indexes")
            self._require_stable_roots()
            _rename_noreplace(staging, destination)
            self._require_stable_roots()
            _fsync_directory(staging_root)
            _fsync_directory(indexes_root)
            destination_exists = True
        if not destination_exists:
            raise RagReconciliationBlocked(
                "authenticated RAG preparation has no recoverable filesystem bytes"
            )
        self._verify_version_files(version)
        self._verified_manifest_state(version)
        events = self.vault.rag_index_events()
        if not any(
            event.kind == "PUBLISHED"
            and event.prepared_event_id == pending.event_id
            for event in events
        ):
            self.vault.mark_rag_index_published(pending.event_id)
        chain, state = self._verified_manifest_state(version)
        if state in {
            RunState.CANCELLED,
            RunState.COMPLETED,
            RunState.FAILED,
        }:
            raise RagReconciliationBlocked(
                "RAG manifest changed before selection"
            )
        self.vault.select_rag_index_version(pending.event_id)
        self._append_reconciled_terminal(version, chain)
        selected = self.current_version()
        if selected != version:
            raise RagReconciliationBlocked(
                "RAG reconciliation did not select exact bytes"
            )
        return selected

    def start(
        self,
        plan: TrainingPlan,
        grant: ApprovalGrant,
        event_sink: Callable[[TrainingEvent], None] | None = None,
    ) -> TrainingRunSnapshot:
        if not callable(event_sink) and event_sink is not None:
            raise TypeError("event_sink must be callable")
        sink = event_sink or (lambda _event: None)
        self.approval_verifier(plan, grant)
        documents, chunks, dataset = self._revalidate_plan(plan)
        if self.vault.pending_rag_index_event() is not None:
            raise RagReconciliationBlocked("an interrupted RAG publication requires reconciliation")
        self._require_stable_roots()
        self._verify_root_inventory()
        run = None
        staging: Path | None = None
        staging_identity: os.stat_result | None = None
        staging_private = False
        prepared_event_id = None
        started = self._now()
        try:
            run = self.manifest_writer.start_run(
                plan, self.audit_hardware
            )
            self._emit(
                run.run_id, 1, RunState.RUNNING, "rag-accepted",
                0, len(documents), started, sink,
            )
            staging_root = self._require_stable_root("staging")
            staging = staging_root / run.run_id
            os.mkdir(staging, 0o700)
            staging_identity = os.lstat(staging)
            if os.name == "nt":
                _set_private_windows_acl(staging, directory=True)
            else:
                os.chmod(staging, 0o700, follow_symlinks=False)
            _require_private_directory(
                staging, expected_device=self._roots["staging"][2]
            )
            staging_private = True
            self._require_stable_root("staging")
            _fsync_directory(staging_root)
            first_vector: tuple[float, ...] | None = None
            embedded_count = 0

            def vector_batches():
                nonlocal first_vector, embedded_count
                for offset in range(0, len(chunks), 64):
                    texts = tuple(
                        chunk.text for chunk in chunks[offset : offset + 64]
                    )
                    vectors = _validate_vectors(
                        self.embedder.embed(texts), len(texts)
                    )
                    if first_vector is None and vectors:
                        first_vector = vectors[0]
                    embedded_count += len(vectors)
                    yield vectors

            index_payload = self.index_backend.build_batches(
                vector_batches(),
                dimension=_DIMENSION,
                expected_count=len(chunks),
            )
            if embedded_count != len(chunks):
                raise RagValidationError(
                    "local index backend did not consume every vector batch"
                )
            if type(index_payload) is not bytes or not 1 <= len(index_payload) <= _MAX_INDEX_BYTES:
                raise RagValidationError("serialized index bytes are invalid")
            loaded = self.index_backend.load(index_payload, dimension=_DIMENSION)
            if chunks:
                self_query = tuple(
                    self.index_backend.search(loaded, first_vector, top_k=1)
                )
                if (
                    not self_query
                    or self_query[0][1] != 0
                    or not math.isfinite(float(self_query[0][0]))
                    or float(self_query[0][0]) < 0.999
                ):
                    raise RagValidationError("serialized index failed its fixed self-query")
            corpus_sha256 = hashlib.sha256(
                _canonical_json([
                    {
                        "opaque_document_id": chunk.opaque_document_id,
                        "ordinal": chunk.ordinal,
                        "text": chunk.text,
                        "text_sha256": chunk.text_sha256,
                    }
                    for chunk in chunks
                ])
            ).hexdigest()
            metadata_clear = _canonical_json({
                "chunks": [
                    {
                        "opaque_document_id": chunk.opaque_document_id,
                        "ordinal": chunk.ordinal,
                        "text": chunk.text,
                        "text_sha256": chunk.text_sha256,
                    }
                    for chunk in chunks
                ],
                "config_sha256": plan.config_sha256,
                "corpus_sha256": corpus_sha256,
                "dataset_sha256": plan.dataset_sha256,
                "dimension": _DIMENSION,
                "embedding_model_sha256": self.embedder.identity_sha256,
                "schema": "LIVED_RAG_INDEX_METADATA_V1",
            })
            if len(metadata_clear) > _MAX_METADATA_CLEAR_BYTES:
                raise RagValidationError(
                    "RAG metadata exceeds its local cleartext bound"
                )
            metadata = self.vault.seal_local_blob(
                "rag-index-metadata",
                _metadata_blob_id(run.run_id),
                metadata_clear,
            )
            if not 1 <= len(metadata) <= _MAX_METADATA_BYTES:
                raise RagValidationError(
                    "sealed RAG metadata exceeds its local bound"
                )
            self._write_private_exclusive(
                staging / "index.faiss",
                index_payload,
                root_name="staging",
            )
            self._write_private_exclusive(
                staging / "metadata.json",
                metadata,
                root_name="staging",
            )
            inventory = _canonical_json({
                "files": [
                    {
                        "bytes": len(index_payload),
                        "path": "index.faiss",
                        "sha256": hashlib.sha256(index_payload).hexdigest(),
                    },
                    {
                        "bytes": len(metadata),
                        "path": "metadata.json",
                        "sha256": hashlib.sha256(metadata).hexdigest(),
                    },
                ],
                "schema": "LIVED_RAG_INDEX_INVENTORY_V1",
            })
            self._write_private_exclusive(
                staging / "INVENTORY.json",
                inventory,
                root_name="staging",
            )
            _fsync_directory(staging)
            self._require_stable_root("staging")
            version = RagIndexVersion(
                version_id=run.run_id,
                dataset_sha256=plan.dataset_sha256,
                corpus_sha256=corpus_sha256,
                config_sha256=plan.config_sha256,
                embedding_model_sha256=self.embedder.identity_sha256,
                index_sha256=hashlib.sha256(index_payload).hexdigest(),
                index_bytes=len(index_payload),
                metadata_sha256=hashlib.sha256(metadata).hexdigest(),
                metadata_bytes=len(metadata),
                inventory_sha256=hashlib.sha256(inventory).hexdigest(),
                dataset_bytes=len(dataset),
                document_count=len(documents),
                chunk_count=len(chunks),
            )
            prepared = self.vault.prepare_rag_index_version(version)
            prepared_event_id = prepared.event.event_id
            self._checkpoint("after_prepared")
            staging_root = self._require_stable_root("staging")
            indexes_root = self._require_stable_root("indexes")
            destination = indexes_root / version.version_id
            self._require_stable_roots()
            _rename_noreplace(staging, destination)
            self._require_stable_roots()
            _fsync_directory(staging_root)
            _fsync_directory(indexes_root)
            self._checkpoint("after_filesystem_publication")
            self._verify_version_files(version)
            self.vault.mark_rag_index_published(prepared_event_id)
            self._checkpoint("after_published")
            self.vault.select_rag_index_version(prepared_event_id)
            self._checkpoint("after_selected")
            self._emit(
                run.run_id, 2, RunState.RUNNING, "rag-embedded",
                len(documents), len(documents), started, sink,
            )
            self._emit(run.run_id, 3, RunState.RUNNING, "rag-published", len(documents), len(documents), started, sink)
            terminal = self._emit(run.run_id, 4, RunState.COMPLETED, "rag-complete", len(documents), len(documents), started, sink)
            return TrainingRunSnapshot(
                run_id=run.run_id,
                mode=TrainingMode.RAG,
                state=RunState.COMPLETED,
                sequence=terminal.sequence,
                phase=terminal.phase,
                completed_units=terminal.completed_units,
                total_units=terminal.total_units,
                epoch=None,
                loss=None,
                learning_rate=None,
                throughput=None,
                elapsed_seconds=terminal.elapsed_seconds,
                eta_seconds=None,
                gpu=None,
                last_checkpoint_sha256=None,
                updated_at=terminal.recorded_at,
                dataset_sha256=plan.dataset_sha256,
                config_sha256=plan.config_sha256,
                parent_kind=plan.parent_kind,
                parent_identity_sha256=plan.parent_identity_sha256,
            )
        except BaseException as failure:
            recovery_errors: list[BaseException] = []
            if prepared_event_id is None and run is not None:
                if staging is not None:
                    if staging_identity is None:
                        recovery_errors.append(
                            RagValidationError(
                                "owned RAG staging identity is unavailable"
                            )
                        )
                    else:
                        try:
                            self._cleanup_owned_staging(
                                staging,
                                staging_identity,
                                require_private=staging_private,
                            )
                        except BaseException as error:
                            recovery_errors.append(error)
                try:
                    sequence = len(
                        self.manifest_writer.verify_run_chain(run.run_id)
                    )
                    self._emit(
                        run.run_id,
                        sequence,
                        RunState.FAILED,
                        "rag-failed",
                        0,
                        len(documents),
                        started,
                        lambda _event: None,
                    )
                except BaseException as error:
                    recovery_errors.append(error)
            if recovery_errors:
                raise RagValidationError(
                    "RAG failure recovery did not complete"
                ) from failure
            raise

    def build(
        self,
        plan: TrainingPlan,
        grant: ApprovalGrant,
        event_sink: Callable[[TrainingEvent], None],
    ) -> TrainingRunSnapshot:
        return self.start(plan, grant, event_sink)

    def _cleanup_owned_staging(
        self,
        path: Path,
        identity: os.stat_result,
        *,
        require_private: bool = True,
    ) -> None:
        root = self._require_stable_root("staging")
        if Path(os.path.commonpath((root, path))) != root:
            raise RagValidationError("RAG staging escaped its private root")
        try:
            current = _checked_lstat(path)
        except FileNotFoundError:
            return
        if _entry_identity(current) != _entry_identity(identity):
            raise RagValidationError(
                "RAG staging identity changed during cleanup"
            )
        if require_private:
            _require_private_directory(
                path, expected_device=self._roots["staging"][2]
            )
            entries = self._bounded_directory_entries(path, maximum=3)
            allowed = {
                "INVENTORY.json", "index.faiss", "metadata.json"
            }
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                if (
                    entry.name not in allowed
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    raise RagValidationError(
                        "RAG staging contains an unsafe object"
                    )
            for entry in entries:
                os.unlink(entry.path)
        else:
            entries = self._bounded_directory_entries(path, maximum=1)
            if entries:
                raise RagValidationError(
                    "unvalidated RAG staging is not empty"
                )
        os.rmdir(path)
        _fsync_directory(root)
        self._require_stable_root("staging")

    def _require_active_runtime(
        self, version: RagIndexVersion
    ) -> None:
        _configuration_bytes, configuration_sha256 = _configuration(
            self.embedder.identity_sha256,
            self.index_backend.identity_sha256,
        )
        if (
            version.embedding_model_sha256
            != self.embedder.identity_sha256
            or version.config_sha256 != configuration_sha256
        ):
            raise RagValidationError(
                "selected RAG index differs from the active local runtime"
            )

    def _verify_version_files(
        self, version: RagIndexVersion, *, staging: bool = False
    ):
        self._require_active_runtime(version)
        root_name = "staging" if staging else "indexes"
        root = self._require_stable_root(root_name)
        directory = root / version.version_id
        try:
            _require_private_directory(
                directory, expected_device=self._roots[root_name][2]
            )
            directory_chain = _snapshot_chain(
                directory, require_target=True
            )
            entries = sorted(
                entry.name
                for entry in self._bounded_directory_entries(
                    directory, maximum=3
                )
            )
        except (OSError, UnsafePathError) as error:
            raise RagValidationError(
                "published RAG index directory is unsafe"
            ) from error
        if entries != ["INVENTORY.json", "index.faiss", "metadata.json"]:
            raise RagValidationError(
                "published RAG index inventory is incomplete"
            )
        inventory = self._read_private_bytes(
            directory / "INVENTORY.json",
            root_name=root_name,
            maximum=1_048_576,
        )
        index_payload = self._read_private_bytes(
            directory / "index.faiss",
            root_name=root_name,
            maximum=version.index_bytes,
        )
        metadata = self._read_private_bytes(
            directory / "metadata.json",
            root_name=root_name,
            maximum=version.metadata_bytes,
        )
        if (
            hashlib.sha256(inventory).hexdigest()
            != version.inventory_sha256
            or len(index_payload) != version.index_bytes
            or hashlib.sha256(index_payload).hexdigest()
            != version.index_sha256
            or len(metadata) != version.metadata_bytes
            or hashlib.sha256(metadata).hexdigest()
            != version.metadata_sha256
        ):
            raise RagValidationError(
                "published RAG index differs from its authenticated ledger"
            )
        inventory_document = _strict_json(
            inventory, maximum=1_048_576
        )
        expected_inventory = {
            "files": [
                {
                    "bytes": version.index_bytes,
                    "path": "index.faiss",
                    "sha256": version.index_sha256,
                },
                {
                    "bytes": version.metadata_bytes,
                    "path": "metadata.json",
                    "sha256": version.metadata_sha256,
                },
            ],
            "schema": "LIVED_RAG_INDEX_INVENTORY_V1",
        }
        if inventory_document != expected_inventory:
            raise RagValidationError(
                "published RAG inventory does not bind its exact files"
            )
        try:
            metadata_clear = self.vault.open_local_blob(
                "rag-index-metadata",
                _metadata_blob_id(version.version_id),
                metadata,
            )
        except (ValueError, VaultAuthenticationError) as error:
            raise RagValidationError(
                "published RAG metadata failed authentication"
            ) from error
        metadata_document = _strict_json(
            metadata_clear, maximum=_MAX_METADATA_CLEAR_BYTES
        )
        expected_keys = {
            "chunks", "config_sha256", "corpus_sha256",
            "dataset_sha256", "dimension", "embedding_model_sha256",
            "schema",
        }
        if (
            type(metadata_document) is not dict
            or set(metadata_document) != expected_keys
            or metadata_document["schema"]
            != "LIVED_RAG_INDEX_METADATA_V1"
            or metadata_document["dataset_sha256"]
            != version.dataset_sha256
            or metadata_document["corpus_sha256"]
            != version.corpus_sha256
            or metadata_document["config_sha256"]
            != version.config_sha256
            or metadata_document["embedding_model_sha256"]
            != version.embedding_model_sha256
            or type(metadata_document["dimension"]) is not int
            or metadata_document["dimension"] != _DIMENSION
            or type(metadata_document["chunks"]) is not list
            or len(metadata_document["chunks"]) != version.chunk_count
        ):
            raise RagValidationError(
                "published RAG metadata does not bind its ledger version"
            )
        expected_ordinals: dict[str, int] = {}
        canonical_chunks = []
        for record in metadata_document["chunks"]:
            if (
                type(record) is not dict
                or set(record)
                != {
                    "opaque_document_id", "ordinal", "text",
                    "text_sha256",
                }
            ):
                raise RagValidationError(
                    "published RAG chunk metadata schema is invalid"
                )
            opaque = record["opaque_document_id"]
            ordinal = record["ordinal"]
            text = record["text"]
            digest = record["text_sha256"]
            if (
                type(opaque) is not str
                or _HEX32.fullmatch(opaque) is None
                or type(ordinal) is not int
                or ordinal != expected_ordinals.get(opaque, 0)
                or type(text) is not str
                or not text
                or len(text) > _CHUNK_SIZE
                or _unsafe_control(text)
                or type(digest) is not str
                or _SHA256.fullmatch(digest) is None
                or hashlib.sha256(text.encode("utf-8")).hexdigest()
                != digest
            ):
                raise RagValidationError(
                    "published RAG chunk metadata is invalid"
                )
            expected_ordinals[opaque] = ordinal + 1
            canonical_chunks.append(
                {
                    "opaque_document_id": opaque,
                    "ordinal": ordinal,
                    "text": text,
                    "text_sha256": digest,
                }
            )
        if (
            len(expected_ordinals) != version.document_count
            or hashlib.sha256(
                _canonical_json(canonical_chunks)
            ).hexdigest()
            != version.corpus_sha256
        ):
            raise RagValidationError(
                "published RAG corpus identity is invalid"
            )
        try:
            loaded = self.index_backend.load(
                index_payload, dimension=_DIMENSION
            )
        except RagValidationError:
            raise
        except Exception as error:
            raise RagValidationError(
                "published RAG index deserialization failed"
            ) from error
        ntotal = getattr(loaded, "ntotal", None)
        dimension = getattr(loaded, "d", None)
        if (
            type(ntotal) is not int
            or ntotal != version.chunk_count
            or type(dimension) is not int
            or dimension != _DIMENSION
        ):
            raise RagValidationError(
                "published RAG vector shape or count is invalid"
            )
        if (
            _snapshot_chain(directory, require_target=True)
            != directory_chain
        ):
            raise RagValidationError(
                "published RAG directory changed during verification"
            )
        self._require_stable_root(root_name)
        return index_payload, metadata_document, loaded

    def versions(self) -> tuple[RagIndexVersion, ...]:
        """Return every authenticated, published local RAG dataset version."""

        published: list[RagIndexVersion] = []
        seen: set[str] = set()
        for event in self.vault.rag_index_events():
            if event.kind != "PUBLISHED":
                continue
            version = event.version
            if version.version_id in seen:
                raise RagValidationError("published RAG version identity was reused")
            self._verify_version_files(version)
            seen.add(version.version_id)
            published.append(version)
        return tuple(published)

    def current_version(self) -> RagIndexVersion | None:
        version = self.vault.selected_rag_index_version()
        if version is None:
            return None
        self._verify_version_files(version)
        return version

    selected_index = current_version

    def _read_selected_index_bytes_for_test(self) -> bytes:
        version = self.current_version()
        if version is None:
            raise RagValidationError("no RAG index is selected")
        payload, _metadata, _index = self._verify_version_files(version)
        return payload

    def search(self, query: str, top_k: int) -> tuple[RagSearchResult, ...]:
        if type(query) is not str or not query or query != query.strip():
            raise ValueError("query must be a nonempty trimmed string")
        if len(query.encode("utf-8")) > _MAX_QUERY_BYTES:
            raise ValueError("query exceeds the local search bound")
        if _unsafe_control(query) or _SECRET.search(query):
            raise ValueError("query contains unsafe control or secret material")
        if type(top_k) is not int or not 1 <= top_k <= _MAX_RESULTS:
            raise ValueError("top_k is outside the local search bound")
        version = self.vault.selected_rag_index_version()
        if version is None:
            raise RagValidationError(
                "no authenticated RAG index is selected"
            )
        _payload, metadata, index = self._verify_version_files(version)
        vector = _embed_bounded(self.embedder, (query,))[0]
        raw = tuple(self.index_backend.search(index, vector, top_k=top_k))
        chunks = metadata["chunks"]
        results = []
        for score, ordinal in sorted(raw, key=lambda item: (-float(item[0]), int(item[1]))):
            if (
                type(ordinal) is not int
                or not 0 <= ordinal < len(chunks)
                or type(score) not in (int, float)
                or not math.isfinite(float(score))
            ):
                raise RagValidationError("local index returned an invalid result")
            record = chunks[ordinal]
            if type(record) is not dict or set(record) != {
                "opaque_document_id", "ordinal", "text", "text_sha256"
            }:
                raise RagValidationError("local RAG metadata record is invalid")
            text = self.local_text_screen(record["text"])
            if type(text) is not str or not text or _unsafe_control(text):
                raise RagValidationError("local result screening returned invalid text")
            results.append(RagSearchResult(
                opaque_document_id=record["opaque_document_id"],
                chunk_ordinal=record["ordinal"],
                score=float(score),
                text=text,
            ))
        return tuple(results)


__all__ = [
    "EmbeddingBackend",
    "EmbeddingModelInventory",
    "EmbeddingRuntimeBlocked",
    "IndexBackend",
    "NativeFaissBackend",
    "PinnedLocalEmbedder",
    "RagCacheError",
    "RagChunk",
    "RagEngine",
    "RagError",
    "RagReconciliationBlocked",
    "RagSearchResult",
    "RagValidationError",
    "chunk_text",
    "verify_embedding_model_inventory",
]
