from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import studio.paths as paths_module
from studio.paths import (
    PathRaceError,
    PathReadLimitError,
    StudioPaths,
    UnsafePathError,
)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def make_private_directory(path: Path) -> None:
    if os.name == "nt":
        paths_module._set_private_windows_acl(path, directory=True)
    else:
        os.chmod(path, 0o700, follow_symlinks=False)


def make_private_file(path: Path) -> None:
    if os.name == "nt":
        paths_module._set_private_windows_acl(path, directory=False)
    else:
        os.chmod(path, 0o600, follow_symlinks=False)


def create_model_tree(release: Path, payload: bytes = b"installed model") -> Path:
    model = release / "resources" / "embedding-model" / "all-MiniLM-L6-v2"
    directories = (
        release,
        release / "resources",
        release / "resources" / "embedding-model",
        model,
    )
    if os.name == "nt":
        # Elevated Windows runners can assign Administrators as the default
        # owner. Build the private-at-creation fixture required by this reader.
        for directory in directories:
            paths_module._create_private_windows_directory(directory)
    else:
        model.mkdir(parents=True)
        for directory in directories:
            make_private_directory(directory)
    target = model / "vocab.txt"
    if os.name == "nt":
        with os.fdopen(paths_module._create_private_windows_file(target), "wb") as stream:
            stream.write(payload)
    else:
        target.write_bytes(payload)
        make_private_file(target)
    return model


@contextmanager
def directory_reparse(link: Path, target: Path):
    if os.name == "nt":
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise unittest.SkipTest(f"junction creation unavailable: {result.stderr}")
    else:
        link.symlink_to(target, target_is_directory=True)
    try:
        yield
    finally:
        if link.exists() or link.is_symlink():
            if os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()


class InstalledEmbeddingPathTests(unittest.TestCase):
    def test_reset_state_and_receipts_are_contained_private_directories(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            for name in ("source_reset_state", "source_reset_receipts"):
                self.assertTrue(hasattr(paths, name), f"missing {name}")
                directory = getattr(paths, name)
                self.assertEqual(directory, Path("workspace/studio_data") / name)
                paths_module._require_private_directory(directory)
                self.assertEqual(paths.secure_directory(f"studio_data/{name}", create=False), directory.resolve())

    def test_source_rights_has_a_contained_private_directory(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            paths = StudioPaths(Path("workspace")).ensure()
            self.assertTrue(hasattr(paths, "source_rights"), "source rights path is missing")
            self.assertEqual(paths.source_rights, Path("workspace/studio_data/source_rights"))
            self.assertTrue(paths.source_rights.is_dir())
            paths_module._require_private_directory(paths.source_rights)
            self.assertEqual(
                paths.secure_directory("studio_data/source_rights", create=False),
                paths.source_rights.resolve(),
            )

    def test_installed_model_root_must_be_absolute_and_have_the_fixed_layout(self):
        with self.assertRaises(UnsafePathError):
            StudioPaths(
                Path("workspace"),
                installed_embedding_model_root=Path(
                    "resources/embedding-model/all-MiniLM-L6-v2"
                ),
            )

        with tempfile.TemporaryDirectory() as temporary:
            arbitrary = Path(temporary).resolve() / "all-MiniLM-L6-v2"
            arbitrary.mkdir()
            make_private_directory(arbitrary)
            with self.assertRaises(UnsafePathError):
                StudioPaths(
                    Path("workspace"), installed_embedding_model_root=arbitrary
                )

    def test_installed_model_is_independent_of_mutable_workspace_fallback(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            model = create_model_tree(Path("release").resolve(), b"trusted installed")
            fallback = (
                Path("workspace/resources/embedding-model/all-MiniLM-L6-v2")
            )
            paths_module._make_checked_directory(Path("workspace"))
            fallback.mkdir(parents=True)
            (fallback / "vocab.txt").write_bytes(b"mutable workspace")

            paths = StudioPaths(
                Path("workspace"), installed_embedding_model_root=model
            ).ensure()

            self.assertEqual(
                paths.secure_embedding_read_bytes("vocab.txt", maximum=64),
                b"trusted installed",
            )
            self.assertEqual(paths.secure_embedding_directory(), model)

            (model / "vocab.txt").unlink()
            with self.assertRaises((FileNotFoundError, UnsafePathError)):
                paths.secure_embedding_read_bytes("vocab.txt", maximum=64)

    @unittest.skipUnless(os.name == "posix", "Linux release ancestry permissions")
    def test_installed_model_rejects_a_writable_release_layout_ancestor(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            release = Path("release").resolve()
            model = create_model_tree(release)
            os.chmod(release / "resources", 0o770, follow_symlinks=False)

            with self.assertRaisesRegex(UnsafePathError, "ancestor|private"):
                StudioPaths(
                    Path("workspace"), installed_embedding_model_root=model
                )

    def test_development_mode_keeps_the_workspace_relative_model_behavior(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            model = create_model_tree(
                Path("workspace").resolve(), b"workspace development model"
            )
            paths = StudioPaths(Path("workspace")).ensure()

            self.assertEqual(
                paths.secure_embedding_read_bytes("vocab.txt", maximum=64),
                b"workspace development model",
            )
            self.assertEqual(paths.secure_embedding_directory(), model)

    def test_embedding_reads_reject_operator_absolute_and_traversal_values(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            model = create_model_tree(Path("release").resolve())
            paths = StudioPaths(
                Path("workspace"), installed_embedding_model_root=model
            ).ensure()

            for value in ("../outside.txt", "nested/../../outside.txt", model / "vocab.txt"):
                with self.subTest(value=value):
                    with self.assertRaises(UnsafePathError):
                        paths.secure_embedding_read_bytes(value, maximum=64)

    def test_embedding_reads_reject_hard_link_files(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            model = create_model_tree(Path("release").resolve())
            outside = Path("outside.txt")
            outside.write_bytes(b"outside")
            alias = model / "alias.txt"
            try:
                os.link(outside, alias)
            except OSError as error:
                raise unittest.SkipTest(f"hard-link creation unavailable: {error}")
            make_private_file(alias)
            paths = StudioPaths(
                Path("workspace"), installed_embedding_model_root=model
            ).ensure()

            with self.assertRaisesRegex(UnsafePathError, "hard link"):
                paths.secure_embedding_read_bytes("alias.txt", maximum=64)

    def test_embedding_reads_reject_symbolic_link_files(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            model = create_model_tree(Path("release").resolve())
            outside = Path("outside.txt")
            outside.write_bytes(b"outside")
            alias = model / "alias.txt"
            try:
                alias.symlink_to(outside)
            except OSError as error:
                raise unittest.SkipTest(f"symbolic-link creation unavailable: {error}")
            paths = StudioPaths(
                Path("workspace"), installed_embedding_model_root=model
            ).ensure()

            with self.assertRaisesRegex(UnsafePathError, "reparse|symbolic"):
                paths.secure_embedding_read_bytes("alias.txt", maximum=64)

    def test_embedding_read_limit_cannot_be_bypassed_by_a_growing_file(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            model = create_model_tree(Path("release").resolve(), b"1234")
            paths = StudioPaths(
                Path("workspace"), installed_embedding_model_root=model
            ).ensure()
            with self.assertRaises(PathReadLimitError):
                paths.secure_embedding_read_bytes("vocab.txt", maximum=3)

            target = model / "vocab.txt"
            original_read = paths_module._read_descriptor

            def grow_after_open(descriptor: int, maximum: int) -> bytes:
                target.write_bytes(b"12345")
                return original_read(descriptor, maximum)

            with patch("studio.paths._read_descriptor", side_effect=grow_after_open):
                with self.assertRaises((PathRaceError, PathReadLimitError)):
                    paths.secure_embedding_read_bytes("vocab.txt", maximum=4)

    def test_constructor_rejects_a_reparse_point_in_installed_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            outside_release = Path("outside-release").resolve()
            model = create_model_tree(outside_release)
            with directory_reparse(Path("release-link"), outside_release):
                linked_model = (
                    Path("release-link/resources/embedding-model/all-MiniLM-L6-v2")
                    .resolve(strict=False)
                )
                # Preserve the lexical link in the supplied path; resolving it would erase
                # the object the production validation must reject.
                linked_model = Path(temporary).resolve() / "release-link" / model.relative_to(outside_release)
                with self.assertRaisesRegex(UnsafePathError, "reparse|symbolic"):
                    StudioPaths(
                        Path("workspace"),
                        installed_embedding_model_root=linked_model,
                    )

    def test_installed_ancestor_replacement_is_rejected_on_every_use(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            release = Path("release").resolve()
            model = create_model_tree(release, b"original")
            replacement = Path("replacement").resolve()
            create_model_tree(replacement, b"replacement")
            paths = StudioPaths(
                Path("workspace"), installed_embedding_model_root=model
            ).ensure()

            os.replace(release, Path("old-release").resolve())
            os.replace(replacement, release)

            with self.assertRaises(PathRaceError):
                paths.secure_embedding_directory()
            with self.assertRaises(PathRaceError):
                paths.secure_embedding_read_bytes("vocab.txt", maximum=64)

    def test_embedding_read_rejects_a_file_identity_change_during_read(self):
        with tempfile.TemporaryDirectory() as temporary, working_directory(Path(temporary)):
            model = create_model_tree(Path("release").resolve(), b"before")
            paths = StudioPaths(
                Path("workspace"), installed_embedding_model_root=model
            ).ensure()
            target = model / "vocab.txt"
            original_read = paths_module._read_descriptor

            def mutate_after_open(descriptor: int, maximum: int) -> bytes:
                target.write_bytes(b"after payload with a different size")
                return original_read(descriptor, maximum)

            with patch("studio.paths._read_descriptor", side_effect=mutate_after_open):
                with self.assertRaises(PathRaceError):
                    paths.secure_embedding_read_bytes("vocab.txt", maximum=64)


@unittest.skipUnless(os.name == "nt", "Windows owner handling")
class WindowsPrivateOwnerTests(unittest.TestCase):
    def test_new_directory_has_private_owner_and_acl_at_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            new = Path(temporary) / "new"
            with patch.object(paths_module, "_set_private_windows_acl") as set_acl:
                paths_module._make_checked_directory(new)
            set_acl.assert_not_called()
            paths_module._require_private_windows_acl(new)

    def test_existing_nonprivate_directory_is_rejected_without_acl_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            original_acl = subprocess.run(
                ["icacls", os.fspath(existing)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            with self.assertRaises(UnsafePathError):
                paths_module._make_checked_directory(existing)
            final_acl = subprocess.run(
                ["icacls", os.fspath(existing)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(final_acl, original_acl)

    def test_existing_directory_with_foreign_owner_is_not_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "existing"
            existing.mkdir()
            original_acl = subprocess.run(
                ["icacls", os.fspath(existing)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            with patch.object(
                paths_module, "_windows_current_user_sid", return_value="S-1-0-0"
            ):
                with self.assertRaisesRegex(UnsafePathError, "owner"):
                    paths_module._make_checked_directory(existing)
            final_acl = subprocess.run(
                ["icacls", os.fspath(existing)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertEqual(final_acl, original_acl)

    def test_replacement_after_private_creation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            moved = root / "moved"
            replacement = root / "replacement"
            replacement.mkdir()

            def create_then_replace(path: Path):
                path.mkdir()
                identity = paths_module._entry_identity(paths_module._checked_lstat(path))
                path.rename(moved)
                replacement.rename(path)
                return identity

            with (
                patch.object(
                    paths_module,
                    "_create_private_windows_directory",
                    side_effect=create_then_replace,
                    create=True,
                ),
                patch.object(paths_module, "_set_private_windows_acl"),
                patch.object(paths_module, "_require_private_directory"),
            ):
                with self.assertRaisesRegex(PathRaceError, "created directory changed"):
                    paths_module._make_checked_directory(target)

    def test_directory_appearing_during_creation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"

            def competing_creation(path: Path):
                path.mkdir()
                raise FileExistsError("directory already exists")

            with patch.object(
                paths_module,
                "_create_private_windows_directory",
                side_effect=competing_creation,
            ):
                with self.assertRaisesRegex(PathRaceError, "appeared during creation"):
                    paths_module._make_checked_directory(target)


if __name__ == "__main__":
    unittest.main()
