"""Set up and run the synthetic local review/export preview in one command."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / ".venv-preview"
MARKER = ENVIRONMENT / ".lived-preview-env"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def main() -> int:
    if sys.version_info < (3, 12):
        raise SystemExit("The local preview requires Python 3.12 or newer.")
    if ENVIRONMENT.exists():
        if not MARKER.is_file():
            raise SystemExit("The preview environment already exists without its ownership marker; refusing to change it.")
    else:
        ENVIRONMENT.mkdir()
        MARKER.write_text("liveD local preview environment\n", encoding="utf-8")
    if not PYTHON.is_file():
        venv.create(ENVIRONMENT, with_pip=True)
    print("Preparing the local demo environment from the pinned PyPI cryptography wheel.", flush=True)
    subprocess.run(
        [
            str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
            "cryptography==50.0.1",
        ],
        cwd=ROOT,
        check=True,
    )
    print("Running the synthetic review and encrypted-export demo.", flush=True)
    subprocess.run(
        [str(PYTHON), "-B", "-I", str(ROOT / "scripts/demo_review_export.py")],
        cwd=ROOT,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
