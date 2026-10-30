"""List local or SSH-reachable NVIDIA GPUs without launching training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Callable, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from studio.gpu_selection import (  # noqa: E402
    GpuInventory, GpuTarget, discover_gpus, load_selection, save_selection,
    select_gpu, selection_document,
)


def _inventory_document(inventory: GpuInventory) -> dict[str, object]:
    return {
        "schema": "LIVED_GPU_INVENTORY_V1",
        "target": {"kind": inventory.target.kind, "ssh_alias": inventory.target.ssh_alias},
        "observed_at_utc": inventory.observed_at.isoformat().replace("+00:00", "Z"),
        "devices": [asdict(device) for device in inventory.devices],
        "training_qualified": False,
        "qualification_note": "Device discovery does not verify Torch, CUDA, the model, or a training recipe.",
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    discoverer: Callable[[GpuTarget], GpuInventory] = discover_gpus,
    clock: Callable[[], datetime] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    list_parser = commands.add_parser("list", help="list observed NVIDIA GPUs")
    list_parser.add_argument("--remote", metavar="SSH_ALIAS", help="strict OpenSSH host alias")
    choose_parser = commands.add_parser("choose", help="choose an exact observed GPU UUID")
    choose_parser.add_argument("--remote", metavar="SSH_ALIAS", help="strict OpenSSH host alias")
    choose_parser.add_argument("--gpu", required=True, metavar="GPU_UUID")
    choose_parser.add_argument("--output", type=Path, metavar="NEW_JSON_FILE")
    refresh_parser = commands.add_parser("refresh", help="re-probe a saved GPU choice")
    refresh_parser.add_argument("--selection", type=Path, required=True, metavar="JSON_FILE")
    arguments = parser.parse_args(argv)
    try:
        prior = load_selection(arguments.selection) if arguments.command == "refresh" else None
        target = prior.target if prior is not None else (
            GpuTarget("ssh", arguments.remote) if arguments.remote else GpuTarget("local")
        )
        inventory = discoverer(target)
        if inventory.target != target:
            raise ValueError("GPU probe target differs from requested target")
        if arguments.command == "list":
            document = _inventory_document(inventory)
        elif arguments.command == "choose":
            chosen = select_gpu(
                inventory, arguments.gpu, now=None if clock is None else clock()
            )
            if arguments.output is not None:
                save_selection(chosen, arguments.output)
            document = selection_document(chosen)
        else:
            assert prior is not None
            chosen = select_gpu(
                inventory, prior.gpu_uuid, now=None if clock is None else clock()
            )
            if chosen.binding_sha256 != prior.binding_sha256:
                raise ValueError("saved GPU target binding differs from fresh observation")
            document = selection_document(chosen)
        print(json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True), file=out)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=err)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
