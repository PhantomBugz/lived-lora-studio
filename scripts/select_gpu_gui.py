"""Inspect and choose a local or SSH-hosted NVIDIA GPU for the preview.

This standalone picker records an observed hardware choice only. It does not
configure the training runtime, qualify a recipe, or start training.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import sys
import tkinter as tk
from tkinter import filedialog, ttk


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studio.gpu_selection import (  # noqa: E402
    GpuDevice,
    GpuInventory,
    GpuProbeError,
    GpuSelection,
    GpuSelectionError,
    GpuTarget,
    discover_gpus,
    select_gpu,
)

SYNTHETIC_BANNER = "SYNTHETIC DEMO — NO HARDWARE PROBE"


def synthetic_demo_inventory() -> GpuInventory:
    """Supply a labeled sample without querying this machine or an SSH host."""
    return GpuInventory(
        target=GpuTarget(kind="local"),
        observed_at=datetime.now(timezone.utc),
        devices=(
            GpuDevice(
                "GPU-00000000-0000-0000-0000-000000000001",
                "Synthetic NVIDIA sample — 24 GiB",
                24576,
                20480,
                "8.6",
                "synthetic-driver",
                "physical",
            ),
            GpuDevice(
                "GPU-00000000-0000-0000-0000-000000000002",
                "Synthetic NVIDIA sample — 48 GiB",
                49152,
                45056,
                "9.0",
                "synthetic-driver",
                "physical",
            ),
        ),
        training_qualified=False,
    )


def _target_label(target: GpuTarget) -> str:
    return "this computer" if target.kind == "local" else f"SSH host {target.ssh_alias}"


def format_inventory_rows(inventory: GpuInventory) -> tuple[tuple[str, ...], ...]:
    """Preserve each native UUID and show measured memory without qualification claims."""
    if type(inventory) is not GpuInventory:
        raise TypeError("inventory must be GpuInventory")

    def memory(mib: int | None) -> str:
        return "unknown" if mib is None else f"{mib / 1024:.1f} GiB"

    return tuple(
        (
            device.uuid,
            device.name,
            memory(device.total_mib),
            memory(device.free_mib),
            device.compute_capability or "unknown",
            device.device_kind,
        )
        for device in inventory.devices
    )


def selection_message(selection: GpuSelection, *, synthetic_demo: bool = False) -> str:
    if type(selection) is not GpuSelection:
        raise TypeError("selection must be GpuSelection")
    if synthetic_demo:
        return (
            f"{SYNTHETIC_BANNER}. Selected sample {selection.gpu_uuid}. "
            "This is not a real device, cannot be saved, and does not start training."
        )
    return (
        f"Selected {selection.gpu_uuid} on {_target_label(selection.target)}. "
        "This is an observed hardware choice; it does not start training or "
        "prove that a training runtime is ready."
    )


class GpuPickerModel:
    """Keep asynchronous observations and choices tied to one target generation."""

    def __init__(self, target: GpuTarget, *, synthetic_demo: bool = False) -> None:
        if type(target) is not GpuTarget:
            raise TypeError("target must be GpuTarget")
        if type(synthetic_demo) is not bool:
            raise TypeError("synthetic_demo must be bool")
        self.target = target
        self.synthetic_demo = synthetic_demo
        self.inventory: GpuInventory | None = None
        self.selection: GpuSelection | None = None
        self.generation = 0

    @property
    def can_save(self) -> bool:
        return self.selection is not None and not self.synthetic_demo

    def invalidate(self) -> None:
        self.generation += 1
        self.inventory = None
        self.selection = None

    def set_target(self, target: GpuTarget) -> None:
        if type(target) is not GpuTarget:
            raise TypeError("target must be GpuTarget")
        if target != self.target:
            self.target = target
            self.invalidate()

    def begin_refresh(self) -> tuple[int, GpuTarget]:
        self.invalidate()
        return self.generation, self.target

    def accept_inventory(self, generation: int, inventory: GpuInventory) -> bool:
        if type(inventory) is not GpuInventory:
            raise TypeError("inventory must be GpuInventory")
        if generation != self.generation or inventory.target != self.target:
            return False
        self.inventory = inventory
        self.selection = None
        return True

    def choose_gpu(self, gpu_uuid: str) -> GpuSelection:
        if self.inventory is None:
            raise GpuSelectionError("Refresh the selected target before choosing a GPU")
        choice = select_gpu(self.inventory, gpu_uuid)
        self.selection = choice
        return choice

    def save_choice(self, path: Path | str) -> Path:
        if not self.can_save or self.selection is None or self.inventory is None:
            raise GpuSelectionError("Choose a real observed GPU before saving")
        fresh = select_gpu(self.inventory, self.selection.gpu_uuid)
        if fresh != self.selection:
            raise GpuSelectionError("GPU observation changed; refresh and choose it again")
        from studio.gpu_selection import save_selection

        return save_selection(fresh, path)


class GpuPickerApp:
    def __init__(self, root: tk.Tk, *, synthetic_demo: bool = False,
                 probe=discover_gpus) -> None:
        self.root = root
        self.root.title("liveD LoRA Studio | GPU picker preview")
        self.root.geometry("1140x600")
        self.synthetic_demo = synthetic_demo
        self.model = GpuPickerModel(GpuTarget(kind="local"), synthetic_demo=synthetic_demo)
        self._probe = probe
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lived-gpu-picker")
        self._active: tuple[int, GpuTarget, Future[GpuInventory]] | None = None
        self._after_id: str | None = None
        self._closed = False
        self._row_uuids: dict[str, str] = {}

        body = ttk.Frame(root, padding=16)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Choose a GPU to inspect", font=("Segoe UI", 17, "bold")).pack(anchor="w")
        if synthetic_demo:
            tk.Label(
                body, text=SYNTHETIC_BANNER,
                fg="#a00000", font=("Segoe UI", 13, "bold"), anchor="w",
            ).pack(anchor="w", fill="x", pady=(4, 4))
        ttk.Label(
            body,
            text="Hardware observation only. Selection does not configure or start training.",
        ).pack(anchor="w", pady=(2, 14))

        target_bar = ttk.Frame(body)
        target_bar.pack(fill="x")
        self.kind = tk.StringVar(value="local")
        self.local_radio = ttk.Radiobutton(
            target_bar, text="This computer", value="local", variable=self.kind,
            command=self._input_changed,
        )
        self.local_radio.pack(side="left")
        self.remote_radio = ttk.Radiobutton(
            target_bar, text="Remote SSH host", value="ssh", variable=self.kind,
            command=self._input_changed,
        )
        self.remote_radio.pack(side="left", padx=(18, 8))
        ttk.Label(target_bar, text="SSH alias:").pack(side="left")
        self.alias = tk.StringVar()
        self.alias_entry = ttk.Entry(target_bar, textvariable=self.alias, width=30)
        self.alias_entry.pack(side="left", padx=(5, 10))
        self.alias.trace_add("write", lambda *_: self._input_changed())
        self.refresh_button = ttk.Button(
            target_bar, text="Reload sample" if synthetic_demo else "Refresh GPUs",
            command=self.refresh,
        )
        self.refresh_button.pack(side="left")

        columns = ("uuid", "name", "total", "free", "compute", "kind")
        table_frame = ttk.Frame(body)
        table_frame.pack(fill="both", expand=True, pady=(14, 8))
        self.table = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="browse", height=11,
        )
        for key, label, width in (
            ("uuid", "GPU UUID", 360), ("name", "Device", 340),
            ("total", "VRAM total", 105), ("free", "VRAM free", 105),
            ("compute", "Compute", 95), ("kind", "Kind", 80),
        ):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, minwidth=65, stretch=False)
        horizontal_scroll = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.table.xview,
        )
        self.table.configure(xscrollcommand=horizontal_scroll.set)
        self.table.pack(fill="both", expand=True)
        horizontal_scroll.pack(fill="x")
        self.table.bind("<<TreeviewSelect>>", lambda _event: self._choose_current())

        self.save_button = ttk.Button(
            body, text="Save selection as JSON…", command=self._save_current, state="disabled",
        )
        self.save_button.pack(anchor="w", pady=(0, 5))

        self.status = tk.StringVar(
            value="Select a target and refresh. No GPU has been qualified for training."
        )
        ttk.Label(body, textvariable=self.status, wraplength=880, justify="left").pack(
            anchor="w", fill="x", pady=(4, 0)
        )
        self._input_changed()
        if synthetic_demo:
            self.local_radio.configure(state="disabled")
            self.remote_radio.configure(state="disabled")
            self.alias_entry.configure(state="disabled")
            self.refresh()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._after_id = self.root.after(100, self._pump)

    def _input_changed(self) -> None:
        if self._closed:
            return
        self.alias_entry.configure(state="normal" if self.kind.get() == "ssh" else "disabled")
        self.model.invalidate()
        self._clear_rows()
        self.save_button.configure(state="disabled")
        self.status.set("Target changed. Refresh GPUs to observe hardware; no training is started.")

    def _target_from_fields(self) -> GpuTarget:
        if self.kind.get() == "local":
            return GpuTarget(kind="local")
        return GpuTarget(kind="ssh", ssh_alias=self.alias.get().strip())

    def _clear_rows(self) -> None:
        self._row_uuids.clear()
        for row_id in self.table.get_children(""):
            self.table.delete(row_id)

    def refresh(self) -> None:
        if self._closed or self._active is not None:
            return
        if self.synthetic_demo:
            generation, _ = self.model.begin_refresh()
            self._clear_rows()
            self._display_inventory(generation, synthetic_demo_inventory())
            return
        try:
            target = self._target_from_fields()
        except (TypeError, ValueError, GpuSelectionError) as error:
            self.status.set(f"Invalid target: {error}")
            return
        self.model.set_target(target)
        generation, target = self.model.begin_refresh()
        self._clear_rows()
        self.status.set(f"Observing NVIDIA GPUs on {_target_label(target)}. No training is started.")
        self.refresh_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self._active = (generation, target, self._executor.submit(self._probe, target))

    def _display_inventory(self, generation: int, inventory: GpuInventory) -> None:
        if not self.model.accept_inventory(generation, inventory):
            return
        for index, row in enumerate(format_inventory_rows(inventory)):
            item_id = f"gpu-{index}"
            self._row_uuids[item_id] = row[0]
            self.table.insert("", "end", iid=item_id, values=row)
        if self.synthetic_demo:
            self.status.set(
                f"{SYNTHETIC_BANNER}. These sample rows are not real hardware, "
                "cannot be saved, and cannot start training."
            )
        else:
            self.status.set(
                f"Observed {len(inventory.devices)} NVIDIA GPU(s) on {_target_label(inventory.target)}. "
                "Choose an exact UUID. Observation does not qualify a training runtime."
            )

    def _pump(self) -> None:
        self._after_id = None
        if self._closed:
            return
        if self._active is not None and self._active[2].done():
            generation, target, future = self._active
            self._active = None
            self.refresh_button.configure(state="normal")
            if generation == self.model.generation and target == self.model.target:
                try:
                    inventory = future.result()
                except (GpuProbeError, GpuSelectionError, OSError, ValueError) as error:
                    self.status.set(f"GPU observation failed: {error}. No training was started.")
                else:
                    self._display_inventory(generation, inventory)
        self._after_id = self.root.after(100, self._pump)

    def _choose_current(self) -> None:
        if self._closed:
            return
        items = self.table.selection()
        if not items:
            return
        gpu_uuid = self._row_uuids.get(items[0])
        if gpu_uuid is None:
            return
        try:
            choice = self.model.choose_gpu(gpu_uuid)
        except GpuSelectionError as error:
            self.status.set(f"GPU choice unavailable: {error}. Refresh the target.")
            return
        self.status.set(selection_message(choice, synthetic_demo=self.synthetic_demo))
        self.save_button.configure(state="normal" if self.model.can_save else "disabled")

    def _save_current(self) -> None:
        if self._closed or not self.model.can_save:
            return
        destination = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save observed GPU selection",
            defaultextension=".json",
            initialfile="lived-gpu-selection.json",
            filetypes=(("JSON", "*.json"),),
        )
        if not destination:
            return
        try:
            saved_path = self.model.save_choice(Path(destination))
        except FileExistsError:
            self.status.set("That file already exists. Choose a new name; no file was overwritten.")
        except (GpuSelectionError, OSError, ValueError) as error:
            self.status.set(f"Could not save the GPU choice: {error}. No training was started.")
        else:
            self.status.set(
                f"Saved observed GPU choice to {saved_path}. "
                "This file does not configure or start training."
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.model.invalidate()
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and choose a GPU for the liveD preview")
    parser.add_argument(
        "--synthetic-demo", action="store_true",
        help="show fake sample GPUs without probing local or remote hardware",
    )
    arguments = parser.parse_args()
    root = tk.Tk()
    GpuPickerApp(root, synthetic_demo=arguments.synthetic_demo)
    root.mainloop()


if __name__ == "__main__":
    main()
