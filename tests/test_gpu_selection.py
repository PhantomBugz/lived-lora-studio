"""Behavioral checks for read-only NVIDIA target selection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from studio.hardware_inventory import ProcessOutput


selection = None
if importlib.util.find_spec("studio.gpu_selection") is not None:
    from studio import gpu_selection as selection
cli = None
if importlib.util.find_spec("scripts.select_gpu") is not None:
    from scripts import select_gpu as cli


FIRST_UUID = "GPU-11111111-1111-1111-1111-111111111111"
SECOND_UUID = "GPU-22222222-2222-2222-2222-222222222222"
OBSERVED_AT = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
TWO_GPU_CSV = (
    f'{FIRST_UUID}, "NVIDIA RTX, fixture", 24576, 20000, 8.6, 560.35\n'
    f"{SECOND_UUID}, NVIDIA fixture 2, 16384, 12000, 7.5, 560.35\n"
).encode("utf-8")


class GpuSelectionTests(unittest.TestCase):
    def test_probe_rejects_unbounded_or_nonfinite_timeout_before_launch(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        calls = []

        def runner(*_args, **_kwargs):
            calls.append(True)
            return ProcessOutput(TWO_GPU_CSV, 0)

        for seconds in (0, -1, 8.01, float("nan"), float("inf"), True, "8"):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                selection.discover_gpus(
                    selection.GpuTarget("local"),
                    process_runner=runner,
                    tool_resolver=lambda _name: Path("C:/trusted/nvidia-smi.exe"),
                    timeout_seconds=seconds,
                )
        self.assertEqual(calls, [])

    def test_device_and_inventory_reject_malformed_caller_records(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        with self.assertRaises(ValueError):
            selection.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 12000, "invalid", "560.35")
        with self.assertRaises(ValueError):
            selection.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 25000, "8.6", "560.35")
        device = selection.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 12000, "8.6", "560.35")
        with self.assertRaises(ValueError):
            selection.GpuInventory(selection.GpuTarget("local"), OBSERVED_AT.replace(tzinfo=None), (device,))
        with self.assertRaises(ValueError):
            selection.GpuInventory(selection.GpuTarget("local"), OBSERVED_AT, (device,), True)

    def test_selection_rejects_naive_current_time(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        device = selection.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 12000, "8.6", "560.35")
        inventory = selection.GpuInventory(selection.GpuTarget("local"), OBSERVED_AT, (device,))
        with self.assertRaises(selection.GpuSelectionError):
            selection.select_gpu(inventory, FIRST_UUID, now=OBSERVED_AT.replace(tzinfo=None))

    def test_save_refuses_a_choice_whose_uuid_was_swapped_after_selection(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        inventory = selection.discover_gpus(
            selection.GpuTarget("local"),
            process_runner=lambda *_args, **_kwargs: ProcessOutput(TWO_GPU_CSV, 0),
            tool_resolver=lambda _name: Path("C:/trusted/nvidia-smi.exe"),
            clock=lambda: OBSERVED_AT,
        )
        genuine = selection.select_gpu(inventory, FIRST_UUID, now=OBSERVED_AT)
        forged = replace(genuine, gpu_uuid=SECOND_UUID)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory) / "forged.json"
            with self.assertRaises(selection.GpuSelectionError):
                selection.save_selection(forged, output)
            self.assertFalse(output.exists())

    def test_choice_rejects_any_ambiguous_uuid_inventory(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        first = selection.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 12000, "8.6", "560.35")
        second = selection.GpuDevice(SECOND_UUID, "NVIDIA B", 16384, 8000, "7.5", "560.35")
        inventory = selection.GpuInventory(
            selection.GpuTarget("local"), OBSERVED_AT, (first, second, second)
        )
        with self.assertRaises(selection.GpuSelectionError):
            selection.select_gpu(inventory, FIRST_UUID, now=OBSERVED_AT)

    def test_choice_rejects_stale_missing_and_old_cuda_devices(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        supported = selection.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 12000, "8.6", "560.35")
        old_cuda = selection.GpuDevice(SECOND_UUID, "NVIDIA B", 8192, 4000, "6.1", "560.35")
        inventory = selection.GpuInventory(
            selection.GpuTarget("local"), OBSERVED_AT, (supported, old_cuda)
        )
        with self.assertRaises(selection.GpuSelectionError):
            selection.select_gpu(inventory, FIRST_UUID, now=OBSERVED_AT + timedelta(seconds=61))
        with self.assertRaises(selection.GpuSelectionError):
            selection.select_gpu(
                inventory, "GPU-33333333-3333-3333-3333-333333333333", now=OBSERVED_AT
            )
        with self.assertRaises(selection.GpuSelectionError):
            selection.select_gpu(inventory, SECOND_UUID, now=OBSERVED_AT)

    def test_explicit_selection_save_is_json_and_never_overwrites(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        inventory = selection.discover_gpus(
            selection.GpuTarget("ssh", "gpu-lab"),
            process_runner=lambda *_args, **_kwargs: ProcessOutput(TWO_GPU_CSV, 0),
            tool_resolver=lambda _name: Path("C:/trusted/ssh.exe"),
            clock=lambda: OBSERVED_AT,
        )
        chosen = selection.select_gpu(inventory, FIRST_UUID, now=OBSERVED_AT)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory) / "gpu-selection.json"
            selection.save_selection(chosen, output)
            original = output.read_bytes()
            document = json.loads(original)
            self.assertEqual(document["schema"], "LIVED_GPU_SELECTION_V1")
            self.assertEqual(document["target"], {"kind": "ssh", "ssh_alias": "gpu-lab"})
            self.assertEqual(document["gpu_uuid"], FIRST_UUID)
            self.assertFalse(document["training_qualified"])
            self.assertNotIn("password", original.decode("utf-8").lower())
            with self.assertRaises(FileExistsError):
                selection.save_selection(chosen, output)
            self.assertEqual(output.read_bytes(), original)

    def test_saved_selection_rejects_changed_target_or_gpu_binding(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        inventory = selection.discover_gpus(
            selection.GpuTarget("ssh", "gpu-lab"),
            process_runner=lambda *_args, **_kwargs: ProcessOutput(TWO_GPU_CSV, 0),
            tool_resolver=lambda _name: Path("C:/trusted/ssh.exe"),
            clock=lambda: OBSERVED_AT,
        )
        chosen = selection.select_gpu(inventory, FIRST_UUID, now=OBSERVED_AT)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            output = Path(directory) / "gpu-selection.json"
            selection.save_selection(chosen, output)
            self.assertEqual(selection.load_selection(output).gpu_uuid, FIRST_UUID)
            document = json.loads(output.read_text(encoding="utf-8"))
            document["target"]["ssh_alias"] = "other-lab"
            output.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(selection.GpuSelectionError):
                selection.load_selection(output)

    def test_control_sequences_in_device_names_are_not_rendered_as_gpu_choices(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        unsafe = TWO_GPU_CSV.replace(b"NVIDIA fixture 2", b"NVIDIA \x1b[31m fixture 2")
        with self.assertRaises(selection.GpuProbeError):
            selection.discover_gpus(
                selection.GpuTarget("local"),
                process_runner=lambda *_args, **_kwargs: ProcessOutput(unsafe, 0),
                tool_resolver=lambda _name: Path("C:/trusted/nvidia-smi.exe"),
                clock=lambda: OBSERVED_AT,
            )

    def test_partial_or_duplicate_native_queries_never_create_a_choice(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        for result in (
            ProcessOutput(TWO_GPU_CSV, 0, truncated=True),
            ProcessOutput(TWO_GPU_CSV, 0, timed_out=True),
            ProcessOutput(TWO_GPU_CSV + TWO_GPU_CSV.splitlines(keepends=True)[0], 0),
        ):
            with self.subTest(result=result), self.assertRaises(selection.GpuProbeError):
                selection.discover_gpus(
                    selection.GpuTarget("local"),
                    process_runner=lambda *_args, **_kwargs: result,
                    tool_resolver=lambda _name: Path("C:/trusted/nvidia-smi.exe"),
                    clock=lambda: OBSERVED_AT,
                )

    def test_remote_discovery_uses_strict_noninteractive_ssh_with_fixed_query(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        invocations = []

        def runner(argv, *, deadline, max_output_bytes):
            invocations.append((argv, max_output_bytes))
            return ProcessOutput(TWO_GPU_CSV, 0)

        inventory = selection.discover_gpus(
            selection.GpuTarget("ssh", "gpu-lab"),
            process_runner=runner,
            tool_resolver=lambda name: Path("C:/trusted/ssh.exe") if name == "ssh" else None,
            clock=lambda: OBSERVED_AT,
        )
        self.assertEqual(len(inventory.devices), 2)
        self.assertEqual(inventory.target.ssh_alias, "gpu-lab")
        argv, limit = invocations[0]
        self.assertEqual(argv[0], str(Path("C:/trusted/ssh.exe")))
        self.assertEqual(limit, 64 * 1024)
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("StrictHostKeyChecking=yes", argv)
        self.assertIn("ForwardAgent=no", argv)
        self.assertEqual(
            argv[-5:],
            (
                "--", "gpu-lab", "/usr/bin/nvidia-smi",
                "--query-gpu=uuid,name,memory.total,memory.free,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ),
        )

    def test_remote_alias_cannot_supply_ssh_options_or_shell_syntax(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")
        for alias in ("-oProxyCommand=sh", "gpu-host;id", "gpu host", "gpu\nhost", "user@gpu"):
            with self.subTest(alias=alias), self.assertRaises(ValueError):
                selection.GpuTarget("ssh", alias)

    def test_local_discovery_and_uuid_choice_do_not_claim_training_ready(self):
        self.assertIsNotNone(selection, "selectable GPU inventory is missing")

        def runner(_argv, *, deadline, max_output_bytes):
            return ProcessOutput(TWO_GPU_CSV, 0)

        inventory = selection.discover_gpus(
            selection.GpuTarget("local"),
            process_runner=runner,
            tool_resolver=lambda name: Path("C:/trusted/nvidia-smi.exe"),
            clock=lambda: OBSERVED_AT,
        )
        chosen = selection.select_gpu(
            inventory, SECOND_UUID, now=OBSERVED_AT + timedelta(seconds=5)
        )
        self.assertEqual([device.uuid for device in inventory.devices], [FIRST_UUID, SECOND_UUID])
        self.assertEqual(inventory.devices[0].name, "NVIDIA RTX, fixture")
        self.assertEqual((chosen.target.kind, chosen.gpu_uuid), ("local", SECOND_UUID))
        self.assertFalse(inventory.training_qualified)
        self.assertFalse(chosen.training_qualified)


class GpuCliTests(unittest.TestCase):
    def test_refresh_refuses_when_saved_gpu_disappears(self):
        self.assertIsNotNone(cli, "GPU selector CLI is missing")
        old_inventory = selection.GpuInventory(
            selection.GpuTarget("local"), OBSERVED_AT,
            (selection.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 12000, "8.6", "560.35"),),
        )
        chosen = selection.select_gpu(old_inventory, FIRST_UUID, now=OBSERVED_AT)
        new_inventory = selection.GpuInventory(
            old_inventory.target, OBSERVED_AT + timedelta(minutes=2),
            (selection.GpuDevice(SECOND_UUID, "NVIDIA B", 24576, 12000, "8.6", "560.35"),),
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            saved_file = Path(directory) / "chosen.json"
            selection.save_selection(chosen, saved_file)
            output, errors = io.StringIO(), io.StringIO()
            code = cli.main(
                ["refresh", "--selection", str(saved_file)],
                discoverer=lambda _target: new_inventory,
                clock=lambda: new_inventory.observed_at,
                stdout=output,
                stderr=errors,
            )
        self.assertEqual(code, 2)
        self.assertEqual(output.getvalue(), "")
        self.assertIn("selected GPU UUID", errors.getvalue())

    def test_refresh_reprobes_saved_target_and_same_uuid_without_qualifying_training(self):
        self.assertIsNotNone(cli, "GPU selector CLI is missing")
        old_inventory = selection.discover_gpus(
            selection.GpuTarget("ssh", "gpu-lab"),
            process_runner=lambda *_args, **_kwargs: ProcessOutput(TWO_GPU_CSV, 0),
            tool_resolver=lambda _name: Path("C:/trusted/ssh.exe"),
            clock=lambda: OBSERVED_AT,
        )
        chosen = selection.select_gpu(old_inventory, FIRST_UUID, now=OBSERVED_AT)
        refreshed_at = OBSERVED_AT + timedelta(hours=2)
        new_inventory = selection.discover_gpus(
            chosen.target,
            process_runner=lambda *_args, **_kwargs: ProcessOutput(TWO_GPU_CSV, 0),
            tool_resolver=lambda _name: Path("C:/trusted/ssh.exe"),
            clock=lambda: refreshed_at,
        )
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            saved_file = Path(directory) / "chosen.json"
            selection.save_selection(chosen, saved_file)
            output, errors = io.StringIO(), io.StringIO()
            code = cli.main(
                ["refresh", "--selection", str(saved_file)],
                discoverer=lambda _target: new_inventory,
                clock=lambda: refreshed_at + timedelta(seconds=5),
                stdout=output,
                stderr=errors,
            )
        document = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(document["gpu_uuid"], FIRST_UUID)
        self.assertEqual(document["observed_at_utc"], "2026-10-30T14:00:00Z")
        self.assertFalse(document["training_qualified"])

    def test_choose_writes_explicit_selection_and_reports_observed_only(self):
        self.assertIsNotNone(cli, "GPU selector CLI is missing")
        inventory = selection.discover_gpus(
            selection.GpuTarget("local"),
            process_runner=lambda *_args, **_kwargs: ProcessOutput(TWO_GPU_CSV, 0),
            tool_resolver=lambda _name: Path("C:/trusted/nvidia-smi.exe"),
            clock=lambda: OBSERVED_AT,
        )
        output, errors = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            destination = Path(directory) / "chosen.json"
            code = cli.main(
                ["choose", "--gpu", SECOND_UUID, "--output", str(destination)],
                discoverer=lambda _target: inventory,
                clock=lambda: OBSERVED_AT + timedelta(seconds=5),
                stdout=output,
                stderr=errors,
            )
            saved = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(json.loads(output.getvalue()), saved)
        self.assertEqual(saved["gpu_uuid"], SECOND_UUID)
        self.assertFalse(saved["training_qualified"])

    def test_list_outputs_machine_readable_remote_inventory_without_training_claim(self):
        self.assertIsNotNone(cli, "GPU selector CLI is missing")
        inventory = selection.discover_gpus(
            selection.GpuTarget("ssh", "gpu-lab"),
            process_runner=lambda *_args, **_kwargs: ProcessOutput(TWO_GPU_CSV, 0),
            tool_resolver=lambda _name: Path("C:/trusted/ssh.exe"),
            clock=lambda: OBSERVED_AT,
        )
        output, errors = io.StringIO(), io.StringIO()
        code = cli.main(
            ["list", "--remote", "gpu-lab"],
            discoverer=lambda target: inventory,
            stdout=output,
            stderr=errors,
        )
        document = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(errors.getvalue(), "")
        self.assertEqual(document["target"], {"kind": "ssh", "ssh_alias": "gpu-lab"})
        self.assertEqual([item["uuid"] for item in document["devices"]], [FIRST_UUID, SECOND_UUID])
        self.assertFalse(document["training_qualified"])


if __name__ == "__main__":
    unittest.main()
