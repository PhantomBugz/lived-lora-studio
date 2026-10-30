"""Headless state tests for the developer-preview GPU picker."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


FIRST_UUID = "GPU-11111111-1111-1111-1111-111111111111"
SECOND_UUID = "GPU-22222222-2222-2222-2222-222222222222"


class GpuPickerModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("scripts.select_gpu_gui"),
            "The developer-preview GPU picker is not implemented",
        )
        from scripts import select_gpu_gui
        from studio import gpu_selection

        self.picker = select_gpu_gui
        self.api = gpu_selection
        self.local = self.api.GpuTarget(kind="local")
        self.remote = self.api.GpuTarget(kind="ssh", ssh_alias="training-host")
        self.observed_at = datetime.now(timezone.utc)

    def inventory(self, target=None):
        return self.api.GpuInventory(
            target=target or self.local,
            observed_at=self.observed_at,
            devices=(
                self.api.GpuDevice(FIRST_UUID, "NVIDIA A", 24576, 20480, "8.6", "570", "physical"),
                self.api.GpuDevice(SECOND_UUID, "NVIDIA B", 49152, 45056, "9.0", "570", "physical"),
            ),
        )

    def test_exact_uuid_choice_from_current_inventory(self) -> None:
        model = self.picker.GpuPickerModel(self.local)
        generation, target = model.begin_refresh()
        self.assertEqual(target, self.local)
        self.assertTrue(model.accept_inventory(generation, self.inventory()))

        choice = model.choose_gpu(SECOND_UUID)
        self.assertEqual(choice.gpu_uuid, SECOND_UUID)
        self.assertEqual(choice.target, self.local)
        self.assertTrue(model.can_save)
        self.assertEqual(self.picker.format_inventory_rows(model.inventory)[1][0], SECOND_UUID)
        self.assertIn("does not start training", self.picker.selection_message(choice))

    def test_target_change_discards_old_probe_and_selection(self) -> None:
        model = self.picker.GpuPickerModel(self.local)
        old_generation, _ = model.begin_refresh()
        model.set_target(self.remote)

        self.assertFalse(model.accept_inventory(old_generation, self.inventory()))
        self.assertIsNone(model.inventory)
        self.assertIsNone(model.selection)
        with self.assertRaises(self.api.GpuSelectionError):
            model.choose_gpu(FIRST_UUID)

    def test_refresh_invalidates_choice_and_rejects_old_result(self) -> None:
        model = self.picker.GpuPickerModel(self.local)
        first_generation, _ = model.begin_refresh()
        self.assertTrue(model.accept_inventory(first_generation, self.inventory()))
        model.choose_gpu(FIRST_UUID)

        second_generation, _ = model.begin_refresh()
        self.assertIsNone(model.inventory)
        self.assertIsNone(model.selection)
        self.assertFalse(model.can_save)
        self.assertFalse(model.accept_inventory(first_generation, self.inventory()))
        self.assertTrue(model.accept_inventory(second_generation, self.inventory()))

    def test_inventory_for_another_target_is_never_displayed(self) -> None:
        model = self.picker.GpuPickerModel(self.remote)
        generation, _ = model.begin_refresh()
        self.assertFalse(model.accept_inventory(generation, self.inventory(self.local)))
        self.assertIsNone(model.inventory)

    def test_synthetic_demo_is_labeled_and_cannot_be_saved_as_observation(self) -> None:
        model = self.picker.GpuPickerModel(self.local, synthetic_demo=True)
        inventory = self.picker.synthetic_demo_inventory()
        self.assertFalse(inventory.training_qualified)
        generation, _ = model.begin_refresh()
        self.assertTrue(model.accept_inventory(generation, inventory))

        choice = model.choose_gpu(inventory.devices[0].uuid)
        self.assertFalse(choice.training_qualified)
        self.assertFalse(model.can_save)
        self.assertIn("SYNTHETIC DEMO", self.picker.selection_message(choice, synthetic_demo=True))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(self.api.GpuSelectionError):
                model.save_choice(Path(temporary) / "fake-selection.json")

    def test_explicit_save_uses_exclusive_json_output(self) -> None:
        model = self.picker.GpuPickerModel(self.local)
        generation, _ = model.begin_refresh()
        self.assertTrue(model.accept_inventory(generation, self.inventory()))
        model.choose_gpu(SECOND_UUID)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selection.json"
            self.assertEqual(model.save_choice(path), path)
            payload = path.read_bytes()
            document = json.loads(payload)
            self.assertEqual(document["gpu_uuid"], SECOND_UUID)
            self.assertFalse(document["training_qualified"])
            with self.assertRaises(FileExistsError):
                model.save_choice(path)
            self.assertEqual(path.read_bytes(), payload)

    def test_stale_choice_is_not_saved(self) -> None:
        model = self.picker.GpuPickerModel(self.local)
        generation, _ = model.begin_refresh()
        self.assertTrue(model.accept_inventory(generation, self.inventory()))
        model.choose_gpu(FIRST_UUID)
        self.assertIsNotNone(model.inventory)
        model.inventory = replace(
            model.inventory,
            observed_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "stale.json"
            with self.assertRaises(self.api.GpuSelectionError):
                model.save_choice(destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
