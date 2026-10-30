# Choose an NVIDIA GPU target

The first selectable target workflow is designed for local Windows/Linux and remote Linux machines. A remote target is an existing SSH host alias configured by the operator. The picker uses a bounded, read-only NVIDIA inventory query and presents device UUIDs so the user can choose one device explicitly.

## Commands

```shell
python scripts/select_gpu.py list
python scripts/select_gpu.py list --remote SSH_ALIAS
python scripts/select_gpu.py choose --gpu GPU-UUID --output selected-gpu.json
python scripts/select_gpu.py choose --remote SSH_ALIAS --gpu GPU-UUID --output selected-gpu.json
python scripts/select_gpu.py refresh --selection selected-gpu.json
python scripts/select_gpu_gui.py
```

`choose` can print its JSON result or write a target and GPU selection without credentials. The file is created only if the name does not exist. `refresh` re-queries the recorded target and blocks when the chosen UUID is absent or no longer meets the selector's compute-capability floor. Until the runtime path is implemented, no training command consumes that file.

The GUI is optional and needs Tk. Its `--synthetic-demo` mode displays prominently labeled fake devices without a hardware or SSH probe and disables saving. Remote discovery uses a fixed read-only command at `/usr/bin/nvidia-smi` over noninteractive OpenSSH with strict host-key checking. This first selector handles physical NVIDIA GPU UUIDs; MIG partitions are outside the preview contract.

## What the result means

- **Observed** means the target's `nvidia-smi` returned a device UUID and hardware details at that point in time. Free VRAM can change after the query.
- **Selected** means the operator chose that exact target and UUID. A saved selection must be checked again before any future run.
- **Training qualified** requires a compatible CUDA build of PyTorch, the selected GPU visible to that runtime, a matching model and recipe, and a real execution smoke. This preview does not claim those checks have passed.

The picker does not install CUDA or model packages, run training, send datasets, or grant remote access. Keep SSH credentials in the operator's normal SSH configuration and verify the host key before querying a remote machine.

An absent, ambiguous, or changed UUID should block selection. CUDA capability and memory requirements depend on the eventual model and training recipe; the picker must not turn a device name alone into a readiness claim.
