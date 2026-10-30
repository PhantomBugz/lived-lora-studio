# Release status

Checked 2026-10-30. This repository is a source-level developer preview.

## Verified

- The one-command synthetic review/export demo passed on Windows with Python 3.12. It exercised an encrypted local vault, explicit review decisions, approved-only encrypted export, and a plaintext exclusion check.
- The public snapshot passed 24 GPU selector and picker tests using synthetic inventories, and 100 local workflow tests (2 skipped) on Windows with Python 3.12.
- The project code is under Apache-2.0. The public preview is intended to omit bundled model weights and deployment-specific infrastructure.

These checks cover the named demo, selector, and local workflow tests. Source-capture adapters included as import dependencies have not been exercised against live sites or provider accounts in this preview.

## Pending before publication

- Run the source smoke workflow on GitHub-hosted Windows and Linux runners.

## Pending for an installable training release

- Build a GPU-specific CUDA runtime and bind the chosen target and UUID into the reviewed plan, approval, worker, and run record.
- Verify a real LoRA execution and its failure/cancel/recovery behavior on the supported machines.
- Assemble and test complete platform installers, dependencies, current-date release evidence, and rollback.

The preview's GPU selection is for target discovery and planning. It is not evidence that LoRA, QLoRA, or DPO training can run on a chosen card.
