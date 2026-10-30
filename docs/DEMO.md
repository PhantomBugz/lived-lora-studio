# Synthetic review and encrypted export

This demo runs the preview's real vault, review decisions, and encrypted JSONL exporter with an original, locally authored example. It does not connect to a website, model, or GPU.

## Run

From a checkout with Python 3.12 and internet access:

```shell
python scripts/quickstart_preview.py
```

The script creates `.venv-preview` and installs the pinned `cryptography` wheel and its dependencies from PyPI. If that dependency is already installed in your Python environment, you can run the demo without setup:

```shell
python -B -I scripts/demo_review_export.py
```

It creates a temporary workspace, generates fresh disposable passphrases, and removes the workspace when the run finishes. No passphrase is printed.

## Follow the output

1. A synthetic capture enters the local encrypted vault.
2. Three candidate instruction/response pairs are added. One is approved, one rejected, and one left pending.
3. The exporter writes a manifest and an AES-256-GCM encrypted JSONL shard containing only the approved pair.
4. The demo decrypts its own disposable shard in memory to check the selected record, then confirms that none of the sample responses appear as plaintext in the staged files.

Expected status lines from the verified Windows run on 2026-10-30:

```text
Synthetic review: 1 approved, 1 rejected, 1 pending
Encrypted export: 1 approved record; rejected and pending records excluded
Checks passed: encrypted files contain no sample response plaintext
```

This is a source-level check of one local path. [Release status](STATUS.md) lists what remains before a desktop or training release.
