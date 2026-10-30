# Contributing

Thanks for helping improve liveD LoRA Studio. Keep changes focused and reproducible with synthetic data.

## Bug reports

Include the operating system, Python version, exact command, expected result, observed result, and a minimal synthetic reproduction. Remove credentials, private keys, captured pages, browser profiles, vaults, SSH configuration, and real training records from logs or screenshots before sharing them.

For security issues, follow [SECURITY.md](SECURITY.md) and use GitHub private vulnerability reporting. Avoid posting exploit details in a public issue.

## Pull requests

1. Describe the behavior changed and the data or execution boundary it affects.
2. Add a focused test using synthetic data, especially for approval gates, export filtering, target selection, or input parsing.
3. Run the one-command demo and the applicable focused tests before opening the pull request:

   ```shell
   python scripts/quickstart_preview.py
   python -B -m unittest tests.test_gpu_selection tests.test_gpu_picker_model -v
   ```

The preview checks source behavior. Installed desktop operation and GPU training require separate release evidence.
