# Verification

- 16 CPU tests passed on Windows / Python 3.12 / PyTorch 2.13.0+cpu.
- The package wheel built successfully; the training CLI was checked without starting a training run.
- [parity_report.json](parity_report.json): three real held-out RH20T pairs were checked on the original server using CPU / PyTorch 2.8.0. Public and original inference produced **exactly equal embeddings, mappings, paths and filter decisions** for those cases. No model updates were performed and no dataset assets were exported.

These are software and implementation-equivalence checks, not semantic matching accuracy. One real check included unequal sequence lengths (43 human / 47 robot samples).
