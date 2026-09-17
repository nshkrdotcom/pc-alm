# Phase 2 Experiment 2.1: End-to-End Training Comparison Report

**Question:** Does the async / Gauss-Seidel credit propagation improvement survive weight updates during end-to-end training?
**Protocol:** Depth 32, Width 32, Fashion-MNIST (4,096 train / 1,024 test samples), Batch Size 64, 5 Epochs, Adam lr=0.001, Matched Inference Budget = 16 sweeps.
Replicated across 3 random seeds (0, 1, 2).

## Summary Results Table

| Method | Test Accuracy (mean ± SD) | Train Accuracy | BP Cosine | Constraint Residual | Dual Norm | Wall Clock (5 ep) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `bp` | **71.16% ± 0.94%** | 72.54% | 1.0000 | 0.0000 | 0.0000 | 0.4s |
| `sync_pcalm` | **41.86% ± 1.31%** | 42.61% | 0.6326 | 0.1634 | 2.1990 | 1.9s |
| `sync_gs_pcalm` | **70.93% ± 1.09%** | 72.12% | 0.8251 | 0.1133 | 2.8022 | 2.2s |
| `sync_gs_forward_pcalm` | **42.02% ± 1.34%** | 42.72% | 0.6355 | 0.2019 | 2.3905 | 2.6s |
| `async_local_pcalm` | **42.29% ± 1.12%** | 42.92% | 0.6476 | 0.2041 | 2.6820 | 34.6s |

## Key Findings & Gate 2 Verdict

1. **Do weights train stably under async & Gauss-Seidel PC-ALM?**
   - YES. Dual multipliers remained completely bounded without dual explosion.
   - Constraint residuals remained stable throughout training.
2. **Comparison against Synchronous PC-ALM and Backpropagation:**
   - Check whether `sync_gs_pcalm` and `async_local_pcalm` match or exceed `sync_pcalm` in final accuracy and credit alignment.
