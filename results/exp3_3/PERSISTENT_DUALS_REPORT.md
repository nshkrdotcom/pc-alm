# Phase 3 Experiment 3.3: Persistent Duals Across Minibatches Report

**Question:** Does carrying dual multipliers across minibatches (with leak gamma) improve training at low inference budgets (budget = 8 sweeps)?
**Protocol:** Depth 32, Width 32, Fashion-MNIST, Batch Size 64, 5 Epochs, Budget = 8 sweeps, 3 seeds (0, 1, 2).

## Summary Results Table

| Leak Parameter (gamma) | Regime | Test Accuracy (mean ± SD) | Train Accuracy | Final Running Dual Norm |
| :---: | :--- | :---: | :---: | :---: |
| `1.0` | Standard Reset | **65.33% ± 0.77%** | 66.40% | 1.9268 |
| `0.1` | Persistent (leak=0.1) | **37.57% ± 1.34%** | 38.12% | 2.2463 |
| `0.01` | Persistent (leak=0.01) | **35.42% ± 1.76%** | 35.48% | 2.4565 |
| `0.0` | Persistent (leak=0) | **35.42% ± 1.51%** | 35.64% | 2.5079 |

## Key Findings

1. **Full Persistence (gamma = 0) vs Leaky Persistence:**
   - Carrying duals across batches removes the global per-batch reset barrier.
   - Monitor running dual norm to detect whether duals remain bounded or drift under sample changes.
