# Protocol F0 Sensitivity Check: Minibatch Shuffling & True Async Probe (C5)

## 1. Executive Summary & Acceptance Criteria
- **Acceptance Criterion (C5):** If the ~9-point reverse vs forward gap at $B=64$ moves by > 1 percentage point under epoch shuffling, the unshuffled F0 grid is flagged `[artifact-suspect]`. If it holds, F0 is retained with note *unshuffled, sensitivity-checked*.
- **Observed Shuffled Gap (Reverse vs Forward at B=64):** **9.12 percentage points** (Unshuffled was 9.07 pp; $\Delta = 0.05$ pp).
- **Verdict:** **PASSED**. Directional advantage is fully preserved under minibatch shuffling.

## 2. Comprehensive Comparison Table (Shuffled vs Unshuffled)

| Method | Budget ($B$) | Shuffled Acc (mean ± SD) | 95% Bootstrap CI | Unshuffled Acc | $\Delta$ (Shuffle Impact) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `async_local_pcalm` | 64 | **79.52% ± 0.46%** | [79.13%, 79.86%] | N/A | N/A |
| `sync_gs_forward_pcalm` | 64 | **75.69% ± 0.78%** | [75.08%, 76.30%] | 75.80% | -0.11 pp |
| `sync_gs_pcalm` | 64 | **84.81% ± 0.19%** | [84.67%, 84.97%] | 84.87% | -0.05 pp |
| `sync_pcalm` | 64 | **75.72% ± 0.80%** | [75.09%, 76.34%] | 75.73% | -0.02 pp |
| `sync_gs_pcalm` | 128 | **84.76% ± 0.18%** | [84.62%, 84.90%] | 84.66% | +0.10 pp |
| `sync_pcalm` | 128 | **84.76% ± 0.31%** | [84.53%, 85.01%] | 84.80% | -0.04 pp |

## 3. Findings on True Per-Step Async Coordinate Descent (C4)
- `async_local_pcalm` at $B=64$ achieves **79.52% ± 0.46%**.
- Comparison with other schedules at $B=64$:
  - Reverse GS: 84.81%
  - Random Order GS (fixed perm): 77.42%
  - True Async Local (per-step random): 79.52%
  - Forward GS: 75.69%
  - Jacobi: 75.72%
- **Conclusion:** True per-step random coordinate selection does not match Reverse GS, confirming the negative thesis result under the canonical protocol.
