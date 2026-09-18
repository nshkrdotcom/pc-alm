# Forward Milestone F2: Block Gauss-Seidel Pareto Frontier

## 1. Executive Summary & Confidence Ladder Certification
- **Milestone F2 Goal:** Map the empirical Pareto frontier between serial work ($W$) and parallel hardware critical-path latency ($T_{\text{crit}}$) by sweeping block sizes $K \in \{1, 2, 4, 8, 16, 31\}$ on a 32-layer MLP ($n=31$ hidden layers) at budget $B=64$ across 5 seeds on Fashion-MNIST.
- **Key Empirical Finding:**
  - The transition between serial Reverse GS ($K=1$, 84.36%) and parallel Jacobi ($K=31$, 75.41%) is **not smooth**; it exhibits a **sharp cliff**:
    - $K=1$ ($T_{\text{crit}}=1,984$): **84.36% ± 0.20%**
    - $K=2$ ($T_{\text{crit}}=1,024$): **78.16% ± 0.66%** (-6.20 pp drop)
    - $K=4$ ($T_{\text{crit}}=512$): **75.59% ± 0.91%** (-8.77 pp drop, collapsing to Jacobi baseline)
    - $K \in \{8, 16, 31\}$ ($T_{\text{crit}} \in \{256, 128, 64\}$): **~75.4%–75.6%** (identical to uncoordinated Jacobi).
- **Status:** **`[established]`**. Pipelined block relaxation cannot preserve directional credit flow unless the block size is strictly $K \le 2$.

---

## 2. Block-GS Pareto Frontier Table ($B=64$, 5 seeds each)

| Block Size ($K$) | Blocks ($M$) | Critical Path ($T_{\text{crit}}$) | Latency Reduction vs Rev GS | Test Accuracy (mean ± SD) | 95% Bootstrap CI | Dual Norm | Wall Clock |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** (Pure Rev GS) | 31 | 1984 | $1.0\times$ (baseline) | **84.36% ± 0.20%** | [84.18%, 84.47%] | 2.37 | 55.2s |
| **2** | 16 | 1024 | **$1.94\times$ faster** | **78.16% ± 0.66%** | [77.69%, 78.74%] | 2.76 | 54.9s |
| **4** | 8 | 512 | **$3.88\times$ faster** | **75.59% ± 0.91%** | [74.85%, 76.27%] | 2.86 | 55.6s |
| **8** | 4 | 256 | **$7.75\times$ faster** | **75.55% ± 0.80%** | [74.93%, 76.18%] | 2.89 | 56.6s |
| **16** | 2 | 128 | **$15.50\times$ faster** | **75.65% ± 0.66%** | [75.15%, 76.17%] | 2.93 | 55.5s |
| **31** (Pure Jacobi) | 1 | 64 | **$31.00\times$ faster** | **75.41% ± 0.73%** | [74.83%, 75.95%] | 2.94 | 56.2s |

---

## 3. Theoretical & Physical Implications

### 1. The Intra-Block Staleness Threshold
- When $K$ layers are grouped into an uncoordinated parallel block, each layer within the block uses activity from the previous sweep rather than updated adjacent layer states.
- For $K=2$, each pair of layers updates simultaneously, introducing a single-step local delay. This incurs a moderate -6.20 pp degradation.
- For $K \ge 4$, intra-block delay exceeds the stability margin of the local predictive coding dynamics at $B=64$, immediately destroying the benefit of inter-block sequential ordering.

### 2. The Architectural Hardware Trade-off
- If the hardware objective is maximum accuracy (~84.5%):
  - **Serial hardware / Low core count:** Use fine-grained Reverse GS ($K=1$) at $B=64$.
  - **Massively parallel hardware ($P \ge 31$):** Run pure Jacobi ($K=31$) at higher budget ($B=112$), which reaches **84.81%** in just **112 parallel cycles**, outperforming Reverse GS's 1,984 sequential cycles by **$17.7\times$ lower latency**.
- Intermediate block pipelining ($K \ge 4$) offers no Pareto advantage over high-budget Jacobi.
