# Remediation R1 & R2: Budget Sweep & Parallelism Accounting Report

## 1. Executive Summary & Acceptance Criteria

This experiment resolves Blocking Defects **R1** (budget starvation confound) and **R2** (critical-path parallel accounting and forward-GS control).
It evaluates training across budgets $B \in \{8, 16, 32, 48, 64, 96, 128\}$, specifically testing the published regime $B = 2L = 64$ and reporting all three work metrics:
1. `layer_update_work`: total coordinate updates.
2. `sequential_depth`: longest serial dependency chain.
3. `parallel_work`: total work $\times$ depth.

## 2. Test Accuracy vs Budget Across Schedules (Depth 32, Fashion-MNIST)

| Method | Budget ($B$) | Layer Work | Seq Depth | Parallel Work | Test Acc (mean ± SD) | Train Acc | Own BP Cosine | Wall Clock |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `bp` | 0 | 0 | 0 | 0 | **53.45% ± 5.48%** | 53.91% | 1.0000 | 0.2s |
| `async_local_pcalm` | 8 | 248 | 248 | 61504 | **20.57% ± 3.68%** | 21.01% | 0.7431 | 6.8s |
| `async_local_pcalm` | 16 | 496 | 496 | 246016 | **23.76% ± 3.19%** | 24.33% | 0.7502 | 12.8s |
| `async_local_pcalm` | 32 | 992 | 992 | 984064 | **41.08% ± 3.36%** | 42.25% | 0.8074 | 25.7s |
| `async_local_pcalm` | 48 | 1488 | 1488 | 2214144 | **50.72% ± 3.44%** | 51.90% | 0.9095 | 38.5s |
| `async_local_pcalm` | 64 | 1984 | 1984 | 3936256 | **52.73% ± 3.45%** | 53.48% | 0.9799 | 51.0s |
| `async_local_pcalm` | 96 | 2976 | 2976 | 8856576 | **52.93% ± 4.34%** | 53.71% | 0.9817 | 72.4s |
| `async_local_pcalm` | 128 | 3968 | 3968 | 15745024 | **53.52% ± 5.12%** | 54.51% | 0.9806 | 95.9s |
| `sync_gs_forward_pcalm` | 8 | 248 | 248 | 61504 | **20.64% ± 3.55%** | 21.04% | 0.7422 | 0.9s |
| `sync_gs_forward_pcalm` | 16 | 496 | 496 | 246016 | **23.44% ± 2.92%** | 24.32% | 0.7427 | 1.0s |
| `sync_gs_forward_pcalm` | 32 | 992 | 992 | 984064 | **31.12% ± 4.01%** | 31.35% | 0.7513 | 1.5s |
| `sync_gs_forward_pcalm` | 48 | 1488 | 1488 | 2214144 | **53.26% ± 3.82%** | 54.48% | 0.8684 | 2.0s |
| `sync_gs_forward_pcalm` | 64 | 1984 | 1984 | 3936256 | **52.73% ± 3.38%** | 53.74% | 0.9591 | 2.3s |
| `sync_gs_forward_pcalm` | 96 | 2976 | 2976 | 8856576 | **52.93% ± 4.22%** | 54.04% | 0.9630 | 3.2s |
| `sync_gs_forward_pcalm` | 128 | 3968 | 3968 | 15745024 | **53.52% ± 5.73%** | 54.38% | 0.9569 | 4.0s |
| `sync_gs_pcalm` | 8 | 248 | 248 | 61504 | **43.95% ± 4.02%** | 45.25% | 0.7773 | 0.7s |
| `sync_gs_pcalm` | 16 | 496 | 496 | 246016 | **53.84% ± 4.10%** | 54.61% | 0.8411 | 0.9s |
| `sync_gs_pcalm` | 32 | 992 | 992 | 984064 | **52.73% ± 3.41%** | 53.81% | 0.9623 | 1.2s |
| `sync_gs_pcalm` | 48 | 1488 | 1488 | 2214144 | **52.41% ± 3.78%** | 53.24% | 0.9556 | 1.6s |
| `sync_gs_pcalm` | 64 | 1984 | 1984 | 3936256 | **53.39% ± 4.10%** | 53.97% | 0.9618 | 1.8s |
| `sync_gs_pcalm` | 96 | 2976 | 2976 | 8856576 | **53.65% ± 4.65%** | 54.72% | 0.9570 | 2.4s |
| `sync_gs_pcalm` | 128 | 3968 | 3968 | 15745024 | **53.06% ± 5.00%** | 53.17% | 0.9391 | 3.0s |
| `sync_pcalm` | 8 | 248 | 8 | 1984 | **20.57% ± 3.64%** | 21.08% | 0.7408 | 0.5s |
| `sync_pcalm` | 16 | 496 | 16 | 7936 | **23.24% ± 2.87%** | 24.24% | 0.7400 | 0.7s |
| `sync_pcalm` | 32 | 992 | 32 | 31744 | **30.79% ± 4.23%** | 30.89% | 0.7452 | 1.2s |
| `sync_pcalm` | 48 | 1488 | 48 | 71424 | **52.99% ± 3.50%** | 54.31% | 0.8481 | 1.6s |
| `sync_pcalm` | 64 | 1984 | 64 | 126976 | **52.47% ± 3.55%** | 54.02% | 0.9613 | 2.1s |
| `sync_pcalm` | 96 | 2976 | 96 | 285696 | **53.26% ± 3.57%** | 53.76% | 0.9589 | 2.9s |
| `sync_pcalm` | 128 | 3968 | 128 | 507904 | **53.19% ± 4.34%** | 54.52% | 0.9646 | 3.7s |

## 3. Analysis of Jacobi Convergence Regime (R1)

- **Published Operating Regime ($B = 2L = 64$)**:
  - The published PC-ALM operating point is $T \approx 2L = 64$.
  - At $B=8$ and $B=16 < L=32$, Jacobi PC-ALM achieves only **20.57%** and **23.24%** test accuracy with BP cosine alignment of only 0.7400.
  - However, at $B=48$ ($1.5L$), Jacobi jumps directly to **52.99% ± 3.50%** (matching BP baseline **53.45% ± 5.48%**).
  - At $B=64$ ($2L$), Jacobi achieves **52.47% ± 3.55%** with BP cosine alignment of **0.9613**.
  - Beyond $B \ge 64$ ($B=96, 128$), Jacobi remains rock-stable at **53.26%** and **53.19%** with BP cosine of **0.9646**.
- **Definitive Verdict on R1**:
  - **The previous claim that Jacobi fails at depth 32 was an artifact of budget starvation.**
  - Jacobi PC-ALM converges stably and reaches full parity with Backpropagation once given $B \ge 1.5L$ sweeps.
  - Reverse Gauss-Seidel does NOT offer an asymptotic accuracy advantage over Jacobi; it offers a **3x–4x sweep-budget acceleration in the sub-$2L$ regime** ($B=16$ vs $B=48..64$).

## 4. Critical-Path Sequential Depth vs Parallel Work Analysis (R2)

| Schedule | Parity Budget ($B^*$) | Layer Updates ($B \times 31$) | Sequential Depth ($T_{\text{crit}}$) | Parallel Work ($W \times T$) |
| :--- | :---: | :---: | :---: | :---: |
| `sync_gs_pcalm` (Reverse GS) | **16** | **496** | 496 | 246,016 |
| `sync_pcalm` (Jacobi) | **48** | 1,488 | **48** | **71,424** |
| `sync_pcalm` ($B=2L=64$) | **64** | 1,984 | **64** | **126,976** |

- **Sequential Critical-Path Paradox**:
  - Reverse GS requires only 496 total layer updates to match BP, compared to 1,488 for Jacobi at $B=48$.
  - However, Reverse GS enforces strict serial dependencies: layer $i-1$ cannot begin until layer $i$ finishes. Its sequential depth is **496 cycles**.
  - Jacobi updates all 31 layers simultaneously per sweep. Its sequential depth at $B=48$ is only **48 cycles** (and 64 cycles at $B=64$).
  - **Verdict on Parallel Advantage**: On an idealized parallel accelerator with $P \ge L=32$ processing cores, **Jacobi reaches BP accuracy in 1/10th the wall-clock cycles of reverse Gauss-Seidel (48 cycles vs 496 cycles)**, and requires **3.4x less parallel work** (71,424 vs 246,016). The "Gauss-Seidel win" exists *only* on strictly single-threaded serial hardware.

## 5. Forward-GS Control & Order-Specificity Statement (R2)

- **Control Comparison**:
  - At $B=16$: Reverse GS achieves **53.84%**, while Forward GS achieves **23.44%** (identical to Jacobi's 23.24%).
  - At $B=32$: Reverse GS achieves **52.73%**, while Forward GS achieves **31.12%** (identical to Jacobi's 30.79%).
  - Forward GS only reaches BP parity when $B \ge 48$ (**53.26%**), matching Jacobi's exact convergence trajectory.
- **Definitive Order-Specificity Verdict**:
  - The acceleration in Gauss-Seidel is **strictly directional (output-to-input reverse propagation)**.
  - Generic sequential coordinate descent provides zero acceleration over Jacobi. Forward Gauss-Seidel provides no benefit whatsoever until Jacobi itself reaches convergence depth.

## 6. Asynchronous PC-ALM Verdict

- `async_local_pcalm` samples layer updates randomly with immediate local dual updates.
- At $B=8, 16$: `async_local_pcalm` scores **20.57%** and **23.76%**, exactly tracking Jacobi and Forward GS.
- At $B=32$: `async_local_pcalm` reaches **41.08%**, interpolating between Jacobi (30.79%) and Reverse GS (52.73%).
- At $B \ge 48$: `async_local_pcalm` reaches BP parity (**50.72%** at B=48, **52.73%** at B=64).
- **Core Thesis Verdict**:
  - Asynchronous coordinate updates do NOT improve credit propagation speed or work efficiency over reverse Gauss-Seidel.
  - Random asynchrony behaves as a noisy, intermediate interpolant that sacrifices both the $L$-way parallelism of Jacobi and the optimal $O(L)$ reverse propagation of Gauss-Seidel.

## 7. What Would Change Our Mind

- **Jacobi Parity Reversal**: If testing on CIFAR-10 or ImageNet at Depth 128 shows Jacobi diverging or failing to match BP at $B=2L$, while reverse GS succeeds, that would indicate an algorithmic depth barrier beyond simple credit starvation.
- **Hardware Realization**: If actual communication interconnect latency between cores exceeds local computation time such that Jacobi's synchronization barrier dominates, reverse GS pipelining could regain wall-clock superiority despite higher sequential depth.
