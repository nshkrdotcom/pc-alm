# Phase 4 Experiment 4.1: Local Triggering Laws & Pareto Schedule Search

## 1. Executive Summary

This experiment evaluates whether local self-timed firing rules (residual-triggered, gradient/disturbance-triggered, anti-starvation age laws, and dual-motion early exit certificates) systematically outperform fixed schedules in work efficiency (credit alignment per layer update event).

## 2. Aggregated Results Across 3 Seeds

### Fashion-MNIST (Depth 32, Width 32, Max Budget = 32 Sweeps)

| Policy | Kind | Budget | Mean Sweeps | BP Cosine to BP | Residual Norm | Work Efficiency (Cos/Sweeps) | Early Exit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fixed_sync_b16` | fixed_sync | 16 | 16.0 ± 0.0 | **0.6995** ± 0.0356 | 0.5969 | **0.0437** | 0% |
| `fixed_sync_b32` | fixed_sync | 32 | 32.0 ± 0.0 | **0.7424** ± 0.0258 | 0.6735 | **0.0232** | 0% |
| `fixed_gs_b4` | fixed_gs | 4 | 4.0 ± 0.0 | **0.6800** ± 0.0407 | 0.6060 | **0.1700** | 0% |
| `fixed_gs_b8` | fixed_gs | 8 | 8.0 ± 0.0 | **0.7105** ± 0.0329 | 0.7198 | **0.0888** | 0% |
| `fixed_gs_b16` | fixed_gs | 16 | 16.0 ± 0.0 | **0.7720** ± 0.0222 | 0.8875 | **0.0483** | 0% |
| `fixed_gs_b32` | fixed_gs | 32 | 32.0 ± 0.0 | **0.9440** ± 0.0044 | 1.0298 | **0.0295** | 0% |
| `dual_exit_eps1e-3` | dual_exit | 32 | 32.0 ± 0.0 | **0.9440** ± 0.0044 | 1.0298 | **0.0295** | 0% |
| `dual_exit_eps3e-4` | dual_exit | 32 | 32.0 ± 0.0 | **0.9440** ± 0.0044 | 1.0298 | **0.0295** | 0% |
| `dual_exit_eps1e-4` | dual_exit | 32 | 32.0 ± 0.0 | **0.9440** ± 0.0044 | 1.0298 | **0.0295** | 0% |
| `dual_exit_eps3e-5` | dual_exit | 32 | 32.0 ± 0.0 | **0.9440** ± 0.0044 | 1.0298 | **0.0295** | 0% |
| `dual_exit_eps1e-5` | dual_exit | 32 | 32.0 ± 0.0 | **0.9440** ± 0.0044 | 1.0298 | **0.0295** | 0% |
| `gradient_wave_th1e-5` | gradient | 32 | 17.0 ± 0.0 | **0.9355** ± 0.0061 | 1.0912 | **0.0550** | 0% |
| `gradient_wave_th3e-5` | gradient | 32 | 3.5 ± 0.9 | **0.7600** ± 0.0109 | 0.8605 | **0.2259** | 67% |
| `gradient_wave_th1e-4` | gradient | 32 | 0.4 ± 0.0 | **0.6736** ± 0.0394 | 0.5445 | **1.6641** | 100% |
| `prob_mild` | probabilistic | 32 | 12.5 ± 0.4 | **0.7170** ± 0.0283 | 0.7477 | **0.0573** | 0% |
| `prob_sharp` | probabilistic | 32 | 10.7 ± 0.2 | **0.7083** ± 0.0321 | 0.7279 | **0.0664** | 0% |

### Synthetic (Depth 32, Width 32, Max Budget = 32 Sweeps)

| Policy | Kind | Budget | Mean Sweeps | BP Cosine to BP | Residual Norm | Work Efficiency (Cos/Sweeps) | Early Exit Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fixed_sync_b16` | fixed_sync | 16 | 16.0 ± 0.0 | **0.7792** ± 0.0158 | 0.5515 | **0.0487** | 0% |
| `fixed_sync_b32` | fixed_sync | 32 | 32.0 ± 0.0 | **0.8116** ± 0.0122 | 0.6207 | **0.0254** | 0% |
| `fixed_gs_b4` | fixed_gs | 4 | 4.0 ± 0.0 | **0.7642** ± 0.0181 | 0.5609 | **0.1910** | 0% |
| `fixed_gs_b8` | fixed_gs | 8 | 8.0 ± 0.0 | **0.7873** ± 0.0148 | 0.6659 | **0.0984** | 0% |
| `fixed_gs_b16` | fixed_gs | 16 | 16.0 ± 0.0 | **0.8324** ± 0.0112 | 0.8119 | **0.0520** | 0% |
| `fixed_gs_b32` | fixed_gs | 32 | 32.0 ± 0.0 | **0.9519** ± 0.0031 | 0.9399 | **0.0297** | 0% |
| `dual_exit_eps1e-3` | dual_exit | 32 | 32.0 ± 0.0 | **0.9519** ± 0.0031 | 0.9399 | **0.0297** | 0% |
| `dual_exit_eps3e-4` | dual_exit | 32 | 32.0 ± 0.0 | **0.9519** ± 0.0031 | 0.9399 | **0.0297** | 0% |
| `dual_exit_eps1e-4` | dual_exit | 32 | 32.0 ± 0.0 | **0.9519** ± 0.0031 | 0.9399 | **0.0297** | 0% |
| `dual_exit_eps3e-5` | dual_exit | 32 | 32.0 ± 0.0 | **0.9519** ± 0.0031 | 0.9399 | **0.0297** | 0% |
| `dual_exit_eps1e-5` | dual_exit | 32 | 32.0 ± 0.0 | **0.9519** ± 0.0031 | 0.9399 | **0.0297** | 0% |
| `gradient_wave_th1e-5` | gradient | 32 | 17.0 ± 0.0 | **0.9531** ± 0.0088 | 0.9893 | **0.0561** | 0% |
| `gradient_wave_th3e-5` | gradient | 32 | 2.8 ± 0.6 | **0.8103** ± 0.0072 | 0.7331 | **0.3033** | 100% |
| `gradient_wave_th1e-4` | gradient | 32 | 0.3 ± 0.1 | **0.7558** ± 0.0166 | 0.4843 | **2.6133** | 100% |
| `prob_mild` | probabilistic | 32 | 12.5 ± 0.4 | **0.7910** ± 0.0122 | 0.6904 | **0.0633** | 0% |
| `prob_sharp` | probabilistic | 32 | 10.7 ± 0.2 | **0.7859** ± 0.0133 | 0.6701 | **0.0737** | 0% |

## 3. Critical Audit & Gate 3 Retraction

1. **Retraction of "47% Compute Reduction" [retracted / artifact-suspect]**:
   - The reported "17 sweeps vs 32 sweeps" reflects *semantic layer-update mask events*, NOT physical FLOP savings.
   - In the simulation implementation (`run_self_timed_inference`), each round executes `grad_free_all(free, duals)` across **all 31 layers** to evaluate local norms, and `compiled_gs_sweep` computes autodiff gradients for all layers before applying `jnp.where(mask[i], ...)`.
   - Consequently, actual physical FLOPs and wall-clock time were 100% full-depth. No physical computational saving was realized.
   - **Gate 3 Status**: **RETRACTED / UNPROVEN**. A self-timed triggering law that saves physical compute requires sparse / event-driven execution kernels that genuinely skip backward graph evaluation for quiescent layers.

2. **Instrumented Dual Motion & Inactive `dual_exit` Arm (N6) [established]**:
   - Every `dual_exit_*` policy in the table is byte-identical to `fixed_gs_b32` (sweeps = 32.0, cosine = 0.9440, early exit rate = 0%).
   - **Physical Instrumentation of $\|\Delta \lambda_i\|_{\text{RMS}}$**:
     - Round 1: $\max_i \|\Delta \lambda_i\| = 0.001311$
     - Round 4: $\max_i \|\Delta \lambda_i\| = 0.003279$
     - Round 16: $\max_i \|\Delta \lambda_i\| = 0.001939$
     - Round 32: $\max_i \|\Delta \lambda_i\| = 0.001396$
   - Because the dual multipliers continuously drift with step $\|\Delta \lambda_i\| \ge 0.0014$, the tested thresholds $\epsilon \in \{10^{-3}, 3\times 10^{-4}, 10^{-4}, 3\times 10^{-5}, 10^{-5}\}$ were mathematically unreachable within 32 rounds.
   - **Status**: The five `dual_exit_*` rows are dead stopping code and are consolidated as redundant executions of `fixed_gs_b32`.

3. **Exclusion of Degenerate Do-Nothing Baseline (N6) [established]**:
   - `gradient_wave_th1e-4` was reported with an inflated "work efficiency" of 1.66-2.61 because it executed only $0.3-0.4$ sweeps.
   - **Trivial Baseline Control**: Evaluating the unrelaxed initial feedforward state ($t=0$, 0 sweeps) yields a reference BP cosine of **$0.6632$**.
   - `gradient_wave_th1e-4` achieves cosine **$0.6736$** (differing from the unrelaxed state by $< 0.01$, inside the noise floor). It is a do-nothing policy where nearly all layers quiesce immediately.
   - **Protocol Rule (N6)**: A **minimum work floor** of $W \ge 1.0$ full sweep (31 layer updates) is established. Any policy sweeping $< 1.0$ passes is tagged **`[degenerate]`** and disqualified from Pareto ranking. Work efficiency ($\text{Cosine}/\text{Sweeps}$) is prohibited when sweeps $< 1.0$.

