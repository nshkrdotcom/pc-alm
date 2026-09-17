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

1. **Retraction of "47% Compute Reduction" [Artifact-Suspect / Retracted]**:
   - The reported "17 sweeps vs 32 sweeps" reflects *semantic layer-update mask events*, NOT physical FLOP savings.
   - In the simulation implementation (`run_self_timed_inference`), each round executes `grad_free_all(free, duals)` across **all 31 layers** to evaluate local norms, and `compiled_gs_sweep` computes autodiff gradients for all layers before applying `jnp.where(mask[i], ...)`.
   - Consequently, actual physical FLOPs and wall-clock time were 100% full-depth. No physical computational saving was realized.
   - **Gate 3 Status**: **RETRACTED / UNPROVEN**. A self-timed triggering law that saves physical compute requires sparse / event-driven execution kernels that genuinely skip backward graph evaluation for quiescent layers.

2. **Semantic Event Dynamics [Suggestive]**:
   - The finding that only 17 reverse layer sweeps are required for credit alignment once masked updates are applied is an algorithmic property of credit stabilization.
   - However, until mapped to hardware/kernels where quiescent layers truly bypass FLOP execution, this remains an unvalidated physical efficiency claim.
