# Phase 0 Baseline Validation & Measurement Ontology

**Date:** 2026-09-17  
**Repository:** `pc-alm`  
**Status:** All baseline checks verified.

---

## 1. Reproduction & Verification Benchmarks

### 1.1 Test Suite Status
- Executed `uv run --extra test pytest`: **39 passed in 18.34s** across:
  - `tests/test_async_analysis.py`
  - `tests/test_async_inference.py`
  - `tests/test_async_metrics.py`
  - `tests/test_formulation.py`
  - `tests/test_inputs.py`

### 1.2 Smoke Tests
1. **End-to-end sync training smoke test** (`configs/smoke.yaml`):
   - Command: `python train.py --config configs/smoke.yaml`
   - Result: Successful single-epoch run on synthetic data (`grad_cos_to_bp = 0.8492`, `final_train_acc = 0.1094`, `final_test_acc = 0.0938`).
2. **Async dynamics smoke test** (`configs/async_smoke.yaml`):
   - Command: `python scripts/run_async_dynamics.py --config configs/async_smoke.yaml`
   - Result: Successful run across `sync`, `random_coordinate`, `random_block`, `heterogeneous_rates`, `bounded_staleness`, and `fully_async_local`. All diagnostics and credit front exponents evaluated without divergence.

### 1.3 Trajectory Equivalence (Depth 6, Width 6)
- Verified equivalence between synchronous reference inference (`run_sync_reference_state`) and async inference running full-block updates (`random_block` with `block_size = depth - 1`, no staleness) at budget = 4, inner steps = 2:
  - **Activity tensor absolute error:** $\le 1.19 \times 10^{-7}$
  - **Dual multiplier tensor absolute error:** $\le 2.09 \times 10^{-7}$
- Confirmed deterministic replay across seeds.

### 1.4 Finite Differences Gradient Verification
- Analytic backpropagation weight gradients computed via `pcalm.async_metrics.bp_weight_gradients` were verified against central finite differences in `float64` across all layers:
  $$\Delta W_{ij}^{\text{num}} = \frac{\mathcal{L}(W + \epsilon e_{ij}) - \mathcal{L}(W - \epsilon e_{ij})}{2\epsilon}, \quad \epsilon = 10^{-6}$$
  - **Layer 0 max absolute difference:** $6.74 \times 10^{-11}$
  - **Layer 1 max absolute difference:** $7.72 \times 10^{-11}$
  - **Layer 2 max absolute difference:** $9.17 \times 10^{-11}$
  - **Layer 3 max absolute difference:** $5.80 \times 10^{-11}$
- Conclusion: Backpropagation gradients and reference alignments are mathematically exact.

---

## 2. Measurement Ontology

To prevent spurious conclusions (such as confusing norm collapse or schedule artifacts with true algorithmic superiority), all subsequent experiments will interpret metrics against the following ontology:

| Metric | Mathematical Definition | Trivial Improvement Mode (False Positive) | Alternative Explanation / Confound |
| :--- | :--- | :--- | :--- |
| `grad_cos_to_bp` | $\frac{\langle \nabla_W \mathcal{E}_{\text{AL}}, \nabla_W \mathcal{L}_{\text{BP}} \rangle}{\|\nabla_W \mathcal{E}_{\text{AL}}\| \|\nabla_W \mathcal{L}_{\text{BP}}\|}$ | Gradient norm collapse in early layers where remaining active layers happen to align with output gradient. | Higher cosine may reflect Gauss-Seidel sequential pass rather than asynchronous dynamics. |
| `early_grad_norm_ratio` | $\min_{l < L/2} \frac{\|g_l\|}{\|g_{L-1}\|}$ | Output layer gradient norm drops due to loss plateau, inflating the ratio without improving early credit. | Early layers receiving spurious noise amplification rather than true credit. |
| `t_90` | Sweeps / work units to reach $0.90 \times \text{final\_cos}$ | Rapid early alignment followed by premature plateau at poor asymptotic cosine. | Schedule prioritizes output-adjacent layers first, showing artificial early front speed. |
| `residual_mean` / `total_residual_norm` | $\frac{1}{L}\sum_{l=1}^{L-1} \|h_l - \sigma(W_l h_{l-1})\|_2$ | Activations collapsing to zero ($h_l \to 0$), trivializing residual satisfaction. | Stale duals failing to penalize constraint violations, masking instability. |
| `dual_norm` | $\sum_{l=1}^{L-1} \|\lambda_l\|_2$ | Multipliers staying near zero due to low $\alpha$ or dead units, signaling absence of constraint pressure. | Divergence / dual explosion under moving weights or unstable step size. |
| `final_test_acc` | $\frac{1}{N_{\text{test}}} \sum_{i} \mathbb{I}(\hat{y}_i = y_i)$ | Overfitting to train set, or test set leakage. | More total FLOPs/work executed per wall-clock second rather than superior learning rule. |
| `total_layer_update_work` | $\sum_t \sum_i \mathbb{I}(\text{layer } i \text{ updated at } t)$ | Counting masked updates while full-graph compute runs. | Under-counting coordinate events or ignoring dual update overhead. |
| `critical_path_steps` | $\max_{\text{path}} \sum_{e \in \text{path}} \text{cost}(e)$ | Assuming zero-latency cross-layer communication. | Confusing serial Gauss-Seidel sweeps with parallel Jacobi clock cycles. |

> **Prohibited Metric**: `parallel_work = work * depth` has no physical units or standard meaning and is **permanently prohibited**.

---

## 3. Protocol Rules for Subsequent Phases

1. **Two-Axis Work & Latency Accounting:** All schedule comparisons must report both `total_layer_update_work` and `critical_path_steps`.
2. **Canonical Protocol Precedence:** All official benchmark comparisons must cite and follow `PROTOCOLS.md` (Protocol F0).
3. **Norm Sanity Checks:** Any reported increase in `grad_cos_to_bp` must be accompanied by non-collapsed `dual_norm` and `early_grad_norm_ratio`.
4. **Seed Replication & Significance:** $\ge 5$ random seeds with 95% bootstrap confidence intervals. Differences $< 1.5\sigma$ are labeled `[not distinguishable from null]`.

---

## 4. Formal Confidence Tag Ladder (N4)

Every empirical and theoretical claim across reports and worklogs must carry an explicit confidence tag adhering to the following strict criteria:

| Confidence Tag | Operational Definition & Acceptance Threshold | Demotion / Revocation Trigger |
| :--- | :--- | :--- |
| **`[established]`** | Independently reproduced across **$\ge 2$ scales or depths** (e.g. Depth 16 and 32, or full dataset), with 95% bootstrap confidence intervals excluding the null hypothesis ($Z \ge 2.0\sigma$), and negative controls excluding trivial confounding mechanisms. | Single-configuration result, unisolated mechanism, or CI overlapping null. |
| **`[suggestive]`** | Evaluated on a **single scale/depth/dataset**, directionally consistent across random seeds with separation from baseline ($Z \ge 1.5\sigma$), but lacking multi-scale replication or complete mechanism isolation. | $Z < 1.5\sigma$ relative to binomial noise floor or variance. |
| **`[not distinguishable from null]`** | Observed effect size is **$< 1.5\sigma$** relative to empirical standard error or test set binomial noise floor. | Cannot be claimed as an algorithmic win or difference. |
| **`[artifact-suspect]`** | Metric definition, measurement apparatus, or simulated counter was found to confound, inflate, or fabricate the observed quantity (e.g. counting masked FLOPs or multiplying work by depth). | Requires immediate retraction of associated claims. |
| **`[retracted]`** | Demonstrated to be empirically false, mathematically invalid, physically non-realizable, or refuted by controlled replication. | Permanent revocation. |
| **`[conjecture / proof sketch]`** | Formal mathematical statement derived analytically under stated assumptions, but pending full discrete-time nonlinear proof or empirical boundary mapping. | Contradiction with established bounds or simulations. |
| **`[superseded]`** | Artifact produced under an earlier, non-canonical, or under-powered exploratory protocol that has been replaced by Canonical Protocol F0. | Informational only; cannot be cited as current benchmark. |

