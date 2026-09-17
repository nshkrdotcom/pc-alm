# Evaluation Protocols Specification

This document establishes the canonical protocols, measurement definitions, and statistical standards for evaluating optimization and inference algorithms in `pc-alm`. All empirical reports must explicitly cite the protocol under which they operated.

---

## 1. Canonical Benchmark Protocol (Protocol F0)

To resolve sample-size artifacts, baseline strength inconsistencies (N1), and test-set binomial noise (N3), all official benchmark evaluations must conform to Protocol F0:

| Parameter | Canonical Value | Rationale |
| :--- | :---: | :--- |
| **Dataset** | Fashion-MNIST | Standard non-trivial image classification benchmark |
| **Train Split** | 60,000 samples | Full canonical training set (prevents under-trained BP artifacts) |
| **Test Split** | 10,000 samples | Binomial standard error $\le 0.46\%$ at $p \approx 0.70$ (enables sub-1% resolution) |
| **Batch Size** | 64 | Matched across all methods |
| **Epoch Count** | 10 epochs | 9,370 minibatches per training run |
| **Architecture** | Depth $L=32$, Width $N=32$ | 31 hidden layers, 784 input, 10 output classes |
| **Activation** | ReLU | Standard nonlinear activation |
| **Weight Optimizer** | Adam (`lr = 0.001`, $\beta_1=0.9, \beta_2=0.999$) | Matched across all methods |
| **State LR ($\eta_h$)** | 0.057390 | Calibrated from `configs/eta_best_by_cell.csv` |
| **AL Multipliers** | $\rho = 1.0, \alpha = 1.0$ | Standard augmented Lagrangian operating point |
| **Seed Replications** | $\ge 5$ seeds ($S \in \{0, 1, 2, 3, 4\}$) | Minimum sample size for bootstrap confidence intervals |

### Statistical Standards:
- Every headline metric must report: **Mean ± SD** and **95% Bootstrap Confidence Intervals** (10,000 resamples).
- For any comparative claim between method $A$ and method $B$:
  $$\Delta = \mu_A - \mu_B, \quad \sigma_{\Delta} = \sqrt{\frac{\sigma_A^2}{N_A} + \frac{\sigma_B^2}{N_B}}, \quad Z = \frac{|\Delta|}{\sigma_{\Delta}}$$
- If $Z < 1.5\sigma$, the difference must be explicitly labeled `[not distinguishable from null]`.

### Work & Latency Accounting Standards:
Every schedule must be evaluated on the two physically meaningful axes:
1. `total_layer_update_work`: Total coordinate primal updates executed across all layers ($W = \sum_t \sum_i \mathbb{I}(\text{layer } i \text{ updated at } t)$). For full sweeps: $W = B \cdot (L-1)$. Units: *layer updates*.
2. `critical_path_steps`: Length of longest serial dependency chain ($T_{\text{crit}}$).
   - Jacobi with $P \ge L-1$ parallel processors: $T_{\text{crit}} = B$.
   - Serial Gauss-Seidel or sequential asynchrony: $T_{\text{crit}} = B \cdot (L-1)$.
   Units: *sequential clock cycles / time steps*.
3. **Prohibited Metric**: `parallel_work = W * T` is a fabricated non-physical quantity with no units and is **permanently prohibited**.

---

## 2. Superseded Exploratory Protocols

The following protocols were used in earlier exploratory phases and are explicitly marked `[superseded]`:

### Protocol A (Phase 2, Exp 2.1)
- **Split:** 4,096 train / 1,024 test samples.
- **Epochs:** 5 epochs (320 minibatches).
- **Seeds:** 3 seeds ($S \in \{0, 1, 2\}$).
- **Reference BP Accuracy:** $71.16\% \pm 0.94\%$.
- **Status:** `[superseded by Protocol F0]`. Not comparable to Protocol B or F0 due to under-trained sample limit.

### Protocol B (Phase 5, Exp R1 Budget Sweep)
- **Split:** 2,048 train / 512 test samples.
- **Epochs:** 4 epochs (128 minibatches).
- **Seeds:** 3 seeds ($S \in \{0, 1, 2\}$).
- **Reference BP Accuracy:** $53.45\% \pm 5.48\%$.
- **Status:** `[superseded by Protocol F0]`. Test sample size ($N=512$) has binomial noise floor $SE \approx 2.2\%$, making sub-point comparisons statistically unresolvable.
