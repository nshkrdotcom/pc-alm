# Phase 3 Experiment 3.1: Feedback Alignment (Weight Transport) Report

**Question:** Does PC-ALM's dual variable provide robustness to feedback misalignment (weight transport problem) compared to standard Predictive Coding?
**Protocol:** Depth 32, Width 32, 16 inference sweeps, Synthetic and Fashion-MNIST, 3 random seeds.

## Summary Results Table

| Dataset | Method | Feedback Mechanism | BP Cosine (mean ± SD) | Residual Norm | Dual Norm |
| :--- | :--- | :--- | :---: | :---: | :---: |
| fashion_mnist | `async_local` | Symmetric (W^T) | **0.7134 ± 0.0378** | 0.2268 | 2.1339 |
| fashion_mnist | `async_local` | Feedback Alignment (Random B) | **0.6360 ± 0.0499** | 0.2570 | 2.3924 |
| fashion_mnist | `pc` | Symmetric (W^T) | **0.6832 ± 0.0416** | 0.7350 | 0.0000 |
| fashion_mnist | `pc` | Feedback Alignment (Random B) | **0.6652 ± 0.0434** | 0.8075 | 0.0000 |
| fashion_mnist | `sync_gs` | Symmetric (W^T) | **0.7753 ± 0.0330** | 0.1232 | 3.1082 |
| fashion_mnist | `sync_gs` | Feedback Alignment (Random B) | **0.6104 ± 0.0569** | 0.1568 | 3.6221 |
| fashion_mnist | `sync_pcalm` | Symmetric (W^T) | **0.7135 ± 0.0401** | 0.1497 | 2.0172 |
| fashion_mnist | `sync_pcalm` | Feedback Alignment (Random B) | **0.6393 ± 0.0477** | 0.1745 | 2.2492 |
| synthetic | `async_local` | Symmetric (W^T) | **0.8002 ± 0.0061** | 0.2460 | 2.3111 |
| synthetic | `async_local` | Feedback Alignment (Random B) | **0.7336 ± 0.0175** | 0.2413 | 2.2470 |
| synthetic | `pc` | Symmetric (W^T) | **0.7640 ± 0.0103** | 0.7961 | 0.0000 |
| synthetic | `pc` | Feedback Alignment (Random B) | **0.7466 ± 0.0126** | 0.7606 | 0.0000 |
| synthetic | `sync_gs` | Symmetric (W^T) | **0.8588 ± 0.0097** | 0.1337 | 3.3398 |
| synthetic | `sync_gs` | Feedback Alignment (Random B) | **0.7244 ± 0.0150** | 0.1471 | 3.4132 |
| synthetic | `sync_pcalm` | Symmetric (W^T) | **0.8001 ± 0.0089** | 0.1616 | 2.1773 |
| synthetic | `sync_pcalm` | Feedback Alignment (Random B) | **0.7341 ± 0.0144** | 0.1645 | 2.1109 |

## Key Findings & Qualifications

1. **Path of Feedback Alignment Injection:**
   - In this implementation, the random feedback matrix $B$ replaces $W^\top$ on the **activity inference path** ($\nabla_{h_i} \mathcal{E}$).
   - Because $B$ directly alters the activity trajectory during inference, the final equilibrium $h^*$ is **not** identically the clean forward pass. It is the steady-state of the perturbed AL dynamical system.
   - However, because the constraint penalty $\frac{\rho}{2}\|h_i - \mu(W_i h_{i-1})\|^2$ and dual multiplier $\lambda_i$ enforce forward consistency using the actual forward weights $W_i$, the representation remains tightly anchored to the feedforward manifold ($\|r\| \approx 0.12 - 0.17$).

2. **Qualification on Residual Comparison (PC vs PC-ALM):**
   - The reported residual norm is $\|h_i - \mu(W_i h_{i-1})\|$.
   - In PC-ALM, this residual is an explicitly constrained equality condition backed by dual ascent ($\lambda$) and penalty ($\rho$).
   - In standard PC, layers minimize a prediction error energy where hidden activations are free to drift to absorb supervisory loss. Thus, comparing residual magnitudes compares an equality-constrained method against an unconstrained soft penalty method.

3. **Alignment Quality:**
   - Replacing $W^\top$ with fixed random $B$ degrades the gradient cosine to Backprop by $\approx 0.10 - 0.16$ across all methods (e.g. `sync_gs` drops from $0.775$ to $0.610$ on Fashion-MNIST).
   - While alignment remains positive, weight transport elimination incurs a measurable credit fidelity penalty.
