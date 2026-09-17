# Experiment 1.2: Dual Freshness vs Activity Freshness Ablation

**Question:** Which neighbor reads are most sensitive to staleness? Does credit degrade from stale activities or stale duals?
**Protocol:** Depth 32, Width 32, staleness delay $\tau = 31$ (1 sweep), 64 sweeps, matched work, 3 seeds (0, 1, 2).

## Summary Results Table

| Dataset | Condition | Description | BP Cosine (mean ± SD) | Early Layer Cosine | Residual Norm |
| :--- | :---: | :--- | :---: | :---: | :---: |
| synthetic | A | Baseline (all fresh) | 0.9704 ± 0.0033 | 0.9283 ± 0.0184 | 0.1735 |
| synthetic | B | h_{i-1} stale (forward activity) | 0.9780 ± 0.0031 | 0.9472 ± 0.0130 | 0.2129 |
| synthetic | C | h_{i+1} stale (backward activity) | 0.9638 ± 0.0029 | 0.9201 ± 0.0106 | 0.2014 |
| synthetic | D | lambda_i stale (own dual) | -0.0520 ± 0.0771 | 0.0179 ± 0.0716 | 50.1631 |
| synthetic | E | lambda_{i+1} stale (downstream dual) | 0.9288 ± 0.0134 | 0.9665 ± 0.0035 | 0.2464 |
| synthetic | F | h_{i-1}, h_{i+1} stale (both activities) | 0.9513 ± 0.0223 | 0.9231 ± 0.0233 | 0.4052 |
| synthetic | G | All stale (activities + duals) | 0.0008 ± 0.0239 | 0.3103 ± 0.1702 | 18544.0362 |
| fashion_mnist | A | Baseline (all fresh) | 0.9648 ± 0.0067 | 0.9408 ± 0.0090 | 0.1630 |
| fashion_mnist | B | h_{i-1} stale (forward activity) | 0.9754 ± 0.0044 | 0.9541 ± 0.0027 | 0.2002 |
| fashion_mnist | C | h_{i+1} stale (backward activity) | 0.9505 ± 0.0057 | 0.9326 ± 0.0027 | 0.1899 |
| fashion_mnist | D | lambda_i stale (own dual) | -0.0941 ± 0.0771 | 0.0249 ± 0.1152 | 46.0185 |
| fashion_mnist | E | lambda_{i+1} stale (downstream dual) | 0.8744 ± 0.0563 | 0.9701 ± 0.0037 | 0.2327 |
| fashion_mnist | F | h_{i-1}, h_{i+1} stale (both activities) | 0.9270 ± 0.0382 | 0.9257 ± 0.0265 | 0.3909 |
| fashion_mnist | G | All stale (activities + duals) | 0.0081 ± 0.0289 | 0.3059 ± 0.1700 | 18346.7200 |

## Analysis & Interpretation

### 1. Qualification of Condition D (Non-Realizable Fault Mode)
- **Condition D (Stale Own Dual $\lambda_i$)** artificially forces a layer to read a delayed copy of its own dual multiplier ($\tau = 31$).
- **Physical Reality:** In any real neuromorphic hardware, distributed actor, or multi-threaded implementation, a layer owns its dual state locally in registers/SRAM. There is zero inter-layer interconnect delay for a node reading its own state. Therefore, Condition D is **unphysical and non-realizable** in real systems.
- **Scientific Value:** Condition D serves strictly as a mathematical contrast: it proves that the local AL multiplier is an active feedback integrator whose closed-loop pole escapes the unit circle under internal delay. It does not constrain hardware design, because no sensible design would delay local state.

### 2. Binding Cross-Layer Constraints (The Real Engineering Content)
The real architectural question is: *which inter-node network delays degrade credit propagation?*
- **Forward Activity Staleness (B):** $h_{i-1}$ stale by $\tau=31$ has **negligible effect** ($\cos = 0.9754$ vs $0.9648$ fresh).
- **Backward Activity Staleness (C):** $h_{i+1}$ stale by $\tau=31$ has **minor effect** ($\cos = 0.9505$).
- **Both Activities Stale (F):** Concurrently stale forward and backward activities preserve high credit alignment ($\cos = 0.9270$).
- **Downstream Dual Staleness (E):** Stale downstream credit $\lambda_{i+1}$ causes modest degradation ($\cos = 0.8744$), making it the *most sensitive realizable cross-layer delay*, but still fully stable.

### 3. Engineering Conclusion
Real cross-layer asynchronous communication buffers carrying activities $h$ and credit $\lambda_{i+1}$ are thoroughly delay-tolerant at $\tau = 31$ sweeps. System designers need only ensure that local dual accumulators are updated on-node.
