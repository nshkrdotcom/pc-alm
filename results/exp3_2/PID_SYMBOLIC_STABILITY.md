# Phase 3 Experiment 3.2: Symbolic Stability Analysis of PID D-Term

## 1. System Formulation

The composite error with proportional (P), integral (I), and derivative (D) action is:
$$e(t) = \rho r(t) + \lambda(t) + \delta \big(r(t) - r(t-1)\big)$$

Introducing the auxiliary memory state $p(t) = r(t-1)$, the coupled 3x3 discrete update matrix is:
$$x(t+1) = \begin{pmatrix} r(t+1) \\ \lambda(t+1) \\ p(t+1) \end{pmatrix} = M x(t)$$
where with dimensionless step size $k = \eta_h \sigma^2$:
$$M = \begin{pmatrix} 1 - k(\rho + \delta) & -k & k\delta \\ \alpha [1 - k(\rho + \delta)] & 1 - \alpha k & \alpha k \delta \\ 1 & 0 & 0 \end{pmatrix}$$

## 2. Characteristic Polynomial & Jury Stability

- Characteristic equation: $\det(zI - M) = z^3 + a_2 z^2 + a_1 z + a_0 = 0$
- At $\delta = 0$ (Standard PC-ALM): $P(z) = z \cdot [z^2 - (2 - k\rho - k\alpha)z + (1 - k\rho)]$, yielding the known bound $k(2\rho + \alpha) < 4 \implies \alpha < 2$ (at $k=1, \rho=1$).
- Jury stability criteria for 3rd-order system:
  1. $P(1) = \alpha k > 0$ (satisfied for all positive $\alpha, k$)
  2. $P(-1) = \alpha k + 4\delta k + 2k\rho - 4 < 0 \implies \alpha < \frac{4 - 2k\rho - 4k\delta}{k}$
  3. $|a_0| = |\delta k| < 1$
  4. $|1 - a_0^2| > |a_1 - a_0 a_2|$

## 3. Stability Boundary Sweep (k = 1.0, rho = 1.0)

| Derivative Term (delta) | Maximum Stable alpha | Limiting Jury Condition | Physical Effect |
| :---: | :---: | :--- | :--- |
| `+0.20` | 1.2000 | $P(-1) < 0$ | High-frequency Nyquist mode ($z=-1$) amplified; restricts $\alpha$ |
| `0.00` (Baseline) | **2.0000** | $P(-1) < 0$ | Standard PC-ALM bound ($\alpha < 2.0$) |
| `-0.20` | 2.8000 | $P(-1) < 0$ | Damps alternating sign oscillations at $z=-1$; raises $\alpha$ ceiling |
| `-0.25` (Optimal) | **3.0000** | Condition 4 boundary | **Maximum stable $\alpha = 3.00$ (50% expansion)** |

## 4. Conclusion & Scientific Decision

- In discrete time, forward finite-difference derivative terms $r(t) - r(t-1)$ act with gain $2$ at the Nyquist frequency $z = -1$.
- Consequently, positive derivative feedback $\delta > 0$ actually worsens discrete ringing at high step sizes ($k \approx 1$), reducing the allowable dual step $\alpha$.
- Conversely, a mild negative momentum / relaxation term $\delta \approx -0.25$ damps the discrete Nyquist mode, expanding the stable dual gain region from $\alpha < 2.0$ to $\alpha < 3.0$.
- However, in practice, our Experiment 1.1 and 2.1 results showed that **Gauss-Seidel dual interleaving** already eliminates dual explosion and restores full BP accuracy (70.93% vs 71.16%) without needing an extra hyperparameter $\delta$ or state buffer $p(t) = r(t-1)$. Thus, asynchronous Gauss-Seidel remains our primary architectural choice.
