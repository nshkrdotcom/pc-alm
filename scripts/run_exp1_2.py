"""Experiment 1.2: Dual Freshness vs Activity Freshness Ablation

Systematically isolates the sensitivity of credit propagation to staleness in:
Condition A (baseline): h_{i-1} fresh, h_{i+1} fresh, lambda_i fresh, lambda_{i+1} fresh
Condition B: h_{i-1} stale, h_{i+1} fresh, lambda_i fresh, lambda_{i+1} fresh (forward activity stale)
Condition C: h_{i-1} fresh, h_{i+1} stale, lambda_i fresh, lambda_{i+1} fresh (backward activity stale)
Condition D: h_{i-1} fresh, h_{i+1} fresh, lambda_i stale, lambda_{i+1} fresh (own dual stale)
Condition E: h_{i-1} fresh, h_{i+1} fresh, lambda_i fresh, lambda_{i+1} stale (downstream dual stale)
Condition F: h_{i-1}, h_{i+1} stale, lambda_i, lambda_{i+1} fresh (both activities stale)
Condition G: all stale (standard bounded staleness tau=31)

Run across 3 seeds on Synthetic and Fashion-MNIST at depth 32, matched work (64 sweeps).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from pcalm.async_metrics import bp_weight_gradients, weight_gradient_alignment
from pcalm.data import load_dataset
from pcalm.inference import (
    al_energy_shifted,
    constraint_residuals,
    free_init,
    zero_duals_like,
)
from pcalm.model import activation_fn, block_pred, init_params, model_scales, skip_mask


def run_selective_staleness(
    params,
    scales,
    skips,
    x,
    y,
    grad_free,
    *,
    condition: str,
    tau: int,
    sweeps: int,
    state_lr: float,
    rho: float,
    alpha: float,
    phi,
    seed: int,
):
    n_layers = len(params) - 1
    free = list(free_init(params, scales, skips, x, phi))
    residuals0 = constraint_residuals(params, scales, skips, x, free, phi)
    duals = list(zero_duals_like(residuals0))
    effective_lr = state_lr * x.shape[0]

    # History tracks (free, duals)
    history: deque[tuple[list[jax.Array], list[jax.Array]]] = deque(maxlen=tau + 1)
    history.append(([jnp.array(f) for f in free], [jnp.array(d) for d in duals]))

    rng = np.random.default_rng(seed)
    total_events = sweeps * n_layers

    for step in range(total_events):
        i = int(rng.integers(0, n_layers))

        stale_free, stale_duals = history[0]

        # Construct local read view for layer i
        read_free = list(free)
        read_duals = list(duals)

        if condition == "A":  # all fresh
            pass
        elif condition == "B":  # h_{i-1} stale
            if i > 0:
                read_free[i - 1] = stale_free[i - 1]
        elif condition == "C":  # h_{i+1} stale
            if i < n_layers - 1:
                read_free[i + 1] = stale_free[i + 1]
        elif condition == "D":  # lambda_i stale
            read_duals[i] = stale_duals[i]
        elif condition == "E":  # lambda_{i+1} stale
            if i < n_layers - 1:
                read_duals[i + 1] = stale_duals[i + 1]
        elif condition == "F":  # both activities stale
            if i > 0:
                read_free[i - 1] = stale_free[i - 1]
            if i < n_layers - 1:
                read_free[i + 1] = stale_free[i + 1]
        elif condition == "G":  # all stale
            if i > 0:
                read_free[i - 1] = stale_free[i - 1]
            if i < n_layers - 1:
                read_free[i + 1] = stale_free[i + 1]
            read_duals[i] = stale_duals[i]
            if i < n_layers - 1:
                read_duals[i + 1] = stale_duals[i + 1]

        # Always use own activity
        read_free[i] = free[i]

        grads = grad_free(tuple(read_free), tuple(read_duals))
        free[i] = free[i] - effective_lr * grads[i]

        # Local dual update
        z_prev = x if i == 0 else read_free[i - 1]
        pred = block_pred(params[i], scales[i], skips[i], z_prev, phi, is_first=(i == 0))
        local_res = free[i] - pred
        duals[i] = duals[i] + alpha * local_res

        history.append(([jnp.array(f) for f in free], [jnp.array(d) for d in duals]))

    return tuple(free), tuple(duals)


def main():
    root_dir = Path("results/exp1_2")
    root_dir.mkdir(parents=True, exist_ok=True)

    depth = 32
    width = 32
    n_free = depth - 1
    tau = n_free  # 1 full sweep delay = 31 events
    sweeps = 64
    seeds = [0, 1, 2]
    conditions = ["A", "B", "C", "D", "E", "F", "G"]
    datasets = ["synthetic", "fashion_mnist"]

    with Path("configs/eta_by_depth.csv").open() as f:
        synthetic_rates = {int(r["L"]): float(r["state_lr"]) for r in csv.DictReader(f)}

    with Path("configs/eta_best_by_cell.csv").open() as f:
        calibration = list(csv.DictReader(f))
    fashion_cell = next(
        r for r in calibration
        if r["dataset"] == "fashion_mnist" and r["activation"] == "relu" and int(r["N"]) == 32 and int(r["L"]) == 32
    )
    fashion_lr = float(fashion_cell["eta_best_1_over_lambda_median"])

    phi = activation_fn("relu")
    results = []

    for dset in datasets:
        for seed in seeds:
            key = jax.random.PRNGKey(seed)
            if dset == "synthetic":
                input_dim = 128
                output_dim = 10
                state_lr = synthetic_rates[depth]
                x = jax.random.normal(jax.random.PRNGKey(seed + 1), (32, input_dim))
                y = jax.nn.one_hot(jax.random.randint(jax.random.PRNGKey(seed + 2), (32,), 0, output_dim), output_dim)
            else:
                input_dim = 784
                output_dim = 10
                train_x, train_y, _, _ = load_dataset(
                    "fashion_mnist",
                    data_dir=Path("data"),
                    train_subset=256,
                    test_subset=64,
                    seed=seed,
                )
                x = jnp.asarray(train_x[:32])
                y = jnp.asarray(train_y[:32])

            scales = model_scales(width=width, depth=depth, input_dim=input_dim)
            skips = skip_mask(depth)
            params = init_params(key, depth=depth, width=width, input_dim=input_dim, output_dim=output_dim)
            bp_grads = bp_weight_gradients(params, scales, skips, x, y, phi)

            # JIT-compile gradient of AL energy for this cell
            grad_free = jax.jit(jax.grad(lambda f, d: al_energy_shifted(params, scales, skips, x, y, f, d, 1.0, phi), argnums=0))

            for cond in conditions:
                sched_seed = 101 + 1009 * seed
                final_free, final_duals = run_selective_staleness(
                    params, scales, skips, x, y, grad_free,
                    condition=cond, tau=tau, sweeps=sweeps,
                    state_lr=state_lr, rho=1.0, alpha=1.0, phi=phi,
                    seed=sched_seed,
                )
                cos, layer_cosines = weight_gradient_alignment(
                    params, scales, skips, x, y, final_free, final_duals, rho=1.0, phi=phi, bp_grads=bp_grads
                )
                residuals = constraint_residuals(params, scales, skips, x, final_free, phi)
                res_norm = float(np.sqrt(sum(float(jnp.sum(r * r)) for r in residuals)))
                dual_norm = float(np.sqrt(sum(float(jnp.sum(d * d)) for d in final_duals)))
                early_cos = float(np.mean(layer_cosines[:max(1, depth // 4)]))

                results.append({
                    "dataset": dset,
                    "seed": seed,
                    "condition": cond,
                    "bp_cosine": float(cos),
                    "early_cosine": early_cos,
                    "residual_norm": res_norm,
                    "dual_norm": dual_norm,
                })
                print(f"[{dset}] seed {seed} Cond {cond}: cos={float(cos):.4f}, early_cos={early_cos:.4f}, res={res_norm:.4f}", flush=True)

    # Save raw
    (root_dir / "raw_results.json").write_text(json.dumps(results, indent=2))

    # Summary
    cond_descriptions = {
        "A": "Baseline (all fresh)",
        "B": "h_{i-1} stale (forward activity)",
        "C": "h_{i+1} stale (backward activity)",
        "D": "lambda_i stale (own dual)",
        "E": "lambda_{i+1} stale (downstream dual)",
        "F": "h_{i-1}, h_{i+1} stale (both activities)",
        "G": "All stale (activities + duals)",
    }

    summary_rows = []
    for dset in datasets:
        for cond in conditions:
            group = [r for r in results if r["dataset"] == dset and r["condition"] == cond]
            cosines = [g["bp_cosine"] for g in group]
            early_cosines = [g["early_cosine"] for g in group]
            residuals = [g["residual_norm"] for g in group]
            summary_rows.append({
                "dataset": dset,
                "condition": cond,
                "description": cond_descriptions[cond],
                "bp_cosine_mean": float(np.mean(cosines)),
                "bp_cosine_std": float(np.std(cosines)),
                "early_cosine_mean": float(np.mean(early_cosines)),
                "early_cosine_std": float(np.std(early_cosines)),
                "residual_norm_mean": float(np.mean(residuals)),
            })

    # Summary CSV
    with (root_dir / "summary_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Markdown Report
    lines = [
        "# Experiment 1.2: Dual Freshness vs Activity Freshness Ablation",
        "",
        "**Question:** Which neighbor reads are most sensitive to staleness? Does credit degrade from stale activities or stale duals?",
        "**Protocol:** Depth 32, Width 32, staleness delay $\\tau = 31$ (1 sweep), 64 sweeps, matched work, 3 seeds (0, 1, 2).",
        "",
        "## Summary Results Table",
        "",
        "| Dataset | Condition | Description | BP Cosine (mean ± SD) | Early Layer Cosine | Residual Norm |",
        "| :--- | :---: | :--- | :---: | :---: | :---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {row['condition']} | {row['description']} | "
            f"{row['bp_cosine_mean']:.4f} ± {row['bp_cosine_std']:.4f} | "
            f"{row['early_cosine_mean']:.4f} ± {row['early_cosine_std']:.4f} | "
            f"{row['residual_norm_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## Analysis & Interpretation",
        "",
        "Compare the degradation relative to Condition A (all fresh):",
        "- **Forward Activity Staleness (B):** $h_{i-1}$ stale effects.",
        "- **Backward Activity Staleness (C):** $h_{i+1}$ stale effects.",
        "- **Own Dual Staleness (D):** $\\lambda_i$ stale effects.",
        "- **Downstream Dual Staleness (E):** $\\lambda_{i+1}$ stale effects.",
        "- **All Activities Stale (F) vs All Stale (G):** Relative contribution of activities vs duals.",
    ])

    report_path = root_dir / "DUAL_VS_ACTIVITY_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
