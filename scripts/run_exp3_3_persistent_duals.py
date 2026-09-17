"""Phase 3 Experiment 3.3: Persistent Duals Across Minibatches

Tests whether retaining dual variables across minibatches with leakage parameter gamma:
  lambda_{init} = (1 - gamma) * lambda_{final_previous_batch}
eliminates the per-batch reset barrier and provides credit-space variance reduction.

Compares:
- gamma = 1.0: Standard PC-ALM (dual reset to 0 at every batch)
- gamma = 0.1: Fast leak
- gamma = 0.01: Slow leak
- gamma = 0.0: Full persistence

Protocol:
- Fashion-MNIST (4,096 train / 1,024 test samples), Batch Size 64, Depth 32, Width 32
- Reverse Gauss-Seidel (sync_gs) inference with budget = 8 sweeps (low-budget test where warm-starting matters most)
- 3 random seeds (0, 1, 2), 5 epochs each.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from pcalm.data import load_dataset
from pcalm.inference import (
    al_energy_shifted,
    bp_loss,
    constraint_residuals,
    free_init,
    zero_duals_like,
)
from pcalm.metrics import mse_ce_accuracy, tree_cos
from pcalm.model import (
    activation_fn,
    block_pred,
    init_params,
    logits,
    model_scales,
    skip_mask,
)
from pcalm.optim import adam_apply, adam_init


def make_persistent_step_fn(scales, skips, phi, state_lr: float, rho: float, alpha: float, budget: int, learning_rate: float):
    n_layers = len(skips)

    def run_sync_gs_with_init_duals(params, x, y, init_duals):
        free = tuple(free_init(params, scales, skips, x, phi))
        n = len(free)
        effective_lr = state_lr * x.shape[0]

        def single_sweep(carry, _):
            f_curr, d_curr = carry
            for i in range(n - 1, -1, -1):
                def energy_i(z_i):
                    f = tuple(z_i if j == i else f_curr[j] for j in range(n))
                    return al_energy_shifted(params, scales, skips, x, y, f, d_curr, rho, phi)
                g_i = jax.grad(energy_i)(f_curr[i])
                z_new = f_curr[i] - effective_lr * g_i
                f_curr = tuple(z_new if j == i else f_curr[j] for j in range(n))

                z_prev = x if i == 0 else f_curr[i - 1]
                pred = block_pred(params[i], scales[i], skips[i], z_prev, phi, is_first=(i == 0))
                r_i = z_new - pred
                lam_new = d_curr[i] + alpha * r_i
                d_curr = tuple(lam_new if j == i else d_curr[j] for j in range(n))
            return (f_curr, d_curr), None

        (final_free, final_duals), _ = jax.lax.scan(single_sweep, (free, init_duals), xs=None, length=budget)
        return final_free, final_duals

    @jax.jit
    def step_persistent(params, opt_state, x, y, running_duals, gamma: float):
        # Apply leak to running duals: (1 - gamma) * running_duals
        init_duals = tuple((1.0 - gamma) * d for d in running_duals)
        free, final_duals = run_sync_gs_with_init_duals(params, x, y, init_duals)
        free_stopped = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
        duals_stopped = jax.tree_util.tree_map(jax.lax.stop_gradient, final_duals)
        grads = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free_stopped, duals_stopped, rho, phi))(params)
        new_params, new_opt = adam_apply(params, grads, opt_state, learning_rate)
        return new_params, new_opt, duals_stopped

    return step_persistent


def main():
    root_dir = Path("results/exp3_3")
    root_dir.mkdir(parents=True, exist_ok=True)

    depth = 32
    width = 32
    budget = 8  # Low budget where warm-starting should show biggest leverage
    batch_size = 64
    epochs = 5
    seeds = [0, 1, 2]
    gammas = [1.0, 0.1, 0.01, 0.0]

    train_samples = 4096
    test_samples = 1024

    with Path("configs/eta_best_by_cell.csv").open() as f:
        calibration = list(csv.DictReader(f))
    fashion_cell = next(
        r for r in calibration
        if r["dataset"] == "fashion_mnist" and r["activation"] == "relu" and int(r["N"]) == 32 and int(r["L"]) == 32
    )
    state_lr = float(fashion_cell["eta_best_1_over_lambda_median"])
    rho = 1.0
    alpha = 1.0
    learning_rate = 0.001

    scales = model_scales(width, depth, 784)
    skips = skip_mask(depth)
    phi = activation_fn("relu")

    step_fn = make_persistent_step_fn(scales, skips, phi, state_lr, rho, alpha, budget, learning_rate)

    @jax.jit
    def eval_batch(params, x, y):
        return mse_ce_accuracy(logits(params, scales, skips, x, phi), y)

    def evaluate_model(params, X, Y):
        total_acc, total_loss, count = 0.0, 0.0, 0
        for i in range(0, len(X), batch_size):
            stop = min(i + batch_size, len(X))
            xb = jnp.asarray(X[i:stop])
            yb = jnp.asarray(Y[i:stop])
            mse, ce, acc = eval_batch(params, xb, yb)
            n = stop - i
            total_acc += float(acc) * n
            total_loss += float(ce) * n
            count += n
        return total_loss / count, total_acc / count

    all_runs = []

    for gamma in gammas:
        for seed in seeds:
            gamma_label = f"gamma_{gamma:g}"
            print(f"\n=== Persistent Duals: {gamma_label} | Seed: {seed} ===", flush=True)
            run_dir = root_dir / f"{gamma_label}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            train_x, train_y, test_x, test_y = load_dataset(
                "fashion_mnist",
                data_dir=Path("data"),
                train_subset=train_samples,
                test_subset=test_samples,
                seed=seed,
            )

            key = jax.random.PRNGKey(seed)
            params = init_params(key, depth=depth, width=width, input_dim=784, output_dim=10)
            opt_state = adam_init(params)

            # Initialize running duals to zero
            residuals0 = constraint_residuals(params, scales, skips, jnp.asarray(train_x[:batch_size]), free_init(params, scales, skips, jnp.asarray(train_x[:batch_size]), phi), phi)
            running_duals = tuple(zero_duals_like(residuals0))

            rows = []
            rng = np.random.default_rng(seed)
            t_start = time.perf_counter()
            step_count = 0

            te_loss, te_acc = evaluate_model(params, test_x, test_y)
            print(f"Epoch 0: Test Acc = {te_acc:.4f}", flush=True)

            for epoch in range(1, epochs + 1):
                indices = rng.permutation(len(train_x))
                for i in range(0, len(train_x), batch_size):
                    batch_idx = indices[i : i + batch_size]
                    if len(batch_idx) < batch_size:
                        continue
                    xb = jnp.asarray(train_x[batch_idx])
                    yb = jnp.asarray(train_y[batch_idx])

                    params, opt_state, running_duals = step_fn(params, opt_state, xb, yb, running_duals, float(gamma))
                    step_count += 1

                elapsed = time.perf_counter() - t_start
                tr_loss, tr_acc = evaluate_model(params, train_x, train_y)
                te_loss, te_acc = evaluate_model(params, test_x, test_y)
                dual_norm = float(jnp.sqrt(sum(jnp.sum(d * d) for d in running_duals)))

                rows.append({
                    "epoch": epoch,
                    "step": step_count,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "test_loss": te_loss,
                    "test_acc": te_acc,
                    "running_dual_norm": dual_norm,
                    "elapsed_seconds": elapsed,
                })
                print(f"Epoch {epoch}: Test Acc = {te_acc:.4f}, Train Acc = {tr_acc:.4f}, Dual Norm = {dual_norm:.4f} ({elapsed:.1f}s)", flush=True)

            with (run_dir / "metrics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            all_runs.append({
                "gamma": gamma,
                "label": gamma_label,
                "seed": seed,
                "final_train_acc": rows[-1]["train_acc"],
                "final_test_acc": rows[-1]["test_acc"],
                "final_dual_norm": rows[-1]["running_dual_norm"],
                "total_time_s": rows[-1]["elapsed_seconds"],
            })

    # Summary
    grouped = {}
    for r in all_runs:
        k = r["gamma"]
        grouped.setdefault(k, []).append(r)

    summary_rows = []
    for gamma, grp in sorted(grouped.items(), reverse=True):
        accs = [g["final_test_acc"] for g in grp]
        tr_accs = [g["final_train_acc"] for g in grp]
        duals = [g["final_dual_norm"] for g in grp]
        summary_rows.append({
            "gamma": gamma,
            "description": "Standard Reset" if gamma == 1.0 else f"Persistent (leak={gamma:g})",
            "test_acc_mean": float(np.mean(accs)),
            "test_acc_std": float(np.std(accs)),
            "train_acc_mean": float(np.mean(tr_accs)),
            "dual_norm_mean": float(np.mean(duals)),
        })

    with (root_dir / "summary_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Markdown Report
    lines = [
        "# Phase 3 Experiment 3.3: Persistent Duals Across Minibatches Report",
        "",
        "**Question:** Does carrying dual multipliers across minibatches (with leak gamma) improve training at low inference budgets (budget = 8 sweeps)?",
        "**Protocol:** Depth 32, Width 32, Fashion-MNIST, Batch Size 64, 5 Epochs, Budget = 8 sweeps, 3 seeds (0, 1, 2).",
        "",
        "## Summary Results Table",
        "",
        "| Leak Parameter (gamma) | Regime | Test Accuracy (mean ± SD) | Train Accuracy | Final Running Dual Norm |",
        "| :---: | :--- | :---: | :---: | :---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['gamma']}` | {row['description']} | "
            f"**{row['test_acc_mean'] * 100:.2f}% ± {row['test_acc_std'] * 100:.2f}%** | "
            f"{row['train_acc_mean'] * 100:.2f}% | {row['dual_norm_mean']:.4f} |"
        )

    lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Full Persistence (gamma = 0) vs Leaky Persistence:**",
        "   - Carrying duals across batches removes the global per-batch reset barrier.",
        "   - Monitor running dual norm to detect whether duals remain bounded or drift under sample changes.",
    ])

    report_path = root_dir / "PERSISTENT_DUALS_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
