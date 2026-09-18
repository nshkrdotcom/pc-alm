#!/usr/bin/env python3
"""Forward Milestone F1: Depth Scaling Sweep (L in {16, 32, 64}) under Protocol F0.

Promotes empirical findings from [suggestive] to [established] on the Confidence
Ladder by evaluating depth scaling across >= 2 architectures:
1. bp: Standard Backpropagation baseline
2. sync_gs_pcalm: Reverse Gauss-Seidel at B=64
3. sync_pcalm: Jacobi at B=64 and B=128
4. sync_gs_forward_pcalm: Forward Gauss-Seidel at B=64

All runs use full Fashion-MNIST (60k/10k), 10 epochs, batch size 64, remainder dropping
(937 batches/epoch), epoch shuffling, and 5 seeds (0..4).
"""

import csv
import math
import sys
import time
from pathlib import Path
from typing import Any
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp

from pcalm.data import load_dataset
from pcalm.model import (
    init_params,
    model_scales,
    skip_mask,
    activation_fn,
    logits,
    block_pred,
)
from pcalm.metrics import mse_ce_accuracy
from pcalm.inference import (
    al_energy_shifted,
    constraint_residuals,
    free_init,
    zero_duals_like,
)
from pcalm.training import adam_init, adam_apply

def bootstrap_ci(data: list[float], num_resamples: int = 10000, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    if n <= 1:
        return float(data[0]), float(data[0])
    resamples = rng.choice(data, size=(num_resamples, n), replace=True)
    resample_means = np.mean(resamples, axis=1)
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(resample_means, alpha * 100))
    high = float(np.percentile(resample_means, (1.0 - alpha) * 100))
    return low, high

def main():
    root_dir = Path("results/f1_depth_sweep")
    root_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root_dir / "depth_sweep_metrics.csv"

    depths = [16, 64]  # L=32 is already canonically benchmarked, but can be added
    width = 32
    batch_size = 64
    epochs = 10
    seeds = [0, 1, 2, 3, 4]

    state_lr = 0.057390
    rho = 1.0
    alpha = 1.0
    learning_rate = 0.001

    phi = activation_fn("relu")

    all_rows = []
    if csv_path.exists():
        with csv_path.open() as f:
            all_rows = list(csv.DictReader(f))
            for r in all_rows:
                r["seed"] = int(r["seed"])
                r["depth"] = int(r["depth"])
                r["budget_sweeps"] = int(r["budget_sweeps"])
                r["test_acc"] = float(r["test_acc"])

    print("=================================================================", flush=True)
    print("  RUNNING FORWARD MILESTONE F1: DEPTH SCALING SWEEP L in {16, 64}", flush=True)
    print("=================================================================", flush=True)

    for depth in depths:
        n = depth - 1
        scales = model_scales(width, depth, 784)
        skips = skip_mask(depth)

        # 1. Inference operators
        def infer_sync(params, x, y, budget: int):
            free = tuple(free_init(params, scales, skips, x, phi))
            duals = tuple(zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi)))
            effective_lr = state_lr * x.shape[0]

            def outer(carry, _):
                f_curr, d_curr = carry
                def energy(f_):
                    return al_energy_shifted(params, scales, skips, x, y, f_, d_curr, rho, phi)
                grads = jax.grad(energy)(f_curr)
                f_next = tuple(z - effective_lr * g for z, g in zip(f_curr, grads))
                residuals = constraint_residuals(params, scales, skips, x, f_next, phi)
                d_next = tuple(lam + alpha * r for lam, r in zip(d_curr, residuals))
                return (f_next, d_next), None

            (final_free, final_duals), _ = jax.lax.scan(outer, (free, duals), xs=None, length=budget)
            return final_free, final_duals

        def infer_gs(params, x, y, budget: int):
            free = tuple(free_init(params, scales, skips, x, phi))
            duals = tuple(zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi)))
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

            (final_free, final_duals), _ = jax.lax.scan(single_sweep, (free, duals), xs=None, length=budget)
            return final_free, final_duals

        def infer_gs_fwd(params, x, y, budget: int):
            free = tuple(free_init(params, scales, skips, x, phi))
            duals = tuple(zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi)))
            effective_lr = state_lr * x.shape[0]

            def single_sweep(carry, _):
                f_curr, d_curr = carry
                for i in range(n):
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

            (final_free, final_duals), _ = jax.lax.scan(single_sweep, (free, duals), xs=None, length=budget)
            return final_free, final_duals

        def make_step_fn(infer_fn, budget: int):
            @jax.jit
            def step(params, opt_state, x, y):
                free, duals = infer_fn(params, x, y, budget)
                free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
                duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
                grads = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi))(params)
                return adam_apply(params, grads, opt_state, learning_rate)
            return step

        @jax.jit
        def step_bp(params, opt_state, x, y):
            def loss(p):
                pred = logits(p, scales, skips, x, phi)
                _, ce, _ = mse_ce_accuracy(pred, y)
                return ce
            grads = jax.grad(loss)(params)
            return adam_apply(params, grads, opt_state, learning_rate)

        def make_diag_fn(infer_fn, budget: int):
            @jax.jit
            def diag(params, x, y):
                free, duals = infer_fn(params, x, y, budget)
                free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
                duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
                residuals = constraint_residuals(params, scales, skips, x, free, phi)
                res_norm = jnp.sqrt(sum(jnp.sum(r * r) for r in residuals))
                dual_norm = jnp.sqrt(sum(jnp.sum(lam * lam) for lam in duals))
                return res_norm, dual_norm
            return diag

        @jax.jit
        def eval_batch(params, x, y):
            return mse_ce_accuracy(logits(params, scales, skips, x, phi), y)

        def evaluate_model(params, X, Y):
            X_dev = jnp.asarray(X)
            Y_dev = jnp.asarray(Y)
            total_acc, total_loss, count = 0.0, 0.0, 0
            chunk_size = 512
            for i in range(0, len(X), chunk_size):
                stop = min(i + chunk_size, len(X))
                xb = X_dev[i:stop]
                yb = Y_dev[i:stop]
                mse, ce, acc = eval_batch(params, xb, yb)
                n_samples = stop - i
                total_acc += float(acc) * n_samples
                total_loss += float(ce) * n_samples
                count += n_samples
            return total_loss / count, total_acc / count

        # Define configurations for this depth
        configs = [
            ("bp", 0, step_bp, None),
            ("sync_gs_pcalm", 64, make_step_fn(infer_gs, 64), make_diag_fn(infer_gs, 64)),
            ("sync_pcalm", 64, make_step_fn(infer_sync, 64), make_diag_fn(infer_sync, 64)),
            ("sync_pcalm", 128, make_step_fn(infer_sync, 128), make_diag_fn(infer_sync, 128)),
            ("sync_gs_forward_pcalm", 64, make_step_fn(infer_gs_fwd, 64), make_diag_fn(infer_gs_fwd, 64)),
        ]

        for method, budget, step_fn, diag_fn in configs:
            for seed in seeds:
                done = any(r["depth"] == depth and r["method"] == method and r["budget_sweeps"] == budget and r["seed"] == seed for r in all_rows)
                if done:
                    print(f"[CACHED] L={depth:2d} | {method:22s} | B={budget:3d} | Seed {seed}", flush=True)
                    continue

                print(f"[L={depth:2d} | {method:22s} | B={budget:3d} | S={seed}] Starting run...", flush=True)
                train_x, train_y, test_x, test_y = load_dataset(
                    "fashion_mnist", data_dir=Path("data"), train_subset=60000, test_subset=10000, seed=seed
                )
                params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width, input_dim=784, output_dim=10)
                opt_state = adam_init(params)

                n_train = len(train_x)
                num_batches = n_train // batch_size
                n_used = num_batches * batch_size
                rng = np.random.default_rng(seed * 1000 + depth * 100 + budget)

                # Preload training arrays to device
                train_x_gpu = jnp.asarray(train_x)
                train_y_gpu = jnp.asarray(train_y)

                t0 = time.perf_counter()
                for epoch in range(epochs):
                    perm = rng.permutation(n_train)[:n_used]
                    for b_idx in range(num_batches):
                        idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
                        xb, yb = train_x_gpu[idx], train_y_gpu[idx]
                        params, opt_state = step_fn(params, opt_state, xb, yb)

                elapsed = time.perf_counter() - t0

                tr_loss, tr_acc = evaluate_model(params, train_x[:2048], train_y[:2048])
                te_loss, te_acc = evaluate_model(params, test_x, test_y)

                if diag_fn is not None:
                    res, dual = diag_fn(params, train_x_gpu[:batch_size], train_y_gpu[:batch_size])
                    res_val, dual_val = float(res), float(dual)
                else:
                    res_val, dual_val = 0.0, 0.0

                layer_work = budget * n if method != "bp" else 0
                seq_depth = budget if method == "sync_pcalm" else (budget * n if method != "bp" else 0)

                row = {
                    "depth": depth,
                    "method": method,
                    "seed": seed,
                    "budget_sweeps": budget,
                    "total_layer_update_work": layer_work,
                    "critical_path_steps": seq_depth,
                    "train_acc_probe_2048": float(tr_acc),
                    "test_acc": float(te_acc),
                    "residual_norm": res_val,
                    "dual_norm": dual_val,
                    "wall_clock_sec": float(elapsed),
                    "shuffled": True,
                }
                all_rows.append(row)
                print(
                    f"[L={depth:2d} | {method:22s} | B={budget:3d} | S={seed}] "
                    f"TestAcc={te_acc*100:5.2f}% | WallClock={elapsed:.1f}s",
                    flush=True,
                )

                with csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(all_rows)

    # Statistical Aggregation
    grouped: dict[tuple[int, str, int], list[dict[str, Any]]] = {}
    for r in all_rows:
        grouped.setdefault((int(r["depth"]), r["method"], int(r["budget_sweeps"])), []).append(r)

    summary_rows = []
    for (d, m, b), g in grouped.items():
        accs = [float(x["test_acc"]) for x in g]
        m_mean = float(np.mean(accs))
        m_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        ci_low, ci_high = bootstrap_ci(accs)

        summary_rows.append({
            "depth": d,
            "method": m,
            "budget_sweeps": b,
            "total_layer_update_work": g[0]["total_layer_update_work"],
            "critical_path_steps": g[0]["critical_path_steps"],
            "test_acc_mean": m_mean,
            "test_acc_std": m_std,
            "test_acc_ci95_low": ci_low,
            "test_acc_ci95_high": ci_high,
            "residual_mean": float(np.mean([float(x["residual_norm"]) for x in g])),
            "dual_norm_mean": float(np.mean([float(x["dual_norm"]) for x in g])),
            "wall_clock_mean": float(np.mean([float(x["wall_clock_sec"]) for x in g])),
        })

    summary_csv = root_dir / "depth_sweep_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = [
        "# Milestone F1: Depth Scaling Sweep ($L \\in \\{16, 64\\}$) under Protocol F0",
        "",
        "## 1. Executive Summary & Confidence Ladder Promotion",
        "- Evaluates depth scaling across $L=16$ and $L=64$ under Protocol F0 (epoch shuffling, exact remainder dropping).",
        "- Tests whether the Reverse GS vs Forward GS directional gap holds, widens, or collapses with network depth.",
        "",
        "## 2. Cross-Depth Summary Table",
        "",
        "| Depth ($L$) | Method | Budget ($B$) | Total Work ($W$) | Critical Path ($T_{\\text{crit}}$) | Test Accuracy (mean ± SD) | 95% Bootstrap CI |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in sorted(summary_rows, key=lambda x: (x["depth"], x["budget_sweeps"], x["method"])):
        report.append(
            f"| {s['depth']} | `{s['method']}` | {s['budget_sweeps']} | {s['total_layer_update_work']} | {s['critical_path_steps']} | "
            f"**{s['test_acc_mean']*100:.2f}% ± {s['test_acc_std']*100:.2f}%** | "
            f"[{s['test_acc_ci95_low']*100:.2f}%, {s['test_acc_ci95_high']*100:.2f}%] |"
        )

    report_path = root_dir / "DEPTH_SWEEP_REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"\nReport generated at {report_path}", flush=True)

if __name__ == "__main__":
    main()
