"""Remediation R1 & R2: Comprehensive Budget Sweep & Parallelism Accounting

Evaluates training performance across budgets B in {8, 16, 32, 48, 64, 96, 128}:
- sync_pcalm (Jacobi parallel)
- sync_gs_pcalm (Reverse Gauss-Seidel)
- sync_gs_forward_pcalm (Forward Gauss-Seidel control)
- async_local_pcalm (Random coordinate)
- bp (Standard Backprop baseline)

Full work ontology recorded:
- layer_update_work: Total local primal layer updates = B * (L - 1)
- sequential_depth: Longest critical-path dependency chain (B for Jacobi; B * (L - 1) for GS/async)
- parallel_work: layer_update_work * sequential_depth
- wall_clock_sec: Actual elapsed physical training time
- own_bp_cosine: Gradient alignment evaluated using that method's own inference state

Dataset: Fashion-MNIST at Depth 32, Width 32, 3 random seeds (0, 1, 2).
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    Params,
    activation_fn,
    block_pred,
    init_params,
    logits,
    model_scales,
    skip_mask,
)
from pcalm.optim import adam_apply, adam_init


def make_budget_infer_fns(scales, skips, phi, state_lr: float, rho: float, alpha: float):
    n = len(scales) - 1

    def run_sync_pcalm(params, x, y, budget: int):
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

    def run_sync_gs(params, x, y, budget: int):
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

    def run_sync_gs_forward(params, x, y, budget: int):
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

    def run_async_local(params, x, y, rng_key, budget: int):
        free = tuple(free_init(params, scales, skips, x, phi))
        duals = tuple(zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi)))
        effective_lr = state_lr * x.shape[0]
        total_steps = budget * n
        selected_layers = jax.random.randint(rng_key, (total_steps,), 0, n)

        def single_step(carry, layer_idx):
            f_curr, d_curr = carry
            def energy(f_):
                return al_energy_shifted(params, scales, skips, x, y, f_, d_curr, rho, phi)
            grads = jax.grad(energy)(f_curr)
            f_next = tuple(jnp.where(i == layer_idx, z - effective_lr * g, z) for i, (z, g) in enumerate(zip(f_curr, grads)))
            residuals = constraint_residuals(params, scales, skips, x, f_next, phi)
            d_next = tuple(jnp.where(i == layer_idx, lam + alpha * r, lam) for i, (lam, r) in enumerate(zip(d_curr, residuals)))
            return (f_next, d_next), None

        (final_free, final_duals), _ = jax.lax.scan(single_step, (free, duals), selected_layers)
        return final_free, final_duals

    return run_sync_pcalm, run_sync_gs, run_sync_gs_forward, run_async_local


def main():
    root_dir = Path("results/exp_r1_budget_sweep")
    root_dir.mkdir(parents=True, exist_ok=True)

    depth = 32
    width = 32
    n_free = depth - 1
    batch_size = 64
    epochs = 4
    seeds = [0, 1, 2]
    budgets = [8, 16, 32, 48, 64, 96, 128]
    iterative_methods = ["sync_pcalm", "sync_gs_pcalm", "sync_gs_forward_pcalm", "async_local_pcalm"]

    train_samples = 2048
    test_samples = 512

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

    infer_sync, infer_gs, infer_gs_fwd, infer_async = make_budget_infer_fns(scales, skips, phi, state_lr, rho, alpha)

    @jax.jit
    def step_bp(params, opt_state, x, y):
        grads = jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
        return adam_apply(params, grads, opt_state, learning_rate)

    def make_step_fn(infer_fn, takes_rng=False, budget=16):
        @jax.jit
        def step(params, opt_state, x, y, k=None):
            if takes_rng:
                free, duals = infer_fn(params, x, y, k, budget)
            else:
                free, duals = infer_fn(params, x, y, budget)
            free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
            duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
            grads = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi))(params)
            return adam_apply(params, grads, opt_state, learning_rate)
        return step

    def make_diag_fn(infer_fn, takes_rng=False, budget=16):
        @jax.jit
        def diag(params, x, y, k=None):
            bp_g = jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)
            if takes_rng:
                free, duals = infer_fn(params, x, y, k, budget)
            else:
                free, duals = infer_fn(params, x, y, budget)
            free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
            duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
            grads = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi))(params)
            cos = tree_cos(grads, bp_g)
            residuals = constraint_residuals(params, scales, skips, x, free, phi)
            res_norm = jnp.sqrt(sum(jnp.sum(r * r) for r in residuals))
            dual_norm = jnp.sqrt(sum(jnp.sum(lam * lam) for lam in duals))
            return cos, res_norm, dual_norm
        return diag

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

    all_rows = []

    # 1. Backpropagation Baseline (one run per seed)
    for seed in seeds:
        print(f"\n--- BP Baseline | Seed {seed} ---", flush=True)
        train_x, train_y, test_x, test_y = load_dataset(
            "fashion_mnist", data_dir=Path("data"), train_subset=train_samples, test_subset=test_samples, seed=seed
        )
        params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width, input_dim=784, output_dim=10)
        opt_state = adam_init(params)
        rng = np.random.default_rng(seed)

        t0 = time.perf_counter()
        for epoch in range(epochs):
            for i in range(0, len(train_x), batch_size):
                xb, yb = jnp.asarray(train_x[i:i+batch_size]), jnp.asarray(train_y[i:i+batch_size])
                params, opt_state = step_bp(params, opt_state, xb, yb)
        elapsed = time.perf_counter() - t0

        tr_loss, tr_acc = evaluate_model(params, train_x, train_y)
        te_loss, te_acc = evaluate_model(params, test_x, test_y)

        row = {
            "method": "bp",
            "seed": seed,
            "budget_sweeps": 0,
            "layer_update_work": 0,
            "sequential_depth": 0,
            "parallel_work": 0,
            "train_acc": float(tr_acc),
            "test_acc": float(te_acc),
            "own_bp_cos": 1.0,
            "residual_norm": 0.0,
            "dual_norm": 0.0,
            "wall_clock_sec": float(elapsed),
        }
        all_rows.append(row)
        print(f"BP Seed {seed}: Test Acc = {te_acc:.4f} ({elapsed:.1f}s)", flush=True)

    # 2. Iterative Methods across Budgets
    for budget in budgets:
        print(f"\n=======================================================", flush=True)
        print(f"       COMPILING & EXECUTING BUDGET B = {budget}", flush=True)
        print(f"=======================================================", flush=True)

        step_sync = make_step_fn(infer_sync, budget=budget)
        step_gs = make_step_fn(infer_gs, budget=budget)
        step_gs_fwd = make_step_fn(infer_gs_fwd, budget=budget)
        step_async = make_step_fn(infer_async, takes_rng=True, budget=budget)

        diag_sync = make_diag_fn(infer_sync, budget=budget)
        diag_gs = make_diag_fn(infer_gs, budget=budget)
        diag_gs_fwd = make_diag_fn(infer_gs_fwd, budget=budget)
        diag_async = make_diag_fn(infer_async, takes_rng=True, budget=budget)

        step_map = {
            "sync_pcalm": lambda p, o, x, y, k: step_sync(p, o, x, y),
            "sync_gs_pcalm": lambda p, o, x, y, k: step_gs(p, o, x, y),
            "sync_gs_forward_pcalm": lambda p, o, x, y, k: step_gs_fwd(p, o, x, y),
            "async_local_pcalm": lambda p, o, x, y, k: step_async(p, o, x, y, k),
        }
        diag_map = {
            "sync_pcalm": lambda p, x, y, k: diag_sync(p, x, y),
            "sync_gs_pcalm": lambda p, x, y, k: diag_gs(p, x, y),
            "sync_gs_forward_pcalm": lambda p, x, y, k: diag_gs_fwd(p, x, y),
            "async_local_pcalm": lambda p, x, y, k: diag_async(p, x, y, k),
        }

        for method in iterative_methods:
            for seed in seeds:
                train_x, train_y, test_x, test_y = load_dataset(
                    "fashion_mnist", data_dir=Path("data"), train_subset=train_samples, test_subset=test_samples, seed=seed
                )
                params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width, input_dim=784, output_dim=10)
                opt_state = adam_init(params)
                step_fn = step_map[method]
                diag_fn = diag_map[method]

                t0 = time.perf_counter()
                step_cnt = 0
                for epoch in range(epochs):
                    for i in range(0, len(train_x), batch_size):
                        xb, yb = jnp.asarray(train_x[i:i+batch_size]), jnp.asarray(train_y[i:i+batch_size])
                        k = jax.random.PRNGKey(seed * 10000 + step_cnt)
                        params, opt_state = step_fn(params, opt_state, xb, yb, k)
                        step_cnt += 1
                elapsed = time.perf_counter() - t0

                tr_loss, tr_acc = evaluate_model(params, train_x, train_y)
                te_loss, te_acc = evaluate_model(params, test_x, test_y)
                cos, res, dual = diag_fn(params, jnp.asarray(train_x[:batch_size]), jnp.asarray(train_y[:batch_size]), jax.random.PRNGKey(seed + 999))

                layer_work = budget * n_free
                seq_depth = budget if method == "sync_pcalm" else (budget * n_free)
                par_work = layer_work * seq_depth

                row = {
                    "method": method,
                    "seed": seed,
                    "budget_sweeps": budget,
                    "layer_update_work": layer_work,
                    "sequential_depth": seq_depth,
                    "parallel_work": par_work,
                    "train_acc": float(tr_acc),
                    "test_acc": float(te_acc),
                    "own_bp_cos": float(cos),
                    "residual_norm": float(res),
                    "dual_norm": float(dual),
                    "wall_clock_sec": float(elapsed),
                }
                all_rows.append(row)
                print(
                    f"[{method:21s} | B={budget:3d} | S={seed}] "
                    f"TestAcc={te_acc*100:5.2f}% | Cos={float(cos):6.4f} | "
                    f"SeqDepth={seq_depth:4d} ({elapsed:.1f}s)",
                    flush=True,
                )

                # Incremental CSV write
                csv_path = root_dir / "budget_sweep_metrics.csv"
                with csv_path.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(all_rows)

    # Summary grouped by (method, budget)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for r in all_rows:
        grouped.setdefault((r["method"], r["budget_sweeps"]), []).append(r)

    summary_rows = []
    for (m, b), g in grouped.items():
        summary_rows.append({
            "method": m,
            "budget_sweeps": b,
            "layer_update_work": g[0]["layer_update_work"],
            "sequential_depth": g[0]["sequential_depth"],
            "parallel_work": g[0]["parallel_work"],
            "test_acc_mean": float(np.mean([x["test_acc"] for x in g])),
            "test_acc_std": float(np.std([x["test_acc"] for x in g])),
            "train_acc_mean": float(np.mean([x["train_acc"] for x in g])),
            "own_bp_cos_mean": float(np.mean([x["own_bp_cos"] for x in g])),
            "residual_mean": float(np.mean([x["residual_norm"] for x in g])),
            "dual_norm_mean": float(np.mean([x["dual_norm"] for x in g])),
            "wall_clock_mean": float(np.mean([x["wall_clock_sec"] for x in g])),
        })

    summary_csv = root_dir / "summary_metrics.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Markdown Report
    report = [
        "# Remediation R1 & R2: Budget Sweep & Parallelism Accounting Report",
        "",
        "## 1. Executive Summary & Acceptance Criteria",
        "",
        "This experiment resolves Blocking Defects **R1** (budget starvation confound) and **R2** (critical-path parallel accounting and forward-GS control).",
        "It evaluates training across budgets $B \\in \\{8, 16, 32, 48, 64, 96, 128\\}$, specifically testing the published regime $B = 2L = 64$ and reporting all three work metrics:",
        "1. `layer_update_work`: total coordinate updates.",
        "2. `sequential_depth`: longest serial dependency chain.",
        "3. `parallel_work`: total work $\\times$ depth.",
        "",
        "## 2. Test Accuracy vs Budget Across Schedules (Depth 32, Fashion-MNIST)",
        "",
        "| Method | Budget ($B$) | Layer Work | Seq Depth | Parallel Work | Test Acc (mean ± SD) | Train Acc | Own BP Cosine | Wall Clock |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in sorted(summary_rows, key=lambda x: (x["method"] != "bp", x["method"], x["budget_sweeps"])):
        report.append(
            f"| `{s['method']}` | {s['budget_sweeps']} | {s['layer_update_work']} | {s['sequential_depth']} | {s['parallel_work']} | "
            f"**{s['test_acc_mean']*100:.2f}% ± {s['test_acc_std']*100:.2f}%** | "
            f"{s['train_acc_mean']*100:.2f}% | {s['own_bp_cos_mean']:.4f} | {s['wall_clock_mean']:.1f}s |"
        )

    report.extend([
        "",
        "## 3. Analysis of Jacobi Convergence Regime (R1)",
        "",
        "- **Published Operating Regime**: Standard PC-ALM operates at $B \\approx 2L = 64$. At $B=16 < L=32$, Jacobi is geometrically starved of reaching early layers.",
        "- **Empirical Finding**: As $B$ increases from 8 to 128, track when Jacobi (`sync_pcalm`) achieves parity with `bp` (~71%).",
        "- **Framing Correction**: If Jacobi catches up to BP at $B \\ge 2L$, the Gauss-Seidel 'win' is strictly an acceleration / work-efficiency statement in the sub-$2L$ regime, NOT a failure of synchronous PC-ALM to converge.",
        "",
        "## 4. Critical-Path Sequential Depth vs Parallel Work Analysis (R2)",
        "",
        "- Under full $L$-way hardware parallelism, Jacobi executes sweep $B$ in critical-path depth $O(B)$ sequential time steps.",
        "- In contrast, Gauss-Seidel traverses layers sequentially, requiring critical-path depth $O(B \\cdot L)$ sequential time steps.",
        "- Therefore, on an ideal massively parallel architecture with $P \\ge L$ cores, Jacobi at $B=64$ has sequential depth 64, whereas reverse GS at $B=16$ has sequential depth $16 \\times 31 = 496$.",
        "",
        "## 5. Forward-GS Control & Order-Specificity Statement (R2)",
        "",
        "- `sync_gs_forward_pcalm` traverses input-to-output ($0 \\to L-1$) with immediate local duals.",
        "- Comparison against `sync_gs_pcalm` ($L-1 \\to 0$) reveals whether the GS benefit is generic sequential coordinate descent or strictly directional reverse credit propagation.",
        "",
        "## 6. Asynchronous PC-ALM Verdict",
        "",
        "- `async_local_pcalm` samples layer updates randomly with immediate local duals.",
        "- Evaluates whether coordinate asynchrony aids credit propagation or destroys directional ordering.",
        "",
        "## 7. What Would Change Our Mind",
        "",
        "- If `sync_pcalm` fails to reach BP accuracy even at $B=128$, then the depth bottleneck in Jacobi is algorithmic instability rather than credit reach starvation.",
        "- If `sync_gs_forward_pcalm` matches `sync_gs_pcalm`, then the advantage is non-directional sequential coordinate conditioning rather than reverse credit backpropagation.",
        "- If `async_local_pcalm` matches `sync_gs_pcalm` at large $B$, then asynchronous execution is viable with sufficient budget.",
    ])

    report_path = root_dir / "BUDGET_SWEEP_REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"\nReport generated at {report_path}", flush=True)


if __name__ == "__main__":
    main()
