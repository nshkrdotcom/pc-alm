#!/usr/bin/env python3
"""Forward Milestone F2: Block Gauss-Seidel Sweep under Protocol F0.

Evaluates intermediate points on the Pareto frontier between serial Reverse GS (block_size=1)
and parallel Jacobi (block_size=31) on L=32 (n=31 hidden layers).

Block sizes evaluated: K in {1, 2, 4, 8, 16, 31}.
Sweep budget: B=64.
5 seeds (0..4) on full Fashion-MNIST (60k/10k), 10 epochs, epoch shuffling, remainder dropping.
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
    root_dir = Path("results/f2_block_gs")
    root_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root_dir / "block_gs_metrics.csv"

    depth = 32
    width = 32
    batch_size = 64
    epochs = 10
    budget = 64
    seeds = [0, 1, 2, 3, 4]
    block_sizes = [1, 2, 4, 8, 16, 31]

    state_lr = 0.057390
    rho = 1.0
    alpha = 1.0
    learning_rate = 0.001
    n = depth - 1  # 31

    scales = model_scales(width, depth, 784)
    skips = skip_mask(depth)
    phi = activation_fn("relu")

    def build_blocks(block_size: int, n_layers: int) -> list[list[int]]:
        """Partition layers 0..n-1 into contiguous blocks, ordered in reverse for Reverse Block-GS."""
        blocks = []
        # Reverse block order: highest layers first
        indices = list(range(n_layers))
        # Split into blocks of size block_size from right to left
        rem = n_layers % block_size
        chunks = []
        curr = n_layers
        while curr > 0:
            start = max(0, curr - block_size)
            chunks.append(list(range(start, curr)))
            curr = start
        return chunks  # Already in reverse block order

    def make_block_gs_infer(block_size: int):
        blocks = build_blocks(block_size, n)
        num_blocks = len(blocks)

        def infer_block_gs(params, x, y, sweeps: int):
            free = tuple(free_init(params, scales, skips, x, phi))
            duals = tuple(zero_duals_like(constraint_residuals(params, scales, skips, x, free, phi)))
            effective_lr = state_lr * x.shape[0]

            def single_sweep(carry, _):
                f_curr, d_curr = carry
                for blk in blocks:
                    # Parallel gradient evaluation for all layers in this block
                    def energy_blk(blk_states):
                        f_temp = list(f_curr)
                        for idx, layer_idx in enumerate(blk):
                            f_temp[layer_idx] = blk_states[idx]
                        return al_energy_shifted(params, scales, skips, x, y, tuple(f_temp), d_curr, rho, phi)

                    blk_curr = tuple(f_curr[l] for l in blk)
                    blk_grads = jax.grad(energy_blk)(blk_curr)

                    # Update all layers in the block simultaneously
                    f_temp = list(f_curr)
                    d_temp = list(d_curr)
                    for idx, layer_idx in enumerate(blk):
                        z_new = blk_curr[idx] - effective_lr * blk_grads[idx]
                        f_temp[layer_idx] = z_new
                        z_prev = x if layer_idx == 0 else f_curr[layer_idx - 1]
                        pred = block_pred(params[layer_idx], scales[layer_idx], skips[layer_idx], z_prev, phi, is_first=(layer_idx == 0))
                        r_i = z_new - pred
                        d_temp[layer_idx] = d_curr[layer_idx] + alpha * r_i

                    f_curr = tuple(f_temp)
                    d_curr = tuple(d_temp)

                return (f_curr, d_curr), None

            (final_free, final_duals), _ = jax.lax.scan(single_sweep, (free, duals), xs=None, length=sweeps)
            return final_free, final_duals

        return infer_block_gs, num_blocks

    all_rows = []
    if csv_path.exists():
        with csv_path.open() as f:
            all_rows = list(csv.DictReader(f))
            for r in all_rows:
                r["seed"] = int(r["seed"])
                r["block_size"] = int(r["block_size"])
                r["test_acc"] = float(r["test_acc"])

    print("=================================================================", flush=True)
    print("  RUNNING FORWARD MILESTONE F2: BLOCK GAUSS-SEIDEL SWEEP", flush=True)
    print("=================================================================", flush=True)

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

    for block_size in block_sizes:
        print(f"\n>>> Compiling Block-GS with K={block_size} <<<", flush=True)
        infer_fn, num_blocks = make_block_gs_infer(block_size)

        @jax.jit
        def step(params, opt_state, x, y):
            free, duals = infer_fn(params, x, y, budget)
            free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
            duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
            grads = jax.grad(lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi))(params)
            return adam_apply(params, grads, opt_state, learning_rate)

        @jax.jit
        def diag(params, x, y):
            free, duals = infer_fn(params, x, y, budget)
            free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
            duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
            residuals = constraint_residuals(params, scales, skips, x, free, phi)
            res_norm = jnp.sqrt(sum(jnp.sum(r * r) for r in residuals))
            dual_norm = jnp.sqrt(sum(jnp.sum(lam * lam) for lam in duals))
            return res_norm, dual_norm

        crit_path_per_sweep = num_blocks
        total_crit_path = budget * crit_path_per_sweep
        total_layer_work = budget * n

        for seed in seeds:
            done = any(r["block_size"] == block_size and r["seed"] == seed for r in all_rows)
            if done:
                print(f"[CACHED] K={block_size:2d} | Seed {seed}", flush=True)
                continue

            print(f"[Block-GS | K={block_size:2d} | S={seed}] Starting run...", flush=True)
            train_x, train_y, test_x, test_y = load_dataset(
                "fashion_mnist", data_dir=Path("data"), train_subset=60000, test_subset=10000, seed=seed
            )
            params = init_params(jax.random.PRNGKey(seed), depth=depth, width=width, input_dim=784, output_dim=10)
            opt_state = adam_init(params)

            n_train = len(train_x)
            num_batches = n_train // batch_size
            n_used = num_batches * batch_size
            rng = np.random.default_rng(seed * 1000 + block_size * 10)

            train_x_gpu = jnp.asarray(train_x)
            train_y_gpu = jnp.asarray(train_y)

            t0 = time.perf_counter()
            for epoch in range(epochs):
                perm = rng.permutation(n_train)[:n_used]
                for b_idx in range(num_batches):
                    idx = perm[b_idx * batch_size : (b_idx + 1) * batch_size]
                    xb, yb = train_x_gpu[idx], train_y_gpu[idx]
                    params, opt_state = step(params, opt_state, xb, yb)

            elapsed = time.perf_counter() - t0

            tr_loss, tr_acc = evaluate_model(params, train_x[:2048], train_y[:2048])
            te_loss, te_acc = evaluate_model(params, test_x, test_y)
            res, dual = diag(params, train_x_gpu[:batch_size], train_y_gpu[:batch_size])

            row = {
                "block_size": block_size,
                "num_blocks": num_blocks,
                "seed": seed,
                "budget_sweeps": budget,
                "total_layer_update_work": total_layer_work,
                "critical_path_steps": total_crit_path,
                "train_acc_probe_2048": float(tr_acc),
                "test_acc": float(te_acc),
                "residual_norm": float(res),
                "dual_norm": float(dual),
                "wall_clock_sec": float(elapsed),
                "shuffled": True,
            }
            all_rows.append(row)
            print(
                f"[Block-GS | K={block_size:2d} | S={seed}] "
                f"TestAcc={te_acc*100:5.2f}% | WallClock={elapsed:.1f}s",
                flush=True,
            )

            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_rows)

    # Statistical Aggregation
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in all_rows:
        grouped.setdefault(int(r["block_size"]), []).append(r)

    summary_rows = []
    for k, g in sorted(grouped.items()):
        accs = [float(x["test_acc"]) for x in g]
        m_mean = float(np.mean(accs))
        m_std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
        ci_low, ci_high = bootstrap_ci(accs)

        summary_rows.append({
            "block_size": k,
            "num_blocks": g[0]["num_blocks"],
            "budget_sweeps": budget,
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

    summary_csv = root_dir / "block_gs_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report = [
        "# Forward Milestone F2: Block Gauss-Seidel Pareto Frontier",
        "",
        "## 1. Executive Summary",
        "- Explores intermediate Block-GS operators with block sizes $K \\in \\{1, 2, 4, 8, 16, 31\\}$ at budget $B=64$.",
        "- Bridges the gap between serial Reverse GS ($K=1$, critical path 1,984) and parallel Jacobi ($K=31$, critical path 64).",
        "",
        "## 2. Block-GS Pareto Frontier Table",
        "",
        "| Block Size ($K$) | Blocks ($M$) | Critical Path ($T_{\\text{crit}}$) | Test Accuracy (mean ± SD) | 95% Bootstrap CI | Wall Clock |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for s in summary_rows:
        report.append(
            f"| {s['block_size']} | {s['num_blocks']} | {s['critical_path_steps']} | "
            f"**{s['test_acc_mean']*100:.2f}% ± {s['test_acc_std']*100:.2f}%** | "
            f"[{s['test_acc_ci95_low']*100:.2f}%, {s['test_acc_ci95_high']*100:.2f}%] | "
            f"{s['wall_clock_mean']:.1f}s |"
        )

    report_path = root_dir / "BLOCK_GS_REPORT.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"\nReport generated at {report_path}", flush=True)

if __name__ == "__main__":
    main()
