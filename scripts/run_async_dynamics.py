from __future__ import annotations

import argparse
import csv
import json
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcalm.async_inference import (  # noqa: E402
    ASYNC_MODES,
    AsyncRunResult,
    AsyncSchedule,
    assert_finite_state,
    run_async_inference,
    run_sync_reference_state,
)
from pcalm.async_metrics import (  # noqa: E402
    bp_weight_gradients,
    credit_fractions,
    credit_front,
    current_state_metrics,
    fit_propagation_exponent,
    state_l2_distance,
    state_relative_l2_distance,
    weight_gradient_alignment,
)
from pcalm.data import load_dataset  # noqa: E402
from pcalm.model import activation_fn, init_params, model_scales, skip_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare synchronous PC-ALM with explicit asynchronous scheduling variants at fixed weights."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/async_smoke.yaml"))
    parser.add_argument("--output-dir", type=Path, help="Override config output_dir.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots", action="store_true", help="Generate summary PNG plots regardless of config setting.")
    return parser.parse_args()


def load_values(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    if not isinstance(values, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return values


def require_keys(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unknown keys in {where}: {sorted(unknown)}")


def mode_label(spec: dict[str, Any]) -> str:
    mode = spec["mode"]
    if mode == "sync":
        return "sync"
    if mode in {"random_coordinate", "random_permutation", "forward_ordered_sweep", "reverse_ordered_sweep"}:
        return f"{mode}_seed{spec.get('scheduler_seed', 0)}"
    if mode == "random_block":
        return f"random_block_b{spec.get('block_size', 1)}_seed{spec.get('scheduler_seed', 0)}"
    if mode == "heterogeneous_rates":
        return f"heterogeneous_rates_h{spec.get('heterogeneous_rate_strength', 0.0):g}_seed{spec.get('scheduler_seed', 0)}"
    if mode == "bounded_staleness":
        return f"bounded_staleness_tau{spec.get('tau_max', 0)}_seed{spec.get('scheduler_seed', 0)}"
    if mode == "fully_async_local":
        return (
            f"fully_async_local_tau{spec.get('tau_max', 0)}"
            f"_h{spec.get('heterogeneous_rate_strength', 0.0):g}_seed{spec.get('scheduler_seed', 0)}"
        )
    raise ValueError(f"unknown mode: {mode}")


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fields})


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def max_abs_state(free, duals) -> float:
    vals = [float(jnp.max(jnp.abs(value))) for value in (*free, *duals)]
    return max(vals, default=0.0)


def diagnostic_rows(
    *,
    mode: str,
    label: str,
    checkpoint_index: int,
    event_index: int,
    layer_update_events: int,
    dual_update_events: int,
    global_dual_updates: int,
    sweep_equivalents: float,
    max_staleness_observed: int,
    free,
    actual_duals,
    gradient_duals,
    params,
    scales,
    skips,
    x,
    y,
    phi,
    rho: float,
    bp_grads,
    reference_free,
    reference_dual_norms: list[float],
    front_thresholds: list[float],
    front_eps: float,
):
    metrics = current_state_metrics(params, scales, skips, x, free, actual_duals, phi)
    fractions = credit_fractions(metrics["dual_norms"], reference_dual_norms, front_eps)
    grad_cos, grad_layer_cos, grad_norms, bp_norms = weight_gradient_alignment(
        params,
        scales,
        skips,
        x,
        y,
        free,
        gradient_duals,
        rho=rho,
        phi=phi,
        bp_grads=bp_grads,
        return_norms=True,
    )
    distance = state_l2_distance(free, reference_free)
    relative_distance = state_relative_l2_distance(free, reference_free)
    finite = all(np.isfinite(np.asarray(v)).all() for v in (*free, *actual_duals))

    trace_row = {
        "mode": mode,
        "mode_label": label,
        "checkpoint_index": checkpoint_index,
        "event_index": event_index,
        "layer_update_events": layer_update_events,
        "sweep_equivalents": sweep_equivalents,
        "dual_update_events": dual_update_events,
        "global_dual_updates": global_dual_updates,
        "max_staleness_observed": max_staleness_observed,
        "total_residual_norm": metrics["total_residual_norm"],
        "total_dual_norm": metrics["total_dual_norm"],
        "total_activity_norm": metrics["total_activity_norm"],
        "weight_grad_cos_to_bp": grad_cos,
        "state_l2_to_sync_reference_terminal": distance,
        "state_relative_l2_to_sync_reference_terminal": relative_distance,
        "max_abs_state": max_abs_state(free, actual_duals),
        "finite": finite,
    }

    n_free = len(free)
    layer_rows = []
    for layer_ix in range(n_free):
        layer_rows.append(
            {
                "mode": mode,
                "mode_label": label,
                "checkpoint_index": checkpoint_index,
                "layer_update_events": layer_update_events,
                "sweep_equivalents": sweep_equivalents,
                "layer_index_input_to_output": layer_ix,
                "distance_from_output": n_free - 1 - layer_ix,
                "activity_norm": metrics["activity_norms"][layer_ix],
                "residual_norm": metrics["residual_norms"][layer_ix],
                "dual_norm": metrics["dual_norms"][layer_ix],
                "reference_dual_norm": reference_dual_norms[layer_ix],
                "credit_fraction": fractions[layer_ix],
            }
        )

    gradient_rows = []
    depth = len(params)
    for layer_ix, cosine in enumerate(grad_layer_cos):
        gradient_rows.append(
            {
                "mode": mode,
                "mode_label": label,
                "checkpoint_index": checkpoint_index,
                "layer_update_events": layer_update_events,
                "sweep_equivalents": sweep_equivalents,
                "weight_layer_index_input_to_output": layer_ix,
                "weight_distance_from_output": depth - 1 - layer_ix,
                "weight_grad_cos_to_bp": cosine,
                "weight_grad_norm": grad_norms[layer_ix],
                "bp_weight_grad_norm": bp_norms[layer_ix],
            }
        )

    front_rows = []
    for threshold in front_thresholds:
        front = credit_front(fractions, threshold)
        front_rows.append(
            {
                "mode": mode,
                "mode_label": label,
                "checkpoint_index": checkpoint_index,
                "layer_update_events": layer_update_events,
                "sweep_equivalents": sweep_equivalents,
                "threshold": threshold,
                "R_contiguous_layers": front.contiguous_reach_layers,
                "R_any_layers": front.any_reach_layers,
                "dispersion_gap_layers": front.dispersion_gap_layers,
                "active_layer_count": front.active_layer_count,
                "active_distance_mean": front.active_distance_mean,
                "active_distance_std": front.active_distance_std,
            }
        )
    return trace_row, layer_rows, gradient_rows, front_rows


def maybe_plot(trace_rows: list[dict[str, Any]], front_rows: list[dict[str, Any]], output_dir: Path) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))
    import matplotlib.pyplot as plt

    labels = list(dict.fromkeys(row["mode_label"] for row in trace_rows))
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for label in labels:
        rows = [row for row in trace_rows if row["mode_label"] == label]
        ax.plot([r["sweep_equivalents"] for r in rows], [r["total_residual_norm"] for r in rows], label=label)
    ax.set_xlabel("sweep equivalents (layer_update_events / free_layers)")
    ax.set_ylabel("total residual norm")
    ax.set_title("PC-ALM scheduling dynamics: residual norm")
    ax.legend(fontsize=7)
    fig.savefig(output_dir / "residual_norm.png", dpi=180)
    plt.close(fig)

    thresholds = sorted({float(row["threshold"]) for row in front_rows})
    if thresholds:
        threshold = thresholds[0]
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        for label in labels:
            rows = [
                row for row in front_rows
                if row["mode_label"] == label and float(row["threshold"]) == threshold
            ]
            ax.plot([r["sweep_equivalents"] for r in rows], [r["R_contiguous_layers"] for r in rows], label=label)
        ax.set_xlabel("sweep equivalents (layer_update_events / free_layers)")
        ax.set_ylabel("R(t): contiguous layers reached from output")
        ax.set_title(f"Credit front at threshold {threshold:g}")
        ax.legend(fontsize=7)
        fig.savefig(output_dir / "credit_front.png", dpi=180)
        plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    jax.config.update('jax_default_matmul_precision', 'highest')
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [*sorted(Path('pcalm').glob('*.py')), Path(__file__)]}
    print(f"Backend: {jax.default_backend()}; devices: {jax.devices()}", flush=True)
    args = parse_args()
    values = load_values(args.config)
    require_keys(
        values,
        {"dataset", "output_dir", "seed", "batch_size", "model", "data", "inference", "modes", "plots"},
        "root",
    )
    model = dict(values.get("model", {}) or {})
    data = dict(values.get("data", {}) or {})
    inference = dict(values.get("inference", {}) or {})
    modes = list(values.get("modes", []) or [])
    require_keys(model, {"width", "depth", "activation", "input_dim", "output_dim"}, "model")
    require_keys(data, {"train_subset", "test_subset"}, "data")
    require_keys(
        inference,
        {
            "work_sweep_equivalents",
            "reference_budget",
            "state_lr",
            "rho",
            "alpha",
            "inner_steps",
            "weight_credit_timing",
            "diagnostic_interval_sweeps",
            "front_thresholds",
            "front_eps",
            "divergence_threshold",
        },
        "inference",
    )
    if not modes:
        raise ValueError("modes must contain at least one scheduler")

    allowed_mode_keys = {
        "mode",
        "scheduler_seed",
        "block_size",
        "heterogeneous_rate_strength",
        "tau_max",
    }
    labels: list[str] = []
    for ix, spec in enumerate(modes):
        if not isinstance(spec, dict):
            raise ValueError(f"modes[{ix}] must be a mapping")
        require_keys(spec, allowed_mode_keys, f"modes[{ix}]")
        if spec.get("mode") not in {"sync", *ASYNC_MODES}:
            raise ValueError(f"unknown mode in modes[{ix}]: {spec.get('mode')}")
        labels.append(mode_label(spec))
    if len(labels) != len(set(labels)):
        raise ValueError("mode labels are not unique; use distinct scheduler seeds or parameters")

    seed = int(values.get("seed", 0))
    batch_size = int(values.get("batch_size", 16))
    width = int(model.get("width", 32))
    depth = int(model.get("depth", 32))
    activation = str(model.get("activation", "relu"))
    input_dim = int(model.get("input_dim", 784))
    output_dim = int(model.get("output_dim", 10))
    if depth < 2:
        raise ValueError("model depth must be at least 2")
    n_free = depth - 1

    state_lr = float(inference.get("state_lr", 0.25))
    rho = float(inference.get("rho", 1.0))
    alpha = float(inference.get("alpha", 1.0))
    inner_steps = int(inference.get("inner_steps", 1))
    weight_credit_timing = str(inference.get("weight_credit_timing", "pre_dual_energy"))
    work_sweeps = float(inference.get("work_sweep_equivalents", 64))
    reference_budget = int(inference.get("reference_budget", 128))
    diagnostic_interval_sweeps = float(inference.get("diagnostic_interval_sweeps", 1.0))
    front_thresholds = [float(x) for x in inference.get("front_thresholds", [0.1, 0.25, 0.5])]
    front_eps = float(inference.get("front_eps", 1e-12))
    divergence_threshold = float(inference.get("divergence_threshold", 1e6))

    layer_update_budget_f = work_sweeps * n_free
    layer_update_budget = int(round(layer_update_budget_f))
    if not math.isclose(layer_update_budget_f, layer_update_budget, rel_tol=0, abs_tol=1e-9):
        raise ValueError("work_sweep_equivalents * number_of_free_layers must be an integer")
    control_budget_f = work_sweeps / inner_steps
    control_budget = int(round(control_budget_f))
    if not math.isclose(control_budget_f, control_budget, rel_tol=0, abs_tol=1e-9):
        raise ValueError("sync control requires work_sweep_equivalents / inner_steps to be an integer")
    if reference_budget < control_budget:
        raise ValueError("reference_budget must be at least the same-work synchronous control budget")

    checkpoint_events_f = diagnostic_interval_sweeps * n_free
    checkpoint_events = int(round(checkpoint_events_f))
    if checkpoint_events < 1 or not math.isclose(checkpoint_events_f, checkpoint_events, rel_tol=0, abs_tol=1e-9):
        raise ValueError("diagnostic_interval_sweeps * number_of_free_layers must be a positive integer")

    dataset = str(values.get("dataset", "synthetic"))
    x_train, y_train, _, _ = load_dataset(
        dataset,
        train_subset=int(data.get("train_subset", max(batch_size, 64))),
        test_subset=int(data.get("test_subset", 32)),
        seed=seed,
        data_dir=args.data_dir,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    if x_train.shape[0] < batch_size:
        raise ValueError("train_subset must be at least batch_size")
    x = jnp.asarray(x_train[:batch_size])
    y = jnp.asarray(y_train[:batch_size])

    phi = activation_fn(activation)
    scales = model_scales(width, depth, input_dim)
    skips = skip_mask(depth)
    params = init_params(
        jax.random.PRNGKey(seed),
        depth=depth,
        width=width,
        input_dim=input_dim,
        output_dim=output_dim,
        dtype=jnp.float32,
    )
    bp_grads = bp_weight_gradients(params, scales, skips, x, y, phi)

    sync_cache: dict[int, Any] = {}

    def sync_state(budget: int):
        if budget not in sync_cache:
            sync_cache[budget] = run_sync_reference_state(
                params,
                scales,
                skips,
                x,
                y,
                state_lr=state_lr,
                rho=rho,
                alpha=alpha,
                budget=budget,
                inner_steps=inner_steps,
                weight_credit_timing=weight_credit_timing,
                phi=phi,
            )
            assert_finite_state(sync_cache[budget].free, sync_cache[budget].actual_duals)
        return sync_cache[budget]

    reference = sync_state(reference_budget)
    reference_metrics = current_state_metrics(params, scales, skips, x, reference.free, reference.actual_duals, phi)
    reference_dual_norms = list(reference_metrics["dual_norms"])
    same_work_sync = sync_state(control_budget)

    output_dir = args.output_dir or Path(str(values.get("output_dir", "results/async_dynamics/run")))
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    front_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    mode_summaries: dict[str, Any] = {}

    for spec, label in zip(modes, labels):
        mode = str(spec["mode"])
        mode_front_start = len(front_rows)
        mode_trace_start = len(trace_rows)
        result: AsyncRunResult | None = None
        status = "ok"

        if mode == "sync":
            budgets = [0]
            next_diag = diagnostic_interval_sweeps
            for budget in range(1, control_budget + 1):
                sweeps = budget * inner_steps
                if sweeps + 1e-12 >= next_diag or budget == control_budget:
                    budgets.append(budget)
                    while next_diag <= sweeps + 1e-12:
                        next_diag += diagnostic_interval_sweeps

            for checkpoint_index, budget in enumerate(budgets):
                state = sync_state(budget)
                events = budget * inner_steps
                layer_events = events * n_free
                dual_events = budget * n_free
                row, rows_l, rows_g, rows_f = diagnostic_rows(
                    mode=mode,
                    label=label,
                    checkpoint_index=checkpoint_index,
                    event_index=events,
                    layer_update_events=layer_events,
                    dual_update_events=dual_events,
                    global_dual_updates=budget,
                    sweep_equivalents=layer_events / n_free,
                    max_staleness_observed=0,
                    free=state.free,
                    actual_duals=state.actual_duals,
                    gradient_duals=state.actual_duals,
                    params=params,
                    scales=scales,
                    skips=skips,
                    x=x,
                    y=y,
                    phi=phi,
                    rho=rho,
                    bp_grads=bp_grads,
                    reference_free=reference.free,
                    reference_dual_norms=reference_dual_norms,
                    front_thresholds=front_thresholds,
                    front_eps=front_eps,
                )
                trace_rows.append(row)
                row['published_weight_grad_cos_to_bp'] = weight_gradient_alignment(
                    params, scales, skips, x, y, state.free, state.weight_duals,
                    rho=rho, phi=phi, bp_grads=bp_grads)[0]
                layer_rows.extend(rows_l)
                gradient_rows.extend(rows_g)
                front_rows.extend(rows_f)
        else:
            result = run_async_inference(
                params,
                scales,
                skips,
                x,
                y,
                schedule=AsyncSchedule(
                    mode=mode,  # type: ignore[arg-type]
                    layer_update_budget=layer_update_budget,
                    state_lr=state_lr,
                    rho=rho,
                    alpha=alpha,
                    inner_steps=inner_steps,
                    block_size=int(spec.get("block_size", 1)),
                    scheduler_seed=int(spec.get("scheduler_seed", 0)),
                    heterogeneous_rate_strength=float(spec.get("heterogeneous_rate_strength", 0.0)),
                    tau_max=int(spec.get("tau_max", 0)),
                    checkpoint_every_events=checkpoint_events,
                    divergence_threshold=divergence_threshold,
                ),
                phi=phi,
            )
            status = result.status
            for checkpoint_index, checkpoint in enumerate(result.checkpoints):
                row, rows_l, rows_g, rows_f = diagnostic_rows(
                    mode=mode,
                    label=label,
                    checkpoint_index=checkpoint_index,
                    event_index=checkpoint.event_index,
                    layer_update_events=checkpoint.layer_update_events,
                    dual_update_events=checkpoint.dual_update_events,
                    global_dual_updates=checkpoint.global_dual_updates,
                    sweep_equivalents=checkpoint.sweep_equivalents,
                    max_staleness_observed=checkpoint.max_staleness_observed,
                    free=checkpoint.free,
                    actual_duals=checkpoint.duals,
                    gradient_duals=checkpoint.duals,
                    params=params,
                    scales=scales,
                    skips=skips,
                    x=x,
                    y=y,
                    phi=phi,
                    rho=rho,
                    bp_grads=bp_grads,
                    reference_free=reference.free,
                    reference_dual_norms=reference_dual_norms,
                    front_thresholds=front_thresholds,
                    front_eps=front_eps,
                )
                trace_rows.append(row)
                layer_rows.extend(rows_l)
                gradient_rows.extend(rows_g)
                front_rows.extend(rows_f)

            for event in result.events:
                event_rows.append(
                    {
                        "mode": mode,
                        "mode_label": label,
                        "event_index": event.event_index,
                        "selected_layers_input_to_output": ";".join(str(ix) for ix in event.selected_layers),
                        "selected_layer_count": len(event.selected_layers),
                        "layer_update_events": event.layer_update_events,
                        "sweep_equivalents": event.layer_update_events / n_free,
                        "dual_update_events": event.dual_update_events,
                        "global_dual_updates": event.global_dual_updates,
                        "requested_staleness": event.requested_staleness,
                        "staleness_used": event.staleness_used,
                        "read_version": event.read_version,
                        "state_version_before": event.state_version_before,
                        "global_dual_update": event.global_dual_update,
                    }
                )

        mode_trace = trace_rows[mode_trace_start:]
        mode_front = front_rows[mode_front_start:]
        final_row = mode_trace[-1]
        final_free = same_work_sync.free if mode == "sync" else result.final_free
        final_duals = same_work_sync.actual_duals if mode == "sync" else result.final_duals
        beta = {}
        for threshold in front_thresholds:
            points = [
                (float(row["sweep_equivalents"]), int(row["R_contiguous_layers"]))
                for row in mode_front
                if math.isclose(float(row["threshold"]), threshold)
            ]
            beta[str(threshold)] = fit_propagation_exponent(points)

        mode_summaries[label] = {
            "mode": mode,
            "status": status,
            "final_layer_update_events": int(final_row["layer_update_events"]),
            "final_sweep_equivalents": float(final_row["sweep_equivalents"]),
            "final_dual_update_events": int(final_row["dual_update_events"]),
            "final_total_residual_norm": float(final_row["total_residual_norm"]),
            "final_total_dual_norm": float(final_row["total_dual_norm"]),
            "final_weight_grad_cos_to_bp": float(final_row["weight_grad_cos_to_bp"]),
            "final_state_l2_to_sync_reference_terminal": float(final_row["state_l2_to_sync_reference_terminal"]),
            "final_state_l2_to_sync_same_work": state_l2_distance(final_free, same_work_sync.free),
            "final_dual_l2_to_sync_same_work": float(
                np.sqrt(sum(float(jnp.sum((a - b) ** 2)) for a, b in zip(final_duals, same_work_sync.actual_duals)))
            ),
            "max_staleness_observed": 0 if result is None else result.max_staleness_observed,
            "layer_rates": None if result is None else list(result.layer_rates),
            "layer_update_counts": None if result is None else np.bincount(
                [i for event in result.events for i in event.selected_layers], minlength=n_free).tolist(),
            "propagation_exponent_by_threshold": beta,
            "tau_max_sweep_equivalents": int(spec.get("tau_max", 0)) / n_free,
        }
        print(f"Completed {label}: {status}; cosine={final_row['weight_grad_cos_to_bp']:.5f}; elapsed={time.perf_counter()-started:.1f}s", flush=True)

    resolved_config = {
        **values,
        "output_dir": str(output_dir),
        "_resolved": {
            "number_of_free_layers": n_free,
            "layer_update_budget": layer_update_budget,
            "sync_control_budget": control_budget,
            "sync_control_sweep_equivalents": control_budget * inner_steps,
            "reference_budget": reference_budget,
            "reference_sweep_equivalents": reference_budget * inner_steps,
            "checkpoint_every_layer_update_events": checkpoint_events,
            "work_accounting": "sweep_equivalents = layer_update_events / number_of_free_layers",
        },
    }
    summary = {
        "runtime_seconds": time.perf_counter() - started,
        "backend": jax.default_backend(),
        "matmul_precision": "highest",
        "source_sha256": source_hashes,
        "gradient_diagnostic_timing": "post_dual_energy_for_all_modes; published sync timing separately in trace.csv",
        "status": "ok" if all(v["status"] == "ok" for v in mode_summaries.values()) else "has_diverged_modes",
        "dataset": dataset,
        "seed": seed,
        "batch_size": batch_size,
        "model": {
            "width": width,
            "depth": depth,
            "activation": activation,
            "input_dim": input_dim,
            "output_dim": output_dim,
            "number_of_free_layers": n_free,
        },
        "inference": {
            "state_lr": state_lr,
            "rho": rho,
            "alpha": alpha,
            "inner_steps": inner_steps,
            "weight_credit_timing": weight_credit_timing,
            "work_sweep_equivalents": work_sweeps,
            "layer_update_budget": layer_update_budget,
            "reference_budget": reference_budget,
            "reference_sweep_equivalents": reference_budget * inner_steps,
            "front_thresholds": front_thresholds,
            "front_eps": front_eps,
        },
        "reference_terminal_dual_norms_input_to_output": reference_dual_norms,
        "modes": mode_summaries,
    }

    write_json(resolved_config, output_dir / "config.json")
    write_json(summary, output_dir / "summary.json")
    write_csv(trace_rows, output_dir / "trace.csv")
    write_csv(layer_rows, output_dir / "layer_trace.csv")
    write_csv(gradient_rows, output_dir / "gradient_trace.csv")
    write_csv(front_rows, output_dir / "front.csv")
    write_csv(event_rows, output_dir / "events.csv")
    if not event_rows:
        (output_dir / "events.csv").write_text("mode,mode_label,event_index\n")

    if bool(values.get("plots", False)) or args.plots:
        maybe_plot(trace_rows, front_rows, output_dir)

    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
