from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .inference import (
    al_energy_shifted,
    constraint_residuals,
    free_init,
    run_pcalm,
    zero_duals_like,
)
from .model import Params, block_pred

AsyncMode = Literal[
    "random_coordinate",
    "random_block",
    "heterogeneous_rates",
    "bounded_staleness",
    "fully_async_local",
]

ASYNC_MODES: tuple[str, ...] = (
    "random_coordinate",
    "random_block",
    "heterogeneous_rates",
    "bounded_staleness",
    "fully_async_local",
)


@dataclass(frozen=True)
class AsyncSchedule:
    """Execution semantics for the diagnostic asynchronous simulator.

    `layer_update_budget` is the primary work budget. A single update of one free
    activity layer counts as one unit, irrespective of scheduler mode. Therefore
    `layer_update_budget / n_free_layers` is the sweep-equivalent budget.

    For all modes except `fully_async_local`, the dual update cadence is retained
    from published PC-ALM: after `inner_steps * n_free_layers` primal layer
    updates, all constraint duals are updated together. This isolates primal
    scheduling semantics. `fully_async_local` removes that remaining barrier and
    updates only the selected layer's owned dual after each local event.
    """

    mode: AsyncMode
    layer_update_budget: int
    state_lr: float
    rho: float
    alpha: float
    inner_steps: int = 1
    block_size: int = 1
    scheduler_seed: int = 0
    heterogeneous_rate_strength: float = 0.0
    tau_max: int = 0
    checkpoint_every_events: int | None = None
    divergence_threshold: float = 1e6


@dataclass(frozen=True)
class AsyncEventRecord:
    event_index: int
    selected_layers: tuple[int, ...]
    layer_update_events: int
    dual_update_events: int
    global_dual_updates: int
    requested_staleness: int
    staleness_used: int
    read_version: int
    state_version_before: int
    global_dual_update: bool


@dataclass(frozen=True)
class AsyncCheckpoint:
    event_index: int
    layer_update_events: int
    dual_update_events: int
    global_dual_updates: int
    sweep_equivalents: float
    max_staleness_observed: int
    free: tuple[jax.Array, ...]
    duals: tuple[jax.Array, ...]


@dataclass(frozen=True)
class AsyncRunResult:
    schedule: AsyncSchedule
    status: str
    final_free: tuple[jax.Array, ...]
    final_duals: tuple[jax.Array, ...]
    events: tuple[AsyncEventRecord, ...]
    checkpoints: tuple[AsyncCheckpoint, ...]
    layer_rates: tuple[float, ...]
    layer_update_events: int
    dual_update_events: int
    global_dual_updates: int
    max_staleness_observed: int


@dataclass(frozen=True)
class SyncReferenceState:
    """A state obtained only through the official `run_pcalm` implementation."""

    free: tuple[jax.Array, ...]
    actual_duals: tuple[jax.Array, ...]
    weight_duals: tuple[jax.Array, ...]


def run_sync_reference_state(
    params: Params,
    scales,
    skips,
    x,
    y,
    *,
    state_lr: float,
    rho: float,
    alpha: float,
    budget: int,
    inner_steps: int,
    weight_credit_timing: str,
    phi,
) -> SyncReferenceState:
    """Thin tracing adapter around Sakana's unchanged synchronous oracle.

    `run_pcalm` returns the dual state selected for weight credit. When
    `weight_credit_timing="pre_dual_energy"`, the actual post-iteration dual state
    is reconstructed from that returned pre-update state and the official
    residual equation. No synchronous primal update is reimplemented here.
    """

    if budget < 0:
        raise ValueError("budget must be non-negative")
    if budget == 0:
        free = free_init(params, scales, skips, x, phi)
        residuals = constraint_residuals(params, scales, skips, x, free, phi)
        duals = zero_duals_like(residuals)
        return SyncReferenceState(tuple(free), tuple(duals), tuple(duals))

    free, weight_duals = run_pcalm(
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
    if weight_credit_timing == "post_dual_energy":
        actual_duals = weight_duals
    else:
        residuals = constraint_residuals(params, scales, skips, x, free, phi)
        actual_duals = [lam + alpha * residual for lam, residual in zip(weight_duals, residuals)]
    return SyncReferenceState(tuple(free), tuple(actual_duals), tuple(weight_duals))


def heterogeneous_layer_rates(n_layers: int, strength: float, seed: int) -> np.ndarray:
    """Return seeded normalized log-normal event rates.

    `strength=0` is exactly uniform. Larger values create reproducible rate
    heterogeneity without changing any numerical learning-rate parameter.
    """

    if n_layers < 1:
        raise ValueError("n_layers must be positive")
    if strength < 0:
        raise ValueError("heterogeneous_rate_strength must be non-negative")
    if strength == 0:
        return np.full(n_layers, 1.0 / n_layers, dtype=np.float64)
    rng = np.random.default_rng(seed)
    logits = strength * rng.normal(size=n_layers)
    logits -= np.max(logits)
    rates = np.exp(logits)
    return rates / rates.sum()


def assert_finite_state(free, duals) -> None:
    for kind, arrays in (("activity", free), ("dual", duals)):
        for layer_ix, value in enumerate(arrays):
            if not np.isfinite(np.asarray(value)).all():
                raise FloatingPointError(f"non-finite {kind} state at layer {layer_ix}")


def _max_abs_state(free, duals) -> float:
    values = [float(jnp.max(jnp.abs(x))) for x in (*free, *duals)]
    return max(values, default=0.0)


def _local_constraint_residual(
    params: Params,
    scales,
    skips,
    x,
    free,
    read_free,
    layer_ix: int,
    phi,
) -> jax.Array:
    z_prev = x if layer_ix == 0 else read_free[layer_ix - 1]
    pred = block_pred(
        params[layer_ix],
        scales[layer_ix],
        skips[layer_ix],
        z_prev,
        phi,
        is_first=(layer_ix == 0),
    )
    return free[layer_ix] - pred


def _validate_schedule(schedule: AsyncSchedule, n_layers: int) -> None:
    if schedule.mode not in ASYNC_MODES:
        raise ValueError(f"unknown async mode: {schedule.mode}")
    if schedule.layer_update_budget < 0:
        raise ValueError("layer_update_budget must be non-negative")
    if schedule.state_lr < 0:
        raise ValueError("state_lr must be non-negative")
    if schedule.rho <= 0:
        raise ValueError("rho must be positive")
    if schedule.inner_steps < 1:
        raise ValueError("inner_steps must be at least 1")
    if schedule.block_size < 1:
        raise ValueError("block_size must be at least 1")
    if schedule.block_size > n_layers:
        raise ValueError("block_size cannot exceed number of free layers")
    if schedule.tau_max < 0:
        raise ValueError("tau_max must be non-negative")
    if schedule.checkpoint_every_events is not None and schedule.checkpoint_every_events < 1:
        raise ValueError("checkpoint_every_events must be positive when provided")
    if schedule.divergence_threshold <= 0:
        raise ValueError("divergence_threshold must be positive")


def _selected_layers(
    schedule: AsyncSchedule,
    rng: np.random.Generator,
    rates: np.ndarray,
    n_layers: int,
    max_block_size: int,
) -> tuple[int, ...]:
    if schedule.mode == "random_block":
        count = min(schedule.block_size, max_block_size)
        chosen = rng.choice(n_layers, size=count, replace=False)
        return tuple(int(x) for x in chosen)
    if schedule.mode in {"heterogeneous_rates", "fully_async_local"}:
        layer = int(rng.choice(n_layers, p=rates))
        return (layer,)
    layer = int(rng.integers(0, n_layers))
    return (layer,)


def run_async_inference(
    params: Params,
    scales,
    skips,
    x,
    y,
    *,
    schedule: AsyncSchedule,
    phi,
) -> AsyncRunResult:
    """Run an explicit non-JIT event simulator for async PC-ALM variants.

    The model, supervised loss, constraints, shifted augmented-Lagrangian energy,
    activity step scaling, and state variables are shared with `pcalm.inference`.
    Only execution/scheduling semantics differ.
    """

    free = list(free_init(params, scales, skips, x, phi))
    residuals0 = constraint_residuals(params, scales, skips, x, free, phi)
    duals = list(zero_duals_like(residuals0))
    n_layers = len(free)
    _validate_schedule(schedule, n_layers)

    effective_lr = schedule.state_lr * x.shape[0]

    def energy(free_, duals_):
        return al_energy_shifted(params, scales, skips, x, y, free_, duals_, schedule.rho, phi)

    grad_free = jax.grad(energy, argnums=0)

    rate_strength = schedule.heterogeneous_rate_strength if schedule.mode in {"heterogeneous_rates", "fully_async_local"} else 0.0
    rates = heterogeneous_layer_rates(n_layers, rate_strength, schedule.scheduler_seed + 104729)
    rng = np.random.default_rng(schedule.scheduler_seed)

    # History is versioned after each scheduler event. JAX arrays are immutable,
    # so retaining tuples is sufficient to preserve the old versions.
    history: deque[tuple[int, tuple[jax.Array, ...]]] = deque(maxlen=max(1, schedule.tau_max + 1))
    history.append((0, tuple(free)))

    events: list[AsyncEventRecord] = []
    checkpoints: list[AsyncCheckpoint] = []
    layer_update_events = 0
    dual_update_events = 0
    global_dual_updates = 0
    state_version = 0
    max_staleness_observed = 0
    status = "ok"

    def record_checkpoint(event_index: int) -> None:
        checkpoints.append(
            AsyncCheckpoint(
                event_index=event_index,
                layer_update_events=layer_update_events,
                dual_update_events=dual_update_events,
                global_dual_updates=global_dual_updates,
                sweep_equivalents=layer_update_events / n_layers,
                max_staleness_observed=max_staleness_observed,
                free=tuple(free),
                duals=tuple(duals),
            )
        )

    record_checkpoint(event_index=0)
    checkpoint_every = schedule.checkpoint_every_events
    next_checkpoint = checkpoint_every if checkpoint_every is not None else None
    dual_interval = schedule.inner_steps * n_layers
    event_index = 0

    while layer_update_events < schedule.layer_update_budget:
        event_index += 1
        remaining_total = schedule.layer_update_budget - layer_update_events
        if schedule.mode == "fully_async_local":
            max_block = 1
        else:
            to_dual_boundary = dual_interval - (layer_update_events % dual_interval)
            max_block = min(remaining_total, to_dual_boundary)

        selected = _selected_layers(schedule, rng, rates, n_layers, max_block)
        if len(selected) > remaining_total:
            selected = selected[:remaining_total]

        requested_staleness = 0
        staleness_used = 0
        read_version = state_version
        if schedule.mode in {"bounded_staleness", "fully_async_local"} and schedule.tau_max > 0:
            requested_staleness = int(rng.integers(0, schedule.tau_max + 1))
            staleness_used = min(requested_staleness, state_version, len(history) - 1)
            read_version, stale_free = history[-1 - staleness_used]
            read_free = list(stale_free)
        else:
            read_free = list(free)

        # A local unit always owns its current activity. Staleness applies only
        # to the neighboring activity values it observes.
        for layer_ix in selected:
            read_free[layer_ix] = free[layer_ix]

        grads = grad_free(read_free, duals)
        new_free = list(free)
        for layer_ix in selected:
            new_free[layer_ix] = free[layer_ix] - effective_lr * grads[layer_ix]
        free = new_free
        layer_update_events += len(selected)

        did_global_dual_update = False
        if schedule.mode == "fully_async_local":
            layer_ix = selected[0]
            local_residual = _local_constraint_residual(
                params, scales, skips, x, free, read_free, layer_ix, phi
            )
            duals[layer_ix] = duals[layer_ix] + schedule.alpha * local_residual
            dual_update_events += 1
        elif layer_update_events % dual_interval == 0:
            residuals = constraint_residuals(params, scales, skips, x, free, phi)
            duals = [lam + schedule.alpha * r for lam, r in zip(duals, residuals)]
            dual_update_events += n_layers
            global_dual_updates += 1
            did_global_dual_update = True

        assert_finite_state(free, duals)
        if _max_abs_state(free, duals) > schedule.divergence_threshold:
            status = "diverged"

        max_staleness_observed = max(max_staleness_observed, staleness_used)
        events.append(
            AsyncEventRecord(
                event_index=event_index,
                selected_layers=selected,
                layer_update_events=layer_update_events,
                dual_update_events=dual_update_events,
                global_dual_updates=global_dual_updates,
                requested_staleness=requested_staleness,
                staleness_used=staleness_used,
                read_version=read_version,
                state_version_before=state_version,
                global_dual_update=did_global_dual_update,
            )
        )

        state_version += 1
        history.append((state_version, tuple(free)))

        if next_checkpoint is not None and layer_update_events >= next_checkpoint:
            record_checkpoint(event_index)
            while next_checkpoint <= layer_update_events:
                next_checkpoint += checkpoint_every

        if status != "ok":
            break

    if checkpoints[-1].layer_update_events != layer_update_events:
        record_checkpoint(event_index)

    return AsyncRunResult(
        schedule=schedule,
        status=status,
        final_free=tuple(free),
        final_duals=tuple(duals),
        events=tuple(events),
        checkpoints=tuple(checkpoints),
        layer_rates=tuple(float(x) for x in rates),
        layer_update_events=layer_update_events,
        dual_update_events=dual_update_events,
        global_dual_updates=global_dual_updates,
        max_staleness_observed=max_staleness_observed,
    )
