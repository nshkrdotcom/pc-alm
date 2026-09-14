from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np

from .inference import al_energy_shifted, bp_loss, constraint_residuals
from .metrics import tree_cos


@dataclass(frozen=True)
class FrontMeasurement:
    threshold: float
    contiguous_reach_layers: int
    any_reach_layers: int
    dispersion_gap_layers: int
    active_layer_count: int
    active_distance_std: float
    active_distance_mean: float


def layer_l2_norms(values) -> list[float]:
    return [float(jnp.linalg.norm(value)) for value in values]


def total_l2_norm(values) -> float:
    if not values:
        return 0.0
    sq = jnp.stack([jnp.sum(value * value) for value in values])
    return float(jnp.sqrt(jnp.sum(sq)))


def state_l2_distance(free, reference_free) -> float:
    diffs = [a - b for a, b in zip(free, reference_free)]
    return total_l2_norm(diffs)


def state_relative_l2_distance(free, reference_free, eps: float = 1e-30) -> float:
    return state_l2_distance(free, reference_free) / max(total_l2_norm(reference_free), eps)


def current_state_metrics(params, scales, skips, x, free, duals, phi) -> dict[str, object]:
    residuals = constraint_residuals(params, scales, skips, x, free, phi)
    residual_norms = layer_l2_norms(residuals)
    dual_norms = layer_l2_norms(duals)
    activity_norms = layer_l2_norms(free)
    return {
        "residuals": residuals,
        "residual_norms": residual_norms,
        "dual_norms": dual_norms,
        "activity_norms": activity_norms,
        "total_residual_norm": total_l2_norm(residuals),
        "total_dual_norm": total_l2_norm(duals),
        "total_activity_norm": total_l2_norm(free),
    }


def _array_cos(a: jax.Array, b: jax.Array) -> float:
    dot = jnp.sum(a * b)
    denom = jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b))
    return float(dot / jnp.maximum(denom, 1e-30))


def bp_weight_gradients(params, scales, skips, x, y, phi):
    return jax.grad(lambda p: bp_loss(p, scales, skips, x, y, phi))(params)


def weight_gradient_alignment(
    params,
    scales,
    skips,
    x,
    y,
    free,
    duals,
    *,
    rho: float,
    phi,
    bp_grads=None,
) -> tuple[float, list[float]]:
    free = jax.tree_util.tree_map(jax.lax.stop_gradient, free)
    duals = jax.tree_util.tree_map(jax.lax.stop_gradient, duals)
    if bp_grads is None:
        bp_grads = bp_weight_gradients(params, scales, skips, x, y, phi)
    method_grads = jax.grad(
        lambda p: al_energy_shifted(p, scales, skips, x, y, free, duals, rho, phi)
    )(params)
    total = float(tree_cos(method_grads, bp_grads))
    layerwise = [_array_cos(a, b) for a, b in zip(method_grads, bp_grads)]
    return total, layerwise


def credit_fractions(dual_norms: Iterable[float], reference_dual_norms: Iterable[float], eps: float) -> list[float]:
    return [float(value / (reference + eps)) for value, reference in zip(dual_norms, reference_dual_norms)]


def credit_front(credit_fraction_input_to_output: Iterable[float], threshold: float) -> FrontMeasurement:
    """Measure backward credit reach from the output side.

    Input order is the repository's natural free-layer order: index 0 is closest
    to the input, index L-1 is closest to the output. For front calculations we
    reverse this orientation so distance 0 is the output-adjacent constraint and
    increasing distance moves backward toward the input.

    `contiguous_reach_layers` is the number of consecutive output-side layers at
    or above threshold. `any_reach_layers` is the deepest threshold crossing even
    if gaps exist. Their difference is a simple dispersion/smearing indicator.
    """

    fractions = np.asarray(list(credit_fraction_input_to_output), dtype=np.float64)
    if threshold <= 0:
        raise ValueError("front threshold must be positive")
    active_by_distance = fractions[::-1] >= threshold

    contiguous = 0
    for active in active_by_distance:
        if not active:
            break
        contiguous += 1

    active_distances = np.flatnonzero(active_by_distance)
    if active_distances.size:
        any_reach = int(active_distances.max()) + 1
        mean = float(active_distances.mean())
        std = float(active_distances.std())
    else:
        any_reach = 0
        mean = float("nan")
        std = float("nan")

    return FrontMeasurement(
        threshold=float(threshold),
        contiguous_reach_layers=contiguous,
        any_reach_layers=any_reach,
        dispersion_gap_layers=any_reach - contiguous,
        active_layer_count=int(active_distances.size),
        active_distance_std=std,
        active_distance_mean=mean,
    )


def fit_propagation_exponent(points: Iterable[tuple[float, int]]) -> dict[str, float | int | str | None]:
    """Fit R(t) ~ t^beta only when the trace has minimal dynamic range."""

    usable = sorted({(float(t), int(r)) for t, r in points if t > 0 and r > 0})
    if len(usable) < 3:
        return {"beta": None, "r2": None, "n": len(usable), "reason": "fewer_than_3_positive_points"}

    t = np.asarray([p[0] for p in usable], dtype=np.float64)
    r = np.asarray([p[1] for p in usable], dtype=np.float64)
    if np.unique(r).size < 2 or r.max() / r.min() < 2.0:
        return {"beta": None, "r2": None, "n": len(usable), "reason": "insufficient_front_dynamic_range"}
    if t.max() / t.min() < 2.0:
        return {"beta": None, "r2": None, "n": len(usable), "reason": "insufficient_time_dynamic_range"}

    log_t = np.log(t)
    log_r = np.log(r)
    beta, intercept = np.polyfit(log_t, log_r, deg=1)
    pred = beta * log_t + intercept
    ss_res = float(np.sum((log_r - pred) ** 2))
    ss_tot = float(np.sum((log_r - log_r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"beta": float(beta), "r2": float(r2), "n": len(usable), "reason": None}
