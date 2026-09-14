from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pcalm.async_inference import (
    AsyncSchedule,
    assert_finite_state,
    run_async_inference,
    run_sync_reference_state,
)
from pcalm.model import activation_fn, init_params, model_scales, skip_mask


def small_case(depth: int = 5):
    params = init_params(jax.random.PRNGKey(0), depth=depth, width=4, input_dim=3, output_dim=2)
    scales = model_scales(width=4, depth=depth, input_dim=3)
    skips = skip_mask(depth)
    phi = activation_fn("tanh")
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    y = jax.nn.one_hot(jnp.arange(5) % 2, 2)
    return params, scales, skips, phi, x, y


def run(mode: str, **updates):
    params, scales, skips, phi, x, y = small_case()
    values = dict(
        mode=mode,
        layer_update_budget=12,
        state_lr=0.08,
        rho=1.0,
        alpha=1.0,
        inner_steps=1,
        scheduler_seed=17,
        checkpoint_every_events=4,
    )
    values.update(updates)
    result = run_async_inference(params, scales, skips, x, y, schedule=AsyncSchedule(**values), phi=phi)
    return result


def test_async_seed_is_deterministic():
    a = run("bounded_staleness", tau_max=3)
    b = run("bounded_staleness", tau_max=3)
    assert [e.selected_layers for e in a.events] == [e.selected_layers for e in b.events]
    assert [e.staleness_used for e in a.events] == [e.staleness_used for e in b.events]
    for x, y in zip(a.final_free, b.final_free):
        assert jnp.allclose(x, y)
    for x, y in zip(a.final_duals, b.final_duals):
        assert jnp.allclose(x, y)


def test_tau_zero_is_exactly_nonstale_coordinate_schedule():
    coord = run("random_coordinate")
    stale0 = run("bounded_staleness", tau_max=0)
    assert [e.selected_layers for e in coord.events] == [e.selected_layers for e in stale0.events]
    assert all(e.staleness_used == 0 for e in stale0.events)
    for x, y in zip(coord.final_free, stale0.final_free):
        assert jnp.allclose(x, y, atol=1e-7, rtol=1e-7)
    for x, y in zip(coord.final_duals, stale0.final_duals):
        assert jnp.allclose(x, y, atol=1e-7, rtol=1e-7)


def test_delay_draws_do_not_change_layer_selection():
    coord = run('random_coordinate')
    for mode, extra in [('bounded_staleness', {'tau_max': 4}),
                        ('fully_async_local', {'tau_max': 2})]:
        other = run(mode, **extra)
        assert [e.selected_layers for e in coord.events] == [e.selected_layers for e in other.events]


def test_bounded_staleness_never_exceeds_tau_or_available_history():
    result = run("bounded_staleness", tau_max=2, layer_update_budget=20)
    assert result.max_staleness_observed <= 2
    for event in result.events:
        assert 0 <= event.staleness_used <= 2
        assert event.state_version_before - event.read_version == event.staleness_used


def test_coordinate_event_mutates_only_selected_activity_layer():
    params, scales, skips, phi, x, y = small_case()
    sync0 = run_sync_reference_state(
        params, scales, skips, x, y,
        state_lr=0.08, rho=1.0, alpha=1.0, budget=0, inner_steps=1,
        weight_credit_timing="post_dual_energy", phi=phi,
    )
    result = run_async_inference(
        params, scales, skips, x, y,
        schedule=AsyncSchedule(
            mode="random_coordinate", layer_update_budget=1, state_lr=0.08,
            rho=1.0, alpha=1.0, scheduler_seed=9,
        ),
        phi=phi,
    )
    selected = set(result.events[0].selected_layers)
    for ix, (before, after) in enumerate(zip(sync0.free, result.final_free)):
        if ix not in selected:
            assert jnp.array_equal(before, after)


def test_block_event_mutates_only_selected_activity_layers():
    params, scales, skips, phi, x, y = small_case(depth=6)
    sync0 = run_sync_reference_state(
        params, scales, skips, x, y,
        state_lr=0.08, rho=1.0, alpha=1.0, budget=0, inner_steps=1,
        weight_credit_timing="post_dual_energy", phi=phi,
    )
    result = run_async_inference(
        params, scales, skips, x, y,
        schedule=AsyncSchedule(
            mode="random_block", layer_update_budget=2, block_size=2, state_lr=0.08,
            rho=1.0, alpha=1.0, scheduler_seed=9,
        ),
        phi=phi,
    )
    selected = set(result.events[0].selected_layers)
    assert len(selected) == 2
    for ix, (before, after) in enumerate(zip(sync0.free, result.final_free)):
        if ix not in selected:
            assert jnp.array_equal(before, after)


def test_full_block_reduces_to_official_reference_semantics():
    params, scales, skips, phi, x, y = small_case(depth=5)
    n_free = len(params) - 1
    budget = 3
    inner_steps = 2
    reference = run_sync_reference_state(
        params, scales, skips, x, y,
        state_lr=0.08, rho=1.0, alpha=0.7, budget=budget, inner_steps=inner_steps,
        weight_credit_timing="post_dual_energy", phi=phi,
    )
    async_result = run_async_inference(
        params, scales, skips, x, y,
        schedule=AsyncSchedule(
            mode="random_block",
            layer_update_budget=budget * inner_steps * n_free,
            block_size=n_free,
            state_lr=0.08,
            rho=1.0,
            alpha=0.7,
            inner_steps=inner_steps,
            scheduler_seed=3,
        ),
        phi=phi,
    )
    for expected, actual in zip(reference.free, async_result.final_free):
        assert jnp.allclose(expected, actual, atol=2e-6, rtol=2e-6)
    for expected, actual in zip(reference.actual_duals, async_result.final_duals):
        assert jnp.allclose(expected, actual, atol=2e-6, rtol=2e-6)


def test_trace_accounting_is_consistent():
    result = run("random_block", block_size=3, layer_update_budget=17, checkpoint_every_events=4)
    n_free = len(result.final_free)
    assert result.layer_update_events == 17
    assert sum(len(e.selected_layers) for e in result.events) == 17
    assert result.checkpoints[0].layer_update_events == 0
    assert result.checkpoints[-1].layer_update_events == 17
    assert result.checkpoints[-1].sweep_equivalents == pytest.approx(17 / n_free)
    assert result.dual_update_events == result.global_dual_updates * n_free


def test_fully_async_local_dual_work_is_local_event_work():
    result = run("fully_async_local", layer_update_budget=13, tau_max=2)
    assert result.dual_update_events == 13
    assert result.global_dual_updates == 0


def test_fully_async_event_mutates_only_owned_activity_and_dual():
    result = run('fully_async_local', layer_update_budget=16, checkpoint_every_events=1, tau_max=3)
    for event, before, after in zip(result.events, result.checkpoints, result.checkpoints[1:]):
        for i in range(len(result.final_free)):
            if i not in event.selected_layers:
                np.testing.assert_array_equal(before.free[i],after.free[i])
                np.testing.assert_array_equal(before.duals[i],after.duals[i])


def test_nonfinite_state_is_rejected():
    with pytest.raises(FloatingPointError, match="non-finite activity"):
        assert_finite_state([jnp.asarray([jnp.nan])], [jnp.asarray([0.0])])


@pytest.mark.parametrize('mode,extra', [
    ('random_coordinate', {}), ('random_block', {'block_size': 3}),
    ('heterogeneous_rates', {'heterogeneous_rate_strength': 1.0}),
    ('bounded_staleness', {'tau_max': 4}),
    ('fully_async_local', {'tau_max': 3, 'heterogeneous_rate_strength': .7}),
])
def test_compiled_events_match_original_eager_trajectory(mode, extra):
    eager = run(mode, jit_events=False, layer_update_budget=24, checkpoint_every_events=1, **extra)
    compiled = run(mode, jit_events=True, layer_update_budget=24, checkpoint_every_events=1, **extra)
    assert eager.events == compiled.events
    assert eager.status == compiled.status
    for a, b in zip(eager.checkpoints, compiled.checkpoints):
        for x, y in zip((*a.free, *a.duals), (*b.free, *b.duals)):
            np.testing.assert_allclose(x, y, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize('mode', ['random_permutation', 'forward_ordered_sweep', 'reverse_ordered_sweep'])
def test_ordered_sweeps_determinism_counts_and_locality(mode):
    a = run(mode, layer_update_budget=12, checkpoint_every_events=1)
    b = run(mode, layer_update_budget=12, checkpoint_every_events=1, jit_events=False)
    assert a.events == b.events
    n = len(a.final_free)
    for start in range(0, 12, n):
        order = [e.selected_layers[0] for e in a.events[start:start+n]]
        assert sorted(order) == list(range(n))
        if mode == 'forward_ordered_sweep': assert order == list(range(n))
        if mode == 'reverse_ordered_sweep': assert order == list(range(n-1,-1,-1))
    assert a.layer_update_events == a.dual_update_events == 12
    for event, before, after, eager in zip(a.events, a.checkpoints, a.checkpoints[1:], b.checkpoints[1:]):
        for i in range(n):
            if i not in event.selected_layers:
                np.testing.assert_array_equal(before.free[i], after.free[i])
            np.testing.assert_allclose(after.free[i], eager.free[i], atol=3e-6, rtol=3e-6)
            np.testing.assert_allclose(after.duals[i], eager.duals[i], atol=3e-6, rtol=3e-6)


def test_depth32_gpu_fusion_preserves_event_trajectory():
    # This case fails at default GPU matmul precision despite small tanh tests
    # passing: fusion can select reduced-precision kernels for the full gradient.
    depth, width, input_dim = 32, 32, 128
    p = init_params(jax.random.PRNGKey(0), depth=depth, width=width, input_dim=input_dim, output_dim=10)
    x = jnp.ones((32,input_dim)); y=jax.nn.one_hot(jnp.arange(32)%10,10)
    results=[]
    for compiled in (False,True):
        results.append(run_async_inference(p,model_scales(width,depth,input_dim),skip_mask(depth),x,y,
            schedule=AsyncSchedule(mode='random_coordinate',layer_update_budget=62,state_lr=.234285,
                rho=1.,alpha=1.,jit_events=compiled,checkpoint_every_events=1),phi=activation_fn('relu')))
    assert results[0].events == results[1].events
    for a,b in zip(results[0].checkpoints,results[1].checkpoints):
        for u,v in zip((*a.free,*a.duals),(*b.free,*b.duals)):
            np.testing.assert_allclose(u,v,atol=2e-5,rtol=2e-5)
