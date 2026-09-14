from __future__ import annotations

import math

from pcalm.async_metrics import credit_front, fit_propagation_exponent


def test_credit_front_orientation_and_dispersion():
    # Natural order is input -> output. Reversed, the active mask is:
    # distance from output 0..3 = [True, True, False, True].
    front = credit_front([0.6, 0.1, 0.8, 0.9], threshold=0.5)
    assert front.contiguous_reach_layers == 2
    assert front.any_reach_layers == 4
    assert front.dispersion_gap_layers == 2
    assert front.active_layer_count == 3


def test_propagation_exponent_fit_requires_dynamic_range():
    insufficient = fit_propagation_exponent([(1, 1), (2, 1), (4, 1)])
    assert insufficient["beta"] is None
    assert insufficient["reason"] == "insufficient_front_dynamic_range"

    ballistic = fit_propagation_exponent([(1, 1), (2, 2), (4, 4), (8, 8)])
    assert math.isclose(ballistic["beta"], 1.0, rel_tol=1e-6)
    assert ballistic["r2"] > 0.999999


def test_compiled_alignment_matches_eager_equations():
    import jax
    import numpy as np
    from test_async_inference import small_case
    from pcalm.async_inference import run_sync_reference_state
    from pcalm.async_metrics import weight_gradient_alignment, bp_weight_gradients, _array_cos
    from pcalm.inference import al_energy_shifted, run_pcalm
    from pcalm.metrics import tree_cos
    p, scales, skips, phi, x, y = small_case()
    state = run_sync_reference_state(p, scales, skips, x, y, state_lr=.08, rho=1., alpha=1.,
        budget=4, inner_steps=1, weight_credit_timing='post_dual_energy', phi=phi)
    eager_free, eager_duals = run_pcalm(p, scales, skips, x, y, state_lr=.08, rho=1., alpha=1.,
        budget=4, inner_steps=1, weight_credit_timing='post_dual_energy', phi=phi)
    for a,b in zip((*state.free,*state.actual_duals),(*eager_free,*eager_duals)):
        np.testing.assert_allclose(a,b,atol=3e-6,rtol=3e-6)
    bp=bp_weight_gradients(p,scales,skips,x,y,phi)
    g=jax.grad(lambda pp: al_energy_shifted(pp,scales,skips,x,y,state.free,state.actual_duals,1.,phi))(p)
    total,layers=weight_gradient_alignment(p,scales,skips,x,y,state.free,state.actual_duals,rho=1.,phi=phi,bp_grads=bp)
    np.testing.assert_allclose(total,tree_cos(g,bp),atol=3e-6,rtol=3e-6)
    np.testing.assert_allclose(layers,[_array_cos(a,b) for a,b in zip(g,bp)],atol=3e-6,rtol=3e-6)
    from pcalm.async_metrics import current_state_metrics
    from pcalm.inference import constraint_residuals
    measured=current_state_metrics(p,scales,skips,x,state.free,state.actual_duals,phi)
    residuals=constraint_residuals(p,scales,skips,x,state.free,phi)
    for key, arrays in [('residual',residuals),('dual',state.actual_duals),('activity',state.free)]:
        norms=[np.linalg.norm(np.asarray(a)) for a in arrays]
        np.testing.assert_allclose(measured[key+'_norms'],norms,atol=3e-6,rtol=3e-6)
        np.testing.assert_allclose(measured['total_'+key+'_norm'],np.linalg.norm(norms),atol=3e-6,rtol=3e-6)
