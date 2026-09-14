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
