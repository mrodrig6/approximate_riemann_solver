"""The shallow-water exact Riemann solver, wet and dry.

The twin of ``tests/euler/test_exact.py``, with the dry-bed cases that have
no Euler counterpart: a vacuum is pathological for a gas, a dry bed is not.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shallow_water import (
    GRAVITY,
    PROBLEMS,
    depth_function,
    get_problem,
    is_dry_bed,
    solve_exact,
    solve_star_depth,
)

G = GRAVITY


def test_depth_function_vanishes_at_the_root():
    for name in ("dam_break", "two_bores", "two_rarefactions", "subcritical_step"):
        prob = get_problem(name)
        h_star, _, _ = solve_star_depth(prob.left, prob.right, prob.gravity)
        assert abs(depth_function(h_star, prob.left, prob.right, prob.gravity)) < 1e-10


def test_depth_function_is_monotone_increasing():
    prob = get_problem("dam_break")
    hs = np.logspace(-3, 1, 200)
    f = np.array([depth_function(h, prob.left, prob.right, prob.gravity) for h in hs])
    assert np.all(np.diff(f) > 0)


def test_symmetric_problem_has_zero_star_velocity():
    """Mirror-symmetric data must give ``u_star = 0`` exactly."""
    sol = solve_exact([1.0, 2.5], [1.0, -2.5], 0.05)
    assert sol.u_star == pytest.approx(0.0, abs=1e-12)
    assert sol.left_wave == sol.right_wave == "bore"


def test_wave_patterns_are_identified():
    assert solve_exact([1.0, 0.0], [0.1, 0.0], 0.05).left_wave == "rarefaction"
    assert solve_exact([1.0, 0.0], [0.1, 0.0], 0.05).right_wave == "bore"
    both_bores = solve_exact([1.0, 2.5], [1.0, -2.5], 0.05)
    assert (both_bores.left_wave, both_bores.right_wave) == ("bore", "bore")
    both_rare = solve_exact([1.0, -2.5], [1.0, 2.5], 0.05)
    assert (both_rare.left_wave, both_rare.right_wave) == (
        "rarefaction",
        "rarefaction",
    )


def test_ritter_solution_for_a_dry_dam_break():
    """The dry dam break has a closed-form answer (Ritter).

    Over dry bed, the depth at the dam site is ``4 h_L / 9`` for all time,
    the velocity there is ``2 sqrt(g h_L) / 3``, and the wet/dry front
    advances at ``2 sqrt(g h_L)``.
    """
    h_l, t = 1.0, 0.05
    sol = solve_exact([h_l, 0.0], [0.0, 0.0], t, n=4001)
    assert sol.dry

    at_dam = np.argmin(np.abs(sol.x - 0.5))
    assert sol.h[at_dam] == pytest.approx(4.0 * h_l / 9.0, rel=1e-3)
    assert sol.u[at_dam] == pytest.approx(2.0 * np.sqrt(G * h_l) / 3.0, rel=1e-3)

    # Front position, to within a cell.
    front = sol.x[np.argmax(sol.h < 1e-12)]
    assert front == pytest.approx(0.5 + 2.0 * np.sqrt(G * h_l) * t, abs=2 * sol.x[0])


def test_dry_bed_generation_is_detected():
    """Fast separation opens a dry region: ``u_R - u_L >= 2(a_L + a_R)``."""
    assert not is_dry_bed([1.0, -5.0], [1.0, 5.0])     # 10 < 4 sqrt(g) = 12.53
    assert is_dry_bed([1.0, -7.0], [1.0, 7.0])         # 14 > 12.53
    sol = solve_exact([1.0, -7.0], [1.0, 7.0], 0.03)
    assert sol.dry
    assert np.isnan(sol.h_star)
    assert np.min(sol.h) == 0.0          # a genuinely dry region appears
    assert np.all(sol.h >= 0.0)


def test_star_depth_refuses_dry_data():
    with pytest.raises(ValueError, match="dry"):
        solve_star_depth([1.0, -7.0], [1.0, 7.0])


def test_exact_solver_validates_input():
    with pytest.raises(ValueError, match="positive"):
        solve_exact([1.0, 0.0], [0.1, 0.0], 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        solve_exact([-1.0, 0.0], [0.1, 0.0], 0.05)
    with pytest.raises(ValueError, match="both sides are dry"):
        solve_exact([0.0, 0.0], [0.0, 0.0], 0.05)


def test_far_field_recovers_the_initial_states():
    sol = solve_exact([1.0, 0.0], [0.1, 0.0], 0.05, n=1000)
    assert sol.h[0] == pytest.approx(1.0)
    assert sol.h[-1] == pytest.approx(0.1)


def test_depth_is_never_negative_anywhere():
    for prob in PROBLEMS.values():
        sol = solve_exact(
            prob.left, prob.right, prob.t_end, gravity=prob.gravity,
            n=997, x_diaphragm=prob.x_diaphragm,
        )
        assert np.all(sol.h >= 0.0), prob.name
        assert np.all(np.isfinite(sol.u)), prob.name
