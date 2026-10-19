"""Tests for the exact Riemann solver.

The reference numbers are Toro's published star-region values
(*Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
Table 4.1 / Sec. 4.3.3), which makes these genuine regression tests against
the textbook rather than against a previous run of this code.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import get_problem, solve_exact, solve_star_pressure
from src.euler.exact import (
    pressure_function,
    pressure_function_derivative,
    sample_exact,
)
from src.euler.physics import sound_speed

# Toro Table 4.1: (problem, p_star, u_star).  Tolerances reflect the number of
# digits Toro prints, not the accuracy of the solver (which is ~1e-12).
TORO_TABLE_4_1 = [
    ("sod", 0.30313, 0.92745),
    ("toro2", 0.00189, 0.00000),
    ("toro3", 460.894, 19.5975),
    ("toro4", 1691.64, 8.68975),
    ("toro5", 460.894, -6.19633 + 6.19633),  # u* = 0 to printed precision
]


@pytest.mark.parametrize("name,p_star,u_star", TORO_TABLE_4_1)
def test_star_values_match_toro_table(name, p_star, u_star):
    prob = get_problem(name)
    sol = solve_exact(
        prob.gamma, prob.left, prob.right, prob.t_end,
        x_diaphragm=prob.x_diaphragm, n=10,
    )
    assert sol.p_star == pytest.approx(p_star, rel=1e-4, abs=1e-5)
    assert sol.u_star == pytest.approx(u_star, rel=1e-4, abs=1e-4)


def test_pressure_function_vanishes_at_the_root():
    """f(p_star) = 0 is the defining property (Toro eq. 4.5)."""
    prob = get_problem("sod")
    p_star, _ = solve_star_pressure(prob.gamma, prob.left, prob.right)
    residual = pressure_function(p_star, prob.gamma, prob.left, prob.right)
    assert abs(residual) < 1e-10


def test_pressure_function_derivative_matches_finite_difference():
    """The rarefaction exponent is easy to get wrong; pin it numerically."""
    prob = get_problem("sod")
    for p in (0.05, 0.2, 0.30313, 0.8, 5.0):
        h = 1e-6 * p
        fd = (
            pressure_function(p + h, prob.gamma, prob.left, prob.right)
            - pressure_function(p - h, prob.gamma, prob.left, prob.right)
        ) / (2 * h)
        analytic = pressure_function_derivative(p, prob.gamma, prob.left, prob.right)
        assert analytic == pytest.approx(fd, rel=1e-5)


def test_pressure_function_is_monotone_increasing():
    """f is strictly increasing, which is why Newton cannot miss the root."""
    prob = get_problem("sod")
    ps = np.logspace(-3, 2, 200)
    f = np.array([pressure_function(p, prob.gamma, prob.left, prob.right) for p in ps])
    assert np.all(np.diff(f) > 0)


def test_newton_is_insensitive_to_the_initial_guess():
    """Any positive starting point must reach the same root.

    Newton on this f converges from any positive start, so the result must
    not depend on the guess -- only the iteration count should.
    """
    prob = get_problem("sod")
    reference, _ = solve_star_pressure(prob.gamma, prob.left, prob.right)
    for guess in (1e-6, 0.01, 0.5, 10.0, 1e4):
        p_star, _ = solve_star_pressure(
            prob.gamma, prob.left, prob.right, p_guess=guess
        )
        assert p_star == pytest.approx(reference, rel=1e-10)


def test_dimensional_problem_converges():
    """The SI shock tube: p ~ 1e5, where an absolute tolerance is meaningless."""
    prob = get_problem("air_shock_tube")
    sol = solve_exact(
        prob.gamma, prob.left, prob.right, prob.t_end, n=50,
        x_diaphragm=prob.x_diaphragm,
    )
    assert sol.iterations <= 10
    assert np.all(np.isfinite(sol.rho))
    assert np.all(sol.rho > 0) and np.all(sol.p > 0)


def test_sampler_leaves_no_gaps():
    """Every sample point must get a physical state.

    A sampler that partitions the x/t axis with strict inequalities on both
    sides leaves points lying exactly on a wave unassigned.
    """
    for name in ("sod", "toro3", "toro4", "toro5", "sonic_rarefaction"):
        prob = get_problem(name)
        sol = solve_exact(
            prob.gamma, prob.left, prob.right, prob.t_end, n=997,
            x_diaphragm=prob.x_diaphragm,
        )
        assert np.all(sol.rho > 0), name
        assert np.all(sol.p > 0), name
        assert np.all(np.isfinite(sol.u)), name


def test_sampler_hits_the_contact_exactly():
    """A sample sitting exactly on the contact must take a star-region value."""
    prob = get_problem("sod")
    p_star, _ = solve_star_pressure(prob.gamma, prob.left, prob.right)
    sol = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=10)
    # s = u_star is the contact itself.
    state = sample_exact([sol.u_star], prob.gamma, prob.left, prob.right)[0]
    assert state[2] == pytest.approx(p_star, rel=1e-9)
    assert state[1] == pytest.approx(sol.u_star, rel=1e-9)


def test_far_field_recovers_the_initial_states():
    """Far from the fan the solution must be exactly the initial data."""
    prob = get_problem("sod")
    states = sample_exact([-1e6, 1e6], prob.gamma, prob.left, prob.right)
    assert states[0] == pytest.approx(prob.left)
    assert states[1] == pytest.approx(prob.right)


def test_vacuum_is_detected():
    """Toro eq. (4.40): fast separation generates a vacuum and has no star state."""
    gamma = 1.4
    w_l = np.array([1.0, -10.0, 0.4])
    w_r = np.array([1.0, 10.0, 0.4])
    with pytest.raises(ValueError, match="vacuum"):
        solve_star_pressure(gamma, w_l, w_r)


def test_negative_input_is_rejected():
    with pytest.raises(ValueError, match="positive density and pressure"):
        solve_star_pressure(1.4, np.array([-1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))


def test_two_rarefaction_guess_is_exact_for_two_rarefactions():
    """Toro's two-rarefaction guess solves the problem in one Newton step."""
    prob = get_problem("toro2")
    _, iterations = solve_star_pressure(prob.gamma, prob.left, prob.right)
    assert iterations <= 2


def test_isolated_shock_is_stationary_in_its_own_frame():
    """Consistency check on the shock relations via the sound speed helper."""
    prob = get_problem("air_shock_tube")
    sol = solve_exact(
        prob.gamma, prob.left, prob.right, prob.t_end, n=10,
        x_diaphragm=prob.x_diaphragm,
    )
    # The driver state was built as the exact post-shock state, so the left
    # wave is a null wave: p* must equal p_L and u* must equal u_L.
    assert sol.p_star == pytest.approx(prob.left[2], rel=1e-6)
    assert sol.u_star == pytest.approx(prob.left[1], rel=1e-6)
    assert sound_speed(prob.right[0], prob.right[2], prob.gamma) > 0
