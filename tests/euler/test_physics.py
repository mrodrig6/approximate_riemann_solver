"""Equation of state, variable conversions and the physical Euler flux.

The twin of ``tests/shallow_water/test_physics.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import conservative_to_primitive, primitive_to_conservative, solve, solve_exact
from src.euler.physics import (
    check_gamma,
    enthalpy,
    euler_flux,
    pressure,
    sound_speed,
    specific_internal_energy,
)

GAMMA = 1.4


def test_primitive_conservative_roundtrip():
    rng = np.random.default_rng(0)
    w = np.column_stack(
        [
            rng.uniform(0.1, 5.0, 50),    # rho
            rng.uniform(-3.0, 3.0, 50),   # u
            rng.uniform(0.1, 100.0, 50),  # p
        ]
    )
    u = primitive_to_conservative(w, GAMMA)
    assert conservative_to_primitive(u, GAMMA) == pytest.approx(w, rel=1e-12)

def test_single_state_roundtrip_keeps_shape():
    w = np.array([1.0, 0.5, 2.0])
    u = primitive_to_conservative(w, GAMMA)
    assert u.shape == (3,)
    assert conservative_to_primitive(u, GAMMA).shape == (3,)

def test_eos_identities():
    """H = (E + p)/rho, e = p/((gamma-1) rho), a^2 = gamma p / rho."""
    w = np.array([2.0, 1.5, 7.0])
    u = primitive_to_conservative(w, GAMMA)
    assert pressure(u, GAMMA) == pytest.approx(w[2])
    assert enthalpy(u, GAMMA) == pytest.approx((u[2] + w[2]) / w[0])
    assert sound_speed(w[0], w[2], GAMMA) ** 2 == pytest.approx(GAMMA * w[2] / w[0])
    assert specific_internal_energy(w[0], w[2], GAMMA) == pytest.approx(
        w[2] / ((GAMMA - 1.0) * w[0])
    )

def test_euler_flux_matches_its_definition():
    w = np.array([1.3, -0.7, 2.2])
    u = primitive_to_conservative(w, GAMMA)
    rho, vel, p = w
    expected = [rho * vel, rho * vel * vel + p, vel * (u[2] + p)]
    assert euler_flux(u, GAMMA) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [1.0, 0.9, 0.0, -1.4])
def test_check_gamma_rejects_gamma_at_or_below_one(bad):
    """``gamma > 1`` is load-bearing, not decorative.

    The closure divides by ``gamma - 1`` throughout, so ``gamma = 1`` divides
    by zero and ``gamma < 1`` makes the internal energy negative while every
    formula stays arithmetically valid.

    Regression: gamma was unvalidated, and the failure surfaced far from its
    cause.  ``gamma = 1`` gave an infinite total energy, and the time loop
    then reported a non-physical state and blamed the *flux* -- advising the
    user to try a different one when no flux would have helped.
    """
    with pytest.raises(ValueError, match="gamma must be greater than 1"):
        check_gamma(bad)


def test_check_gamma_accepts_the_usual_values():
    assert check_gamma(1.4) == 1.4
    assert check_gamma(5.0 / 3.0) == pytest.approx(5.0 / 3.0)
    assert isinstance(check_gamma(1.4), float)


@pytest.mark.parametrize("bad", [1.0, 0.0, -1.0])
def test_the_solvers_reject_a_bad_gamma_at_the_door(bad):
    """Both entry points validate, so neither fails downstream."""
    with pytest.raises(ValueError, match="gamma must be greater than 1"):
        solve(bad, [1.0, 0.0, 1.0], [0.125, 0.0, 0.1], 0.2, nx=32)
    with pytest.raises(ValueError, match="gamma must be greater than 1"):
        solve_exact(bad, [1.0, 0.0, 1.0], [0.125, 0.0, 0.1], 0.2, n=32)
