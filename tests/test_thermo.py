"""Equation of state, variable conversions and the physical Euler flux."""

from __future__ import annotations

import numpy as np
import pytest

from src import conservative_to_primitive, primitive_to_conservative
from src.euler.thermo import (
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

def test_thermo_identities():
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
