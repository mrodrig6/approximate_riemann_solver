"""The shallow-water numerical fluxes, and the Roe eigenstructure.

The twin of ``tests/euler/test_fluxes.py``: the Roe U-property and wave
strengths, consistency of every flux, and upwinding where the flow is
supercritical -- the shallow-water analogue of supersonic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shallow_water import GRAVITY, flux_code, primitive_to_conservative
from src.shallow_water.fluxes import (
    hll_flux,
    interface_eigensystem,
    lax_friedrichs_flux,
    roe_average,
    rusanov_flux,
)
from src.shallow_water.physics import swe_flux

G = GRAVITY


def _cons(w):
    return primitive_to_conservative(np.asarray(w, dtype=float))


def _flux(u_cell):
    out = np.empty(2)
    swe_flux(u_cell, G, out)
    return out


def _evaluate(name, u_l, u_r, dx=0.01, dt=0.001):
    out = np.empty(2)
    if name == "hll":
        hll_flux(u_l, u_r, G, out)
    elif name == "rusanov":
        rusanov_flux(u_l, u_r, G, out)
    elif name == "lax_friedrichs":
        lax_friedrichs_flux(u_l, u_r, G, dx, dt, out)
    else:
        raise AssertionError(f"unhandled flux {name}")
    return out


def test_flux_code_accepts_names_aliases_and_integers():
    assert flux_code("HLL") == flux_code("hlle") == flux_code("hll")
    # HLLC reduces to HLL for this system: there is no contact to restore.
    assert flux_code("hllc") == flux_code("hll")
    assert flux_code("llf") == flux_code("rusanov")
    assert flux_code("lf") == flux_code("lax_friedrichs")
    with pytest.raises(ValueError, match="unknown flux"):
        flux_code("ausmup")
    with pytest.raises(ValueError, match="unknown flux code"):
        flux_code(42)


@pytest.mark.parametrize("name", ["hll", "rusanov", "lax_friedrichs"])
@pytest.mark.parametrize("w", [(1.0, 0.0), (0.5, 2.0), (2.0, -1.3)])
def test_flux_is_consistent(name, w):
    """``F(U, U) = F(U)``: the minimum requirement for consistency."""
    u = _cons(w)
    assert _evaluate(name, u, u) == pytest.approx(_flux(u), rel=1e-12, abs=1e-12)


def test_hll_is_upwind_when_supercritical():
    """Supercritical throughout, every wave travels one way: F = F(upwind)."""
    u_l, u_r = _cons([1.0, 8.0]), _cons([0.9, 8.2])
    assert _evaluate("hll", u_l, u_r) == pytest.approx(_flux(u_l), rel=1e-10)
    u_l, u_r = _cons([1.0, -8.0]), _cons([0.9, -8.2])
    assert _evaluate("hll", u_l, u_r) == pytest.approx(_flux(u_r), rel=1e-10)


def test_roe_u_property():
    """``F_R - F_L = sum_k alpha_k lambda_k K_k`` exactly.

    The defining property of the Roe linearisation, and the check that the
    ``sqrt(h)`` weighting and the ``sqrt(g (h_L + h_R)/2)`` celerity are the
    right averages for this system.
    """
    for w_l, w_r in [
        ([1.0, 0.0], [0.1, 0.0]),
        ([2.0, 1.5], [0.4, -0.7]),
        ([1.0, 2.5], [1.0, -2.5]),
        ([0.8, 0.3], [0.8, 0.3]),
    ]:
        u = np.vstack([_cons(w_l), _cons(w_r)])
        lam = np.empty((1, 2))
        k = np.empty((1, 2, 2))
        alpha = np.empty((1, 2))
        interface_eigensystem(u, G, lam, k, alpha)
        rebuilt = sum(alpha[0, m] * lam[0, m] * k[0, m] for m in range(2))
        assert rebuilt == pytest.approx(_flux(u[1]) - _flux(u[0]), rel=1e-10, abs=1e-10)


def test_wave_strengths_reconstruct_the_jump():
    u = np.vstack([_cons([1.0, 0.0]), _cons([0.1, 0.0])])
    lam = np.empty((1, 2))
    k = np.empty((1, 2, 2))
    alpha = np.empty((1, 2))
    interface_eigensystem(u, G, lam, k, alpha)
    rebuilt = sum(alpha[0, m] * k[0, m] for m in range(2))
    assert rebuilt == pytest.approx(u[1] - u[0], rel=1e-10, abs=1e-12)


def test_roe_average_reduces_to_the_state():
    u = np.vstack([_cons([1.3, 0.7]), _cons([1.3, 0.7])])
    u_t, a_t = roe_average(u[0], u[1], G)
    assert u_t == pytest.approx(0.7)
    assert a_t == pytest.approx(np.sqrt(G * 1.3))
