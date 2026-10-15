"""Properties of the individual numerical flux functions.

Mostly *identity* tests rather than stored-number comparisons: each flux has
defining algebraic or physical properties, and checking those pins the
implementation far more tightly than a regression baseline would.
"""

from __future__ import annotations

import numpy as np
import pytest

from riemann import get_problem, solve_exact, solve_roe
from riemann.fluxes import (
    FLUXES,
    ausm_up_flux,
    flux_code,
    hll_flux,
    hllc_flux,
    interface_eigensystem,
    lax_friedrichs_flux,
    m4_minus,
    m4_plus,
    p5_minus,
    p5_plus,
    roe_average,
    rusanov_flux,
)
from riemann.thermo import euler_flux, primitive_to_conservative

GAMMA = 1.4

#: Fluxes callable directly, without an eigensystem argument.
DIRECT_FLUXES = ["hll", "hllc", "rusanov", "lax_friedrichs", "ausmup"]

#: A spread of states for the consistency checks: at rest, subsonic,
#: supersonic both ways, strong and weak.
SAMPLE_STATES = [
    (1.0, 0.0, 1.0),
    (0.125, 0.0, 0.1),
    (1.2, 0.75, 2.0),
    (0.4, -1.9, 0.3),
    (5.99924, 19.5975, 460.894),
    (1.0, -19.59745, 1000.0),
]


def _cons(w):
    """Conservative state from a primitive triple."""
    return primitive_to_conservative(np.asarray(w, dtype=float), GAMMA)


def _pair(w_l, w_r):
    """A two-cell conservative array from two primitive states."""
    return np.vstack([_cons(w_l), _cons(w_r)])


def _evaluate(name, u_l, u_r, dx=0.01, dt=0.001):
    """Call one flux by name and return the result as a fresh array."""
    out = np.empty(3)
    if name == "hll":
        hll_flux(u_l, u_r, GAMMA, out)
    elif name == "hllc":
        hllc_flux(u_l, u_r, GAMMA, out)
    elif name == "rusanov":
        rusanov_flux(u_l, u_r, GAMMA, out)
    elif name == "lax_friedrichs":
        lax_friedrichs_flux(u_l, u_r, GAMMA, dx, dt, out)
    elif name == "ausmup":
        ausm_up_flux(u_l, u_r, GAMMA, out)
    else:
        raise AssertionError(f"unhandled flux {name}")
    return out


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_flux_code_accepts_names_aliases_and_integers():
    assert flux_code("hllc") == flux_code("HLLC") == FLUXES["hllc"]
    # Aliases must resolve to the same scheme.
    assert flux_code("hlle") == flux_code("hll")
    assert flux_code("llf") == flux_code("rusanov")
    assert flux_code("lf") == flux_code("lax_friedrichs")
    assert flux_code("AUSM+-up") == flux_code("ausmup") == flux_code("ausm")
    assert flux_code(flux_code("roe")) == flux_code("roe")
    with pytest.raises(ValueError, match="unknown flux"):
        flux_code("not-a-flux")
    with pytest.raises(ValueError, match="unknown flux code"):
        flux_code(99)


# --------------------------------------------------------------------------- #
# Consistency and upwinding
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", DIRECT_FLUXES)
@pytest.mark.parametrize("w", SAMPLE_STATES)
def test_flux_is_consistent(name, w):
    """Every numerical flux must reduce to the physical flux when U_L == U_R.

    This is the minimum requirement for consistency of the scheme, and it
    catches most algebra slips immediately.
    """
    u = _cons(w)
    assert _evaluate(name, u, u) == pytest.approx(euler_flux(u, GAMMA), rel=1e-12,
                                                  abs=1e-12)

@pytest.mark.parametrize("name", ["hll", "hllc", "ausmup"])
def test_upwind_fluxes_are_exact_when_supersonic(name):
    """At a fully supersonic interface an upwind flux is the upwind physical one.

    HLL and HLLC select the physical flux outright once both wave speeds share
    a sign; AUSM+ gets there through its split functions, since ``M+_(4)`` and
    ``P+_(5)`` collapse to the one-sided values for ``|M| >= 1``.

    Rusanov and Lax-Friedrichs are deliberately excluded: they are *central*
    fluxes with added scalar dissipation, so they keep a
    ``-0.5 S (U_R - U_L)`` term even when every wave travels the same way.
    That is precisely why they remain dissipative in supersonic flow.
    """
    # Supersonic to the right: every wave travels right, so F = F(U_L).
    u_l, u_r = _cons((1.0, 5.0, 1.0)), _cons((0.8, 5.2, 0.9))
    assert _evaluate(name, u_l, u_r) == pytest.approx(
        euler_flux(u_l, GAMMA), rel=1e-10
    )
    # Supersonic to the left: F = F(U_R).
    u_l, u_r = _cons((1.0, -5.0, 1.0)), _cons((0.8, -5.2, 0.9))
    assert _evaluate(name, u_l, u_r) == pytest.approx(
        euler_flux(u_r, GAMMA), rel=1e-10
    )

@pytest.mark.parametrize("name", ["rusanov", "lax_friedrichs"])
def test_central_fluxes_stay_dissipative_when_supersonic(name):
    """The complement: Rusanov and LF do not reduce to the upwind flux."""
    u_l, u_r = _cons((1.0, 5.0, 1.0)), _cons((0.8, 5.2, 0.9))
    assert np.max(
        np.abs(_evaluate(name, u_l, u_r) - euler_flux(u_l, GAMMA))
    ) > 1e-6


# --------------------------------------------------------------------------- #
# Roe: the U-property and the wave decomposition
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "w_l,w_r",
    [
        ([1.0, 0.0, 1.0], [0.125, 0.0, 0.1]),
        ([5.99924, 19.5975, 460.894], [5.99242, -6.19633, 46.0950]),
        ([1.0, -1.3, 2.7], [0.4, 0.9, 0.3]),
        ([2.0, 0.0, 3.0], [2.0, 0.0, 3.0]),  # no jump at all
    ],
)
def test_u_property(w_l, w_r):
    """``F(U_R) - F(U_L) = sum_k alpha_k lambda_k K_k`` exactly (Toro eq. 11.18).

    This is the single most important property of the Roe average; if the
    wave strengths or the eigenvectors are wrong, this breaks.
    """
    u = _pair(w_l, w_r)
    lam = np.empty((1, 3))
    k = np.empty((1, 3, 3))
    alpha = np.empty((1, 3))
    interface_eigensystem(u, GAMMA, lam, k, alpha)

    reconstructed = sum(alpha[0, m] * lam[0, m] * k[0, m] for m in range(3))
    expected = euler_flux(u[1], GAMMA) - euler_flux(u[0], GAMMA)
    assert reconstructed == pytest.approx(expected, rel=1e-10, abs=1e-10)

@pytest.mark.parametrize(
    "w_l,w_r",
    [
        ([1.0, 0.0, 1.0], [0.125, 0.0, 0.1]),
        ([1.0, 2.0, 5.0], [3.0, -1.0, 0.5]),
    ],
)
def test_wave_strengths_reconstruct_the_jump(w_l, w_r):
    """``U_R - U_L = sum_k alpha_k K_k`` (Toro eqs. 11.68-11.70)."""
    u = _pair(w_l, w_r)
    lam = np.empty((1, 3))
    k = np.empty((1, 3, 3))
    alpha = np.empty((1, 3))
    interface_eigensystem(u, GAMMA, lam, k, alpha)
    reconstructed = sum(alpha[0, m] * k[0, m] for m in range(3))
    assert reconstructed == pytest.approx(u[1] - u[0], rel=1e-10, abs=1e-12)

def test_roe_average_reduces_to_the_state_when_states_are_equal():
    w = [1.3, 0.7, 2.1]
    u = _pair(w, w)
    u_t, h_t, a_t = roe_average(u[0], u[1], GAMMA)
    assert u_t == pytest.approx(w[1])
    assert a_t == pytest.approx(np.sqrt(GAMMA * w[2] / w[0]))


# --------------------------------------------------------------------------- #
# AUSM+ split-function identities
# --------------------------------------------------------------------------- #

def test_ausm_split_mach_is_consistent():
    """``M+_(4)(M) + M-_(4)(M) = M`` exactly, for every M.

    This identity holds for any beta -- the cross terms cancel -- and is what
    makes the Mach splitting consistent.  It pins the polynomial forms.
    """
    for m in np.linspace(-3.0, 3.0, 601):
        total = m4_plus(float(m), 0.125) + m4_minus(float(m), 0.125)
        assert total == pytest.approx(float(m), abs=1e-14)

def test_ausm_pressure_splitting_is_consistent():
    """``P+_(5)(M) + P-_(5)(M) = 1`` exactly, for every M.

    Again independent of alpha.  Together with the Mach identity above this
    is a strong check that the fifth-degree polynomials are the right ones.
    """
    for m in np.linspace(-3.0, 3.0, 601):
        if abs(m) < 1e-12:
            continue
        total = p5_plus(float(m), 0.1875) + p5_minus(float(m), 0.1875)
        assert total == pytest.approx(1.0, abs=1e-14)

def test_ausm_pressure_splitting_is_non_negative():
    """The pressure weights are a convex combination, so both are in [0, 1]."""
    for m in np.linspace(-3.0, 3.0, 601):
        if abs(m) < 1e-12:
            continue
        assert -1e-14 <= p5_plus(float(m), 0.1875) <= 1.0 + 1e-14
        assert -1e-14 <= p5_minus(float(m), 0.1875) <= 1.0 + 1e-14


# --------------------------------------------------------------------------- #
# Contact resolution -- the property that separates the families
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["roe", "hllc", "ausmup"])
def test_contact_preserving_fluxes_are_exact_on_a_stationary_contact(name):
    """Roe, HLLC and AUSM+-up carry a stationary contact with zero dissipation.

    A contact has uniform ``u`` and ``p`` and a jump only in ``rho``.  A flux
    that resolves the middle wave leaves it untouched for all time.
    """
    sol = solve_roe(
        GAMMA, [1.0, 0.0, 1.0], [0.125, 0.0, 1.0], 0.4, nx=100,
        flux=name, second_order=False,
    )
    assert np.max(np.abs(sol.p - 1.0)) < 1e-12
    assert np.max(np.abs(sol.u)) < 1e-12
    # Density must still be a clean two-plateau step: no intermediate cells.
    between = np.sum((sol.rho > 0.1425) & (sol.rho < 0.9825))
    assert between == 0

@pytest.mark.parametrize("name", ["hll", "rusanov", "lax_friedrichs"])
def test_two_wave_fluxes_smear_a_stationary_contact(name):
    """HLL, Rusanov and Lax-Friedrichs have no middle wave, so they smear it.

    The complement of the test above: it documents the real cost of dropping
    the contact from the approximate wave structure.
    """
    sol = solve_roe(
        GAMMA, [1.0, 0.0, 1.0], [0.125, 0.0, 1.0], 0.4, nx=100,
        flux=name, second_order=False,
    )
    between = np.sum((sol.rho > 0.1425) & (sol.rho < 0.9825))
    assert between > 10

@pytest.mark.parametrize("name", ["roe", "hllc", "ausmup"])
def test_moving_contact_keeps_pressure_constant(name):
    """A contact advected at uniform speed must not generate pressure.

    This is the test that caught an unstable reconstruction of AUSM+-up's
    ``K_u`` term: with that term 7.5x too large it amplified round-off by
    about 1.5 per step until the pressure error reached 50%.  With the
    coefficients adapted from EMS it passes to machine precision.
    """
    sol = solve_roe(
        GAMMA, [1.0, 1.0, 1.0], [0.125, 1.0, 1.0], 0.3, nx=100,
        flux=name, second_order=False,
    )
    assert np.max(np.abs(sol.p - 1.0)) < 1e-12


# --------------------------------------------------------------------------- #
# What each flux can and cannot do
# --------------------------------------------------------------------------- #

def test_hllc_solves_the_problem_roe_cannot():
    """Toro's Test 2 is the headline case for a positivity-preserving flux."""
    prob = get_problem("toro2")
    with pytest.raises(FloatingPointError):
        solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=200, flux="roe")
    sol = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=200, flux="hllc")
    exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=200)
    # The star region is near vacuum, so judge on a relative error.
    rel = np.mean(np.abs(sol.rho - exact.rho)) / np.max(exact.rho)
    assert rel < 0.02

def test_ausm_up_starts_from_rest():
    """The ``K_p`` term is what lets AUSM+-up start a flow from rest.

    Plain AUSM+ (no ``K_p``) has a mass flux that is identically zero when
    ``u_L = u_R = 0``, whatever the pressure jump, and cannot run Sod's
    problem at all.  The pressure-diffusion term supplies the missing
    pressure-velocity coupling.  This test is the guard against that term
    being dropped or mis-scaled.
    """
    prob = get_problem("sod")
    sol = solve_roe(
        prob.gamma, prob.left, prob.right, prob.t_end, nx=200, flux="ausmup"
    )
    assert np.all(sol.p > 0)
    exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=200)
    assert np.mean(np.abs(sol.rho - exact.rho)) < 0.02

def test_ausm_up_runs_every_problem_at_first_order():
    """The AUSM+-up flux itself handles the whole catalogue.

    At second order it fails on the three most extreme cases (`toro2`,
    `toro3`, `toro5`), exactly as Rusanov does on `toro5`: the failure is in
    the characteristic-based TVD correction this package pairs it with, not
    in the flux.  EMS pairs AUSM with WENO reconstruction instead.
    """
    from riemann.problems import PROBLEMS

    for prob in PROBLEMS.values():
        sol = solve_roe(
            prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
            x_diaphragm=prob.x_diaphragm, flux="ausmup", second_order=False,
        )
        assert np.all(sol.rho > 0), prob.name
        assert np.all(sol.p > 0), prob.name
