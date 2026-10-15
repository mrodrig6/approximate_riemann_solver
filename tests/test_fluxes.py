"""Tests for the numerical flux functions.

Most of these are *identity* tests rather than comparisons against stored
numbers: each flux has defining algebraic properties, and checking those pins
the implementation far more tightly than a regression baseline would.
"""

from __future__ import annotations

import numpy as np
import pytest

from riemann import get_problem, solve_exact, solve_roe
from riemann.fluxes import (
    FLUX_NAMES,
    FLUXES,
    ROBUST_FLUXES,
    _m4_minus,
    _m4_plus,
    _p5_minus,
    _p5_plus,
    ausm_plus_flux,
    flux_code,
    hll_flux,
    hllc_flux,
    lax_friedrichs_flux,
    rusanov_flux,
)
from riemann.thermo import euler_flux, primitive_to_conservative

GAMMA = 1.4

#: Every flux by canonical name.
ALL_FLUXES = sorted(set(FLUX_NAMES.values()))

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
    return primitive_to_conservative(np.asarray(w, dtype=float), GAMMA)


def _evaluate(name, u_l, u_r, dx=0.01, dt=0.001):
    """Call one flux by name and return the result as a fresh array."""
    out = np.empty(3)
    if name == "roe":
        sol = None  # the Roe flux needs its eigenstructure; exercised via solve_roe
        raise NotImplementedError(sol)
    if name == "hll":
        hll_flux(u_l, u_r, GAMMA, out)
    elif name == "hllc":
        hllc_flux(u_l, u_r, GAMMA, out)
    elif name == "rusanov":
        rusanov_flux(u_l, u_r, GAMMA, out)
    elif name == "lax_friedrichs":
        lax_friedrichs_flux(u_l, u_r, GAMMA, dx, dt, out)
    elif name == "ausm":
        ausm_plus_flux(u_l, u_r, GAMMA, out)
    else:
        raise AssertionError(f"unhandled flux {name}")
    return out


DIRECT_FLUXES = ["hll", "hllc", "rusanov", "lax_friedrichs", "ausm"]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_flux_code_accepts_names_aliases_and_integers():
    assert flux_code("hllc") == flux_code("HLLC") == FLUXES["hllc"]
    # Aliases must resolve to the same scheme.
    assert flux_code("hlle") == flux_code("hll")
    assert flux_code("llf") == flux_code("rusanov")
    assert flux_code("lf") == flux_code("lax_friedrichs")
    assert flux_code("AUSM+") == flux_code("ausm")
    assert flux_code(flux_code("roe")) == flux_code("roe")
    with pytest.raises(ValueError, match="unknown flux"):
        flux_code("not-a-flux")
    with pytest.raises(ValueError, match="unknown flux code"):
        flux_code(99)


# --------------------------------------------------------------------------- #
# Consistency: F(U, U) == F(U)
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


@pytest.mark.parametrize("name", ["hll", "hllc", "ausm"])
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
# AUSM+ split-function identities
# --------------------------------------------------------------------------- #

def test_ausm_split_mach_is_consistent():
    """``M+_(4)(M) + M-_(4)(M) = M`` exactly, for every M.

    This identity holds for any beta -- the cross terms cancel -- and is what
    makes the Mach splitting consistent.  It pins the polynomial forms.
    """
    for m in np.linspace(-3.0, 3.0, 601):
        total = _m4_plus(float(m), 0.125) + _m4_minus(float(m), 0.125)
        assert total == pytest.approx(float(m), abs=1e-14)


def test_ausm_pressure_splitting_is_consistent():
    """``P+_(5)(M) + P-_(5)(M) = 1`` exactly, for every M.

    Again independent of alpha.  Together with the Mach identity above this
    is a strong check that the fifth-degree polynomials are the right ones.
    """
    for m in np.linspace(-3.0, 3.0, 601):
        if abs(m) < 1e-12:
            continue
        total = _p5_plus(float(m), 0.1875) + _p5_minus(float(m), 0.1875)
        assert total == pytest.approx(1.0, abs=1e-14)


def test_ausm_pressure_splitting_is_non_negative():
    """The pressure weights are a convex combination, so both are in [0, 1]."""
    for m in np.linspace(-3.0, 3.0, 601):
        if abs(m) < 1e-12:
            continue
        assert -1e-14 <= _p5_plus(float(m), 0.1875) <= 1.0 + 1e-14
        assert -1e-14 <= _p5_minus(float(m), 0.1875) <= 1.0 + 1e-14


# --------------------------------------------------------------------------- #
# Contact resolution -- the property that separates the families
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["roe", "hllc", "ausm"])
def test_contact_preserving_fluxes_are_exact_on_a_stationary_contact(name):
    """Roe, HLLC and AUSM+ carry a stationary contact with zero dissipation.

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


@pytest.mark.parametrize("name", ["roe", "hllc", "ausm"])
def test_moving_contact_keeps_pressure_constant(name):
    """A contact advected at uniform speed must not generate pressure.

    This is the test that caught an unstable reconstruction of AUSM+-up's
    ``p_u`` term: it amplified round-off by ~1.5 per step until the pressure
    error reached 50%.  Plain AUSM+ passes to machine precision.
    """
    sol = solve_roe(
        GAMMA, [1.0, 1.0, 1.0], [0.125, 1.0, 1.0], 0.3, nx=100,
        flux=name, second_order=False,
    )
    assert np.max(np.abs(sol.p - 1.0)) < 1e-12


# --------------------------------------------------------------------------- #
# Behaviour of the full scheme
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL_FLUXES)
def test_uniform_flow_is_a_fixed_point(name):
    w = np.array([1.2, 0.8, 3.4])
    sol = solve_roe(GAMMA, w, w, 0.3, nx=64, flux=name)
    assert sol.rho == pytest.approx(np.full(64, w[0]))
    assert sol.u == pytest.approx(np.full(64, w[1]))
    assert sol.p == pytest.approx(np.full(64, w[2]))


@pytest.mark.parametrize("name", ALL_FLUXES)
def test_every_flux_conserves_mass_and_energy(name):
    """Every flux is in conservation form, so the totals obey the boundary flux."""
    prob = get_problem("sod")
    t_end, nx = 0.15, 100
    if name == "ausm":
        pytest.skip("AUSM+ cannot start a flow from rest; covered separately")
    sol = solve_roe(prob.gamma, prob.left, prob.right, t_end, nx=nx, flux=name)
    initial = primitive_to_conservative(
        np.where((sol.x < 0.5)[:, None], prob.left[None, :], prob.right[None, :]),
        prob.gamma,
    )
    final = sol.conserved.sum(axis=0) * sol.dx
    start = initial.sum(axis=0) * sol.dx
    assert final[0] == pytest.approx(start[0], rel=1e-12)
    assert final[2] == pytest.approx(start[2], rel=1e-12)
    impulse = (prob.left[2] - prob.right[2]) * t_end
    assert final[1] - start[1] == pytest.approx(impulse, rel=1e-10)


@pytest.mark.parametrize("name", sorted(FLUX_NAMES[c] for c in ROBUST_FLUXES))
def test_robust_fluxes_run_every_problem(name):
    """The fluxes advertised as robust must actually run the whole catalogue.

    Including Toro's Test 2, the near-vacuum case where Roe fails outright.
    """
    from riemann.problems import PROBLEMS

    for prob in PROBLEMS.values():
        sol = solve_roe(
            prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
            x_diaphragm=prob.x_diaphragm, flux=name,
        )
        assert np.all(sol.rho > 0), (name, prob.name)
        assert np.all(sol.p > 0), (name, prob.name)


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


def test_accuracy_ordering_on_sod():
    """Roe < HLLC < HLL < Rusanov < Lax-Friedrichs in L1 error.

    This ordering follows directly from how much of the wave structure each
    flux resolves, and it is the single clearest summary of what the fluxes
    differ in.  AUSM+ is excluded: it cannot run a problem started from rest.
    """
    prob = get_problem("sod")
    exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=200)
    err = {}
    for name in ("roe", "hllc", "hll", "rusanov", "lax_friedrichs"):
        sol = solve_roe(
            prob.gamma, prob.left, prob.right, prob.t_end, nx=200, flux=name
        )
        err[name] = np.mean(np.abs(sol.rho - exact.rho))
    assert (
        err["roe"] < err["hllc"] < err["hll"]
        < err["rusanov"] < err["lax_friedrichs"]
    )


def test_ausm_cannot_start_from_rest_but_runs_moving_flow():
    """Characterisation test for the AUSM+ low-Mach limitation.

    Pins the documented behaviour so that adding the AUSM+-up ``M_p`` term
    later is noticed here.
    """
    with pytest.raises(FloatingPointError, match="non-physical"):
        solve_roe(GAMMA, [1.0, 0.0, 1.0], [0.125, 0.0, 0.1], 0.25, nx=200, flux="ausm")
    # The same problem translated to a moving frame runs fine.
    sol = solve_roe(
        GAMMA, [1.0, 0.5, 1.0], [0.125, 0.5, 0.1], 0.25, nx=200, flux="ausm"
    )
    assert np.all(sol.p > 0)


def test_lax_friedrichs_odd_even_decoupling_disables_the_limiter():
    """Classical Lax-Friedrichs decouples the grid into two sub-grids.

    Its dissipation acts on a 2*dx stencil, so the solution develops identical
    adjacent cell pairs.  Every interface then has either a zero local wave
    strength or a zero upwind one, which drives the limiter ratio to zero and
    makes the second-order correction vanish identically -- so for this flux
    alone, second order is indistinguishable from first.
    """
    kwargs = dict(gamma=GAMMA, w_l=[1.0, 0.0, 1.0], w_r=[0.125, 0.0, 0.1],
                  t_end=0.25, nx=200, flux="lax_friedrichs")
    second = solve_roe(second_order=True, **kwargs)
    first = solve_roe(second_order=False, **kwargs)
    assert np.max(np.abs(second.rho - first.rho)) < 1e-12
    # The signature of the decoupling: adjacent cells agree in pairs.  Which
    # of the two phases the pairs start on depends on where the diaphragm
    # falls, so accept either.
    even = np.max(np.abs(first.rho[0:-1:2] - first.rho[1::2]))
    odd = np.max(np.abs(first.rho[1:-1:2] - first.rho[2::2]))
    assert min(even, odd) < 1e-12
    # A local-dissipation flux does not do this.
    rus = solve_roe(**{**kwargs, "flux": "rusanov"}, second_order=True)
    rus1 = solve_roe(**{**kwargs, "flux": "rusanov"}, second_order=False)
    assert np.max(np.abs(rus.rho - rus1.rho)) > 1e-3


@pytest.mark.parametrize("name", ["roe", "hllc", "hll", "rusanov"])
def test_second_order_beats_first_for_each_flux(name):
    prob = get_problem("sod")
    exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=200)
    kw = dict(nx=200, flux=name)
    second = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end,
                       second_order=True, **kw)
    first = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end,
                      second_order=False, **kw)
    assert np.mean(np.abs(second.rho - exact.rho)) < np.mean(
        np.abs(first.rho - exact.rho)
    )


@pytest.mark.parametrize("name", ALL_FLUXES)
def test_labels_are_distinct_and_descriptive(name):
    sol = solve_roe(GAMMA, [1.0, 1.0, 1.0], [0.5, 1.0, 0.8], 0.05, nx=32, flux=name)
    assert sol.flux == name
    assert "order" in sol.label
