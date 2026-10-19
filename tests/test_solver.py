"""Properties of the finite-volume scheme, across all fluxes.

Where :mod:`test_fluxes` checks the flux functions themselves, these check
the scheme built on them: conservation, the time step, convergence, the order
of accuracy, the entropy fix and argument handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import get_problem, solve_exact, solve_roe
from src.euler.fluxes import FLUX_NAMES, ROBUST_FLUXES
from src.euler.problems import PROBLEMS
from src.euler.solver import N_GHOST
from src.euler.thermo import primitive_to_conservative

GAMMA = 1.4

#: Every flux, by canonical name.
ALL_FLUXES = sorted(set(FLUX_NAMES.values()))


# --------------------------------------------------------------------------- #
# Conservation and basic invariants
# --------------------------------------------------------------------------- #

def test_scheme_is_conservative():
    """Each conserved quantity changes only by its net flux through the ends.

    The run stops before any wave reaches an end of the domain, so the gas
    there is still.  With ``u = 0`` the mass flux ``rho*u`` and the energy flux
    ``u(E + p)`` both vanish, so total mass and energy must be unchanged to
    round-off.  The momentum flux is ``rho*u^2 + p = p``, which does *not*
    vanish, so total momentum changes at the exact rate ``p_L - p_R``.  Testing
    that predicted rate is a far stronger check than testing a bare sum.
    """
    prob = get_problem("sod")
    t_end, nx = 0.15, 200
    roe = solve_roe(prob.gamma, prob.left, prob.right, t_end, nx=nx)
    dx = roe.dx
    initial = primitive_to_conservative(
        np.where((roe.x < 0.5)[:, None], prob.left[None, :], prob.right[None, :]),
        prob.gamma,
    )
    final_totals = roe.conserved.sum(axis=0) * dx
    initial_totals = initial.sum(axis=0) * dx

    # Mass and energy: exactly conserved.
    assert final_totals[0] == pytest.approx(initial_totals[0], rel=1e-12)
    assert final_totals[2] == pytest.approx(initial_totals[2], rel=1e-12)

    # Momentum: impulse delivered by the boundary pressures.
    impulse = (prob.left[2] - prob.right[2]) * t_end
    assert final_totals[1] - initial_totals[1] == pytest.approx(impulse, rel=1e-10)

@pytest.mark.parametrize("name", ALL_FLUXES)
def test_every_flux_conserves_mass_and_energy(name):
    """Every flux is in conservation form, so the totals obey the boundary flux."""
    prob = get_problem("sod")
    t_end, nx = 0.15, 100
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

def test_uniform_flow_is_preserved_exactly():
    """A constant state must be a fixed point of the scheme."""
    w = np.array([1.2, 0.8, 3.4])
    roe = solve_roe(1.4, w, w, 0.3, nx=64)
    assert roe.rho == pytest.approx(np.full(64, w[0]))
    assert roe.u == pytest.approx(np.full(64, w[1]))
    assert roe.p == pytest.approx(np.full(64, w[2]))

@pytest.mark.parametrize("name", ALL_FLUXES)
def test_uniform_flow_is_a_fixed_point(name):
    w = np.array([1.2, 0.8, 3.4])
    sol = solve_roe(GAMMA, w, w, 0.3, nx=64, flux=name)
    assert sol.rho == pytest.approx(np.full(64, w[0]))
    assert sol.u == pytest.approx(np.full(64, w[1]))
    assert sol.p == pytest.approx(np.full(64, w[2]))

def test_solution_lands_exactly_on_the_output_time():
    """The last step must be truncated, not allowed to overshoot."""
    prob = get_problem("sod")
    roe = solve_roe(prob.gamma, prob.left, prob.right, 0.1234, nx=100)
    assert roe.t == pytest.approx(0.1234, rel=1e-14)


# --------------------------------------------------------------------------- #
# Accuracy and order
# --------------------------------------------------------------------------- #

def test_isolated_shock_is_captured_sharply():
    """Roe's U-property means a single shock is resolved in very few cells."""
    prob = get_problem("air_shock_tube")
    roe = solve_roe(
        prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
        x_diaphragm=prob.x_diaphragm, limiter="superbee",
    )
    # Count cells whose density lies strictly between the two plateau values.
    lo, hi = prob.right[0], prob.left[0]
    span = hi - lo
    interior = np.sum((roe.rho > lo + 0.02 * span) & (roe.rho < hi - 0.02 * span))
    assert interior <= 3

def test_second_order_beats_first_order():
    prob = get_problem("sod")
    reference = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=200)
    errors = {}
    for label, kwargs in [
        ("first", dict(second_order=False)),
        ("minmod", dict(limiter="minmod")),
        ("superbee", dict(limiter="superbee")),
    ]:
        roe = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=200, **kwargs)
        errors[label] = np.mean(np.abs(roe.rho - reference.rho))
    # Superbee is the most compressive, minmod the most diffusive of the two.
    assert errors["superbee"] < errors["minmod"] < errors["first"]

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

def test_error_decreases_under_refinement():
    prob = get_problem("sod")
    previous = np.inf
    for nx in (50, 100, 200, 400):
        roe = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=nx)
        exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=nx)
        err = np.mean(np.abs(roe.rho - exact.rho))
        assert err < previous
        previous = err

def test_accuracy_ordering_on_sod():
    """Roe < HLLC < HLL < Rusanov < Lax-Friedrichs in L1 error.

    This ordering follows directly from how much of the wave structure each
    flux resolves, and it is the single clearest summary of what the fluxes
    differ in.  AUSM+-up is excluded because it is not a wave-based scheme and
    does not slot into the same ordering.
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


# --------------------------------------------------------------------------- #
# Entropy fix
# --------------------------------------------------------------------------- #

def test_entropy_fix_removes_the_sonic_glitch():
    """Without the fix the error at the sonic point does not converge.

    That non-convergence, rather than the size of the error at any one
    resolution, is the signature of an entropy-violating scheme.
    """
    prob = get_problem("sonic_rarefaction")
    kwargs = dict(
        gamma=prob.gamma, w_l=prob.left, w_r=prob.right, t_end=prob.t_end,
        x_diaphragm=prob.x_diaphragm, second_order=False,
    )
    errors = {True: [], False: []}
    for nx in (100, 400):
        exact = solve_exact(
            prob.gamma, prob.left, prob.right, prob.t_end, n=nx,
            x_diaphragm=prob.x_diaphragm,
        )
        for fix in (True, False):
            roe = solve_roe(nx=nx, entropy_fix=fix, **kwargs)
            # Restrict to the rarefaction fan, away from the contact and shock.
            inside = (roe.x > 0.0) & (roe.x < 0.45)
            errors[fix].append(np.max(np.abs(roe.rho - exact.rho)[inside]))

    # With the fix, refinement helps; without it, the glitch persists.
    assert errors[True][1] < 0.75 * errors[True][0]
    assert errors[False][1] > 0.9 * errors[False][0]
    assert errors[True][1] < errors[False][1]


# --------------------------------------------------------------------------- #
# Robustness across the problem catalogue
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["sod", "toro3", "toro4", "toro5",
                                  "sonic_rarefaction", "air_shock_tube"])
def test_every_runnable_problem_stays_physical(name):
    prob = get_problem(name)
    roe = solve_roe(
        prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
        x_diaphragm=prob.x_diaphragm,
    )
    assert np.all(roe.rho > 0), name
    assert np.all(roe.p > 0), name
    assert np.all(np.isfinite(roe.e)), name

@pytest.mark.parametrize("name", sorted(FLUX_NAMES[c] for c in ROBUST_FLUXES))
def test_robust_fluxes_run_every_problem(name):
    """The fluxes advertised as robust must actually run the whole catalogue.

    Including Toro's Test 2, the near-vacuum case where Roe fails outright.
    """
    for prob in PROBLEMS.values():
        sol = solve_roe(
            prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
            x_diaphragm=prob.x_diaphragm, flux=name,
        )
        assert np.all(sol.rho > 0), (name, prob.name)
        assert np.all(sol.p > 0), (name, prob.name)

def test_toro2_fails_as_documented():
    """Roe's scheme is known to fail on the near-vacuum '123' problem.

    This is a *characterisation* test: it pins down a documented limitation of
    the method (Einfeldt et al. 1991; Toro Sec. 11.4), so that if a future
    positivity-preserving flux (HLLE/HLLC) is added, this test is the reminder
    to revisit it.
    """
    prob = get_problem("toro2")
    assert prob.flux_note, "toro2 must carry an explanatory note"
    with pytest.raises(FloatingPointError, match="non-physical state"):
        solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=200)


# --------------------------------------------------------------------------- #
# Arguments and grid layout
# --------------------------------------------------------------------------- #

def test_invalid_arguments_are_rejected():
    w_l, w_r = np.array([1.0, 0.0, 1.0]), np.array([0.125, 0.0, 0.1])
    with pytest.raises(ValueError, match="t_end"):
        solve_roe(1.4, w_l, w_r, 0.0)
    with pytest.raises(ValueError, match="nx"):
        solve_roe(1.4, w_l, w_r, 0.1, nx=2)
    with pytest.raises(ValueError, match="cfl"):
        solve_roe(1.4, w_l, w_r, 0.1, cfl=1.5)
    with pytest.raises(ValueError, match="positive density"):
        solve_roe(1.4, np.array([0.0, 0.0, 1.0]), w_r, 0.1)

@pytest.mark.parametrize("name", ALL_FLUXES)
def test_labels_are_distinct_and_descriptive(name):
    sol = solve_roe(GAMMA, [1.0, 1.0, 1.0], [0.5, 1.0, 0.8], 0.05, nx=32, flux=name)
    assert sol.flux == name
    assert "order" in sol.label

def test_ghost_cell_count_is_enough_for_the_limiter_stencil():
    """The limiter reads one interface beyond the first interior interface."""
    assert N_GHOST >= 2
