"""Tests for the Roe flux, the limiters and the finite-volume driver."""

from __future__ import annotations

import numpy as np
import pytest

from riemann import get_problem, solve_exact, solve_roe
from riemann.limiters import LIMITERS, limiter_code, limiter_phi
from riemann.roe import interface_eigensystem, roe_average
from riemann.solver import N_GHOST
from riemann.thermo import euler_flux, primitive_to_conservative

GAMMA = 1.4


def _conservative_pair(w_l, w_r):
    """Build a two-cell conservative array from two primitive states."""
    return np.vstack(
        [primitive_to_conservative(w_l, GAMMA), primitive_to_conservative(w_r, GAMMA)]
    )


# --------------------------------------------------------------------------- #
# The U-property: the defining feature of the Roe linearisation
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
    u = _conservative_pair(w_l, w_r)
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
    u = _conservative_pair(w_l, w_r)
    lam = np.empty((1, 3))
    k = np.empty((1, 3, 3))
    alpha = np.empty((1, 3))
    interface_eigensystem(u, GAMMA, lam, k, alpha)
    reconstructed = sum(alpha[0, m] * k[0, m] for m in range(3))
    assert reconstructed == pytest.approx(u[1] - u[0], rel=1e-10, abs=1e-12)


def test_roe_average_reduces_to_the_state_when_states_are_equal():
    w = [1.3, 0.7, 2.1]
    u = _conservative_pair(w, w)
    u_t, h_t, a_t = roe_average(u[0], u[1], GAMMA)
    assert u_t == pytest.approx(w[1])
    assert a_t == pytest.approx(np.sqrt(GAMMA * w[2] / w[0]))


# --------------------------------------------------------------------------- #
# Limiters
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", [n for n in LIMITERS if n != "none"])
def test_limiters_are_inside_the_sweby_tvd_region(name):
    """``0 <= phi(r) <= min(2r, 2)`` for ``r > 0``, and ``phi = 0`` for ``r <= 0``."""
    code = limiter_code(name)
    for r in np.linspace(-3.0, 6.0, 901):
        phi = limiter_phi(float(r), code)
        if r <= 0.0:
            assert phi == 0.0
        else:
            assert 0.0 <= phi <= min(2.0 * r, 2.0) + 1e-12, (name, r, phi)


@pytest.mark.parametrize("name", [n for n in LIMITERS if n != "none"])
def test_limiters_are_second_order_at_r_equals_one(name):
    """Every limiter must satisfy ``phi(1) = 1`` to recover second order."""
    assert limiter_phi(1.0, limiter_code(name)) == pytest.approx(1.0)


def test_limiter_code_accepts_names_and_integers():
    assert limiter_code("superbee") == limiter_code(0) == 0
    assert limiter_code("MinMod") == limiter_code("min_mod") == 1
    with pytest.raises(ValueError):
        limiter_code("not-a-limiter")
    with pytest.raises(ValueError):
        limiter_code(99)


# --------------------------------------------------------------------------- #
# Conservation, accuracy and robustness of the full scheme
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


def test_uniform_flow_is_preserved_exactly():
    """A constant state must be a fixed point of the scheme."""
    w = np.array([1.2, 0.8, 3.4])
    roe = solve_roe(1.4, w, w, 0.3, nx=64)
    assert roe.rho == pytest.approx(np.full(64, w[0]))
    assert roe.u == pytest.approx(np.full(64, w[1]))
    assert roe.p == pytest.approx(np.full(64, w[2]))


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


def test_solution_lands_exactly_on_the_output_time():
    """The MATLAB loop overshot the end time by up to one full step."""
    prob = get_problem("sod")
    roe = solve_roe(prob.gamma, prob.left, prob.right, 0.1234, nx=100)
    assert roe.t == pytest.approx(0.1234, rel=1e-14)


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


def test_error_decreases_under_refinement():
    prob = get_problem("sod")
    previous = np.inf
    for nx in (50, 100, 200, 400):
        roe = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=nx)
        exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=nx)
        err = np.mean(np.abs(roe.rho - exact.rho))
        assert err < previous
        previous = err


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


def test_ghost_cell_count_is_enough_for_the_limiter_stencil():
    """The limiter reads one interface beyond the first interior interface."""
    assert N_GHOST >= 2
