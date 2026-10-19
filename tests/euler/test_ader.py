"""The one-step ADER scheme.

The defining claims are that it is second-order accurate in space *and* time
on smooth data, that it reaches that in a single step with no Runge-Kutta
stages, and that it is a genuinely different scheme from the flux-limited
TVD one rather than a relabelling of it.  Each is checked here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import (
    acoustic_pulse,
    advected_pulse,
    cell_centres,
    get_problem,
    solve_exact,
    solve_field,
    solve,
)

GAMMA = 1.4


def _order(errors, resolutions):
    """Least-squares convergence rate from an error sequence."""
    return -np.polyfit(np.log(resolutions), np.log(errors), 1)[0]


def test_ader2_is_second_order_on_an_advected_pulse():
    """Against the analytic answer: a pulse translated at constant speed.

    Velocity and pressure are uniform, so the exact solution is the initial
    density profile shifted by ``u t``.  The limiter is off: a limiter clips
    smooth extrema and would mask the formal order.
    """
    t_end, u0 = 0.2, 1.0
    resolutions = np.array([50, 100, 200, 400])
    errors = []
    for nx in resolutions:
        x = cell_centres(int(nx))
        sol = solve_field(
            GAMMA, advected_pulse(x, velocity=u0), t_end,
            scheme="ader2", limiter="none", cfl=0.8,
        )
        exact = advected_pulse(x - u0 * t_end, velocity=u0)[:, 0]
        errors.append(np.mean(np.abs(sol.rho - exact)))
    assert _order(np.array(errors), resolutions) > 1.9


def test_ader2_is_second_order_on_a_nonlinear_pulse():
    """Against a fine-grid reference: a smooth isentropic acoustic pulse.

    This one excites the genuinely non-linear fields, where schemes of the
    same formal order actually differ.
    """
    t_end = 0.15
    ref_nx = 3200
    ref = solve_field(
        GAMMA, acoustic_pulse(cell_centres(ref_nx)), t_end,
        scheme="ader2", limiter="none", cfl=0.8,
    )
    resolutions = np.array([50, 100, 200, 400])
    errors = []
    for nx in resolutions:
        x = cell_centres(int(nx))
        sol = solve_field(
            GAMMA, acoustic_pulse(x), t_end,
            scheme="ader2", limiter="none", cfl=0.8,
        )
        errors.append(np.mean(np.abs(sol.rho - np.interp(x, ref.x, ref.rho))))
    assert _order(np.array(errors), resolutions) > 1.9


def test_ader2_differs_from_the_tvd_scheme():
    """They are distinct schemes, not two names for one.

    On a *pure contact* the two coincide exactly -- both reduce to limited
    Lax-Wendroff on the one active field -- so the discriminating test has to
    excite the non-linear fields.
    """
    x = cell_centres(200)
    kw = dict(limiter="none", cfl=0.8)
    ader = solve_field(GAMMA, acoustic_pulse(x), 0.15, scheme="ader2", **kw)
    tvd = solve_field(GAMMA, acoustic_pulse(x), 0.15, scheme="tvd", **kw)
    assert np.max(np.abs(ader.rho - tvd.rho)) > 1e-5

    # ... and on the pure contact they agree, which is a consistency check
    # rather than a coincidence.
    ader_c = solve_field(GAMMA, advected_pulse(x), 0.2, scheme="ader2", **kw)
    tvd_c = solve_field(GAMMA, advected_pulse(x), 0.2, scheme="tvd", **kw)
    assert np.max(np.abs(ader_c.rho - tvd_c.rho)) < 1e-12


def test_ader2_takes_one_step_per_step():
    """ADER is one-step: the step count is set by the CFL condition alone.

    A two-stage Runge-Kutta scheme of the same order would need twice the
    Riemann solves for the same number of time steps; ADER reaches second
    order within the single update, so its step count matches the
    first-order scheme's at the same CFL number.
    """
    prob = get_problem("sod")
    kw = dict(nx=200, cfl=0.9, limiter="minmod")
    ader = solve(prob.gamma, prob.left, prob.right, prob.t_end,
                     scheme="ader2", **kw)
    first = solve(prob.gamma, prob.left, prob.right, prob.t_end,
                      second_order=False, **kw)
    assert abs(ader.steps - first.steps) <= 1


#: The shock tubes ADER2 runs.  ``toro2`` (near-vacuum) and ``toro5`` (slowly
#: moving contact) are absent because it fails on them for every limiter; that
#: is asserted below rather than left to be noticed, so the documented
#: limitation and the test suite cannot drift apart.
ADER_SHOCK_TUBES = ["sod", "toro3", "toro4", "sonic_rarefaction", "air_shock_tube"]
ADER_FAILURES = ["toro2", "toro5"]
LIMITERS = ["superbee", "minmod", "vanleer", "vanalbada"]


@pytest.mark.parametrize("name", ADER_SHOCK_TUBES)
def test_ader2_stays_physical_on_the_shock_tubes(name):
    prob = get_problem(name)
    sol = solve(
        prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
        x_diaphragm=prob.x_diaphragm, scheme="ader2", limiter="minmod",
    )
    assert np.all(sol.rho > 0), name
    assert np.all(sol.p > 0), name


@pytest.mark.parametrize("limiter", LIMITERS)
def test_ader2_survives_a_non_physical_reconstruction(limiter):
    """A limited slope can still extrapolate to a non-physical state.

    ``toro3`` has a pressure ratio of 1e5, and on it the boundary-extrapolated
    states go non-physical at some interfaces.  Those states cannot be handed
    to the wave model at all -- it divides by the density to form the velocity
    and takes the square root of the pressure to form the sound speed -- so
    the scheme checks them and falls back to the cell averages there.

    Regression: only the *resulting* Godunov state was checked, and only its
    density, which is both too late (the division has already happened) and
    NaN-blind.  With superbee this problem used to abort.
    """
    prob = get_problem("toro3")
    sol = solve(
        prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
        x_diaphragm=prob.x_diaphragm, scheme="ader2", limiter=limiter,
    )
    assert np.all(sol.rho > 0.0)
    assert np.all(sol.p > 0.0)


@pytest.mark.parametrize("name", ADER_FAILURES)
@pytest.mark.parametrize("limiter", LIMITERS)
def test_ader2_reports_its_documented_failures_cleanly(name, limiter):
    """The two problems ADER2 cannot run must say so, for every limiter.

    ``pytest.raises(FloatingPointError)`` is the point: a bare
    ``ZeroDivisionError`` out of a kernel would not satisfy it.  If a future
    change makes either problem run, this test fails and ``docs/ader.md``
    needs updating with it.
    """
    prob = get_problem(name)
    with pytest.raises(FloatingPointError):
        solve(
            prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
            x_diaphragm=prob.x_diaphragm, scheme="ader2", limiter=limiter,
        )


def test_ader2_is_competitive_on_sod():
    """It should land in the same accuracy band as the TVD scheme.

    Not a claim that one beats the other -- at second order they are
    comparable, and ADER's advantage is the template, not the constant.
    """
    prob = get_problem("sod")
    exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=200)
    ader = solve(prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
                     scheme="ader2")
    tvd = solve(prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
                    scheme="tvd")
    e_ader = np.mean(np.abs(ader.rho - exact.rho))
    e_tvd = np.mean(np.abs(tvd.rho - exact.rho))
    assert e_ader < 3.0 * e_tvd
    assert e_ader < 0.01


def test_ader2_is_conservative():
    prob = get_problem("sod")
    t_end, nx = 0.15, 100
    sol = solve(prob.gamma, prob.left, prob.right, t_end, nx=nx,
                    scheme="ader2")
    from src.euler.physics import primitive_to_conservative

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


def test_uniform_flow_is_a_fixed_point_for_ader():
    w = np.array([1.2, 0.8, 3.4])
    sol = solve(GAMMA, w, w, 0.3, nx=64, scheme="ader2")
    assert sol.rho == pytest.approx(np.full(64, w[0]))
    assert sol.u == pytest.approx(np.full(64, w[1]))
    assert sol.p == pytest.approx(np.full(64, w[2]))


def test_unknown_scheme_is_rejected():
    with pytest.raises(ValueError, match="unknown scheme"):
        solve(GAMMA, [1.0, 0.0, 1.0], [0.125, 0.0, 0.1], 0.1, scheme="ader7")


def test_solve_field_validates_its_input():
    x = cell_centres(32)
    with pytest.raises(ValueError, match=r"shape \(nx, 3\)"):
        solve_field(GAMMA, np.ones((32, 2)), 0.1)
    with pytest.raises(ValueError, match="positive density"):
        bad = acoustic_pulse(x)
        bad[0, 0] = -1.0
        solve_field(GAMMA, bad, 0.1)


def test_ader2_galilean_sensitivity_is_the_reason_toro5_fails():
    """Pins *why* ADER2 fails Toro's test 5, so the reason cannot drift.

    Test 5 is test 3 with a uniform -19.59745 added to both velocities, and
    the Euler equations are Galilean invariant, so the two differ only by the
    frame.  This scheme nevertheless runs test 3 and fails the shifted
    version, because ``_limited_slopes`` limits componentwise on the
    conserved variables where the TVD correction limits on the characteristic
    wave strengths.  The TVD scheme, on identical data, survives every shift
    asserted here.

    If limiting ever moves to primitive or characteristic variables this test
    fails, which is the intended signal: the limitations section of
    ``docs/ader.md`` would then be describing something that no longer
    happens.
    """
    prob = get_problem("toro3")
    t_end = 0.004          # short enough to keep the waves on the grid

    def run(u0, **kw):
        w_l = np.array(prob.left, dtype=float)
        w_r = np.array(prob.right, dtype=float)
        w_l[1] += u0
        w_r[1] += u0
        return solve(prob.gamma, w_l, w_r, t_end, nx=200, limiter="minmod", **kw)

    # Unshifted, and a small shift: ADER is fine.
    assert np.all(run(0.0, scheme="ader2").p > 0.0)
    assert np.all(run(-1.0, scheme="ader2").p > 0.0)

    # The frame alone breaks it, and not symmetrically in the sign.
    for u0 in (-5.0, -19.59745):
        with pytest.raises(FloatingPointError):
            run(u0, scheme="ader2")
    assert np.all(run(+19.59745, scheme="ader2").p > 0.0)

    # The TVD scheme runs every one of those frames, which is what makes this
    # an implementation limitation rather than a property of the data.
    for u0 in (0.0, -1.0, -5.0, -19.59745, +19.59745):
        assert np.all(run(u0, flux="hllc").p > 0.0), u0
