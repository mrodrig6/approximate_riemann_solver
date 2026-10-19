"""The shallow-water finite-volume scheme.

The twin of ``tests/euler/test_solver.py``: fixed points, conservation,
agreement with the exact solution, second order beating first, and the
dry-bed dam break against Ritter's analytic solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.limiters import LIMITERS
from src.shallow_water import (
    FLUX_NAMES,
    GRAVITY,
    PROBLEMS,
    get_problem,
    primitive_to_conservative,
    solve,
    solve_exact,
    solve_field,
)

ALL_FLUXES = sorted(set(FLUX_NAMES.values()))
G = GRAVITY


@pytest.mark.parametrize("flux", ALL_FLUXES)
def test_still_water_is_a_fixed_point(flux):
    """A lake at rest must stay a lake at rest, exactly.

    The well-balancing check in its simplest form: with a flat bed, uniform
    depth and zero velocity, every flux difference must cancel.
    """
    w = np.column_stack([np.full(64, 1.7), np.zeros(64)])
    sol = solve_field(w, 0.2, flux=flux)
    assert sol.h == pytest.approx(np.full(64, 1.7))
    assert sol.u == pytest.approx(np.zeros(64), abs=1e-13)


@pytest.mark.parametrize("flux", ALL_FLUXES)
def test_uniform_flow_is_a_fixed_point(flux):
    w = np.column_stack([np.full(64, 1.2), np.full(64, 0.8)])
    sol = solve_field(w, 0.2, flux=flux)
    assert sol.h == pytest.approx(np.full(64, 1.2))
    assert sol.u == pytest.approx(np.full(64, 0.8))


def test_scheme_is_conservative():
    """Mass is conserved exactly; momentum changes by the boundary pressures.

    The run stops before any wave reaches an end, so the water there is
    still: the mass flux ``h u`` vanishes while the momentum flux is the
    hydrostatic term ``g h^2 / 2``, which does not.
    """
    prob = get_problem("dam_break")
    nx, t_end = 200, prob.t_end
    sol = solve(prob.left, prob.right, t_end, nx=nx, x_diaphragm=prob.x_diaphragm)
    initial = primitive_to_conservative(
        np.where((sol.x < 0.5)[:, None], prob.left[None, :], prob.right[None, :])
    )
    final = sol.conserved.sum(axis=0) * sol.dx
    start = initial.sum(axis=0) * sol.dx
    assert final[0] == pytest.approx(start[0], rel=1e-12)

    h_l, h_r = prob.w_l[0], prob.w_r[0]
    impulse = 0.5 * G * (h_l * h_l - h_r * h_r) * t_end
    assert final[1] - start[1] == pytest.approx(impulse, rel=1e-9)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
@pytest.mark.parametrize("flux", ALL_FLUXES)
def test_every_flux_runs_every_problem(name, flux):
    prob = get_problem(name)
    sol = solve(
        prob.left, prob.right, prob.t_end, gravity=prob.gravity, nx=200,
        x_diaphragm=prob.x_diaphragm, cfl=prob.cfl, flux=flux,
    )
    assert np.all(sol.h >= 0.0), (name, flux)
    assert np.all(np.isfinite(sol.u)), (name, flux)


@pytest.mark.parametrize("name", sorted(PROBLEMS))
def test_numerical_matches_the_exact_solution(name):
    prob = get_problem(name)
    exact = solve_exact(
        prob.left, prob.right, prob.t_end, gravity=prob.gravity, n=200,
        x_diaphragm=prob.x_diaphragm,
    )
    sol = solve(
        prob.left, prob.right, prob.t_end, gravity=prob.gravity, nx=200,
        x_diaphragm=prob.x_diaphragm, cfl=prob.cfl, flux="roe",
    )
    scale = max(np.max(np.abs(exact.h)), 1e-30)
    assert np.mean(np.abs(sol.h - exact.h)) / scale < 0.02, name


def test_second_order_beats_first_order():
    prob = get_problem("dam_break")
    exact = solve_exact(prob.left, prob.right, prob.t_end, n=200)
    kw = dict(nx=200, x_diaphragm=prob.x_diaphragm)
    second = solve(prob.left, prob.right, prob.t_end, second_order=True, **kw)
    first = solve(prob.left, prob.right, prob.t_end, second_order=False, **kw)
    assert np.mean(np.abs(second.h - exact.h)) < np.mean(np.abs(first.h - exact.h))


def test_order_of_accuracy_on_a_smooth_pulse():
    """First order converges at ~1 and the TVD scheme at ~2, as for Euler."""
    def init(x):
        return np.column_stack(
            [1.0 + 0.2 * np.exp(-200.0 * (x - 0.3) ** 2), np.full_like(x, 1.0)]
        )

    resolutions = np.array([50, 100, 200, 400])
    rates = {}
    for second in (False, True):
        ref = solve_field(
            init((np.arange(3200) + 0.5) / 3200), 0.05,
            cfl=0.8, limiter="none", second_order=second,
        )
        errors = []
        for nx in resolutions:
            x = (np.arange(int(nx)) + 0.5) / int(nx)
            sol = solve_field(
                init(x), 0.05, cfl=0.8, limiter="none", second_order=second
            )
            errors.append(np.mean(np.abs(sol.h - np.interp(x, ref.x, ref.h))))
        rates[second] = -np.polyfit(np.log(resolutions), np.log(errors), 1)[0]

    assert rates[False] > 0.85
    assert rates[True] > 1.9


def test_dry_dam_break_matches_ritter():
    """The numerical dry dam break tracks the analytic Ritter solution."""
    prob = get_problem("dam_break_dry")
    exact = solve_exact(prob.left, prob.right, prob.t_end, n=400)
    sol = solve(
        prob.left, prob.right, prob.t_end, nx=400, cfl=prob.cfl, flux="hll"
    )
    assert np.all(sol.h >= 0.0)
    assert np.mean(np.abs(sol.h - exact.h)) < 0.01
    # The depth at the dam site is the Ritter value 4 h_L / 9.
    at_dam = np.argmin(np.abs(sol.x - 0.5))
    assert sol.h[at_dam] == pytest.approx(4.0 / 9.0, rel=0.05)


def test_solver_validates_input():
    with pytest.raises(ValueError, match="non-negative"):
        solve([-1.0, 0.0], [0.1, 0.0], 0.05)
    with pytest.raises(ValueError, match="nx"):
        solve([1.0, 0.0], [0.1, 0.0], 0.05, nx=2)
    with pytest.raises(ValueError, match="cfl"):
        solve([1.0, 0.0], [0.1, 0.0], 0.05, cfl=1.5)
    with pytest.raises(ValueError, match=r"shape \(nx, 2\)"):
        solve_field(np.ones((16, 3)), 0.05)


@pytest.mark.parametrize("limiter", [n for n in LIMITERS if n != "none"])
def test_every_limiter_runs(limiter):
    prob = get_problem("dam_break")
    sol = solve(prob.left, prob.right, prob.t_end, nx=100, limiter=limiter)
    assert np.all(sol.h > 0.0)
