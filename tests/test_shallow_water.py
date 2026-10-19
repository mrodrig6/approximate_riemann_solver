"""The shallow-water equations: exact solver, fluxes and the scheme.

Structured like the Euler tests, and checking the same kinds of property:
defining identities where they exist, analytic solutions where they are
known, and characterisations of the documented differences from the Euler
case -- two waves instead of three, and a dry bed instead of a vacuum.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.limiters import LIMITERS
from src.shallow_water import (
    FLUX_NAMES,
    GRAVITY,
    PROBLEMS,
    celerity,
    conservative_to_primitive,
    depth_function,
    flux_code,
    get_problem,
    interface_eigensystem,
    is_dry_bed,
    list_problems,
    primitive_to_conservative,
    roe_average,
    solve,
    solve_exact,
    solve_field,
    solve_star_depth,
)
from src.shallow_water.fluxes import (
    hll_flux,
    lax_friedrichs_flux,
    rusanov_flux,
)
from src.shallow_water.physics import swe_flux

ALL_FLUXES = sorted(set(FLUX_NAMES.values()))
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


# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #

def test_primitive_conservative_roundtrip():
    rng = np.random.default_rng(0)
    w = np.column_stack([rng.uniform(0.05, 5.0, 40), rng.uniform(-3.0, 3.0, 40)])
    assert conservative_to_primitive(primitive_to_conservative(w)) == pytest.approx(
        w, rel=1e-12
    )


def test_flux_matches_its_definition():
    w = np.array([2.0, 1.5])
    u = _cons(w)
    h, vel = w
    assert _flux(u) == pytest.approx([h * vel, h * vel * vel + 0.5 * G * h * h])


def test_celerity_and_dry_cells():
    assert celerity(4.0, G) == pytest.approx(np.sqrt(4.0 * G))
    # A dry cell has no velocity to report, and its flux is hydrostatic only.
    assert conservative_to_primitive(np.array([0.0, 0.0]))[1] == 0.0
    assert _flux(np.array([0.0, 0.0])) == pytest.approx([0.0, 0.0])


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


# --------------------------------------------------------------------------- #
# Exact solver
# --------------------------------------------------------------------------- #

def test_depth_function_vanishes_at_the_root():
    for name in ("dam_break", "two_bores", "two_rarefactions", "subcritical_step"):
        prob = get_problem(name)
        h_star, _, _ = solve_star_depth(prob.left, prob.right, prob.gravity)
        assert abs(depth_function(h_star, prob.left, prob.right, prob.gravity)) < 1e-10


def test_depth_function_is_monotone_increasing():
    prob = get_problem("dam_break")
    hs = np.logspace(-3, 1, 200)
    f = np.array([depth_function(h, prob.left, prob.right, prob.gravity) for h in hs])
    assert np.all(np.diff(f) > 0)


def test_symmetric_problem_has_zero_star_velocity():
    """Mirror-symmetric data must give ``u_star = 0`` exactly."""
    sol = solve_exact([1.0, 2.5], [1.0, -2.5], 0.05)
    assert sol.u_star == pytest.approx(0.0, abs=1e-12)
    assert sol.left_wave == sol.right_wave == "bore"


def test_wave_patterns_are_identified():
    assert solve_exact([1.0, 0.0], [0.1, 0.0], 0.05).left_wave == "rarefaction"
    assert solve_exact([1.0, 0.0], [0.1, 0.0], 0.05).right_wave == "bore"
    both_bores = solve_exact([1.0, 2.5], [1.0, -2.5], 0.05)
    assert (both_bores.left_wave, both_bores.right_wave) == ("bore", "bore")
    both_rare = solve_exact([1.0, -2.5], [1.0, 2.5], 0.05)
    assert (both_rare.left_wave, both_rare.right_wave) == (
        "rarefaction",
        "rarefaction",
    )


def test_ritter_solution_for_a_dry_dam_break():
    """The dry dam break has a closed-form answer (Ritter).

    Over dry bed, the depth at the dam site is ``4 h_L / 9`` for all time,
    the velocity there is ``2 sqrt(g h_L) / 3``, and the wet/dry front
    advances at ``2 sqrt(g h_L)``.
    """
    h_l, t = 1.0, 0.05
    sol = solve_exact([h_l, 0.0], [0.0, 0.0], t, n=4001)
    assert sol.dry

    at_dam = np.argmin(np.abs(sol.x - 0.5))
    assert sol.h[at_dam] == pytest.approx(4.0 * h_l / 9.0, rel=1e-3)
    assert sol.u[at_dam] == pytest.approx(2.0 * np.sqrt(G * h_l) / 3.0, rel=1e-3)

    # Front position, to within a cell.
    front = sol.x[np.argmax(sol.h < 1e-12)]
    assert front == pytest.approx(0.5 + 2.0 * np.sqrt(G * h_l) * t, abs=2 * sol.x[0])


def test_dry_bed_generation_is_detected():
    """Fast separation opens a dry region: ``u_R - u_L >= 2(a_L + a_R)``."""
    assert not is_dry_bed([1.0, -5.0], [1.0, 5.0])     # 10 < 4 sqrt(g) = 12.53
    assert is_dry_bed([1.0, -7.0], [1.0, 7.0])         # 14 > 12.53
    sol = solve_exact([1.0, -7.0], [1.0, 7.0], 0.03)
    assert sol.dry
    assert np.isnan(sol.h_star)
    assert np.min(sol.h) == 0.0          # a genuinely dry region appears
    assert np.all(sol.h >= 0.0)


def test_star_depth_refuses_dry_data():
    with pytest.raises(ValueError, match="dry"):
        solve_star_depth([1.0, -7.0], [1.0, 7.0])


def test_exact_solver_validates_input():
    with pytest.raises(ValueError, match="positive"):
        solve_exact([1.0, 0.0], [0.1, 0.0], 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        solve_exact([-1.0, 0.0], [0.1, 0.0], 0.05)
    with pytest.raises(ValueError, match="both sides are dry"):
        solve_exact([0.0, 0.0], [0.0, 0.0], 0.05)


def test_far_field_recovers_the_initial_states():
    sol = solve_exact([1.0, 0.0], [0.1, 0.0], 0.05, n=1000)
    assert sol.h[0] == pytest.approx(1.0)
    assert sol.h[-1] == pytest.approx(0.1)


def test_depth_is_never_negative_anywhere():
    for prob in PROBLEMS.values():
        sol = solve_exact(
            prob.left, prob.right, prob.t_end, gravity=prob.gravity,
            n=997, x_diaphragm=prob.x_diaphragm,
        )
        assert np.all(sol.h >= 0.0), prob.name
        assert np.all(np.isfinite(sol.u)), prob.name


# --------------------------------------------------------------------------- #
# Fluxes
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# The scheme
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Catalogue
# --------------------------------------------------------------------------- #

def test_problem_catalogue_is_well_formed():
    assert len(PROBLEMS) >= 6
    for name, prob in PROBLEMS.items():
        assert prob.name == name
        assert prob.description and prob.reference
        assert prob.gravity > 0.0 and prob.t_end > 0.0
        assert prob.x_left < prob.x_diaphragm < prob.x_right
        assert 0.0 < prob.cfl <= 1.0
        assert prob.left.shape == prob.right.shape == (2,)
        assert prob.left[0] >= 0.0 and prob.right[0] >= 0.0
    assert [n for n, _ in list_problems()] == list(PROBLEMS)


def test_get_problem_is_case_insensitive_and_errors_helpfully():
    assert get_problem("DAM_BREAK") is get_problem("dam_break")
    with pytest.raises(KeyError, match="unknown shallow-water problem"):
        get_problem("nope")


@pytest.mark.parametrize("limiter", [n for n in LIMITERS if n != "none"])
def test_every_limiter_runs(limiter):
    prob = get_problem("dam_break")
    sol = solve(prob.left, prob.right, prob.t_end, nx=100, limiter=limiter)
    assert np.all(sol.h > 0.0)
