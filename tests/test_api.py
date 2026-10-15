"""Tests for the supporting API: thermodynamics, the problem catalogue, the CLI."""

from __future__ import annotations

import numpy as np
import pytest

from riemann import cli, conservative_to_primitive, primitive_to_conservative
from riemann.problems import PROBLEMS, get_problem, list_problems
from riemann.thermo import (
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


def test_problem_catalogue_is_well_formed():
    assert len(PROBLEMS) >= 7
    for name, prob in PROBLEMS.items():
        assert prob.name == name
        assert prob.description and prob.reference
        assert prob.gamma > 1.0
        assert prob.t_end > 0.0
        assert prob.x_left < prob.x_diaphragm < prob.x_right
        assert prob.left.shape == prob.right.shape == (3,)
        assert prob.left[0] > 0 and prob.left[2] > 0
        assert prob.right[0] > 0 and prob.right[2] > 0


def test_list_problems_matches_catalogue():
    assert [name for name, _ in list_problems()] == list(PROBLEMS)


def test_get_problem_is_case_insensitive_and_errors_helpfully():
    assert get_problem("SOD") is get_problem("sod")
    with pytest.raises(KeyError, match="unknown problem"):
        get_problem("nope")


def test_problem_kwargs_feed_the_solvers():
    from riemann import solve_roe

    kwargs = get_problem("sod").kwargs()
    roe = solve_roe(nx=50, **kwargs)
    assert roe.steps > 0


def test_cli_runs_and_reports(capsys):
    assert cli.main(["sod", "--nx", "50"]) == 0
    out = capsys.readouterr().out
    assert "p* = 0.30313" in out
    assert "L1" in out


def test_cli_reports_documented_failure(capsys):
    """toro2 must exit non-zero with the explanatory message, not a traceback."""
    assert cli.main(["toro2", "--nx", "50"]) == 1
    captured = capsys.readouterr()
    assert "Roe run FAILED" in captured.err
    assert "positivity" in captured.err or "non-physical" in captured.err


def test_cli_rejects_unknown_problem(capsys):
    """An unknown name gets a listing of the valid ones, not a traceback."""
    assert cli.main(["definitely-not-a-problem"]) == 2
    err = capsys.readouterr().err
    assert "unknown problem" in err
    assert "sod" in err


def test_cli_first_order_and_limiter_flags(capsys):
    assert cli.main(["sod", "--nx", "50", "--first-order", "--limiter", "minmod"]) == 0
    assert "1st order" in capsys.readouterr().out
