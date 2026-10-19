"""The built-in test-problem catalogue."""

from __future__ import annotations

import pytest

from src.euler.problems import PROBLEMS, get_problem, list_problems


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
    from src import solve

    kwargs = get_problem("sod").kwargs()
    roe = solve(nx=50, **kwargs)
    assert roe.steps > 0
