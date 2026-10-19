"""The shallow-water problem catalogue.

The twin of ``tests/euler/test_problems.py``.
"""

from __future__ import annotations

import pytest

from src.shallow_water import PROBLEMS, get_problem, list_problems


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
