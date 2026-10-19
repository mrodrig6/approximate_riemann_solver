"""Shallow-water variables, closure and physical flux.

The twin of ``tests/euler/test_physics.py``: the defining identities of the
closure, and the conversions between primitive and conservative variables.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shallow_water import (
    GRAVITY,
    celerity,
    conservative_to_primitive,
    is_dry_bed,
    primitive_to_conservative,
    solve,
    solve_exact,
    solve_star_depth,
)
from src.shallow_water.physics import check_gravity
from src.shallow_water.physics import swe_flux

G = GRAVITY


def _cons(w):
    return primitive_to_conservative(np.asarray(w, dtype=float))


def _flux(u_cell):
    out = np.empty(2)
    swe_flux(u_cell, G, out)
    return out


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


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0.0, -9.81, -1.0])
def test_check_gravity_rejects_a_non_positive_gravity(bad):
    """Regression, and the worst failure mode found in this package.

    The celerity is ``sqrt(g h)``, so ``g = 0`` makes every wave speed zero
    and ``g < 0`` makes them NaN.  Either way the maximum signal speed came
    out ``0.0``, which the time loop reads as its legitimate "everything is
    dry and at rest" case -- so it set ``t = t_end`` and returned the initial
    condition unchanged, with ``steps = 0`` and **no error at all**.  A
    plausible-looking wrong answer is worse than a crash, which is why this
    is checked at every entry point rather than once.
    """
    with pytest.raises(ValueError, match="gravity must be positive"):
        check_gravity(bad)


def test_every_shallow_water_entry_point_validates_gravity():
    for bad in (0.0, -9.81):
        with pytest.raises(ValueError, match="gravity must be positive"):
            solve([1.0, 0.0], [0.1, 0.0], 0.05, nx=32, gravity=bad)
        with pytest.raises(ValueError, match="gravity must be positive"):
            solve_exact([1.0, 0.0], [0.1, 0.0], 0.05, gravity=bad)
        with pytest.raises(ValueError, match="gravity must be positive"):
            solve_star_depth([1.0, 0.0], [0.1, 0.0], gravity=bad)
        with pytest.raises(ValueError, match="gravity must be positive"):
            is_dry_bed([1.0, 0.0], [0.1, 0.0], gravity=bad)


def test_a_genuinely_dry_domain_still_short_circuits():
    """The early exit the bug was hiding behind is itself correct.

    An all-dry domain really has nothing to do, and must still return rather
    than divide by a zero signal speed.
    """
    sol = solve([0.0, 0.0], [0.0, 0.0], 0.1, nx=32)
    assert sol.steps == 0
    assert np.all(sol.h == 0.0)
