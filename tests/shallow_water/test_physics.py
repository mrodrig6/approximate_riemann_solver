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
    primitive_to_conservative,
)
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
