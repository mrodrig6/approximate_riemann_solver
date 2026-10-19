"""Flux limiters: Sweby's TVD region, second-order consistency, naming."""

from __future__ import annotations

import numpy as np
import pytest

from src.core.limiters import LIMITERS, limiter_code, limiter_phi


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
