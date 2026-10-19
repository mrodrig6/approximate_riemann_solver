"""The Numba shim, and the fast-math relaxations the kernels are built with."""

from __future__ import annotations

import pytest

from src.core import jit


def test_njit_accepts_both_decorator_spellings():
    @jit.njit
    def bare(x):
        return x + 1.0

    @jit.njit(cache=False)
    def called(x):
        return x + 2.0

    assert bare(1.0) == 2.0
    assert called(1.0) == 3.0


@pytest.mark.skipif(not jit.HAVE_NUMBA, reason="no Numba, so no fast-math flags")
def test_fastmath_keeps_nan_and_inf_semantics():
    """``nnan`` and ``ninf`` must stay out of the fast-math set.

    They licence the compiler to assume no operand is ever a NaN or an
    infinity, and under that assumption a comparison against a NaN may be
    folded away.  The solvers depend on those comparisons: every positivity
    guard is written so that a NaN *fails* it, which is what turns a step
    that has already gone bad into a reported error rather than a silent
    one.  Granting ``nnan`` would tell the compiler that the case the guards
    exist to catch cannot happen.

    Regression: the shim used to pass ``fastmath=True``, which is the whole
    set.  Measured cost of dropping these two on Sod at 1600 cells: none
    outside run-to-run noise.
    """
    assert "nnan" not in jit.FASTMATH
    assert "ninf" not in jit.FASTMATH
    # The relaxations the speed actually comes from are still granted.
    assert "reassoc" in jit.FASTMATH
    assert "contract" in jit.FASTMATH
