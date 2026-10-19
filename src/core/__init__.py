"""Machinery shared by both systems of equations.

Nothing in here knows about the Euler equations or about shallow water.  The
split is deliberate and is the reason this subpackage exists: a TVD limiter is
a function of the smoothness ratio alone, the Numba shim is a decorator, and
the plotting helpers take arrays and labels.  All three are reused verbatim by
:mod:`src.euler` and :mod:`src.shallow_water`, so they live beside neither.

* :mod:`src.core.jit` -- the optional-Numba shim, so ``njit`` is importable
  whether or not Numba is installed;
* :mod:`src.core.limiters` -- superbee, minmod, van Leer and van Albada;
* :mod:`src.core.plotting` -- comparison figures.

:mod:`src.core.plotting` imports matplotlib lazily, inside the functions that
draw, so importing this subpackage costs nothing and does not require
matplotlib to be installed.
"""

from __future__ import annotations

from .jit import HAVE_NUMBA, njit, numba_version
from .limiters import LIMITER_NAMES, LIMITERS, limiter_code, limiter_phi
from .plotting import (
    plot_comparison,
    plot_series,
    plot_shallow_water,
    save_figure,
    use_paper_style,
)

__all__ = [
    "HAVE_NUMBA",
    "LIMITERS",
    "LIMITER_NAMES",
    "limiter_code",
    "limiter_phi",
    "njit",
    "numba_version",
    "plot_comparison",
    "plot_series",
    "plot_shallow_water",
    "save_figure",
    "use_paper_style",
]
