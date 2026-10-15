"""Exact and approximate Riemann solvers for the 1-D Euler equations.

A Python port of the MATLAB code of the Rodriguez Flow Research group at Brown
University, accelerated with Numba.  The package provides

* :func:`riemann.exact.solve_exact` -- the exact Riemann solver (Newton
  iteration for the star pressure plus self-similar sampling of the wave fan);
* :func:`riemann.solver.solve_roe` -- Roe's approximate Riemann solver with a
  Harten-Hyman entropy fix and a second-order TVD flux-limiter extension;
* :mod:`riemann.problems` -- Toro's standard test problems;
* :mod:`riemann.plotting` -- four-panel comparison figures.

The formulation, the equation numbers and the test cases all follow Toro,
*Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
Springer 2009.  See ``docs/`` for the full write-up.

Quick start
-----------
>>> from riemann import solve_exact, solve_roe, get_problem
>>> prob = get_problem("sod")
>>> exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end)
>>> roe = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=200)
>>> round(exact.p_star, 5)
0.30313
"""

from __future__ import annotations

from ._jit import HAVE_NUMBA, numba_version
from .exact import ExactSolution, sample_exact, solve_exact, solve_star_pressure
from .limiters import LIMITERS, limiter_phi
from .plotting import plot_comparison, save_figure
from .problems import PROBLEMS, RiemannProblem, get_problem, list_problems
from .roe import roe_average, roe_fluxes
from .solver import RoeSolution, solve_roe
from .thermo import conservative_to_primitive, primitive_to_conservative

__version__ = "1.0.0"

__all__ = [
    "ExactSolution",
    "HAVE_NUMBA",
    "LIMITERS",
    "PROBLEMS",
    "RiemannProblem",
    "RoeSolution",
    "__version__",
    "conservative_to_primitive",
    "get_problem",
    "limiter_phi",
    "list_problems",
    "numba_version",
    "plot_comparison",
    "primitive_to_conservative",
    "roe_average",
    "roe_fluxes",
    "sample_exact",
    "save_figure",
    "solve_exact",
    "solve_roe",
    "solve_star_pressure",
]
