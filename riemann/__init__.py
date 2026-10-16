"""Exact and approximate Riemann solvers for the 1-D Euler equations.

The package provides

* :func:`riemann.exact.solve_exact` -- the exact Riemann solver: Newton
  iteration for the star-region pressure, then self-similar sampling of the
  wave fan;
* :func:`riemann.solver.solve_roe` -- an explicit finite-volume solver with a
  choice of numerical flux and of second-order scheme (flux-limited TVD, or
  the one-step ADER scheme of :mod:`riemann.ader`);
* :func:`riemann.solver.solve_field` -- the same solver on an arbitrary
  initial field, for order-of-accuracy studies;
* :mod:`riemann.fluxes` -- the flux functions: Roe, HLL, HLLC, AUSM+-up,
  Rusanov and Lax-Friedrichs;
* :mod:`riemann.limiters` -- superbee, minmod, van Leer, van Albada;
* :mod:`riemann.problems` -- standard test problems;
* :mod:`riemann.plotting` -- comparison figures;
* :mod:`riemann.shallow_water` -- the same solvers for the shallow-water
  equations.

Formulation, notation and equation numbers follow Toro, *Riemann Solvers and
Numerical Methods for Fluid Dynamics*, 3rd ed., Springer 2009.  See ``docs/``
for the full write-up.

Quick start
-----------
>>> from riemann import get_problem, solve_exact, solve_roe
>>> prob = get_problem("sod")
>>> exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end)
>>> round(exact.p_star, 5)
0.30313
>>> roe = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=200)
>>> hllc = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, flux="hllc")
"""

from __future__ import annotations

from ._jit import HAVE_NUMBA, numba_version
from .exact import ExactSolution, sample_exact, solve_exact, solve_star_pressure
from .fluxes import (
    FLUXES,
    ROBUST_FLUXES,
    flux_code,
    interface_fluxes,
    roe_average,
)
from .limiters import LIMITERS, limiter_phi
from .plotting import plot_comparison, plot_shallow_water, save_figure
from .problems import (
    PROBLEMS,
    RiemannProblem,
    acoustic_pulse,
    advected_pulse,
    cell_centres,
    get_problem,
    list_problems,
)
from .solver import RoeSolution, solve_field, solve_roe
from .thermo import conservative_to_primitive, primitive_to_conservative

__version__ = "1.1.0"

__all__ = [
    "ExactSolution",
    "FLUXES",
    "HAVE_NUMBA",
    "LIMITERS",
    "PROBLEMS",
    "ROBUST_FLUXES",
    "RiemannProblem",
    "RoeSolution",
    "__version__",
    "acoustic_pulse",
    "advected_pulse",
    "cell_centres",
    "conservative_to_primitive",
    "flux_code",
    "get_problem",
    "interface_fluxes",
    "limiter_phi",
    "list_problems",
    "numba_version",
    "plot_comparison",
    "plot_shallow_water",
    "primitive_to_conservative",
    "roe_average",
    "sample_exact",
    "save_figure",
    "solve_exact",
    "solve_field",
    "solve_roe",
    "solve_star_pressure",
]
