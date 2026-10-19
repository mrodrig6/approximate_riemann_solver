"""Shallow-water equations: exact and approximate Riemann solvers.

A parallel to the Euler solvers in the parent package, for the system

    dU/dt + dF(U)/dx = 0,
    U = [h, h*u],   F = [h*u, h*u^2 + g*h^2/2].

Everything is organised the same way, and the two share the limiter
functions, the Numba shim and the plotting style:

================ =====================================================
Module           Contents
================ =====================================================
:mod:`.physics`  variables, closure, physical flux
:mod:`.exact`    exact Riemann solver, wet and dry bed
:mod:`.fluxes`   Roe, HLL, Rusanov, Lax-Friedrichs; dispatch; TVD term
:mod:`.solver`   finite-volume driver, CFL, boundary conditions
:mod:`.problems` dam-break test problems
================ =====================================================

The module names and their contents match :mod:`src.euler` slot for slot, so
a reader who knows one subpackage knows where to look in the other.  The two
differences are both absences in :mod:`.fluxes` -- no HLLC and no AUSM+-up --
and both are explained there.

What differs from the Euler case
--------------------------------
* **Two waves, both genuinely non-linear.**  No contact discontinuity, and
  therefore no HLLC: with no middle wave to restore it would reduce exactly
  to HLL, so ``flux="hllc"`` is accepted as an alias for it.  It also means
  HLL is not at the accuracy disadvantage it has for a gas -- its two-wave
  model *is* the full wave structure -- which is why it is the default here.
* **Dry bed instead of vacuum.**  ``h = 0`` is a routine engineering case,
  not a pathology, so the exact solver solves it rather than refusing, and
  the approximate solvers handle a wetting front.

Quick start
-----------
>>> from src.shallow_water import get_problem, solve, solve_exact
>>> prob = get_problem("dam_break")
>>> exact = solve_exact(prob.left, prob.right, prob.t_end)
>>> round(exact.h_star, 6)
0.396175
>>> numerical = solve(prob.left, prob.right, prob.t_end, nx=200)
"""

from __future__ import annotations

from .exact import (
    ShallowWaterSolution,
    depth_function,
    is_dry_bed,
    solve_exact,
    solve_star_depth,
)
from .fluxes import (
    FLUX_NAMES,
    FLUXES,
    ROBUST_FLUXES,
    flux_code,
    interface_eigensystem,
    interface_fluxes,
    roe_average,
)
from .physics import (
    GRAVITY,
    celerity,
    conservative_to_primitive,
    primitive_to_conservative,
    swe_flux,
)
from .problems import PROBLEMS, ShallowWaterProblem, get_problem, list_problems
from .solver import ShallowWaterResult, solve, solve_field

__all__ = [
    "FLUXES",
    "FLUX_NAMES",
    "GRAVITY",
    "PROBLEMS",
    "ROBUST_FLUXES",
    "ShallowWaterProblem",
    "ShallowWaterResult",
    "ShallowWaterSolution",
    "celerity",
    "conservative_to_primitive",
    "depth_function",
    "flux_code",
    "get_problem",
    "interface_eigensystem",
    "interface_fluxes",
    "is_dry_bed",
    "list_problems",
    "primitive_to_conservative",
    "roe_average",
    "solve",
    "solve_exact",
    "solve_field",
    "solve_star_depth",
    "swe_flux",
]
