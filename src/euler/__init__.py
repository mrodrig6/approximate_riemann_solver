"""The 1-D Euler equations: equation of state, exact solver, fluxes, schemes.

The conservative form is

    U = [rho, rho*u, E],
    F = [rho*u, rho*u^2 + p, u*(E + p)],

closed by an ideal-gas equation of state.  Three wave families, one of them
linearly degenerate (the contact) -- which is the wave the HLLC flux exists to
restore and HLL averages away.

================ =====================================================
Module           Contents
================ =====================================================
:mod:`.physics`  variables, closure, physical flux
:mod:`.exact`    exact Riemann solver
:mod:`.fluxes`   Roe, HLL, HLLC, AUSM+-up, Rusanov, LF; dispatch; TVD
:mod:`.ader`     the one-step ADER alternative to the TVD correction
:mod:`.solver`   finite-volume driver, CFL, boundary conditions
:mod:`.problems` Toro's test problems and smooth initial fields
================ =====================================================

The module names and their contents match :mod:`src.shallow_water` slot for
slot, so a reader who knows one subpackage knows where to look in the other.
:mod:`.ader` is the one module with no counterpart there: the scheme is
written for this system and has not been carried across.

Notation and equation numbers follow Toro, *Riemann Solvers and Numerical
Methods for Fluid Dynamics*, 3rd ed., Springer 2009.
"""

from __future__ import annotations

from .ader import ader2_fluxes
from .exact import ExactSolution, sample_exact, solve_exact, solve_star_pressure
from .fluxes import (
    FLUX_NAMES,
    FLUXES,
    ROBUST_FLUXES,
    flux_code,
    interface_fluxes,
    roe_average,
)
from .physics import (
    conservative_to_primitive,
    euler_flux,
    pressure,
    primitive_to_conservative,
    sound_speed,
)
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

__all__ = [
    "ExactSolution",
    "FLUXES",
    "FLUX_NAMES",
    "PROBLEMS",
    "ROBUST_FLUXES",
    "RiemannProblem",
    "RoeSolution",
    "acoustic_pulse",
    "ader2_fluxes",
    "advected_pulse",
    "cell_centres",
    "conservative_to_primitive",
    "euler_flux",
    "flux_code",
    "get_problem",
    "interface_fluxes",
    "list_problems",
    "pressure",
    "primitive_to_conservative",
    "roe_average",
    "sample_exact",
    "solve_exact",
    "solve_field",
    "solve_roe",
    "solve_star_pressure",
    "sound_speed",
]
