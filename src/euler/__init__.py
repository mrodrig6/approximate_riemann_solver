"""The 1-D Euler equations: equation of state, exact solver, fluxes, schemes.

The conservative form is

    U = [rho, rho*u, E],
    F = [rho*u, rho*u^2 + p, u*(E + p)],

closed by an ideal-gas equation of state.  Three wave families, one of them
linearly degenerate (the contact) -- which is the wave the HLLC flux exists to
restore and HLL averages away.

The modules, in the order the solver uses them:

* :mod:`src.euler.thermo` -- equation of state, variable conversions, the
  physical flux.  Depends on nothing else here;
* :mod:`src.euler.exact` -- the exact Riemann solver, used as the reference
  solution everywhere;
* :mod:`src.euler.fluxes` -- the numerical fluxes (Roe, HLL, HLLC, AUSM+-up,
  Rusanov, Lax-Friedrichs) and the flux-limited TVD correction;
* :mod:`src.euler.ader` -- the one-step ADER alternative to that correction;
* :mod:`src.euler.solver` -- the time loop, which selects between the two;
* :mod:`src.euler.problems` -- Toro's test problems and smooth initial fields.

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
from .thermo import (
    conservative_to_primitive,
    euler_flux,
    pressure,
    primitive_to_conservative,
    sound_speed,
)

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
