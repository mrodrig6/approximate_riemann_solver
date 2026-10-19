"""Numerical flux functions for the shallow-water equations.

One module per family, all interchangeable through
:func:`~src.shallow_water.fluxes.interface.interface_fluxes`, laid out to
match :mod:`src.euler.fluxes` module for module:

=============== ==================================================== ==========
Module          Fluxes                                               Waves
=============== ==================================================== ==========
:mod:`.roe`     Roe, with a Harten entropy fix                        2
:mod:`.hll`     HLL (Einfeldt wave speeds)                            2
:mod:`.central` Rusanov, Lax-Friedrichs                               0 (scalar)
=============== ==================================================== ==========

The same families as for the Euler system, less two, and both absences are in
the equations rather than in the code:

* **no HLLC**, because there is no contact wave for it to restore -- both
  fields here are genuinely non-linear, so HLLC reduces exactly to HLL and
  :mod:`.registry` accepts the name as an alias;
* **no AUSM+-up**, because the AUSM family splits the flux by Mach number
  into a convective and a pressure part.  A shallow-water flux has no
  pressure term in that sense; the ``g h^2 / 2`` term *is* the depth, not a
  separate thermodynamic variable.  Shallow-water members of the family exist
  in the literature but are a research question, not a transcription.

Against those, HLL loses less here than it does for a gas: with no contact in
the system its two-wave model *is* the full wave structure, which is why it is
the default rather than a fallback.

:mod:`.registry` holds the names and integer codes, and
:mod:`.interface` the dispatch and the second-order TVD correction.
"""

from __future__ import annotations

from .central import lax_friedrichs_flux, rusanov_flux
from .hll import einfeldt_wave_speeds, hll_flux
from .interface import interface_fluxes
from .registry import FLUX_NAMES, FLUXES, ROBUST_FLUXES, flux_code
from .roe import (
    interface_eigensystem,
    interface_eigensystem_wet,
    roe_average,
    roe_flux,
)

__all__ = [
    "FLUXES",
    "FLUX_NAMES",
    "ROBUST_FLUXES",
    "einfeldt_wave_speeds",
    "flux_code",
    "hll_flux",
    "interface_eigensystem",
    "interface_eigensystem_wet",
    "interface_fluxes",
    "lax_friedrichs_flux",
    "roe_average",
    "roe_flux",
    "rusanov_flux",
]
