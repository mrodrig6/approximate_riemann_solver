"""Numerical flux functions for the 1-D Euler equations.

One module per family, all interchangeable through
:func:`~riemann.fluxes.interface.interface_fluxes`:

=============== ==================================================== ==========
Module          Fluxes                                               Waves
=============== ==================================================== ==========
:mod:`.roe`     Roe, with the Harten-Hyman entropy fix               3
:mod:`.hll`     HLL (Einfeldt wave speeds), HLLC                     2, 3
:mod:`.ausm`    AUSM+-up                                             3 (split)
:mod:`.central` Rusanov, Lax-Friedrichs                              0 (scalar)
=============== ==================================================== ==========

They fall into three families, and knowing which is which predicts how each
behaves:

* **Central / Rusanov-type** -- one scalar dissipation coefficient applied to
  the whole state vector.  Unconditionally robust and trivially
  positivity-preserving, but they smear contacts badly, since the same large
  coefficient damps every wave family.
* **HLL-type** -- an approximate wave structure with two (HLL) or three
  (HLLC) waves and a consistent integral average between them.
  Positivity-preserving for the right wave-speed estimates; HLLC restores the
  contact that HLL averages away.
* **Flux-difference (Roe) and flux-vector (AUSM+-up)** -- both resolve all
  three fields and carry a contact exactly.  Roe is additionally exact for an
  isolated shock; AUSM+-up is the cheapest here, forming neither an
  eigensystem nor a wave-speed estimate.

:mod:`.registry` holds the names and integer codes, and
:mod:`.interface` the dispatch and the second-order TVD correction.
"""

from __future__ import annotations

from .ausm import ausm_up_flux, m4_minus, m4_plus, p5_minus, p5_plus
from .central import lax_friedrichs_flux, rusanov_flux
from .hll import einfeldt_wave_speeds, hll_flux, hllc_flux
from .interface import interface_fluxes
from .registry import FLUX_NAMES, FLUXES, ROBUST_FLUXES, flux_code
from .roe import interface_eigensystem, roe_average, roe_flux

__all__ = [
    "FLUXES",
    "FLUX_NAMES",
    "ROBUST_FLUXES",
    "ausm_up_flux",
    "einfeldt_wave_speeds",
    "flux_code",
    "hll_flux",
    "hllc_flux",
    "interface_eigensystem",
    "interface_fluxes",
    "lax_friedrichs_flux",
    "m4_minus",
    "m4_plus",
    "p5_minus",
    "p5_plus",
    "roe_average",
    "roe_flux",
    "rusanov_flux",
]
