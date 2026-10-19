"""Central fluxes with scalar dissipation: Lax-Friedrichs and Rusanov.

Both have the form

    F = 0.5 (F_L + F_R) - 0.5 S (U_R - U_L),

a central average damped by a single coefficient ``S`` applied to the whole
state vector.  They differ only in the choice of ``S``.

One scalar damps both characteristic fields equally, which makes these the
most dissipative fluxes here, as in the Euler case.  What they lose relative
to that case is less, though: with no contact in the system there is no
linearly degenerate field being smeared far worse than the rest.

Rusanov takes its coefficient from :func:`~src.shallow_water.fluxes.hll
.einfeldt_wave_speeds` rather than from a local ``|u| + a``, which is where
the dry-bed handling lives -- the Euler twin of this module needs no such
import.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Sec. 7.3.
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ..physics import swe_flux
from .hll import einfeldt_wave_speeds

__all__ = ["lax_friedrichs_flux", "rusanov_flux"]


@njit
def lax_friedrichs_flux(u_l, u_r, gravity, dx, dt, out):
    """Classical Lax-Friedrichs: central average damped by the mesh speed."""
    f_l = np.empty(2)
    f_r = np.empty(2)
    swe_flux(u_l, gravity, f_l)
    swe_flux(u_r, gravity, f_r)
    s = dx / dt
    for m in range(2):
        out[m] = 0.5 * (f_l[m] + f_r[m]) - 0.5 * s * (u_r[m] - u_l[m])


@njit
def rusanov_flux(u_l, u_r, gravity, out):
    """Rusanov (local Lax-Friedrichs): damped by the local maximum speed."""
    s_l, s_r = einfeldt_wave_speeds(u_l, u_r, gravity)
    s_plus = max(abs(s_l), abs(s_r))
    f_l = np.empty(2)
    f_r = np.empty(2)
    swe_flux(u_l, gravity, f_l)
    swe_flux(u_r, gravity, f_r)
    for m in range(2):
        out[m] = 0.5 * (f_l[m] + f_r[m]) - 0.5 * s_plus * (u_r[m] - u_l[m])
