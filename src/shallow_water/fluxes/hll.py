"""The HLL flux, and the wave-speed estimates every flux here is built on.

HLL assumes two waves spanning the interface and one intermediate state,
obtained from the integral form of the conservation laws.

**There is no HLLC.**  HLLC exists to put back the contact wave that HLL
averages away, and the shallow-water system has no contact: both of its
fields are genuinely non-linear.  With no middle wave to restore, HLLC
reduces exactly to HLL, so :mod:`~src.shallow_water.fluxes.registry` accepts
the name as an alias rather than implementing a second function that would
compute the same numbers.  This is the one place where the symmetry with
:mod:`src.euler.fluxes.hll` stops, and it stops for a reason in the
equations rather than one in the code.

The same absence cuts the other way for accuracy: HLL is not at the
disadvantage here that it is for a gas, because its two-wave model *is* the
full wave structure.  That, with its positivity, makes it the default.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Sec. 7.4 and Chapter 10.
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ..physics import DRY_TOLERANCE, celerity, swe_flux

__all__ = ["einfeldt_wave_speeds", "hll_flux"]


@njit
def einfeldt_wave_speeds(u_l, u_r, gravity):
    """Einfeldt-style wave-speed estimates, with dry-bed handling.

    Wet on both sides, the estimates bracket every wave:

        S_L = min(u_L - a_L, u~ - a~),   S_R = max(u_R + a_R, u~ + a~).

    Where one side is dry the Roe average is undefined, and the correct
    outer speed is the wet/dry front of the rarefaction running onto the dry
    bed, ``u +- 2a`` -- the same front the exact solver uses.  Getting this
    case right is what lets the approximate solvers run a dam break onto dry
    ground at all.
    """
    h_l = u_l[0]
    h_r = u_r[0]
    left_wet = h_l > DRY_TOLERANCE
    right_wet = h_r > DRY_TOLERANCE

    if left_wet and right_wet:
        vel_l = u_l[1] / h_l
        vel_r = u_r[1] / h_r
        a_l = celerity(h_l, gravity)
        a_r = celerity(h_r, gravity)
        sq_l = np.sqrt(h_l)
        sq_r = np.sqrt(h_r)
        u_t = (sq_l * vel_l + sq_r * vel_r) / (sq_l + sq_r)
        a_t = np.sqrt(0.5 * gravity * (h_l + h_r))
        return min(vel_l - a_l, u_t - a_t), max(vel_r + a_r, u_t + a_t)

    if left_wet:  # dry to the right
        vel_l = u_l[1] / h_l
        a_l = celerity(h_l, gravity)
        return vel_l - a_l, vel_l + 2.0 * a_l

    if right_wet:  # dry to the left
        vel_r = u_r[1] / h_r
        a_r = celerity(h_r, gravity)
        return vel_r - 2.0 * a_r, vel_r + a_r

    return 0.0, 0.0  # dry both sides


@njit
def hll_flux(u_l, u_r, gravity, out):
    """HLL flux: two waves, one intermediate state.

    With no contact in the system there is nothing for HLL to average away,
    so unlike the Euler case it is not at an accuracy disadvantage against a
    three-wave solver -- its two-wave model *is* the full wave structure.
    That, plus its positivity, makes it the natural default here.
    """
    s_l, s_r = einfeldt_wave_speeds(u_l, u_r, gravity)
    f_l = np.empty(2)
    f_r = np.empty(2)
    swe_flux(u_l, gravity, f_l)
    swe_flux(u_r, gravity, f_r)

    if s_l >= 0.0:
        for m in range(2):
            out[m] = f_l[m]
        return
    if s_r <= 0.0:
        for m in range(2):
            out[m] = f_r[m]
        return
    inv = 1.0 / (s_r - s_l)
    for m in range(2):
        out[m] = (
            s_r * f_l[m] - s_l * f_r[m] + s_l * s_r * (u_r[m] - u_l[m])
        ) * inv
