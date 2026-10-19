"""HLL-family fluxes: HLL (with Einfeldt wave speeds) and HLLC.

Both assume an approximate wave structure spanning the interface and obtain
the state(s) inside it from the integral form of the conservation laws.  HLL
assumes two waves and one intermediate state; HLLC adds the contact back, so
the fan carries two.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Sec. 10.3 (HLL), Sec. 10.4 (HLLC), Sec. 10.5 (wave-speed estimates).
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ..physics import euler_flux, pressure

__all__ = ["einfeldt_wave_speeds", "hll_flux", "hllc_flux"]


@njit
def einfeldt_wave_speeds(u_l, u_r, gamma):
    """Einfeldt's estimates of the outer wave speeds (Toro Sec. 10.5).

        S_L = min(u_L - a_L, u~ - a~),    S_R = max(u_R + a_R, u~ + a~)

    with ``u~``, ``a~`` the Roe averages.  Taking the minimum over both the
    physical left speed and the Roe speed (and the maximum on the right) is
    what guarantees the estimates bracket every wave in the exact solution,
    and that bracketing is what makes the resulting HLL flux
    positivity-preserving.  Davis' cheaper estimate, from the physical speeds
    alone, can fail to bracket a strong shock.

    Returns ``(S_L, S_R)``.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    # Roe averages; see src.euler.fluxes.roe.roe_average for the derivation.
    sq_l, sq_r = np.sqrt(rho_l), np.sqrt(rho_r)
    h_l = (u_l[2] + p_l) / rho_l
    h_r = (u_r[2] + p_r) / rho_r
    u_t = (sq_l * vel_l + sq_r * vel_r) / (sq_l + sq_r)
    h_t = (sq_l * h_l + sq_r * h_r) / (sq_l + sq_r)
    a_t = np.sqrt((gamma - 1.0) * (h_t - 0.5 * u_t * u_t))

    s_l = min(vel_l - a_l, u_t - a_t)
    s_r = max(vel_r + a_r, u_t + a_t)
    return s_l, s_r


@njit
def hll_flux(u_l, u_r, gamma, out):
    """HLL flux with Einfeldt wave speeds, written into ``out``.

    Integrating the conservation law over a control volume spanning the two
    waves gives the single intermediate state, and hence (Toro eq. 10.21)

        F = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)

    for ``S_L <= 0 <= S_R``, and the upwind physical flux otherwise.

    With no middle wave in the assumed structure, the contact discontinuity is
    averaged away, so HLL smears contacts more than even first-order Roe.  It
    is, however, unconditionally positivity-preserving with these wave speeds,
    which is why it succeeds on near-vacuum problems where Roe fails.
    """
    s_l, s_r = einfeldt_wave_speeds(u_l, u_r, gamma)

    if s_l >= 0.0:
        f = euler_flux(u_l, gamma)
        for m in range(3):
            out[m] = f[m]
        return
    if s_r <= 0.0:
        f = euler_flux(u_r, gamma)
        for m in range(3):
            out[m] = f[m]
        return

    f_l = euler_flux(u_l, gamma)
    f_r = euler_flux(u_r, gamma)
    inv = 1.0 / (s_r - s_l)
    for m in range(3):
        out[m] = (
            s_r * f_l[m] - s_l * f_r[m] + s_l * s_r * (u_r[m] - u_l[m])
        ) * inv


@njit
def hllc_flux(u_l, u_r, gamma, out):
    """HLLC flux -- HLL with the contact wave restored -- written into ``out``.

    Requiring pressure and normal velocity to be continuous across the middle
    wave fixes its speed (Toro eq. 10.37):

        S_* = [p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R)]
              / [rho_L (S_L - u_L) - rho_R (S_R - u_R)]

    and the Rankine-Hugoniot conditions across the outer waves give the two
    star states (Toro eq. 10.39), for K in {L, R}:

        U*_K = rho_K (S_K - u_K)/(S_K - S_*)
               * [1, S_*, E_K/rho_K + (S_* - u_K)(S_* + p_K/(rho_K (S_K - u_K)))]

    with ``F*_K = F_K + S_K (U*_K - U_K)``.  The branch is selected by the
    signs of ``S_L``, ``S_*`` and ``S_R``.

    Restoring the contact is the point: HLLC resolves contacts and shear about
    as sharply as Roe while keeping HLL's positivity, and needs no entropy
    fix.  It is the best general-purpose choice here.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)

    s_l, s_r = einfeldt_wave_speeds(u_l, u_r, gamma)

    if s_l >= 0.0:
        f = euler_flux(u_l, gamma)
        for m in range(3):
            out[m] = f[m]
        return
    if s_r <= 0.0:
        f = euler_flux(u_r, gamma)
        for m in range(3):
            out[m] = f[m]
        return

    # The denominator cannot vanish while S_L < 0 < S_R: both bracketed terms
    # then carry the same sign.
    num = (
        p_r - p_l
        + rho_l * vel_l * (s_l - vel_l)
        - rho_r * vel_r * (s_r - vel_r)
    )
    den = rho_l * (s_l - vel_l) - rho_r * (s_r - vel_r)
    s_star = num / den

    if s_star >= 0.0:
        f = euler_flux(u_l, gamma)
        coef = rho_l * (s_l - vel_l) / (s_l - s_star)
        e_term = u_l[2] / rho_l + (s_star - vel_l) * (
            s_star + p_l / (rho_l * (s_l - vel_l))
        )
        out[0] = f[0] + s_l * (coef - u_l[0])
        out[1] = f[1] + s_l * (coef * s_star - u_l[1])
        out[2] = f[2] + s_l * (coef * e_term - u_l[2])
    else:
        f = euler_flux(u_r, gamma)
        coef = rho_r * (s_r - vel_r) / (s_r - s_star)
        e_term = u_r[2] / rho_r + (s_star - vel_r) * (
            s_star + p_r / (rho_r * (s_r - vel_r))
        )
        out[0] = f[0] + s_r * (coef - u_r[0])
        out[1] = f[1] + s_r * (coef * s_star - u_r[1])
        out[2] = f[2] + s_r * (coef * e_term - u_r[2])
