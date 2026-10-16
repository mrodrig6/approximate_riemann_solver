"""Central fluxes with scalar dissipation: Lax-Friedrichs and Rusanov.

Both have the form

    F = 0.5 (F_L + F_R) - 0.5 S (U_R - U_L),

a central average damped by a single coefficient ``S`` applied to the whole
state vector.  They differ only in the choice of ``S``.

Because one scalar damps all three characteristic fields equally, these are
the most dissipative fluxes available here -- a contact is smeared about as
badly as the fastest acoustic wave demands.  In exchange they are trivially
robust and need no wave-speed estimate, eigensystem or iteration.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Sec. 5.3.4 (Lax-Friedrichs) and Sec. 10.5 (Rusanov).
"""

from __future__ import annotations

import numpy as np

from .._jit import njit
from ..thermo import euler_flux, pressure

__all__ = ["lax_friedrichs_flux", "rusanov_flux"]


@njit
def lax_friedrichs_flux(u_l, u_r, gamma, dx, dt, out):
    """Classical Lax-Friedrichs flux, written into ``out``.

    The dissipation coefficient is the *mesh* speed ``dx / dt``, which by the
    CFL condition is at least the fastest signal speed anywhere on the grid.
    Applying that global maximum at every interface makes this the most
    dissipative scheme here by a wide margin; it serves as the baseline the
    others are measured against.

    One consequence is worth knowing: ``dx / dt`` acts on a ``2 dx`` stencil,
    which leaves the even and odd sub-grids uncoupled.  The solution develops
    identical adjacent cell pairs, and that pattern drives every flux-limiter
    ratio to zero, so the second-order correction has no effect on this flux.
    """
    f_l = euler_flux(u_l, gamma)
    f_r = euler_flux(u_r, gamma)
    s = dx / dt
    for m in range(3):
        out[m] = 0.5 * (f_l[m] + f_r[m]) - 0.5 * s * (u_r[m] - u_l[m])


@njit
def rusanov_flux(u_l, u_r, gamma, out):
    """Rusanov (local Lax-Friedrichs) flux, written into ``out``.

    Same form as Lax-Friedrichs but with the *local* maximum signal speed,

        S = max(|u_L| + a_L, |u_R| + a_R),

    which is a large improvement at no extra cost.  Still one scalar for all
    three fields, so contacts remain smeared.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    s_plus = max(abs(vel_l) + a_l, abs(vel_r) + a_r)

    f_l = euler_flux(u_l, gamma)
    f_r = euler_flux(u_r, gamma)
    for m in range(3):
        out[m] = 0.5 * (f_l[m] + f_r[m]) - 0.5 * s_plus * (u_r[m] - u_l[m])
