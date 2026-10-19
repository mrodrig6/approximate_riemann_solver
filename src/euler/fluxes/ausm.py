"""AUSM+-up: advection upstream splitting, all-speed variant.

AUSM-family schemes do not decompose the jump into waves.  They split the
flux into a *convective* part carried by an interface velocity and a separate
*pressure* part,

    F = u_1/2 Psi_(upwind) + [0, p_1/2, 0],   Psi = [rho, rho u, E + p],

with the convective vector taken from whichever side the interface velocity
points away from.  Because pressure is handled entirely by ``p_1/2``, the
convective part carries no pressure dissipation, and the scheme reproduces a
contact discontinuity -- stationary or moving -- exactly.

References: Liou, *A sequel to AUSM, Part II: AUSM+-up for all speeds*,
J. Comput. Phys. 214:137 (2006) for the scheme; Liou, J. Comput. Phys.
129:364 (1996) for the split polynomials.  The coefficients ``K_p`` and
``K_u`` and the arithmetic-mean interface sound speed follow the benchmarked
implementation in EMS, ``src/riemann_solvers/ausm6.cpp``.
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ..physics import pressure

__all__ = [
    "ausm_up_flux",
    "m4_plus",
    "m4_minus",
    "p5_plus",
    "p5_minus",
]

# beta and alpha fix the split polynomials.  They are not tunable: they are
# the unique values that make polynomials of the stated degrees smooth at
# |M| = 1.
BETA = 0.125     # in the split Mach polynomials M+_(4), M-_(4)
ALPHA = 0.1875   # = 3/16, in the pressure splittings P+_(5), P-_(5)

# K_p and K_u scale the two "-up" diffusion terms.  These are genuine
# parameters of the scheme; the values are EMS's.
KP = 0.4         # pressure diffusion in the interface velocity
KU = 0.2         # velocity diffusion in the interface pressure


@njit
def _m1_plus(m):
    """``M+_(1) = 0.5 (M + |M|)``."""
    return 0.5 * (m + abs(m))


@njit
def _m1_minus(m):
    """``M-_(1) = 0.5 (M - |M|)``."""
    return 0.5 * (m - abs(m))


@njit
def _m2_plus(m):
    """``M+_(2) = +0.25 (M + 1)^2``."""
    return 0.25 * (m + 1.0) * (m + 1.0)


@njit
def _m2_minus(m):
    """``M-_(2) = -0.25 (M - 1)^2``."""
    return -0.25 * (m - 1.0) * (m - 1.0)


@njit
def m4_plus(m, beta):
    """Fourth-degree split Mach number ``M+_(4)``.

    ``M+_(1)`` for ``|M| >= 1``, else ``M+_(2) (1 - 16 beta M-_(2))``.  With
    :func:`m4_minus` it satisfies ``M+_(4)(M) + M-_(4)(M) = M`` for every
    ``M`` and every ``beta`` -- the cross terms cancel.  Expanded at
    ``beta = 1/8`` and ``|M| < 1`` this is ``M^4/8 + M/2 + 3/8``.
    """
    if abs(m) >= 1.0:
        return _m1_plus(m)
    return _m2_plus(m) * (1.0 - 16.0 * beta * _m2_minus(m))


@njit
def m4_minus(m, beta):
    """Fourth-degree split Mach number ``M-_(4)``."""
    if abs(m) >= 1.0:
        return _m1_minus(m)
    return _m2_minus(m) * (1.0 + 16.0 * beta * _m2_plus(m))


@njit
def p5_plus(m, alpha):
    """Fifth-degree pressure splitting ``P+_(5)``.

    ``M+_(1)/M`` for ``|M| >= 1`` (1 upstream, 0 downstream), else
    ``M+_(2)[(2 - M) - 16 alpha M M-_(2)]``.  With :func:`p5_minus` it
    satisfies ``P+_(5)(M) + P-_(5)(M) = 1`` for every ``M`` and every
    ``alpha``, which is what makes the pressure splitting consistent.
    Expanded at ``alpha = 3/16`` and ``|M| < 1`` this is
    ``3 M^5/16 - 5 M^3/8 + 15 M/16 + 1/2``.
    """
    if abs(m) >= 1.0:
        return _m1_plus(m) / m
    return _m2_plus(m) * ((2.0 - m) - 16.0 * alpha * m * _m2_minus(m))


@njit
def p5_minus(m, alpha):
    """Fifth-degree pressure splitting ``P-_(5)``."""
    if abs(m) >= 1.0:
        return _m1_minus(m) / m
    return _m2_minus(m) * ((-2.0 - m) + 16.0 * alpha * m * _m2_plus(m))


@njit
def ausm_up_flux(u_l, u_r, gamma, out):
    """AUSM+-up flux, written into ``out``.

    The interface velocity and pressure are built from polynomial splittings
    of each side's Mach number ``M_K = u_K / a_1/2``:

        u_1/2 = a_1/2 [M+_(4)(M_L) + M-_(4)(M_R)]
                + K_p max(1 - Mbar^2, 0) (p_L - p_R) / (rho_1/2 a_1/2),

        p_1/2 = P+_(5)(M_L) p_L + P-_(5)(M_R) p_R
                + K_u P+_(5) P-_(5) rho_1/2 a_1/2 (u_L - u_R),

    with ``a_1/2`` and ``rho_1/2`` the arithmetic means of the one-sided
    values and ``Mbar^2 = (M_L^2 + M_R^2)/2``.

    The two extra terms are the "-up" of the name.  ``K_p`` is a
    pressure-diffusion term in the velocity and ``K_u`` a velocity-diffusion
    term in the pressure; ``max(1 - Mbar^2, 0)`` switches the first off once
    the interface is supersonic, where upwinding already supplies enough
    dissipation.

    The ``K_p`` term is not optional.  Without it the split Mach numbers give
    ``M_1/2 = 0`` exactly whenever ``u_L = u_R = 0``, however large the
    pressure jump, so both the mass flux and the energy flux ``u_1/2 (E + p)``
    vanish.  The momentum equation still feels the pressure difference and
    accelerates the gas, and with the total energy fixed the growing kinetic
    energy is subtracted from the internal energy -- which goes negative in
    the first step on the light side of a shock tube.  ``K_p`` supplies the
    missing pressure-velocity coupling.

    Both ``K_p`` and ``K_u`` are diffusion coefficients, and the consistency
    identities on the split polynomials say nothing about them: too large a
    ``K_u`` is anti-dissipative and destabilises a moving contact.  The
    invariant that pins them is that a contact advected at uniform speed must
    not generate pressure, which the test suite checks.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    a_half = 0.5 * (a_l + a_r)
    rho_half = 0.5 * (rho_l + rho_r)

    mach_l = vel_l / a_half
    mach_r = vel_r / a_half
    mbar_sq = 0.5 * (mach_l * mach_l + mach_r * mach_r)

    u_half = a_half * (
        m4_plus(mach_l, BETA) + m4_minus(mach_r, BETA)
    ) + KP * max(1.0 - mbar_sq, 0.0) * (p_l - p_r) / (rho_half * a_half)

    p5p = p5_plus(mach_l, ALPHA)
    p5m = p5_minus(mach_r, ALPHA)
    p_half = (
        p5p * p_l
        + p5m * p_r
        + KU * p5p * p5m * rho_half * a_half * (vel_l - vel_r)
    )

    # Upwind Psi = [rho, rho u, E + p] on the sign of the interface velocity.
    if u_half > 0.0:
        out[0] = u_half * rho_l
        out[1] = u_half * u_l[1] + p_half
        out[2] = u_half * (u_l[2] + p_l)
    else:
        out[0] = u_half * rho_r
        out[1] = u_half * u_r[1] + p_half
        out[2] = u_half * (u_r[2] + p_r)
