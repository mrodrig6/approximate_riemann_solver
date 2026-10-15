"""Numerical flux functions, and the flux-limited second-order wrapper.

This module is the pluggable part of the solver: it holds every first-order
interface flux, dispatches between them, and adds the TVD correction that
makes the scheme second order.  :mod:`riemann.solver` drives it and never
needs to know which flux is in use.

The fluxes
----------

=================  ====================================================
Name               Origin
=================  ====================================================
``lax_friedrichs`` Classical Lax-Friedrichs (Toro Sec. 5.3.4)
``rusanov``        Rusanov / local Lax-Friedrichs (Toro Sec. 10.5)
``hll``            Harten-Lax-van Leer with Einfeldt wave speeds
                   (Toro Secs. 10.3, 10.5); i.e. the HLLE variant
``hllc``           HLL with a restored Contact wave (Toro Sec. 10.4)
``roe``            Roe with the Harten-Hyman entropy fix (Toro Ch. 11)
``ausmup``         AUSM+-up, Liou, J. Comput. Phys. 214:137 (2006)
=================  ====================================================

They fall into three families, and it is worth knowing which is which
because it predicts how each behaves:

* **Central / Rusanov-type** (``lax_friedrichs``, ``rusanov``).  One scalar
  dissipation coefficient applied to the whole state vector.  Unconditionally
  robust and trivially positivity-preserving, but they smear the contact badly
  because the same large coefficient is applied to every wave family.
* **HLL-type** (``hll``, ``hllc``).  An approximate wave structure with two
  (HLL) or three (HLLC) waves and a consistent integral average between them.
  Positivity-preserving for the right wave-speed estimates, and HLLC restores
  the contact that HLL averages away.
* **Flux-vector / AUSM-type** (``ausmup``) and **flux-difference** (``roe``).
  Both resolve all three fields exactly at a contact.  Roe is additionally
  exact for an isolated shock; AUSM+-up is built for uniform accuracy from
  low Mach number to hypersonic.

Second-order extension
----------------------
The flux-limited correction of Toro eq. (14.49) is applied **wave by wave on
the Roe characteristic decomposition**, whichever first-order flux is
selected.  This is deliberate and is the usual construction: the correction is
an antidiffusive term that needs *some* characteristic splitting to limit
field by field, and the Roe eigensystem is the natural one for the Euler
equations.  The resulting scheme is "flux X plus a characteristic-based TVD
correction", so the choice of ``flux`` governs the first-order dissipation
(and hence the robustness and the contact resolution) while the limiter
governs the order of accuracy.

Note that ``flux="roe"`` with ``second_order=False`` is exactly the scheme
documented in :mod:`riemann.roe`; the other fluxes simply replace its
first-order part.
"""

from __future__ import annotations

import numpy as np

from ._jit import njit
from .limiters import limiter_phi
from .roe import interface_eigensystem, roe_flux
from .thermo import euler_flux, pressure

__all__ = [
    "FLUXES",
    "FLUX_NAMES",
    "ROBUST_FLUXES",
    "flux_code",
    "interface_fluxes",
    "roe_fluxes",
]

# --------------------------------------------------------------------------- #
# Flux registry.  Integer codes so the dispatch compiles under Numba, which
# cannot take a Python function into a nopython kernel without extra machinery.
# --------------------------------------------------------------------------- #

FLUX_ROE = 0
FLUX_HLL = 1
FLUX_HLLC = 2
FLUX_RUSANOV = 3
FLUX_LF = 4
FLUX_AUSMUP = 5

#: Accepted flux names, including aliases, mapped to their integer code.
FLUXES = {
    "roe": FLUX_ROE,
    "hll": FLUX_HLL,
    # Our HLL uses Einfeldt's Roe-average wave-speed estimates, which is
    # precisely what the literature calls HLLE, so accept that spelling too.
    "hlle": FLUX_HLL,
    "hllc": FLUX_HLLC,
    "rusanov": FLUX_RUSANOV,
    # Rusanov is local Lax-Friedrichs; both names are in common use.
    "llf": FLUX_RUSANOV,
    "locallaxfriedrichs": FLUX_RUSANOV,
    "laxfriedrichs": FLUX_LF,
    "lf": FLUX_LF,
    "ausmup": FLUX_AUSMUP,
    "ausmplusup": FLUX_AUSMUP,   # "ausm+-up" and "ausm+up" normalise to this
    "ausm": FLUX_AUSMUP,         # convenient short form
}

#: Canonical display name for each code (inverse of :data:`FLUXES`).
FLUX_NAMES = {
    FLUX_ROE: "roe",
    FLUX_HLL: "hll",
    FLUX_HLLC: "hllc",
    FLUX_RUSANOV: "rusanov",
    FLUX_LF: "lax_friedrichs",
    FLUX_AUSMUP: "ausmup",
}

#: Fluxes that run every built-in problem at second order, Toro's near-vacuum
#: Test 2 included.  Used by the CLI and the tests.
#:
#: HLL and HLLC are positivity-preserving by construction with Einfeldt wave
#: speeds, and Lax-Friedrichs is so diffusive that nothing troubles it.
#: Rusanov is deliberately *absent*: it is positivity-preserving as a
#: first-order flux, but that proof does not extend to the second-order TVD
#: correction, and the combination does break down on Toro's Test 5.  The
#: distinction is worth keeping -- "the flux is positivity-preserving" and
#: "the scheme is" are not the same statement.
ROBUST_FLUXES = frozenset({FLUX_HLL, FLUX_HLLC, FLUX_LF})


def flux_code(flux):
    """Normalise a flux specifier (name or integer code) to its integer code."""
    if isinstance(flux, str):
        key = flux.strip().lower()
        for ch in (" ", "_", "-", "+"):
            key = key.replace(ch, "")
        if key not in FLUXES:
            raise ValueError(
                f"unknown flux {flux!r}; choose from "
                f"{', '.join(sorted(set(FLUX_NAMES.values())))}"
            )
        return FLUXES[key]
    code = int(flux)
    if code not in FLUX_NAMES:
        raise ValueError(
            f"unknown flux code {code}; valid codes are {sorted(FLUX_NAMES)}"
        )
    return code


# --------------------------------------------------------------------------- #
# Central / Rusanov-type fluxes
# --------------------------------------------------------------------------- #

@njit
def lax_friedrichs_flux(u_l, u_r, gamma, dx, dt, out):
    """Classical Lax-Friedrichs flux (Toro Sec. 5.3.4).

        F = 0.5 (F_L + F_R) - 0.5 (dx/dt) (U_R - U_L)

    The dissipation coefficient is the *mesh* speed ``dx/dt``, which by the
    CFL condition is at least the fastest signal speed anywhere on the grid.
    Applying that global maximum at every interface makes this the most
    dissipative scheme here by a wide margin -- it is included as the
    reference point that shows what the smarter fluxes buy, not as a
    recommendation.
    """
    f_l = euler_flux(u_l, gamma)
    f_r = euler_flux(u_r, gamma)
    s = dx / dt
    for m in range(3):
        out[m] = 0.5 * (f_l[m] + f_r[m]) - 0.5 * s * (u_r[m] - u_l[m])


@njit
def rusanov_flux(u_l, u_r, gamma, out):
    """Rusanov (local Lax-Friedrichs) flux (Toro Sec. 10.5).

        F = 0.5 (F_L + F_R) - 0.5 S+ (U_R - U_L),
        S+ = max(|u_L| + a_L, |u_R| + a_R)

    Identical in form to Lax-Friedrichs but with the *local* maximum signal
    speed instead of the global mesh speed, which is a large improvement at
    no extra cost.  Still a single scalar coefficient for all three fields,
    so the contact is smeared about as badly as by first-order Roe.
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


# --------------------------------------------------------------------------- #
# HLL-type fluxes
# --------------------------------------------------------------------------- #

@njit
def _einfeldt_wave_speeds(u_l, u_r, gamma):
    """Einfeldt's wave-speed estimates (Toro Sec. 10.5).

        S_L = min(u_L - a_L, u~ - a~),   S_R = max(u_R + a_R, u~ + a~)

    where ``u~``, ``a~`` are the Roe averages.  Taking the *minimum* over both
    the physical left speed and the Roe speed (and the maximum on the right)
    is what guarantees the estimates bracket every wave in the exact solution,
    which in turn is what makes the resulting HLL flux positivity-preserving.
    Using only the physical speeds (Davis' estimate) is cheaper but can fail
    to bracket a strong shock.

    Returns ``(S_L, S_R)``.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    # Roe averages (same construction as riemann.roe.roe_average).
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
    """HLL flux with Einfeldt wave speeds (Toro Sec. 10.3, eq. 10.21).

    HLL assumes a two-wave structure -- a left and a right wave with a single
    constant state between them -- and obtains that state by applying the
    integral form of the conservation laws over the wave fan:

        F_HLL = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)

    for ``S_L <= 0 <= S_R``, and the upwind physical flux otherwise.

    Because the assumed structure has no middle wave, the contact
    discontinuity is averaged away: HLL smears contacts even more than
    first-order Roe does.  Shocks, on the other hand, it captures well, and it
    is unconditionally positivity-preserving with these wave speeds -- which
    is why it succeeds on Toro's Test 2 where Roe fails outright.
    """
    s_l, s_r = _einfeldt_wave_speeds(u_l, u_r, gamma)

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
    """HLLC flux -- HLL with the Contact wave restored (Toro Sec. 10.4).

    HLLC adds a middle wave of speed ``S_*`` between the two HLL waves, so the
    fan carries two star states rather than one.  ``S_*`` follows from
    requiring pressure and normal velocity to be continuous across it
    (Toro eq. 10.37):

        S_* = [p_R - p_L + rho_L u_L (S_L - u_L) - rho_R u_R (S_R - u_R)]
              / [rho_L (S_L - u_L) - rho_R (S_R - u_R)]

    and the star states from the Rankine-Hugoniot conditions across the outer
    waves (Toro eq. 10.39):

        U*_K = rho_K (S_K - u_K)/(S_K - S_*)
               * [1, S_*, E_K/rho_K + (S_* - u_K)(S_* + p_K/(rho_K (S_K - u_K)))]

    with ``F*_K = F_K + S_K (U*_K - U_K)``.

    Restoring the contact is the whole point: HLLC resolves contacts and shear
    about as sharply as Roe, while keeping HLL's positivity.  For this code it
    is the best general-purpose choice, and it is the flux that makes Toro's
    Test 2 runnable.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)

    s_l, s_r = _einfeldt_wave_speeds(u_l, u_r, gamma)

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

    # Contact speed.  The denominator cannot vanish while S_L < 0 < S_R,
    # because both bracketed terms then have the same sign.
    num = (
        p_r - p_l
        + rho_l * vel_l * (s_l - vel_l)
        - rho_r * vel_r * (s_r - vel_r)
    )
    den = rho_l * (s_l - vel_l) - rho_r * (s_r - vel_r)
    s_star = num / den

    if s_star >= 0.0:
        # Left star state.
        f = euler_flux(u_l, gamma)
        coef = rho_l * (s_l - vel_l) / (s_l - s_star)
        e_term = u_l[2] / rho_l + (s_star - vel_l) * (
            s_star + p_l / (rho_l * (s_l - vel_l))
        )
        out[0] = f[0] + s_l * (coef - u_l[0])
        out[1] = f[1] + s_l * (coef * s_star - u_l[1])
        out[2] = f[2] + s_l * (coef * e_term - u_l[2])
    else:
        # Right star state.
        f = euler_flux(u_r, gamma)
        coef = rho_r * (s_r - vel_r) / (s_r - s_star)
        e_term = u_r[2] / rho_r + (s_star - vel_r) * (
            s_star + p_r / (rho_r * (s_r - vel_r))
        )
        out[0] = f[0] + s_r * (coef - u_r[0])
        out[1] = f[1] + s_r * (coef * s_star - u_r[1])
        out[2] = f[2] + s_r * (coef * e_term - u_r[2])


# --------------------------------------------------------------------------- #
# AUSM-type flux
# --------------------------------------------------------------------------- #
#
# AUSM+-up constants.  Two groups, with different standing.
#
# beta = 1/8 and alpha = 3/16 fix the split Mach and pressure polynomials
# (Liou, J. Comput. Phys. 129:364, 1996).  They are not tunable: they are the
# unique values that make polynomials of the stated degrees smooth at
# |M| = 1.  Expanded, they give exactly the closed polynomial forms used in
# EMS -- see the note in `ausm_up_flux`.
_AUSM_BETA = 0.125     # beta in the split Mach polynomials M+_(4), M-_(4)
_AUSM_ALPHA = 0.1875   # alpha = 3/16 in the pressure splittings P+_(5), P-_(5)

# K_p and K_u scale the two "-up" diffusion terms (Liou, J. Comput. Phys.
# 214:137, 2006).  These *are* tunable, and the values below are the ones used
# and benchmarked in EMS (the Eulerian Multicomponent Solver of the same
# research group), `src/riemann_solvers/ausm6.cpp`.  They are not the values
# an earlier version of this file guessed at; see `ausm_up_flux` for what went
# wrong and why it matters.
_AUSM_KP = 0.4         # pressure-diffusion term in the interface velocity
_AUSM_KU = 0.2         # velocity-diffusion term in the interface pressure


@njit
def _m1_plus(m):
    """First-degree split Mach number ``M+_(1) = 0.5 (M + |M|)``."""
    return 0.5 * (m + abs(m))


@njit
def _m1_minus(m):
    """First-degree split Mach number ``M-_(1) = 0.5 (M - |M|)``."""
    return 0.5 * (m - abs(m))


@njit
def _m2_plus(m):
    """Second-degree split Mach number ``M+_(2) = +0.25 (M + 1)^2``."""
    return 0.25 * (m + 1.0) * (m + 1.0)


@njit
def _m2_minus(m):
    """Second-degree split Mach number ``M-_(2) = -0.25 (M - 1)^2``."""
    return -0.25 * (m - 1.0) * (m - 1.0)


@njit
def _m4_plus(m, beta):
    """Fourth-degree split Mach number ``M+_(4)``.

    ``M+_(1)`` for ``|M| >= 1``, else ``M+_(2) (1 - 16 beta M-_(2))``.
    Together with :func:`_m4_minus` it satisfies the consistency identity
    ``M+_(4)(M) + M-_(4)(M) = M`` for every ``M`` and every ``beta`` -- the
    cross terms cancel -- which the test suite checks directly.
    """
    if abs(m) >= 1.0:
        return _m1_plus(m)
    return _m2_plus(m) * (1.0 - 16.0 * beta * _m2_minus(m))


@njit
def _m4_minus(m, beta):
    """Fourth-degree split Mach number ``M-_(4)``."""
    if abs(m) >= 1.0:
        return _m1_minus(m)
    return _m2_minus(m) * (1.0 + 16.0 * beta * _m2_plus(m))


@njit
def _p5_plus(m, alpha):
    """Fifth-degree pressure splitting ``P+_(5)``.

    ``M+_(1)/M`` for ``|M| >= 1`` (i.e. 1 upstream, 0 downstream), else
    ``M+_(2)[(2 - M) - 16 alpha M M-_(2)]``.  With :func:`_p5_minus` it
    satisfies ``P+_(5)(M) + P-_(5)(M) = 1`` for every ``M`` and every
    ``alpha``, which is what makes the pressure splitting consistent.
    """
    if abs(m) >= 1.0:
        return _m1_plus(m) / m
    return _m2_plus(m) * ((2.0 - m) - 16.0 * alpha * m * _m2_minus(m))


@njit
def _p5_minus(m, alpha):
    """Fifth-degree pressure splitting ``P-_(5)``."""
    if abs(m) >= 1.0:
        return _m1_minus(m) / m
    return _m2_minus(m) * ((-2.0 - m) + 16.0 * alpha * m * _m2_plus(m))


@njit
def ausm_up_flux(u_l, u_r, gamma, out):
    """AUSM+-up flux (Liou, J. Comput. Phys. 214:137, 2006).

    AUSM-family schemes do not decompose the jump into waves.  They split the
    flux into a *convective* part carried by an interface velocity and a
    separate *pressure* part:

        F = u_1/2 * Psi_(upwind) + [0, p_1/2, 0],   Psi = [rho, rho u, E + p]

    with the convective vector upwinded on the sign of ``u_1/2``.  Because the
    pressure is handled entirely by ``p_1/2``, the convective part carries no
    pressure dissipation, and the scheme reproduces a contact discontinuity --
    stationary or moving -- exactly.

    The interface velocity and pressure are built from polynomial splittings of
    each side's Mach number ``M_K = u_K / a_1/2``, plus the two "-up" terms
    that give the scheme its name:

        u_1/2 = a_1/2 [M+_(4)(M_L) + M-_(4)(M_R)]
                + K_p max(1 - Mbar^2, 0) (p_L - p_R) / (rho_1/2 a_1/2),

        p_1/2 = P+_(5)(M_L) p_L + P-_(5)(M_R) p_R
                + K_u P+_(5) P-_(5) rho_1/2 a_1/2 (u_L - u_R),

    where ``a_1/2 = (a_L + a_R)/2``, ``rho_1/2 = (rho_L + rho_R)/2`` and
    ``Mbar^2 = (M_L^2 + M_R^2)/2``.

    The ``K_p`` term is the pressure-velocity coupling that plain AUSM+ lacks.
    Without it the mass flux is identically zero whenever ``u_L = u_R = 0``,
    however large the pressure jump, so AUSM+ alone cannot start a flow from
    rest -- it cannot run Sod's problem at all.  The ``K_u`` term is the
    corresponding velocity diffusion in the pressure.

    Provenance, and a warning about reconstructing this from memory
    ---------------------------------------------------------------
    The split polynomials ``M+-_(4)`` and ``P+-_(5)`` are standard and are
    pinned by two exact identities, checked in the test suite:

        M+_(4)(M) + M-_(4)(M) = M,      P+_(5)(M) + P-_(5)(M) = 1,

    both independent of ``alpha`` and ``beta``.  Expanding the forms used here
    reproduces, to machine precision, the closed polynomials written out in
    EMS ``src/riemann_solvers/ausm6.cpp::M4P5``, e.g. for ``|M| < 1``

        M+_(4) = M^4/8 + M/2 + 3/8,
        P+_(5) = 3 M^5/16 - 5 M^3/8 + 15 M/16 + 1/2.

    The ``-up`` terms are a different matter.  An earlier version of this file
    reconstructed them without access to the 2006 paper and got them wrong in
    three compounding ways: it used Liou's critical-speed construction for
    ``a_1/2`` instead of the arithmetic mean, it carried an ``f_a`` all-speed
    scaling that is absent here, and -- the fatal one -- it wrote the ``K_u``
    term with ``K_u = 0.75`` against ``(rho_L + rho_R)`` rather than
    ``K_u = 0.2`` against ``rho_1/2``, making that term **7.5 times too
    large**.  The result was anti-dissipative: on a plain moving contact it
    amplified round-off by about 1.5 per step until the pressure error reached
    50%, with a measured stability threshold at roughly 0.35 in its own
    parameterisation -- just below the 0.75 it was using, and far below the
    1.5 it effectively applied.

    The coefficients and the ``a_1/2`` definition here are taken from EMS,
    where they are benchmarked.  The sign and structure of both terms agree
    with what was reconstructed; only the scaling was wrong, which is exactly
    the kind of error that identity tests cannot catch and a moving-contact
    test can.  ``test_moving_contact_keeps_pressure_constant`` is that test.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    # Interface averages: plain arithmetic means of the one-sided values.
    a_half = 0.5 * (a_l + a_r)
    rho_half = 0.5 * (rho_l + rho_r)

    mach_l = vel_l / a_half
    mach_r = vel_r / a_half
    mbar_sq = 0.5 * (mach_l * mach_l + mach_r * mach_r)

    # --- interface velocity, with the K_p pressure-diffusion term --------- #
    # max(1 - Mbar^2, 0) switches the term off once the interface is
    # supersonic, where the upwinding already supplies enough dissipation.
    u_half = a_half * (
        _m4_plus(mach_l, _AUSM_BETA) + _m4_minus(mach_r, _AUSM_BETA)
    ) + _AUSM_KP * max(1.0 - mbar_sq, 0.0) * (p_l - p_r) / (rho_half * a_half)

    # --- interface pressure, with the K_u velocity-diffusion term --------- #
    p5p = _p5_plus(mach_l, _AUSM_ALPHA)
    p5m = _p5_minus(mach_r, _AUSM_ALPHA)
    p_half = (
        p5p * p_l
        + p5m * p_r
        + _AUSM_KU * p5p * p5m * rho_half * a_half * (vel_l - vel_r)
    )

    # --- assemble: upwind Psi = [rho, rho u, E + p] on the sign of u_1/2 -- #
    if u_half > 0.0:
        out[0] = u_half * rho_l
        out[1] = u_half * u_l[1] + p_half
        out[2] = u_half * (u_l[2] + p_l)
    else:
        out[0] = u_half * rho_r
        out[1] = u_half * u_r[1] + p_half
        out[2] = u_half * (u_r[2] + p_r)


# --------------------------------------------------------------------------- #
# Dispatch and the grid-level loop
# --------------------------------------------------------------------------- #

@njit
def _first_order_flux(
    u_l, u_r, gamma, dx, dt, flux_code_, use_entropy_fix, lam, kmat, alpha, out
):
    """Evaluate the selected first-order flux at one interface, into ``out``.

    ``lam``, ``kmat`` and ``alpha`` are the Roe eigenstructure of this
    interface.  Only the Roe branch uses them; they are passed unconditionally
    so that the dispatch has one signature, which keeps it compilable.
    """
    if flux_code_ == FLUX_ROE:
        roe_flux(u_l, u_r, gamma, lam, kmat, alpha, use_entropy_fix, out)
    elif flux_code_ == FLUX_HLL:
        hll_flux(u_l, u_r, gamma, out)
    elif flux_code_ == FLUX_HLLC:
        hllc_flux(u_l, u_r, gamma, out)
    elif flux_code_ == FLUX_RUSANOV:
        rusanov_flux(u_l, u_r, gamma, out)
    elif flux_code_ == FLUX_LF:
        lax_friedrichs_flux(u_l, u_r, gamma, dx, dt, out)
    else:  # FLUX_AUSMUP
        ausm_up_flux(u_l, u_r, gamma, out)


@njit
def interface_fluxes(
    u_all, gamma, dx, dt, flux_code_, limiter_code_, use_entropy_fix,
    second_order,
):
    """Fluxes at every interface of the (ghosted) grid.

    Parameters
    ----------
    u_all : ndarray, shape (n_cells, 3)
        Conservative states including two ghost cells at each end.
    gamma : float
        Ratio of specific heats.
    dx, dt : float
        Mesh spacing and the step being taken.  ``dt`` enters the second-order
        correction through the Courant numbers, and the Lax-Friedrichs flux
        through ``dx/dt``, so the flux is not a function of the states alone.
    flux_code_ : int
        Which first-order flux to use; see :data:`FLUXES`.
    limiter_code_ : int
        Which flux limiter to use; see :mod:`riemann.limiters`.
    use_entropy_fix : bool
        Harten-Hyman entropy fix.  Only meaningful for ``flux="roe"`` -- the
        other fluxes here are already entropy-satisfying, because none of them
        represents a wave family by a single eigenvalue that cannot change
        sign.  It is ignored by them.
    second_order : bool
        Add the flux-limited correction of Toro eq. (14.49).
    Returns
    -------
    flux : ndarray, shape (n_cells - 1, 3)
        The first and last interfaces are left at zero: they cannot be formed
        (the limiter needs an upwind neighbour interface) and are never used,
        since only interior cells are updated.
    """
    n_cells = u_all.shape[0]
    n_if = n_cells - 1

    lam_all = np.empty((n_if, 3))
    k_all = np.empty((n_if, 3, 3))
    alpha_all = np.empty((n_if, 3))
    # The Roe eigenstructure is needed by the Roe flux itself and by the
    # limiter correction (for any flux).  Skip it when neither applies.
    if second_order or flux_code_ == FLUX_ROE:
        interface_eigensystem(u_all, gamma, lam_all, k_all, alpha_all)

    flux = np.zeros((n_if, 3))
    f = np.empty(3)

    for j in range(1, n_if - 1):
        u_l = u_all[j]
        u_r = u_all[j + 1]
        lam = lam_all[j]
        alpha = alpha_all[j]
        kmat = k_all[j]

        _first_order_flux(
            u_l, u_r, gamma, dx, dt, flux_code_, use_entropy_fix,
            lam, kmat, alpha, f,
        )

        # ---- second-order TVD correction, Toro eq. (14.49) --------------- #
        # Applied on the Roe characteristic fields whichever base flux was
        # used; see this module's docstring.
        if second_order:
            for k in range(3):
                nu = lam[k] * dt / dx
                # The weight vanishes identically when nu == 0 (which implies
                # lambda == 0), so there is nothing to add and no upwind
                # direction to pick.
                if nu == 0.0:
                    continue
                sgn = 1.0 if nu > 0.0 else -1.0
                j_up = j - 1 if nu > 0.0 else j + 1
                num = alpha_all[j_up, k]
                den = alpha[k]
                # Guard the ratio: where the local wave is absent the whole
                # correction is proportional to den and vanishes anyway, so
                # falling back to r = 0 (phi = 0) is safe and harmless.
                if den != 0.0 and abs(den) > 1.0e-12 * abs(num):
                    r = num / den
                else:
                    r = 0.0
                phi = limiter_phi(r, limiter_code_)
                w = 0.5 * lam[k] * (sgn - nu) * alpha[k] * phi
                for m in range(3):
                    f[m] += w * kmat[k, m]

        for m in range(3):
            flux[j, m] = f[m]

    return flux


def roe_fluxes(u_all, gamma, dx, dt, limiter_code_, use_entropy_fix, second_order):
    """Backwards-compatible wrapper: :func:`interface_fluxes` with the Roe flux.

    Kept so that code written against the previous release keeps working.
    New code should call :func:`interface_fluxes` and pass ``flux_code_``.
    """
    return interface_fluxes(
        u_all, gamma, dx, dt, FLUX_ROE, limiter_code_, use_entropy_fix,
        second_order,
    )
