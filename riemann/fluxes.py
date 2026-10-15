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
``ausm``           AUSM+, Liou, J. Comput. Phys. 129:364 (1996)
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
* **Flux-vector / AUSM-type** (``ausm``) and **flux-difference** (``roe``).
  Both resolve all three fields exactly at a contact.  Roe is additionally
  exact for an isolated shock.

See :func:`ausm_plus_flux` for a note on what is *not* implemented here: the
low-Mach ``-up`` extension of AUSM+, and the SLAU scheme.

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
FLUX_AUSM = 5

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
    "ausm": FLUX_AUSM,
    "ausmplus": FLUX_AUSM,   # "ausm+" normalises to this
}

#: Canonical display name for each code (inverse of :data:`FLUXES`).
FLUX_NAMES = {
    FLUX_ROE: "roe",
    FLUX_HLL: "hll",
    FLUX_HLLC: "hllc",
    FLUX_RUSANOV: "rusanov",
    FLUX_LF: "lax_friedrichs",
    FLUX_AUSM: "ausm",
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
# AUSM+ constants, Liou, "A sequel to AUSM: AUSM+", J. Comput. Phys. 129:364
# (1996).  These are not tunable: beta = 1/8 and alpha = 3/16 are the unique
# values that make the split Mach and pressure polynomials of the stated
# degrees smooth at |M| = 1, which is what the scheme is built on.
_AUSM_BETA = 0.125     # beta in the split Mach polynomials M+_(4), M-_(4)
_AUSM_ALPHA = 0.1875   # alpha = 3/16 in the pressure splittings P+_(5), P-_(5)


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
def ausm_plus_flux(u_l, u_r, gamma, out):
    """AUSM+ flux (Liou, J. Comput. Phys. 129:364, 1996).

    AUSM-family schemes split the flux into a *convective* part carried by an
    interface mass flux and a *pressure* part, rather than decomposing the
    jump into waves:

        F = mdot_1/2 * Psi_(upwind) + [0, p_1/2, 0],   Psi = [1, u, H]

    The interface Mach number and pressure come from polynomial splittings of
    each side's Mach number,

        M_1/2 = M+_(4)(M_L) + M-_(4)(M_R),
        p_1/2 = P+_(5)(M_L) p_L + P-_(5)(M_R) p_R,

    and the convective vector is upwinded on the sign of ``mdot``.  Because
    the pressure is handled entirely by the separate ``p_1/2`` term, the
    convective part carries no pressure dissipation, and AUSM+ reproduces a
    contact discontinuity -- stationary or moving -- exactly.  The test suite
    checks both.

    Known limitation: it cannot start a flow from rest
    ---------------------------------------------------
    AUSM+ has no pressure-velocity coupling in the *mass* flux.  At an
    interface with ``u_L = u_R = 0`` the split Mach numbers give
    ``M_1/2 = M+_(4)(0) + M-_(4)(0) = 0`` exactly, however large the pressure
    jump is, so both the mass flux and the energy flux ``mdot H`` vanish.  The
    momentum equation still feels the pressure difference and accelerates the
    gas, but with the total energy held fixed the growing kinetic energy is
    subtracted from the internal energy.  On the light side of a shock tube
    the same impulse produces a much larger velocity, and the internal energy
    goes negative in the first step.

    The practical consequence is sharp: this flux cannot run Sod's problem, or
    any other case started from rest.  Measured here on Sod with a uniform
    initial velocity ``u_0``, it fails for ``u_0 <= 0.01`` and succeeds from
    ``u_0 >= 0.1`` -- i.e. it needs a local Mach number of roughly 0.1 to
    function.  It runs the cases in this repository that begin with moving gas
    (``toro2``, ``toro5``, ``sonic_rarefaction``, ``air_shock_tube``) perfectly
    well.

    That deficiency is not a defect of this implementation; it is the exact
    deficiency AUSM+-up was written to cure, by adding the ``M_p`` term below.

    What is **not** implemented here
    --------------------------------
    This is AUSM+ (1996), not AUSM+-up (Liou, J. Comput. Phys. 214:137, 2006).
    AUSM+-up adds two low-Mach diffusion terms, ``M_p`` in the mass flux and
    ``p_u`` in the pressure, scaled by a function ``f_a`` of a reference Mach
    number, so that the dissipation stays proportional to the flow speed
    rather than the sound speed as ``M -> 0``.

    Those terms are omitted deliberately.  Their coefficients could not be
    checked against the 2006 paper from the environment this port was written
    in (its network policy reaches GitHub and the package registries only), and
    a reconstruction of ``p_u`` from memory proved to be anti-dissipative: on a
    plain moving contact it amplified round-off by a factor of about 1.5 per
    step, reaching a 50% pressure error, and the measured stability threshold
    sat near ``K_u = 0.35`` against the published value of 0.75.  Shipping an
    unverifiable term that is demonstrably unstable on a textbook problem
    would be worse than shipping the well-established scheme it extends, so
    what is here is plain AUSM+, whose split functions satisfy their exact
    consistency identities and which preserves both stationary and moving
    contacts to machine precision.

    For low-Mach work, add the ``-up`` terms from the 2006 paper (and, for
    SLAU, Shima & Kitamura, AIAA J. 49:1693, 2011); the structure below is
    arranged so that they slot into ``m_half`` and ``p_half``.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)
    h_l = (u_l[2] + p_l) / rho_l   # total enthalpy
    h_r = (u_r[2] + p_r) / rho_r

    # --- interface speed of sound (Liou's critical-speed construction) ---- #
    # a* is the speed at which the flow would be sonic for the given total
    # enthalpy.  Taking the minimum of the two one-sided values makes the
    # interface sound speed respect an incoming shock from either side.
    fac = 2.0 * (gamma - 1.0) / (gamma + 1.0)
    a_star_l = np.sqrt(fac * h_l)
    a_star_r = np.sqrt(fac * h_r)
    a_hat_l = a_star_l * a_star_l / max(a_star_l, vel_l)
    a_hat_r = a_star_r * a_star_r / max(a_star_r, -vel_r)
    a_half = min(a_hat_l, a_hat_r)

    mach_l = vel_l / a_half
    mach_r = vel_r / a_half

    # --- interface Mach number and pressure ------------------------------- #
    m_half = _m4_plus(mach_l, _AUSM_BETA) + _m4_minus(mach_r, _AUSM_BETA)
    p_half = (
        _p5_plus(mach_l, _AUSM_ALPHA) * p_l
        + _p5_minus(mach_r, _AUSM_ALPHA) * p_r
    )

    # --- assemble: upwind the convective vector on the sign of mdot ------- #
    if m_half > 0.0:
        mdot = a_half * m_half * rho_l
        out[0] = mdot
        out[1] = mdot * vel_l + p_half
        out[2] = mdot * h_l
    else:
        mdot = a_half * m_half * rho_r
        out[0] = mdot
        out[1] = mdot * vel_r + p_half
        out[2] = mdot * h_r


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
    else:  # FLUX_AUSM
        ausm_plus_flux(u_l, u_r, gamma, out)


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
