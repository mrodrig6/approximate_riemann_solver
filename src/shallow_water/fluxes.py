"""Numerical fluxes for the shallow-water equations, and the TVD correction.

The same four families as for the Euler system, minus one.  There is no HLLC
here: HLLC exists to restore the contact wave that HLL averages away, and the
shallow-water system has no contact -- both of its fields are genuinely
non-linear.  With no middle wave to put back, HLLC reduces exactly to HLL, so
the name is accepted as an alias rather than implemented separately.

A single module rather than the one-per-family package used for the Euler
fluxes: with two equations and four fluxes the whole thing is shorter than
any one of those files.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Chapters 7-10; the TVD correction follows the same construction
as :mod:`src.euler.fluxes.interface`.
"""

from __future__ import annotations

import numpy as np

from ..core.jit import njit
from ..core.limiters import limiter_phi
from .physics import DRY_TOLERANCE, celerity, swe_flux

__all__ = ["FLUXES", "FLUX_NAMES", "flux_code", "interface_fluxes"]

FLUX_ROE = 0
FLUX_HLL = 1
FLUX_RUSANOV = 2
FLUX_LF = 3

#: Accepted spellings.  ``hllc`` maps to HLL deliberately: see the module
#: docstring.
FLUXES = {
    "roe": FLUX_ROE,
    "hll": FLUX_HLL,
    "hlle": FLUX_HLL,
    "hllc": FLUX_HLL,
    "rusanov": FLUX_RUSANOV,
    "llf": FLUX_RUSANOV,
    "laxfriedrichs": FLUX_LF,
    "lf": FLUX_LF,
}

FLUX_NAMES = {
    FLUX_ROE: "roe",
    FLUX_HLL: "hll",
    FLUX_RUSANOV: "rusanov",
    FLUX_LF: "lax_friedrichs",
}


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


@njit
def _wave_speeds(u_l, u_r, gravity):
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
    s_l, s_r = _wave_speeds(u_l, u_r, gravity)
    s_plus = max(abs(s_l), abs(s_r))
    f_l = np.empty(2)
    f_r = np.empty(2)
    swe_flux(u_l, gravity, f_l)
    swe_flux(u_r, gravity, f_r)
    for m in range(2):
        out[m] = 0.5 * (f_l[m] + f_r[m]) - 0.5 * s_plus * (u_r[m] - u_l[m])


@njit
def hll_flux(u_l, u_r, gravity, out):
    """HLL flux: two waves, one intermediate state.

    With no contact in the system there is nothing for HLL to average away,
    so unlike the Euler case it is not at an accuracy disadvantage against a
    three-wave solver -- its two-wave model *is* the full wave structure.
    That, plus its positivity, makes it the natural default here.
    """
    s_l, s_r = _wave_speeds(u_l, u_r, gravity)
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


@njit
def roe_flux(u_l, u_r, gravity, lam, kmat, alpha, use_entropy_fix, out):
    """Roe flux, with a Harten entropy fix at a transonic rarefaction.

        F = F(U_L) + sum_k min(lambda_k, 0) alpha_k K_k

    The entropy fix is the simple Harten form: an eigenvalue smaller in
    magnitude than a threshold proportional to the local wave strength is
    replaced by a parabola through it.  The failure it repairs is the same
    one as for a gas -- a single eigenvalue cannot change sign inside a
    rarefaction fan that straddles the interface, so the scheme admits an
    unphysical stationary discontinuity at the critical (Froude = 1) point.

    Falls back to HLL when either side is dry: the Roe average divides by
    ``sqrt(h_L) + sqrt(h_R)`` and the linearisation has no meaning there.
    """
    if u_l[0] <= DRY_TOLERANCE or u_r[0] <= DRY_TOLERANCE:
        hll_flux(u_l, u_r, gravity, out)
        return

    f_l = np.empty(2)
    swe_flux(u_l, gravity, f_l)

    vel_l = u_l[1] / u_l[0]
    vel_r = u_r[1] / u_r[0]
    a_l = celerity(u_l[0], gravity)
    a_r = celerity(u_r[0], gravity)

    for k in range(2):
        lam_k = lam[k]
        if use_entropy_fix:
            # Harten's threshold: the spread of the physical speeds of this
            # family across the interface.
            if k == 0:
                delta = max(0.0, (vel_r - a_r) - (vel_l - a_l))
            else:
                delta = max(0.0, (vel_r + a_r) - (vel_l + a_l))
            if abs(lam_k) < delta and delta > 0.0:
                lam_k = 0.5 * (lam_k * lam_k / delta + delta)
                if lam[k] < 0.0:
                    lam_k = -lam_k
        w = min(lam_k, 0.0) * alpha[k]
        for m in range(2):
            f_l[m] += w * kmat[k, m]

    for m in range(2):
        out[m] = f_l[m]


@njit
def _first_order_flux(
    u_l, u_r, gravity, dx, dt, flux_code_, use_entropy_fix, lam, kmat, alpha, out
):
    """Dispatch to the selected first-order flux."""
    if flux_code_ == FLUX_ROE:
        roe_flux(u_l, u_r, gravity, lam, kmat, alpha, use_entropy_fix, out)
    elif flux_code_ == FLUX_HLL:
        hll_flux(u_l, u_r, gravity, out)
    elif flux_code_ == FLUX_RUSANOV:
        rusanov_flux(u_l, u_r, gravity, out)
    else:  # FLUX_LF
        lax_friedrichs_flux(u_l, u_r, gravity, dx, dt, out)


@njit
def interface_fluxes(
    u_all, gravity, dx, dt, flux_code_, limiter_code_, use_entropy_fix, second_order
):
    """Fluxes at every interface of the (ghosted) grid.

    The second-order correction is the same flux-limited term as for the
    Euler system (Toro eq. 14.49), applied wave by wave on the Roe
    characteristic fields -- here two fields rather than three.  It is
    switched off at any interface touching a dry cell, where the
    characteristic decomposition does not exist and a limited antidiffusive
    term would drive the depth negative.

    Returns an array of shape ``(n_cells - 1, 2)``; the first and last
    interfaces are left at zero and never used.
    """
    n_cells = u_all.shape[0]
    n_if = n_cells - 1

    lam_all = np.empty((n_if, 2))
    k_all = np.empty((n_if, 2, 2))
    alpha_all = np.empty((n_if, 2))
    wet_if = np.empty(n_if, dtype=np.bool_)
    for j in range(n_if):
        wet_if[j] = (
            u_all[j, 0] > DRY_TOLERANCE and u_all[j + 1, 0] > DRY_TOLERANCE
        )

    need_eig = second_order or flux_code_ == FLUX_ROE
    if need_eig:
        # Only form the eigensystem where both sides are wet; leave the rest
        # zeroed so a dry interface cannot produce a NaN that spreads.
        for j in range(n_if):
            if not wet_if[j]:
                for k in range(2):
                    lam_all[j, k] = 0.0
                    alpha_all[j, k] = 0.0
                    for m in range(2):
                        k_all[j, k, m] = 0.0
        interface_eigensystem_wet(u_all, gravity, lam_all, k_all, alpha_all, wet_if)

    flux = np.zeros((n_if, 2))
    f = np.empty(2)

    for j in range(1, n_if - 1):
        u_l = u_all[j]
        u_r = u_all[j + 1]
        lam = lam_all[j]
        alpha = alpha_all[j]
        kmat = k_all[j]

        _first_order_flux(
            u_l, u_r, gravity, dx, dt, flux_code_, use_entropy_fix,
            lam, kmat, alpha, f,
        )

        if second_order and wet_if[j] and wet_if[j - 1] and wet_if[j + 1]:
            for k in range(2):
                nu = lam[k] * dt / dx
                if nu == 0.0:
                    continue
                sgn = 1.0 if nu > 0.0 else -1.0
                j_up = j - 1 if nu > 0.0 else j + 1
                num = alpha_all[j_up, k]
                den = alpha[k]
                if den != 0.0 and abs(den) > 1.0e-12 * abs(num):
                    r = num / den
                else:
                    r = 0.0
                phi = limiter_phi(r, limiter_code_)
                w = 0.5 * lam[k] * (sgn - nu) * alpha[k] * phi
                for m in range(2):
                    f[m] += w * kmat[k, m]

        for m in range(2):
            flux[j, m] = f[m]

    return flux


@njit
def interface_eigensystem_wet(u_all, gravity, lam_all, k_all, alpha_all, wet_if):
    """:func:`~src.shallow_water.physics.interface_eigensystem`, wet only.

    Skips interfaces touching a dry cell, where the square-root weighting of
    the Roe average divides by zero.
    """
    n_if = u_all.shape[0] - 1
    for j in range(n_if):
        if not wet_if[j]:
            continue
        h_l = u_all[j, 0]
        h_r = u_all[j + 1, 0]
        sq_l = np.sqrt(h_l)
        sq_r = np.sqrt(h_r)
        vel_l = u_all[j, 1] / h_l
        vel_r = u_all[j + 1, 1] / h_r
        u_t = (sq_l * vel_l + sq_r * vel_r) / (sq_l + sq_r)
        a_t = np.sqrt(0.5 * gravity * (h_l + h_r))

        lam_all[j, 0] = u_t - a_t
        lam_all[j, 1] = u_t + a_t
        k_all[j, 0, 0] = 1.0
        k_all[j, 0, 1] = u_t - a_t
        k_all[j, 1, 0] = 1.0
        k_all[j, 1, 1] = u_t + a_t

        d0 = u_all[j + 1, 0] - u_all[j, 0]
        d1 = u_all[j + 1, 1] - u_all[j, 1]
        alpha_all[j, 0] = (d0 * (u_t + a_t) - d1) / (2.0 * a_t)
        alpha_all[j, 1] = (-d0 * (u_t - a_t) + d1) / (2.0 * a_t)
