"""Flux dispatch at every interface, and the second-order TVD correction.

The single entry point the solver calls.  It selects the first-order flux by
integer code, then optionally adds the flux-limited antidiffusive term,
exactly as :mod:`src.euler.fluxes.interface` does -- here on two
characteristic fields rather than three.

The correction is switched off at any interface touching a dry cell, where
the characteristic decomposition does not exist and a limited antidiffusive
term would drive the depth negative.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Chapter 10; the correction itself is Toro, *Riemann Solvers and
Numerical Methods for Fluid Dynamics*, 3rd ed., eq. 14.49.
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ...core.limiters import limiter_phi
from ..physics import DRY_TOLERANCE
from .central import lax_friedrichs_flux, rusanov_flux
from .hll import hll_flux
from .registry import FLUX_HLL, FLUX_ROE, FLUX_RUSANOV
from .roe import interface_eigensystem_wet, roe_flux

__all__ = ["interface_fluxes"]


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
