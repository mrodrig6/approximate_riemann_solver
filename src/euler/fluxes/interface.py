"""Interface fluxes for the whole grid: flux dispatch plus the TVD correction.

This is the single entry point the finite-volume driver calls.  It selects the
first-order flux, then optionally adds the flux-limited correction that makes
the scheme second order.

The correction (Toro eq. 14.49) is applied **wave by wave on the Roe
characteristic decomposition**, whichever first-order flux is selected.  That
is the usual construction: the correction is an antidiffusive term and needs
*some* characteristic splitting to limit field by field, and the Roe
eigensystem is the natural one for the Euler equations.  The practical
division of labour is that ``flux`` governs the first-order dissipation --
and hence the robustness and the contact resolution -- while ``limiter``
governs the order of accuracy.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Sec. 13.8 and Sec. 14.4.
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ...core.limiters import limiter_phi
from .ausm import ausm_up_flux
from .central import lax_friedrichs_flux, rusanov_flux
from .hll import hll_flux, hllc_flux
from .registry import FLUX_HLL, FLUX_HLLC, FLUX_LF, FLUX_ROE, FLUX_RUSANOV
from .roe import interface_eigensystem, roe_flux

__all__ = ["interface_fluxes"]


@njit
def _first_order_flux(
    u_l, u_r, gamma, dx, dt, flux_code_, use_entropy_fix, lam, kmat, alpha, out
):
    """Evaluate the selected first-order flux at one interface, into ``out``.

    ``lam``, ``kmat`` and ``alpha`` are the interface's Roe eigenstructure.
    Only the Roe branch uses them; they are passed unconditionally so the
    dispatch keeps one signature, which is what makes it compilable.
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
    u_all, gamma, dx, dt, flux_code_, limiter_code_, use_entropy_fix, second_order
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
        through ``dx / dt``, so the flux is not a function of the states alone.
    flux_code_ : int
        Which first-order flux to use; see :mod:`src.euler.fluxes.registry`.
    limiter_code_ : int
        Which flux limiter to use; see :mod:`src.core.limiters`.
    use_entropy_fix : bool
        Harten-Hyman entropy fix.  Only the Roe flux uses it: the others
        represent no wave family by a single eigenvalue that cannot change
        sign, so they are already entropy-satisfying.
    second_order : bool
        Add the flux-limited correction.

    Returns
    -------
    flux : ndarray, shape (n_cells - 1, 3)
        The first and last interfaces are left at zero: they cannot be formed,
        since the limiter needs an upwind neighbour interface, and they are
        never used, since only interior cells are updated.
    """
    n_cells = u_all.shape[0]
    n_if = n_cells - 1

    lam_all = np.empty((n_if, 3))
    k_all = np.empty((n_if, 3, 3))
    alpha_all = np.empty((n_if, 3))
    # The Roe eigenstructure is needed by the Roe flux itself and by the
    # limiter correction for any flux.  Skip it when neither applies.
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

        if second_order:
            for k in range(3):
                nu = lam[k] * dt / dx
                # The weight vanishes identically when nu == 0 (which implies
                # lambda == 0): nothing to add, and no upwind direction.
                if nu == 0.0:
                    continue
                sgn = 1.0 if nu > 0.0 else -1.0
                j_up = j - 1 if nu > 0.0 else j + 1
                num = alpha_all[j_up, k]
                den = alpha[k]
                # Guard the smoothness ratio: where the local wave is absent
                # the whole correction is proportional to den and vanishes
                # anyway, so falling back to r = 0 (phi = 0) is harmless.
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
