"""Roe's approximate Riemann solver with a Harten-Hyman entropy fix and a
second-order TVD flux-limiter extension.

References (Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed.):

* Chapter 11, Sec. 11.2-11.3: the Roe linearisation, the Roe-average state,
  the eigenstructure and the wave strengths.
* Chapter 11, Sec. 11.3 (Harten-Hyman): the entropy fix for a transonic
  rarefaction.
The flux-limited second-order extension and the dispatch over the other
flux functions live in :mod:`riemann.fluxes`; this module supplies the Roe
eigenstructure that both the Roe flux and that limiter correction are built
from.

The idea
--------
Roe replaces the nonlinear Jacobian ``A(U) = dF/dU`` by a constant matrix
``A~(U_L, U_R)`` evaluated at a specially averaged state, chosen so that

    F(U_R) - F(U_L) = A~ (U_R - U_L)                      (Toro eq. 11.18)

holds *exactly* -- the "U-property".  The linear problem then has an exact
solution built from the eigenvectors of ``A~``, giving the flux

    F_{i+1/2} = F(U_L) + sum_k min(lambda_k, 0) alpha_k K_k   (Toro eq. 11.27)

where ``lambda_k``, ``K_k`` are the eigenvalues/right eigenvectors of ``A~``
and ``alpha_k`` are the wave strengths obtained by projecting the jump
``U_R - U_L`` onto the eigenvectors.

The U-property is what makes Roe's solver resolve isolated shocks *exactly*
(one-cell, no smearing), and is why it remains the workhorse upwind flux
despite needing an entropy fix.

Grid / index conventions used throughout this module
---------------------------------------------------
``U`` has shape ``(n_cells, 3)`` and *includes* two ghost cells at each end.
Interface ``j`` lies between cells ``j`` and ``j + 1``, so there are
``n_cells - 1`` interfaces.  Fluxes are only computed for interfaces
``1 .. n_cells - 3``, which is exactly the set needed to update every interior
cell *and* to reach the upwind neighbour interface ``j -/+ 1`` that the flux
limiter needs.

Bugs fixed relative to the MATLAB original (``roe_flux_e.m``, ``flx_lim.m``)
---------------------------------------------------------------------------
1. In the right-transonic branch the MATLAB wrote
   ``flx = fl - lam3b*al3*K(3,:)``.  The Harten-Hyman flux for that branch is
   built from the *right* physical flux: ``F_R - lam3b*al3*K3``.  Using ``fl``
   there is simply the wrong state and produced a locally inconsistent flux.
2. The right star state used the *left* star pressure and velocity
   (``pstar``, ``ustar`` computed from ``U_L + alpha_1 K_1``) together with the
   right star density.  Both star states are now built from their own
   conservative states, ``U_L + alpha_1 K_1`` and ``U_R - alpha_3 K_3``.
3. The second-order limiter correction was applied only to the single
   transonic wave inside the entropy-fix branches, and to all three waves
   otherwise.  It is now applied uniformly to all three waves, so the scheme's
   order of accuracy does not silently drop at transonic interfaces.
4. ``flx_lim.m`` divided by the local wave strength with no guard, giving
   ``Inf``/``NaN`` wherever a wave was absent (e.g. the entropy wave in a
   uniform region).  The ratio is now guarded.
5. ``flx_lim.m`` recomputed the full Roe average of the *upwind* interface on
   every call, i.e. three redundant Roe averages per interface per step.  Wave
   strengths for all interfaces are now computed once per step into an array.
"""

from __future__ import annotations

import numpy as np

from ._jit import njit
from .thermo import euler_flux, pressure

__all__ = ["roe_average", "interface_eigensystem", "roe_flux"]


@njit
def roe_average(u_l, u_r, gamma):
    """Roe-averaged velocity, total enthalpy and sound speed (Toro eq. 11.60).

    The averages are weighted by ``sqrt(rho)``::

        u~ = (sqrt(rho_L) u_L + sqrt(rho_R) u_R) / (sqrt(rho_L) + sqrt(rho_R))
        H~ = (sqrt(rho_L) H_L + sqrt(rho_R) H_R) / (sqrt(rho_L) + sqrt(rho_R))
        a~ = sqrt((gamma - 1) (H~ - u~^2 / 2))

    This particular weighting is not an aesthetic choice: it is the unique one
    (for an ideal gas) that makes the U-property hold exactly.

    Returns
    -------
    (u_tilde, h_tilde, a_tilde) : tuple[float, float, float]
    """
    rho_l = u_l[0]
    rho_r = u_r[0]
    sq_l = np.sqrt(rho_l)
    sq_r = np.sqrt(rho_r)

    vel_l = u_l[1] / rho_l
    vel_r = u_r[1] / rho_r
    p_l = pressure(u_l, gamma)
    p_r = pressure(u_r, gamma)
    h_l = (u_l[2] + p_l) / rho_l   # total enthalpy, not static
    h_r = (u_r[2] + p_r) / rho_r

    u_tilde = (sq_l * vel_l + sq_r * vel_r) / (sq_l + sq_r)
    h_tilde = (sq_l * h_l + sq_r * h_r) / (sq_l + sq_r)
    a_tilde = np.sqrt((gamma - 1.0) * (h_tilde - 0.5 * u_tilde * u_tilde))
    return u_tilde, h_tilde, a_tilde


@njit
def interface_eigensystem(u_all, gamma, lam_all, k_all, alpha_all):
    """Fill the Roe eigenstructure for every interface of ``u_all``.

    For each interface ``j`` (between cells ``j`` and ``j + 1``) this stores

    * ``lam_all[j, :]``  -- eigenvalues ``[u~ - a~, u~, u~ + a~]``
      (Toro eq. 11.64);
    * ``k_all[j, m, :]`` -- the ``m``-th right eigenvector ``K_m``
      (Toro eq. 11.65)::

          K_1 = [1, u~ - a~, H~ - u~ a~]
          K_2 = [1, u~,      u~^2 / 2  ]
          K_3 = [1, u~ + a~, H~ + u~ a~]

    * ``alpha_all[j, :]`` -- wave strengths from projecting the jump
      ``dU = U_R - U_L`` onto the eigenvectors (Toro eqs. 11.68-11.70)::

          alpha_2 = ((gamma - 1)/a~^2) [ dU_1 (H~ - u~^2) + u~ dU_2 - dU_3 ]
          alpha_1 = (1/(2 a~)) [ dU_1 (u~ + a~) - dU_2 - a~ alpha_2 ]
          alpha_3 = dU_1 - (alpha_1 + alpha_2)

    Doing this in one pass (rather than recomputing the upwind interface inside
    the limiter, as the MATLAB did) cuts the Roe averages per step by 3x.
    """
    n_if = u_all.shape[0] - 1
    for j in range(n_if):
        u_l = u_all[j]
        u_r = u_all[j + 1]
        u_t, h_t, a_t = roe_average(u_l, u_r, gamma)

        lam_all[j, 0] = u_t - a_t
        lam_all[j, 1] = u_t
        lam_all[j, 2] = u_t + a_t

        k_all[j, 0, 0] = 1.0
        k_all[j, 0, 1] = u_t - a_t
        k_all[j, 0, 2] = h_t - u_t * a_t
        k_all[j, 1, 0] = 1.0
        k_all[j, 1, 1] = u_t
        k_all[j, 1, 2] = 0.5 * u_t * u_t
        k_all[j, 2, 0] = 1.0
        k_all[j, 2, 1] = u_t + a_t
        k_all[j, 2, 2] = h_t + u_t * a_t

        d0 = u_r[0] - u_l[0]  # jump in rho
        d1 = u_r[1] - u_l[1]  # jump in rho*u
        d2 = u_r[2] - u_l[2]  # jump in E

        a2 = ((gamma - 1.0) / (a_t * a_t)) * (
            d0 * (h_t - u_t * u_t) + u_t * d1 - d2
        )
        a1 = (1.0 / (2.0 * a_t)) * (d0 * (u_t + a_t) - d1 - a_t * a2)
        a3 = d0 - (a1 + a2)

        alpha_all[j, 0] = a1
        alpha_all[j, 1] = a2
        alpha_all[j, 2] = a3


@njit
def _negative_wave_speeds(u_l, u_r, gamma, lam, kmat, alpha, use_entropy_fix, lam_neg):
    """Effective negative wave speeds, with the Harten-Hyman entropy fix.

    The first-order Roe flux is ``F_L + sum_k lam~_k alpha_k K_k`` with
    ``lam~_k = min(lam_k, 0)``.  That form is entropy-violating at a
    *transonic* rarefaction -- one whose fan straddles the interface, so that
    the physical characteristic speed changes sign inside the wave while the
    single Roe eigenvalue cannot.  The symptom is the familiar unphysical
    "rarefaction shock" sitting at the sonic point.

    Harten and Hyman's fix (Toro Sec. 11.3) replaces the offending eigenvalue
    by a weighted speed that interpolates across the sonic point:

        lam1_bar = lam1_L (lam1_* - lam1) / (lam1_* - lam1_L)   if lam1_L < 0 < lam1_*
        lam3_bar = lam3_R (lam3 - lam3_*) / (lam3_R - lam3_*)   if lam3_* < 0 < lam3_R

    where ``lam1_L = u_L - a_L``, ``lam3_R = u_R + a_R`` are the physical edge
    speeds of the fan and ``lam1_*``, ``lam3_*`` are the corresponding speeds
    evaluated in the Roe star states ``U_L + alpha_1 K_1`` and
    ``U_R - alpha_3 K_3``.

    For the left-transonic case we set ``lam~_1 = lam1_bar`` directly.  For the
    right-transonic case Harten-Hyman gives ``F = F_R - lam3_bar alpha_3 K_3``;
    using the U-property ``F_R - F_L = sum_k lam_k alpha_k K_k`` and the fact
    that waves 1 and 2 are then both left-going, this is equivalent to setting
    ``lam~_3 = lam3 - lam3_bar`` in the ``F_L``-based form, which is what we
    do so that a single code path produces the flux.

    ``kmat`` is the ``(3, 3)`` matrix of right eigenvectors for this
    interface, so that the energy components ``K_1[2] = H~ - u~ a~`` and
    ``K_3[2] = H~ + u~ a~`` needed for the star states are reused rather than
    recomputed from a second Roe average.

    ``lam_neg`` is filled in place with the three effective speeds.
    """
    for k in range(3):
        lam_neg[k] = min(lam[k], 0.0)

    if not use_entropy_fix:
        return

    rho_l = u_l[0]
    rho_r = u_r[0]
    vel_l = u_l[1] / rho_l
    vel_r = u_r[1] / rho_r
    p_l = pressure(u_l, gamma)
    p_r = pressure(u_r, gamma)
    if p_l <= 0.0 or p_r <= 0.0:
        return
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)

    lam1_l = vel_l - a_l   # left edge (head) of a left rarefaction fan
    lam3_r = vel_r + a_r   # right edge (head) of a right rarefaction fan

    # --- left acoustic wave ------------------------------------------------ #
    # Star state behind wave 1 in the Roe linearisation: U*_L = U_L + a1 K1.
    rho_sl = rho_l + alpha[0]
    if rho_sl > 0.0:
        mom_sl = u_l[1] + alpha[0] * lam[0]
        # K1[2] = H~ - u~ a~, and lam[0] = u~ - a~, so this reuses them.
        ene_sl = u_l[2] + alpha[0] * kmat[0, 2]
        vel_sl = mom_sl / rho_sl
        p_sl = (gamma - 1.0) * (ene_sl - 0.5 * mom_sl * mom_sl / rho_sl)
        if p_sl > 0.0:
            lam1_star = vel_sl - np.sqrt(gamma * p_sl / rho_sl)
            if lam1_l < 0.0 < lam1_star:
                denom = lam1_star - lam1_l
                if denom != 0.0:
                    lam_neg[0] = lam1_l * (lam1_star - lam[0]) / denom
                    return  # a transonic wave 1 precludes a transonic wave 3

    # --- right acoustic wave ----------------------------------------------- #
    # Star state ahead of wave 3: U*_R = U_R - a3 K3.
    rho_sr = rho_r - alpha[2]
    if rho_sr > 0.0:
        mom_sr = u_r[1] - alpha[2] * lam[2]
        ene_sr = u_r[2] - alpha[2] * kmat[2, 2]
        vel_sr = mom_sr / rho_sr
        p_sr = (gamma - 1.0) * (ene_sr - 0.5 * mom_sr * mom_sr / rho_sr)
        if p_sr > 0.0:
            lam3_star = vel_sr + np.sqrt(gamma * p_sr / rho_sr)
            if lam3_star < 0.0 < lam3_r:
                denom = lam3_r - lam3_star
                if denom != 0.0:
                    lam3_bar = lam3_r * (lam[2] - lam3_star) / denom
                    lam_neg[2] = lam[2] - lam3_bar


@njit
def roe_flux(u_l, u_r, gamma, lam, kmat, alpha, use_entropy_fix, out):
    """First-order Roe flux at one interface, written into ``out``.

    Implements Toro eq. (11.27),

        F = F(U_L) + sum_k min(lambda_k, 0) alpha_k K_k,

    with ``min(lambda_k, 0)`` replaced by the Harten-Hyman speeds at a
    transonic rarefaction when ``use_entropy_fix`` is set.

    Parameters
    ----------
    u_l, u_r : ndarray, shape (3,)
        Conservative states either side of the interface.
    gamma : float
        Ratio of specific heats.
    lam, kmat, alpha : ndarray
        The interface's Roe eigenvalues ``(3,)``, right eigenvectors
        ``(3, 3)`` and wave strengths ``(3,)``, as filled by
        :func:`interface_eigensystem`.  They are passed in rather than
        recomputed because the caller already needs them for the flux limiter.
    use_entropy_fix : bool
        Apply the Harten-Hyman entropy fix.
    out : ndarray, shape (3,)
        Output buffer, overwritten.
    """
    lam_neg = np.empty(3)
    _negative_wave_speeds(u_l, u_r, gamma, lam, kmat, alpha, use_entropy_fix, lam_neg)

    f = euler_flux(u_l, gamma)
    for k in range(3):
        w = lam_neg[k] * alpha[k]
        for m in range(3):
            f[m] += w * kmat[k, m]
    for m in range(3):
        out[m] = f[m]
