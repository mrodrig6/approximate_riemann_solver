"""Roe's approximate Riemann solver, with the Harten-Hyman entropy fix.

Roe replaces the nonlinear Jacobian ``A(U) = dF/dU`` by a constant matrix
``A~(U_L, U_R)`` evaluated at a specially averaged state, then solves the
resulting linear Riemann problem exactly.  The averaging is chosen so that

    F(U_R) - F(U_L) = A~ (U_R - U_L)                      (Toro eq. 11.18)

holds *exactly* -- the "U-property".  That is what makes a Roe scheme capture
an isolated shock in one or two cells: a discontinuity that solves the
nonlinear problem also solves the linearised one.

This module supplies both the Roe flux itself and the eigenstructure that the
flux-limited second-order correction uses for *every* flux, since that
correction needs a characteristic splitting to limit field by field.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Sec. 11.2 (linearisation, averages, eigenstructure, wave strengths)
and Sec. 11.3 (the entropy fix).
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ..physics import euler_flux, pressure

__all__ = ["roe_average", "interface_eigensystem", "roe_flux"]


@njit
def roe_average(u_l, u_r, gamma):
    """Roe-averaged velocity, total enthalpy and sound speed (Toro eq. 11.60).

    The averages are weighted by ``sqrt(rho)``::

        u~ = (sqrt(rho_L) u_L + sqrt(rho_R) u_R) / (sqrt(rho_L) + sqrt(rho_R))
        H~ = (sqrt(rho_L) H_L + sqrt(rho_R) H_R) / (sqrt(rho_L) + sqrt(rho_R))
        a~ = sqrt((gamma - 1)(H~ - u~^2 / 2))

    That weighting is not an aesthetic choice: for an ideal gas it is the
    unique one making the U-property hold exactly.  ``H`` is the *total*
    enthalpy ``(E + p) / rho``, not the static one.

    Returns ``(u_tilde, h_tilde, a_tilde)``.
    """
    rho_l = u_l[0]
    rho_r = u_r[0]
    sq_l = np.sqrt(rho_l)
    sq_r = np.sqrt(rho_r)

    vel_l = u_l[1] / rho_l
    vel_r = u_r[1] / rho_r
    p_l = pressure(u_l, gamma)
    p_r = pressure(u_r, gamma)
    h_l = (u_l[2] + p_l) / rho_l
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
    * ``k_all[j, m, :]`` -- the ``m``-th right eigenvector (Toro eq. 11.65)::

          K_1 = [1, u~ - a~, H~ - u~ a~]
          K_2 = [1, u~,      u~^2 / 2  ]
          K_3 = [1, u~ + a~, H~ + u~ a~]

    * ``alpha_all[j, :]`` -- wave strengths from projecting the jump
      ``dU = U_R - U_L`` onto the eigenvectors (Toro eqs. 11.68-11.70)::

          alpha_2 = ((gamma - 1)/a~^2)[dU_1 (H~ - u~^2) + u~ dU_2 - dU_3]
          alpha_1 = (1/(2 a~))[dU_1 (u~ + a~) - dU_2 - a~ alpha_2]
          alpha_3 = dU_1 - (alpha_1 + alpha_2)

    Filling all interfaces in one pass means each Roe average is formed once
    per step, rather than again for every wave family that needs it.
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

    The Roe flux is ``F_L + sum_k lam~_k alpha_k K_k`` with
    ``lam~_k = min(lam_k, 0)``.  That form is entropy-violating at a
    *transonic* rarefaction -- one whose fan straddles the interface, so the
    physical characteristic speed changes sign inside the wave while the
    single Roe eigenvalue representing the family cannot.  The symptom is an
    unphysical "rarefaction shock" at the sonic point whose size does not
    shrink under mesh refinement.

    Harten and Hyman's fix (Toro Sec. 11.3) replaces the offending eigenvalue
    by a speed interpolated across the sonic point:

        lam1_bar = lam1_L (lam1_* - lam1)/(lam1_* - lam1_L)  if lam1_L < 0 < lam1_*
        lam3_bar = lam3_R (lam3 - lam3_*)/(lam3_R - lam3_*)  if lam3_* < 0 < lam3_R

    where ``lam1_L = u_L - a_L`` and ``lam3_R = u_R + a_R`` are the physical
    edge speeds of the fan, and ``lam1_*``, ``lam3_*`` the corresponding
    speeds in the Roe star states ``U_L + alpha_1 K_1`` and
    ``U_R - alpha_3 K_3``.

    For a transonic wave 1 we set ``lam~_1 = lam1_bar`` directly.  For a
    transonic wave 3 Harten-Hyman gives ``F = F_R - lam3_bar alpha_3 K_3``;
    by the U-property, and because waves 1 and 2 are then both left-going,
    that is the same as setting ``lam~_3 = lam3 - lam3_bar`` in the
    ``F_L``-based form, so one code path covers both.  The two cases are
    mutually exclusive: a transonic wave 1 requires ``lam1_* > 0``, which
    forces ``lam3_* > 0``.

    ``kmat`` is the interface's ``(3, 3)`` eigenvector matrix, so the energy
    components ``K_1[2]`` and ``K_3[2]`` needed for the star states are reused
    rather than recomputed.  ``lam_neg`` is filled in place.
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

    lam1_l = vel_l - a_l   # head of a left rarefaction fan
    lam3_r = vel_r + a_r   # head of a right rarefaction fan

    # --- left acoustic wave: star state U*_L = U_L + alpha_1 K_1 ---------- #
    rho_sl = rho_l + alpha[0]
    if rho_sl > 0.0:
        mom_sl = u_l[1] + alpha[0] * lam[0]
        ene_sl = u_l[2] + alpha[0] * kmat[0, 2]
        vel_sl = mom_sl / rho_sl
        p_sl = (gamma - 1.0) * (ene_sl - 0.5 * mom_sl * mom_sl / rho_sl)
        if p_sl > 0.0:
            lam1_star = vel_sl - np.sqrt(gamma * p_sl / rho_sl)
            if lam1_l < 0.0 < lam1_star:
                denom = lam1_star - lam1_l
                if denom != 0.0:
                    lam_neg[0] = lam1_l * (lam1_star - lam[0]) / denom
                    return  # precludes a transonic wave 3

    # --- right acoustic wave: star state U*_R = U_R - alpha_3 K_3 --------- #
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
    """Roe flux at one interface, written into ``out``.

    Implements Toro eq. (11.27),

        F = F(U_L) + sum_k min(lambda_k, 0) alpha_k K_k,

    with ``min(lambda_k, 0)`` replaced by the Harten-Hyman speeds at a
    transonic rarefaction when ``use_entropy_fix`` is set.

    ``lam``, ``kmat`` and ``alpha`` are this interface's eigenvalues,
    eigenvectors and wave strengths, as filled by
    :func:`interface_eigensystem`.  They are passed in rather than recomputed
    because the caller already holds them for the flux limiter.

    Roe's one hard limitation is that nothing here guarantees a positive
    updated internal energy.  Near a vacuum star state it fails outright, in
    the first time step and at first order as well as second; a
    positivity-preserving flux such as HLL or HLLC is required there.
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
