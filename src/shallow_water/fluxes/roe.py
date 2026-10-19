"""Roe's approximate Riemann solver, with a Harten entropy fix.

Roe replaces the nonlinear Jacobian ``A(U) = dF/dU`` by a constant matrix
``A~(U_L, U_R)`` evaluated at a specially averaged state, then solves the
resulting linear Riemann problem exactly.  As for the Euler equations, the
averages are the unique ones making the U-property

    F(U_R) - F(U_L) = A~ (U_R - U_L)

hold exactly, and the velocity is weighted by the square root of the first
conserved variable.  The celerity average is the simpler of the two, because
``F`` is quadratic in ``U`` here where for a gas it is not.

This module supplies the Roe flux, the averaged state, and the eigenstructure
that the flux-limited second-order correction uses for *every* flux, since
that correction needs a characteristic splitting to limit field by field --
the same division of labour as :mod:`src.euler.fluxes.roe`.

The one addition the shallow-water case needs is a dry bed.  The square-root
weighting divides by ``sqrt(h_L) + sqrt(h_R)``, so the linearisation has no
meaning when either side is dry; :func:`roe_flux` falls back to HLL there,
and :func:`interface_eigensystem_wet` skips those interfaces rather than
producing a NaN that would spread across the grid.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Sec. 7.5 (the averages and eigenstructure) and Sec. 8.4 (the
entropy fix).
"""

from __future__ import annotations

import numpy as np

from ...core.jit import njit
from ..physics import DRY_TOLERANCE, celerity, swe_flux
from .hll import hll_flux

__all__ = [
    "interface_eigensystem",
    "interface_eigensystem_wet",
    "roe_average",
    "roe_flux",
]


@njit
def roe_average(u_l, u_r, gravity):
    """Roe-averaged velocity and celerity for the shallow-water system.

    As for the Euler equations, the averages are the unique ones making the
    U-property ``F_R - F_L = A~ (U_R - U_L)`` hold exactly, and as there the
    velocity is weighted by the square root of the first conserved variable:

        u~ = (sqrt(h_L) u_L + sqrt(h_R) u_R) / (sqrt(h_L) + sqrt(h_R)),
        a~ = sqrt(g (h_L + h_R) / 2).

    The celerity average is the simpler of the two -- an arithmetic mean of
    the depths under the square root -- because ``F`` is quadratic in ``U``
    here, where for a gas it is not.

    Returns ``(u_tilde, a_tilde)``.
    """
    h_l = u_l[0]
    h_r = u_r[0]
    sq_l = np.sqrt(h_l)
    sq_r = np.sqrt(h_r)
    vel_l = u_l[1] / h_l
    vel_r = u_r[1] / h_r
    u_tilde = (sq_l * vel_l + sq_r * vel_r) / (sq_l + sq_r)
    a_tilde = np.sqrt(0.5 * gravity * (h_l + h_r))
    return u_tilde, a_tilde


@njit
def interface_eigensystem(u_all, gravity, lam_all, k_all, alpha_all):
    """Fill the Roe eigenstructure for every interface of ``u_all``.

    For each interface ``j`` this stores the eigenvalues
    ``[u~ - a~, u~ + a~]``, the right eigenvectors

        K_1 = [1, u~ - a~],    K_2 = [1, u~ + a~],

    and the wave strengths from projecting ``dU = U_R - U_L`` onto them:

        alpha_1 = ( dU_1 (u~ + a~) - dU_2) / (2 a~),
        alpha_2 = (-dU_1 (u~ - a~) + dU_2) / (2 a~).

    With only two fields the projection is a 2x2 inversion and can be written
    out directly, where the Euler case needs the three-term formula.
    """
    n_if = u_all.shape[0] - 1
    for j in range(n_if):
        u_l = u_all[j]
        u_r = u_all[j + 1]
        u_t, a_t = roe_average(u_l, u_r, gravity)

        lam_all[j, 0] = u_t - a_t
        lam_all[j, 1] = u_t + a_t

        k_all[j, 0, 0] = 1.0
        k_all[j, 0, 1] = u_t - a_t
        k_all[j, 1, 0] = 1.0
        k_all[j, 1, 1] = u_t + a_t

        d0 = u_r[0] - u_l[0]
        d1 = u_r[1] - u_l[1]
        alpha_all[j, 0] = (d0 * (u_t + a_t) - d1) / (2.0 * a_t)
        alpha_all[j, 1] = (-d0 * (u_t - a_t) + d1) / (2.0 * a_t)


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
