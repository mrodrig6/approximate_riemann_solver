"""ADER: a one-step scheme built on the derivative Riemann problem.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
3rd ed., Chapter 19 (ADER for linear systems) and Chapter 20 (the non-linear
case, the derivative Riemann problem and the Cauchy-Kovalewski procedure).

What ADER is for
----------------
The TVD scheme of :mod:`src.euler.fluxes.interface` reaches second order by
adding an antidiffusive correction to a first-order flux.  ADER gets there a
different way, and the difference matters for how it generalises: it
reconstructs a polynomial in each cell, solves a Riemann problem *for the
derivatives as well as the states*, and uses the Cauchy-Kovalewski procedure
to turn time derivatives into space derivatives.  The interface state is then
known as a function of time within the step, so the flux can be integrated
over the step by quadrature.

The result is a genuinely **one-step** scheme: no Runge-Kutta stages, one
Riemann solve per interface per step, and -- in the general ADER formulation
-- arbitrary order from the same template, by carrying more terms of the
Taylor series in time.  This module implements the second-order member,
ADER2, which is the one that can be written without a symbolic
Cauchy-Kovalewski expansion.

The algorithm
-------------
For each cell a limited linear reconstruction gives a slope ``s_i``, and
hence boundary-extrapolated values ``U_i +- s_i/2``.  At interface
``i + 1/2``, with ``U_L = U_i + s_i/2`` and ``U_R = U_{i+1} - s_{i+1}/2``:

1. **Leading term.**  Solve the classical Riemann problem ``(U_L, U_R)`` and
   take the state on the interface, ``U*(0)``.  This is the Godunov state and
   is what a first-order scheme would use.

2. **Derivative term.**  The spatial derivative is itself discontinuous
   across the interface, with left and right values ``s_i / dx`` and
   ``s_{i+1} / dx``.  Evolve it with the system *linearised about* ``U*(0)``
   and take the value on the interface: projecting each side onto the
   characteristic fields of ``A(U*(0))`` and picking the upwind component
   field by field,

       (dU/dx)* = sum_k beta_k K_k,   beta_k = alpha_k^L if lambda_k > 0
                                               alpha_k^R otherwise.

3. **Cauchy-Kovalewski.**  The governing equation itself converts the time
   derivative into the space derivative:

       dU/dt = -A(U) dU/dx   =>   (dU/dt)* = -sum_k lambda_k beta_k K_k.

   No Jacobian is ever formed: the eigen-decomposition that step 2 already
   needs supplies the product directly.

4. **Flux.**  The interface state is now linear in time within the step,
   ``U(tau) = U*(0) + tau (dU/dt)*``, so the time integral of the physical
   flux is second-order accurate by the midpoint rule:

       F_{i+1/2} = F( U*(0) + (dt/2) (dU/dt)* ).

Note what does *not* appear: no numerical flux function.  ADER evaluates the
*physical* flux at a state, so the only Riemann solve is the one producing
``U*(0)``.  Here that state comes from the HLLC solver, which is cheap,
positivity-preserving and resolves the contact exactly.

Accuracy
--------
Second order in space and time on smooth solutions, verified by grid
convergence on a smooth isentropic pulse in the test suite.  At
discontinuities the slope limiter reduces the reconstruction to
piecewise-constant and the scheme degrades gracefully to first-order Godunov,
exactly as the TVD scheme does.
"""

from __future__ import annotations

import numpy as np

from ..core.jit import njit
from .fluxes.hll import einfeldt_wave_speeds
from ..core.limiters import limiter_phi
from .physics import euler_flux, pressure

__all__ = ["ader2_fluxes"]


@njit
def _is_physical(u_cell, gamma):
    """True only for a state with positive density and positive pressure.

    Two details matter and both are deliberate.

    The density test comes first and returns early, because
    :func:`~src.euler.physics.pressure` divides by the density -- asking for
    the pressure of a zero-density state is itself the error this guard
    exists to prevent.

    The tests are written as ``> 0.0`` rather than ``not <= 0.0`` so that a
    NaN, which compares ``False`` against everything, is reported as
    non-physical.  A reconstruction that has already produced a NaN is
    exactly the case that must reach the fallback.
    """
    if not (u_cell[0] > 0.0):
        return False
    return pressure(u_cell, gamma) > 0.0


@njit
def _godunov_state_hllc(u_l, u_r, gamma, out):
    """The state on the interface, ``x/t = 0``, from the HLLC wave model.

    HLLC's assumed fan has the two outer waves and the contact, so sampling
    it at the interface gives either an unmodified input state or one of the
    two star states (Toro eq. 10.39) -- the same states its flux is built
    from.  That makes this the natural companion to the HLLC flux, and it
    resolves a contact exactly.
    """
    rho_l, rho_r = u_l[0], u_r[0]
    vel_l, vel_r = u_l[1] / rho_l, u_r[1] / rho_r
    p_l, p_r = pressure(u_l, gamma), pressure(u_r, gamma)
    s_l, s_r = einfeldt_wave_speeds(u_l, u_r, gamma)

    if s_l >= 0.0:
        for m in range(3):
            out[m] = u_l[m]
        return
    if s_r <= 0.0:
        for m in range(3):
            out[m] = u_r[m]
        return

    num = (
        p_r - p_l
        + rho_l * vel_l * (s_l - vel_l)
        - rho_r * vel_r * (s_r - vel_r)
    )
    den = rho_l * (s_l - vel_l) - rho_r * (s_r - vel_r)
    s_star = num / den

    if s_star >= 0.0:
        coef = rho_l * (s_l - vel_l) / (s_l - s_star)
        out[0] = coef
        out[1] = coef * s_star
        out[2] = coef * (
            u_l[2] / rho_l
            + (s_star - vel_l) * (s_star + p_l / (rho_l * (s_l - vel_l)))
        )
    else:
        coef = rho_r * (s_r - vel_r) / (s_r - s_star)
        out[0] = coef
        out[1] = coef * s_star
        out[2] = coef * (
            u_r[2] / rho_r
            + (s_star - vel_r) * (s_star + p_r / (rho_r * (s_r - vel_r)))
        )


@njit
def _state_eigensystem(u_state, gamma, lam, kmat):
    """Eigenvalues and right eigenvectors of ``A(U)`` at a single state.

    Same structure as the Roe eigensystem, but evaluated at an actual state
    rather than at an average of two, since ADER linearises about the
    Godunov state ``U*(0)`` rather than across a jump.
    """
    rho = u_state[0]
    vel = u_state[1] / rho
    p = pressure(u_state, gamma)
    a = np.sqrt(gamma * p / rho)
    h = (u_state[2] + p) / rho     # total enthalpy

    lam[0] = vel - a
    lam[1] = vel
    lam[2] = vel + a

    kmat[0, 0] = 1.0
    kmat[0, 1] = vel - a
    kmat[0, 2] = h - vel * a
    kmat[1, 0] = 1.0
    kmat[1, 1] = vel
    kmat[1, 2] = 0.5 * vel * vel
    kmat[2, 0] = 1.0
    kmat[2, 1] = vel + a
    kmat[2, 2] = h + vel * a
    return vel, h, a


@njit
def _project(d, vel, h, a, gamma, alpha):
    """Characteristic components of ``d`` in the eigenbasis at a state.

    Solves ``d = sum_k alpha_k K_k`` in closed form; the inversion is the
    same one the Roe wave strengths use (Toro eqs. 11.68-11.70), evaluated
    here at a single state.
    """
    a2 = ((gamma - 1.0) / (a * a)) * (
        d[0] * (h - vel * vel) + vel * d[1] - d[2]
    )
    a1 = (1.0 / (2.0 * a)) * (d[0] * (vel + a) - d[1] - a * a2)
    alpha[0] = a1
    alpha[1] = a2
    alpha[2] = d[0] - (a1 + a2)


@njit
def _limited_slopes(u_all, limiter_code_, slopes):
    """Limited linear-reconstruction slopes, one per cell, into ``slopes``.

    The slope is written as ``phi(r) * (U_{i+1} - U_i)`` with
    ``r = (U_i - U_{i-1}) / (U_{i+1} - U_i)``, so the same limiter functions
    that the TVD flux correction uses apply unchanged here.

    Two caveats, both measured rather than supposed.  The Sweby bound makes
    each *conserved component* monotone; it says nothing about the pressure,
    which is a difference of large numbers and can still come out negative.
    And limiting componentwise on the conserved variables is **not Galilean
    invariant**, where the TVD correction limits on the characteristic wave
    strengths and is: adding a uniform velocity to Toro's test 3 and varying
    it alone makes this scheme fail from about -5 onwards while the TVD
    scheme runs throughout, which is why ADER fails Toro's test 5 (the same
    problem shifted by -19.59745).  Limiting on primitive or characteristic
    variables is the standard fix and is not done here; see the limitations
    section of ``docs/ader.md``.
    """
    n_cells = u_all.shape[0]
    for i in range(n_cells):
        for m in range(3):
            slopes[i, m] = 0.0
    for i in range(1, n_cells - 1):
        for m in range(3):
            back = u_all[i, m] - u_all[i - 1, m]
            fwd = u_all[i + 1, m] - u_all[i, m]
            if fwd != 0.0 and abs(fwd) > 1.0e-12 * abs(back):
                r = back / fwd
            else:
                r = 0.0
            slopes[i, m] = limiter_phi(r, limiter_code_) * fwd


@njit
def ader2_fluxes(u_all, gamma, dx, dt, limiter_code_):
    """Second-order ADER interface fluxes for the whole (ghosted) grid.

    Parameters
    ----------
    u_all : ndarray, shape (n_cells, 3)
        Conservative states including two ghost cells at each end.
    gamma : float
        Ratio of specific heats.
    dx, dt : float
        Mesh spacing and the step being taken.  ``dt`` enters through the
        midpoint-rule evaluation time ``dt / 2``.
    limiter_code_ : int
        Slope limiter for the reconstruction; see :mod:`src.core.limiters`.

    Returns
    -------
    flux : ndarray, shape (n_cells - 1, 3)
        The first and last interfaces are left at zero, as for the TVD
        fluxes: they lack a complete reconstruction stencil and are never
        used, since only interior cells are updated.
    """
    n_cells = u_all.shape[0]
    n_if = n_cells - 1

    slopes = np.empty((n_cells, 3))
    _limited_slopes(u_all, limiter_code_, slopes)

    flux = np.zeros((n_if, 3))
    u_l = np.empty(3)
    u_r = np.empty(3)
    u_star = np.empty(3)
    d_l = np.empty(3)
    d_r = np.empty(3)
    alpha_l = np.empty(3)
    alpha_r = np.empty(3)
    lam = np.empty(3)
    kmat = np.empty((3, 3))
    u_mid = np.empty(3)

    for j in range(1, n_if - 1):
        # Boundary-extrapolated states either side of the interface.
        for m in range(3):
            u_l[m] = u_all[j, m] + 0.5 * slopes[j, m]
            u_r[m] = u_all[j + 1, m] - 0.5 * slopes[j + 1, m]
            d_l[m] = slopes[j, m] / dx
            d_r[m] = slopes[j + 1, m] / dx

        # 1. Leading term: the Godunov state on the interface.
        #
        # The extrapolated states are checked *before* the wave model sees
        # them, not only afterwards.  A limited slope can still push a
        # boundary-extrapolated state to zero density or negative pressure on
        # a strong enough jump, and the wave model cannot survive either: it
        # divides by the density to form the velocity, and takes the square
        # root of the pressure to form the sound speed.  Checking only the
        # resulting state is too late -- the division has already happened.
        if not (_is_physical(u_l, gamma) and _is_physical(u_r, gamma)):
            for m in range(3):
                u_l[m] = u_all[j, m]
                u_r[m] = u_all[j + 1, m]
                d_l[m] = 0.0
                d_r[m] = 0.0

        _godunov_state_hllc(u_l, u_r, gamma, u_star)

        # The cell averages are physical -- the time loop checks that every
        # step -- and HLLC with Einfeldt speeds is positivity-preserving, so
        # this second guard should not fire once the first has.  It is kept
        # because "should not" is not "cannot", and the cost is two
        # comparisons at an interface that is already in trouble.
        if not _is_physical(u_star, gamma):
            for m in range(3):
                u_l[m] = u_all[j, m]
                u_r[m] = u_all[j + 1, m]
                d_l[m] = 0.0
                d_r[m] = 0.0
            _godunov_state_hllc(u_l, u_r, gamma, u_star)

        # 2 and 3. Derivative Riemann problem, then Cauchy-Kovalewski.
        vel, h, a = _state_eigensystem(u_star, gamma, lam, kmat)
        _project(d_l, vel, h, a, gamma, alpha_l)
        _project(d_r, vel, h, a, gamma, alpha_r)

        # dU/dt = -sum_k lambda_k beta_k K_k, upwinding each field.
        for m in range(3):
            u_mid[m] = u_star[m]
        for k in range(3):
            beta = alpha_l[k] if lam[k] > 0.0 else alpha_r[k]
            w = -0.5 * dt * lam[k] * beta
            for m in range(3):
                u_mid[m] += w * kmat[k, m]

        # 4. Midpoint rule on the physical flux.
        if _is_physical(u_mid, gamma):
            f = euler_flux(u_mid, gamma)
        else:
            # Keep the step physical if the half-step state is not.
            f = euler_flux(u_star, gamma)
        for m in range(3):
            flux[j, m] = f[m]

    return flux
