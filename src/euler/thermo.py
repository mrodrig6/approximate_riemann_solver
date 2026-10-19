"""Ideal-gas thermodynamics and conversions between primitive and conservative
variables for the one-dimensional Euler equations.

Notation follows Toro, *Riemann Solvers and Numerical Methods for Fluid
Dynamics* (3rd ed., Springer 2009), Chapter 3.  The conservative form is

    dU/dt + dF(U)/dx = 0,
    U = [rho, rho*u, E],
    F = [rho*u, rho*u^2 + p, u*(E + p)],

closed by the ideal-gas caloric equation of state (Toro eq. 3.9)

    E = rho*e + 0.5*rho*u^2,     e = p / ((gamma - 1) * rho).

Throughout the code ``W = [rho, u, p]`` denotes the *primitive* state vector
and ``U = [rho, rho*u, E]`` the *conservative* one.
"""

from __future__ import annotations

import numpy as np

from ..core.jit import njit

__all__ = [
    "primitive_to_conservative",
    "conservative_to_primitive",
    "pressure",
    "sound_speed",
    "enthalpy",
    "euler_flux",
    "specific_internal_energy",
]


# --------------------------------------------------------------------------- #
# Scalar kernels (compiled).  These are the versions called from inside the
# finite-volume sweep, so they take plain floats / 1-D arrays only.
# --------------------------------------------------------------------------- #

@njit
def pressure(u_cell, gamma):
    """Pressure from a single conservative state ``[rho, rho*u, E]``.

    Inverts the caloric EOS: ``p = (gamma - 1) * (E - 0.5 * (rho*u)^2 / rho)``.
    """
    return (gamma - 1.0) * (u_cell[2] - 0.5 * u_cell[1] * u_cell[1] / u_cell[0])


@njit
def sound_speed(rho, p, gamma):
    """Ideal-gas sound speed ``a = sqrt(gamma * p / rho)`` (Toro eq. 3.17)."""
    return np.sqrt(gamma * p / rho)


@njit
def enthalpy(u_cell, gamma):
    """Specific *total* enthalpy ``H = (E + p) / rho`` (Toro eq. 11.59).

    The Roe average is built from ``H``, not from the static enthalpy, which is
    why this helper returns the total quantity.
    """
    p = pressure(u_cell, gamma)
    return (u_cell[2] + p) / u_cell[0]


@njit
def euler_flux(u_cell, gamma):
    """Physical flux ``F(U)`` for one conservative state (Toro eq. 3.3)."""
    rho = u_cell[0]
    mom = u_cell[1]
    ene = u_cell[2]
    vel = mom / rho
    p = pressure(u_cell, gamma)
    out = np.empty(3)
    out[0] = mom
    out[1] = mom * vel + p
    out[2] = vel * (ene + p)
    return out


@njit
def primitive_to_conservative_1(w, gamma):
    """Convert one primitive state ``[rho, u, p]`` to ``[rho, rho*u, E]``."""
    out = np.empty(3)
    out[0] = w[0]
    out[1] = w[0] * w[1]
    out[2] = w[2] / (gamma - 1.0) + 0.5 * w[0] * w[1] * w[1]
    return out


# --------------------------------------------------------------------------- #
# Vectorised helpers (NumPy).  Used for I/O, post-processing and plotting,
# where readability beats speed.
# --------------------------------------------------------------------------- #

def primitive_to_conservative(w, gamma):
    """Vectorised ``[rho, u, p] -> [rho, rho*u, E]``.

    Parameters
    ----------
    w : array_like, shape (3,) or (n, 3)
        Primitive state(s).
    gamma : float
        Ratio of specific heats.
    """
    w = np.atleast_2d(np.asarray(w, dtype=float))
    rho, vel, p = w[:, 0], w[:, 1], w[:, 2]
    out = np.empty_like(w)
    out[:, 0] = rho
    out[:, 1] = rho * vel
    out[:, 2] = p / (gamma - 1.0) + 0.5 * rho * vel * vel
    return out.squeeze() if out.shape[0] == 1 else out


def conservative_to_primitive(u, gamma):
    """Vectorised ``[rho, rho*u, E] -> [rho, u, p]``."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    rho, mom, ene = u[:, 0], u[:, 1], u[:, 2]
    out = np.empty_like(u)
    out[:, 0] = rho
    out[:, 1] = mom / rho
    out[:, 2] = (gamma - 1.0) * (ene - 0.5 * mom * mom / rho)
    return out.squeeze() if out.shape[0] == 1 else out


def specific_internal_energy(rho, p, gamma):
    """Specific internal energy ``e = p / ((gamma - 1) * rho)``.

    This is the quantity plotted in the fourth panel of every figure, and it is
    the diagnostic that most clearly exposes the contact discontinuity: ``e``
    jumps across the contact while ``u`` and ``p`` do not.
    """
    return np.asarray(p) / ((gamma - 1.0) * np.asarray(rho))
