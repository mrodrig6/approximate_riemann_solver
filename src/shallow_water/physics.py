"""Variables, closure and physical flux for the shallow-water equations.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Chapters 2 and 5; the numerical methods follow the same
Godunov-type development as the Euler case.

The equations
-------------
The one-dimensional shallow-water equations over a flat, frictionless bed
express conservation of mass and momentum for a layer of fluid whose depth is
small compared with the wavelength of the motion:

    dU/dt + dF(U)/dx = 0,
    U = [h, h*u],
    F = [h*u, h*u^2 + 0.5*g*h^2].

``h`` is the water depth, ``u`` the depth-averaged velocity and ``g`` the
acceleration due to gravity.  There is no equation of state to supply and no
energy equation: the ``0.5*g*h^2`` term is the depth-integrated hydrostatic
pressure, and the system closes itself.

Why the same machinery works
----------------------------
The Jacobian has eigenvalues

    lambda_1 = u - a,   lambda_2 = u + a,    a = sqrt(g*h),

with ``a`` the celerity of a small gravity wave.  Both fields are genuinely
non-linear, so the solution of a Riemann problem is a shock ("bore") or a
rarefaction on each side of a constant middle state.

Two differences from the Euler system are worth holding on to, because they
explain most of what differs downstream:

* **Two waves, not three.**  There is no linearly degenerate field, hence no
  contact discontinuity.  That is why HLLC is not offered here: with no
  middle wave to restore, it reduces exactly to HLL.
* **The analogue of vacuum is a dry bed**, ``h = 0``, and unlike vacuum in a
  gas it is a routine engineering case -- a dam break onto dry ground.  The
  exact solver handles it rather than refusing it.

Throughout, ``W = [h, u]`` denotes the *primitive* state and ``U = [h, h*u]``
the *conservative* one.

The Roe average and the eigenstructure built from it live in
:mod:`src.shallow_water.fluxes.roe`, beside the flux that needs them, as they
do for the Euler system.  This module is the closure and nothing more.
"""

from __future__ import annotations

import numpy as np

from ..core.jit import njit

__all__ = [
    "GRAVITY",
    "celerity",
    "conservative_to_primitive",
    "primitive_to_conservative",
    "swe_flux",
]

#: Standard gravity, m/s^2.  The default everywhere in this subpackage.
GRAVITY = 9.81

#: Depth below which a cell is treated as dry.  Not a tuning knob so much as
#: a statement that a depth of a micron is not a wet bed; the wave speeds
#: ``u = (h*u)/h`` are meaningless below it.
DRY_TOLERANCE = 1.0e-10


@njit
def celerity(h, gravity):
    """Gravity-wave celerity ``a = sqrt(g h)``.

    The shallow-water counterpart of the sound speed: the speed at which an
    infinitesimal disturbance of the free surface travels relative to the
    fluid.  The Froude number ``u / a`` plays the role of the Mach number,
    with subcritical and supercritical flow corresponding to subsonic and
    supersonic.
    """
    return np.sqrt(gravity * h)


@njit
def swe_flux(u_cell, gravity, out):
    """Physical flux ``F(U) = [h u, h u^2 + g h^2 / 2]``, written into ``out``."""
    h = u_cell[0]
    hu = u_cell[1]
    if h <= DRY_TOLERANCE:
        out[0] = 0.0
        out[1] = 0.5 * gravity * h * h
        return
    out[0] = hu
    out[1] = hu * hu / h + 0.5 * gravity * h * h


def primitive_to_conservative(w, gravity=GRAVITY):
    """Vectorised ``[h, u] -> [h, h u]``."""
    w = np.atleast_2d(np.asarray(w, dtype=float))
    out = np.empty_like(w)
    out[:, 0] = w[:, 0]
    out[:, 1] = w[:, 0] * w[:, 1]
    return out.squeeze() if out.shape[0] == 1 else out


def conservative_to_primitive(u, gravity=GRAVITY):
    """Vectorised ``[h, h u] -> [h, u]``.

    A dry cell has no defined velocity; it is reported as zero rather than
    as a division by zero.
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    out = np.empty_like(u)
    out[:, 0] = u[:, 0]
    out[:, 1] = np.where(u[:, 0] > DRY_TOLERANCE, u[:, 1] / np.maximum(u[:, 0], DRY_TOLERANCE), 0.0)
    return out.squeeze() if out.shape[0] == 1 else out
