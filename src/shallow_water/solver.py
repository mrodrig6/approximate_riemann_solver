"""Explicit finite-volume driver for the shallow-water equations.

The same Godunov-type update as the Euler driver:

    U_i^{n+1} = U_i^n - (dt / dx) [ F_{i+1/2} - F_{i-1/2} ],

with transmissive ghost cells and a CFL-limited step.  It is a separate
driver rather than a shared one because the state vector has two components
instead of three and the maximum signal speed is defined differently; under
Numba a shared version would have to branch on the system inside every
kernel, which costs more in clarity than the duplicated loop does.

Two things the Euler driver does not have to deal with:

* **Dry cells.**  A cell may legitimately have zero depth, so the maximum
  signal speed skips dry cells rather than dividing by their depth, and the
  update clips any depth that round-off drives slightly negative.
* **A depth-based CFL condition**, ``S_max = max(|u| + sqrt(g h))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.jit import njit
from ..core.limiters import LIMITER_NAMES, limiter_code
from .fluxes import FLUX_NAMES, flux_code, interface_fluxes
from .physics import (
    DRY_TOLERANCE,
    GRAVITY,
    conservative_to_primitive,
    primitive_to_conservative,
)

__all__ = ["ShallowWaterResult", "solve", "solve_field", "N_GHOST"]

#: Ghost cells per side; two, as for the Euler driver, because the limiter
#: reads one interface beyond the first interior one.
N_GHOST = 2


@njit
def _apply_transmissive_bc(u_all, n_ghost):
    """Zero-gradient boundary conditions."""
    n_cells = u_all.shape[0]
    for g in range(n_ghost):
        for m in range(2):
            u_all[g, m] = u_all[n_ghost, m]
            u_all[n_cells - 1 - g, m] = u_all[n_cells - 1 - n_ghost, m]


@njit
def _max_signal_speed(u_all, gravity, n_ghost):
    """``S_max = max(|u| + sqrt(g h))`` over the wet interior cells.

    Dry cells are skipped rather than contributing a division by zero.
    Returns ``-1.0`` for a negative depth, which the caller treats as a
    failure, and ``0.0`` if every interior cell is dry.
    """
    n_cells = u_all.shape[0]
    s_max = 0.0
    for i in range(n_ghost, n_cells - n_ghost):
        h = u_all[i, 0]
        if h < 0.0:
            return -1.0
        if h <= DRY_TOLERANCE:
            continue
        vel = u_all[i, 1] / h
        s = abs(vel) + np.sqrt(gravity * h)
        if s > s_max:
            s_max = s
    return s_max


@njit
def _advance(
    u_all, gravity, dx, dt, n_ghost, flux_code_, limiter_code_, use_entropy_fix,
    second_order,
):
    """One conservative time step, returning the new state array."""
    flux = interface_fluxes(
        u_all, gravity, dx, dt, flux_code_, limiter_code_, use_entropy_fix,
        second_order,
    )
    n_cells = u_all.shape[0]
    u_new = u_all.copy()
    lam = dt / dx
    for i in range(n_ghost, n_cells - n_ghost):
        for m in range(2):
            u_new[i, m] = u_all[i, m] - lam * (flux[i, m] - flux[i - 1, m])
        # A wetting front can leave a depth a hair below zero; clip it and
        # drop the discharge with it, rather than letting a negative depth
        # reach sqrt().
        if u_new[i, 0] <= DRY_TOLERANCE:
            u_new[i, 0] = 0.0
            u_new[i, 1] = 0.0
    _apply_transmissive_bc(u_new, n_ghost)
    return u_new


@dataclass
class ShallowWaterResult:
    """Result of a shallow-water run on a uniform grid.

    Attributes
    ----------
    x : ndarray, shape (nx,)
        Cell centres.
    h, u : ndarray, shape (nx,)
        Depth and depth-averaged velocity.
    t : float
        Final time, equal to the requested one.
    steps : int
        Time steps taken.
    gravity, cfl, dx : float
    flux, limiter : str
    second_order, entropy_fix : bool
    """

    x: np.ndarray
    h: np.ndarray
    u: np.ndarray
    t: float
    steps: int
    gravity: float
    cfl: float
    dx: float
    flux: str
    limiter: str
    second_order: bool
    entropy_fix: bool
    conserved: np.ndarray = field(repr=False, default=None)

    _PRETTY = {
        "roe": "Roe",
        "hll": "HLL",
        "rusanov": "Rusanov",
        "lax_friedrichs": "Lax-Friedrichs",
    }

    @property
    def discharge(self):
        """Depth-integrated discharge ``h u``."""
        return self.h * self.u

    @property
    def label(self):
        """A short description, suitable for a plot legend."""
        name = self._PRETTY.get(self.flux, self.flux)
        if self.second_order:
            return f"{name}, 2nd order, {self.limiter}"
        return f"{name}, 1st order"


def solve(
    w_l,
    w_r,
    t_end,
    gravity=GRAVITY,
    x_left=0.0,
    x_right=1.0,
    nx=200,
    x_diaphragm=0.5,
    cfl=0.9,
    flux="hll",
    limiter="superbee",
    second_order=True,
    entropy_fix=True,
    max_steps=200_000,
):
    """Solve a shallow-water Riemann problem numerically.

    Parameters
    ----------
    w_l, w_r : array_like, shape (2,)
        Left and right primitive states ``[h, u]``.  Either depth may be
        zero, for a dam break onto a dry bed.
    t_end : float
        Output time.
    gravity : float
        Acceleration due to gravity.
    x_left, x_right, nx, x_diaphragm
        Grid and the position of the initial discontinuity.
    cfl : float
        Courant number in (0, 1].  Use 0.45 or below for a dry-bed problem:
        the wetting front advances by more than the usual estimate suggests,
        and the standard bound is not conservative there.
    flux : str or int
        ``"hll"`` (default; ``"hllc"`` is an accepted alias for it),
        ``"roe"``, ``"rusanov"`` or ``"lax_friedrichs"``.
    limiter : str or int
        Limiter for the second-order correction.
    second_order, entropy_fix : bool
    max_steps : int
        Safety cap on the number of time steps.

    Returns
    -------
    ShallowWaterResult
    """
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    if w_l[0] < 0.0 or w_r[0] < 0.0:
        raise ValueError("depths must be non-negative")
    if nx < 4:
        raise ValueError("nx must be at least 4")

    dx = (x_right - x_left) / nx
    x = x_left + dx * (np.arange(nx) + 0.5)
    w_init = np.where((x < x_diaphragm)[:, None], w_l[None, :], w_r[None, :])

    return solve_field(
        w_init, t_end, gravity=gravity, x_left=x_left, x_right=x_right,
        cfl=cfl, flux=flux, limiter=limiter, second_order=second_order,
        entropy_fix=entropy_fix, max_steps=max_steps,
    )


def solve_field(
    w_init,
    t_end,
    gravity=GRAVITY,
    x_left=0.0,
    x_right=1.0,
    cfl=0.9,
    flux="hll",
    limiter="superbee",
    second_order=True,
    entropy_fix=True,
    max_steps=200_000,
):
    """Advance an arbitrary initial field of primitive states ``[h, u]``.

    :func:`solve` is a thin wrapper over this; working with a field directly
    allows smooth initial data, and hence an order-of-accuracy study.
    """
    w_init = np.atleast_2d(np.asarray(w_init, dtype=float))
    if w_init.ndim != 2 or w_init.shape[1] != 2:
        raise ValueError("w_init must have shape (nx, 2)")
    nx = w_init.shape[0]
    if nx < 4:
        raise ValueError("nx must be at least 4")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0, 1]")
    if np.any(w_init[:, 0] < 0.0):
        raise ValueError("initial field must have non-negative depth")

    code = limiter_code(limiter)
    fcode = flux_code(flux)
    gravity = float(gravity)

    dx = (x_right - x_left) / nx
    x = x_left + dx * (np.arange(nx) + 0.5)

    u_all = np.empty((nx + 2 * N_GHOST, 2))
    interior = slice(N_GHOST, N_GHOST + nx)
    u_all[interior] = primitive_to_conservative(w_init, gravity)
    _apply_transmissive_bc(u_all, N_GHOST)

    t = 0.0
    steps = 0
    while t < t_end and steps < max_steps:
        s_max = _max_signal_speed(u_all, gravity, N_GHOST)
        if s_max < 0.0:
            raise FloatingPointError(
                f"flux={FLUX_NAMES[fcode]!r} produced a negative depth at step "
                f"{steps}, t = {t:.6g}.  On a dry-bed problem, reduce the CFL "
                "number to about 0.45 or use second_order=False; the usual "
                "CFL bound is not conservative at a wetting front."
            )
        if s_max == 0.0:
            # Everything is dry and at rest; nothing can happen.
            t = t_end
            break
        dt = cfl * dx / s_max
        if t + dt > t_end:
            dt = t_end - t

        u_all = _advance(
            u_all, gravity, dx, dt, N_GHOST, fcode, code, entropy_fix,
            second_order,
        )
        if not np.all(np.isfinite(u_all)):
            raise FloatingPointError(
                f"solution became non-finite at step {steps}, t = {t:.6g}"
            )
        t += dt
        steps += 1

    if t < t_end:
        raise RuntimeError(
            f"reached the {max_steps}-step cap at t = {t:.6g} < t_end = {t_end:.6g}"
        )

    u_int = u_all[interior]
    prim = conservative_to_primitive(u_int, gravity)

    return ShallowWaterResult(
        x=x,
        h=prim[:, 0],
        u=prim[:, 1],
        t=float(t),
        steps=steps,
        gravity=gravity,
        cfl=float(cfl),
        dx=float(dx),
        flux=FLUX_NAMES[fcode],
        limiter=LIMITER_NAMES[code],
        second_order=bool(second_order),
        entropy_fix=bool(entropy_fix),
        conserved=u_int.copy(),
    )
