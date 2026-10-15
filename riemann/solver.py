"""Explicit finite-volume driver: conservative update, CFL time step and
boundary conditions.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*
(3rd ed.), Chapter 6 ("The Method of Godunov for Non-linear Systems"),
Secs. 6.3 (boundary conditions) and 6.4 (the CFL time-step restriction).

The scheme is the conservative one-step update (Toro Sec. 6.2)

    U_i^{n+1} = U_i^n - (dt / dx) [ F_{i+1/2} - F_{i-1/2} ],

with ``F`` supplied by :func:`riemann.fluxes.interface_fluxes`, which
selects among the available numerical fluxes.

Bugs fixed relative to the MATLAB original (``plot_roe_eue.m``)
--------------------------------------------------------------
1. The time step used ``max |u + a|``, which is not the maximum signal speed:
   the left-going acoustic family travels at ``u - a``, so for ``u < 0`` the
   fastest wave is missed and the step is too large.  It now uses
   ``max(|u| + a)`` (Toro Sec. 6.4, where ``S_max`` is so defined).
2. The last step was not clipped to the output time, so the solution was
   reported at some ``t > T``, up to one full step late.  ``dt`` is now
   truncated so the final time is exactly ``T``.
3. A note on *not* ramping the CFL number.  Toro (Sec. 6.4) suggests reducing
   the Courant number over the first few steps of a Godunov-type run, because
   the wave speeds present in the discontinuous initial data under-predict
   those that appear immediately after.  That advice does **not** transfer to
   this flux-limited scheme, and an earlier version of this port that did ramp
   it broke Toro's Test 5.  The reason is visible in the second-order
   correction of Toro eq. (14.49), whose magnitude is proportional to
   ``1 - |nu_k|``: as ``nu -> 0`` the correction grows until it exactly cancels
   the upwind numerical diffusion, and the scheme degenerates to the
   dissipation-free central scheme.  Shrinking ``dt`` therefore makes this
   scheme *less* robust, not more.  The step is set purely by the CFL condition
   and by the output time.
4. The MATLAB loop simply never updated cells 1, 2, NX-1, NX, which works only
   as long as no wave reaches the boundary.  Proper transmissive ghost cells
   (Toro Sec. 6.3) are now applied every step, so the solver stays valid if a
   wave does exit the domain.
5. There was no cap on the number of steps and no check for NaN, so a bad
   input spun an infinite loop in silence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._jit import njit
from .fluxes import FLUX_NAMES, flux_code, interface_fluxes
from .limiters import LIMITER_NAMES, limiter_code
from .thermo import (
    conservative_to_primitive,
    primitive_to_conservative,
    specific_internal_energy,
)

__all__ = ["RoeSolution", "solve_roe", "N_GHOST"]

#: Ghost cells per side.  Two are needed, not one: the flux limiter at the
#: first interior interface reads the wave strengths of the interface one
#: further upwind.
N_GHOST = 2


@njit
def _apply_transmissive_bc(u_all, n_ghost):
    """Zero-gradient (transmissive) boundary conditions, Toro Sec. 6.3.

    Ghost values copy the nearest interior cell.  For a shock tube run to a
    time before any wave reaches the ends of the tube this is indistinguishable
    from holding the initial states fixed, but it stays well posed afterwards:
    waves leave the domain instead of reflecting off it.
    """
    n_cells = u_all.shape[0]
    for g in range(n_ghost):
        for m in range(3):
            u_all[g, m] = u_all[n_ghost, m]
            u_all[n_cells - 1 - g, m] = u_all[n_cells - 1 - n_ghost, m]


@njit
def _max_signal_speed(u_all, gamma, n_ghost):
    """``S_max = max_i (|u_i| + a_i)`` over the interior cells (Toro Sec. 6.4).

    Using ``|u| + a`` rather than the MATLAB's ``|u + a|`` is what makes this
    an upper bound on *both* acoustic families, ``u - a`` and ``u + a``.
    """
    n_cells = u_all.shape[0]
    s_max = 0.0
    for i in range(n_ghost, n_cells - n_ghost):
        rho = u_all[i, 0]
        vel = u_all[i, 1] / rho
        p = (gamma - 1.0) * (u_all[i, 2] - 0.5 * u_all[i, 1] * u_all[i, 1] / rho)
        if p <= 0.0 or rho <= 0.0:
            # Signal the caller rather than taking sqrt of a negative number.
            return -1.0
        a = np.sqrt(gamma * p / rho)
        s = abs(vel) + a
        if s > s_max:
            s_max = s
    return s_max


@njit
def _advance(
    u_all, gamma, dx, dt, n_ghost, flux_code_, limiter_code_, use_entropy_fix,
    second_order,
):
    """One conservative time step (Toro Sec. 6.2, the Godunov update), in place.

    Returns the new state array; the interior cells are updated from the
    interface fluxes and the ghost cells are refreshed afterwards.
    """
    flux = interface_fluxes(
        u_all, gamma, dx, dt, flux_code_, limiter_code_, use_entropy_fix,
        second_order,
    )
    n_cells = u_all.shape[0]
    u_new = u_all.copy()
    lam = dt / dx
    for i in range(n_ghost, n_cells - n_ghost):
        for m in range(3):
            # flux[i]   lives at interface i   == i + 1/2 for cell i
            # flux[i-1] lives at interface i-1 == i - 1/2 for cell i
            u_new[i, m] = u_all[i, m] - lam * (flux[i, m] - flux[i - 1, m])
    _apply_transmissive_bc(u_new, n_ghost)
    return u_new


def _positivity_message(fcode, steps, t):
    """Explain a positivity failure in terms of the flux that produced it.

    Which fluxes can fail, and why, differs enough between them that one
    generic message would be useless.  Keeping this out of the hot path means
    it costs nothing until something actually goes wrong.
    """
    name = FLUX_NAMES[fcode]
    lines = [
        f"flux={name!r} produced a non-physical state (negative pressure or "
        f"density) at step {steps}, t = {t:.6g}.",
    ]
    if name == "roe":
        lines.append(
            "Roe's linearisation does not preserve positivity of internal "
            "energy: near a vacuum star state it is known to fail outright "
            "(Einfeldt, Munz, Roe & Sjoegreen, J. Comput. Phys. 92:273, 1991; "
            "Toro Sec. 11.4).  Toro's Test 2 (the '123' problem) is the "
            "standard example, and fails even at first order."
        )
    elif name == "ausm":
        lines.append(
            "AUSM+ has no pressure-velocity coupling in its mass flux, so it "
            "cannot start a flow from rest: with u = 0 on both sides the "
            "interface Mach number is exactly zero whatever the pressure "
            "jump, and the internal energy goes negative immediately.  It "
            "needs a local Mach number of roughly 0.1 to function.  See "
            "riemann.fluxes.ausm_plus_flux."
        )
    elif name in ("rusanov", "lax_friedrichs"):
        lines.append(
            f"{name} is positivity-preserving as a first-order flux, but that "
            "proof does not cover the second-order TVD correction, which is "
            "the usual cause of a failure here.  Try second_order=False or "
            "limiter='minmod'."
        )
    lines.append(
        "Remedy: use a more robust flux -- flux='hllc' (recommended) or "
        "'hll' run every built-in problem.  Failing that, a more diffusive "
        "limiter (minmod) or second_order=False may help.  Note that "
        "*lowering* the CFL number is not a remedy: the second-order "
        "correction scales as 1 - |nu|, so a smaller step removes "
        "dissipation rather than adding it."
    )
    return "\n".join(lines)


@dataclass
class RoeSolution:
    """Result of a Roe-solver run on a uniform grid.

    Attributes
    ----------
    x : ndarray, shape (nx,)
        Cell-centre coordinates.
    rho, u, p, e : ndarray, shape (nx,)
        Cell-averaged density, velocity, pressure and specific internal energy.
    t : float
        Final time actually reached (equal to the requested time).
    steps : int
        Number of time steps taken.
    gamma, cfl, dx : float
        Run parameters, kept for reproducibility and for the plot labels.
    flux : str
        Name of the numerical flux used.
    limiter : str
        Name of the flux limiter used.
    second_order, entropy_fix : bool
        Which options were active.
    """

    x: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    e: np.ndarray
    t: float
    steps: int
    gamma: float
    cfl: float
    dx: float
    flux: str
    limiter: str
    second_order: bool
    entropy_fix: bool
    conserved: np.ndarray = field(repr=False, default=None)

    #: Display names for the plot legend, where the conventional
    #: capitalisation differs from the lowercase option name.
    _PRETTY = {
        "roe": "Roe",
        "hll": "HLL",
        "hllc": "HLLC",
        "rusanov": "Rusanov",
        "lax_friedrichs": "Lax-Friedrichs",
        "ausm": "AUSM+",
    }

    @property
    def label(self):
        """A short human-readable description, suitable for a plot legend."""
        name = self._PRETTY.get(self.flux, self.flux)
        if self.second_order:
            return f"{name}, 2nd order, {self.limiter}"
        return f"{name}, 1st order"


def solve_roe(
    gamma,
    w_l,
    w_r,
    t_end,
    x_left=0.0,
    x_right=1.0,
    nx=200,
    x_diaphragm=0.5,
    cfl=0.9,
    flux="roe",
    limiter="superbee",
    second_order=True,
    entropy_fix=True,
    max_steps=200_000,
):
    """Solve a Riemann problem numerically with the TVD Roe scheme.

    Parameters
    ----------
    gamma : float
        Ratio of specific heats.
    w_l, w_r : array_like, shape (3,)
        Left and right primitive states ``[rho, u, p]``.
    t_end : float
        Output time.
    x_left, x_right : float
        Domain bounds.
    nx : int
        Number of interior finite-volume cells.
    x_diaphragm : float
        Position of the initial discontinuity.  Cells are initialised from the
        left or right state according to whether their *centre* lies to the
        left or right of it, which is the cell-centred convention the exact
        sampler also uses.
    cfl : float
        Courant number.  The TVD-limited scheme is stable for ``cfl <= 1``;
        ``0.9`` is a sensible working value and ``0.75`` is comfortably safe.
    flux : str or int
        Numerical flux: ``"roe"``, ``"hll"`` (equivalently ``"hlle"``),
        ``"hllc"``, ``"rusanov"``, ``"lax_friedrichs"`` or ``"ausm"``.
        See :mod:`riemann.fluxes`.  ``"hllc"`` is the best general-purpose
        choice; ``"roe"`` is the default for backwards compatibility.
    limiter : str or int
        Flux limiter: ``"superbee"``, ``"minmod"``, ``"vanleer"``,
        ``"vanalbada"`` or ``"none"``.  Integers 0/1 map to superbee/minmod to
        match the MATLAB ``flim`` flag.
    second_order : bool
        Include the flux-limited second-order correction.  Setting this
        ``False`` gives the plain first-order Roe scheme, which is useful for
        showing how much the TVD term actually buys.
    entropy_fix : bool
        Apply the Harten-Hyman entropy fix.  Setting this ``False`` reproduces
        the entropy-violating rarefaction glitch at a sonic point.  Only the
        Roe flux uses it; the others ignore it.
    max_steps : int
        Safety cap on the number of time steps.

    Returns
    -------
    RoeSolution
    """
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if nx < 4:
        raise ValueError("nx must be at least 4")
    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0, 1]")

    code = limiter_code(limiter)
    fcode = flux_code(flux)
    gamma = float(gamma)
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    if w_l[0] <= 0 or w_l[2] <= 0 or w_r[0] <= 0 or w_r[2] <= 0:
        raise ValueError("initial states must have positive density and pressure")

    dx = (x_right - x_left) / nx
    x = x_left + dx * (np.arange(nx) + 0.5)

    # Allocate with ghost cells and set the initial condition.
    u_all = np.empty((nx + 2 * N_GHOST, 3))
    u_left = primitive_to_conservative(w_l, gamma)
    u_right = primitive_to_conservative(w_r, gamma)
    interior = slice(N_GHOST, N_GHOST + nx)
    u_all[interior] = np.where(
        (x < x_diaphragm)[:, None], u_left[None, :], u_right[None, :]
    )
    _apply_transmissive_bc(u_all, N_GHOST)

    t = 0.0
    steps = 0
    while t < t_end and steps < max_steps:
        s_max = _max_signal_speed(u_all, gamma, N_GHOST)
        if s_max <= 0.0:
            raise FloatingPointError(
                _positivity_message(fcode, steps, t)
            )
        dt = cfl * dx / s_max

        # Land exactly on the output time instead of overshooting it.
        if t + dt > t_end:
            dt = t_end - t

        u_all = _advance(
            u_all, gamma, dx, dt, N_GHOST, fcode, code, entropy_fix,
            second_order,
        )
        if not np.all(np.isfinite(u_all)):
            raise FloatingPointError(
                f"solution became non-finite at step {steps}, t = {t:.6g}"
            )
        t += dt
        steps += 1

    if t < t_end:
        # Only reachable by exhausting the step cap.
        raise RuntimeError(
            f"reached the {max_steps}-step cap at t = {t:.6g} < t_end = {t_end:.6g}"
        )

    u_int = u_all[interior]
    prim = conservative_to_primitive(u_int, gamma)
    rho, vel, p = prim[:, 0], prim[:, 1], prim[:, 2]

    return RoeSolution(
        x=x,
        rho=rho,
        u=vel,
        p=p,
        e=specific_internal_energy(rho, p, gamma),
        t=float(t),
        steps=steps,
        gamma=gamma,
        cfl=float(cfl),
        dx=float(dx),
        flux=FLUX_NAMES[fcode],
        limiter=LIMITER_NAMES[code],
        second_order=bool(second_order),
        entropy_fix=bool(entropy_fix),
        conserved=u_int.copy(),
    )
