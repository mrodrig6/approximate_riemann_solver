"""Explicit finite-volume driver: conservative update, CFL time step and
boundary conditions.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*
(3rd ed.), Chapter 6 ("The Method of Godunov for Non-linear Systems"),
Secs. 6.3 (boundary conditions) and 6.4 (the CFL time-step restriction).

The scheme is the conservative one-step update (Toro Sec. 6.2)

    U_i^{n+1} = U_i^n - (dt / dx) [ F_{i+1/2} - F_{i-1/2} ],

with ``F`` supplied by :func:`src.euler.fluxes.interface_fluxes`, which
selects among the available numerical fluxes.

Notes on the time step
----------------------
The step is set purely by the CFL condition and by the output time; it is
truncated so the run lands exactly on the requested time rather than
overshooting it.

Do **not** reduce the Courant number hoping for robustness.  Toro suggests
that for a first-order Godunov scheme, but it does not transfer to a
flux-limited one: the second-order correction of Toro eq. (14.49) has
magnitude proportional to ``1 - |nu_k|``, so as ``nu -> 0`` it grows until it
exactly cancels the upwind numerical diffusion and the scheme degenerates to
the dissipation-free central scheme.  A smaller step therefore makes this
scheme *less* robust.  If a run is unstable, change the limiter or the flux.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.jit import njit
from .ader import ader2_fluxes
from .fluxes import FLUX_NAMES, flux_code, interface_fluxes
from ..core.limiters import LIMITER_NAMES, limiter_code
from .physics import (
    conservative_to_primitive,
    primitive_to_conservative,
    specific_internal_energy,
)

__all__ = ["RoeSolution", "solve_roe", "solve_field", "N_GHOST"]

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

    Note ``|u| + a``, not ``|u + a|``: only the former bounds *both* acoustic
    families, ``u - a`` and ``u + a``, when ``u < 0``.
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
    second_order, use_ader,
):
    """One conservative time step (Toro Sec. 6.2, the Godunov update), in place.

    Returns the new state array; the interior cells are updated from the
    interface fluxes and the ghost cells are refreshed afterwards.
    """
    if use_ader:
        flux = ader2_fluxes(u_all, gamma, dx, dt, limiter_code_)
    else:
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


def _positivity_message(fcode, steps, t, scheme="tvd"):
    """Explain a positivity failure in terms of the flux that produced it.

    Which fluxes can fail, and why, differs enough between them that one
    generic message would be useless.  Keeping this out of the hot path means
    it costs nothing until something actually goes wrong.
    """
    name = "ader2" if scheme == "ader2" else FLUX_NAMES[fcode]
    lines = [
        f"{name} produced a non-physical state (negative pressure or "
        f"density) at step {steps}, t = {t:.6g}.",
    ]
    if scheme == "ader2":
        lines.append(
            "ADER2 reconstructs a linear polynomial in each cell; on a very "
            "strong jump the reconstruction itself can go non-physical.  Try "
            "limiter='minmod', a smaller CFL number, or scheme='tvd'."
        )
    if name == "roe":
        lines.append(
            "Roe's linearisation does not preserve positivity of internal "
            "energy: near a vacuum star state it is known to fail outright "
            "(Einfeldt, Munz, Roe & Sjoegreen, J. Comput. Phys. 92:273, 1991; "
            "Toro Sec. 11.4).  Toro's Test 2 (the '123' problem) is the "
            "standard example, and fails even at first order."
        )
    elif name in ("ausmup", "rusanov", "lax_friedrichs"):
        lines.append(
            f"The {name} flux runs every built-in problem at first order; "
            "failures at second order come from the characteristic-based TVD "
            "correction, not from the flux.  Try second_order=False or "
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
    scheme : str
        ``"tvd"`` or ``"ader2"``.
    flux : str
        Name of the numerical flux used (``"ader2"`` for the ADER scheme,
        which uses no separate numerical flux).
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
    scheme: str
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
        "ausmup": "AUSM+-up",
        "ader2": "ADER2",
    }

    @property
    def label(self):
        """A short human-readable description, suitable for a plot legend."""
        if self.scheme == "ader2":
            return f"ADER2, {self.limiter}"
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
    scheme="tvd",
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
        See :mod:`src.euler.fluxes`.  ``"hllc"`` is the best general-purpose
        choice; ``"roe"`` is the default for backwards compatibility.
    limiter : str or int
        Limiter: ``"superbee"``, ``"minmod"``, ``"vanleer"``, ``"vanalbada"``
        or ``"none"``.  It limits the antidiffusive flux correction under
        ``scheme="tvd"`` and the reconstruction slope under
        ``scheme="ader2"``.
    scheme : str
        ``"tvd"`` (default) for the flux-limited second-order scheme, or
        ``"ader2"`` for the one-step ADER scheme of :mod:`src.euler.ader`.
        ADER supplies its own interface state, so ``flux``, ``entropy_fix``
        and ``second_order`` do not apply to it.
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
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    if w_l[0] <= 0 or w_l[2] <= 0 or w_r[0] <= 0 or w_r[2] <= 0:
        raise ValueError("initial states must have positive density and pressure")
    if nx < 4:
        raise ValueError("nx must be at least 4")

    dx = (x_right - x_left) / nx
    x = x_left + dx * (np.arange(nx) + 0.5)
    w_init = np.where((x < x_diaphragm)[:, None], w_l[None, :], w_r[None, :])

    return solve_field(
        gamma, w_init, t_end,
        x_left=x_left, x_right=x_right, cfl=cfl, flux=flux, limiter=limiter,
        scheme=scheme, second_order=second_order, entropy_fix=entropy_fix,
        max_steps=max_steps,
    )


def solve_field(
    gamma,
    w_init,
    t_end,
    x_left=0.0,
    x_right=1.0,
    cfl=0.9,
    flux="roe",
    limiter="superbee",
    scheme="tvd",
    second_order=True,
    entropy_fix=True,
    max_steps=200_000,
):
    """Advance an arbitrary initial field, not just two constant states.

    :func:`solve_roe` is a thin wrapper around this: it builds the
    two-state field and hands it over.  Working with a field directly is what
    makes an order-of-accuracy study possible, since measuring the formal
    order of a scheme needs *smooth* initial data -- on a discontinuity every
    scheme is first order in :math:`L_1` whatever its formal order.

    Parameters
    ----------
    gamma : float
        Ratio of specific heats.
    w_init : array_like, shape (nx, 3)
        Primitive states ``[rho, u, p]`` at the cell centres of a uniform
        grid spanning ``[x_left, x_right]``.  ``nx`` is taken from its shape.
    t_end, x_left, x_right, cfl, flux, limiter, scheme, second_order,
    entropy_fix, max_steps
        As for :func:`solve_roe`.

    Returns
    -------
    RoeSolution
    """
    w_init = np.atleast_2d(np.asarray(w_init, dtype=float))
    if w_init.ndim != 2 or w_init.shape[1] != 3:
        raise ValueError("w_init must have shape (nx, 3)")
    nx = w_init.shape[0]
    if nx < 4:
        raise ValueError("nx must be at least 4")
    if t_end <= 0.0:
        raise ValueError("t_end must be positive")
    if not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must lie in (0, 1]")
    if np.any(w_init[:, 0] <= 0.0) or np.any(w_init[:, 2] <= 0.0):
        raise ValueError("initial field must have positive density and pressure")
    scheme = str(scheme).strip().lower()
    if scheme not in ("tvd", "ader2", "ader"):
        raise ValueError(f"unknown scheme {scheme!r}; choose 'tvd' or 'ader2'")
    use_ader = scheme in ("ader2", "ader")

    code = limiter_code(limiter)
    fcode = flux_code(flux)
    gamma = float(gamma)

    dx = (x_right - x_left) / nx
    x = x_left + dx * (np.arange(nx) + 0.5)

    # Allocate with ghost cells and set the initial condition.
    u_all = np.empty((nx + 2 * N_GHOST, 3))
    interior = slice(N_GHOST, N_GHOST + nx)
    u_all[interior] = primitive_to_conservative(w_init, gamma)
    _apply_transmissive_bc(u_all, N_GHOST)

    t = 0.0
    steps = 0
    while t < t_end and steps < max_steps:
        s_max = _max_signal_speed(u_all, gamma, N_GHOST)
        if s_max <= 0.0:
            raise FloatingPointError(
                _positivity_message(
                    fcode, steps, t, scheme="ader2" if use_ader else "tvd"
                )
            )
        dt = cfl * dx / s_max

        # Land exactly on the output time instead of overshooting it.
        if t + dt > t_end:
            dt = t_end - t

        u_all = _advance(
            u_all, gamma, dx, dt, N_GHOST, fcode, code, entropy_fix,
            second_order, use_ader,
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
        scheme="ader2" if use_ader else "tvd",
        flux="ader2" if use_ader else FLUX_NAMES[fcode],
        limiter=LIMITER_NAMES[code],
        second_order=bool(second_order),
        entropy_fix=bool(entropy_fix),
        conserved=u_int.copy(),
    )
