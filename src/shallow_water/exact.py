"""Exact Riemann solver for the shallow-water equations.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Chapter 5.

Structure of the solution
-------------------------
The Riemann problem has two waves and one constant middle state.  Each wave
is a bore (shock) or a rarefaction, and across the middle state both depth
and velocity are constant -- there is no contact discontinuity to separate
``h`` from ``u``, as there is in a gas.  The whole solution is therefore
pinned down by the single scalar ``h_star`` solving

    f(h) = f_L(h, h_L) + f_R(h, h_R) + (u_R - u_L) = 0,

with the one-sided depth functions

    f_K(h) = 2 (sqrt(g h) - sqrt(g h_K))                      h <= h_K  (rarefaction)
           = (h - h_K) sqrt( (g/2) (h + h_K) / (h h_K) )      h >  h_K  (bore)

This is the same construction as the pressure function of the Euler exact
solver, with depth in place of pressure; ``f`` is again monotone increasing
and concave, so Newton converges from any positive start.

Dry bed
-------
The shallow-water analogue of vacuum is a dry bed, ``h = 0``, and unlike
vacuum in a gas it is an ordinary case: a dam break onto dry ground is the
canonical test problem.  Three situations arise, and all three are solved
here rather than refused:

* ``h_R = 0`` -- a left rarefaction running out onto dry bed, with the wet/dry
  front at ``u_L + 2 a_L``;
* ``h_L = 0`` -- the mirror image, front at ``u_R - 2 a_R``;
* **dry-bed generation**, when the two sides separate fast enough that
  ``(u_R - u_L) >= 2 (a_L + a_R)``: both rarefactions run out and a dry
  region opens between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.jit import njit
from .physics import DRY_TOLERANCE, GRAVITY, check_gravity

__all__ = ["ShallowWaterSolution", "depth_function", "solve_star_depth", "solve_exact"]


@njit
def _f_single(h, h_k, gravity):
    """One-sided depth function ``f_K(h)``.

    Rarefaction branch from the Riemann invariant ``u +- 2 sqrt(g h)``;
    bore branch from the Rankine-Hugoniot conditions for the system.
    """
    if h <= h_k:  # rarefaction
        return 2.0 * (np.sqrt(gravity * h) - np.sqrt(gravity * h_k))
    # bore
    return (h - h_k) * np.sqrt(0.5 * gravity * (h + h_k) / (h * h_k))


@njit
def _fprime_single(h, h_k, gravity):
    """Derivative ``df_K/dh``.

    Rarefaction: ``sqrt(g / h)``.  Bore: differentiating the square root
    above, with ``Q = (g/2)(h + h_K)/(h h_K)`` and ``dQ/dh = -g/(2 h^2)``,

        f' = sqrt(Q) - (h - h_K) g / (4 h^2 sqrt(Q)).
    """
    if h <= h_k:
        return np.sqrt(gravity / h)
    q = 0.5 * gravity * (h + h_k) / (h * h_k)
    return np.sqrt(q) - (h - h_k) * gravity / (4.0 * h * h * np.sqrt(q))


@njit
def depth_function(h, w_l, w_r, gravity):
    """``f(h) = f_L + f_R + (u_R - u_L)``; its root is the star-region depth."""
    return (
        _f_single(h, w_l[0], gravity)
        + _f_single(h, w_r[0], gravity)
        + (w_r[1] - w_l[1])
    )


@njit
def _depth_function_derivative(h, w_l, w_r, gravity):
    return _fprime_single(h, w_l[0], gravity) + _fprime_single(h, w_r[0], gravity)


@njit
def _two_rarefaction_guess(w_l, w_r, gravity):
    """Starting depth from the two-rarefaction approximation.

    Exact when both waves really are rarefactions, which makes it a good
    guess everywhere and the exact answer for the common dam-break case.
    Equating the two Riemann invariants across the star region gives

        h* = (0.5 (a_L + a_R) + 0.25 (u_L - u_R))^2 / g.
    """
    a_l = np.sqrt(gravity * w_l[0])
    a_r = np.sqrt(gravity * w_r[0])
    root = 0.5 * (a_l + a_r) + 0.25 * (w_l[1] - w_r[1])
    return root * root / gravity


def is_dry_bed(w_l, w_r, gravity=GRAVITY):
    """Does this data have, or generate, a dry region?

    True when either side starts dry, or when the two sides separate faster
    than the rarefactions can fill the gap,
    ``u_R - u_L >= 2 (a_L + a_R)`` -- the shallow-water counterpart of the
    vacuum condition for a gas.
    """
    gravity = check_gravity(gravity)
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    if w_l[0] <= DRY_TOLERANCE or w_r[0] <= DRY_TOLERANCE:
        return True
    a_l = np.sqrt(gravity * w_l[0])
    a_r = np.sqrt(gravity * w_r[0])
    return (w_r[1] - w_l[1]) >= 2.0 * (a_l + a_r)


def solve_star_depth(w_l, w_r, gravity=GRAVITY, tol=1.0e-12, max_iter=100):
    """Newton-Raphson solve of ``f(h_star) = 0``.

    Returns ``(h_star, u_star, iterations)``.  Raises if the data are dry or
    dry-generating, since then no star region exists; callers should test
    with :func:`is_dry_bed` first, as :func:`solve_exact` does.
    """
    gravity = check_gravity(gravity)
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    if is_dry_bed(w_l, w_r, gravity):
        raise ValueError(
            "initial data are dry or generate a dry bed, so there is no star "
            "region; use solve_exact, which handles the dry cases"
        )

    h = float(_two_rarefaction_guess(w_l, w_r, gravity))
    h = max(h, DRY_TOLERANCE)
    for it in range(1, max_iter + 1):
        f = depth_function(h, w_l, w_r, gravity)
        df = _depth_function_derivative(h, w_l, w_r, gravity)
        h_new = h - f / df
        if h_new <= 0.0:
            h_new = 0.5 * h
        change = 2.0 * abs(h_new - h) / (h_new + h)
        h = h_new
        if change < tol:
            u_star = 0.5 * (w_l[1] + w_r[1]) + 0.5 * (
                _f_single(h, w_r[0], gravity) - _f_single(h, w_l[0], gravity)
            )
            return h, float(u_star), it

    raise ValueError(
        f"Newton-Raphson for h_star did not converge in {max_iter} iterations"
    )


@njit
def _sample_wet(s, w_l, w_r, h_star, u_star, gravity, out):
    """Sample the wet-bed solution at similarity coordinate ``s = x / t``.

    Partitions the whole ``s`` axis: left of the middle state or right of it,
    then by whether that side's wave is a bore (one speed to test) or a
    rarefaction (head and tail speeds bracketing a fan whose interior the
    Riemann invariants give in closed form).
    """
    h_l, u_l = w_l[0], w_l[1]
    h_r, u_r = w_r[0], w_r[1]
    a_l = np.sqrt(gravity * h_l)
    a_r = np.sqrt(gravity * h_r)
    a_star = np.sqrt(gravity * h_star)

    if s <= u_star:
        # ------------------------------------------------ left of the middle
        if h_star > h_l:
            # Left bore.  Speed from Rankine-Hugoniot, written with the
            # standard factor q_L = sqrt( (h* + h_L) h* / (2 h_L^2) ).
            q_l = np.sqrt(0.5 * (h_star + h_l) * h_star / (h_l * h_l))
            s_l = u_l - a_l * q_l
            if s <= s_l:
                out[0] = h_l
                out[1] = u_l
            else:
                out[0] = h_star
                out[1] = u_star
        else:
            # Left rarefaction.
            s_head = u_l - a_l
            s_tail = u_star - a_star
            if s <= s_head:
                out[0] = h_l
                out[1] = u_l
            elif s >= s_tail:
                out[0] = h_star
                out[1] = u_star
            else:
                # Inside the fan: u + 2a is constant along it.
                u_fan = (u_l + 2.0 * a_l + 2.0 * s) / 3.0
                a_fan = (u_l + 2.0 * a_l - s) / 3.0
                out[0] = a_fan * a_fan / gravity
                out[1] = u_fan
    else:
        # ----------------------------------------------- right of the middle
        if h_star > h_r:
            q_r = np.sqrt(0.5 * (h_star + h_r) * h_star / (h_r * h_r))
            s_r = u_r + a_r * q_r
            if s >= s_r:
                out[0] = h_r
                out[1] = u_r
            else:
                out[0] = h_star
                out[1] = u_star
        else:
            s_head = u_r + a_r
            s_tail = u_star + a_star
            if s >= s_head:
                out[0] = h_r
                out[1] = u_r
            elif s <= s_tail:
                out[0] = h_star
                out[1] = u_star
            else:
                u_fan = (u_r - 2.0 * a_r + 2.0 * s) / 3.0
                a_fan = (-u_r + 2.0 * a_r + s) / 3.0
                out[0] = a_fan * a_fan / gravity
                out[1] = u_fan


@njit
def _sample_dry(s, w_l, w_r, gravity, out):
    """Sample a solution containing a dry region.

    Covers all three dry cases with one partition.  Where a side is wet its
    rarefaction runs out to a wet/dry front at ``u +- 2a``; beyond that front
    the bed is dry, reported as ``h = 0`` with zero velocity.
    """
    h_l, u_l = w_l[0], w_l[1]
    h_r, u_r = w_r[0], w_r[1]
    left_wet = h_l > DRY_TOLERANCE
    right_wet = h_r > DRY_TOLERANCE

    if left_wet:
        a_l = np.sqrt(gravity * h_l)
        s_head_l = u_l - a_l
        s_front_l = u_l + 2.0 * a_l   # wet/dry front of the left fan
    else:
        a_l = 0.0
        s_head_l = 0.0
        s_front_l = 0.0
    if right_wet:
        a_r = np.sqrt(gravity * h_r)
        s_head_r = u_r + a_r
        s_front_r = u_r - 2.0 * a_r   # wet/dry front of the right fan
    else:
        a_r = 0.0
        s_head_r = 0.0
        s_front_r = 0.0

    if left_wet and s <= s_head_l:
        out[0] = h_l
        out[1] = u_l
        return
    if right_wet and s >= s_head_r:
        out[0] = h_r
        out[1] = u_r
        return
    if left_wet and s < s_front_l:
        u_fan = (u_l + 2.0 * a_l + 2.0 * s) / 3.0
        a_fan = (u_l + 2.0 * a_l - s) / 3.0
        out[0] = a_fan * a_fan / gravity
        out[1] = u_fan
        return
    if right_wet and s > s_front_r:
        u_fan = (u_r - 2.0 * a_r + 2.0 * s) / 3.0
        a_fan = (-u_r + 2.0 * a_r + s) / 3.0
        out[0] = a_fan * a_fan / gravity
        out[1] = u_fan
        return
    out[0] = 0.0
    out[1] = 0.0


@njit
def _sample_many(s_vals, w_l, w_r, h_star, u_star, gravity, dry, out):
    for i in range(s_vals.shape[0]):
        if dry:
            _sample_dry(s_vals[i], w_l, w_r, gravity, out[i])
        else:
            _sample_wet(s_vals[i], w_l, w_r, h_star, u_star, gravity, out[i])


@dataclass
class ShallowWaterSolution:
    """Exact solution of one shallow-water Riemann problem, sampled on a grid.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Cell-centre coordinates.
    h, u : ndarray, shape (n,)
        Depth and depth-averaged velocity.
    h_star, u_star : float
        Star-region depth and velocity; both ``nan`` for a dry solution,
        which has no star region.
    gravity, t : float
    left_wave, right_wave : str
        ``"bore"``, ``"rarefaction"`` or ``"dry"``.
    iterations : int
        Newton iterations for ``h_star``; 0 for a dry solution.
    dry : bool
        Whether the solution contains a dry region.
    """

    x: np.ndarray
    h: np.ndarray
    u: np.ndarray
    h_star: float
    u_star: float
    gravity: float
    t: float
    left_wave: str
    right_wave: str
    iterations: int
    dry: bool

    @property
    def discharge(self):
        """Depth-integrated discharge ``h u``, the second conserved variable."""
        return self.h * self.u

    @property
    def froude(self):
        """Froude number ``|u| / sqrt(g h)``; zero where the bed is dry."""
        a = np.sqrt(self.gravity * np.maximum(self.h, DRY_TOLERANCE))
        return np.where(self.h > DRY_TOLERANCE, np.abs(self.u) / a, 0.0)


def solve_exact(
    w_l,
    w_r,
    t,
    gravity=GRAVITY,
    x_left=0.0,
    x_right=1.0,
    n=1000,
    x_diaphragm=0.5,
    tol=1.0e-12,
):
    """Solve and sample the exact shallow-water Riemann problem.

    Parameters
    ----------
    w_l, w_r : array_like, shape (2,)
        Left and right primitive states ``[h, u]``.  Either depth may be
        zero, for a dry bed.
    t : float
        Output time, positive.
    gravity : float
        Acceleration due to gravity.
    x_left, x_right, n, x_diaphragm
        Sampling grid and the position of the initial discontinuity.
    tol : float
        Relative tolerance for the Newton iteration.

    Returns
    -------
    ShallowWaterSolution
    """
    gravity = check_gravity(gravity)
    if t <= 0.0:
        raise ValueError("output time t must be positive")
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    if w_l[0] < 0.0 or w_r[0] < 0.0:
        raise ValueError("depths must be non-negative")
    if w_l[0] <= DRY_TOLERANCE and w_r[0] <= DRY_TOLERANCE:
        raise ValueError("both sides are dry; there is nothing to solve")

    dry = is_dry_bed(w_l, w_r, gravity)
    if dry:
        h_star, u_star, iterations = np.nan, np.nan, 0
        left_wave = "rarefaction" if w_l[0] > DRY_TOLERANCE else "dry"
        right_wave = "rarefaction" if w_r[0] > DRY_TOLERANCE else "dry"
    else:
        h_star, u_star, iterations = solve_star_depth(w_l, w_r, gravity, tol=tol)
        left_wave = "bore" if h_star > w_l[0] else "rarefaction"
        right_wave = "bore" if h_star > w_r[0] else "rarefaction"

    dx = (x_right - x_left) / n
    x = x_left + dx * (np.arange(n) + 0.5)
    s_vals = (x - x_diaphragm) / t

    out = np.empty((n, 2))
    _sample_many(
        s_vals, w_l, w_r,
        0.0 if dry else h_star, 0.0 if dry else u_star,
        gravity, dry, out,
    )

    return ShallowWaterSolution(
        x=x,
        h=out[:, 0],
        u=out[:, 1],
        h_star=float(h_star),
        u_star=float(u_star),
        gravity=float(gravity),
        t=float(t),
        left_wave=left_wave,
        right_wave=right_wave,
        iterations=int(iterations),
        dry=bool(dry),
    )
