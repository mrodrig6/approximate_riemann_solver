"""Exact Riemann solver for the 1-D Euler equations with an ideal gas.

The algorithm follows Toro, *Riemann Solvers and Numerical Methods for Fluid
Dynamics* (3rd ed.), Chapter 4 ("The Riemann Problem for the Euler
Equations"), with the initial-guess recipe of Chapter 9, Sec. 9.5.

Structure of the solution
-------------------------
The Riemann problem

    U(x, 0) = U_L for x < 0,   U_R for x > 0

has a self-similar solution made of three waves separated by two constant
"star" states.  The left and right waves are each either a shock or a
rarefaction; the middle wave is always a contact discontinuity.  Pressure and
normal velocity are continuous across the contact, so the whole solution is
pinned down by the single scalar ``p_star`` solving Toro eq. (4.5):

    f(p) = f_L(p, W_L) + f_R(p, W_R) + (u_R - u_L) = 0.

``f`` is monotone increasing and concave in ``p``, so Newton-Raphson converges
from any positive starting point; a good starting point (``initial_pressure_guess``
below) just makes it converge in 3-5 iterations instead of 20.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.jit import njit
from .physics import check_gamma, sound_speed, specific_internal_energy

__all__ = [
    "ExactSolution",
    "pressure_function",
    "pressure_function_derivative",
    "initial_pressure_guess",
    "solve_star_pressure",
    "solve_exact",
    "sample_exact",
]


# --------------------------------------------------------------------------- #
# The pressure function f(p) and its derivative
# --------------------------------------------------------------------------- #

@njit
def _f_single(p, rho_k, p_k, a_k, gamma):
    """One-sided pressure function ``f_K(p)`` (Toro eqs. 4.6-4.7).

    Shock branch (``p > p_K``), from the Rankine-Hugoniot relations::

        f_K = (p - p_K) * sqrt(A_K / (p + B_K)),
        A_K = 2 / ((gamma + 1) * rho_K),   B_K = p_K * (gamma - 1)/(gamma + 1)

    Rarefaction branch (``p <= p_K``), from the isentropic + Riemann-invariant
    relations::

        f_K = (2 * a_K / (gamma - 1)) * ((p / p_K)**((gamma - 1)/(2 gamma)) - 1)
    """
    if p > p_k:  # shock
        a_big = 2.0 / ((gamma + 1.0) * rho_k)
        b_big = p_k * (gamma - 1.0) / (gamma + 1.0)
        return (p - p_k) * np.sqrt(a_big / (p + b_big))
    # rarefaction
    return (2.0 * a_k / (gamma - 1.0)) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


@njit
def _fprime_single(p, rho_k, p_k, a_k, gamma):
    """Derivative ``df_K/dp`` (Toro eqs. 4.37, differentiating 4.6-4.7).

    Note the rarefaction exponent: ``-(gamma + 1) / (2 * gamma)``, not
    ``-(gamma - 1) / (2 * gamma)``.  Getting it wrong costs Newton its
    quadratic convergence without preventing it from finding the root, so the
    derivative is checked against a finite difference in the test suite.
    """
    if p > p_k:  # shock
        a_big = 2.0 / ((gamma + 1.0) * rho_k)
        b_big = p_k * (gamma - 1.0) / (gamma + 1.0)
        return np.sqrt(a_big / (b_big + p)) * (
            1.0 - (p - p_k) / (2.0 * (b_big + p))
        )
    # rarefaction
    return (1.0 / (rho_k * a_k)) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))


@njit
def pressure_function(p, gamma, w_l, w_r):
    """``f(p) = f_L(p) + f_R(p) + (u_R - u_L)``, Toro eq. (4.5).

    The root of this function is the star-region pressure.
    """
    rho_l, u_l, p_l = w_l[0], w_l[1], w_l[2]
    rho_r, u_r, p_r = w_r[0], w_r[1], w_r[2]
    a_l = sound_speed(rho_l, p_l, gamma)
    a_r = sound_speed(rho_r, p_r, gamma)
    return (
        _f_single(p, rho_l, p_l, a_l, gamma)
        + _f_single(p, rho_r, p_r, a_r, gamma)
        + (u_r - u_l)
    )


@njit
def pressure_function_derivative(p, gamma, w_l, w_r):
    """``df/dp``; strictly positive, so ``f`` is monotone (Toro Sec. 4.3.1)."""
    rho_l, p_l = w_l[0], w_l[2]
    rho_r, p_r = w_r[0], w_r[2]
    a_l = sound_speed(rho_l, p_l, gamma)
    a_r = sound_speed(rho_r, p_r, gamma)
    return _fprime_single(p, rho_l, p_l, a_l, gamma) + _fprime_single(
        p, rho_r, p_r, a_r, gamma
    )


# --------------------------------------------------------------------------- #
# Initial guess and Newton iteration
# --------------------------------------------------------------------------- #

@njit
def initial_pressure_guess(gamma, w_l, w_r):
    """Adaptive starting pressure for the Newton iteration (Toro Sec. 9.5).

    Picks between three *non-iterative* approximate solvers depending on how
    strong the initial pressure ratio is:

    * **PVRS** (primitive-variable Riemann solver, Toro eq. 9.20) for weak
      waves -- the linearised answer;
    * **Two-rarefaction** (Toro eq. 9.32), which is exact when both waves are
      rarefactions, used when PVRS undershoots ``min(p_L, p_R)``;
    * **Two-shock** (Toro eq. 9.42) when PVRS overshoots ``max(p_L, p_R)``.

    Each is cheap and each is asymptotically exact in its own regime, so the
    resulting guess is uniformly good across regimes, including dimensional
    problems at ~1e5 Pa where a fixed guess would be nowhere near the root.
    """
    rho_l, u_l, p_l = w_l[0], w_l[1], w_l[2]
    rho_r, u_r, p_r = w_r[0], w_r[1], w_r[2]
    a_l = sound_speed(rho_l, p_l, gamma)
    a_r = sound_speed(rho_r, p_r, gamma)

    # Pressure ratio above which we distrust the linearised PVRS estimate.
    q_user = 2.0

    # --- PVRS estimate ---------------------------------------------------- #
    cup = 0.25 * (rho_l + rho_r) * (a_l + a_r)
    p_pv = 0.5 * (p_l + p_r) + 0.5 * (u_l - u_r) * cup
    p_pv = max(p_pv, 1.0e-8 * min(p_l, p_r))

    p_min = min(p_l, p_r)
    p_max = max(p_l, p_r)
    q_max = p_max / p_min

    if q_max <= q_user and p_min <= p_pv <= p_max:
        return p_pv

    if p_pv < p_min:
        # --- Two-rarefaction approximation (exact for two rarefactions) --- #
        p_q = (p_l / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
        u_m = (
            p_q * u_l / a_l
            + u_r / a_r
            + (2.0 / (gamma - 1.0)) * (p_q - 1.0)
        ) / (p_q / a_l + 1.0 / a_r)
        pt_l = 1.0 + 0.5 * (gamma - 1.0) * (u_l - u_m) / a_l
        pt_r = 1.0 + 0.5 * (gamma - 1.0) * (u_m - u_r) / a_r
        expo = 2.0 * gamma / (gamma - 1.0)
        return 0.5 * (p_l * pt_l ** expo + p_r * pt_r ** expo)

    # --- Two-shock approximation -------------------------------------------- #
    ge_l = np.sqrt(
        (2.0 / ((gamma + 1.0) * rho_l)) / (p_l * (gamma - 1.0) / (gamma + 1.0) + p_pv)
    )
    ge_r = np.sqrt(
        (2.0 / ((gamma + 1.0) * rho_r)) / (p_r * (gamma - 1.0) / (gamma + 1.0) + p_pv)
    )
    return (ge_l * p_l + ge_r * p_r - (u_r - u_l)) / (ge_l + ge_r)


def solve_star_pressure(gamma, w_l, w_r, tol=1.0e-12, max_iter=100, p_guess=None):
    """Newton-Raphson solve of ``f(p_star) = 0`` (Toro Sec. 4.3.2).

    Parameters
    ----------
    gamma : float
        Ratio of specific heats.
    w_l, w_r : array_like, shape (3,)
        Left and right primitive states ``[rho, u, p]``.
    tol : float
        Convergence tolerance on the *relative* pressure change
        ``2 |p_new - p_old| / |p_new + p_old|``.  Being relative, it is
        dimensionally consistent: the same ``tol`` works for Sod's ``p ~ 1``
        and for a shock tube at ``p ~ 1e5 Pa``.
    max_iter : int
        Iteration cap.  Newton on this ``f`` converges quadratically, so
        hitting the cap means something is wrong with the input, not that the
        problem is merely hard.
    p_guess : float, optional
        Override the adaptive initial guess.  Mostly useful for teaching, to
        show how a poor guess slows convergence down.

    Returns
    -------
    (p_star, iterations) : tuple[float, int]

    Raises
    ------
    ValueError
        If the data generate a vacuum, or if Newton fails to converge.
    """
    gamma = check_gamma(gamma)
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    _check_positivity(w_l, w_r)
    _check_vacuum(gamma, w_l, w_r)

    p = float(initial_pressure_guess(gamma, w_l, w_r)) if p_guess is None else float(p_guess)
    if p <= 0.0:
        raise ValueError("initial pressure guess must be positive")

    for it in range(1, max_iter + 1):
        f = pressure_function(p, gamma, w_l, w_r)
        df = pressure_function_derivative(p, gamma, w_l, w_r)
        p_new = p - f / df
        # Newton on a concave monotone f can overshoot to p <= 0 from a bad
        # guess; bisect toward zero instead of letting the powers go complex.
        if p_new <= 0.0:
            p_new = 0.5 * p
        change = 2.0 * abs(p_new - p) / (p_new + p)
        p = p_new
        if change < tol:
            return p, it

    raise ValueError(
        f"Newton-Raphson for p_star did not converge in {max_iter} iterations "
        f"(last relative change {change:.3e}); check the input states"
    )


def _check_positivity(w_l, w_r):
    """Reject non-physical input before it turns into NaNs deep in a kernel."""
    for name, w in (("left", w_l), ("right", w_r)):
        if w[0] <= 0.0 or w[2] <= 0.0:
            raise ValueError(
                f"{name} state must have positive density and pressure, got "
                f"rho={w[0]!r}, p={w[2]!r}"
            )


def _check_vacuum(gamma, w_l, w_r):
    """Vacuum-generation check, Toro eq. (4.40).

    If the two states separate faster than the rarefaction fans can fill the
    gap, ``2(a_L + a_R)/(gamma - 1) <= u_R - u_L``, no star region exists and
    ``f(p)`` has no positive root.
    """
    a_l = float(sound_speed(w_l[0], w_l[2], gamma))
    a_r = float(sound_speed(w_r[0], w_r[2], gamma))
    du_crit = 2.0 * (a_l + a_r) / (gamma - 1.0)
    if du_crit <= (w_r[1] - w_l[1]):
        raise ValueError(
            "initial data generate a vacuum (Toro eq. 4.40): "
            f"u_R - u_L = {w_r[1] - w_l[1]:.6g} >= 2(a_L + a_R)/(gamma-1) = {du_crit:.6g}"
        )


# --------------------------------------------------------------------------- #
# Self-similar sampling of the wave fan
# --------------------------------------------------------------------------- #

@njit
def _sample_one(s, gamma, w_l, w_r, p_star, u_star):
    """Sample the exact solution at a single similarity coordinate ``s = x / t``.

    Implements Toro's ``SAMPLE`` routine (Sec. 4.5).  The ``x/t`` axis is
    partitioned without gaps: first by which side of the contact we are on,
    then, on that side, by the wave structure (shock -> one speed to test;
    rarefaction -> head and tail speeds bracketing the fan, inside which the
    generalised Riemann invariants give a closed-form profile).

    Returns ``[rho, u, p]``.
    """
    rho_l, u_l, p_l = w_l[0], w_l[1], w_l[2]
    rho_r, u_r, p_r = w_r[0], w_r[1], w_r[2]
    a_l = np.sqrt(gamma * p_l / rho_l)
    a_r = np.sqrt(gamma * p_r / rho_r)
    # mu = (gamma - 1) / (gamma + 1) appears all over the shock relations.
    mu = (gamma - 1.0) / (gamma + 1.0)

    out = np.empty(3)

    if s <= u_star:
        # ------------------------------------------------------ left of contact
        if p_star > p_l:
            # Left shock.  Density ratio from Rankine-Hugoniot (Toro eq. 4.19),
            # shock speed from eq. (4.20).
            rho_star_l = rho_l * (p_star / p_l + mu) / (mu * p_star / p_l + 1.0)
            s_l = u_l - a_l * np.sqrt(
                (gamma + 1.0) / (2.0 * gamma) * (p_star / p_l)
                + (gamma - 1.0) / (2.0 * gamma)
            )
            if s <= s_l:
                out[0] = rho_l
                out[1] = u_l
                out[2] = p_l
            else:
                out[0] = rho_star_l
                out[1] = u_star
                out[2] = p_star
        else:
            # Left rarefaction: isentropic, so rho follows p^(1/gamma)
            # (Toro eq. 4.23); head/tail speeds from eqs. (4.25)-(4.26).
            rho_star_l = rho_l * (p_star / p_l) ** (1.0 / gamma)
            a_star_l = a_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
            s_head = u_l - a_l          # left-most characteristic of the fan
            s_tail = u_star - a_star_l  # right-most characteristic of the fan
            if s <= s_head:
                out[0] = rho_l
                out[1] = u_l
                out[2] = p_l
            elif s >= s_tail:
                out[0] = rho_star_l
                out[1] = u_star
                out[2] = p_star
            else:
                # Inside the fan (Toro eq. 4.24): the "left fan" solution.
                fac = 2.0 / (gamma + 1.0) + (mu / a_l) * (u_l - s)
                out[0] = rho_l * fac ** (2.0 / (gamma - 1.0))
                out[1] = (2.0 / (gamma + 1.0)) * (
                    a_l + 0.5 * (gamma - 1.0) * u_l + s
                )
                out[2] = p_l * fac ** (2.0 * gamma / (gamma - 1.0))
    else:
        # ----------------------------------------------------- right of contact
        if p_star > p_r:
            # Right shock (Toro eqs. 4.32-4.33).
            rho_star_r = rho_r * (p_star / p_r + mu) / (mu * p_star / p_r + 1.0)
            s_r = u_r + a_r * np.sqrt(
                (gamma + 1.0) / (2.0 * gamma) * (p_star / p_r)
                + (gamma - 1.0) / (2.0 * gamma)
            )
            if s >= s_r:
                out[0] = rho_r
                out[1] = u_r
                out[2] = p_r
            else:
                out[0] = rho_star_r
                out[1] = u_star
                out[2] = p_star
        else:
            # Right rarefaction (Toro eqs. 4.35-4.36).
            rho_star_r = rho_r * (p_star / p_r) ** (1.0 / gamma)
            a_star_r = a_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
            s_head = u_r + a_r          # right-most characteristic of the fan
            s_tail = u_star + a_star_r  # left-most characteristic of the fan
            if s >= s_head:
                out[0] = rho_r
                out[1] = u_r
                out[2] = p_r
            elif s <= s_tail:
                out[0] = rho_star_r
                out[1] = u_star
                out[2] = p_star
            else:
                fac = 2.0 / (gamma + 1.0) - (mu / a_r) * (u_r - s)
                out[0] = rho_r * fac ** (2.0 / (gamma - 1.0))
                out[1] = (2.0 / (gamma + 1.0)) * (
                    -a_r + 0.5 * (gamma - 1.0) * u_r + s
                )
                out[2] = p_r * fac ** (2.0 * gamma / (gamma - 1.0))

    return out


@njit
def _sample_many(s_vals, gamma, w_l, w_r, p_star, u_star):
    """Vectorised wrapper around :func:`_sample_one` (compiled loop)."""
    n = s_vals.shape[0]
    out = np.empty((n, 3))
    for i in range(n):
        out[i, :] = _sample_one(s_vals[i], gamma, w_l, w_r, p_star, u_star)
    return out


@dataclass
class ExactSolution:
    """Exact solution of one Riemann problem, sampled on a grid.

    Attributes
    ----------
    x : ndarray, shape (n,)
        Cell-centre coordinates in the physical domain.
    rho, u, p, e : ndarray, shape (n,)
        Density, velocity, pressure and specific internal energy.
    p_star, u_star : float
        Star-region pressure and velocity (constant across the contact).
    gamma, t : float
        Ratio of specific heats and the output time.
    left_wave, right_wave : str
        ``"shock"`` or ``"rarefaction"`` -- the wave pattern that was realised.
    iterations : int
        Newton iterations used for ``p_star``.
    """

    x: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    p: np.ndarray
    e: np.ndarray
    p_star: float
    u_star: float
    gamma: float
    t: float
    left_wave: str
    right_wave: str
    iterations: int

    @property
    def primitive(self):
        """The solution as an ``(n, 3)`` array of ``[rho, u, p]``."""
        return np.column_stack((self.rho, self.u, self.p))


def solve_exact(
    gamma,
    w_l,
    w_r,
    t,
    x_left=0.0,
    x_right=1.0,
    n=1000,
    x_diaphragm=0.5,
    tol=1.0e-12,
    p_guess=None,
):
    """Solve and sample the exact Riemann problem on a cell-centred grid.

    Parameters
    ----------
    gamma : float
        Ratio of specific heats.
    w_l, w_r : array_like, shape (3,)
        Left and right primitive states ``[rho, u, p]``.
    t : float
        Output time.  Must be positive: the solution is a function of ``x/t``
        only, and ``t = 0`` is the (discontinuous) initial condition.
    x_left, x_right : float
        Domain bounds.
    n : int
        Number of sample points (cell centres).
    x_diaphragm : float
        Position of the initial discontinuity.
    tol, p_guess
        Passed to :func:`solve_star_pressure`.

    Returns
    -------
    ExactSolution
    """
    if t <= 0.0:
        raise ValueError("output time t must be positive")
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)

    p_star, iterations = solve_star_pressure(gamma, w_l, w_r, tol=tol, p_guess=p_guess)

    # Star velocity, Toro eq. (4.9).  Equivalent to u_L - f_L(p*) and to
    # u_R + f_R(p*); the symmetric average is the numerically kindest form.
    f_l = float(_f_single(p_star, w_l[0], w_l[2], float(sound_speed(w_l[0], w_l[2], gamma)), gamma))
    f_r = float(_f_single(p_star, w_r[0], w_r[2], float(sound_speed(w_r[0], w_r[2], gamma)), gamma))
    u_star = 0.5 * (w_l[1] + w_r[1]) + 0.5 * (f_r - f_l)

    dx = (x_right - x_left) / n
    x = x_left + dx * (np.arange(n) + 0.5)
    # The solution is self-similar about the diaphragm, so sample in s = xi / t
    # with xi measured from the diaphragm.
    s_vals = (x - x_diaphragm) / t

    prim = _sample_many(s_vals, gamma, w_l, w_r, p_star, u_star)
    rho, u, p = prim[:, 0], prim[:, 1], prim[:, 2]

    return ExactSolution(
        x=x,
        rho=rho,
        u=u,
        p=p,
        e=specific_internal_energy(rho, p, gamma),
        p_star=float(p_star),
        u_star=float(u_star),
        gamma=float(gamma),
        t=float(t),
        left_wave="shock" if p_star > w_l[2] else "rarefaction",
        right_wave="shock" if p_star > w_r[2] else "rarefaction",
        iterations=int(iterations),
    )


def sample_exact(s_vals, gamma, w_l, w_r, tol=1.0e-12):
    """Sample the exact solution directly in similarity coordinates ``s = x/t``.

    Useful for checking a numerical flux at an interface, where only the state
    at ``s = 0`` is wanted.
    """
    w_l = np.asarray(w_l, dtype=float)
    w_r = np.asarray(w_r, dtype=float)
    p_star, _ = solve_star_pressure(gamma, w_l, w_r, tol=tol)
    a_l = float(sound_speed(w_l[0], w_l[2], gamma))
    a_r = float(sound_speed(w_r[0], w_r[2], gamma))
    f_l = float(_f_single(p_star, w_l[0], w_l[2], a_l, gamma))
    f_r = float(_f_single(p_star, w_r[0], w_r[2], a_r, gamma))
    u_star = 0.5 * (w_l[1] + w_r[1]) + 0.5 * (f_r - f_l)
    return _sample_many(np.atleast_1d(np.asarray(s_vals, dtype=float)),
                        gamma, w_l, w_r, p_star, u_star)
