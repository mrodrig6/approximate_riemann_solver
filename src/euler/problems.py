"""Catalogue of standard Riemann-problem test cases.

The first five are Toro's test problems from *Riemann Solvers and Numerical
Methods for Fluid Dynamics* (3rd ed.), Sec. 4.3.3 and Table 4.1 / Sec. 6.4.
They are the de facto acceptance suite for a 1-D Euler code because between
them they exercise every wave pattern and every failure mode:

===== ================================= ==========================================
Test  Wave pattern                      What it stresses
===== ================================= ==========================================
1     rarefaction / contact / shock     Sod's problem: the basic sanity check.
2     rarefaction / contact / rarefact. The "123" problem; a near-vacuum star
                                        region, so positivity and the Newton
                                        initial guess.
3     rarefaction / contact / shock     Left half of the Woodward-Colella blast;
                                        a very strong (1e5) pressure ratio.
4     shock / contact / shock           Right half of the blast: two strong
                                        shocks colliding.
5     rarefaction / contact / shock     A slowly-moving contact -- the classic
                                        demonstration that Roe smears contacts
                                        and that a compressive limiter helps.
===== ================================= ==========================================

``sonic_rarefaction`` is added for the entropy fix: it is set up so the left
rarefaction fan straddles ``x = x_diaphragm``, which is exactly the transonic
situation where an unfixed Roe solver produces an entropy-violating glitch.

``air_shock_tube`` is a dimensional case: SI units, a right-moving Mach ~1.35
shock in air at standard conditions.  It is the one problem here whose
pressures are ~1e5 rather than ~1, so it exercises the scale invariance of
the exact solver's initial guess and convergence criterion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RiemannProblem",
    "PROBLEMS",
    "acoustic_pulse",
    "advected_pulse",
    "cell_centres",
    "get_problem",
    "list_problems",
]


@dataclass(frozen=True)
class RiemannProblem:
    """A complete, self-contained Riemann-problem specification.

    Attributes
    ----------
    name : str
        Short key used by the CLI and the figure scripts.
    description : str
        One-line human-readable summary.
    w_l, w_r : tuple[float, float, float]
        Left and right primitive states ``(rho, u, p)``.
    gamma : float
        Ratio of specific heats.
    t_end : float
        Conventional output time for this test.
    x_left, x_right, x_diaphragm : float
        Domain and the initial discontinuity position.
    reference : str
        Where the case comes from, so a reader can look it up in Toro.
    flux_note : str
        Non-empty if some flux is *expected* to struggle (or fail) on this
        case, so that a known, documented limitation of a method is not
        mistaken for a broken implementation.  The CLI prints it only when the
        selected flux is not one of :data:`src.euler.fluxes.ROBUST_FLUXES`.
    """

    name: str
    description: str
    w_l: tuple
    w_r: tuple
    gamma: float
    t_end: float
    x_left: float = 0.0
    x_right: float = 1.0
    x_diaphragm: float = 0.5
    reference: str = ""
    flux_note: str = ""

    @property
    def left(self):
        """Left primitive state as a float array ``[rho, u, p]``."""
        return np.array(self.w_l, dtype=float)

    @property
    def right(self):
        """Right primitive state as a float array ``[rho, u, p]``."""
        return np.array(self.w_r, dtype=float)

    def kwargs(self):
        """The geometry/physics keywords shared by the exact and Roe solvers."""
        return dict(
            gamma=self.gamma,
            w_l=self.left,
            w_r=self.right,
            t_end=self.t_end,
            x_left=self.x_left,
            x_right=self.x_right,
            x_diaphragm=self.x_diaphragm,
        )


#: All built-in test problems, keyed by name.
PROBLEMS = {
    p.name: p
    for p in [
        RiemannProblem(
            name="sod",
            description="Sod's problem: left rarefaction, contact, right shock",
            w_l=(1.0, 0.0, 1.0),
            w_r=(0.125, 0.0, 0.1),
            gamma=1.4,
            t_end=0.25,
            reference="Toro Test 1, Sec. 4.3.3 / Table 4.1",
        ),
        RiemannProblem(
            name="toro2",
            description="123 problem: two strong rarefactions, near-vacuum star state",
            w_l=(1.0, -2.0, 0.4),
            w_r=(1.0, 2.0, 0.4),
            gamma=1.4,
            t_end=0.15,
            reference="Toro Test 2, Sec. 4.3.3",
            flux_note=(
                "Roe's scheme FAILS on this problem, at first and second order "
                "alike: the star region is so close to vacuum that the "
                "linearised solver drives the internal energy negative in the "
                "first time step.  This is a documented property of the method, "
                "not a defect of this implementation (Einfeldt, Munz, Roe & "
                "Sjoegreen 1991; Toro Sec. 11.4).  Use a positivity-preserving "
                "flux instead: --flux hllc (recommended), hll, rusanov or "
                "lax_friedrichs all run it at second order, and ausmup runs "
                "it at first order."
            ),
        ),
        RiemannProblem(
            name="toro3",
            description="Left half of the Woodward-Colella blast: pressure ratio 1e5",
            w_l=(1.0, 0.0, 1000.0),
            w_r=(1.0, 0.0, 0.01),
            gamma=1.4,
            t_end=0.012,
            reference="Toro Test 3, Sec. 4.3.3",
        ),
        RiemannProblem(
            name="toro4",
            description="Right half of the blast: two colliding strong shocks",
            w_l=(5.99924, 19.5975, 460.894),
            w_r=(5.99242, -6.19633, 46.0950),
            gamma=1.4,
            t_end=0.035,
            reference="Toro Test 4, Sec. 4.3.3",
        ),
        RiemannProblem(
            name="toro5",
            description="Slowly-moving contact discontinuity (stresses contact resolution)",
            w_l=(1.0, -19.59745, 1000.0),
            w_r=(1.0, -19.59745, 0.01),
            gamma=1.4,
            t_end=0.012,
            reference="Toro Test 5, Sec. 4.3.3",
            flux_note=(
                "Sensitive to the time step: because the second-order "
                "correction scales as 1 - |nu|, an artificially small CFL "
                "number removes the scheme's dissipation and breaks positivity "
                "here.  Run it at CFL ~ 0.9.  The Rusanov flux also fails here "
                "at second order (its positivity proof covers the first-order "
                "flux, not the TVD correction); HLLC and HLL are fine."
            ),
        ),
        RiemannProblem(
            name="sonic_rarefaction",
            description="Transonic left rarefaction: exercises the entropy fix",
            # u_L > 0 shifts the whole left fan to the right so that the sonic
            # point u - a = 0 sits inside the domain near the diaphragm.
            w_l=(1.0, 0.75, 1.0),
            w_r=(0.125, 0.0, 0.1),
            gamma=1.4,
            t_end=0.2,
            x_diaphragm=0.3,
            reference="Toro Sec. 11.3 (entropy fix); cf. Sec. 6.4 Test 1 variant",
        ),
        RiemannProblem(
            name="air_shock_tube",
            description="Dimensional air shock tube (SI units), Mach ~1.35 right-running shock",
            # Post-shock driver state on the left, ambient air on the right.
            w_l=(1.204 * 1.8105824688, 225.894320023, 101325.0 * 2.35438333333),
            w_r=(1.204, 0.0, 101325.0),
            gamma=1.4,
            t_end=3.0e-4,
            reference="dimensional shock tube in air at standard conditions",
        ),
    ]
}


def get_problem(name):
    """Look up a built-in problem by name, with a helpful error message."""
    key = str(name).strip().lower()
    if key not in PROBLEMS:
        raise KeyError(
            f"unknown problem {name!r}; available problems are "
            f"{', '.join(sorted(PROBLEMS))}"
        )
    return PROBLEMS[key]


def list_problems():
    """Return ``(name, description)`` pairs for every built-in problem."""
    return [(p.name, p.description) for p in PROBLEMS.values()]


# --------------------------------------------------------------------------- #
# Smooth initial conditions, for order-of-accuracy studies
# --------------------------------------------------------------------------- #
#
# Every problem above contains a discontinuity, and near one *any* scheme
# makes an O(1) error over an O(dx) region -- so the L1 error converges at
# first order however high the formal order of the method is.  Measuring the
# formal order therefore needs smooth data, which is what these supply.  Feed
# them to :func:`src.euler.solver.solve_field`.


def advected_pulse(x, gamma=1.4, velocity=1.0, pressure=1.0, amplitude=0.5,
                   centre=0.3, width=200.0):
    """A smooth density pulse advected at constant velocity.

    Velocity and pressure are uniform, so only the linearly degenerate field
    is excited and the exact solution is simply the initial profile
    translated: ``rho(x, t) = rho_0(x - u t)``.  Having an analytic answer
    makes this the cleanest order check available, at the cost of testing
    only the contact field.

    Returns an ``(n, 3)`` array of primitive states.
    """
    x = np.asarray(x, dtype=float)
    rho = 1.0 + amplitude * np.exp(-width * (x - centre) ** 2)
    return np.column_stack(
        [rho, np.full_like(x, velocity), np.full_like(x, pressure)]
    )


def acoustic_pulse(x, gamma=1.4, amplitude=0.2, centre=0.5, width=200.0):
    """A smooth isentropic pressure pulse at rest.

    The density follows the isentrope ``rho = p**(1/gamma)``, so the data are
    smooth and physical; the pulse then splits into left- and right-running
    acoustic waves.  Unlike :func:`advected_pulse` this excites the genuinely
    non-linear fields, which is where schemes of the same formal order
    actually differ -- but it has no closed-form solution, so convergence
    must be measured against a fine-grid reference.

    Returns an ``(n, 3)`` array of primitive states.
    """
    x = np.asarray(x, dtype=float)
    p = 1.0 + amplitude * np.exp(-width * (x - centre) ** 2)
    return np.column_stack([p ** (1.0 / gamma), np.zeros_like(x), p])


def cell_centres(nx, x_left=0.0, x_right=1.0):
    """Cell-centre coordinates of a uniform grid, matching the solvers'."""
    dx = (x_right - x_left) / nx
    return x_left + dx * (np.arange(nx) + 0.5)
