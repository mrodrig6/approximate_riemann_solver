"""Standard shallow-water test problems.

The dam-break family is to the shallow-water equations what the shock tube
is to the Euler equations: a single discontinuity in depth, released at
``t = 0``.  The cases below cover the four wave patterns plus the two dry-bed
situations.

Reference: Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*,
Wiley 2001, Chapter 5 and the test set of Chapter 6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .physics import GRAVITY

__all__ = ["ShallowWaterProblem", "PROBLEMS", "get_problem", "list_problems"]


@dataclass(frozen=True)
class ShallowWaterProblem:
    """A complete shallow-water Riemann problem.

    Attributes
    ----------
    name, description : str
    w_l, w_r : tuple[float, float]
        Left and right primitive states ``(h, u)``.
    t_end : float
        Conventional output time.
    gravity : float
    x_left, x_right, x_diaphragm : float
    cfl : float
        A CFL number known to work for this case.  Dry-bed problems want a
        smaller one than wet problems, because the wetting front outruns the
        usual signal-speed estimate.
    reference : str
    """

    name: str
    description: str
    w_l: tuple
    w_r: tuple
    t_end: float
    gravity: float = GRAVITY
    x_left: float = 0.0
    x_right: float = 1.0
    x_diaphragm: float = 0.5
    cfl: float = 0.9
    reference: str = ""

    @property
    def left(self):
        """Left primitive state as a float array ``[h, u]``."""
        return np.array(self.w_l, dtype=float)

    @property
    def right(self):
        """Right primitive state as a float array ``[h, u]``."""
        return np.array(self.w_r, dtype=float)

    def kwargs(self):
        """Geometry and physics keywords shared by the exact and FV solvers."""
        return dict(
            w_l=self.left,
            w_r=self.right,
            gravity=self.gravity,
            x_left=self.x_left,
            x_right=self.x_right,
            x_diaphragm=self.x_diaphragm,
        )


PROBLEMS = {
    p.name: p
    for p in [
        ShallowWaterProblem(
            name="dam_break",
            description="Dam break on a wet bed: rarefaction and a right-running bore",
            w_l=(1.0, 0.0),
            w_r=(0.1, 0.0),
            t_end=0.05,
            reference="Stoker's dam-break solution; Toro SW Sec. 5.3",
        ),
        ShallowWaterProblem(
            name="dam_break_dry",
            description="Dam break onto a dry bed: the Ritter solution",
            w_l=(1.0, 0.0),
            w_r=(0.0, 0.0),
            t_end=0.05,
            cfl=0.45,
            reference="Ritter's dry-bed solution; Toro SW Sec. 5.5",
        ),
        ShallowWaterProblem(
            name="two_bores",
            description="Colliding streams: a bore running each way",
            w_l=(1.0, 2.5),
            w_r=(1.0, -2.5),
            t_end=0.05,
            reference="Toro SW Ch. 6 test set",
        ),
        ShallowWaterProblem(
            name="two_rarefactions",
            description="Separating streams: a rarefaction each way, shallow middle state",
            w_l=(1.0, -2.5),
            w_r=(1.0, 2.5),
            t_end=0.05,
            reference="Toro SW Ch. 6 test set",
        ),
        ShallowWaterProblem(
            name="dry_generation",
            description="Fast separation: the rarefactions cannot fill the gap, so a dry region opens",
            # Dry bed is generated when u_R - u_L >= 2(a_L + a_R); here that
            # threshold is 4 sqrt(9.81) = 12.53, and the data exceed it.
            w_l=(1.0, -7.0),
            w_r=(1.0, 7.0),
            t_end=0.03,
            cfl=0.45,
            reference="dry-bed generation; Toro SW Sec. 5.5",
        ),
        ShallowWaterProblem(
            name="subcritical_step",
            description="Weak depth step, subcritical throughout",
            w_l=(1.0, 0.5),
            w_r=(0.8, 0.5),
            t_end=0.05,
            reference="smooth-ish case for accuracy comparison",
        ),
    ]
}


def get_problem(name):
    """Look up a built-in shallow-water problem by name."""
    key = str(name).strip().lower()
    if key not in PROBLEMS:
        raise KeyError(
            f"unknown shallow-water problem {name!r}; available problems are "
            f"{', '.join(sorted(PROBLEMS))}"
        )
    return PROBLEMS[key]


def list_problems():
    """Return ``(name, description)`` pairs for every built-in problem."""
    return [(p.name, p.description) for p in PROBLEMS.values()]
