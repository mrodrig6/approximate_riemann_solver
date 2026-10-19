#!/usr/bin/env python3
"""Dam break on a wet bed and on a dry bed.

The shallow-water counterpart of the shock-tube example.  Run::

    python examples/shallow_water_dam_break.py

The dry case is the interesting one: it has a closed-form answer (Ritter's
solution) and it is the case a solver written only for wet beds cannot do at
all.
"""

from __future__ import annotations

import os
import sys

# Let this example run straight from a checkout, with or without the package
# installed (`pip install -e .`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import plot_shallow_water, save_figure
from src.shallow_water import GRAVITY, get_problem, solve, solve_exact

for name in ("dam_break", "dam_break_dry"):
    prob = get_problem(name)
    print(f"\n{prob.name}: {prob.description}")
    print(f"  left  (h, u) = ({prob.left[0]:g}, {prob.left[1]:g}) m, m/s")
    print(f"  right (h, u) = ({prob.right[0]:g}, {prob.right[1]:g}) m, m/s")

    exact = solve_exact(
        prob.left, prob.right, prob.t_end, gravity=prob.gravity, n=2000,
        x_diaphragm=prob.x_diaphragm,
    )
    if exact.dry:
        # Ritter's solution: these three numbers are exact for all time.
        h_l = prob.left[0]
        print("  dry bed, so there is no star region.  Ritter's solution gives")
        print(f"    depth at the dam site : {4.0 * h_l / 9.0:.6f} m")
        print(f"    velocity there        : {2.0 * np.sqrt(GRAVITY * h_l) / 3.0:.6f} m/s")
        print(f"    front speed           : {2.0 * np.sqrt(GRAVITY * h_l):.6f} m/s")
    else:
        print(f"  h* = {exact.h_star:.6f} m, u* = {exact.u_star:.6f} m/s "
              f"({exact.iterations} Newton iterations)")
    print(f"  wave pattern: {exact.left_wave} | {exact.right_wave}")

    sol = solve(
        prob.left, prob.right, prob.t_end, gravity=prob.gravity, nx=200,
        x_diaphragm=prob.x_diaphragm, cfl=prob.cfl, flux="hll",
    )
    reference = solve_exact(
        prob.left, prob.right, prob.t_end, gravity=prob.gravity, n=200,
        x_diaphragm=prob.x_diaphragm,
    )
    print(f"  {sol.label}: {sol.steps} steps to t = {sol.t:g} s")
    for field in ("h", "u"):
        err = np.mean(np.abs(getattr(sol, field) - getattr(reference, field)))
        print(f"    L1 error in {field} = {err:.4e}")

    fig, _ = plot_shallow_water(
        exact=exact, numerical=[sol],
        title=f"{prob.description}  ($t = {prob.t_end:g}$ s)",
    )
    print("  wrote", save_figure(fig, f"{name}.png"))
