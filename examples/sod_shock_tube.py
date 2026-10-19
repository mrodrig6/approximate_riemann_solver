#!/usr/bin/env python3
"""Simplest possible example: Sod's shock tube, exact vs. Roe.

Run it from the repository root::

    python examples/sod_shock_tube.py

It prints the star-region state and the error against the exact solution, and
writes ``sod_example.png``.
"""

from __future__ import annotations

import os
import sys

# Let this example run straight from a checkout, with or without the package
# installed (`pip install -e .`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import plot_comparison, save_figure, solve_exact, solve_roe

# ---------------------------------------------------------------------------
# Problem definition.
# ---------------------------------------------------------------------------
gamma = 1.4                     # ratio of specific heats
w_left = [1.000, 0.0, 1.0]      # [rho, u, p] to the left of the diaphragm
w_right = [0.125, 0.0, 0.1]     # [rho, u, p] to the right
t_end = 0.25                    # output time
nx = 200                        # finite-volume cells for the Roe solver

# ---------------------------------------------------------------------------
# Exact solution.  Sampled finely (2000 points) because it costs nothing and
# makes a better reference curve to plot the 200 numerical cells against.
# ---------------------------------------------------------------------------
exact = solve_exact(gamma, w_left, w_right, t_end, n=2000)
print(f"star-region pressure p* = {exact.p_star:.8f}")
print(f"star-region velocity u* = {exact.u_star:.8f}")
print(f"wave pattern            : {exact.left_wave} | contact | {exact.right_wave}")
print(f"Newton iterations       : {exact.iterations}")

# ---------------------------------------------------------------------------
# Numerical solution: Roe's approximate solver, second order with the superbee
# limiter and the Harten-Hyman entropy fix (both on by default).
# ---------------------------------------------------------------------------
roe = solve_roe(gamma, w_left, w_right, t_end, nx=nx, cfl=0.9, limiter="superbee")
print(f"\n{roe.label}: {roe.steps} steps to t = {roe.t}")

# ---------------------------------------------------------------------------
# Error.  Sample the exact solution on the *same* grid so the comparison is
# cell-by-cell rather than interpolated.
# ---------------------------------------------------------------------------
reference = solve_exact(gamma, w_left, w_right, t_end, n=nx)
for field in ("rho", "u", "p", "e"):
    err = np.mean(np.abs(getattr(roe, field) - getattr(reference, field)))
    print(f"  L1 error in {field:<3} = {err:.4e}")

fig, _ = plot_comparison(
    exact=exact,
    numerical=[roe],
    title=f"Sod's shock tube at $t = {t_end}$, {nx} cells",
)
print("\nwrote", save_figure(fig, "sod_example.png"))
