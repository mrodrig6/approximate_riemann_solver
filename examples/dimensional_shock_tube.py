#!/usr/bin/env python3
"""A dimensional (SI) air shock tube.

Worth running as its own example because it is the only case here with
pressures of order 1e5 rather than order 1.  Nothing is tuned for it: the
exact solver uses Toro's adaptive initial guess and a *relative* convergence
criterion, both of which are scale-free, so the same code handles this and
the non-dimensional problems alike.

Run::

    python examples/dimensional_shock_tube.py
"""

from __future__ import annotations

import os
import sys

# Let this example run straight from a checkout, with or without the package
# installed (`pip install -e .`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import get_problem, plot_comparison, save_figure, solve_exact, solve

problem = get_problem("air_shock_tube")
print(problem.description)
print(f"reference: {problem.reference}\n")

rho_r, _, p_r = problem.right
print(f"ambient air : rho = {rho_r:.4f} kg/m^3, p = {p_r:.0f} Pa")
rho_l, u_l, p_l = problem.left
print(f"driver gas  : rho = {rho_l:.4f} kg/m^3, u = {u_l:.2f} m/s, p = {p_l:.0f} Pa")

exact = solve_exact(
    problem.gamma, problem.left, problem.right, problem.t_end,
    n=2000, x_diaphragm=problem.x_diaphragm,
)
print(f"\np* = {exact.p_star:.2f} Pa, u* = {exact.u_star:.4f} m/s")
print(f"wave pattern: {exact.left_wave} | contact | {exact.right_wave}")
print(f"Newton iterations: {exact.iterations}")
# The left state was constructed as the exact post-shock state of a right-running
# shock, so the left wave is vanishingly weak and the solution is essentially a
# single shock travelling into the ambient air.

roe = solve(
    problem.gamma, problem.left, problem.right, problem.t_end,
    nx=200, x_diaphragm=problem.x_diaphragm, limiter="superbee",
)
print(f"\n{roe.label}: {roe.steps} steps to t = {roe.t:g} s")

fig, _ = plot_comparison(
    exact=exact,
    numerical=[roe],
    title=f"Air shock tube, $t = {problem.t_end * 1e3:g}$ ms",
    dimensional=True,
    xlabel="$x$ (m)",
)
print("\nwrote", save_figure(fig, "air_shock_tube_example.png"))
