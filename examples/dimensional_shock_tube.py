#!/usr/bin/env python3
"""The dimensional (SI) air shock tube from the original MATLAB driver.

This is the port of ``test_input2o1.m``.  It is worth running as its own
example because it is the case that exposed two bugs in the MATLAB exact
solver: a hard-coded initial pressure guess of ``0.5`` and an *absolute*
Newton tolerance of ``1e-15``, neither of which means anything when the
pressures are of order ``1e5`` Pa.  The Python solver uses Toro's adaptive
initial guess and a relative tolerance, so the same code handles both this
case and the non-dimensional ones without tuning.

Run::

    python examples/dimensional_shock_tube.py
"""

from __future__ import annotations

import os
import sys

# Let this example run straight from a checkout, with or without the package
# installed (`pip install -e .`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from riemann import get_problem, plot_comparison, save_figure, solve_exact, solve_roe

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

roe = solve_roe(
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
