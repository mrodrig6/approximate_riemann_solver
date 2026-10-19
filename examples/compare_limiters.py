#!/usr/bin/env python3
"""Compare flux limiters, and first vs. second order, on Sod's problem.

Shows the point of the TVD machinery: all four runs use the identical Roe
flux and differ only in how the second-order correction is limited.  Run::

    python examples/compare_limiters.py
"""

from __future__ import annotations

import os
import sys

# Let this example run straight from a checkout, with or without the package
# installed (`pip install -e .`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import plot_comparison, save_figure, solve_exact, solve_roe

gamma, t_end, nx = 1.4, 0.25, 100
w_left, w_right = [1.0, 0.0, 1.0], [0.125, 0.0, 0.1]

exact = solve_exact(gamma, w_left, w_right, t_end, n=2000)
reference = solve_exact(gamma, w_left, w_right, t_end, n=nx)

# (solver keyword overrides, legend label) for each configuration to compare.
configs = [
    (dict(second_order=False), "1st-order Roe"),
    (dict(limiter="minmod"), "2nd order, minmod"),
    (dict(limiter="vanleer"), "2nd order, van Leer"),
    (dict(limiter="superbee"), "2nd order, superbee"),
]

runs = []
print(f"{'configuration':<24} {'L1 (rho)':>12} {'steps':>7}")
for overrides, label in configs:
    roe = solve_roe(gamma, w_left, w_right, t_end, nx=nx, **overrides)
    err = np.mean(np.abs(roe.rho - reference.rho))
    print(f"{label:<24} {err:>12.4e} {roe.steps:>7d}")
    runs.append((roe, label))

fig, axes = plot_comparison(
    exact=exact,
    numerical=runs,
    fields=("rho", "e"),
    title=f"Flux limiters compared (Sod, {nx} cells)",
    figsize=(9.0, 3.8),
)
# Zoom onto the contact discontinuity, which is what the limiters mainly affect.
for ax in axes[:2]:
    ax.set_xlim(0.6, 0.85)
print("\nwrote", save_figure(fig, "limiters_example.png"))
