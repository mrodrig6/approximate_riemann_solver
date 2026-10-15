#!/usr/bin/env python3
"""Compare every numerical flux on the same problem.

All runs use the identical grid, limiter and CFL number; only the first-order
flux differs.  Run::

    python examples/compare_fluxes.py

The contact discontinuity is what separates the fluxes -- every one of them
captures the shock in one or two cells -- so the figure zooms in on it.
"""

from __future__ import annotations

import os
import sys

# Let this example run straight from a checkout, with or without the package
# installed (`pip install -e .`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from riemann import plot_comparison, save_figure, solve_exact, solve_roe

gamma, t_end, nx = 1.4, 0.25, 100
w_left, w_right = [1.0, 0.0, 1.0], [0.125, 0.0, 0.1]

exact = solve_exact(gamma, w_left, w_right, t_end, n=2000)
reference = solve_exact(gamma, w_left, w_right, t_end, n=nx)

# The two plateau values of density across the contact, used to measure how
# many cells the contact is smeared over.
RHO_LO, RHO_HI = 0.2656, 0.4263


def contact_width(sol):
    """Number of cells whose density lies strictly between the two plateaux."""
    span = RHO_HI - RHO_LO
    inside = (sol.rho > RHO_LO + 0.05 * span) & (sol.rho < RHO_HI - 0.05 * span)
    return int(np.sum(inside[(sol.x > 0.6) & (sol.x < 0.85)]))


runs = []
print(f"Sod's problem, {nx} cells, superbee limiter, CFL 0.9\n")
print(f"{'flux':<18} {'contact':>9} {'L1 (rho)':>12}")
for name in ("roe", "hllc", "hll", "ausm", "rusanov", "lax_friedrichs"):
    try:
        sol = solve_roe(gamma, w_left, w_right, t_end, nx=nx, flux=name)
    except FloatingPointError:
        # AUSM+ has no pressure-velocity coupling in its mass flux, so it
        # cannot start a flow from rest.  See riemann.fluxes.ausm_plus_flux.
        print(f"{name:<18} {'--':>9} {'cannot start from rest':>12}")
        continue
    err = np.mean(np.abs(sol.rho - reference.rho))
    print(f"{name:<18} {contact_width(sol):>6d} cells {err:>12.4e}")
    runs.append((sol, sol.label.split(",")[0]))

fig, axes = plot_comparison(
    exact=exact,
    numerical=runs,
    fields=("rho", "e"),
    title=f"Numerical fluxes compared (Sod, {nx} cells)",
    figsize=(9.0, 3.8),
)
# Zoom onto the contact, where the fluxes differ.
for ax in axes[:2]:
    ax.set_xlim(0.6, 0.85)

print("\nwrote", save_figure(fig, "fluxes_example.png"))
print(
    "\nAUSM+ runs the same problem once the gas is moving, e.g.\n"
    "    solve_roe(1.4, [1, 0.5, 1], [0.125, 0.5, 0.1], 0.25, flux='ausm')"
)
