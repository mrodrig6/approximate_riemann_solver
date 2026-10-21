# Dimensional shock

This case checks the same algorithms with SI inputs and a pressure near 100 kPa. The
reference is a single-shock jump-relation construction, not independent experimental
data.

## A dimensional case

All the cases above are non-dimensional. `air_shock_tube` is a Mach $\approx 1.35$ shock
running into air at standard conditions, in SI units, output at $t = 0.3$ ms.

![Air shock tube in SI units](../figures/air_shock_tube.png)

The left state was constructed as the exact post-shock state of a right-running shock,
so the left wave is a null wave and the solution is essentially a single shock —
captured here in one or two cells.

This case is worth keeping precisely because of its units. A fixed initial pressure
guess, or an *absolute* Newton tolerance, would be meaningless at $p \approx 10^{5}$ Pa.
Toro's adaptive initial guess and a relative convergence criterion are both scale-free,
so the same code runs this case and Sod's without any tuning — here in 3 Newton
iterations.

## Quantitative agreement

`examples/dimensional_shock_tube.py` prints p-star = 238557.89 Pa and u-star = 225.8943
m/s in 3 Newton iterations.
`tests/euler/test_exact.py::test_isolated_shock_is_stationary_in_its_own_frame` checks
these values against the left post-shock state to relative 1e-6. The numerical profile
uses 42 steps at 200 cells; its relative density, velocity, and pressure errors are
9.9353e-4, 2.4710e-3, and 1.1768e-3. Recompute with `scripts/record_validation.py`; this
is agreement with the shock construction, not an experimental validation.

## Figure source

Rebuild these images with `python scripts/make_figures.py` from the repository root.
Source lives beside the images in [make_figures.py](../figures/make_figures.py):
`figure_air_shock_tube`.
