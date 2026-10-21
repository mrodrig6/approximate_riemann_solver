# Flux comparison

These runs hold the Sod data, grid, CFL, and limiter fixed while changing the base flux.
Accuracy claims apply to this benchmark, not an ordering for every flow.

## The numerical fluxes compared

Six fluxes, the same problem, the same limiter, the same grid — only the first-order
flux differs. The contact separates the fluxes clearly. Shock width also depends on flux
and settings; it is not limited to one or two cells for every flux.

![Numerical fluxes at the contact discontinuity](../figures/flux_comparison.png)

| Flux           | Contact width (100 cells) | $L_1$ error in $\rho$ (200 cells) |
| -------------- | ------------------------- | --------------------------------- |
| Roe            | 2 cells                   | $1.44 \times 10^{-3}$             |
| HLLC           | 3 cells                   | $2.10 \times 10^{-3}$             |
| AUSM+-up       | 4 cells                   | $6.59 \times 10^{-3}$             |
| HLL            | 8 cells                   | $4.04 \times 10^{-3}$             |
| Rusanov        | 19 cells                  | $1.28 \times 10^{-2}$             |
| Lax-Friedrichs | 23 cells                  | $2.22 \times 10^{-2}$             |

Full discussion, including the AUSM+-up coefficients adapted from EMS and why
Lax-Friedrichs ignores the TVD correction entirely, is in
[Euler flux methods](../method/fluxes.md).

## Reproduce the measurement

Run `python scripts/record_validation.py > /tmp/riemann-results.json`. The table uses
200 cells for density error and 100 for width, with superbee and CFL 0.9. Width counts
cells in `0.6 < x < 0.85` whose density lies between 5% and 95% of the interval
`[0.2656, 0.4263]`. This matches `examples/compare_fluxes.py` and the figure source. The
measured LF width is 23; the old 24-cell count used a different spatial mask.

## Figure source

Rebuild these images with `python scripts/make_figures.py` from the repository root.
Source lives beside the images in [make_figures.py](../figures/make_figures.py):
`figure_flux_comparison`.
