# Limiter comparison

This comparison changes the limiter while holding the Roe flux, initial data, and grid
fixed. It measures error against the exact Sod solution rather than a saved numerical
result.

## Order of accuracy and the limiters

All three runs below use the identical Roe flux. They differ only in whether the
second-order correction of [finite-volume method](../method/finite-volume.md) is
applied, and with which limiter.

![Effect of the flux limiter, zoomed on the contact](../figures/limiter_comparison.png)

Measured on Sod's problem at 100 cells, counting cells whose density lies strictly
between the two plateau values across the contact:

| Scheme                 | Contact width (cells) | $L_1$ error in $\rho$ |
| ---------------------- | --------------------- | --------------------- |
| First-order Roe        | 14                    | $1.51 \times 10^{-2}$ |
| Second order, minmod   | 7                     | $5.67 \times 10^{-3}$ |
| Second order, van Leer | 4                     | $3.86 \times 10^{-3}$ |
| Second order, superbee | 2                     | $2.62 \times 10^{-3}$ |

The ordering is exactly what the limiter functions predict: superbee sits on the upper
(compressive) edge of Sweby's TVD region and minmod on the lower (diffusive) edge, with
van Leer between them. None of the runs oscillates, which is the TVD property doing its
job.

The shock, by contrast, is captured in one or two cells by *every* variant. That is
Roe's U-property: an isolated shock is an exact solution of the linearised problem, so
there is nothing for the scheme to smear.

## Figure source

Rebuild these images with `python scripts/make_figures.py` from the repository root.
Source lives beside the images in [make_figures.py](../figures/make_figures.py):
`figure_order_and_limiters`.
