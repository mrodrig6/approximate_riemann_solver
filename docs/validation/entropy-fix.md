# Entropy-fix comparison

This comparison isolates the Harten–Hyman correction using first-order Roe on a
transonic rarefaction. Refinement distinguishes an entropy defect from ordinary grid
error.

## The entropy fix

The transonic case of [Roe theory](../theory/roe.md#the-entropy-fix). The initial data
are chosen so the left rarefaction fan straddles the diaphragm and contains a sonic
point, where $u - a$ changes sign inside the wave. Both runs are first order, so that
the effect is not masked by the limiter.

![Entropy glitch with and without the Harten-Hyman fix](../figures/entropy_fix.png)

Without the fix (orange squares) a step of fixed height sits at the sonic point: the
unphysical "rarefaction shock". The decisive observation is the comparison *between* the
two panels. Quadrupling the resolution from 100 to 400 cells leaves the unfixed error
essentially unchanged, while the fixed solution converges normally:

|                                                           | 100 cells             | 400 cells             |
| --------------------------------------------------------- | --------------------- | --------------------- |
| Max error in $\rho$ across the fan, **with** Harten–Hyman | $4.21 \times 10^{-2}$ | $2.72 \times 10^{-2}$ |
| Max error in $\rho$ across the fan, **without**           | $5.40 \times 10^{-2}$ | $5.86 \times 10^{-2}$ |

An ordinary discretisation error would shrink under refinement. One that does not is a
sign that the scheme is converging to the wrong weak solution — which is precisely what
an entropy-violating scheme does. `tests/euler/test_solver.py` asserts this
non-convergence, so the fix cannot be silently disabled.

## Figure source

Rebuild these images with `python scripts/make_figures.py` from the repository root.
Source lives beside the images in [make_figures.py](../figures/make_figures.py):
`figure_entropy_fix`.
