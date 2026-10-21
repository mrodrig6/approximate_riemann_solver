# Convergence

Smooth-data runs measure formal order; shock-tube runs measure error in the presence of
discontinuities. A fine-grid acoustic reference is numerical evidence, while the
advected density pulse has an analytic reference.

## Grid convergence

$L_1$ error in density on Sod's problem, from 50 to 1600 cells.

![Grid convergence](../figures/convergence.png)

All three curves have slope $\approx -1$, **not** $-2$, and that is the correct result
rather than a defect. For the formal order, measured on smooth data where it is visible,
see the [smooth-data measurement](#order-of-accuracy). The exact solution contains a
shock and a contact; near a discontinuity any scheme makes an $\mathcal{O}(1)$ error
over an $\mathcal{O}(\Delta x)$-wide region, which contributes $\mathcal{O}(\Delta x)$
to the $L_1$ norm however accurate the scheme is elsewhere. A genuine second-order
*rate* is only observable on a smooth test problem.

What the second-order correction delivers is a smaller constant: the superbee curve sits
about a factor of 6 below the first-order one, so it reaches the same accuracy on a grid
roughly six times coarser.

______________________________________________________________________

## Order of accuracy

Measuring a formal order requires **smooth** initial data. Every problem in the
Riemann-problem catalogue contains a discontinuity, and near one any scheme makes an
$\mathcal{O}(1)$ error over an $\mathcal{O}(\Delta x)$ region — so the $L_1$ rate is
capped at 1 whatever the formal order. That is exactly why the
[grid-convergence figure](convergence.md) on Sod's problem shows slope $-1$ for every
scheme.

`solve_field` exists for this: it advances an arbitrary initial field rather than two
constant states, and `advected_pulse` and `acoustic_pulse` supply smooth data.

![Order of accuracy on a smooth acoustic pulse](../figures/order_of_accuracy.png)

| Problem                | Reference | 1st order | TVD   | ADER2 |
| ---------------------- | --------- | --------- | ----- | ----- |
| Advected density pulse | analytic  | 0.854     | 1.970 | 1.970 |
| Smooth acoustic pulse  | fine grid | 0.960     | 1.989 | 2.001 |

The limiter is switched off for these measurements: a limiter clips smooth extrema,
which caps the observed rate below the formal one.

## Measurement settings

These rates were fitted to density mean absolute errors at 50, 100, 200, and 400 cells
with CFL 0.8 and `limiter="none"`. Advection uses the analytic translated profile at
t=0.2; acoustics uses a 3200-cell ADER2 reference at t=0.15.
`scripts/record_validation.py` emits the errors and log-log least-squares rates.
First-order advection is approaching its asymptotic rate on this grid range, so
reporting exactly 1.00 would overstate the measurement. The figure uses an additional
800-cell point and a 6400-cell acoustic reference; its settings are recorded in
`figure_order_of_accuracy`.

## Figure source

Rebuild these images with `python scripts/make_figures.py` from the repository root.
Source lives beside the images in [make_figures.py](../figures/make_figures.py):
`figure_convergence`, `figure_order_of_accuracy`.
