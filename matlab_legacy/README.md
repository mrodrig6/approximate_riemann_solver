# MATLAB implementation (archived)

The original MATLAB sources, kept for provenance on this branch alone: they
were removed from `master`, where the maintained implementation is the Python
package in
[`src/`](https://github.com/mrodrig6/approximate_riemann_solver/tree/master/src).
See the
[documentation index](https://github.com/mrodrig6/approximate_riemann_solver/blob/master/docs/index.md).

This branch holds nothing else, and is not merged back into `master`. The
full history of these files is intact, so `git log` and `git log --follow`
work as they always did.

```
main.m                                 driver for Sod's problem
test_input2o1.m                        driver for the dimensional air shock tube
src/exact_solver/f_ps.m                pressure function f(p)
src/exact_solver/fp_ps.m               its derivative
src/exact_solver/root_find.m           Newton-Raphson for p_star
src/exact_solver/plot_rp.m             sample and plot the exact solution
src/approximate_solver/roe_flux_e.m    Roe flux with entropy fix
src/approximate_solver/flx_lim.m       flux limiter
src/approximate_solver/plot_roe_eue.m  time loop and plot
```
