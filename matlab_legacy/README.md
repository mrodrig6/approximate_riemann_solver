# MATLAB implementation (archived)

The original MATLAB sources, kept for provenance. The maintained
implementation is the Python package in [`../riemann/`](../riemann/); see
[`../docs/`](../docs/index.md).

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
