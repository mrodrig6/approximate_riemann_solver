# Original MATLAB implementation (archived)

These are the original MATLAB sources, kept verbatim for provenance and for
anyone who wants to compare the port against them.  **They are no longer the
maintained implementation** -- see the Python package in `riemann/` at the
repository root.

```
main.m                        driver for Sod's problem
test_input2o1.m               driver for the dimensional air shock tube
src/exact_solver/f_ps.m       pressure function f(p)
src/exact_solver/fp_ps.m      its derivative
src/exact_solver/root_find.m  Newton-Raphson for p_star
src/exact_solver/plot_rp.m    sample + plot the exact solution
src/approximate_solver/roe_flux_e.m   Roe flux with entropy fix
src/approximate_solver/flx_lim.m      flux limiter
src/approximate_solver/plot_roe_eue.m time loop + plot
```

The Python port is not a line-by-line translation: it fixes several bugs
present in these files, and separates computation from plotting.  Both the
file-by-file mapping and the full list of corrections are in
[`../docs/porting-notes.md`](../docs/porting-notes.md).
