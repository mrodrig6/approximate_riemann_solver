# Python API

The public `src` package exports the Euler solver and shared plotting tools. Import
`src.shallow_water` explicitly for the second physical system; the two `solve` functions
take different arguments.

```python
from src import get_problem, solve_exact, solve, plot_comparison, save_figure

problem = get_problem("sod")
exact = solve_exact(problem.gamma, problem.left, problem.right, problem.t_end)
numerical = solve(problem.gamma, problem.left, problem.right, problem.t_end,
                  flux="hllc", nx=200)
print(f"p* = {exact.p_star:.8f}, u* = {exact.u_star:.8f}")
fig, _ = plot_comparison(exact=exact, numerical=[numerical])
save_figure(fig, "python-sod.png")
```

Recorded output:

```text
p* = 0.30313018, u* = 0.92745262
```

## Entry points

| Import                                                                                        | Purpose                                                           |
| --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `src.solve`, `src.solve_field`                                                                | Euler Riemann data or arbitrary initial field                     |
| `src.solve_exact`, `src.solve_star_pressure`, `src.sample_exact`                              | Euler exact profile, scalar pressure solve, self-similar sampling |
| `src.get_problem`, `src.list_problems`, `src.PROBLEMS`                                        | Euler catalogue                                                   |
| `src.cell_centres`, `src.advected_pulse`, `src.acoustic_pulse`                                | Initial-field helpers                                             |
| `src.primitive_to_conservative`, `src.conservative_to_primitive`                              | Euler state conversion                                            |
| `src.FLUXES`, `src.ROBUST_FLUXES`, `src.flux_code`, `src.interface_fluxes`, `src.roe_average` | Euler flux selection and low-level kernels                        |
| `src.LIMITERS`, `src.limiter_phi`                                                             | Shared limiter catalogue and evaluation                           |
| `src.HAVE_NUMBA`, `src.numba_version`                                                         | Installed acceleration availability                               |
| `src.plot_comparison`, `src.plot_shallow_water`, `src.save_figure`                            | Plotting and saving                                               |
| `src.shallow_water.solve`, `solve_field`, `solve_exact`, `solve_star_depth`                   | Shallow-water numerical/exact APIs                                |
| `src.shallow_water.get_problem`, `list_problems`, `PROBLEMS`                                  | Shallow-water catalogue                                           |

Result classes are `src.EulerResult`, `src.ExactSolution`,
`src.shallow_water.ShallowWaterResult`, and `src.shallow_water.ShallowWaterSolution`.
Catalogue classes are `RiemannProblem` and `ShallowWaterProblem`. Fields and units are
in [output](../running/output.md) and [parameters](../model/parameters.md).

## Solver signatures

These signatures correspond to the current source. Import the shallow-water module
rather than shadowing the Euler names when using both systems.

```text
src.solve(gamma, w_l, w_r, t_end, x_left=0.0, x_right=1.0, nx=200, x_diaphragm=0.5, cfl=0.9, flux='roe', limiter='superbee', scheme='tvd', second_order=True, entropy_fix=True, max_steps=200000)
```

```text
src.solve_field(gamma, w_init, t_end, x_left=0.0, x_right=1.0, cfl=0.9, flux='roe', limiter='superbee', scheme='tvd', second_order=True, entropy_fix=True, max_steps=200000)
```

```text
src.solve_exact(gamma, w_l, w_r, t, x_left=0.0, x_right=1.0, n=1000, x_diaphragm=0.5, tol=1e-12, p_guess=None)
```

```text
src.shallow_water.solve(w_l, w_r, t_end, gravity=9.81, x_left=0.0, x_right=1.0, nx=200, x_diaphragm=0.5, cfl=0.9, flux='hll', limiter='superbee', second_order=True, entropy_fix=True, max_steps=200000)
```

```text
src.shallow_water.solve_field(w_init, t_end, gravity=9.81, x_left=0.0, x_right=1.0, cfl=0.9, flux='hll', limiter='superbee', second_order=True, entropy_fix=True, max_steps=200000)
```

```text
src.shallow_water.solve_exact(w_l, w_r, t, gravity=9.81, x_left=0.0, x_right=1.0, n=1000, x_diaphragm=0.5, tol=1e-12)
```

See [shallow-water workflows](../running/shallow-water.md) for wet/dry data and the
[extension example](extending.md) for a smooth custom field. Euler
`RiemannProblem.kwargs()` includes `t_end` for the numerical solver; the exact API
instead takes `t`. Avoid blindly forwarding that dictionary to `solve_exact`.
