# Troubleshooting

This page maps solver and CLI errors to their causes and remedies. Text in backticks is
a message or stable message prefix; values such as step counts and input names vary with
the run.

| Error or message                                                                                                  | Cause                                         | Remedy and implementation                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `unknown problem`                                                                                                 | Euler catalogue key not found                 | Use a name from `python -m src --help`; custom states need Python (`src.euler.problems.get_problem`)                         |
| `unknown shallow-water problem`                                                                                   | Wrong shallow-water key                       | Use `src.shallow_water.list_problems()`                                                                                      |
| `unknown flux`, `unknown flux code`                                                                               | Registry entry not found                      | Select a supported flux for the correct system; see compatibility                                                            |
| `unknown limiter`, `unknown limiter code`                                                                         | Limiter name/code not found                   | Use `minmod`, `superbee`, `vanleer`, `vanalbada`, or `none`                                                                  |
| `unknown scheme`                                                                                                  | Euler reconstruction name not supported       | Use `tvd` or `ader2` (`src.euler.solver.solve_field`)                                                                        |
| `gamma must be greater than 1`                                                                                    | Invalid ideal-gas closure                     | Supply a finite physical ratio greater than one; normal air examples use 1.4                                                 |
| `gravity must be positive`                                                                                        | Nonpositive gravity                           | Supply positive gravity (`src.shallow_water.physics.check_gravity`)                                                          |
| `initial states must have positive density and pressure`, `initial field must have positive density and pressure` | Invalid Euler data                            | Check `[rho, u, p]` order and positive density/pressure                                                                      |
| `depths must be non-negative`, `initial field must have non-negative depth`                                       | Invalid shallow-water data                    | Check `[h, u]` order and nonnegative depth                                                                                   |
| `w_init must have shape (nx, 3)`, `w_init must have shape (nx, 2)`                                                | Wrong state array shape                       | One primitive state per row; Euler has 3 columns, shallow water 2                                                            |
| `nx must be at least 4`                                                                                           | Too few numerical cells                       | Increase `nx`                                                                                                                |
| `t_end must be positive`, `output time t must be positive`                                                        | Zero or negative requested time               | Supply positive time; the exact fan uses position divided by time                                                            |
| `cfl must lie in (0, 1]`                                                                                          | Invalid CFL                                   | Use a number in the accepted interval; this alone does not guarantee positivity                                              |
| `initial pressure guess must be positive`                                                                         | Invalid `p_guess`                             | Omit it to use the adaptive guess, or give positive pressure                                                                 |
| `vacuum`                                                                                                          | Euler separation exceeds the vacuum threshold | The exact Euler implementation rejects this input; see exact Euler theory                                                    |
| `dry bed` in `solve_star_depth`                                                                                   | No wet intermediate state                     | Call shallow-water `solve_exact` for dry sampling instead of requesting a star depth                                         |
| `both sides are dry; there is nothing to solve`                                                                   | Exact shallow-water data have no water        | The numerical solver supports an all-dry field; there is no exact fan to calculate                                           |
| `Newton-Raphson for p_star did not converge`, `Newton-Raphson for h_star did not converge`                        | Scalar iteration cap exhausted                | Inspect physical inputs and units, initial guess and tolerance; do not assume a result was obtained                          |
| `numerical run FAILED` / `produced a non-physical state`                                                          | Euler positivity guard failed                 | Roe near vacuum: choose HLLC/HLL. AUSM or Rusanov TVD failure: try first order. ADER failure: switch to TVD/HLLC; see matrix |
| `produced a negative depth`                                                                                       | Wet/dry update failed                         | Use catalogue CFL, about 0.45 for dry cases, or first order (`src.shallow_water.solver`)                                     |
| `solution became non-finite`                                                                                      | An update produced NaN or infinity            | Inspect initial states, units, flux/reconstruction pairing; reduce complexity and compare exact solution                     |
| `reached the ...-step cap`                                                                                        | `max_steps` exhausted before requested time   | Inspect speed and scale; increase cap only after diagnosing why so many steps are needed                                     |
| Matplotlib import error                                                                                           | Plot extra missing                            | Install `.[plot]`; solver-only operation does not need Matplotlib                                                            |
| Display or font-cache error                                                                                       | Headless session or unwritable cache          | Omit `--show`, use `MPLBACKEND=Agg`, and select writable `MPLCONFIGDIR`/`XDG_CACHE_HOME`                                     |

## Observed Roe failure

The following excerpt is copied from `python -m src toro2` (exit 1), using the default
grid and CFL. It is a known limitation asserted by the tests, not an installation
failure.

```text
numerical run FAILED:
roe produced a non-physical state (negative pressure or density) at step 1, t = 0.00163736.
```

Run `python -m src toro2 --flux hllc` to complete this catalogue case. Roe also fails at
first order on these data. The second-order TVD correction can fail even when its base
flux is positivity preserving: a smaller CFL is not a general remedy for that
correction. ADER has different reconstruction limits; its error advice does not imply
that a limiter change rescues the documented `toro2` and `toro5` failures.

See [compatibility](../reference/compatibility.md),
[ADER validation](../validation/ader.md), and
[exact Euler theory](../theory/exact-euler.md). The CLI catches `FloatingPointError`;
other invalid-input exceptions can still display a traceback.
