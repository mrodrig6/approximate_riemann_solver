# Euler workflows

Use the CLI for the built-in shock tubes and Python for custom Euler states or smooth
initial fields. `src.euler.solver.solve` constructs a two-state field and delegates to
`solve_field`.

| Goal                               | Selection                          | Restriction                                            |
| ---------------------------------- | ---------------------------------- | ------------------------------------------------------ |
| Standard first check               | `python -m src sod`                | Default Roe/TVD/superbee                               |
| General catalogue runs             | `python -m src toro2 --flux hllc`  | Tested catalogue, not universal positivity             |
| Sharp isolated shock               | Roe with entropy fix               | Roe fails near vacuum                                  |
| Diffusive two-wave baseline        | HLL                                | Smears Euler contacts                                  |
| Cheap split flux                   | AUSM+-up                           | Use first order for `toro2`, `toro3`, `toro5`          |
| Cheap central flux                 | Rusanov                            | `toro5` fails with the default second-order correction |
| Teaching baseline                  | Lax–Friedrichs                     | Odd/even decoupling can cancel TVD correction          |
| One-step higher-order construction | `python -m src sod --scheme ader2` | Second order only; `toro2` and `toro5` fail            |
| Smooth convergence experiment      | Python `solve_field`               | Use smooth inputs; shocks conceal formal order         |

Roe remains the default for backward compatibility; HLLC is often a useful first
alternative. The four TVD limiters trade diffusion against compression; `none` disables
limiting and can oscillate. Choose according to the [method](../method/finite-volume.md)
and [comparison](../validation/limiters.md).

See [compatibility](../reference/compatibility.md) before choosing a flux for a strong
jump, and [troubleshooting](troubleshooting.md) for observed failures. WENO is a
possible future companion for AUSM, not an option provided here.
