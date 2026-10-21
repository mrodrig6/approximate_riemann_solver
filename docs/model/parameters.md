# Parameters and notation

This table gives the code names, defaults, and units of the two numerical Riemann
solvers. Use a consistent unit system; Euler catalogue cases are nondimensional except
the SI air case.

| Code name      | System        | Default      | Meaning or units                                            |
| -------------- | ------------- | ------------ | ----------------------------------------------------------- |
| `gamma`        | Euler         | `Required`   | Dimensionless                                               |
| `w_l`          | Euler         | `Required`   | Euler: [density, velocity, pressure]; SW: [depth, velocity] |
| `w_r`          | Euler         | `Required`   | Same as w_l                                                 |
| `t_end`        | Euler         | `Required`   | Time                                                        |
| `x_left`       | Euler         | `0.0`        | Length                                                      |
| `x_right`      | Euler         | `1.0`        | Length                                                      |
| `nx`           | Euler         | `200`        | Cell count                                                  |
| `x_diaphragm`  | Euler         | `0.5`        | Length                                                      |
| `cfl`          | Euler         | `0.9`        | Dimensionless                                               |
| `flux`         | Euler         | `'roe'`      | Name or integer code                                        |
| `limiter`      | Euler         | `'superbee'` | Name or integer code                                        |
| `scheme`       | Euler         | `'tvd'`      | Name                                                        |
| `second_order` | Euler         | `True`       | Boolean                                                     |
| `entropy_fix`  | Euler         | `True`       | Boolean                                                     |
| `max_steps`    | Euler         | `200000`     | Update count                                                |
| `w_l`          | Shallow water | `Required`   | Euler: [density, velocity, pressure]; SW: [depth, velocity] |
| `w_r`          | Shallow water | `Required`   | Same as w_l                                                 |
| `t_end`        | Shallow water | `Required`   | Time                                                        |
| `gravity`      | Shallow water | `9.81`       | Length/time²                                                |
| `x_left`       | Shallow water | `0.0`        | Length                                                      |
| `x_right`      | Shallow water | `1.0`        | Length                                                      |
| `nx`           | Shallow water | `200`        | Cell count                                                  |
| `x_diaphragm`  | Shallow water | `0.5`        | Length                                                      |
| `cfl`          | Shallow water | `0.9`        | Dimensionless                                               |
| `flux`         | Shallow water | `'hll'`      | Name or integer code                                        |
| `limiter`      | Shallow water | `'superbee'` | Name or integer code                                        |
| `second_order` | Shallow water | `True`       | Boolean                                                     |
| `entropy_fix`  | Shallow water | `True`       | Boolean                                                     |
| `max_steps`    | Shallow water | `200000`     | Update count                                                |

`solve_field` takes `w_init` instead of two states, infers `nx` from its shape, and has
no diaphragm argument. Euler fields have shape `(nx, 3)`; shallow-water fields have
shape `(nx, 2)`. Exact solvers use `t` for time, `n=1000` for sample count, `tol=1e-12`,
and Euler optionally accepts `p_guess=None`. See
[exact algorithms](../method/exact-solvers.md).

| Symbol / code                                          | Meaning                                                | Units                                     |
| ------------------------------------------------------ | ------------------------------------------------------ | ----------------------------------------- |
| $\rho$ / `rho`                                         | Density                                                | Mass/length³                              |
| $u$ / `u`                                              | Velocity                                               | Length/time                               |
| $p$ / `p`                                              | Pressure                                               | Mass/(length time²)                       |
| $E$, $e$                                               | Total energy density; specific internal energy         | Energy/volume; energy/mass                |
| $a$, $H$                                               | Sound speed; specific total enthalpy                   | Length/time; energy/mass                  |
| $\gamma$ / `gamma`                                     | Ratio of specific heats, greater than one              | Dimensionless                             |
| $p_{\ast}$, $u_{\ast}$                                 | Intermediate pressure and velocity                     | Pressure; velocity                        |
| $\tilde{u}$, $\tilde{H}$, $\tilde{a}$                  | Roe-averaged quantities                                | As their physical counterparts            |
| $\tilde{\lambda}_k$, $\tilde{K}_k$, $\tilde{\alpha}_k$ | Characteristic speed, right eigenvector, wave strength | See [Roe decomposition](../theory/roe.md) |
| $\nu_k$, $\phi(r)$                                     | Wave Courant number; limiter                           | Dimensionless                             |
| $S_L$, $S_R$, $S_{\ast}$                               | HLL/HLLC wave speeds                                   | Length/time                               |
| $M^\pm_{(4)}$, $P^\pm_{(5)}$                           | AUSM Mach/pressure split functions                     | Dimensionless                             |
| $K_p$, $K_u$                                           | AUSM diffusion coefficients                            | Dimensionless; 0.4 and 0.2                |
| $h$ / `h`                                              | Depth                                                  | Length                                    |
| $g$ / `gravity`                                        | Gravitational acceleration                             | Length/time²; default 9.81                |
| `DRY_TOLERANCE`                                        | Dry-depth threshold                                    | Depth units; `1e-10`                      |

Plotting with `dimensional=True` adds SI labels without changing these values. See
[output](../running/output.md).
