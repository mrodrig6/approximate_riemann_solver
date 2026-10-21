# Command line

`src.cli.main` runs one built-in Euler problem with exact and numerical solvers. It
prints an error report and optionally saves or displays a figure; custom states and
shallow water use the [Python API](../api/package.md).

| Command                   | Purpose                                                                    |
| ------------------------- | -------------------------------------------------------------------------- |
| `python -m src [problem]` | Run an Euler problem; defaults to `sod`                                    |
| `riemann [problem]`       | Installed console entry point for the same function                        |
| `python -m src --help`    | List options, problem names, and flux aliases                              |
| `python -m src --version` | Print the API version; see [version mismatch](../reference/limitations.md) |

All flags belong to the run command. There are no subcommands or configuration files.

| Flag                | Default         | Applies to / effect                                                                |
| ------------------- | --------------- | ---------------------------------------------------------------------------------- |
| `problem`           | `sod`           | Any built-in Euler catalogue key                                                   |
| `--nx NX`           | `200`           | Numerical grid, at least 4 cells                                                   |
| `--cfl CFL`         | `0.9`           | Time step; `0 < CFL <= 1`                                                          |
| `--flux FLUX`       | `roe`           | TVD only; valid name still required under ADER                                     |
| `--limiter NAME`    | `superbee`      | TVD correction or ADER slope; `minmod`, `none`, `superbee`, `vanalbada`, `vanleer` |
| `--scheme NAME`     | `tvd`           | `tvd` or `ader2`                                                                   |
| `--first-order`     | Off             | TVD only; suppress correction for the selected flux                                |
| `--no-entropy-fix`  | Off             | Roe with TVD only                                                                  |
| `--t-end T_END`     | Catalogue value | Override output time; must be positive                                             |
| `--n-exact N_EXACT` | `1000`          | Exact curve resolution; error report still samples on `nx` cells                   |
| `--plot PATH`       | None            | Write figure using Matplotlib                                                      |
| `--show`            | Off             | Open plot window; requires a graphical session                                     |
| `--dimensional`     | Off             | SI plot labels, intended for `air_shock_tube`; does not rescale inputs             |
| `--help`, `-h`      | —               | Print help and exit                                                                |
| `--version`         | —               | Print version and exit                                                             |

Canonical Euler fluxes are `roe`, `hll`, `hllc`, `ausmup`, `rusanov`, and
`lax_friedrichs`. Aliases include `hlle` → HLL, `llf` → Rusanov, `lf` → LF, and `ausm` →
AUSM+-up. `src.euler.fluxes.registry.flux_code` ignores spaces, underscores, hyphens,
plus signs, and letter case. See [flux formulas](../method/fluxes.md).

## Exit status

| Status | Meaning                                       |
| ------ | --------------------------------------------- |
| `0`    | Run, help, or version completed               |
| `1`    | Reported numerical positivity failure         |
| `2`    | Unknown problem/flux or argument-parser error |

Some invalid physical inputs raise uncaught `ValueError`, and a step-limit failure
raises `RuntimeError`; these are not converted into the friendly positivity report. See
[troubleshooting](troubleshooting.md). The [catalogue](../model/problems.md) lists every
problem's states and output time.
