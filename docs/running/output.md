# Output and plotting

The numerical APIs return data objects; plotting is optional. `src.cli.main` prints a
summary, compares the numerical and exact solutions, and passes their arrays to
`src.core.plotting.plot_comparison` when a figure is requested.

| Result                                       | Fields and meaning                                                                                                                              |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Euler numerical `EulerResult`                | `x`, `rho`, `u`, `p`, `e`, `conserved`, `t`, `steps`, `gamma`, `cfl`, `dx`, `scheme`, `flux`, `limiter`, `second_order`, `entropy_fix`, `label` |
| Euler exact `ExactSolution`                  | Sample coordinates/primitive fields, `p_star`, `u_star`, `iterations`, wave labels, `gamma`, `t`, `primitive`                                   |
| Shallow-water numerical `ShallowWaterResult` | `x`, `h`, `u`, `conserved`, `discharge`, `t`, `steps`, `gravity`, `cfl`, `dx`, flux/limiter/options and `label`                                 |
| Shallow-water exact `ShallowWaterSolution`   | `x`, `h`, `u`, intermediate depth/velocity, wave labels, `dry`, `iterations`, `t`, `gravity`, `discharge`, `froude`                             |

Numerical samples are finite-volume cell values. Exact samples are point values at those
cell centres. `conserved` contains density/momentum/energy for Euler, or depth/discharge
for shallow water. Dry exact solutions use `NaN` for the nonexistent intermediate state,
not for the sampled physical depth field.

## Error report

For each Euler field, the CLI computes `mean(abs(numerical - exact))`. The reported
relative L1 divides that mean by `max(abs(exact))` (with a `1e-30` floor). It is not a
percentage, an integral without normalization by domain length, or an average of
pointwise relative errors. Both solutions are evaluated at the same cell centres for
this comparison.

## Figures

`plot_comparison` plots Euler density, velocity, pressure, and specific internal energy
by default. `plot_shallow_water` plots depth and velocity. `save_figure` saves a
Matplotlib figure to a path; choose the format with its suffix. Relative output paths
are relative to the process's working directory. These commands do not save a CSV,
checkpoint, animation, or simulation restart file.

Use `--plot sod.png` to save a CLI comparison. `--dimensional` changes plot labels to SI
units without converting data. Plot labels use Matplotlib mathtext and do not require
TeX. For headless machines, set `MPLBACKEND=Agg` and omit `--show`.

See the [API examples](../api/package.md) and
[quick-start transcript](../getting-started/quick-start.md). The
[documentation guide](../reference/documentation.md) explains committed figure sources.
