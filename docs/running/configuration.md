# Configuration

The package uses function arguments and CLI flags, not input files or a
configuration-file hierarchy. `src.euler.problems` and `src.shallow_water.problems` hold
reusable built-in data.

| Source                        | Selection and precedence                                                               |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| Python function signature     | Supplies defaults for omitted keyword arguments                                        |
| Python explicit argument      | Overrides that function's default                                                      |
| CLI `problem`                 | Selects catalogue physics, states, geometry, and output time                           |
| CLI flags                     | Override grid, CFL, flux, limiter, scheme, output time, and plotting options           |
| Shallow-water catalogue `cfl` | Must be passed explicitly when using `solve`; it is not inferred from states           |
| Environment                   | `NUMBA_DISABLE_JIT=1` disables compilation; `MPLBACKEND=Agg` selects headless plotting |

The CLI uses its own `--cfl=0.9` default for Euler. `--t-end` takes precedence over the
catalogue output time. It does not read parameter files or expose flags for `gamma`,
initial states, domain bounds, diaphragm position, gravity, or `max_steps`; use Python
for those.

All parameters, defaults, shapes, and units are listed once in
[parameters and notation](../model/parameters.md). ADER ignores some TVD options; see
the [flag applicability table](command-line.md).
