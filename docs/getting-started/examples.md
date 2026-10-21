# Recipes

These commands run from the repository root after installation with `[all]`. Each
example writes figures into the current directory; use a scratch directory and absolute
script paths if you want to keep outputs elsewhere.

```bash
# Exact and Roe shock-tube comparison.
python examples/sod_shock_tube.py
# Six base fluxes, with the same limiter and grid.
python examples/compare_fluxes.py
# First order versus limited second order.
python examples/compare_limiters.py
# Dimensional inputs and SI plot labels.
python examples/dimensional_shock_tube.py
# Shallow water on wet and dry beds.
python examples/shallow_water_dam_break.py
```

| Script                                | Files saved                          | Detailed result                                     |
| ------------------------------------- | ------------------------------------ | --------------------------------------------------- |
| `examples/sod_shock_tube.py`          | `sod_example.png`                    | [Sod](../validation/sod.md)                         |
| `examples/compare_fluxes.py`          | `fluxes_example.png`                 | [Flux comparison](../validation/flux-comparison.md) |
| `examples/compare_limiters.py`        | `limiters_example.png`               | [Limiter comparison](../validation/limiters.md)     |
| `examples/dimensional_shock_tube.py`  | `air_shock_tube_example.png`         | [Dimensional shock](../validation/dimensional.md)   |
| `examples/shallow_water_dam_break.py` | `dam_break.png`, `dam_break_dry.png` | [Shallow water](../validation/shallow-water.md)     |

```bash
# Run the near-vacuum case with a flux that survives it.
python -m src toro2 --flux hllc
# Select the one-step second-order scheme.
python -m src sod --scheme ader2
# Use a diffusive limiter with the TVD scheme.
python -m src sod --limiter minmod
# Label the dimensional case in SI units.
python -m src air_shock_tube --dimensional --plot air.png
```

Use the [Python API](../api/package.md) for custom left/right states and
[custom initial fields](../api/extending.md) for smooth data. The
[performance guide](../running/performance.md) covers benchmarking; the
[documentation guide](../reference/documentation.md) covers rebuilding all figures.
