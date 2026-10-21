# Approximate Riemann Solver

[![CI](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/tests.yml/badge.svg)](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/tests.yml)
[![Docs build](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/docs.yml/badge.svg)](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/docs.yml)
[![Lint](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/lint.yml/badge.svg)](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/lint.yml)
[![Formatting](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/formatting.yml/badge.svg)](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/formatting.yml)
[![Spelling](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/spelling.yml/badge.svg)](https://github.com/mrodrig6/approximate_riemann_solver/actions/workflows/spelling.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Solve one-dimensional shock-tube and dam-break problems in Python, comparing exact
solutions with finite-volume approximations. The package supports ideal-gas Euler flow
and shallow water on a flat bed, for education and numerical experiments.

![Sod shock tube: exact density, velocity, pressure and internal energy compared with a second-order Roe calculation](docs/figures/sod_comparison.png)

- Exact Euler Riemann solutions with intermediate pressure and wave sampling.
- Six Euler numerical fluxes: Roe, HLLC, HLL, AUSM+-up, Rusanov and Lax–Friedrichs.
- Limited second-order TVD updates and the one-step Euler ADER2 scheme.
- Exact and approximate shallow-water solutions on wet and dry beds.
- Built-in reference problems, plotting helpers and optional Numba acceleration.

Use custom initial fields through the Python API, or add fluxes, limiters and systems
through the documented [integration points](docs/api/extending.md).

## Install

Requires Python 3.9 or newer. From the repository root:

```bash
python -m pip install -e '.[all]'
```

`all` includes acceleration, plotting and tests; see
[installation](docs/getting-started/installation.md) for smaller extras and
documentation tools.

## Quick start

```bash
# Compare exact and numerical Sod solutions, then save a figure.
python -m src sod --plot sod.png
# Change the numerical flux.
python -m src sod --flux hllc
# Run the near-vacuum case with HLLC.
python -m src toro2 --flux hllc
# Select the one-step ADER2 scheme.
python -m src sod --scheme ader2
# Run wet- and dry-bed dam breaks from Python.
python examples/shallow_water_dam_break.py
```

```python
from src import solve_exact, solve

left, right = [1.0, 0.0, 1.0], [0.125, 0.0, 0.1]
exact = solve_exact(1.4, left, right, 0.25)
numerical = solve(1.4, left, right, 0.25, flux="hllc")
assert abs(exact.p_star - 0.30313018) < 1e-8
```

Follow the [complete quick start](docs/getting-started/quick-start.md) for recorded
output and an explanation of each line.

## Documentation

| Section         | Pages                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Getting started | [Installation](docs/getting-started/installation.md) · [First shock tube](docs/getting-started/quick-start.md) · [Recipes](docs/getting-started/examples.md)                                                                                                                                                                                                                                                                                                                                                                 |
| Running         | [Command line](docs/running/command-line.md) · [Configuration](docs/running/configuration.md) · [Output and plotting](docs/running/output.md) · [Euler workflows](docs/running/euler.md) · [Shallow-water workflows](docs/running/shallow-water.md) · [Performance](docs/running/performance.md) · [Troubleshooting](docs/running/troubleshooting.md)                                                                                                                                                                        |
| Python API      | [Python API](docs/api/package.md) · [Extending the package](docs/api/extending.md)                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Model           | [Euler model](docs/model/euler.md) · [Shallow-water model](docs/model/shallow-water.md) · [Parameters and notation](docs/model/parameters.md) · [Problem catalogues](docs/model/problems.md)                                                                                                                                                                                                                                                                                                                                 |
| Method          | [Exact-solver algorithms](docs/method/exact-solvers.md) · [Finite-volume TVD method](docs/method/finite-volume.md) · [Euler numerical fluxes](docs/method/fluxes.md) · [ADER2](docs/method/ader.md) · [Shallow-water numerical method](docs/method/shallow-water.md)                                                                                                                                                                                                                                                         |
| Theory          | [Exact Euler Riemann solution](docs/theory/exact-euler.md) · [Roe linearisation and entropy](docs/theory/roe.md) · [Exact shallow-water solution](docs/theory/exact-shallow-water.md)                                                                                                                                                                                                                                                                                                                                        |
| Validation      | [Test suite](docs/validation/tests.md) · [Sod benchmark](docs/validation/sod.md) · [Toro reference cases](docs/validation/toro.md) · [Dimensional shock](docs/validation/dimensional.md) · [Limiter comparison](docs/validation/limiters.md) · [Entropy-fix comparison](docs/validation/entropy-fix.md) · [Convergence](docs/validation/convergence.md) · [Flux comparison](docs/validation/flux-comparison.md) · [ADER comparisons](docs/validation/ader.md) · [Shallow-water validation](docs/validation/shallow-water.md) |
| Reference       | [Compatibility](docs/reference/compatibility.md) · [Code map](docs/reference/code-map.md) · [Repository map](docs/reference/repository.md) · [CI checks](docs/reference/ci.md) · [Documentation maintenance](docs/reference/documentation.md) · [Limits and future work](docs/reference/limitations.md) · [References and citation](docs/reference/bibliography.md)                                                                                                                                                          |

Managed by the Rodriguez Flow Research group at Brown University; developer: Mauro
Rodriguez Jr. The methods follow E. F. Toro's textbooks; see
[references and citation](docs/reference/bibliography.md).

## License

[MIT](LICENSE).
