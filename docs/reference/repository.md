# Repository map

This table maps the top-level files and directories to their jobs. The
[code map](code-map.md) lists module ownership and solver calls.

| Path                 | Purpose                                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `README.md`          | Short project front door and website home source                                                |
| `pyproject.toml`     | Packaging, extras, console entry point, pytest and spelling configuration                       |
| `src/`               | Public API, CLI, numerical systems and shared helpers                                           |
| `tests/`             | Existing numerical and CLI tests                                                                |
| `examples/`          | Five executable demonstrations                                                                  |
| `scripts/`           | Benchmark, math checker, example runner, documentation checks and compatible figure entry point |
| `docs/`              | Eight documentation sections, site hook, assets and figure sources                              |
| `docs/figures/`      | Committed images and their generator                                                            |
| `mkdocs.yml`         | Material theme, navigation, strict validation, math and diagram support                         |
| `.github/workflows/` | Tests, lint, formatting, spelling and docs build; no deployment                                 |
| `.gitignore`         | Local caches, generated example plots and site output                                           |
| `.mdformat.toml`     | Markdown formatting configuration                                                               |
| `ruff.toml`          | Lint configuration for documentation tooling                                                    |
| `LICENSE`            | MIT license                                                                                     |

The Python package is named `src`; it is not a conventional source-root folder to strip
from imports. Its original MATLAB implementation is archived on the
[matlab-legacy branch](https://github.com/mrodrig6/approximate_riemann_solver/tree/matlab-legacy).
