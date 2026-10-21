# Installation

Install the package from a checkout with Python 3.9 or newer. NumPy is required;
acceleration, plotting, and tests are optional extras declared in `pyproject.toml`.

## Install from source

```bash
git clone https://github.com/mrodrig6/approximate_riemann_solver.git
cd approximate_riemann_solver
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[all]'
```

These shell commands use Bash. On Windows, activate `.venv` using its `Scripts`
directory instead. Editable installation keeps Python imports pointed at the checkout.

| Extra    | Adds                         | When needed                              |
| -------- | ---------------------------- | ---------------------------------------- |
| No extra | NumPy                        | Solver API and CLI without plots         |
| `fast`   | Numba                        | Compiled numerical kernels               |
| `plot`   | Matplotlib                   | Plot functions and `--plot`              |
| `test`   | pytest                       | Test suite                               |
| `all`    | Numba, Matplotlib, pytest    | Full numerical development workflow      |
| `docs`   | Material MkDocs              | Build or serve documentation             |
| `dev`    | Ruff, mdformat and codespell | Documentation lint, formatting, spelling |

For documentation work, use Python 3.12 and run
`python -m pip install -e '.[all,docs,dev]'`; see the
[tooling compatibility](../reference/compatibility.md#python-and-optional-acceleration).
There is no additional setup script, database, service, or credential requirement. The
repository has no dependency lockfile: compatible dependency releases are selected for
your Python version. See [compatibility](../reference/compatibility.md).

## Verify

```bash
python -m pip check
python -m pytest -q
python -m src sod
```

On the recorded Linux/Python 3.12 run, `pip check` printed
`No broken requirements found.` and pytest completed with 263 passed. The complete
solver output is explained in the [quick start](quick-start.md). Expected Roe/ADER
numerical failures are asserted within passing tests; see
[test coverage](../validation/tests.md).

For headless plotting use `MPLBACKEND=Agg`. If the home cache is unwritable, point
`XDG_CACHE_HOME` and `MPLCONFIGDIR` to writable directories. No LaTeX installation is
required for figures.
