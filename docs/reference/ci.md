# CI checks

Each workflow runs on pushes and pull requests and has read-only repository permissions.
These checks build and test the project; none publishes a website or enables GitHub
Pages.

Run the commands below from the repository root after installing `.[all,docs,dev]`.

| Workflow         | Checks                                                                               | Local equivalent                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests.yml`      | Python 3.9 and latest stable without Numba; Python 3.12 accelerated and disabled JIT | `python -m pytest -q`; `NUMBA_DISABLE_JIT=1 python -m pytest -q`                                                                                              |
| `lint.yml`       | Ruff on documentation tooling; structural math checks                                | `ruff check docs/hooks.py docs/figures/make_figures.py scripts`; `python scripts/check_math.py`                                                               |
| `formatting.yml` | Markdown formatting, without rewriting numerical source                              | `mdformat --check README.md docs`                                                                                                                             |
| `spelling.yml`   | Documentation and tooling spelling                                                   | `codespell`                                                                                                                                                   |
| `docs.yml`       | Executable docs examples, all five scripts, figure generation, strict links/build    | `python scripts/check_docs.py`; `python scripts/run_examples.py`; `python scripts/make_figures.py --output-dir /tmp/riemann-figures`; `mkdocs build --strict` |

The no-Numba jobs install `[plot,test]` in a clean interpreter and assert that Numba is
absent. This exercises the shim's ImportError fallback. The Python 3.12 job installs
`[all]`; disabling JIT there exercises uncompiled kernels with Numba still importable. A
Numba-specific flag test is skipped when Numba is absent, and that skip is expected.

`3.x` asks `actions/setup-python` for the latest stable Python available when the job
runs. This is not a claim that every optional compiler backend supports that
interpreter. See [compatibility](compatibility.md). CI results are available after the
branch is pushed; local results do not imply those remote jobs have passed.

The docs job uploads the built site as an Actions artifact for review. It has no
deployment step, Pages permission, Pages environment, or `gh-pages` branch.
