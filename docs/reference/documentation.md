# Documentation maintenance

Build the site with Material MkDocs and edit each fact on its canonical page. The root
README supplies the home page through `docs/hooks.py`; the hook rewrites README-relative
links for the generated site.

```bash
python -m pip install -e '.[all,docs,dev]'
python scripts/check_math.py
python scripts/check_docs.py
mdformat --check README.md docs
codespell
mkdocs build --strict
```

`mkdocs serve` is optional for local interactive review. The strict build fails on
missing documents, broken internal links, and missing anchors. The hook maps repository
file links to their GitHub source URLs for site visitors. On GitHub, source Markdown
retains repository-relative links.

## Figures

`docs/figures/make_figures.py` is the source next to all eleven images. Each figure
function names its output; the page captions name the function. A source comment gives
the rebuild command. `scripts/make_figures.py` remains a compatible entry point.

```bash
# Refresh the committed documentation images.
python scripts/make_figures.py
# Check generation without replacing the committed images.
python scripts/make_figures.py --output-dir /tmp/riemann-figures
```

The solver is deterministic, but image bytes and font layout may vary with Matplotlib
and fonts. Numerical results and their tolerances are the validation target; identical
PNG bytes across platforms are not promised. Each image needs descriptive alt text.
Mermaid source is kept in its page; Material renders it as a diagram and GitHub can
render the same fence.

## Equations

Use inline `$...$` and one equation per unindented display block with `$$` delimiters on
separate lines, with blank lines around it. Write star quantities with `\ast`. Keep
equation references in prose and avoid custom macros, TeX packages, row-separator
environments, and cosmetic spacing macros. These are project portability conventions,
not a complete list of every renderer's capabilities.

`scripts/check_math.py` recursively checks delimiter balance and the restricted syntax
in README/docs. It is a structural checker, not a TeX engine or browser renderer.
Material uses MathJax; figure labels use Matplotlib mathtext, which has different
support. No TeX installation is needed.

## Checks and examples

`scripts/check_docs.py` executes complete Python and supported shell examples in
isolated temporary working directories, captures their output, checks recorded output
blocks, and runs package doctests. It also checks source-file links and required
figure-source captions. Installation and CI-maintenance command blocks are documented
instructions, not recursively executed by that checker. See [CI](ci.md) for the complete
validation commands.

Format documentation with `mdformat README.md docs`. Never replace equation signs or
spelling in code identifiers merely to satisfy prose tooling. `codespell` has a small
explicit scientific-word allowlist in `pyproject.toml`.

## Recompute numerical tables

Run `python scripts/record_validation.py > /tmp/riemann-results.json` to record star
states, profile errors, contact widths, entropy errors, and smooth convergence rates.
This writes no tracked files. Compare the emitted numbers with the validation pages
before changing a table or claiming extra digits of agreement.
