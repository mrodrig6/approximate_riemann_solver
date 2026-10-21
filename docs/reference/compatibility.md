# Compatibility

This matrix records tested equation, flux, and reconstruction combinations. A check mark
means the stated catalogue run succeeds, not unconditional positivity or support for
every platform.

## Robustness

Which fluxes run which problems, at second order with superbee:

| Problem               | Roe | HLLC | HLL | AUSM+-up | Rusanov | LF  |
| --------------------- | --- | ---- | --- | -------- | ------- | --- |
| `sod`                 | ✓   | ✓    | ✓   | ✓        | ✓       | ✓   |
| `toro2` (near vacuum) | ✗   | ✓    | ✓   | ✗ 2nd    | ✓       | ✓   |
| `toro3`               | ✓   | ✓    | ✓   | ✗ 2nd    | ✓       | ✓   |
| `toro4`               | ✓   | ✓    | ✓   | ✓        | ✓       | ✓   |
| `toro5`               | ✓   | ✓    | ✓   | ✗ 2nd    | ✗ 2nd   | ✓   |
| `sonic_rarefaction`   | ✓   | ✓    | ✓   | ✓        | ✓       | ✓   |
| `air_shock_tube`      | ✓   | ✓    | ✓   | ✓        | ✓       | ✓   |

"✗ 2nd" means the *flux* runs the problem but the second-order scheme does not: every
one of those cases succeeds with `second_order=False`. Only Roe on `toro2` is a failure
of the flux itself.

Three entries deserve comment.

**Roe on `toro2`** is the near-vacuum breakdown described in
[Roe theory](../running/troubleshooting.md). HLL and HLLC run it, which is the whole
point of a positivity-preserving flux, and resolves the long-standing roadmap item in
this repository:

![Toro Test 2 with HLLC and HLL](../figures/toro2_positivity.png)

Density and velocity are captured well. The overshoot in the specific internal energy
near $x = 0.5$ is the known difficulty of this test — the star region is at
$p_{\ast} \approx 0.0019$, so $e = p / ((\gamma - 1)\rho)$ is a ratio of two very small
numbers — and it appears for every scheme, not only these two.

**Rusanov on `toro5`** is a useful distinction. Rusanov is positivity-preserving *as a
first-order flux*, and at first order it runs `toro5` without trouble. The failure comes
from the second-order TVD correction, which the flux's positivity proof does not cover.
"The flux is positivity-preserving" and "the scheme is positivity-preserving" are
different statements, and only `hll`, `hllc` and `lax_friedrichs` are listed in
`src.euler.fluxes.ROBUST_FLUXES`.

**AUSM+-up on `toro2`, `toro3` and `toro5`** is the same story. The flux runs *every*
built-in problem at first order — `test_ausm_up_runs_every_problem_at_first_order`
asserts exactly that, and on `toro5` it is the most accurate first-order flux of the
six. What fails is the pairing with this package's characteristic-based TVD correction
on the three most extreme cases. That pairing is a choice of this code, not of the
scheme: a WENO reconstruction is the more natural companion for a flux that never forms
an eigensystem, and is what EMS uses.

______________________________________________________________________

## Systems and interfaces

| Feature                                 | Euler                                     | Shallow water                             |
| --------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| CLI                                     | ✓ built-in problems                       | ✗ use Python                              |
| Custom Riemann states / smooth fields   | ✓ Python                                  | ✓ Python                                  |
| Exact solver                            | ✓ rejects generated vacuum                | ✓ wet/dry; rejects both-dry exact data    |
| Roe / HLL / Rusanov / LF                | ✓                                         | ✓ Roe falls back to HLL at dry interfaces |
| HLLC                                    | ✓ separate flux                           | Alias of HLL                              |
| AUSM+-up                                | ✓                                         | ✗                                         |
| ADER2                                   | ✓ documented failures on `toro2`, `toro5` | ✗ not implemented                         |
| Boundary conditions                     | Transmissive only                         | Transmissive only                         |
| Higher dimensions, WENO, bed topography | ✗                                         | ✗                                         |

## Python and optional acceleration

The package declares Python 3.9 or newer. CI tests the oldest declared version and the
latest stable version resolved by `actions/setup-python` (`3.x`) without Numba. A
separate Python 3.12 job tests Numba and disabled JIT. Dependency resolvers select
releases compatible with each interpreter; the latest interpreter may precede Numba
wheel support.

The numerical package supports Python 3.9, but the pinned Markdown formatter plugins in
the `dev` extra require Python 3.10 or newer. The documented docs/lint/format/spelling
workflow uses Python 3.12, matching its CI jobs.

Local validation used Linux x86-64: Python 3.12.14 passed all 263 tests with Numba and
with JIT disabled; Python 3.13.5 without Numba passed 262 tests with one expected skip
for Numba-specific fast-math settings. Other platforms are not asserted as locally
verified. Plotting uses a headless backend in CI; `--show` requires a graphical session.
The remaining [limits](limitations.md) include a packaging/API version mismatch.

## Figure source

Rebuild these images with `python scripts/make_figures.py` from the repository root.
Source lives beside the images in [make_figures.py](../figures/make_figures.py):
`figure_toro2_positivity`.
