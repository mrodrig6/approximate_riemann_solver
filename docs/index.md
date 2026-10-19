# Documentation

Exact and approximate Riemann solvers for the one-dimensional Euler
equations. The formulation, notation, equation numbers and test problems all
follow:

> E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics: A
> Practical Introduction*, 3rd edition, Springer, 2009.
> [doi:10.1007/b79761](https://doi.org/10.1007/b79761)

Each page below cites the specific chapters and equations it implements, so
the code can be read with the book open beside it. The pages follow the
package layout: one for the exact solver, one for the flux functions, one for
the scheme built on them, then the ADER alternative and the shallow-water
system. The last page is about the code rather than the mathematics, and maps
the modules onto everything above.

---

## Pages

### [1. The Riemann problem for the Euler equations](formulation.md)
*Toro Chapters 3 and 4.* The governing equations and their closure; the
structure of the Riemann solution (three waves, two star states); the
reduction to the single scalar equation $f(p_{\ast}) = 0$; why Newton–Raphson
always converges and how the initial guess is chosen; self-similar sampling of
the wave fan; the vacuum condition.

### [2. Numerical flux functions](flux-functions.md)
*Toro Chapter 10; Liou 2006.* The six interchangeable fluxes — Roe, HLLC,
HLL with Einfeldt wave speeds, AUSM+-up, Rusanov and Lax-Friedrichs — grouped
by family; the formulation of each; measured contact resolution, accuracy and
cost; which fluxes run which problems and why.

### [3. Roe's solver in detail](roe-solver.md)
*Toro Chapter 11.* The linearisation and the U-property; the Roe-average
state, eigenstructure and wave strengths; the Harten–Hyman entropy fix for
transonic rarefactions; and the solver's one hard limitation, that it is not
positivity-preserving near vacuum.

### [4. The finite-volume scheme](finite-volume-scheme.md)
*Toro Chapters 6, 13 and 14.* The conservative update, the CFL time step and
the boundary conditions; Godunov's theorem and why a nonlinear scheme is
needed for second order; the flux-limited correction; the smoothness sensor
$r_k$; Sweby's TVD region and the four limiters; stability.

### [5. The ADER scheme](ader.md)
*Toro Chapters 19–20.* A one-step route to second order: a Riemann problem
solved for the derivatives as well as the states, and the Cauchy–Kovalewski
procedure turning time derivatives into space derivatives; the measured
order of accuracy on smooth data, and why that needs smooth data at all.

### [6. The shallow-water equations](shallow-water.md)
*Toro, Shock-Capturing Methods for Free-Surface Shallow Flows.* The same
machinery for a second system: exact solver on wet and dry beds, Roe, HLL,
Rusanov and Lax–Friedrichs fluxes, dam-break results. Why there is no HLLC
here, and why a dry bed is routine where a vacuum is pathological.

### [7. Validation and results](validation.md)
Every figure, with the numbers behind it: Sod's problem against the exact
solution; limiters and order of accuracy; the entropy glitch and its fix;
grid convergence; Toro's test suite; the documented failure on Test 2; a
dimensional (SI) shock tube; what the test suite covers.

### [8. Package architecture](architecture.md)
How the modules of `src/` fit together: the three-layer hierarchy and the rule
that separates them; the annotated directory tree; the import graph; what each
module owns and what it is allowed to import; and where a new flux, limiter,
scheme or system of equations goes.

---

## Quick reference

### Symbols

| Symbol | Meaning |
| --- | --- |
| $\rho, u, p$ | density, velocity, pressure — the primitive state $\mathbf{W}$ |
| $E, e$ | total energy per unit volume; specific internal energy $e = p/((\gamma-1)\rho)$ |
| $a, H$ | sound speed $\sqrt{\gamma p/\rho}$; specific total enthalpy $(E+p)/\rho$ |
| $\gamma$ | ratio of specific heats (1.4 for air) |
| $p_{\ast}, u_{\ast}$ | star-region pressure and velocity |
| $\tilde{u}, \tilde{H}, \tilde{a}$ | Roe-averaged velocity, total enthalpy, sound speed |
| $\tilde{\lambda}_{k}, \tilde{\mathbf{K}}_{k}, \tilde{\alpha}_{k}$ | Roe eigenvalue, right eigenvector and wave strength of family $k$ |
| $\nu_{k}$ | Courant number $\tilde{\lambda}_{k}\Delta t/\Delta x$ of family $k$ |
| $\phi(r)$ | flux limiter |
| $S_L, S_R, S_{\ast}$ | HLL/HLLC left, right and contact wave speeds |
| $M^{\pm}_{(4)}, P^{\pm}_{(5)}$ | AUSM+ split Mach and pressure polynomials |
| $K_p, K_u$ | AUSM+-up pressure- and velocity-diffusion coefficients |
| $h, a = \sqrt{gh}$ | shallow-water depth and gravity-wave celerity |

### Modules

| Module | Contents |
| --- | --- |
| `src/euler/thermo.py` | equation of state, primitive ↔ conservative, Euler flux |
| `src/euler/exact.py` | exact Riemann solver: $f(p)$, Newton, sampling |
| `src/euler/fluxes/roe.py` | Roe average, eigenstructure, Roe flux, entropy fix |
| `src/euler/fluxes/hll.py` | Einfeldt wave speeds, HLL, HLLC |
| `src/euler/fluxes/ausm.py` | split polynomials, AUSM+-up |
| `src/euler/fluxes/central.py` | Rusanov, Lax-Friedrichs |
| `src/euler/fluxes/interface.py` | flux dispatch and the second-order TVD term |
| `src/euler/fluxes/registry.py` | flux names and codes |
| `src/euler/ader.py` | the one-step ADER scheme |
| `src/core/limiters.py` | superbee, minmod, van Leer, van Albada (shared) |
| `src/shallow_water/` | the shallow-water system: physics, exact solver, fluxes, driver, problems |
| `src/euler/solver.py` | conservative update, CFL time step, boundary conditions |
| `src/euler/problems.py` | Toro's test problems and two extras |
| `src/core/plotting.py` | four-panel comparison figures |
| `src/cli.py` | `python -m src` |
| `src/core/jit.py` | optional-Numba shim |

### Built-in problems

| Name | Wave pattern | Source |
| --- | --- | --- |
| `sod` | rarefaction, contact, shock | Toro Test 1 |
| `toro2` | two rarefactions, near-vacuum star state — **Roe fails** | Toro Test 2 |
| `toro3` | rarefaction, contact, shock; pressure ratio $10^{5}$ | Toro Test 3 |
| `toro4` | two colliding strong shocks | Toro Test 4 |
| `toro5` | slowly-moving contact | Toro Test 5 |
| `sonic_rarefaction` | transonic fan — exercises the entropy fix | this repository |
| `air_shock_tube` | Mach 1.35 shock in air, SI units | dimensional case |

---

## A note on the mathematics in these pages

Equations here are written for GitHub's Markdown maths renderer, which is
more restrictive than a LaTeX document. The rules these pages follow, each
of which corresponds to something that does not render:

- **One equation per `$$` block**, with the block on its own line, flush to
  the left margin, and a blank line either side. An indented `$$` inside a
  list item is not recognised as maths at all.
- **No multi-line environments** — no `aligned`, `cases`, `array` or
  `bmatrix`. They need `\\` row separators, and the Markdown parser
  consumes backslashes before the maths renderer sees them. Column vectors
  are written as transposed rows, e.g. $[\rho, \rho u, E]^{T}$, and a
  case split is written as prose.
- **No `\tag` or `\label`.** Equation numbering is not available; a
  reference such as "Toro eq. (4.5)" goes in the surrounding prose.
- **No spacing macros** (`\,`, `\;`, `\!`). They are purely cosmetic and
  every one is another backslash for the parser to eat.
- **No literal `*` inside maths.** GitHub pairs `*` as emphasis *before* the
  maths span is handed to the renderer, and unlike `_` it pairs intraword, so
  two asterisks anywhere in the same paragraph swallow everything between
  them into an `<em>`. Star-region quantities are therefore written
  $p_{\ast}$, $\mathbf{U}^{\ast}_{L}$ and so on, never `p_{*}`.
- **No custom macros**, and nothing needing a LaTeX package.

Figure labels are a separate matter: they use matplotlib's built-in
**mathtext**, never `text.usetex`, so no LaTeX installation is required.
See the module docstring of `src/core/plotting.py` for the rules they follow.

The repository is checked against these rules: every maths span in `docs/`
and the README parses under a real TeX engine, and the structural rules
above are verified mechanically.
