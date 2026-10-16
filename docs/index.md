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
system.

---

## Pages

### [1. The Riemann problem for the Euler equations](formulation.md)
*Toro Chapters 3 and 4.* The governing equations and their closure; the
structure of the Riemann solution (three waves, two star states); the
reduction to the single scalar equation $f(p_{*}) = 0$; why Newton–Raphson
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

---

## Quick reference

### Symbols

| Symbol | Meaning |
| --- | --- |
| $\rho,\ u,\ p$ | density, velocity, pressure — the primitive state $\mathbf{W}$ |
| $E,\ e$ | total energy per unit volume; specific internal energy $e = p/((\gamma-1)\rho)$ |
| $a,\ H$ | sound speed $\sqrt{\gamma p/\rho}$; specific total enthalpy $(E+p)/\rho$ |
| $\gamma$ | ratio of specific heats (1.4 for air) |
| $p_{*},\ u_{*}$ | star-region pressure and velocity |
| $\tilde{u},\ \tilde{H},\ \tilde{a}$ | Roe-averaged velocity, total enthalpy, sound speed |
| $\tilde{\lambda}_{k},\ \tilde{\mathbf{K}}_{k},\ \tilde{\alpha}_{k}$ | Roe eigenvalue, right eigenvector and wave strength of family $k$ |
| $\nu_{k}$ | Courant number $\tilde{\lambda}_{k}\Delta t/\Delta x$ of family $k$ |
| $\phi(r)$ | flux limiter |
| $S_L,\ S_R,\ S_*$ | HLL/HLLC left, right and contact wave speeds |
| $M^{\pm}_{(4)},\ P^{\pm}_{(5)}$ | AUSM+ split Mach and pressure polynomials |
| $K_p,\ K_u$ | AUSM+-up pressure- and velocity-diffusion coefficients |
| $h,\ a = \sqrt{gh}$ | shallow-water depth and gravity-wave celerity |

### Modules

| Module | Contents |
| --- | --- |
| `riemann/thermo.py` | equation of state, primitive ↔ conservative, Euler flux |
| `riemann/exact.py` | exact Riemann solver: $f(p)$, Newton, sampling |
| `riemann/fluxes/roe.py` | Roe average, eigenstructure, Roe flux, entropy fix |
| `riemann/fluxes/hll.py` | Einfeldt wave speeds, HLL, HLLC |
| `riemann/fluxes/ausm.py` | split polynomials, AUSM+-up |
| `riemann/fluxes/central.py` | Rusanov, Lax-Friedrichs |
| `riemann/fluxes/interface.py` | flux dispatch and the second-order TVD term |
| `riemann/fluxes/registry.py` | flux names and codes |
| `riemann/ader.py` | the one-step ADER scheme |
| `riemann/limiters.py` | superbee, minmod, van Leer, van Albada (shared) |
| `riemann/shallow_water/` | the shallow-water system: physics, exact solver, fluxes, driver, problems |
| `riemann/solver.py` | conservative update, CFL time step, boundary conditions |
| `riemann/problems.py` | Toro's test problems and two extras |
| `riemann/plotting.py` | four-panel comparison figures |
| `riemann/cli.py` | `python -m riemann` |
| `riemann/_jit.py` | optional-Numba shim |

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

Equations here are written for GitHub's Markdown maths renderer. They are kept
deliberately plain — one equation per `$$` block, no `aligned`, `cases` or
`bmatrix` environments, no custom macros and no `\\` row breaks, because
those are the constructs that most often fail to render. Column vectors are
written as transposed rows, e.g. $[\rho,\ \rho u,\ E]^{T}$.

Figure labels use matplotlib's built-in **mathtext**, never `text.usetex`.
See the module docstring of `riemann/plotting.py` for why, and for the rules
the labels follow.
