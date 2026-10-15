# Documentation

Exact and approximate Riemann solvers for the one-dimensional Euler
equations. The formulation, notation, equation numbers and test problems all
follow:

> E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics: A
> Practical Introduction*, 3rd edition, Springer, 2009.
> [doi:10.1007/b79761](https://doi.org/10.1007/b79761)

Each page below cites the specific chapters and equations it implements, so
the code can be read with the book open beside it.

---

## Pages

### [1. The Riemann problem for the Euler equations](formulation.md)
*Toro Chapters 3 and 4.* The governing equations and their closure; the
structure of the Riemann solution (three waves, two star states); the
reduction to the single scalar equation $f(p_{*}) = 0$; why Newton–Raphson
always converges and how the initial guess is chosen; self-similar sampling of
the wave fan; the vacuum condition.

### [2. Roe's approximate Riemann solver](roe-solver.md)
*Toro Chapter 11.* The linearisation and the U-property; the Roe-average
state, eigenstructure and wave strengths; the Roe flux; the Harten–Hyman
entropy fix for transonic rarefactions; the solver's one hard limitation
(it is not positivity-preserving near vacuum); the finite-volume driver,
CFL condition and boundary conditions.

### [3. Numerical flux functions](flux-functions.md)
*Toro Chapter 10; Liou 2006.* HLL with Einfeldt wave speeds, HLLC and its
contact wave, AUSM+-up and its flux splitting, Rusanov and Lax-Friedrichs;
measured contact resolution, accuracy and cost for all six; which fluxes run
which problems and why; where the AUSM+-up coefficients come from, and what
a mis-scaled one did; and what is still not implemented (SLAU).

### [4. Second-order accuracy and TVD flux limiters](tvd-extension.md)
*Toro Chapters 13 and 14.* Godunov's theorem and why a nonlinear scheme is
needed; the flux-limited second-order correction; the smoothness sensor
$r_k$; Sweby's TVD region and the four limiters; what second order does and
does not buy; stability.

### [5. Validation and results](validation.md)
Every figure, with the numbers behind it: Sod's problem against the exact
solution; limiters and order of accuracy; the entropy glitch and its fix;
grid convergence; Toro's test suite; the documented failure on Test 2; a
dimensional (SI) shock tube; what the test suite covers.

### [6. Porting notes: MATLAB to Python](porting-notes.md)
File-by-file mapping from the archived MATLAB code, and the complete list of
bugs found and fixed — a wrong exponent in the Newton derivative, an
unbounded iteration with a dimensionally meaningless tolerance, gaps in the
exact sampler, a wrong base state in the entropy fix, an unguarded division
in the limiter, a wrong maximum signal speed, and one bug introduced and
removed during the port itself.

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

### Modules

| Module | Contents |
| --- | --- |
| `riemann/thermo.py` | equation of state, primitive ↔ conservative, Euler flux |
| `riemann/exact.py` | exact Riemann solver: $f(p)$, Newton, sampling |
| `riemann/roe.py` | Roe average, eigenstructure, Roe flux, entropy fix |
| `riemann/fluxes.py` | HLL, HLLC, AUSM+-up, Rusanov, Lax-Friedrichs; dispatch; TVD term |
| `riemann/limiters.py` | superbee, minmod, van Leer, van Albada |
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
| `air_shock_tube` | Mach 1.35 shock in air, SI units | `test_input2o1.m` |

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
