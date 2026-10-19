# The shallow-water equations

> Reference: E. F. Toro, *Shock-Capturing Methods for Free-Surface Shallow
> Flows*, Wiley 2001 — Chapters 2 and 5 for the equations and the exact
> solver, Chapters 7–10 for the numerical fluxes.

```python
from src.shallow_water import get_problem, solve, solve_exact

prob = get_problem("dam_break")
exact = solve_exact(prob.left, prob.right, prob.t_end)
sol = solve(prob.left, prob.right, prob.t_end, nx=200, flux="hll")
```

---

## 1. The equations

For a layer of fluid whose depth is small compared with the wavelength of
the motion, over a flat frictionless bed, conservation of mass and momentum
gives

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = \mathbf{0},$$

$$\mathbf{U} = [h, hu]^{T}, \qquad \mathbf{F} = [hu, hu^{2} + \tfrac{1}{2}gh^{2}]^{T},$$

with $h$ the depth, $u$ the depth-averaged velocity and $g$ gravity.

There is no equation of state to supply and no energy equation: the
$\tfrac{1}{2}gh^{2}$ term is the depth-integrated hydrostatic pressure, and
the system closes itself. That makes it the simplest non-trivial non-linear
hyperbolic system, and the natural second system for a code built around the
Euler equations.

The Jacobian has eigenvalues

$$\lambda_{1} = u - a, \qquad \lambda_{2} = u + a, \qquad a = \sqrt{gh},$$

with $a$ the celerity of a small gravity wave. The Froude number $u/a$ plays
the role of the Mach number: subcritical and supercritical correspond to
subsonic and supersonic.

---

## 2. What differs from the Euler case

Two differences explain nearly everything downstream.

**Two waves, both genuinely non-linear.** There is no linearly degenerate
field, hence no contact discontinuity. A Riemann problem is a bore (shock)
or a rarefaction on each side of one constant middle state, across which
*both* depth and velocity are constant.

The practical consequence is that **HLLC does not exist here**. HLLC's whole
purpose is to restore the contact that HLL averages away; with no contact to
restore it reduces exactly to HLL, so `flux="hllc"` is accepted as an alias
rather than implemented separately. It also means HLL is not at the accuracy
disadvantage it has for a gas — its two-wave model *is* the full wave
structure — which is why HLL, not Roe, is the default flux here.

**A dry bed instead of a vacuum.** The analogue of vacuum is $h = 0$, and
unlike vacuum in a gas it is an ordinary engineering case: a dam break onto
dry ground is *the* canonical test. The exact solver therefore solves the dry
cases rather than refusing them, and the approximate solvers handle a
wetting front.

---

## 3. The exact solver

The same construction as the Euler exact solver, with depth in place of
pressure. Both waves pin the middle state, and eliminating $u_{\ast}$ leaves one
scalar equation in $h_{\ast}$:

$$f(h) = f_{L}(h, h_{L}) + f_{R}(h, h_{R}) + (u_{R} - u_{L}) = 0,$$

with the one-sided depth functions

$$f_{K}(h) = 2\left(\sqrt{gh} - \sqrt{gh_{K}}\right) \qquad h \le h_{K} \text{(rarefaction)},$$

$$f_{K}(h) = (h - h_{K})\sqrt{\frac{g}{2} \frac{h + h_{K}}{h h_{K}}} \qquad h > h_{K} \text{(bore)}.$$

$f$ is monotone increasing and concave, so Newton converges from any positive
start. The starting guess is the two-rarefaction approximation,

$$h_{0} = \frac{1}{g}\left[\tfrac{1}{2}(a_{L} + a_{R}) + \tfrac{1}{4}(u_{L} - u_{R})\right]^{2},$$

which is *exact* when both waves are rarefactions.

### Dry bed

Three dry situations arise, and all three are solved:

| Case | Condition | Solution |
| --- | --- | --- |
| Dry to the right | $h_{R} = 0$ | left rarefaction to a wet/dry front at $u_{L} + 2a_{L}$ |
| Dry to the left | $h_{L} = 0$ | mirror image, front at $u_{R} - 2a_{R}$ |
| Dry generated | $u_{R} - u_{L} \ge 2(a_{L} + a_{R})$ | both rarefactions run out; a dry region opens between them |

The last condition is the shallow-water counterpart of Toro's vacuum
condition for a gas.

The dry dam break has a closed-form answer — **Ritter's solution** — which
makes it a strong check. Over dry bed the depth at the dam site is $4h_{L}/9$
for all time, the velocity there is $\tfrac{2}{3}\sqrt{gh_{L}}$, and the
wet/dry front advances at $2\sqrt{gh_{L}}$. The test suite asserts all three.

---

## 4. The fluxes

| Flux | `flux=` | Notes |
| --- | --- | --- |
| HLL | `hll`, `hlle`, `hllc` | Default. Two waves is the full wave structure here. |
| Roe | `roe` | With a Harten entropy fix; falls back to HLL at a dry interface. |
| Rusanov | `rusanov`, `llf` | Local Lax–Friedrichs. |
| Lax–Friedrichs | `lax_friedrichs`, `lf` | The dissipative baseline. |

The Roe average for this system is

$$\tilde{u} = \frac{\sqrt{h_{L}} u_{L} + \sqrt{h_{R}} u_{R}}{\sqrt{h_{L}} + \sqrt{h_{R}}}, \qquad \tilde{a} = \sqrt{\tfrac{1}{2}g (h_{L} + h_{R})}.$$

As for a gas, these are the unique averages making the U-property
$\mathbf{F}_R - \mathbf{F}_L = \tilde{A}(\mathbf{U}_R - \mathbf{U}_L)$ hold
exactly, and the velocity is again weighted by the square root of the first
conserved variable. The celerity average is the simpler of the two because
$\mathbf{F}$ is quadratic in $\mathbf{U}$ here, where for a gas it is not.
`tests/test_shallow_water.py::test_roe_u_property` checks the identity
numerically.

At a dry interface the Roe average divides by
$\sqrt{h_L} + \sqrt{h_R} = 0$, so the Roe flux falls back to HLL there, and
the wave-speed estimates switch from the Einfeldt form to the wet/dry front
speed $u \pm 2a$ that the exact solver uses. Getting that right is what lets
the approximate solvers run a dry dam break at all.

---

## 5. Results

![Dam break on a wet and a dry bed](figures/shallow_water_dam_break.png)

Relative $L_1$ error in depth at 200 cells, second order with superbee:

| Problem | Pattern | Roe | HLL | Rusanov | LF |
| --- | --- | --- | --- | --- | --- |
| `dam_break` | rarefaction, bore | $1.0\times10^{-3}$ | $2.3\times10^{-3}$ | $6.8\times10^{-3}$ | $1.1\times10^{-2}$ |
| `dam_break_dry` | rarefaction to dry | $1.9\times10^{-3}$ | $2.7\times10^{-3}$ | $9.6\times10^{-3}$ | $1.8\times10^{-2}$ |
| `two_bores` | bore, bore | $2.0\times10^{-3}$ | $2.0\times10^{-3}$ | $4.6\times10^{-3}$ | $9.9\times10^{-3}$ |
| `two_rarefactions` | rarefaction, rarefaction | $3.2\times10^{-3}$ | $3.0\times10^{-3}$ | $4.0\times10^{-3}$ | $1.6\times10^{-2}$ |
| `dry_generation` | dry region opens | $7.5\times10^{-3}$ | $8.8\times10^{-3}$ | $4.3\times10^{-3}$ | $5.5\times10^{-2}$ |
| `subcritical_step` | weak step | $5.1\times10^{-4}$ | $6.1\times10^{-4}$ | $1.6\times10^{-3}$ | $3.2\times10^{-3}$ |

![Shallow-water fluxes at the bore](figures/shallow_water_fluxes.png)

Note how much closer Roe and HLL are here than in the
[Euler comparison](flux-functions.md#5-measured-behaviour), where HLL is
about three times worse. That is the missing contact: for a gas HLL pays for
averaging the contact away, and here there is nothing to average away.

Order of accuracy on a smooth depth pulse, measured as in
[ader.md](ader.md#3-order-of-accuracy): 1.0 at first order, 2.0 with the TVD
correction.

---

## 6. Layout

```
src/shallow_water/
  physics.py    variables, physical flux, Roe average, eigenstructure
  exact.py      exact Riemann solver, wet and dry bed
  fluxes.py     Roe, HLL, Rusanov, Lax-Friedrichs; dispatch; TVD term
  solver.py     finite-volume driver, CFL, boundary conditions
  problems.py   dam-break test problems
```

A single `fluxes.py` rather than the one-module-per-family package the Euler
fluxes use: with two equations and four fluxes the whole thing is shorter
than any one of those files.

The finite-volume driver is a separate file rather than a shared one. The
state vector has two components instead of three and the maximum signal
speed is defined differently, so under Numba a shared driver would have to
branch on the system inside every kernel — which costs more in clarity than
the duplicated loop does. The two packages do share the limiter functions,
the Numba shim and the plotting style.

---

## 7. A caveat on dry beds

The usual CFL bound is **not conservative at a wetting front**: the front
advances faster than $\max(|u| + \sqrt{gh})$ over the cell suggests, because
the depth there is going to zero. The dry-bed problems therefore carry
`cfl=0.45` in the catalogue, and `solve` says so in its error message if a
depth goes negative. This is a property of the front, not of any particular
flux.

---

**Back to:** [Documentation index](index.md)
