# The Riemann problem for the Euler equations

> Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
> Dynamics*, 3rd ed., Springer 2009 — **Chapter 3** (the Euler equations),
> **Chapter 4** (the exact Riemann solver).

---

## 1. Governing equations

The one-dimensional, time-dependent Euler equations express conservation of
mass, momentum and energy for an inviscid, non-conducting compressible fluid.
In conservative form (Toro §3.1):

$$\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = \mathbf{0}$$

with the vector of conserved variables

$$\mathbf{U} = [\rho,\ \rho u,\ E]^{T}$$

and the flux vector

$$\mathbf{F}(\mathbf{U}) = [\rho u,\ \rho u^{2} + p,\ u(E + p)]^{T}.$$

Here $\rho$ is density, $u$ velocity, $p$ pressure and $E$ the **total** energy
per unit volume. The system has three equations and four unknowns, so it needs
a closure. For an ideal (polytropic) gas with constant ratio of specific heats
$\gamma$, the caloric equation of state (Toro eq. 3.9) supplies it:

$$E = \rho e + \tfrac{1}{2}\rho u^{2}, \qquad e = \frac{p}{(\gamma - 1)\rho}.$$

The sound speed follows as

$$a = \sqrt{\frac{\gamma p}{\rho}},$$

and the specific **total** enthalpy, which the Roe average is built from, is

$$H = \frac{E + p}{\rho}.$$

Throughout the code, $\mathbf{W} = [\rho, u, p]^{T}$ denotes the *primitive*
state vector and $\mathbf{U}$ the *conservative* one. `riemann/thermo.py` holds
the conversions between them.

### Why the system is hyperbolic

The Jacobian $\mathbf{A}(\mathbf{U}) = \partial\mathbf{F}/\partial\mathbf{U}$
has three real eigenvalues (Toro eq. 3.18):

$$\lambda_{1} = u - a, \qquad \lambda_{2} = u, \qquad \lambda_{3} = u + a,$$

with a complete set of right eigenvectors. Information therefore travels along
three families of characteristics. Families 1 and 3 are *genuinely nonlinear*
(they produce shocks or rarefactions); family 2 is *linearly degenerate* (it
produces a contact discontinuity, across which $u$ and $p$ are continuous but
$\rho$ and $e$ jump).

---

## 2. The Riemann problem

The Riemann problem is the initial-value problem for the system above with
piecewise-constant data separated by a single discontinuity:

$$\mathbf{U}(x, 0) = \mathbf{U}_{L} \quad \text{for } x < 0, \qquad \mathbf{U}(x, 0) = \mathbf{U}_{R} \quad \text{for } x > 0.$$

It matters for two reasons. Physically it *is* the shock tube. Numerically it
is the building block of every Godunov-type finite-volume method: each cell
interface in a piecewise-constant reconstruction poses exactly this problem,
and its solution supplies the interface flux.

### Structure of the solution

Because neither the equations nor the data contain a length or time scale, the
solution depends on $x$ and $t$ only through the similarity variable

$$s = \frac{x}{t}.$$

It consists of three waves, one per characteristic family, separating four
constant states:

```
            left wave          contact          right wave
        (shock or rarefaction)    |        (shock or rarefaction)
                 \                |                /
      U_L         \     U*_L      |     U*_R      /         U_R
                   \              |              /
  ─────────────────────────────────────────────────────────────▶  x
                               t = 0
```

The two states between the outer waves are the **star region**. Across the
contact, $p$ and $u$ are continuous:

$$p^{*}_{L} = p^{*}_{R} \equiv p_{*}, \qquad u^{*}_{L} = u^{*}_{R} \equiv u_{*}.$$

Density is *not*: $\rho^{*}_{L} \ne \rho^{*}_{R}$ in general. This is the single
most important fact for understanding the numerical results — the contact
shows up in $\rho$ and $e$ but is invisible in $u$ and $p$, which is why the
four-panel figures in [validation.md](validation.md) always plot all four.

There are four possible wave patterns, depending on how $p_{*}$ compares with
$p_L$ and $p_R$:

| $p_{*}$ versus the data | Left wave | Right wave |
| --- | --- | --- |
| $p_{*} > p_L$ and $p_{*} > p_R$ | shock | shock |
| $p_{*} \le p_L$ and $p_{*} > p_R$ | rarefaction | shock |
| $p_{*} > p_L$ and $p_{*} \le p_R$ | shock | rarefaction |
| $p_{*} \le p_L$ and $p_{*} \le p_R$ | rarefaction | rarefaction |

---

## 3. Reducing the problem to a single scalar equation

This is the key step (Toro §4.2). Both outer waves relate the star-region
pressure to the star-region velocity. Eliminating $u_{*}$ between the two
relations leaves one nonlinear algebraic equation in the single unknown
$p_{*}$:

$$f(p) \equiv f_{L}(p, \mathbf{W}_{L}) + f_{R}(p, \mathbf{W}_{R}) + (u_{R} - u_{L}) = 0. \tag{Toro 4.5}$$

The functions $f_{K}$, for $K \in \{L, R\}$, each have two branches.

**Shock branch** ($p > p_{K}$), from the Rankine–Hugoniot jump conditions
(Toro eq. 4.6):

$$f_{K}(p) = (p - p_{K})\left[\frac{A_{K}}{p + B_{K}}\right]^{1/2},$$

with the data constants

$$A_{K} = \frac{2}{(\gamma + 1)\rho_{K}}, \qquad B_{K} = \frac{\gamma - 1}{\gamma + 1}\, p_{K}.$$

**Rarefaction branch** ($p \le p_{K}$), from isentropy plus the generalised
Riemann invariants (Toro eq. 4.7):

$$f_{K}(p) = \frac{2 a_{K}}{\gamma - 1}\left[\left(\frac{p}{p_{K}}\right)^{\frac{\gamma - 1}{2\gamma}} - 1\right].$$

Once $p_{*}$ is known, the star velocity follows directly (Toro eq. 4.9):

$$u_{*} = \tfrac{1}{2}(u_{L} + u_{R}) + \tfrac{1}{2}\left[f_{R}(p_{*}) - f_{L}(p_{*})\right].$$

### Why Newton–Raphson always works here

$f$ is continuous, strictly **monotone increasing** and **concave** on
$p > 0$ (Toro §4.3.1). A monotone concave function has exactly one root, and
Newton's method started from any positive $p$ converges to it — monotonically
from the right, and quadratically near the root. The derivative needed is

$$f_{K}'(p) = \left[\frac{A_{K}}{B_{K} + p}\right]^{1/2}\left[1 - \frac{p - p_{K}}{2(B_{K} + p)}\right] \qquad (p > p_{K}, \text{ shock}),$$

$$f_{K}'(p) = \frac{1}{\rho_{K} a_{K}}\left(\frac{p}{p_{K}}\right)^{-\frac{\gamma + 1}{2\gamma}} \qquad (p \le p_{K}, \text{ rarefaction}).$$

> **Note the exponent $-(\gamma+1)/(2\gamma)$ in the rarefaction derivative.**
> The MATLAB code this package was ported from used $-(\gamma-1)/(2\gamma)$.
> With the wrong derivative Newton degrades to a first-order fixed-point
> iteration: it still converged for Sod's problem, but slowly and only from a
> fortunate starting guess. See [porting-notes.md](porting-notes.md).

### The initial guess

Newton converges from any positive start, but a good start converges in 3–5
iterations instead of 20. The code uses Toro's adaptive recipe (§9.5), which
selects among three *non-iterative* approximate solvers according to how
strong the waves are:

- **PVRS** (primitive-variable Riemann solver, Toro eq. 9.20) when the
  pressure ratio is mild — the linearised estimate

  $$p_{PV} = \tfrac{1}{2}(p_{L} + p_{R}) - \tfrac{1}{8}(u_{R} - u_{L})(\rho_{L} + \rho_{R})(a_{L} + a_{R}).$$

- **Two-rarefaction** (Toro eq. 9.32) when $p_{PV}$ falls below
  $\min(p_L, p_R)$. This one is *exact* when both waves really are
  rarefactions, which is why Toro's Test 2 converges in a single Newton step.
- **Two-shock** (Toro eq. 9.42) when $p_{PV}$ exceeds $\max(p_L, p_R)$.

### Vacuum

If the two states separate faster than the rarefaction fans can fill the gap,
no star region exists. The condition (Toro eq. 4.40) is

$$\frac{2}{\gamma - 1}(a_{L} + a_{R}) \le u_{R} - u_{L},$$

and `solve_star_pressure` raises a `ValueError` rather than letting $f(p)$
hunt for a root that is not there.

---

## 4. Sampling the solution

With $p_{*}$ and $u_{*}$ known, every other quantity follows in closed form,
and the solution at any $(x, t)$ is obtained by evaluating it at
$s = x/t$ (Toro §4.5). The code partitions the whole $s$ axis without gaps:
first by the sign of $s - u_{*}$ (which side of the contact), then by the wave
type on that side.

**Across a shock** (take the left wave; the right is the mirror image), the
Rankine–Hugoniot relations give the post-shock density (Toro eq. 4.19) and
shock speed (Toro eq. 4.20):

$$\rho^{*}_{L} = \rho_{L}\,\frac{\frac{p_{*}}{p_{L}} + \frac{\gamma - 1}{\gamma + 1}}{\frac{\gamma - 1}{\gamma + 1}\frac{p_{*}}{p_{L}} + 1},$$

$$S_{L} = u_{L} - a_{L}\left[\frac{\gamma + 1}{2\gamma}\frac{p_{*}}{p_{L}} + \frac{\gamma - 1}{2\gamma}\right]^{1/2}.$$

The shock is a single discontinuity: the state is $\mathbf{W}_L$ for
$s < S_L$ and the star state for $s > S_L$.

**Across a rarefaction**, the flow is isentropic, so

$$\rho^{*}_{L} = \rho_{L}\left(\frac{p_{*}}{p_{L}}\right)^{1/\gamma}, \qquad a^{*}_{L} = a_{L}\left(\frac{p_{*}}{p_{L}}\right)^{\frac{\gamma - 1}{2\gamma}}.$$

The fan occupies a wedge bounded by its head and tail characteristics,

$$S_{HL} = u_{L} - a_{L}, \qquad S_{TL} = u_{*} - a^{*}_{L},$$

and *inside* the fan the generalised Riemann invariants give the profile
explicitly (Toro eq. 4.24). Writing
$\Phi = \frac{2}{\gamma + 1} + \frac{\gamma - 1}{(\gamma + 1)a_{L}}(u_{L} - s)$:

$$\rho = \rho_{L}\,\Phi^{\frac{2}{\gamma - 1}}, \qquad p = p_{L}\,\Phi^{\frac{2\gamma}{\gamma - 1}},$$

$$u = \frac{2}{\gamma + 1}\left[a_{L} + \frac{\gamma - 1}{2}u_{L} + s\right].$$

A **transonic** rarefaction is one whose fan straddles $s = 0$, i.e.
$S_{HL} < 0 < S_{TL}$. That case is harmless for the exact solver but is
precisely where Roe's approximate solver needs an entropy fix — see
[roe-solver.md](roe-solver.md#4-the-entropy-fix).

---

## 5. What is implemented where

| Concept | Equation | Code |
| --- | --- | --- |
| Primitive ↔ conservative, EOS | Toro 3.9 | `riemann/thermo.py` |
| Pressure function $f(p)$ | Toro 4.5–4.7 | `exact.pressure_function` |
| Its derivative | Toro 4.37 | `exact.pressure_function_derivative` |
| Adaptive initial guess | Toro §9.5 | `exact.initial_pressure_guess` |
| Newton iteration for $p_{*}$ | Toro §4.3.2 | `exact.solve_star_pressure` |
| Vacuum check | Toro 4.40 | `exact._check_vacuum` |
| Self-similar sampling | Toro §4.5 | `exact._sample_one` |

---

**Next:** [Roe's approximate Riemann solver](roe-solver.md)
