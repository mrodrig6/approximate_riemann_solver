# Euler model

The Euler model describes one-dimensional inviscid ideal-gas flow. Inputs use primitive
states `[rho, u, p]` and a constant ratio of specific heats `gamma > 1`.

## Governing equations

The one-dimensional, time-dependent Euler equations express conservation of mass,
momentum and energy for an inviscid, non-conducting compressible fluid. In conservative
form (Toro §3.1):

$$
\frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}(\mathbf{U})}{\partial x} = \mathbf{0}
$$

with the vector of conserved variables

$$
\mathbf{U} = [\rho, \rho u, E]^{T}
$$

and the flux vector

$$
\mathbf{F}(\mathbf{U}) = [\rho u, \rho u^{2} + p, u(E + p)]^{T}.
$$

Here $\rho$ is density, $u$ velocity, $p$ pressure and $E$ the **total** energy per unit
volume. The system has three equations and four unknowns, so it needs a closure. For an
ideal (polytropic) gas with constant ratio of specific heats $\gamma$, the caloric
equation of state (Toro eq. 3.9) supplies it:

$$
E = \rho e + \tfrac{1}{2}\rho u^{2}, \qquad e = \frac{p}{(\gamma - 1)\rho}.
$$

The sound speed follows as

$$
a = \sqrt{\frac{\gamma p}{\rho}},
$$

and the specific **total** enthalpy, which the Roe average is built from, is

$$
H = \frac{E + p}{\rho}.
$$

Throughout the code, $\mathbf{W} = [\rho, u, p]^{T}$ denotes the *primitive* state
vector and $\mathbf{U}$ the *conservative* one. `src/euler/physics.py` holds the
conversions between them.

### Why the system is hyperbolic

The Jacobian $\mathbf{A}(\mathbf{U}) = \partial\mathbf{F}/\partial\mathbf{U}$ has three
real eigenvalues (Toro eq. 3.18):

$$
\lambda_{1} = u - a, \qquad \lambda_{2} = u, \qquad \lambda_{3} = u + a,
$$

with a complete set of right eigenvectors. Information therefore travels along three
families of characteristics. Families 1 and 3 are *genuinely nonlinear* (they produce
shocks or rarefactions); family 2 is *linearly degenerate* (it produces a contact
discontinuity, across which $u$ and $p$ are continuous but $\rho$ and $e$ jump).

______________________________________________________________________

Use consistent units: SI inputs give SI outputs; the standard Toro cases are
nondimensional. The solver omits viscosity, heat conduction, source terms, and
additional spatial dimensions. See [parameters](parameters.md) and the
[problem catalogue](problems.md).
