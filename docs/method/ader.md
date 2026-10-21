# ADER2

`src.euler.ader` implements the second-order one-step ADER scheme. Use `scheme="ader2"`
through the [Python API](../api/package.md) or the [CLI](../running/command-line.md);
higher orders are not implemented.

## Two routes to second order

The scheme of [finite-volume method](finite-volume.md) reaches second order by
*correcting* a first-order flux: compute the upwind flux, then add a limited
antidiffusive term. ADER gets there by a different route, and the difference is
structural rather than cosmetic.

ADER reconstructs a polynomial in each cell, solves a Riemann problem **for the
derivatives as well as the states**, and uses the governing equation itself to turn time
derivatives into space derivatives. The interface state is then known as a *function of
time* across the step, so the flux can be integrated over the step by quadrature.

Two consequences follow:

- It is genuinely **one-step**. No Runge–Kutta stages, one Riemann solve per interface
  per step. A two-stage RK scheme of the same order needs twice the Riemann solves.
- It is a **template**. Carrying more terms of the Taylor series in time, and
  reconstructing a higher-degree polynomial, raises the order without changing the
  structure. That is what the "arbitrary high order" in the name refers to.

This module implements the second-order member, ADER2 — the one that can be written
without a symbolic Cauchy–Kovalewski expansion.

______________________________________________________________________

## The algorithm

A limited linear reconstruction gives a slope $s_i$ in each cell, hence
boundary-extrapolated values $U_i \pm s_i/2$. At interface $i+1/2$, with
$U_L = U_i + s_i/2$ and $U_R = U_{i+1} - s_{i+1}/2$:

**Step 1 — the leading term.** Solve the classical Riemann problem $(U_L, U_R)$ and take
the state on the interface, $U^{\ast}(0)$. This is the Godunov state, and on its own it
is what a first-order scheme would use. Here it comes from the HLLC wave model, which
restores the contact wave. The subsequent predictor is not positivity preserving; see
[ADER validation](../validation/ader.md).

**Step 2 — the derivative Riemann problem.** The spatial derivative is itself
discontinuous across the interface, with values $s_i/\Delta x$ and $s_{i+1}/\Delta x$
either side. Evolve it with the system linearised about $U^{\ast}(0)$, and take the
value on the interface — which means projecting each side onto the characteristic fields
of $A(U^{\ast}(0))$ and picking the upwind component, field by field:

$$
\left(\frac{\partial U}{\partial x}\right)^{\ast} = \sum_{k} \beta_{k} K_{k},
$$

where $\beta_{k} = \alpha_{k}^{L}$ when $\lambda_{k} > 0$ and
$\beta_{k} = \alpha_{k}^{R}$ otherwise, with $\alpha_{k}^{L}$ and $\alpha_{k}^{R}$ the
characteristic components of the two one-sided derivatives.

**Step 3 — Cauchy–Kovalewski.** The governing equation converts the time derivative into
the space derivative:

$$
\frac{\partial U}{\partial t} = -A(U)\frac{\partial U}{\partial x} \quad \Longrightarrow \quad \left(\frac{\partial U}{\partial t}\right)^{\ast} = -\sum_{k} \lambda_{k}\beta_{k}K_{k}.
$$

No Jacobian is ever formed: the eigen-decomposition that step 2 already needs supplies
the matrix–vector product directly.

**Step 4 — the flux.** The interface state is now linear in time within the step, so the
midpoint rule integrates the flux to second order:

$$
F_{i+1/2} = F\left(U^{\ast}(0) + \tfrac{1}{2}\Delta t\left(\frac{\partial U}{\partial t}\right)^{\ast}\right).
$$

Note what does **not** appear: no numerical flux function. ADER evaluates the *physical*
flux at a state. The only Riemann solve is the one producing $U^{\ast}(0)$, which is why
`flux=`, `entropy_fix=` and `second_order=` have no effect under `scheme="ader2"`.
