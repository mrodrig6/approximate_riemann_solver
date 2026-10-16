# The ADER scheme

> Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
> Dynamics*, 3rd ed., Springer 2009 — **Chapter 19** (ADER for linear
> systems) and **Chapter 20** (the non-linear case, the derivative Riemann
> problem and the Cauchy–Kovalewski procedure).

```python
solve_roe(gamma, w_l, w_r, t_end, nx=200, scheme="ader2")
```

---

## 1. Two routes to second order

The scheme of [finite-volume-scheme.md](finite-volume-scheme.md) reaches
second order by *correcting* a first-order flux: compute the upwind flux,
then add a limited antidiffusive term. ADER gets there by a different route,
and the difference is structural rather than cosmetic.

ADER reconstructs a polynomial in each cell, solves a Riemann problem **for
the derivatives as well as the states**, and uses the governing equation
itself to turn time derivatives into space derivatives. The interface state
is then known as a *function of time* across the step, so the flux can be
integrated over the step by quadrature.

Two consequences follow:

- It is genuinely **one-step**. No Runge–Kutta stages, one Riemann solve per
  interface per step. A two-stage RK scheme of the same order needs twice
  the Riemann solves.
- It is a **template**. Carrying more terms of the Taylor series in time, and
  reconstructing a higher-degree polynomial, raises the order without
  changing the structure. That is what the "arbitrary high order" in the name
  refers to.

This module implements the second-order member, ADER2 — the one that can be
written without a symbolic Cauchy–Kovalewski expansion.

---

## 2. The algorithm

A limited linear reconstruction gives a slope $s_i$ in each cell, hence
boundary-extrapolated values $U_i \pm s_i/2$. At interface $i+1/2$, with
$U_L = U_i + s_i/2$ and $U_R = U_{i+1} - s_{i+1}/2$:

**Step 1 — the leading term.** Solve the classical Riemann problem
$(U_L, U_R)$ and take the state on the interface, $U^{\ast}(0)$. This is the
Godunov state, and on its own it is what a first-order scheme would use.
Here it comes from the HLLC wave model, which is cheap, positivity-preserving
and exact at a contact.

**Step 2 — the derivative Riemann problem.** The spatial derivative is itself
discontinuous across the interface, with values $s_i/\Delta x$ and
$s_{i+1}/\Delta x$ either side. Evolve it with the system linearised about
$U^{\ast}(0)$, and take the value on the interface — which means projecting each
side onto the characteristic fields of $A(U^{\ast}(0))$ and picking the upwind
component, field by field:

$$\left(\frac{\partial U}{\partial x}\right)^{\ast} = \sum_{k} \beta_{k} K_{k},$$

where $\beta_{k} = \alpha_{k}^{L}$ when $\lambda_{k} > 0$ and
$\beta_{k} = \alpha_{k}^{R}$ otherwise, with $\alpha_{k}^{L}$ and
$\alpha_{k}^{R}$ the characteristic components of the two one-sided
derivatives.

**Step 3 — Cauchy–Kovalewski.** The governing equation converts the time
derivative into the space derivative:

$$\frac{\partial U}{\partial t} = -A(U)\frac{\partial U}{\partial x} \quad \Longrightarrow \quad \left(\frac{\partial U}{\partial t}\right)^{\ast} = -\sum_{k} \lambda_{k}\beta_{k}K_{k}.$$

No Jacobian is ever formed: the eigen-decomposition that step 2 already needs
supplies the matrix–vector product directly.

**Step 4 — the flux.** The interface state is now linear in time within the
step, so the midpoint rule integrates the flux to second order:

$$F_{i+1/2} = F\left(U^{\ast}(0) + \tfrac{1}{2}\Delta t\left(\frac{\partial U}{\partial t}\right)^{\ast}\right).$$

Note what does **not** appear: no numerical flux function. ADER evaluates the
*physical* flux at a state. The only Riemann solve is the one producing
$U^{\ast}(0)$, which is why `flux=`, `entropy_fix=` and `second_order=` have no
effect under `scheme="ader2"`.

---

## 3. Order of accuracy

Measuring a formal order requires **smooth** initial data. Every problem in
the Riemann-problem catalogue contains a discontinuity, and near one any
scheme makes an $\mathcal{O}(1)$ error over an $\mathcal{O}(\Delta x)$
region — so the $L_1$ rate is capped at 1 whatever the formal order. That is
exactly why the [grid-convergence figure](validation.md#4-grid-convergence)
on Sod's problem shows slope $-1$ for every scheme.

`solve_field` exists for this: it advances an arbitrary initial field rather
than two constant states, and `advected_pulse` and `acoustic_pulse` supply
smooth data.

![Order of accuracy on a smooth acoustic pulse](figures/order_of_accuracy.png)

| Problem | Reference | 1st order | TVD | ADER2 |
| --- | --- | --- | --- | --- |
| Advected density pulse | analytic | 1.00 | 2.00 | 2.00 |
| Smooth acoustic pulse | fine grid | 1.00 | 2.00 | 2.01 |

The limiter is switched off for these measurements: a limiter clips smooth
extrema, which caps the observed rate below the formal one.

---

## 4. Is it actually a different scheme?

Worth checking, because at second order ADER2 and the TVD scheme are close
enough that a bug could hide as "about the same".

On the **smooth acoustic pulse**, which excites the genuinely non-linear
fields, they differ by $5.1 \times 10^{-4}$ at 200 cells, with ADER about 17%
more accurate.

On a **pure contact** they agree to machine precision. That is a consistency
check rather than a coincidence: with only the linearly degenerate field
active, both reduce to limited Lax–Wendroff on that one field, so they *must*
agree. The test suite asserts both facts, since together they distinguish
"two schemes that happen to be close" from "one scheme with two names".

On Sod's problem at 200 cells, ADER2 gives an $L_1$ density error of
$2.1 \times 10^{-3}$ against the TVD scheme's $1.4 \times 10^{-3}$ — the
same band, as expected. At second order ADER's advantage is the one-step
structure and the route to higher order, not the constant.

---

## 5. Limitations

- **Second order only.** Going higher needs more Taylor terms and a
  higher-degree reconstruction; the structure below is arranged for it but
  it is not implemented.
- **The linear reconstruction can go non-physical** on a very strong jump
  before the limiter catches it. The implementation falls back to the cell
  averages at such an interface, and `limiter="minmod"` is the safe choice
  on Toro's tests 3 to 5.
- **No entropy fix is applied**, and none is needed: the leading state comes
  from HLLC, which is entropy-satisfying.

---

**Back to:** [Documentation index](index.md)
