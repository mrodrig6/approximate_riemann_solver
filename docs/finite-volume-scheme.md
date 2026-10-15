# The finite-volume scheme

> Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
> Dynamics*, 3rd ed., Springer 2009 — **Chapter 6** (the conservative update,
> boundary conditions and the CFL condition), **Chapter 13** §13.8 and
> **Chapter 14** §14.4 (the flux-limited second-order extension).

This page covers everything between the numerical flux and the answer: how
the flux is used to advance the solution, how the time step and boundaries
are set, and how the scheme is lifted from first to second order.

---

## 1. The conservative update

Whichever flux of [flux-functions.md](flux-functions.md) is chosen, it enters
the same explicit conservative update (Toro Sec. 6.2):

$$\mathbf{U}_{i}^{\,n+1} = \mathbf{U}_{i}^{\,n} - \frac{\Delta t}{\Delta x}\left[\mathbf{F}_{i+1/2} - \mathbf{F}_{i-1/2}\right].$$

**Time step.** Set by the CFL condition (Toro §6.4) from the maximum signal
speed on the grid,

$$S_{\max} = \max_{i}\left(|u_{i}| + a_{i}\right), \qquad \Delta t = \frac{C_{\text{cfl}}\,\Delta x}{S_{\max}},$$

truncated so the run lands exactly on the output time rather than
overshooting it. Note $|u| + a$, not $|u + a|$: only the former bounds both
acoustic families when $u < 0$.

**Boundaries.** Two transmissive (zero-gradient) ghost cells per side
(Toro Sec. 6.3). Two rather than one, because the flux limiter of §4 below
reads the wave strengths at the interface one further upwind.

---

## 2. The problem with first order

The Roe flux of [roe-solver.md](roe-solver.md) is *monotone*: it produces no
spurious oscillations. That is its virtue and also its cost. A monotone scheme
carries a numerical viscosity of order $\Delta x$, so a contact discontinuity
— which has no self-steepening mechanism to resist diffusion, unlike a shock —
smears steadily as it travels. At 100 cells a Sod contact is spread over
roughly a dozen cells.

The obvious fix is a second-order scheme such as Lax–Wendroff. The obvious fix
does not work: **Godunov's theorem** states that no *linear* scheme of better
than first order can be monotonicity preserving. Lax–Wendroff duly produces
overshoots at every discontinuity.

The resolution is to be **nonlinear** — to let the scheme's own coefficients
depend on the solution, using high order where the solution is smooth and
first order where it is not.

---

## 3. The flux-limited form

Write the second-order flux as the first-order (upwind) flux plus a
correction, and multiply the correction by a limiter. Applied wave by wave to
the Roe decomposition, the interface flux becomes (Toro eq. 14.49):

$$\mathbf{F}_{i+1/2} = \mathbf{F}^{\text{Roe}}_{i+1/2} + \sum_{k=1}^{3} \tfrac{1}{2}\,\tilde{\lambda}_{k}\left[\mathrm{sign}(\nu_{k}) - \nu_{k}\right]\phi(r_{k})\,\tilde{\alpha}_{k}\,\tilde{\mathbf{K}}_{k},$$

where $\nu_{k}$ is the Courant number of wave family $k$,

$$\nu_{k} = \frac{\tilde{\lambda}_{k}\,\Delta t}{\Delta x}.$$

Two things are worth noticing about the weight
$\tfrac{1}{2}\tilde{\lambda}_{k}[\mathrm{sign}(\nu_{k}) - \nu_{k}]$.

First, it is exactly the difference between the Lax–Wendroff and upwind fluxes
for scalar advection. For $\lambda > 0$ that difference is
$\tfrac{1}{2}\lambda(1 - \nu)\Delta$; for $\lambda < 0$ it is
$-\tfrac{1}{2}\lambda(1 + \nu)\Delta$. The $\mathrm{sign}(\nu) - \nu$
form covers both.

Second — and this governs the solver's robustness — its magnitude is
proportional to $1 - |\nu_{k}|$. As $\nu \to 0$ the correction grows until it
exactly cancels the upwind numerical diffusion, and the scheme degenerates to
the dissipation-free *central* scheme. **Shrinking $\Delta t$ therefore makes
this scheme less robust, not more**, which is the opposite of the usual
intuition: the advice to lower the Courant number for robustness holds for a
first-order Godunov scheme and does not transfer here.

---

## 4. The smoothness sensor

The limiter argument $r_{k}$ measures how smooth the solution is *in wave
family $k$*, as the ratio of the upwind wave strength to the local one
(Toro §14.4):

$$r_{k} = \frac{\tilde{\alpha}_{k}^{\text{upwind}}}{\tilde{\alpha}_{k}^{\text{local}}}.$$

For the local interface $i + 1/2$, the upwind one is $i - 1/2$ when
$\nu_{k} > 0$ and $i + 3/2$ when $\nu_{k} < 0$.

Each wave family picks its own upwind direction, from the sign of its own
eigenvalue. Interpreting $r$:

| $r_{k}$ | Meaning | Desired $\phi$ |
| --- | --- | --- |
| $\approx 1$ | successive jumps are equal — the solution is smooth | $\phi \approx 1$ (full second order) |
| $\le 0$ | the jumps have opposite signs — a local extremum | $\phi = 0$ (fall back to upwind) |
| large or small | a discontinuity | bounded $\phi$, so no new extremum appears |

Because the correction is proportional to $\tilde{\alpha}_{k}^{\text{local}}$,
it vanishes wherever that wave is absent, so the ratio must be guarded
against a zero denominator rather than being allowed to produce `Inf`.

---

## 5. The limiters

A limiter is TVD if it lies inside **Sweby's region**,

$$0 \le \phi(r) \le \min(2r,\,2), \qquad \phi(r) = 0 \ \text{for}\ r \le 0,$$

and it recovers second-order accuracy only if $\phi(1) = 1$. Both properties
are checked for every limiter in `tests/test_roe.py`.

| Name | $\phi(r)$ for $r > 0$ | Character |
| --- | --- | --- |
| `superbee` | $\max\left[\min(1, 2r),\ \min(r, 2)\right]$ | The most *compressive* TVD limiter — it sits on the upper edge of Sweby's region. Keeps contacts sharpest; can mildly steepen smooth profiles into staircases. |
| `minmod` | $\min(1, r)$ | The most *diffusive* TVD limiter — the lower edge of the region. Smooth and very robust, but noticeably smeared. |
| `vanleer` | $\dfrac{2r}{1 + r}$ | Smooth and differentiable, mid-region. A good default when neither extreme is wanted. |
| `vanalbada` | $\dfrac{r(1 + r)}{1 + r^{2}}$ | Smooth, approaches 1 quickly near $r = 1$. |
| `none` | $1$ | **Not** TVD — raw Lax–Wendroff. Provided deliberately so the oscillations it produces can be demonstrated. |

The practical difference is large and is quantified in
[validation.md](validation.md#2-order-of-accuracy-and-the-limiters): on Sod at
100 cells, superbee resolves the contact in about 2 cells, minmod in about 6,
and first-order Roe in about 12.

---

## 6. What "second order" does and does not buy

A natural expectation is that the $L_1$ error should now fall as
$\mathcal{O}(\Delta x^{2})$. It does not, and this is not a defect. The exact
solution contains a shock and a contact. Near a discontinuity *any* scheme
makes an $\mathcal{O}(1)$ error over a region of width $\mathcal{O}(\Delta x)$,
contributing $\mathcal{O}(\Delta x)$ to the $L_1$ norm. The measured rate is
therefore about 1 for all the schemes here.

What the second-order correction buys is a much smaller **constant**. On
Sod's problem that is a factor of roughly 6 in the $L_1$ density error at
fixed resolution — equivalently, the same accuracy on a grid six times
coarser. The convergence figure in
[validation.md](validation.md#4-grid-convergence) shows three parallel lines
of slope $\approx -1$, separated by exactly that constant.

Recovering a genuine second-order *rate* requires a smooth test problem (one
with no shocks or contacts), where this scheme does deliver it.

---

## 7. Stability

The scheme is stable for $C_{\text{cfl}} \le 1$. In practice:

- `cfl = 0.9` — the default; recommended, and what Toro uses.
- `cfl = 0.75` — comfortably safe.
- Small values (`< 0.3`) — **not** safer, for the $1 - |\nu|$ reason in §3.
  If a run is unstable, change the limiter to `minmod` or drop to first order;
  do not simply reduce the CFL number.

---

One caveat on the base flux: the correction above is applied on the Roe
characteristic fields *whichever* first-order flux is in use, which is the
usual construction. It works for all of them except classical
Lax-Friedrichs, whose odd-even decoupling drives every limiter ratio to zero
— see [flux-functions.md](flux-functions.md#7-laxfriedrichs-decouples-the-grid).

---

**Next:** [Validation and results](validation.md)
