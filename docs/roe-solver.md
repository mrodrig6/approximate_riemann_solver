# Roe's approximate Riemann solver

> Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
> Dynamics*, 3rd ed., Springer 2009 — **Chapter 11** (the Roe approximate
> Riemann solver), §11.2 (the linearisation and the Roe averages), §11.3
> (the entropy fix), §11.4 (limitations). Chapter 6 for the finite-volume
> framework the flux is used in.

---

## 1. Why approximate at all?

The exact solver of [formulation.md](formulation.md) requires a Newton
iteration at every cell interface, at every time step. For a 200-cell run to
$t = 0.25$ that is roughly 25 000 Newton solves, each with square roots and
fractional powers. Worse, it is not *more accurate* in a finite-volume code:
the flux is used to update cell averages, and the error of the overall scheme
is dominated by the piecewise-constant reconstruction, not by the exactness of
the flux.

Roe's solver replaces the Newton iteration with a single $3 \times 3$
eigendecomposition evaluated in closed form. It is roughly an order of
magnitude cheaper and, for the problems in this repository, indistinguishable
in accuracy.

---

## 2. The Roe linearisation

Roe's idea (Toro §11.2) is to replace the nonlinear Jacobian
$\mathbf{A}(\mathbf{U}) = \partial \mathbf{F}/\partial \mathbf{U}$ by a
*constant* matrix $\tilde{\mathbf{A}}(\mathbf{U}_L, \mathbf{U}_R)$, and then
solve the resulting **linear** Riemann problem exactly. The constant matrix is
required to satisfy three conditions:

1. **Hyperbolicity.** $\tilde{\mathbf{A}}$ has real eigenvalues and a complete
   set of eigenvectors.
2. **Consistency.** $\tilde{\mathbf{A}}(\mathbf{U}, \mathbf{U}) = \mathbf{A}(\mathbf{U})$.
3. **Conservation — the "U-property".** Exactly, not approximately:

   $$\mathbf{F}(\mathbf{U}_{R}) - \mathbf{F}(\mathbf{U}_{L}) = \tilde{\mathbf{A}}\,(\mathbf{U}_{R} - \mathbf{U}_{L}). \tag{Toro 11.18}$$

Condition 3 is the heart of the method. It guarantees that a discontinuity
which is an exact solution of the nonlinear problem — an isolated shock — is
*also* an exact solution of the linearised one. That is why a Roe scheme
captures an isolated steady shock with no smearing at all, typically over one
or two cells, as the air-shock-tube figure in
[validation.md](validation.md#7-a-dimensional-case) shows.

### The Roe-average state

For an ideal gas, the unique averaging that satisfies the U-property weights by
$\sqrt{\rho}$ (Toro eqs. 11.60–11.61):

$$\tilde{u} = \frac{\sqrt{\rho_{L}}\,u_{L} + \sqrt{\rho_{R}}\,u_{R}}{\sqrt{\rho_{L}} + \sqrt{\rho_{R}}}, \qquad \tilde{H} = \frac{\sqrt{\rho_{L}}\,H_{L} + \sqrt{\rho_{R}}\,H_{R}}{\sqrt{\rho_{L}} + \sqrt{\rho_{R}}},$$

$$\tilde{a} = \sqrt{(\gamma - 1)\left(\tilde{H} - \tfrac{1}{2}\tilde{u}^{2}\right)}.$$

Note that $\tilde{H}$ is the *total* enthalpy $H = (E + p)/\rho$, not the
static one. This is `roe.roe_average`, and
`tests/test_roe.py::test_u_property` checks eq. (11.18) numerically to
$10^{-10}$ on several states, including a strong-shock pair.

### Eigenstructure

The eigenvalues and right eigenvectors of $\tilde{\mathbf{A}}$ are the usual
Euler ones, evaluated at the Roe-average state (Toro eqs. 11.64–11.65):

$$\tilde{\lambda}_{1} = \tilde{u} - \tilde{a}, \qquad \tilde{\lambda}_{2} = \tilde{u}, \qquad \tilde{\lambda}_{3} = \tilde{u} + \tilde{a},$$

$$\tilde{\mathbf{K}}_{1} = \left[1,\ \tilde{u} - \tilde{a},\ \tilde{H} - \tilde{u}\tilde{a}\right]^{T},$$

$$\tilde{\mathbf{K}}_{2} = \left[1,\ \tilde{u},\ \tfrac{1}{2}\tilde{u}^{2}\right]^{T},$$

$$\tilde{\mathbf{K}}_{3} = \left[1,\ \tilde{u} + \tilde{a},\ \tilde{H} + \tilde{u}\tilde{a}\right]^{T}.$$

### Wave strengths

Projecting the jump $\Delta\mathbf{U} = \mathbf{U}_{R} - \mathbf{U}_{L}$ onto
the eigenvectors gives the wave strengths $\tilde{\alpha}_{k}$, defined by

$$\Delta\mathbf{U} = \sum_{k=1}^{3} \tilde{\alpha}_{k}\,\tilde{\mathbf{K}}_{k}.$$

For the Euler equations this inversion is available in closed form
(Toro eqs. 11.68–11.70). Writing the components of the jump as
$\Delta_1 = \Delta\rho$, $\Delta_2 = \Delta(\rho u)$, $\Delta_3 = \Delta E$:

$$\tilde{\alpha}_{2} = \frac{\gamma - 1}{\tilde{a}^{2}}\left[\Delta_{1}\left(\tilde{H} - \tilde{u}^{2}\right) + \tilde{u}\,\Delta_{2} - \Delta_{3}\right],$$

$$\tilde{\alpha}_{1} = \frac{1}{2\tilde{a}}\left[\Delta_{1}\left(\tilde{u} + \tilde{a}\right) - \Delta_{2} - \tilde{a}\,\tilde{\alpha}_{2}\right],$$

$$\tilde{\alpha}_{3} = \Delta_{1} - \left(\tilde{\alpha}_{1} + \tilde{\alpha}_{2}\right).$$

Both this decomposition and the U-property are covered by tests.

---

## 3. The Roe flux

With the linear problem solved, the interface flux is obtained by starting
from the left state and adding the contribution of every wave that has
travelled *left* across the interface (Toro eq. 11.27):

$$\mathbf{F}_{i+1/2} = \mathbf{F}(\mathbf{U}_{L}) + \sum_{k\,:\,\tilde{\lambda}_{k} < 0} \tilde{\alpha}_{k}\,\tilde{\lambda}_{k}\,\tilde{\mathbf{K}}_{k}.$$

Equivalently, using the U-property, one may start from the right and subtract
the right-going waves; the two forms agree exactly, which is a useful internal
consistency check. Writing $\tilde{\lambda}^{-}_{k} = \min(\tilde{\lambda}_{k}, 0)$ lets both cases
share one code path, which is what `roe.roe_fluxes` does.

---

## 4. The entropy fix

### The failure mode

The Roe flux has one well-known defect (Toro §11.3). A rarefaction fan is a
*continuous* wave spanning a range of characteristic speeds, but the
linearisation represents each family by a **single** eigenvalue. When the fan
straddles the interface — a **transonic** or **sonic** rarefaction, where
$\lambda_1^{L} = u_L - a_L < 0 < \lambda_1^{*} = u^{*}_{L} - a^{*}_{L}$ — the
single eigenvalue $\tilde{\lambda}_1$ cannot change sign inside the wave. The
scheme then admits a discontinuous, entropy-violating "rarefaction shock"
sitting at the sonic point.

The symptom is unmistakable and is shown in
[validation.md](validation.md#3-the-entropy-fix): a small, fixed-size kink in
the middle of an otherwise smooth expansion fan **that does not shrink when the
grid is refined**. That non-convergence is the real signature — an ordinary
discretisation error would halve as the mesh halves.

### The Harten–Hyman fix

The remedy implemented here is Harten and Hyman's (Toro §11.3): replace the
offending eigenvalue with a speed that interpolates across the sonic point.
For a transonic *left* wave,

$$\bar{\lambda}_{1} = \lambda_{1}^{L}\,\frac{\lambda_{1}^{*} - \tilde{\lambda}_{1}}{\lambda_{1}^{*} - \lambda_{1}^{L}},$$

where $\lambda_{1}^{L} = u_{L} - a_{L}$ is the physical speed at the head of
the fan and $\lambda_{1}^{*} = u^{*}_{L} - a^{*}_{L}$ is the speed at its tail,
evaluated in the Roe star state
$\mathbf{U}^{*}_{L} = \mathbf{U}_{L} + \tilde{\alpha}_{1}\tilde{\mathbf{K}}_{1}$.
The flux becomes

$$\mathbf{F}_{i+1/2} = \mathbf{F}(\mathbf{U}_{L}) + \bar{\lambda}_{1}\,\tilde{\alpha}_{1}\,\tilde{\mathbf{K}}_{1}.$$

The mirror case, a transonic *right* wave with
$\lambda_{3}^{*} < 0 < \lambda_{3}^{R}$, uses

$$\bar{\lambda}_{3} = \lambda_{3}^{R}\,\frac{\tilde{\lambda}_{3} - \lambda_{3}^{*}}{\lambda_{3}^{R} - \lambda_{3}^{*}}, \qquad \mathbf{F}_{i+1/2} = \mathbf{F}(\mathbf{U}_{R}) - \bar{\lambda}_{3}\,\tilde{\alpha}_{3}\,\tilde{\mathbf{K}}_{3},$$

with the star state $\mathbf{U}^{*}_{R} = \mathbf{U}_{R} - \tilde{\alpha}_{3}\tilde{\mathbf{K}}_{3}$.

The two cases are mutually exclusive: a transonic wave 1 requires
$\lambda_{1}^{*} > 0$, which forces $\lambda_{3}^{*} > 0$ as well. The
implementation exploits this, and folds the right-transonic branch into the
left-based form by setting
$\tilde{\lambda}^{-}_{3} = \tilde{\lambda}_{3} - \bar{\lambda}_{3}$, so a
single code path produces the flux in every case. (The MATLAB original used
$\mathbf{F}(\mathbf{U}_{L})$ rather than $\mathbf{F}(\mathbf{U}_{R})$ in the
right-transonic branch — see [porting-notes.md](porting-notes.md).)

---

## 5. Known limitation: Roe's solver is not positivity-preserving

This matters enough to state plainly, because it is the one case in this
repository where the solver **fails outright rather than merely being
inaccurate**.

Nothing in the Roe linearisation guarantees that the updated state has
positive density and positive internal energy. When the star region approaches
vacuum, it does not: the scheme drives the internal energy negative, usually
within the first time step, and the run cannot continue.

The standard demonstration is Toro's Test 2, the "123 problem"
($\rho = 1$, $u = \mp 2$, $p = 0.4$), whose star pressure is
$p_{*} \approx 0.0019$. In this code:

```console
$ python -m riemann toro2
Roe run FAILED:
Roe scheme produced a non-physical state (negative pressure or density) at step 1 ...
```

It fails at first order as well as second, at every CFL number and with every
limiter, which is how one can tell it is a property of the flux and not of the
time-stepping or the reconstruction. This is a documented result — Einfeldt,
Munz, Roe & Sjögreen, *On Godunov-type methods near low densities*,
J. Comput. Phys. **92**:273 (1991); see also Toro §11.4.

The cure is a positivity-preserving flux: **HLL** (with Einfeldt wave speeds,
i.e. HLLE) or **HLLC**, Toro Chapter 10. Both are implemented — see
[flux-functions.md](flux-functions.md) — and both run this problem:

```bash
python -m riemann toro2 --flux hllc
```

The test suite pins both halves of this: `test_toro2_fails_as_documented` for
the Roe flux, and `test_hllc_solves_the_problem_roe_cannot` for the remedy.

---

## 6. The finite-volume driver

The flux is used in the standard explicit conservative update (Toro Sec. 6.2):

$$\mathbf{U}_{i}^{\,n+1} = \mathbf{U}_{i}^{\,n} - \frac{\Delta t}{\Delta x}\left[\mathbf{F}_{i+1/2} - \mathbf{F}_{i-1/2}\right].$$

**Time step.** Set by the CFL condition (Toro §6.4) using the maximum signal
speed over the grid,

$$S_{\max} = \max_{i}\left(|u_{i}| + a_{i}\right), \qquad \Delta t = \frac{C_{\text{cfl}}\,\Delta x}{S_{\max}},$$

truncated so the run lands exactly on the output time. Note $|u| + a$, not
$|u + a|$: the latter misses the left-going family when $u < 0$, and was a bug
in the MATLAB original.

**Boundaries.** Two transmissive (zero-gradient) ghost cells per side
(Toro Sec. 6.3). Two, not one, because the flux limiter of
[tvd-extension.md](tvd-extension.md) reads the wave strengths at the interface
one further upwind.

---

**Next:** [Numerical flux functions](flux-functions.md) ·
[The second-order TVD extension](tvd-extension.md)
