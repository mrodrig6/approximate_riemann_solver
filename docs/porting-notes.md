# Porting notes: MATLAB to Python

The original MATLAB implementation is archived under
[`matlab_legacy/`](../matlab_legacy/). This page records what moved where, and
every behavioural change — in particular the bugs that were fixed along the
way, since a reader comparing the two codes will want to know which
differences are deliberate.

---

## 1. File mapping

| MATLAB | Python | Notes |
| --- | --- | --- |
| `main.m` | `examples/sod_shock_tube.py`, `python -m riemann sod` | Hard-coded parameters became arguments. |
| `test_input2o1.m` | `examples/dimensional_shock_tube.py`, `python -m riemann air_shock_tube` | |
| `src/exact_solver/f_ps.m` | `exact.pressure_function` | |
| `src/exact_solver/fp_ps.m` | `exact.pressure_function_derivative` | Exponent corrected, see §3.1. |
| `src/exact_solver/root_find.m` | `exact.solve_star_pressure` | Rewritten, see §3.2. |
| `src/exact_solver/plot_rp.m` | `exact.solve_exact` + `plotting.plot_comparison` | Computation split from plotting. |
| `src/approximate_solver/roe_flux_e.m` | `roe.roe_fluxes` | See §4. |
| `src/approximate_solver/flx_lim.m` | `limiters.py` + the limiter block of `roe.roe_fluxes` | |
| `src/approximate_solver/plot_roe_eue.m` | `solver.solve_roe` + `plotting.plot_comparison` | Computation split from plotting. |
| — | `thermo.py`, `problems.py`, `cli.py`, `_jit.py` | New. |

---

## 2. Structural changes

**Computation separated from plotting.** `plot_rp.m` and `plot_roe_eue.m` each
solved *and* drew, so there was no way to obtain the numbers without producing
a figure, and no way to overlay a third curve without editing the solver.
Solvers now return plain data objects (`ExactSolution`, `RoeSolution`) and
`riemann/plotting.py` draws them. This is what made the limiter-comparison and
grid-convergence figures possible at all.

**Wave strengths computed once per step.** `flx_lim.m` recomputed the full Roe
average of the *upwind* interface on every call, and was called once per wave
family per interface — three redundant Roe averages per interface per step.
`roe.interface_eigensystem` now fills eigenvalues, eigenvectors and wave
strengths for every interface in a single pass.

**Proper ghost cells.** The MATLAB time loop simply never updated cells 1, 2,
`NX-1` and `NX`, which is adequate only while no wave reaches the boundary.
The Python driver carries two transmissive ghost cells per side (Toro
eq. 6.21) and updates every interior cell. Two per side, not one, because the
limiter reads one interface beyond the first interior interface. Toro's Test 5
is a case where a wave *does* leave the domain.

**Numba.** The flux kernels are decorated with `njit` from `riemann/_jit.py`,
which falls back to a no-op decorator when Numba is absent. The measured
difference is about 0.29 µs/cell/step compiled versus 29.7 µs/cell/step in
pure Python, i.e. roughly 100×. The whole test suite runs in 0.7 s compiled
and 17 s interpreted.

**A test suite and a problem catalogue.** Toro's five test problems plus a
transonic case and the dimensional one are available by name, and 61 tests
check the solvers against Toro's published values and against their own
defining identities.

---

## 3. Bugs fixed in the exact solver

### 3.1 Wrong exponent in the rarefaction derivative

`fp_ps.m` computed the rarefaction branch of $f_{K}'(p)$ as

```matlab
fl = (1/(rl*cl))*(p/pl)^(-(g-1)/(2*g));   % wrong exponent
```

The correct exponent (Toro eq. 4.37, differentiating eq. 4.7) is
$-(\gamma + 1)/(2\gamma)$:

$$f_{K}'(p) = \frac{1}{\rho_{K} a_{K}}\left(\frac{p}{p_{K}}\right)^{-\frac{\gamma + 1}{2\gamma}}.$$

With the wrong derivative, Newton's method degenerates from quadratic to
linear convergence. It still found the root for Sod's problem — which is why
the bug survived — but slowly and only from a fortunate starting point.
`tests/test_exact.py::test_pressure_function_derivative_matches_finite_difference`
is the regression guard.

### 3.2 The Newton iteration

`root_find.m` read:

```matlab
rerr = 1;
while rerr > tol
    pnew = pg - f_ps(pg,g,WL,WR)/fp_ps(pg,g,WL,WR);
    rerr = abs(f_ps(pnew,g,WL,WR) - f_ps(pg,g,WL,WR));
    pg = pnew;
end
```

Four problems, all fixed:

- **The convergence test is an absolute difference of residuals.** With
  `tol = 1e-15` and the dimensional shock tube at $p \approx 10^{5}$ Pa, that
  tolerance is below the floating-point resolution of the residual and can
  never be met. The Python version tests the *relative* pressure change,
  $2|p_{\text{new}} - p_{\text{old}}| / (p_{\text{new}} + p_{\text{old}})$,
  which is dimensionally consistent.
- **No iteration cap**, so a bad input is an infinite loop in silence. Now
  capped, with an informative exception.
- **No positivity guard.** Newton on a concave function can overshoot to
  $p \le 0$, after which the fractional powers go complex. The update is now
  clamped.
- **A hard-coded initial guess of `p = 0.5`**, passed in by both drivers. It
  is nowhere near the answer for a problem at $10^{5}$ Pa. Replaced by Toro's
  adaptive PVRS / two-rarefaction / two-shock guess (§9.5).
  `test_newton_is_insensitive_to_the_initial_guess` checks that the result no
  longer depends on it, over 10 decades of starting value.

### 3.3 Gaps in the sampler

`plot_rp.m` handled the left and right halves of the wave fan in two separate
`if` blocks, each using strict inequalities:

```matlab
if x(i) < Sl*T                       % left of the left shock
elseif (x(i) > Sl*T && x(i) < usl*T) % star region
end
```

A sample point landing exactly on a wave — most easily $x = u_{*}t$, the
contact itself — matched no branch and was left at its initialised value of
zero. The Python sampler partitions the whole $x/t$ axis with no gaps and no
overlaps. `test_sampler_leaves_no_gaps` and
`test_sampler_hits_the_contact_exactly` guard this.

### 3.4 No vacuum check

Data satisfying Toro's vacuum condition (eq. 4.40) have no star region and
hence no positive root of $f(p)$. The MATLAB code would iterate into complex
arithmetic. `solve_star_pressure` now raises a `ValueError` naming the
condition.

---

## 4. Bugs fixed in the Roe solver

### 4.1 Wrong base state in the right-transonic entropy fix

`roe_flux_e.m` had:

```matlab
elseif (lamrs < 0 && lamr > 0)
    flx = fl - lam3b*al3*K(3,:) + ...   % `fl` should be the RIGHT flux
```

Harten and Hyman's fix for a transonic *right* wave is built from the right
physical flux:
$\mathbf{F} = \mathbf{F}(\mathbf{U}_{R}) - \bar{\lambda}_{3}\tilde{\alpha}_{3}\tilde{\mathbf{K}}_{3}$.
Using $\mathbf{F}(\mathbf{U}_{L})$ gives a locally inconsistent flux. The
Python version writes both transonic branches in a single $\mathbf{F}_L$-based
form, using the U-property to convert the right-transonic case — see
[roe-solver.md](roe-solver.md#4-the-entropy-fix).

### 4.2 Mixed-up star states

The MATLAB computed `ustar` and `pstar` from the *left* star state
$\mathbf{U}_L + \tilde{\alpha}_1\tilde{\mathbf{K}}_1$, then reused those values
with the *right* star density `rrstar = rr - al3`. Each star state is now
built from its own conservative state.

### 4.3 The second-order term was dropped at transonic interfaces

In the MATLAB, the two entropy-fix branches returned early and applied the
limiter correction only to the single transonic wave, while the ordinary
branch applied it to all three. The order of accuracy therefore dropped
silently wherever the entropy fix triggered. The correction is now applied
uniformly to all three families.

### 4.4 Unguarded division in the limiter

`flx_lim.m` computed `theta = alp(j)/alpha(j)` with no guard. Wherever a wave
family carries no jump — a uniform region, or the entropy wave in isentropic
flow — this is `0/0` and yields `NaN`, which then propagates through the whole
solution. The ratio is now guarded; since the correction is itself
proportional to the local wave strength, falling back to $r = 0$ there is both
safe and exact.

---

## 5. Bugs fixed in the time loop

### 5.1 Wrong maximum signal speed

`plot_roe_eue.m` set the time step from

```matlab
a_calc = @(usl) usl(:,2)./usl(:,1) + sqrt(...);   % u + a
dt_calc = @(usl) cfl*dx/max(abs(a_calc(usl)));    % max |u + a|
```

$\max|u + a|$ is not the maximum signal speed: the left-going acoustic family
travels at $u - a$, so for $u < 0$ the fastest wave is missed and the step is
too large. Corrected to $\max(|u| + a)$ (Toro §6.4). Toro's Test 5, which has
a uniform background velocity of $-19.6$, is exactly the case that exposes it.

### 5.2 Overshooting the output time

The loop ran `while t < T` with no clipping, so the reported solution was at
some $t > T$, up to one full step late. The step is now truncated so the run
lands exactly on $T$; `test_solution_lands_exactly_on_the_output_time` checks
it.

### 5.3 No safety checks

No step cap and no finiteness check, so a diverging run was an infinite loop
producing `NaN`s in silence. Both are now present, and the positivity failure
raises with an explanation of the Roe near-vacuum limitation rather than a
bare error.

---

## 6. The CFL ramp that broke Test 5

This one was introduced *during* the port and is recorded because the reasoning
behind it is a genuinely counter-intuitive property of flux-limited schemes.

Toro (§6.4) suggests reducing the Courant number over the first few steps of a
Godunov-type run, since the wave speeds in the discontinuous initial data
under-predict those that appear immediately afterwards. An early version of
this port duly multiplied the first five time steps by 0.2. Toro's Test 5 then
failed with negative pressure at step 2, while Test 3 — the same states with a
different background velocity — ran fine.

The diagnosis was that the failure was **independent of both the CFL number
and the limiter**, and got *worse* as the CFL number was reduced. That rules
out an ordinary stability problem. The cause is the weight in the second-order
correction (Toro eq. 14.49):

$$\tfrac{1}{2}\tilde{\lambda}_{k}\left[\mathrm{sign}(\nu_{k}) - \nu_{k}\right] = \tfrac{1}{2}|\tilde{\lambda}_{k}|\left(1 - |\nu_{k}|\right) \times (\pm 1).$$

As $\nu \to 0$ the correction grows until it exactly cancels the upwind
numerical diffusion, and the scheme becomes the dissipation-free central
scheme. Shrinking $\Delta t$ removes the very dissipation that holds a
$10^{5}$ pressure ratio together.

The ramp was removed. Advice that is sound for a first-order Godunov scheme
does not transfer to a flux-limited one — if such a run is unstable, change
the limiter, not the CFL number.

---

## 7. Behaviour that is intentionally unchanged

- The limiter flag `0 = superbee`, `1 = minmod` is still accepted, so old
  driver parameters translate directly (`solve_roe(..., limiter=0)`).
- The default CFL of 0.9 and the default 200 cells match the MATLAB drivers.
- The four plotted fields are the same four, in the same order:
  $\rho$, $u$, $p$ and specific internal energy $e$.
- Results for Sod's problem agree with the MATLAB output to plotting accuracy.

---

**Back to:** [Documentation index](index.md)
