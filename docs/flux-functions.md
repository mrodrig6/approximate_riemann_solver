# Numerical flux functions

> Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
> Dynamics*, 3rd ed., Springer 2009 — **Chapter 10** (HLL and HLLC, and the
> wave-speed estimates), **Sec. 5.3.4** (Lax–Friedrichs), **Sec. 10.5**
> (Rusanov). AUSM+-up is Liou, *A sequel to AUSM, Part II: AUSM+-up for all
> speeds*, J. Comput. Phys. **214**:137 (2006), with the split polynomials
> from Liou, J. Comput. Phys. **129**:364 (1996). Roe has its own page:
> [roe-solver.md](roe-solver.md).

The conservative update of
[finite-volume-scheme.md](finite-volume-scheme.md#1-the-conservative-update)
needs one thing from a Riemann solver: a single flux vector at each cell
interface. Everything else — the time step, the boundary conditions, the TVD
correction — is independent of how that flux is formed. `riemann/fluxes/`
therefore treats the flux as a plug-in, one module per family, selected with
`flux=`:

```python
solve_roe(gamma, w_l, w_r, t_end, nx=200, flux="hllc")
```

```bash
python -m riemann sod --flux hllc
```

---

## 1. The three families

| Flux | `flux=` | Waves | Module | Source |
| --- | --- | --- | --- | --- |
| Roe (+ Harten–Hyman fix) | `roe` | 3 | `fluxes/roe.py` | Toro Ch. 11 |
| HLLC | `hllc` | 3 | `fluxes/hll.py` | Toro Sec. 10.4 |
| HLL / HLLE | `hll`, `hlle` | 2 | `fluxes/hll.py` | Toro Secs. 10.3, 10.5 |
| AUSM+-up | `ausmup`, `ausm` | 3 (split) | `fluxes/ausm.py` | Liou 2006 |
| Rusanov (local LF) | `rusanov`, `llf` | 0 (scalar) | `fluxes/central.py` | Toro Sec. 10.5 |
| Lax–Friedrichs | `lax_friedrichs`, `lf` | 0 (scalar) | `fluxes/central.py` | Toro Sec. 5.3.4 |

**Central / Rusanov-type.** A central average plus one scalar dissipation
coefficient applied to the whole state vector:

$$\mathbf{F} = \tfrac{1}{2}\left(\mathbf{F}_L + \mathbf{F}_R\right) - \tfrac{1}{2} S\left(\mathbf{U}_R - \mathbf{U}_L\right).$$

Lax–Friedrichs takes $S = \Delta x / \Delta t$, the *global* mesh speed;
Rusanov takes the *local* maximum signal speed
$S = \max(|u_L| + a_L, |u_R| + a_R)$. Robust and trivial, but the same large
coefficient hits every field, so contacts smear badly.

**HLL-type.** Assume an approximate wave structure and average the exact
integral form across it. HLL uses two waves and one intermediate state; HLLC
adds the contact back.

**Flux-vector / flux-difference.** Roe decomposes the jump into
characteristic waves; AUSM+-up splits the flux into a convective part carried
by an interface velocity and a separate pressure part. Both resolve the
contact exactly.

---

## 2. HLL

Assume the fan contains just a left wave of speed $S_L$ and a right wave of
speed $S_R$, with one constant state between them. Integrating the
conservation law over a control volume spanning the fan gives (Toro eq. 10.21)

$$\mathbf{F}^{\text{HLL}} = \frac{S_R \mathbf{F}_L - S_L \mathbf{F}_R + S_L S_R\left(\mathbf{U}_R - \mathbf{U}_L\right)}{S_R - S_L}$$

when $S_L \le 0 \le S_R$, and simply the upwind physical flux otherwise.

### Wave-speed estimates matter

The code uses **Einfeldt's** estimates (Toro Sec. 10.5), which is what makes
this the HLLE variant:

$$S_L = \min\left(u_L - a_L, \tilde{u} - \tilde{a}\right), \qquad S_R = \max\left(u_R + a_R, \tilde{u} + \tilde{a}\right),$$

with $\tilde{u}, \tilde{a}$ the Roe averages. Taking the minimum over *both*
the physical and the Roe speed is what guarantees the estimates bracket every
wave in the exact solution, and that bracketing is what makes HLL
positivity-preserving. The cheaper Davis estimate, using only the physical
speeds, can fail to bracket a strong shock.

**The cost of two waves.** With no middle wave in the assumed structure, the
contact is averaged away. HLL smears a contact *more than first-order Roe
does* — 8 cells against 2 on Sod at 100 cells.

---

## 3. HLLC

HLLC restores the middle wave. Requiring pressure and normal velocity to be
continuous across it gives the contact speed (Toro eq. 10.37)

$$S_{\ast} = \frac{p_R - p_L + \rho_L u_L\left(S_L - u_L\right) - \rho_R u_R\left(S_R - u_R\right)}{\rho_L\left(S_L - u_L\right) - \rho_R\left(S_R - u_R\right)},$$

and the Rankine–Hugoniot conditions across the outer waves give the two star
states (Toro eq. 10.39), for $K \in \{L, R\}$:

$$\mathbf{U}^{\ast}_{K} = \rho_K \frac{S_K - u_K}{S_K - S_{\ast}}\left[1, S_{\ast}, \frac{E_K}{\rho_K} + \left(S_{\ast} - u_K\right)\left(S_{\ast} + \frac{p_K}{\rho_K\left(S_K - u_K\right)}\right)\right]^{T},$$

with $\mathbf{F}^{\ast}_{K} = \mathbf{F}_K + S_K\left(\mathbf{U}^{\ast}_{K} - \mathbf{U}_K\right)$.
The flux is then selected by the sign of $S_L$, $S_{\ast}$ and $S_R$.

**This is the best general-purpose choice in this code.** It resolves contacts
essentially as well as Roe, keeps HLL's positivity, needs no entropy fix, and
is measurably cheaper than Roe (0.33 against 0.51 µs per cell per step).

---

## 4. AUSM+-up

AUSM-family schemes do not decompose the jump at all. They split the flux into
a convective part carried by an interface *velocity* and a separate pressure
part:

$$\mathbf{F} = u_{1/2} \mathbf{\Psi}_{\text{upwind}} + \left[0, p_{1/2}, 0\right]^{T}, \qquad \mathbf{\Psi} = \left[\rho, \rho u, E + p\right]^{T},$$

with $\mathbf{\Psi}$ taken from the left when $u_{1/2} > 0$ and from the right
otherwise. The interface velocity and pressure are built from polynomial
splittings of each side's Mach number $M_K = u_K / a_{1/2}$:

$$u_{1/2} = a_{1/2}\left[M^{+}_{(4)}(M_L) + M^{-}_{(4)}(M_R)\right] + K_p \max\left(1 - \bar{M}^{2}, 0\right)\frac{p_L - p_R}{\rho_{1/2} a_{1/2}},$$

$$p_{1/2} = P^{+}_{(5)}(M_L) p_L + P^{-}_{(5)}(M_R) p_R + K_u P^{+}_{(5)}P^{-}_{(5)} \rho_{1/2} a_{1/2}\left(u_L - u_R\right),$$

where $a_{1/2} = \tfrac{1}{2}(a_L + a_R)$,
$\rho_{1/2} = \tfrac{1}{2}(\rho_L + \rho_R)$ and
$\bar{M}^{2} = \tfrac{1}{2}(M_L^{2} + M_R^{2})$.

The two extra terms are the "-up" of the name. $K_p$ is a pressure-diffusion
term in the velocity, $K_u$ a velocity-diffusion term in the pressure, and
the factor $\max(1 - \bar{M}^2, 0)$ switches the first off once the interface
goes supersonic, where upwinding already supplies enough dissipation.

### Why the $K_p$ term is not optional

Without it — that is, for plain AUSM+ (Liou 1996) — the mass flux is
identically zero whenever $u_L = u_R = 0$, since
$M^{+}_{(4)}(0) + M^{-}_{(4)}(0) = 0$ exactly, however large the pressure jump
is. The momentum equation still feels the pressure difference and accelerates
the gas, but with the energy flux also zero the growing kinetic energy is
subtracted from a fixed total energy, and on the light side of a shock tube
the internal energy goes negative in the first step.

Plain AUSM+ therefore **cannot run Sod's problem at all**. The $K_p$ term
supplies the missing pressure–velocity coupling, and AUSM+-up runs it without
difficulty. `test_ausm_up_starts_from_rest` is the guard.

### The split polynomials

$M^{\pm}_{(4)}$ and $P^{\pm}_{(5)}$ are the standard AUSM+ ones, with
$\beta = 1/8$ and $\alpha = 3/16$. These are not tunable — they are the
unique values making polynomials of the stated degrees smooth at $|M| = 1$ —
and they are pinned by two exact identities,

$$M^{+}_{(4)}(M) + M^{-}_{(4)}(M) = M, \qquad P^{+}_{(5)}(M) + P^{-}_{(5)}(M) = 1,$$

which hold for *every* $M$ and independently of $\alpha$ and $\beta$. Both are
checked directly in `tests/test_fluxes.py`. Expanded, they give the closed
forms, for $|M| < 1$:

$$M^{+}_{(4)} = \tfrac{1}{8}M^{4} + \tfrac{1}{2}M + \tfrac{3}{8},$$

$$P^{+}_{(5)} = \tfrac{3}{16}M^{5} - \tfrac{5}{8}M^{3} + \tfrac{15}{16}M + \tfrac{1}{2}.$$

### Where the coefficients come from

$K_p = 0.4$ and $K_u = 0.2$, and the arithmetic-mean $a_{1/2}$, follow the
benchmarked implementation in **EMS** (the Eulerian Multicomponent Solver of
the same research group), `src/riemann_solvers/ausm6.cpp`.

They matter, and they are worth stating separately from $\alpha$ and $\beta$,
because the two groups have different standing. $\alpha$ and $\beta$ are
fixed by the polynomial construction and are pinned by the identities above.
$K_p$ and $K_u$ are *diffusion coefficients*, and those identities say
nothing about them — both hold for every value. A $K_u$ that is too large is
anti-dissipative rather than merely inaccurate: it destabilises a moving
contact, amplifying round-off until the pressure error swamps the solution.

What constrains them is a physical invariant rather than an algebraic one: a
contact advected at uniform speed must not generate pressure.
`test_moving_contact_keeps_pressure_constant` checks exactly that, and is the
guard against the coefficients drifting.

### What is still not implemented: SLAU

The **SLAU** scheme (Shima & Kitamura, AIAA J. **49**:1693, 2011) is not
included: its defining functions $g$ and $\chi$ are not verified here, and
guessing at a scheme's diffusion functions is exactly the mistake the
previous section warns against. `riemann/fluxes/` is structured so it slots
in as another module beside `ausm.py` when the formulation is to hand.

## 5. Measured behaviour

Sod's problem. Contact width is the number of cells with a density strictly
between the two plateau values, at 100 cells with the superbee limiter;
the $L_1$ error is at 200 cells; the cost is at 800 cells.

| Flux | Contact width | $L_1$ error in $\rho$ | µs / cell / step |
| --- | --- | --- | --- |
| Roe | 2 cells | $1.44 \times 10^{-3}$ | 0.48 |
| HLLC | 3 cells | $2.10 \times 10^{-3}$ | 0.31 |
| AUSM+-up | 4 cells | $6.59 \times 10^{-3}$ | 0.24 |
| HLL | 8 cells | $4.04 \times 10^{-3}$ | 0.35 |
| Rusanov | 19 cells | $1.28 \times 10^{-2}$ | 0.36 |
| Lax–Friedrichs | 24 cells | $2.22 \times 10^{-2}$ | 0.34 |

![Numerical fluxes at the contact discontinuity](figures/flux_comparison.png)

For the wave-based fluxes the ordering is not a coincidence or a tuning
artefact: it follows directly from how much of the wave structure each
resolves. Every one of them captures the *shock* in one or two cells — it is
the contact that separates them, exactly as the theory says it should.
`tests/test_fluxes.py::test_accuracy_ordering_on_sod` asserts
Roe < HLLC < HLL < Rusanov < Lax–Friedrichs, so it cannot silently regress.

AUSM+-up sits outside that ordering, and the table shows why it is worth
having: it resolves the contact in 4 cells — second only to Roe and HLLC —
and it is the **cheapest** flux here, because it never forms an eigensystem
or a wave-speed estimate. Its larger $L_1$ error is accumulated away from the
contact, in the rarefaction fan.

---

## 6. Robustness

Which fluxes run which problems, at second order with superbee:

| Problem | Roe | HLLC | HLL | AUSM+-up | Rusanov | LF |
| --- | --- | --- | --- | --- | --- | --- |
| `sod` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `toro2` (near vacuum) | ✗ | ✓ | ✓ | ✗ 2nd | ✓ | ✓ |
| `toro3` | ✓ | ✓ | ✓ | ✗ 2nd | ✓ | ✓ |
| `toro4` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `toro5` | ✓ | ✓ | ✓ | ✗ 2nd | ✗ 2nd | ✓ |
| `sonic_rarefaction` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `air_shock_tube` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

"✗ 2nd" means the *flux* runs the problem but the second-order scheme does
not: every one of those cases succeeds with `second_order=False`. Only Roe on
`toro2` is a failure of the flux itself.

Three entries deserve comment.

**Roe on `toro2`** is the near-vacuum breakdown described in
[roe-solver.md](roe-solver.md#5-known-limitation-roes-solver-is-not-positivity-preserving).
HLL and HLLC run it, which is the whole point of a positivity-preserving
flux, and resolves the long-standing roadmap item in this repository:

![Toro Test 2 with HLLC and HLL](figures/toro2_positivity.png)

Density and velocity are captured well. The overshoot in the specific
internal energy near $x = 0.5$ is the known difficulty of this test — the
star region is at $p_{\ast} \approx 0.0019$, so $e = p / ((\gamma - 1)\rho)$ is a
ratio of two very small numbers — and it appears for every scheme, not only
these two.

**Rusanov on `toro5`** is a useful distinction. Rusanov is
positivity-preserving *as a first-order flux*, and at first order it runs
`toro5` without trouble. The failure comes from the second-order TVD
correction, which the flux's positivity proof does not cover. "The flux is
positivity-preserving" and "the scheme is positivity-preserving" are
different statements, and only `hll`, `hllc` and `lax_friedrichs` are listed
in `riemann.fluxes.ROBUST_FLUXES`.

**AUSM+-up on `toro2`, `toro3` and `toro5`** is the same story. The flux runs
*every* built-in problem at first order —
`test_ausm_up_runs_every_problem_at_first_order` asserts exactly that, and on
`toro5` it is the most accurate first-order flux of the six. What fails is the
pairing with this package's characteristic-based TVD correction on the three
most extreme cases. That pairing is a choice of this code, not of the scheme:
a WENO reconstruction is the more natural companion for a flux that never
forms an eigensystem, and is what EMS uses.

---

## 7. Lax–Friedrichs decouples the grid

Classical Lax–Friedrichs has one further property worth knowing, because it
shows up immediately if you try to use it: **the second-order TVD correction
has no effect on it at all.**

Its dissipation $\tfrac{1}{2}(\Delta x/\Delta t)(\mathbf{U}_R - \mathbf{U}_L)$
acts on a $2\Delta x$ stencil, which leaves the even and odd sub-grids
uncoupled. The solution develops identical adjacent cell pairs, so at every
interface either the local wave strength or the upwind one is exactly zero.
Either way the limiter ratio $r_k$ is driven to zero, $\phi = 0$, and the
correction vanishes identically — measured over a whole Sod run, the
correction weight was nonzero at **0 of 603** interface-wave pairs, and the
second-order and first-order solutions agreed to $8 \times 10^{-14}$.

This is classical odd–even decoupling, and
`test_lax_friedrichs_odd_even_decoupling_disables_the_limiter` pins it. The
local-dissipation flux, Rusanov, does not suffer from it, which is one more
reason to prefer it over Lax–Friedrichs whenever a cheap robust flux is
wanted.

---

## 8. Which to use

- **HLLC** unless there is a reason not to. Best accuracy-to-robustness ratio,
  cheaper than Roe, no entropy fix needed, runs everything.
- **Roe** when the sharpest possible shock matters — it is exact for an
  isolated shock — and the problem is not near vacuum.
- **HLL** when positivity matters more than contact resolution.
- **Rusanov** as a cheap, simple fallback; first order only if you need its
  positivity guarantee.
- **AUSM+-up** when speed matters: it is the cheapest flux here and resolves
  contacts well. Prefer `second_order=False` on very strong shocks, or pair
  it with a WENO reconstruction.
- **Lax–Friedrichs** for teaching, as the baseline the others are measured
  against.

---

**Next:** [Roe's solver in detail](roe-solver.md) ·
[The finite-volume scheme](finite-volume-scheme.md)

**Back to:** [Documentation index](index.md)
