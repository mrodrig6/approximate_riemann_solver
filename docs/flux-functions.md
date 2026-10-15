# Numerical flux functions

> Reference: E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid
> Dynamics*, 3rd ed., Springer 2009 — **Chapter 10** (HLL and HLLC, and the
> wave-speed estimates), **Sec. 5.3.4** (Lax–Friedrichs), **Sec. 10.5**
> (Rusanov). AUSM+ is Liou, *A sequel to AUSM: AUSM+*,
> J. Comput. Phys. **129**:364 (1996). Roe has its own page:
> [roe-solver.md](roe-solver.md).

The finite-volume driver of [roe-solver.md](roe-solver.md#6-the-finite-volume-driver)
needs one thing from a Riemann solver: a single flux vector at each cell
interface. Everything else — the time step, the boundary conditions, the TVD
correction — is independent of how that flux is formed. `riemann/fluxes.py`
therefore treats the flux as a plug-in, selected with `flux=`:

```python
solve_roe(gamma, w_l, w_r, t_end, nx=200, flux="hllc")
```

```bash
python -m riemann sod --flux hllc
```

---

## 1. The three families

| Flux | `flux=` | Waves resolved | Source |
| --- | --- | --- | --- |
| Roe (+ Harten–Hyman fix) | `roe` | 3 | Toro Ch. 11 |
| HLLC | `hllc` | 3 | Toro Sec. 10.4 |
| HLL / HLLE | `hll`, `hlle` | 2 | Toro Secs. 10.3, 10.5 |
| AUSM+ | `ausm` | 3 (split) | Liou 1996 |
| Rusanov (local LF) | `rusanov`, `llf` | 0 (scalar) | Toro Sec. 10.5 |
| Lax–Friedrichs | `lax_friedrichs`, `lf` | 0 (scalar) | Toro Sec. 5.3.4 |

**Central / Rusanov-type.** A central average plus one scalar dissipation
coefficient applied to the whole state vector:

$$\mathbf{F} = \tfrac{1}{2}\left(\mathbf{F}_L + \mathbf{F}_R\right) - \tfrac{1}{2} S\left(\mathbf{U}_R - \mathbf{U}_L\right).$$

Lax–Friedrichs takes $S = \Delta x / \Delta t$, the *global* mesh speed;
Rusanov takes the *local* maximum signal speed
$S = \max(|u_L| + a_L,\ |u_R| + a_R)$. Robust and trivial, but the same large
coefficient hits every field, so contacts smear badly.

**HLL-type.** Assume an approximate wave structure and average the exact
integral form across it. HLL uses two waves and one intermediate state; HLLC
adds the contact back.

**Flux-vector / flux-difference.** Roe decomposes the jump into
characteristic waves; AUSM+ splits the flux into a convective part carried by
a mass flux and a separate pressure part. Both resolve the contact exactly.

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

$$S_L = \min\left(u_L - a_L,\ \tilde{u} - \tilde{a}\right), \qquad S_R = \max\left(u_R + a_R,\ \tilde{u} + \tilde{a}\right),$$

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

$$S_* = \frac{p_R - p_L + \rho_L u_L\left(S_L - u_L\right) - \rho_R u_R\left(S_R - u_R\right)}{\rho_L\left(S_L - u_L\right) - \rho_R\left(S_R - u_R\right)},$$

and the Rankine–Hugoniot conditions across the outer waves give the two star
states (Toro eq. 10.39), for $K \in \{L, R\}$:

$$\mathbf{U}^{*}_{K} = \rho_K \frac{S_K - u_K}{S_K - S_*}\left[1,\ S_*,\ \frac{E_K}{\rho_K} + \left(S_* - u_K\right)\left(S_* + \frac{p_K}{\rho_K\left(S_K - u_K\right)}\right)\right]^{T},$$

with $\mathbf{F}^{*}_{K} = \mathbf{F}_K + S_K\left(\mathbf{U}^{*}_{K} - \mathbf{U}_K\right)$.
The flux is then selected by the sign of $S_L$, $S_*$ and $S_R$.

**This is the best general-purpose choice in this code.** It resolves contacts
essentially as well as Roe, keeps HLL's positivity, needs no entropy fix, and
is measurably cheaper than Roe (0.33 against 0.51 µs per cell per step).

---

## 4. AUSM+

AUSM-family schemes do not decompose the jump at all. They split the flux into
a convective part carried by an interface mass flux and a separate pressure
part:

$$\mathbf{F} = \dot{m}_{1/2}\,\mathbf{\Psi}_{\text{upwind}} + \left[0,\ p_{1/2},\ 0\right]^{T}, \qquad \mathbf{\Psi} = \left[1,\ u,\ H\right]^{T},$$

with the interface Mach number and pressure built from polynomial splittings
of each side's Mach number,

$$M_{1/2} = M^{+}_{(4)}(M_L) + M^{-}_{(4)}(M_R), \qquad p_{1/2} = P^{+}_{(5)}(M_L)\,p_L + P^{-}_{(5)}(M_R)\,p_R.$$

The split polynomials satisfy two exact identities,

$$M^{+}_{(4)}(M) + M^{-}_{(4)}(M) = M, \qquad P^{+}_{(5)}(M) + P^{-}_{(5)}(M) = 1,$$

which hold for every $M$ and independently of the constants $\beta = 1/8$ and
$\alpha = 3/16$. Both are checked directly in `tests/test_fluxes.py`, and
between them they pin the polynomial forms tightly.

Because pressure is handled entirely by the separate $p_{1/2}$ term, the
convective part carries no pressure dissipation, and AUSM+ reproduces a
contact — stationary **or moving** — exactly.

### It cannot start a flow from rest

AUSM+ has no pressure–velocity coupling in the *mass* flux. At an interface
with $u_L = u_R = 0$ the splittings give $M_{1/2} = 0$ exactly, however large
the pressure jump is, so both the mass flux and the energy flux
$\dot{m}H$ vanish. The momentum equation still feels the pressure difference
and accelerates the gas, but with the total energy held fixed, the growing
kinetic energy is subtracted from the internal energy. On the light side of a
shock tube the same impulse produces a far larger velocity, and the internal
energy goes negative in the first step.

Measured here on Sod with a uniform initial velocity $u_0$:

| $u_0$ | 0 | $10^{-6}$ | $10^{-3}$ | $0.01$ | $0.1$ | $0.5$ |
| --- | --- | --- | --- | --- | --- | --- |
| AUSM+ | fails | fails | fails | fails | runs | runs |

So it needs a local Mach number of roughly 0.1 to function. It runs the cases
in this repository that begin with moving gas (`toro2`, `toro5`,
`sonic_rarefaction`, `air_shock_tube`) perfectly well.

This is not a defect of the implementation — it is precisely the deficiency
that **AUSM+-up** (Liou 2006) was written to cure, by adding a
pressure-difference term $M_p$ to the mass flux.

### What is not implemented: AUSM+-up and SLAU

Neither the `-up` low-Mach extension of AUSM+ nor the SLAU scheme
(Shima & Kitamura, AIAA J. **49**:1693, 2011) is included, and the reason is
worth stating plainly rather than leaving as a silent omission.

The environment this work was done in allows outbound network access to
GitHub and the package registries only; the 2006 and 2011 papers, and every
secondary source that restates their coefficients in full, were unreachable.
Their defining pieces — AUSM+-up's $M_p$ and $p_u$ terms and the function
$f_a$, SLAU's functions $g$ and $\chi$ — could therefore not be checked.

A reconstruction of AUSM+-up's $p_u$ term from memory was implemented and
tested, and it is **wrong**: on a plain moving contact it amplified round-off
by a factor of about 1.5 per step until the pressure error reached 50%, and
the measured stability threshold sat near $K_u = 0.35$ against the published
value of $0.75$. The sign agrees with the acoustic Riemann solution, so the
error is structural rather than a slipped minus sign, and it could not be
resolved without the source. Shipping an unverifiable term that is
demonstrably unstable on a textbook problem would be worse than shipping the
well-established scheme it extends.

Both are straightforward to add once the papers are to hand:
`riemann/fluxes.py` is arranged so that $M_p$ and $p_u$ slot into `m_half`
and `p_half` in `ausm_plus_flux`, and the test suite already contains the
moving-contact test that caught the bad reconstruction.

---

## 5. Measured behaviour

Sod's problem. Contact width is the number of cells with a density strictly
between the two plateau values, at 100 cells with the superbee limiter;
the $L_1$ error is at 200 cells; the cost is at 800 cells.

| Flux | Contact width | $L_1$ error in $\rho$ | µs / cell / step |
| --- | --- | --- | --- |
| Roe | 2 cells | $1.44 \times 10^{-3}$ | 0.51 |
| HLLC | 3 cells | $2.10 \times 10^{-3}$ | 0.33 |
| HLL | 8 cells | $4.04 \times 10^{-3}$ | 0.36 |
| AUSM+ | — (cannot run) | — | 0.27 |
| Rusanov | 19 cells | $1.28 \times 10^{-2}$ | 0.36 |
| Lax–Friedrichs | 24 cells | $2.22 \times 10^{-2}$ | 0.36 |

![Numerical fluxes at the contact discontinuity](figures/flux_comparison.png)

The ordering is not a coincidence or a tuning artefact: it follows directly
from how much of the wave structure each flux resolves. Every one of them
captures the *shock* in one or two cells — it is the contact that separates
them, exactly as the theory says it should.

`tests/test_fluxes.py::test_accuracy_ordering_on_sod` asserts the ordering, so
it cannot silently regress.

---

## 6. Robustness

Which fluxes run which problems, at second order with superbee:

| Problem | Roe | HLLC | HLL | AUSM+ | Rusanov | LF |
| --- | --- | --- | --- | --- | --- | --- |
| `sod` | ✓ | ✓ | ✓ | ✗ rest | ✓ | ✓ |
| `toro2` (near vacuum) | ✗ | ✓ | ✓ | ✗ rest | ✓ | ✓ |
| `toro3` | ✓ | ✓ | ✓ | ✗ rest | ✓ | ✓ |
| `toro4` | ✓ | ✓ | ✓ | ✗ rest | ✓ | ✓ |
| `toro5` | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `sonic_rarefaction` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `air_shock_tube` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

"✗ rest" is the AUSM+ start-from-rest limitation of §4, not a near-vacuum
failure.

Two entries deserve comment.

**Roe on `toro2`** is the near-vacuum breakdown described in
[roe-solver.md](roe-solver.md#5-known-limitation-roes-solver-is-not-positivity-preserving).
HLL and HLLC run it, which is the whole point of a positivity-preserving
flux, and resolves the long-standing roadmap item in this repository:

![Toro Test 2 with HLLC and HLL](figures/toro2_positivity.png)

Density and velocity are captured well. The overshoot in the specific
internal energy near $x = 0.5$ is the known difficulty of this test — the
star region is at $p_* \approx 0.0019$, so $e = p / ((\gamma - 1)\rho)$ is a
ratio of two very small numbers — and it appears for every scheme, not only
these two.

**Rusanov on `toro5`** is a useful distinction. Rusanov is
positivity-preserving *as a first-order flux*, and at first order it runs
`toro5` without trouble. The failure comes from the second-order TVD
correction, which the flux's positivity proof does not cover. "The flux is
positivity-preserving" and "the scheme is positivity-preserving" are
different statements, and only `hll`, `hllc` and `lax_friedrichs` are listed
in `riemann.fluxes.ROBUST_FLUXES`.

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

This is classical odd–even decoupling, not a bug, and
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
- **AUSM+** for flows that are everywhere moving; never for a problem started
  from rest.
- **Lax–Friedrichs** for teaching, as the baseline the others are measured
  against.

---

**Back to:** [Documentation index](index.md)
