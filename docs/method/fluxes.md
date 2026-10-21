# Euler numerical fluxes

The TVD scheme selects an Euler base flux through `flux=`. This page records the
formulas and implementation choices;
[measured accuracy](../validation/flux-comparison.md) and
[operating restrictions](../reference/compatibility.md) have separate pages.

## The three families

| Flux                     | `flux=`                | Waves      | Module              | Source                |
| ------------------------ | ---------------------- | ---------- | ------------------- | --------------------- |
| Roe (+ Harten–Hyman fix) | `roe`                  | 3          | `fluxes/roe.py`     | Toro Ch. 11           |
| HLLC                     | `hllc`                 | 3          | `fluxes/hll.py`     | Toro Sec. 10.4        |
| HLL / HLLE               | `hll`, `hlle`          | 2          | `fluxes/hll.py`     | Toro Secs. 10.3, 10.5 |
| AUSM+-up                 | `ausmup`, `ausm`       | 3 (split)  | `fluxes/ausm.py`    | Liou 2006             |
| Rusanov (local LF)       | `rusanov`, `llf`       | 0 (scalar) | `fluxes/central.py` | Toro Sec. 10.5        |
| Lax–Friedrichs           | `lax_friedrichs`, `lf` | 0 (scalar) | `fluxes/central.py` | Toro Sec. 5.3.4       |

**Central / Rusanov-type.** A central average plus one scalar dissipation coefficient
applied to the whole state vector:

$$
\mathbf{F} = \tfrac{1}{2}\left(\mathbf{F}_L + \mathbf{F}_R\right) - \tfrac{1}{2} S\left(\mathbf{U}_R - \mathbf{U}_L\right).
$$

Lax–Friedrichs takes $S = \Delta x / \Delta t$, the *global* mesh speed; Rusanov takes
the *local* maximum signal speed $S = \max(|u_L| + a_L, |u_R| + a_R)$. Robust and
trivial, but the same large coefficient hits every field, so contacts smear badly.

**HLL-type.** Assume an approximate wave structure and average the exact integral form
across it. HLL uses two waves and one intermediate state; HLLC adds the contact back.

**Flux-vector / flux-difference.** Roe decomposes the jump into characteristic waves;
AUSM+-up splits the flux into a convective part carried by an interface velocity and a
separate pressure part. Both resolve the contact exactly.

______________________________________________________________________

## HLL

Assume the fan contains just a left wave of speed $S_L$ and a right wave of speed $S_R$,
with one constant state between them. Integrating the conservation law over a control
volume spanning the fan gives (Toro eq. 10.21)

$$
\mathbf{F}^{\text{HLL}} = \frac{S_R \mathbf{F}_L - S_L \mathbf{F}_R + S_L S_R\left(\mathbf{U}_R - \mathbf{U}_L\right)}{S_R - S_L}
$$

when $S_L \le 0 \le S_R$, and simply the upwind physical flux otherwise.

### Wave-speed estimates matter

The code uses **Einfeldt's** estimates (Toro Sec. 10.5), which is what makes this the
HLLE variant:

$$
S_L = \min\left(u_L - a_L, \tilde{u} - \tilde{a}\right), \qquad S_R = \max\left(u_R + a_R, \tilde{u} + \tilde{a}\right),
$$

with $\tilde{u}, \tilde{a}$ the Roe averages. The bounds combine physical and Roe
characteristic speeds. Positivity results require suitable data, speed bounds, and a CFL
restriction; the second-order correction has additional restrictions. The cheaper Davis
estimate, using only the physical speeds, can fail to bracket a strong shock.

**The cost of two waves.** With no middle wave in the assumed structure, the contact is
averaged away. The Sod comparison uses the same second-order correction for each base
flux: HLL spreads the contact over 8 cells and Roe over 2. See the
[measurement definition](../validation/flux-comparison.md).

______________________________________________________________________

## HLLC

HLLC restores the middle wave. Requiring pressure and normal velocity to be continuous
across it gives the contact speed (Toro eq. 10.37)

$$
S_{\ast} = \frac{p_R - p_L + \rho_L u_L\left(S_L - u_L\right) - \rho_R u_R\left(S_R - u_R\right)}{\rho_L\left(S_L - u_L\right) - \rho_R\left(S_R - u_R\right)},
$$

and the Rankine–Hugoniot conditions across the outer waves give the two star states
(Toro eq. 10.39), for $K \in \{L, R\}$:

$$
\mathbf{U}^{\ast}_{K} = \rho_K \frac{S_K - u_K}{S_K - S_{\ast}}\left[1, S_{\ast}, \frac{E_K}{\rho_K} + \left(S_{\ast} - u_K\right)\left(S_{\ast} + \frac{p_K}{\rho_K\left(S_K - u_K\right)}\right)\right]^{T},
$$

with
$\mathbf{F}^{\ast}_{K} = \mathbf{F}_K + S_K\left(\mathbf{U}^{\ast}_{K} - \mathbf{U}_K\right)$.
The flux is then selected by the sign of $S_L$, $S_{\ast}$ and $S_R$.

HLLC resolves the contact and is a useful general-purpose choice. The installed scheme
runs the built-in catalogue under the conditions in the
[compatibility matrix](../reference/compatibility.md); this is not a positivity
guarantee for arbitrary higher-order reconstructions. Measure costs with the
[benchmark](../running/performance.md).

______________________________________________________________________

## AUSM+-up

AUSM-family schemes do not decompose the jump at all. They split the flux into a
convective part carried by an interface *velocity* and a separate pressure part:

$$
\mathbf{F} = u_{1/2} \mathbf{\Psi}_{\text{upwind}} + \left[0, p_{1/2}, 0\right]^{T}, \qquad \mathbf{\Psi} = \left[\rho, \rho u, E + p\right]^{T},
$$

with $\mathbf{\Psi}$ taken from the left when $u_{1/2} > 0$ and from the right
otherwise. The interface velocity and pressure are built from polynomial splittings of
each side's Mach number $M_K = u_K / a_{1/2}$:

$$
u_{1/2} = a_{1/2}\left[M^{+}_{(4)}(M_L) + M^{-}_{(4)}(M_R)\right] + K_p \max\left(1 - \bar{M}^{2}, 0\right)\frac{p_L - p_R}{\rho_{1/2} a_{1/2}},
$$

$$
p_{1/2} = P^{+}_{(5)}(M_L) p_L + P^{-}_{(5)}(M_R) p_R + K_u P^{+}_{(5)}P^{-}_{(5)} \rho_{1/2} a_{1/2}\left(u_L - u_R\right),
$$

where $a_{1/2} = \tfrac{1}{2}(a_L + a_R)$, $\rho_{1/2} = \tfrac{1}{2}(\rho_L + \rho_R)$
and $\bar{M}^{2} = \tfrac{1}{2}(M_L^{2} + M_R^{2})$.

The two extra terms are the "-up" of the name. $K_p$ is a pressure-diffusion term in the
velocity, $K_u$ a velocity-diffusion term in the pressure, and the factor
$\max(1 - \bar{M}^2, 0)$ switches the first off once the interface goes supersonic,
where upwinding already supplies enough dissipation.

### Why the $K_p$ term is not optional

Without it — that is, for plain AUSM+ (Liou 1996) — the mass flux is identically zero
whenever $u_L = u_R = 0$, since $M^{+}_{(4)}(0) + M^{-}_{(4)}(0) = 0$ exactly, however
large the pressure jump is. The momentum equation still feels the pressure difference
and accelerates the gas, but with the energy flux also zero the growing kinetic energy
is subtracted from a fixed total energy, and on the light side of a shock tube the
internal energy goes negative in the first step.

Without pressure–velocity coupling, this particular rest-state update can produce a
nonphysical state on Sod. This observation concerns this implementation, not every AUSM+
discretisation. The $K_p$ term supplies the missing pressure–velocity coupling, and
AUSM+-up runs it without difficulty. `test_ausm_up_starts_from_rest` is the guard.

### The split polynomials

$M^{\pm}_{(4)}$ and $P^{\pm}_{(5)}$ are the standard AUSM+ ones, with $\beta = 1/8$ and
$\alpha = 3/16$. These are the values chosen by this implementation. The consistency
identities below hold independently of those choices, so they do not by themselves
identify unique coefficients:

$$
M^{+}_{(4)}(M) + M^{-}_{(4)}(M) = M, \qquad P^{+}_{(5)}(M) + P^{-}_{(5)}(M) = 1,
$$

which hold for *every* $M$ and independently of $\alpha$ and $\beta$. Both are checked
directly in `tests/euler/test_fluxes.py`. Expanded, they give the closed forms, for
$|M| < 1$:

$$
M^{+}_{(4)} = \tfrac{1}{8}M^{4} + \tfrac{1}{2}M + \tfrac{3}{8},
$$

$$
P^{+}_{(5)} = \tfrac{3}{16}M^{5} - \tfrac{5}{8}M^{3} + \tfrac{15}{16}M + \tfrac{1}{2}.
$$

### Where the coefficients come from

$K_p = 0.4$ and $K_u = 0.2$, and the arithmetic-mean $a_{1/2}$, follow the benchmarked
implementation in **EMS** (the Eulerian Multicomponent Solver of the same research
group), `src/riemann_solvers/ausm6.cpp`.

They matter, and they are worth stating separately from $\alpha$ and $\beta$, because
the two groups have different standing. $\alpha$ and $\beta$ are fixed in this
implementation; the identities above check consistency but do not determine those
coefficients. $K_p$ and $K_u$ are *diffusion coefficients*, and those identities say
nothing about them — both hold for every value. A $K_u$ that is too large is
anti-dissipative rather than merely inaccurate: it destabilises a moving contact,
amplifying round-off until the pressure error swamps the solution.

What constrains them is a physical invariant rather than an algebraic one: a contact
advected at uniform speed must not generate pressure.
`test_moving_contact_keeps_pressure_constant` checks exactly that, and is the guard
against the coefficients drifting.

## Lax–Friedrichs decouples the grid

Classical Lax–Friedrichs has one further property worth knowing, because it shows up
immediately if you try to use it: **the second-order TVD correction has no effect on it
at all.**

Its dissipation $\tfrac{1}{2}(\Delta x/\Delta t)(\mathbf{U}_R - \mathbf{U}_L)$ acts on a
$2\Delta x$ stencil, which leaves the even and odd sub-grids uncoupled. The solution
develops identical adjacent cell pairs, so at every interface either the local wave
strength or the upwind one is exactly zero. Either way the limiter ratio $r_k$ is driven
to zero, $\phi = 0$, and the correction vanishes identically — measured over a whole Sod
run, the correction weight was nonzero at **0 of 603** interface-wave pairs, and the
second-order and first-order solutions agreed to $8 \times 10^{-14}$.

This is classical odd–even decoupling, and
`test_lax_friedrichs_odd_even_decoupling_disables_the_limiter` pins it. The
local-dissipation flux, Rusanov, does not suffer from it, which is one more reason to
prefer it over Lax–Friedrichs whenever a cheap robust flux is wanted.
