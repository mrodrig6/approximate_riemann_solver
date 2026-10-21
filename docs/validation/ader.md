# ADER comparisons

These checks distinguish ADER2 from the TVD scheme and record its current implementation
limits. `tests/euler/test_ader.py` checks smooth convergence, contact agreement, frame
sensitivity, and expected failures.

## Is it actually a different scheme?

Worth checking, because at second order ADER2 and the TVD scheme are close enough that a
bug could hide as "about the same".

On the **smooth acoustic pulse**, which excites the genuinely non-linear fields, they
differ by $5.1 \times 10^{-4}$ at 200 cells, with ADER about 15% more accurate against
the 3200-cell reference used in [convergence](convergence.md).

On a **pure contact** they agree to machine precision. That is a consistency check
rather than a coincidence: with only the linearly degenerate field active, both reduce
to limited Lax–Wendroff on that one field, so they *must* agree. The test suite asserts
both facts, since together they distinguish "two schemes that happen to be close" from
"one scheme with two names".

On Sod's problem at 200 cells, ADER2 gives an $L_1$ density error of
$2.1 \times 10^{-3}$ against the TVD scheme's $1.4 \times 10^{-3}$ — the same band, as
expected. At second order ADER's advantage is the one-step structure and the route to
higher order, not the constant.

______________________________________________________________________

## Limitations

- **Second order only.** Going higher needs more Taylor terms and a higher-degree
  reconstruction; the structure below is arranged for it but it is not implemented.

- **The linear reconstruction can go non-physical** on a very strong jump before the
  limiter catches it. A boundary-extrapolated state with zero density or negative
  pressure cannot be fed to the wave model at all — it divides by the density to form
  the velocity and takes the square root of the pressure to form the sound speed — so
  the implementation checks the extrapolated states and falls back to the cell averages
  at such an interface.

- **Which problems it runs is narrower than the TVD scheme's.** It runs `sod`, `toro3`,
  `toro4`, `sonic_rarefaction` and `air_shock_tube` with every limiter, and fails on
  `toro2` and `toro5` with every limiter. The limiter does not decide the outcome either
  way, so no limiter choice rescues those two; use `scheme="tvd"` with `flux="hllc"` for
  them. A failure is reported as a `FloatingPointError` naming the step and the time.

  The two failures have **different causes**, and only the first is a property of the
  scheme:

  - `toro2`, the near-vacuum "123 problem", is the expected one. The predictor adds a
    correction directly to the conserved variables with no positivity constraint, so
    nothing stops an update leaving the physical state space when the star-region
    pressure is $1.9 \times 10^{-3}$. Surviving it needs a positivity-preserving limiter
    or an a-posteriori first-order fallback, neither of which is implemented here.
  - `toro5` is a **limitation of this implementation**, not of ADER. It is `toro3` with
    a uniform $-19.59745$ added to both velocities, and the Euler equations are Galilean
    invariant, so the two differ only by the frame. Adding a uniform velocity to `toro3`
    and varying it alone:

| uniform velocity added to `toro3` | ADER2 | TVD  |
| --------------------------------- | ----- | ---- |
| $0$, $-1$                         | runs  | runs |
| $-5$ to $-19.6$                   | fails | runs |
| $+19.6$                           | runs  | runs |

ADER fails at a shift of $-5$ but survives $+19.6$ of the same magnitude, on physics
that is identical in every frame. The likely cause is visible in the source: the TVD
correction limits on the **characteristic** wave strengths
(`alpha_all[j_up, k] / alpha[k]`, field by field), while the ADER reconstruction in
`_limited_slopes` limits **componentwise on the conserved variables**, which is not
Galilean invariant. Limiting the reconstruction on primitive or characteristic variables
instead is the standard remedy and is not implemented yet.

- **No separate Roe entropy fix is applied.** The leading state comes from HLLC. The
  suite covers the transonic catalogue case; this is not a guarantee for every
  reconstructed state.
