# Test suite

The suite checks algebraic identities, conservation, analytic and textbook references,
and documented numerical failures. Expected exceptions are asserted by passing tests;
they are not skipped or expected-failure markers.

## Test suite

```bash
pytest
```

263 tests covering, among others:

- star-region values against Toro's Table 4.1 for all five of his test problems;
- the pressure-function derivative against a finite difference, which pins the
  easily-mistaken rarefaction exponent;
- Newton's independence from the initial guess, across 10 decades;
- Roe's **U-property**,
  $\mathbf{F}_R - \mathbf{F}_L = \sum_k \tilde{\alpha}_k \tilde{\lambda}_k \tilde{\mathbf{K}}_k$,
  to $10^{-10}$;
- reconstruction of the jump,
  $\Delta\mathbf{U} = \sum_k \tilde{\alpha}_k \tilde{\mathbf{K}}_k$;
- each TVD limiter (excluding `none`) inside Sweby's TVD region, and $\phi(1) = 1$;
- exact conservation of mass and energy, and momentum changing at exactly the rate set
  by the boundary pressures;
- a uniform state being a fixed point of the scheme;
- the solution landing exactly on the requested output time;
- the entropy glitch failing to converge when the fix is disabled;
- Toro Test 2 failing for Roe with the documented message, and succeeding for HLLC and
  HLL;
- consistency, `F(U, U) = F(U)`, for every flux over a spread of states;
- the two exact AUSM+ split-function identities, and the moving-contact invariant that
  caught a mis-scaled AUSM+-up coefficient;
- stationary-contact density preservation and moving-contact pressure uniformity for
  Roe, HLLC and AUSM+-up; central/HLL fluxes smear a stationary density jump;
- the measured accuracy ordering Roe < HLLC < HLL < Rusanov < Lax-Friedrichs;
- Lax-Friedrichs odd-even decoupling nullifying the limiter;
- second-order convergence of the TVD and ADER schemes on smooth data, and their
  agreement to machine precision on a pure contact;
- for the shallow-water system: the Roe U-property, Ritter's analytic dry-dam-break
  solution, dry-bed generation, still water as an exact fixed point, and every flux on
  every dam-break problem.

Run `NUMBA_DISABLE_JIT=1 python -m pytest -q` to exercise uncompiled kernels with Numba
installed. A clean installation with `[plot,test]` exercises the actual Numba-absent
fallback. The Numba fast-math flag test is skipped only when Numba is absent; see
[CI](../reference/ci.md).
