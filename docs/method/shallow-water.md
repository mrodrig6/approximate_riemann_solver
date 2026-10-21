# Shallow-water numerical method

`src.shallow_water.solver` combines four base fluxes with the shared TVD limiters. Its
interface code adds wet/dry guards that the Euler driver does not use.

## The fluxes

| Flux           | `flux=`                | Notes                                                                        |
| -------------- | ---------------------- | ---------------------------------------------------------------------------- |
| HLL            | `hll`, `hlle`, `hllc`  | Default; two-wave approximate flux. `hllc` is an alias, not an extra solver. |
| Roe            | `roe`                  | With a Harten entropy fix; falls back to HLL at a dry interface.             |
| Rusanov        | `rusanov`, `llf`       | Local Lax–Friedrichs.                                                        |
| Lax–Friedrichs | `lax_friedrichs`, `lf` | The dissipative baseline.                                                    |

The Roe average for this system is

$$
\tilde{u} = \frac{\sqrt{h_{L}} u_{L} + \sqrt{h_{R}} u_{R}}{\sqrt{h_{L}} + \sqrt{h_{R}}}, \qquad \tilde{a} = \sqrt{\tfrac{1}{2}g (h_{L} + h_{R})}.
$$

As for a gas, these are the unique averages making the U-property
$\mathbf{F}_R - \mathbf{F}_L = \tilde{A}(\mathbf{U}_R - \mathbf{U}_L)$ hold exactly, and
the velocity is again weighted by the square root of the first conserved variable. The
celerity average is the simpler of the two because $\mathbf{F}$ is quadratic in
$\mathbf{U}$ here, where for a gas it is not.
`tests/shallow_water/test_fluxes.py::test_roe_u_property` checks the identity
numerically.

The code falls back to HLL when either side is dry. The Roe average is undefined if both
depths vanish; a one-sided dry interface also needs special wet/dry wave handling, and
the wave-speed estimates switch from the Einfeldt form to the wet/dry front speed
$u \pm 2a$ that the exact solver uses. Getting that right is what lets the approximate
solvers run a dry dam break at all.

______________________________________________________________________

## Dry-cell update

`src.shallow_water.solver._advance` sets depth and discharge to zero whenever the
updated depth is at or below `DRY_TOLERANCE`, including negative values. This clipping
is not a general proof of positivity or exact conservation through wetting fronts.
Catalogue dry cases use `cfl=0.45`; see [workflows](../running/shallow-water.md).
