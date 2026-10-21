# Exact-solver algorithms

The exact solvers find an intermediate pressure or depth, then sample the wave fan. This
page describes numerical safeguards rather than repeating the
[Euler](../theory/exact-euler.md) and [shallow-water](../theory/exact-shallow-water.md)
derivations.

| Operation                | Euler                                                 | Shallow water                                          |
| ------------------------ | ----------------------------------------------------- | ------------------------------------------------------ |
| Entry point              | `src.euler.exact.solve_exact`                         | `src.shallow_water.exact.solve_exact`                  |
| Scalar solve             | `solve_star_pressure` returns pressure and iterations | `solve_star_depth` returns depth, velocity, iterations |
| Initial guess            | Toro adaptive PVRS / two-rarefaction / two-shock      | Two-rarefaction depth                                  |
| Relative tolerance       | `tol=1e-12`                                           | `tol=1e-12`                                            |
| Scalar iteration cap     | `max_iter=100`                                        | `max_iter=100`                                         |
| Nonpositive next iterate | Halve previous iterate                                | Halve previous iterate                                 |
| Vacuum or dry data       | Reject vacuum-generating Euler data                   | Use explicit wet/dry sampling branches                 |
| Sampling                 | Cell centres, `n=1000` by default                     | Cell centres, `n=1000` by default                      |

Both solvers require positive output time and validate their physics parameter. Euler
checks positive initial density and pressure. Shallow water accepts a dry side but the
exact solver rejects two dry sides; the numerical driver can return an all-dry resting
field without stepping.

The convergence criterion uses relative change, so a tolerance is not tied to pressure
units. A poor guess can still exhaust the cap. See
[actual errors and remedies](../running/troubleshooting.md).
