# References and citation

The implementation follows the references below. Equation numbers on the theory and
method pages identify the formulas implemented in source.

| Reference                                                                                                                                              | Used for                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., Springer, 2009, [DOI: 10.1007/b79761](https://doi.org/10.1007/b79761) | Euler equations, exact solver, Roe, fluxes, TVD and ADER        |
| E. F. Toro, *Shock-Capturing Methods for Free-Surface Shallow Flows*, Wiley, 2001                                                                      | Shallow-water equations and wet/dry exact solutions             |
| Einfeldt, Munz, Roe and Sjögreen, J. Comput. Phys. **92**, 273 (1991)                                                                                  | Godunov methods near low density and Roe positivity limitations |
| M.-S. Liou, J. Comput. Phys. **129**, 364 (1996)                                                                                                       | AUSM+ split polynomials                                         |
| M.-S. Liou, *A sequel to AUSM, Part II: AUSM+-up for all speeds*, J. Comput. Phys. **214**, 137 (2006)                                                 | AUSM+-up pressure–velocity coupling                             |
| Shima and Kitamura, AIAA J. **49**, 1693 (2011)                                                                                                        | SLAU, a possible future extension                               |

The AUSM diffusion coefficients follow the research group's EMS implementation,
`src/riemann_solvers/ausm6.cpp`; this file is not part of this repository. It is
provenance, not a runnable local dependency.

Managed by the Rodriguez Flow Research group at Brown University. Developer: Mauro
Rodriguez Jr. (`mauro_rodriguez@brown.edu`). The software is intended for education and
numerical experiments.

When using these methods, cite Toro:

```bibtex
@book{toro2009,
  author = {Toro, Eleuterio F.},
  title = {Riemann Solvers and Numerical Methods for Fluid Dynamics:
           A Practical Introduction},
  edition = {3},
  publisher = {Springer},
  year = {2009},
  doi = {10.1007/b79761}
}
```
