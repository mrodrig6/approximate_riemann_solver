# Limits and future work

This page records unimplemented capabilities and separates them from documented
numerical failures. The [compatibility matrix](compatibility.md) is the canonical list
of operating combinations.

- One spatial dimension and a uniform grid only; transmissive boundaries only.
- Ideal gas with constant `gamma`; no stiffened-gas or Mie–Grüneisen closure.
- Flat frictionless shallow-water bed; no topographic source term or general
  well-balanced treatment.
- ADER2 only, for Euler. Higher-order ADER and shallow-water ADER remain future work.
- SLAU (Shima & Kitamura, 2011) is not implemented. A new module can be added beside
  `ausm.py` once its defining diffusion functions are verified.
- WENO and MUSCL–Hancock reconstruction are not implemented. WENO is mentioned as a
  potential AUSM companion, not an available option.
- Roe can fail near vacuum; higher-order corrections introduce additional restrictions.
  See [troubleshooting](../running/troubleshooting.md).
- The package metadata version is `1.0.0` while `src.__version__` and the CLI report
  `1.1.0`. This pre-existing release-metadata issue remains open; the documentation
  change does not select a release version.

Potential extensions include SLAU, higher-order ADER, shallow-water source terms,
multidimensional splitting, other equations of state, and MUSCL–Hancock. See the
[extension guide](../api/extending.md).
