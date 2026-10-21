# First shock tube

Run Sod's shock tube and compare the exact Euler solution with the second-order Roe
solution. Start in the repository root after [installation](installation.md).

```bash
python -m src sod --plot sod.png
```

Recorded output from Python 3.12.14 with Numba 0.68.0:

```text
problem        : sod -- Sod's problem: left rarefaction, contact, right shock
reference      : Toro Test 1, Sec. 4.3.3 / Table 4.1
gamma, t_end   : 1.4, 0.25
left  (rho,u,p): [1. 0. 1.]
right (rho,u,p): [0.125 0.    0.1  ]
backend        : numba 0.68.0

exact solution : p* = 0.30313018, u* = 0.92745262 (4 Newton iterations)
wave pattern   : left rarefaction, contact, right shock

numerical run  : Roe, 2nd order, superbee, nx = 200, CFL = 0.9
                 122 steps to t = 0.25

error vs exact (on the numerical grid):
  rho  L1 = 1.4416e-03   relative L1 = 1.4416e-03
  u    L1 = 2.3999e-03   relative L1 = 2.5877e-03
  p    L1 = 9.3714e-04   relative L1 = 9.3714e-04
  e    L1 = 6.8311e-03   relative L1 = 2.3939e-03

figure written to sod.png
```

| Output line            | Meaning                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| `problem`, `reference` | Catalogue key, wave description, and textbook source                |
| `gamma, t_end`         | Ratio of specific heats and requested output time                   |
| `left`, `right`        | Initial primitive states: density, velocity, pressure               |
| `backend`              | Installed acceleration backend; exact version varies by environment |
| `exact solution`       | Intermediate pressure, velocity, and Newton iteration count         |
| `wave pattern`         | Wave family on each side of the contact                             |
| `numerical run`        | Flux, reconstruction order, limiter, cell count, CFL                |
| `steps`                | CFL-limited updates; last update lands at the requested time        |
| `error vs exact`       | Mean absolute error and normalized error on numerical cell centres  |
| `figure written`       | Saved comparison image                                              |

The image shows density, velocity, pressure, and specific internal energy. Read the
[output guide](../running/output.md) for field definitions and error normalization. This
run's numbers agree with the [Sod benchmark](../validation/sod.md); timings and backend
versions are environment dependent.

Try another base flux with `python -m src sod --flux hllc`. See [recipes](examples.md)
for custom states, dry beds, and smooth-data runs.
