# Problem catalogues

The catalogue objects store reproducible initial data and geometry. Euler problems are
selectable from the CLI; shallow-water problems use Python.

## Euler

| Key                 | Left state                                     | Right state                   | Output time | Diaphragm | Physics   | CFL | Source                                                    |
| ------------------- | ---------------------------------------------- | ----------------------------- | ----------- | --------- | --------- | --- | --------------------------------------------------------- |
| `sod`               | `(1, 0, 1)`                                    | `(0.125, 0, 0.1)`             | 0.25        | 0.5       | gamma=1.4 | 0.9 | Toro Test 1, Sec. 4.3.3 / Table 4.1                       |
| `toro2`             | `(1, -2, 0.4)`                                 | `(1, 2, 0.4)`                 | 0.15        | 0.5       | gamma=1.4 | 0.9 | Toro Test 2, Sec. 4.3.3                                   |
| `toro3`             | `(1, 0, 1000)`                                 | `(1, 0, 0.01)`                | 0.012       | 0.5       | gamma=1.4 | 0.9 | Toro Test 3, Sec. 4.3.3                                   |
| `toro4`             | `(5.99924, 19.5975, 460.894)`                  | `(5.99242, -6.19633, 46.095)` | 0.035       | 0.5       | gamma=1.4 | 0.9 | Toro Test 4, Sec. 4.3.3                                   |
| `toro5`             | `(1, -19.59745, 1000)`                         | `(1, -19.59745, 0.01)`        | 0.012       | 0.5       | gamma=1.4 | 0.9 | Toro Test 5, Sec. 4.3.3                                   |
| `sonic_rarefaction` | `(1, 0.75, 1)`                                 | `(0.125, 0, 0.1)`             | 0.2         | 0.3       | gamma=1.4 | 0.9 | Toro Sec. 11.3 (entropy fix); cf. Sec. 6.4 Test 1 variant |
| `air_shock_tube`    | `(2.17994129244, 225.894320023, 238557.89125)` | `(1.204, 0, 101325)`          | 0.0003      | 0.5       | gamma=1.4 | 0.9 | dimensional shock tube in air at standard conditions      |

## Shallow water

| Key                | Left state  | Right state  | Output time | Diaphragm | Physics | CFL  | Source                                        |
| ------------------ | ----------- | ------------ | ----------- | --------- | ------- | ---- | --------------------------------------------- |
| `dam_break`        | `(1, 0)`    | `(0.1, 0)`   | 0.05        | 0.5       | g=9.81  | 0.9  | Stoker's dam-break solution; Toro SW Sec. 5.3 |
| `dam_break_dry`    | `(1, 0)`    | `(0, 0)`     | 0.05        | 0.5       | g=9.81  | 0.45 | Ritter's dry-bed solution; Toro SW Sec. 5.5   |
| `two_bores`        | `(1, 2.5)`  | `(1, -2.5)`  | 0.05        | 0.5       | g=9.81  | 0.9  | Toro SW Ch. 6 test set                        |
| `two_rarefactions` | `(1, -2.5)` | `(1, 2.5)`   | 0.05        | 0.5       | g=9.81  | 0.9  | Toro SW Ch. 6 test set                        |
| `dry_generation`   | `(1, -7)`   | `(1, 7)`     | 0.03        | 0.5       | g=9.81  | 0.45 | dry-bed generation; Toro SW Sec. 5.5          |
| `subcritical_step` | `(1, 0.5)`  | `(0.8, 0.5)` | 0.05        | 0.5       | g=9.81  | 0.9  | smooth-ish case for accuracy comparison       |

All current domains span `[0, 1]`. Euler states are `[rho, u, p]`; shallow-water states
are `[h, u]`. Catalogue references identify the construction; a source label does not
claim an independent published numerical value for every row. See
[compatibility](../reference/compatibility.md) for combinations that fail.
