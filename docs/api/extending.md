# Extending the package

Custom initial fields work through the public API. New compiled fluxes or limiters
require source integration; there is no runtime plugin registration API.

## Worked example: a custom initial field

This smooth density pulse uses the existing public API; it does not require editing
solver code. The exact solution is a translation while velocity and pressure stay
uniform.

```python
import numpy as np
from src import cell_centres, solve_field

x = cell_centres(200)
rho = 1.0 + 0.2 * np.exp(-200.0 * (x - 0.3) ** 2)
initial = np.column_stack((rho, np.ones_like(x), np.ones_like(x)))
result = solve_field(1.4, initial, 0.1, flux="hllc", limiter="vanleer")
assert result.rho.shape == (200,)
assert np.all(result.p > 0)
```

## Source integration

The numerical dispatch uses integer codes and compiled branches. Adding a flux is a
source change, not registering a Python callable at runtime. Match the system's
conservative-state signature, register a new code, add dispatch, and test consistency
`F(U, U) = F(U)` and the applicable invariants before comparing full simulations.

## Where new code goes

The following integration points keep changes local to the relevant system; a new flux
requires its module, registry, and dispatch updates.

- **Another flux** (SLAU, for instance): a new module in that system's `fluxes/`, a code
  in its `registry.py`, a branch in its `interface.py`. Nothing outside that directory
  changes — the solver dispatches on the code.
- **Another limiter**: `core/limiters.py`, and both systems get it at once.
- **Another scheme** beside TVD and ADER: a module beside `euler/ader.py` and a branch
  in `solver.py`, which already selects on `scheme=`.
- **Another system of equations**: a new subpackage alongside the two, with the same
  module names as §2 — `physics`, `exact`, `fluxes/`, `solver`, `problems`. It imports
  `core/` and nothing else of the package.
- **Higher-order ADER**: `euler/ader.py` alone; the signature it presents to `solver.py`
  does not change with the order.
