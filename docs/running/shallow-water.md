# Shallow-water workflows

Use `src.shallow_water` from Python for dam breaks and general depth/velocity fields.
The Euler CLI does not accept shallow-water catalogue keys.

```python
from src.shallow_water import get_problem, solve, solve_exact

problem = get_problem("dam_break_dry")
exact = solve_exact(problem.left, problem.right, problem.t_end,
                    **{k: v for k, v in problem.kwargs().items()
                       if k not in ("w_l", "w_r")})
numerical = solve(problem.left, problem.right, problem.t_end,
                  gravity=problem.gravity, x_left=problem.x_left,
                  x_right=problem.x_right,
                  x_diaphragm=problem.x_diaphragm,
                  cfl=problem.cfl, flux="hll")
assert exact.dry
assert numerical.h.min() >= 0
```

`ShallowWaterProblem.kwargs()` provides states and geometry, but omits output time and
CFL. Pass `problem.cfl` explicitly: dry-bed examples use `0.45` because the wetting
front can outrun the usual cell signal-speed estimate. The API default remains `0.9` and
does not infer a catalogue entry from the states.

| Workflow            | Reference                                                                |
| ------------------- | ------------------------------------------------------------------------ |
| Wet dam break       | Exact Stoker construction and numerical bore                             |
| Dry dam break       | Ritter analytic solution                                                 |
| Dry generation      | Two separating wet streams create a dry interval                         |
| Custom smooth field | `src.shallow_water.solve_field(w_init, t_end, ...)` with shape `(nx, 2)` |
| Plot                | `src.plot_shallow_water` and `src.save_figure`                           |

HLL is the default base flux. `hllc` aliases HLL here; AUSM and ADER are not
implemented. See the [catalogue](../model/problems.md),
[wet/dry method](../method/shallow-water.md), and
[validation](../validation/shallow-water.md).
