"""Print reproducible numerical measurements used by the validation pages.

Run from the root: python scripts/record_validation.py > /tmp/riemann-results.json
No tracked files are changed. Timing is deliberately excluded.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (  # noqa: E402
    acoustic_pulse, advected_pulse, cell_centres, get_problem, solve,
    solve_exact, solve_field,
)
from src import shallow_water as sw  # noqa: E402


def euler_run(name, nx=200, **kwargs):
    p = get_problem(name)
    return solve(p.gamma, p.left, p.right, p.t_end, nx=nx,
                 x_diaphragm=p.x_diaphragm, **kwargs)


def errors(name, numerical):
    p = get_problem(name)
    reference = solve_exact(p.gamma, p.left, p.right, p.t_end,
                            n=len(numerical.x), x_diaphragm=p.x_diaphragm)
    return {field: float(np.mean(np.abs(getattr(numerical, field) - getattr(reference, field)))
                        / np.max(np.abs(getattr(reference, field))))
            for field in ("rho", "u", "p")}


def main():
    result = {"toro": {}, "euler_profiles": {}, "fluxes": {}, "shallow_water": {},
              "convergence": {}, "entropy": {}, "ader": {}}
    for name in ("sod", "toro2", "toro3", "toro4", "toro5"):
        p = get_problem(name)
        ex = solve_exact(p.gamma, p.left, p.right, p.t_end)
        result["toro"][name] = dict(p_star=ex.p_star, u_star=ex.u_star)
    for name in ("sod", "toro3", "toro4", "toro5", "sonic_rarefaction", "air_shock_tube"):
        sol = euler_run(name)
        result["euler_profiles"][name] = dict(steps=sol.steps, **errors(name, sol))
    for flux in ("roe", "hllc", "hll", "ausmup", "rusanov", "lax_friedrichs"):
        sol = euler_run("sod", nx=100, flux=flux)
        inside = (sol.rho > .2656 + .05 * (.4263 - .2656)) & (sol.rho < .4263 - .05 * (.4263 - .2656))
        width = int(np.sum(inside[(sol.x > .6) & (sol.x < .85)]))
        result["fluxes"][flux] = dict(contact_width=width, **errors("sod", euler_run("sod", flux=flux)))
    for name, p in sw.PROBLEMS.items():
        result["shallow_water"][name] = {}
        ex = sw.solve_exact(p.left, p.right, p.t_end, n=200,
                            gravity=p.gravity, x_diaphragm=p.x_diaphragm)
        for flux in ("roe", "hll", "rusanov", "lax_friedrichs"):
            sol = sw.solve(p.left, p.right, p.t_end, nx=200,
                           gravity=p.gravity, x_diaphragm=p.x_diaphragm, cfl=p.cfl, flux=flux)
            result["shallow_water"][name][flux] = float(np.mean(np.abs(sol.h-ex.h))/np.max(ex.h))
    resolutions = np.array([50, 100, 200, 400])
    ref = solve_field(1.4, acoustic_pulse(cell_centres(3200)), .15,
                      scheme="ader2", limiter="none", cfl=.8)
    for key, kwargs in (("first", {"second_order": False}),
                        ("tvd", {}), ("ader2", {"scheme": "ader2"})):
        values = {"advected": [], "acoustic": []}
        for nx in resolutions:
            x = cell_centres(int(nx))
            sol = solve_field(1.4, advected_pulse(x), .2, limiter="none", cfl=.8, **kwargs)
            values["advected"].append(float(np.mean(np.abs(sol.rho-advected_pulse(x-.2)[:, 0]))))
            sol = solve_field(1.4, acoustic_pulse(x), .15, limiter="none", cfl=.8, **kwargs)
            values["acoustic"].append(float(np.mean(np.abs(sol.rho-np.interp(x, ref.x, ref.rho)))))
        result["convergence"][key] = {
            kind: {"errors": values[kind], "rate": float(-np.polyfit(np.log(resolutions), np.log(values[kind]), 1)[0])}
            for kind in values
        }
    p = get_problem("sonic_rarefaction")
    for nx in (100, 400):
        ex = solve_exact(p.gamma, p.left, p.right, p.t_end, n=nx, x_diaphragm=p.x_diaphragm)
        result["entropy"][nx] = {}
        for fix in (True, False):
            sol = euler_run(p.name, nx=nx, entropy_fix=fix, second_order=False)
            inside = (sol.x > 0) & (sol.x < .45)
            result["entropy"][nx][str(fix)] = float(np.max(np.abs(sol.rho-ex.rho)[inside]))
    x = cell_centres(200)
    tvd = solve_field(1.4, acoustic_pulse(x), .15, limiter="none", cfl=.8)
    ader = solve_field(1.4, acoustic_pulse(x), .15, limiter="none", cfl=.8, scheme="ader2")
    result["ader"]["acoustic_max_difference"] = float(np.max(np.abs(tvd.rho-ader.rho)))
    result["ader"]["sod_relative_density_error"] = errors("sod", euler_run("sod", scheme="ader2"))["rho"]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
