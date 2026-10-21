"""Run every example in a scratch directory and verify its saved images."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    outputs = {
        "sod_shock_tube": ["sod_example.png"],
        "compare_fluxes": ["fluxes_example.png"],
        "compare_limiters": ["limiters_example.png"],
        "dimensional_shock_tube": ["air_shock_tube_example.png"],
        "shallow_water_dam_break": ["dam_break.png", "dam_break_dry.png"],
    }
    with tempfile.TemporaryDirectory(prefix="riemann-examples-") as directory:
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = str(Path(directory) / "matplotlib")
        env["XDG_CACHE_HOME"] = str(Path(directory) / "cache")
        for script in sorted((root / "examples").glob("*.py")):
            result = subprocess.run([sys.executable, str(script)], cwd=directory, env=env)
            if result.returncode:
                return result.returncode
            for name in outputs[script.stem]:
                path = Path(directory) / name
                if not path.is_file() or not path.stat().st_size:
                    raise RuntimeError(f"{script.name} did not write {name}")
            print(f"PASS {script.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
