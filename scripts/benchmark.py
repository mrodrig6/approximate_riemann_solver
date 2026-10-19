#!/usr/bin/env python3
"""Measure the Numba speed-up on the Roe solver.

Run twice to see both backends::

    python scripts/benchmark.py                      # compiled
    NUMBA_DISABLE_JIT=1 python scripts/benchmark.py  # pure Python

``NUMBA_DISABLE_JIT=1`` makes Numba's decorators pass the Python function
through untouched, which is the same code path taken when Numba is not
installed at all -- so this measures exactly what an unaccelerated user gets.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import get_problem, solve_roe  # noqa: E402
from src.core.jit import HAVE_NUMBA, numba_version  # noqa: E402

JIT_ON = HAVE_NUMBA and os.environ.get("NUMBA_DISABLE_JIT") != "1"


def main():
    prob = get_problem("sod")
    backend = f"numba {numba_version}" if JIT_ON else "pure Python"
    print(f"backend: {backend}\n")

    # First call pays the compilation cost; report it separately so the
    # per-run timings are not polluted by it.
    t0 = time.perf_counter()
    solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=64)
    warmup = time.perf_counter() - t0
    print(f"warm-up (includes JIT compilation): {warmup:7.3f} s\n")

    print(f"{'cells':>7} {'steps':>7} {'time (s)':>10} {'us/cell/step':>14}")
    for nx in (200, 400, 800, 1600):
        t0 = time.perf_counter()
        roe = solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=nx)
        elapsed = time.perf_counter() - t0
        per = 1e6 * elapsed / (nx * roe.steps)
        print(f"{nx:>7} {roe.steps:>7} {elapsed:>10.4f} {per:>14.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
