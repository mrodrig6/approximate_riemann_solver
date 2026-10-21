# Performance

Numba compiles the numerical kernels selected through `src.core.jit.njit`. Without Numba
the same source runs as Python; the shim turns the decorator into a no-op.

```bash
# Warm kernels and measure the compiled solver.
python scripts/benchmark.py
# Run the same benchmark without compilation.
NUMBA_DISABLE_JIT=1 python scripts/benchmark.py
```

`scripts/benchmark.py` prints a separate 64-cell warmup, then times Sod at 200, 400,
800, and 1600 cells. It reports wall time and microseconds per cell per step. Compare
backends on the same machine and record Python, NumPy, Numba, CPU, grid, step count,
warmup, and whether caches already existed.

The previous documentation reported about 0.52 µs per cell per step for Numba and 28.0
µs for Python at 1600 cells (about 54×), and elsewhere used rounded claims near 100×.
Those are historical measurements from an unspecified machine, not a promise for the
current host. Flux-specific historical timings also varied between paragraphs; use a
fresh benchmark rather than treating them as portable rankings.

`cache=True` retains compiled kernels under `__pycache__` across processes. Compilation
and cache loading still have costs; neither should be confused with steady-state
stepping. No accelerator service or GPU is required.

The fast-math set permits reassociation and contraction, but excludes `nnan` and `ninf`
so positivity guards retain their NaN/infinity semantics. `tests/core/test_jit.py`
checks those settings. Compiled and Python arithmetic can differ in the last bits;
compare with numerical tolerances.

The [CI guide](../reference/ci.md) explains actual Numba-absent testing versus
`NUMBA_DISABLE_JIT=1`, which leaves Numba installed.
