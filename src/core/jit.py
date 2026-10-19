"""Optional-Numba shim.

Every compute kernel in this package is decorated with :func:`njit` from this
module rather than importing ``numba`` directly.  If Numba is installed the
decorator is the real ``numba.njit`` and the kernels are compiled to machine
code on first call; if it is missing the decorator degrades to a no-op and the
identical NumPy/Python source still runs, only slower.

That matters because the Roe solver is an explicitly swept finite-volume code:
the hot loop touches one interface at a time and cannot be expressed as a
single vectorised NumPy expression without materialising large temporaries.
Numba recovers compiled-language speed for that loop (typically 50-150x over
pure Python) while keeping the source readable, which is the whole point of a
teaching code.
"""

from __future__ import annotations

__all__ = ["njit", "HAVE_NUMBA", "numba_version"]

try:  # pragma: no cover - exercised implicitly by whichever branch is taken
    from numba import njit as _numba_njit
    from numba import __version__ as numba_version

    HAVE_NUMBA = True

    def njit(*args, **kwargs):
        """``numba.njit`` with defaults suited to this code.

        ``cache=True`` persists the compiled kernels to ``__pycache__`` so the
        ~1 s compilation cost is paid once per machine rather than once per
        process, and ``fastmath=True`` lets LLVM reassociate the floating point
        arithmetic in the flux kernels.  Neither changes the discretisation.
        """
        kwargs.setdefault("cache", True)
        kwargs.setdefault("fastmath", True)
        # Support both @njit and @njit(...) spellings.
        if args and callable(args[0]) and not kwargs:
            return _numba_njit(cache=True, fastmath=True)(args[0])
        return _numba_njit(*args, **kwargs)

except ImportError:  # pragma: no cover
    HAVE_NUMBA = False
    numba_version = None

    def njit(*args, **kwargs):
        """No-op stand-in for ``numba.njit`` used when Numba is absent."""
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _identity(func):
            return func

        return _identity
