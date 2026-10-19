"""Optional-Numba shim.

Every compute kernel in this package is decorated with :func:`njit` from this
module rather than importing ``numba`` directly.  If Numba is installed the
decorator is the real ``numba.njit`` and the kernels are compiled to machine
code on first call; if it is missing the decorator degrades to a no-op and the
identical NumPy/Python source still runs, only slower.

That matters because the solvers are explicitly swept finite-volume codes:
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

    #: LLVM fast-math relaxations the kernels are compiled with.
    #:
    #: Deliberately *not* ``fastmath=True``, which is the full set and includes
    #: ``nnan`` and ``ninf`` -- licences to assume no operand is ever a NaN or
    #: an infinity.  Under those two, a comparison against a NaN may be folded
    #: away at compile time, and the solvers rely on exactly such comparisons:
    #: the positivity guards are written so that a NaN fails them, so that a
    #: step which has already gone bad is reported rather than propagated.
    #: Granting ``nnan`` tells the compiler that the case those guards exist to
    #: catch cannot arise.
    #:
    #: What is kept is what the speed actually comes from: reassociation and
    #: contraction of the arithmetic in the flux kernels, which is what lets
    #: LLVM vectorise and fuse the inner loops.
    FASTMATH = frozenset({"nsz", "arcp", "contract", "afn", "reassoc"})

    def njit(*args, **kwargs):
        """``numba.njit`` with defaults suited to this code.

        ``cache=True`` persists the compiled kernels to ``__pycache__`` so the
        ~1 s compilation cost is paid once per machine rather than once per
        process, and ``fastmath=FASTMATH`` lets LLVM reassociate the floating
        point arithmetic in the flux kernels without also assuming that NaNs
        and infinities cannot occur.

        Neither changes the *discretisation*.  Reassociation does change the
        last bits of the arithmetic, so a compiled run and a pure-Python run
        of the same case agree to roughly 1e-11 rather than exactly, which is
        why the tolerances through the test suite are relative.
        """
        kwargs.setdefault("cache", True)
        # Numba wants a mutable ``set``, and takes a copy of it.
        kwargs.setdefault("fastmath", set(FASTMATH))
        # Support both @njit and @njit(...) spellings.
        if args and callable(args[0]) and not kwargs:
            return _numba_njit(cache=True, fastmath=set(FASTMATH))(args[0])
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
