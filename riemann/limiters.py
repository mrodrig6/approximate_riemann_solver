"""Flux limiters for the second-order TVD extension of the Roe scheme.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*
(3rd ed.), Chapter 13 ("High-Order and TVD Methods for Scalar Equations"),
Sec. 13.8, and Chapter 14, Sec. 14.4 where the limiter is applied wave-by-wave
to the Roe flux.

Why a limiter at all
--------------------
A plain first-order upwind (Roe) flux is monotone but smears a contact
discontinuity over ~20 cells.  Adding the second-order correction term of
Toro eq. (14.49) sharpens it, but on its own it makes the scheme oscillatory
near discontinuities (Godunov's theorem: no linear scheme can be both
second-order accurate and monotonicity preserving).  The fix is to multiply
the correction by a nonlinear *limiter* ``phi(r)``, where

    r_k = alpha_k^upwind / alpha_k^local

is the ratio of the ``k``-th wave strength at the upwind interface to the one
at the local interface.  ``phi`` is designed to lie inside Sweby's TVD region:
it returns ~1 where the solution is smooth (recovering second order) and ~0 at
an extremum or a discontinuity (falling back to first-order upwind).

The limiters are selected by an integer code so that the dispatch compiles
cleanly under Numba (which cannot pass Python functions into a nopython
kernel without extra machinery).
"""

from __future__ import annotations

from ._jit import njit

__all__ = ["LIMITERS", "LIMITER_NAMES", "limiter_code", "limiter_phi"]

#: Mapping from limiter name to the integer code understood by the kernels.
#: Codes 0 and 1 are kept identical to the MATLAB original's ``flim`` flag
#: (0 = superbee, 1 = minmod) so old driver scripts translate directly.
LIMITERS = {
    "superbee": 0,
    "minmod": 1,
    "vanleer": 2,
    "vanalbada": 3,
    "none": 4,  # unlimited: pure second order, oscillatory -- for teaching only
}

#: Inverse of :data:`LIMITERS`, for labelling plots.
LIMITER_NAMES = {code: name for name, code in LIMITERS.items()}


def limiter_code(limiter):
    """Normalise a limiter specifier (name or integer code) to its code.

    Accepts ``"superbee"``, ``"SuperBee"``, ``0`` -- all equivalent.
    """
    if isinstance(limiter, str):
        key = limiter.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        if key not in LIMITERS:
            raise ValueError(
                f"unknown limiter {limiter!r}; choose from {sorted(LIMITERS)}"
            )
        return LIMITERS[key]
    code = int(limiter)
    if code not in LIMITER_NAMES:
        raise ValueError(
            f"unknown limiter code {code}; valid codes are {sorted(LIMITER_NAMES)}"
        )
    return code


@njit
def limiter_phi(r, code):
    """Evaluate the limiter ``phi(r)``.

    All branches return 0 for ``r <= 0`` (opposite-signed neighbouring wave
    strengths, i.e. a local extremum), which is what makes the scheme TVD.

    * ``code == 0`` superbee, ``phi = max(0, min(1, 2r), min(r, 2))``:
      the most compressive limiter in the TVD region, so it keeps contacts
      sharpest, at the price of mild "staircasing" of smooth profiles.
    * ``code == 1`` minmod, ``phi = max(0, min(1, r))``: the most diffusive
      TVD limiter -- smooth, monotone, but noticeably smeared.
    * ``code == 2`` van Leer, ``phi = 2r / (1 + r)``: smooth and
      differentiable, between the two above.
    * ``code == 3`` van Albada, ``phi = r(1 + r) / (1 + r^2)``: smooth, and
      tends to 1 quickly for ``r`` near 1.
    * ``code == 4`` no limiting, ``phi = 1``: the raw second-order scheme.
      Included deliberately so the oscillations it produces can be shown.
    """
    if code == 4:  # unlimited
        return 1.0
    if r <= 0.0:
        return 0.0
    if code == 0:  # superbee
        return max(min(1.0, 2.0 * r), min(r, 2.0))
    if code == 1:  # minmod
        return min(1.0, r)
    if code == 2:  # van Leer
        return 2.0 * r / (1.0 + r)
    if code == 3:  # van Albada
        return r * (1.0 + r) / (1.0 + r * r)
    return 0.0
