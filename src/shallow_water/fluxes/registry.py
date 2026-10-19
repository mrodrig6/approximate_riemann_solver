"""Names and integer codes for the available numerical fluxes.

Integer codes rather than function objects, because the dispatch in
:mod:`src.shallow_water.fluxes.interface` runs inside a Numba kernel, which
cannot take a Python function as an argument without extra machinery.

The code values are this subpackage's own and are deliberately *not* the Euler
ones: there is no AUSM+-up here, and ``hllc`` is an alias rather than a flux
of its own, so a shared table would have holes in it.  Nothing crosses between
the two systems, and :func:`flux_code` is the only way in.
"""

from __future__ import annotations

__all__ = ["FLUXES", "FLUX_NAMES", "ROBUST_FLUXES", "flux_code"]

FLUX_ROE = 0
FLUX_HLL = 1
FLUX_RUSANOV = 2
FLUX_LF = 3

#: Accepted spellings.  ``hllc`` maps to HLL deliberately: see the module
#: docstring.
FLUXES = {
    "roe": FLUX_ROE,
    "hll": FLUX_HLL,
    "hlle": FLUX_HLL,
    "hllc": FLUX_HLL,
    "rusanov": FLUX_RUSANOV,
    "llf": FLUX_RUSANOV,
    "laxfriedrichs": FLUX_LF,
    "lf": FLUX_LF,
}

#: Canonical display name for each code.
FLUX_NAMES = {
    FLUX_ROE: "roe",
    FLUX_HLL: "hll",
    FLUX_RUSANOV: "rusanov",
    FLUX_LF: "lax_friedrichs",
}

#: Fluxes that run every built-in problem at second order.
#:
#: All of them, where the Euler set excludes three.  The reason is the one
#: structural difference between the systems: a dry bed is a regular state of
#: the shallow-water equations, where a vacuum is pathological for a gas.  The
#: Roe linearisation still fails at a dry interface -- the square-root
#: weighting divides by zero -- but there it falls back to HLL for that one
#: interface and the run continues, where the Euler Roe flux has nothing to
#: fall back to and the solver stops.  Kept as a frozenset anyway, so the
#: solver and the tests can ask the question the same way for both systems.
ROBUST_FLUXES = frozenset({FLUX_ROE, FLUX_HLL, FLUX_RUSANOV, FLUX_LF})

def flux_code(flux):
    """Normalise a flux specifier (name or integer code) to its integer code."""
    if isinstance(flux, str):
        key = flux.strip().lower()
        for ch in (" ", "_", "-", "+"):
            key = key.replace(ch, "")
        if key not in FLUXES:
            raise ValueError(
                f"unknown flux {flux!r}; choose from "
                f"{', '.join(sorted(set(FLUX_NAMES.values())))}"
            )
        return FLUXES[key]
    code = int(flux)
    if code not in FLUX_NAMES:
        raise ValueError(
            f"unknown flux code {code}; valid codes are {sorted(FLUX_NAMES)}"
        )
    return code
