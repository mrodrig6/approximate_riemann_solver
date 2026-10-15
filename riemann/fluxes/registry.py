"""Names and integer codes for the available numerical fluxes.

Integer codes rather than function objects, because the dispatch in
:mod:`riemann.fluxes.interface` runs inside a Numba kernel, which cannot take
a Python function as an argument without extra machinery.
"""

from __future__ import annotations

__all__ = ["FLUXES", "FLUX_NAMES", "ROBUST_FLUXES", "flux_code"]

FLUX_ROE = 0
FLUX_HLL = 1
FLUX_HLLC = 2
FLUX_RUSANOV = 3
FLUX_LF = 4
FLUX_AUSMUP = 5

#: Accepted spellings, including aliases, mapped to their integer code.
#: Lookup is case-insensitive and ignores spaces, underscores, hyphens and
#: plus signs, so ``"AUSM+-up"`` and ``"ausmup"`` are the same flux.
FLUXES = {
    "roe": FLUX_ROE,
    # HLL here uses Einfeldt's wave-speed estimates, which is what the
    # literature calls HLLE, so accept that spelling too.
    "hll": FLUX_HLL,
    "hlle": FLUX_HLL,
    "hllc": FLUX_HLLC,
    # Rusanov is local Lax-Friedrichs; both names are in common use.
    "rusanov": FLUX_RUSANOV,
    "llf": FLUX_RUSANOV,
    "locallaxfriedrichs": FLUX_RUSANOV,
    "laxfriedrichs": FLUX_LF,
    "lf": FLUX_LF,
    "ausmup": FLUX_AUSMUP,
    "ausmplusup": FLUX_AUSMUP,
    "ausm": FLUX_AUSMUP,
}

#: Canonical display name for each code.
FLUX_NAMES = {
    FLUX_ROE: "roe",
    FLUX_HLL: "hll",
    FLUX_HLLC: "hllc",
    FLUX_RUSANOV: "rusanov",
    FLUX_LF: "lax_friedrichs",
    FLUX_AUSMUP: "ausmup",
}

#: Fluxes that run every built-in problem at second order, including the
#: near-vacuum Toro Test 2.
#:
#: HLL and HLLC are positivity-preserving by construction with Einfeldt wave
#: speeds, and Lax-Friedrichs is too diffusive to be troubled.  Rusanov and
#: AUSM+-up are absent: both run every problem as *first-order* fluxes, but
#: their positivity does not extend to the second-order TVD correction.  "The
#: flux is positivity-preserving" and "the scheme is" are different claims.
ROBUST_FLUXES = frozenset({FLUX_HLL, FLUX_HLLC, FLUX_LF})


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
