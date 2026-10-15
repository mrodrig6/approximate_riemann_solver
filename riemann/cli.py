"""Command-line interface: ``python -m riemann``.

Runs one of the built-in test problems (or a custom left/right state) with both
the exact and the Roe solver, prints a short error report and optionally writes
a comparison figure.  This is the direct replacement for the MATLAB ``main.m``
and ``test_input2o1.m`` driver scripts, with their hard-coded parameters turned
into flags.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from ._jit import HAVE_NUMBA, numba_version
from .exact import solve_exact
from .fluxes import FLUX_NAMES, ROBUST_FLUXES, flux_code
from .limiters import LIMITERS
from .problems import get_problem, list_problems
from .solver import solve_roe


def build_parser():
    """Construct the argument parser (kept separate so it is testable)."""
    parser = argparse.ArgumentParser(
        prog="python -m riemann",
        description=(
            "Exact and approximate (Roe) Riemann solvers for the 1-D Euler "
            "equations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
        "Built-in problems:\n"
        + "\n".join(f"  {name:<18} {desc}" for name, desc in list_problems())
        + "\n\nFluxes (--flux), roughly most to least accurate:\n"
        "  roe              Roe with the Harten-Hyman entropy fix (Toro Ch. 11)\n"
        "  hllc             HLL with the contact restored (Toro Sec. 10.4)\n"
        "  hll / hlle       HLL, Einfeldt wave speeds (Toro Secs. 10.3, 10.5)\n"
        "  ausmup           AUSM+-up (Liou 2006); alias ausm\n"
        "  rusanov / llf    local Lax-Friedrichs (Toro Sec. 10.5)\n"
        "  lax_friedrichs   classical Lax-Friedrichs (Toro Sec. 5.3.4)"
    ),
    )
    parser.add_argument(
        "problem",
        nargs="?",
        default="sod",
        help="name of a built-in test problem (default: sod)",
    )
    parser.add_argument("--nx", type=int, default=200, help="number of cells (default: 200)")
    parser.add_argument(
        "--cfl", type=float, default=0.9, help="Courant number in (0, 1] (default: 0.9)"
    )
    parser.add_argument(
        "--flux",
        default="roe",
        metavar="FLUX",
        help=(
            "numerical flux: roe, hll (alias hlle), hllc, rusanov, "
            "lax_friedrichs (alias lf), ausmup (default: roe)"
        ),
    )
    parser.add_argument(
        "--limiter",
        default="superbee",
        choices=sorted(LIMITERS),
        help="flux limiter (default: superbee)",
    )
    parser.add_argument(
        "--first-order",
        action="store_true",
        help="drop the second-order TVD correction (plain Roe upwind)",
    )
    parser.add_argument(
        "--no-entropy-fix",
        action="store_true",
        help="disable the Harten-Hyman entropy fix",
    )
    parser.add_argument("--t-end", type=float, default=None, help="override the output time")
    parser.add_argument(
        "--n-exact",
        type=int,
        default=1000,
        help="sample points for the exact solution (default: 1000)",
    )
    parser.add_argument(
        "--plot",
        metavar="PATH",
        default=None,
        help="write a four-panel comparison figure to PATH (e.g. sod.png)",
    )
    parser.add_argument(
        "--show", action="store_true", help="display the figure in a window"
    )
    parser.add_argument(
        "--dimensional",
        action="store_true",
        help="label figure axes with SI units (use with air_shock_tube)",
    )
    parser.add_argument("--version", action="version", version=f"riemann {__version__}")
    return parser


def main(argv=None):
    """Entry point.  Returns a process exit status."""
    args = build_parser().parse_args(argv)

    try:
        problem = get_problem(args.problem)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 2

    # Validate the flux here rather than with argparse `choices`: the registry
    # keys are normalised (no spaces, underscores or hyphens), so a choices
    # list built from them would reject the readable spelling
    # "lax_friedrichs" that the help text and the docs use.
    try:
        flux_code(args.flux)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    t_end = args.t_end if args.t_end is not None else problem.t_end

    backend = f"numba {numba_version}" if HAVE_NUMBA else "pure Python (numba not installed)"
    print(f"problem        : {problem.name} -- {problem.description}")
    print(f"reference      : {problem.reference}")
    print(f"gamma, t_end   : {problem.gamma}, {t_end:g}")
    print(f"left  (rho,u,p): {problem.left}")
    print(f"right (rho,u,p): {problem.right}")
    print(f"backend        : {backend}")

    exact = solve_exact(
        problem.gamma,
        problem.left,
        problem.right,
        t_end,
        x_left=problem.x_left,
        x_right=problem.x_right,
        n=args.n_exact,
        x_diaphragm=problem.x_diaphragm,
    )
    print(
        f"\nexact solution : p* = {exact.p_star:.8g}, u* = {exact.u_star:.8g} "
        f"({exact.iterations} Newton iterations)"
    )
    print(f"wave pattern   : left {exact.left_wave}, contact, right {exact.right_wave}")

    try:
        roe = solve_roe(
            problem.gamma,
            problem.left,
            problem.right,
            t_end,
            x_left=problem.x_left,
            x_right=problem.x_right,
            nx=args.nx,
            x_diaphragm=problem.x_diaphragm,
            cfl=args.cfl,
            flux=args.flux,
            limiter=args.limiter,
            second_order=not args.first_order,
            entropy_fix=not args.no_entropy_fix,
        )
    except FloatingPointError as exc:
        # A documented breakdown of the method (see `roe_note`) should report
        # itself clearly rather than looking like a crash in the port.
        # A documented breakdown of the method should report itself clearly
        # rather than looking like a bug.  The problem's own note is printed
        # here, where it is always relevant, rather than before every run.
        print(f"\nnumerical run FAILED:\n{exc}", file=sys.stderr)
        if problem.flux_note:
            print(f"\nnote on this problem:\n{problem.flux_note}", file=sys.stderr)
        if flux_code(args.flux) not in ROBUST_FLUXES:
            robust = ", ".join(sorted(FLUX_NAMES[c] for c in ROBUST_FLUXES))
            print(f"\nfluxes that run every built-in problem: {robust}",
                  file=sys.stderr)
        return 1
    print(f"\nnumerical run  : {roe.label}, nx = {args.nx}, CFL = {roe.cfl}")
    print(f"                 {roe.steps} steps to t = {roe.t:g}")

    # Error against the exact solution, sampled on the *numerical* grid so the
    # comparison is cell-to-cell rather than interpolated.
    exact_on_grid = solve_exact(
        problem.gamma,
        problem.left,
        problem.right,
        t_end,
        x_left=problem.x_left,
        x_right=problem.x_right,
        n=args.nx,
        x_diaphragm=problem.x_diaphragm,
    )
    print("\nerror vs exact (on the numerical grid):")
    for field in ("rho", "u", "p", "e"):
        num = getattr(roe, field)
        ref = getattr(exact_on_grid, field)
        scale = max(np.max(np.abs(ref)), 1.0e-30)
        l1 = np.mean(np.abs(num - ref))
        print(f"  {field:<4} L1 = {l1:.4e}   relative L1 = {l1 / scale:.4e}")

    if args.plot or args.show:
        from .plotting import plot_comparison, save_figure

        fig, _ = plot_comparison(
            exact=exact,
            numerical=[roe],
            title=f"{problem.name}: {problem.description}  (t = {t_end:g})",
            dimensional=args.dimensional,
        )
        if args.plot:
            save_figure(fig, args.plot)
            print(f"\nfigure written to {args.plot}")
        if args.show:
            import matplotlib.pyplot as plt

            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
