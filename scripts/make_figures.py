#!/usr/bin/env python3
"""Regenerate every figure used in ``docs/``.

Run from the repository root::

    python scripts/make_figures.py

Each figure is written to ``docs/figures/``.  The script is deliberately
self-contained and deterministic -- the solvers have no randomness -- so the
committed figures can always be reproduced and diffed.

Every figure here uses matplotlib's built-in mathtext for its labels, never
``text.usetex``.  See the module docstring of :mod:`src.core.plotting` for why.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Allow running from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")  # no display needed; must precede the pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from src import (  # noqa: E402
    acoustic_pulse,
    cell_centres,
    get_problem,
    solve_exact,
    solve_field,
    solve_roe,
)
from src.shallow_water import get_problem as swe_problem  # noqa: E402
from src.shallow_water import solve as swe_solve  # noqa: E402
from src.shallow_water import solve_exact as swe_solve_exact  # noqa: E402
from src.core.plotting import (  # noqa: E402
    MARKERS,
    PALETTE,
    plot_shallow_water,
    plot_comparison,
    plot_series,
    save_figure,
    use_paper_style,
)

FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures"
)


def _autoscale_y(ax, x_lo, x_hi):
    """Rescale an axis' y limits to the data inside ``[x_lo, x_hi]``."""
    lo, hi = np.inf, -np.inf
    for line in ax.get_lines():
        xd, yd = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
        inside = (xd >= x_lo) & (xd <= x_hi)
        if np.any(inside):
            lo = min(lo, float(np.min(yd[inside])))
            hi = max(hi, float(np.max(yd[inside])))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)


def figure_sod():
    """Headline figure: Sod's problem, exact vs. second-order TVD Roe."""
    prob = get_problem("sod")
    exact = solve_exact(
        prob.gamma, prob.left, prob.right, prob.t_end, n=2000,
        x_diaphragm=prob.x_diaphragm,
    )
    roe = solve_roe(
        prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
        x_diaphragm=prob.x_diaphragm, limiter="superbee",
    )
    fig, _ = plot_comparison(
        exact=exact,
        numerical=[roe],
        title=f"Sod's problem at $t = {prob.t_end}$, 200 cells",
    )
    return save_figure(fig, os.path.join(FIG_DIR, "sod_comparison.png"))


def figure_order_and_limiters():
    """First order vs. minmod vs. superbee, zoomed on the contact.

    This is the figure that justifies the whole TVD machinery: the three runs
    use the identical Roe flux and differ only in the limiter applied to the
    second-order correction.
    """
    prob = get_problem("sod")
    exact = solve_exact(
        prob.gamma, prob.left, prob.right, prob.t_end, n=2000,
        x_diaphragm=prob.x_diaphragm,
    )
    runs = [
        (solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=100,
                   second_order=False), "1st-order Roe"),
        (solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=100,
                   limiter="minmod"), "2nd order, minmod"),
        (solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=100,
                   limiter="superbee"), "2nd order, superbee"),
    ]
    fig, axes = plot_comparison(
        exact=exact,
        numerical=runs,
        fields=("rho", "e"),
        title="Effect of the flux limiter (Sod, 100 cells)",
        figsize=(9.0, 3.8),
    )
    # Zoom both panels onto the contact, where the schemes differ most, and
    # rescale y to the visible window -- otherwise the full-domain y range
    # leaves the interesting part compressed into a sliver of the panel.
    for ax in axes[:2]:
        ax.set_xlim(0.55, 0.9)
        _autoscale_y(ax, 0.55, 0.9)
    return save_figure(fig, os.path.join(FIG_DIR, "limiter_comparison.png"))


def figure_entropy_fix():
    """The entropy glitch, with and without the Harten-Hyman fix.

    Two resolutions are shown side by side because the decisive point is not
    that the unfixed scheme is less accurate -- it is that its error at the
    sonic point does *not* shrink under refinement.
    """
    prob = get_problem("sonic_rarefaction")
    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)

    for ax, nx in zip(axes, (100, 400)):
        exact = solve_exact(
            prob.gamma, prob.left, prob.right, prob.t_end, n=2000,
            x_diaphragm=prob.x_diaphragm,
        )
        kw = dict(
            gamma=prob.gamma, w_l=prob.left, w_r=prob.right, t_end=prob.t_end,
            nx=nx, x_diaphragm=prob.x_diaphragm, second_order=False,
        )
        fixed = solve_roe(entropy_fix=True, **kw)
        unfixed = solve_roe(entropy_fix=False, **kw)

        ax.plot(exact.x, exact.rho, color=PALETTE["exact"], lw=2.0, label="Exact",
                zorder=2)
        ax.plot(unfixed.x, unfixed.rho, ls="none", marker="s", ms=4.5,
                markerfacecolor="none", markeredgewidth=1.1,
                color=PALETTE["series"][1], label="Roe, no entropy fix", zorder=3)
        ax.plot(fixed.x, fixed.rho, ls="none", marker="o", ms=4.0,
                markerfacecolor="none", markeredgewidth=1.1,
                color=PALETTE["series"][0], label="Roe, Harten-Hyman fix", zorder=4)
        # The sonic point: where u - a changes sign inside the left fan.
        ax.axvline(0.301, color="#9a9992", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.annotate(
            "sonic point", xy=(0.301, 0.95), xycoords=("data", "axes fraction"),
            xytext=(4, 0), textcoords="offset points", fontsize=8,
            color="#52514e", va="top",
        )
        ax.set_xlim(0.15, 0.45)
        ax.set_xlabel("$x$")
        ax.set_title(f"{nx} cells", loc="left", color="#52514e")
    axes[0].set_ylabel(r"$\rho$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Transonic rarefaction: the entropy glitch does not vanish under refinement",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return save_figure(fig, os.path.join(FIG_DIR, "entropy_fix.png"))


def figure_toro_suite():
    """Density for the four Toro tests the Roe scheme can handle."""
    names = ["sod", "toro3", "toro4", "toro5"]
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4))
    axes = axes.ravel()
    for ax, name in zip(axes, names):
        prob = get_problem(name)
        exact = solve_exact(
            prob.gamma, prob.left, prob.right, prob.t_end, n=2000,
            x_diaphragm=prob.x_diaphragm,
        )
        roe = solve_roe(
            prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
            x_diaphragm=prob.x_diaphragm, limiter="superbee",
        )
        ax.plot(exact.x, exact.rho, color=PALETTE["exact"], lw=2.0, label="Exact")
        ax.plot(roe.x, roe.rho, ls="none", marker="o", ms=3.5,
                markerfacecolor="none", markeredgewidth=1.0,
                color=PALETTE["series"][0], label="2nd-order Roe, superbee")
        ax.set_title(f"{name}  ($t = {prob.t_end:g}$)", loc="left", color="#52514e")
        ax.set_ylabel(r"$\rho$")
        ax.set_xlabel("$x$")
        ax.margins(y=0.1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Toro's test problems, 200 cells", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return save_figure(fig, os.path.join(FIG_DIR, "toro_suite.png"))


def figure_convergence():
    """Grid convergence in the L1 norm for Sod's problem.

    The expected result is a rate near 1, *not* 2, even though the scheme is
    formally second order: the exact solution contains a shock and a contact,
    and an O(1) error over an O(dx) region contributes O(dx) to the L1 norm no
    matter how accurate the scheme is in the smooth parts.  What the
    second-order correction buys is a much smaller *constant*, which is what
    the figure actually shows.
    """
    resolutions = np.array([50, 100, 200, 400, 800, 1600])
    configs = [
        (dict(second_order=False), "1st-order Roe"),
        (dict(second_order=True, limiter="minmod"), "2nd order, minmod"),
        (dict(second_order=True, limiter="superbee"), "2nd order, superbee"),
    ]
    prob = get_problem("sod")
    curves, labels = [], []
    for kwargs, label in configs:
        errors = []
        for nx in resolutions:
            roe = solve_roe(
                prob.gamma, prob.left, prob.right, prob.t_end, nx=int(nx),
                x_diaphragm=prob.x_diaphragm, **kwargs
            )
            exact = solve_exact(
                prob.gamma, prob.left, prob.right, prob.t_end, n=int(nx),
                x_diaphragm=prob.x_diaphragm,
            )
            errors.append(np.mean(np.abs(roe.rho - exact.rho)))
        curves.append(errors)
        labels.append(label)

    fig, ax = plot_series(
        resolutions, curves, labels,
        xlabel="number of cells",
        ylabel=r"$L_1$ error in $\rho$",
        title="Grid convergence, Sod's problem",
        logx=True, logy=True,
    )
    # A first-order reference slope, anchored to the coarsest superbee point.
    ref = curves[-1][0] * (resolutions[0] / resolutions)
    ax.plot(resolutions, ref, color="#9a9992", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        "slope $-1$", xy=(resolutions[-2], ref[-2]), xytext=(6, 6),
        textcoords="offset points", fontsize=9, color="#52514e",
    )
    fig.tight_layout()
    return save_figure(fig, os.path.join(FIG_DIR, "convergence.png"))


def figure_air_shock_tube():
    """The dimensional (SI) shock-tube case."""
    prob = get_problem("air_shock_tube")
    exact = solve_exact(
        prob.gamma, prob.left, prob.right, prob.t_end, n=2000,
        x_diaphragm=prob.x_diaphragm,
    )
    roe = solve_roe(
        prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
        x_diaphragm=prob.x_diaphragm, limiter="superbee",
    )
    fig, _ = plot_comparison(
        exact=exact,
        numerical=[roe],
        title=(
            "Air shock tube, SI units, "
            rf"$t = {prob.t_end * 1e3:g}$ ms, 200 cells"
        ),
        dimensional=True,
        xlabel="$x$ (m)",
    )
    return save_figure(fig, os.path.join(FIG_DIR, "air_shock_tube.png"))


def figure_flux_comparison():
    """Small multiples: every flux against the exact solution on Sod's problem.

    Small multiples rather than six series on one axis: with six schemes the
    single-panel version becomes a colour-matching exercise, while one panel
    per flux lets each be read on its own and compared by shape.  The view is
    zoomed on the contact, which is where the fluxes actually differ -- all of
    them capture the shock in one or two cells.

    Any flux that cannot run the case is labelled as such rather than being
    silently dropped.
    """
    prob = get_problem("sod")
    nx = 100
    exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=2000)

    names = ["roe", "hllc", "hll", "ausmup", "rusanov", "lax_friedrichs"]
    pretty = {
        "roe": "Roe", "hllc": "HLLC", "hll": "HLL", "ausmup": "AUSM+-up",
        "rusanov": "Rusanov", "lax_friedrichs": "Lax-Friedrichs",
    }

    use_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.6), sharex=True, sharey=True)
    axes = axes.ravel()

    for idx, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(exact.x, exact.rho, color=PALETTE["exact"], lw=2.0, zorder=2,
                label="Exact")
        try:
            sol = solve_roe(
                prob.gamma, prob.left, prob.right, prob.t_end, nx=nx,
                flux=name, limiter="superbee",
            )
            ax.plot(
                sol.x, sol.rho, ls="none", marker=MARKERS[idx % len(MARKERS)],
                ms=4.5, markerfacecolor="none", markeredgewidth=1.1,
                color=PALETTE["series"][idx % len(PALETTE["series"])], zorder=3,
            )
            # Contact width, in cells, as a direct label -- the number the
            # panel is really about.
            lo, hi = 0.2656, 0.4263
            span = hi - lo
            width = int(np.sum((sol.rho > lo + 0.05 * span)
                               & (sol.rho < hi - 0.05 * span)))
            note = f"contact: {width} cells"
        except FloatingPointError:
            note = "cannot start\nfrom rest"
        ax.set_title(pretty[name], loc="left", color="#52514e")
        ax.annotate(
            note, xy=(0.04, 0.08), xycoords="axes fraction", fontsize=8.5,
            color="#52514e",
        )
        ax.set_xlim(0.6, 0.85)
        ax.set_ylim(0.22, 0.47)

    for ax in axes[3:]:
        ax.set_xlabel("$x$")
    for ax in (axes[0], axes[3]):
        ax.set_ylabel(r"$\rho$")

    fig.suptitle(
        "Numerical fluxes at the contact discontinuity (Sod, 100 cells)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, os.path.join(FIG_DIR, "flux_comparison.png"))


def figure_toro2_positivity():
    """Toro's Test 2, the near-vacuum case that Roe cannot run.

    The point of adding positivity-preserving fluxes, in one figure: the exact
    solution has a star region at p* = 0.0019, and HLLC and HLL both resolve
    it, while the Roe flux drives the internal energy negative in the first
    time step and the run stops.
    """
    prob = get_problem("toro2")
    exact = solve_exact(prob.gamma, prob.left, prob.right, prob.t_end, n=2000)

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6))
    fields = ("rho", "u", "e")
    labels = {"rho": r"$\rho$", "u": "$u$", "e": "$e$"}
    titles = {"rho": "Density", "u": "Velocity",
              "e": "Specific internal energy"}

    runs = []
    for idx, name in enumerate(("hllc", "hll")):
        runs.append((
            solve_roe(prob.gamma, prob.left, prob.right, prob.t_end, nx=200,
                      flux=name, limiter="superbee"),
            name.upper(),
            idx,
        ))

    for ax, field in zip(axes, fields):
        ax.plot(exact.x, getattr(exact, field), color=PALETTE["exact"], lw=2.0,
                label="Exact", zorder=2)
        for sol, label, idx in runs:
            ax.plot(
                sol.x, getattr(sol, field), ls="none", marker=MARKERS[idx],
                ms=4.0, markerfacecolor="none", markeredgewidth=1.1,
                color=PALETTE["series"][idx], label=label, zorder=3 + idx,
            )
        ax.set_ylabel(labels[field])
        ax.set_xlabel("$x$")
        ax.set_title(titles[field], loc="left", color="#52514e")
        ax.margins(y=0.1)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "Toro Test 2 ($p_* \\approx 0.0019$): Roe fails outright; "
        "HLLC and HLL do not",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return save_figure(fig, os.path.join(FIG_DIR, "toro2_positivity.png"))


def figure_order_of_accuracy():
    """Formal order of accuracy, measured on smooth data.

    The grid-convergence figure on Sod shows slope -1 for every scheme,
    which is correct but says nothing about their formal order: a
    discontinuity caps the L1 rate at 1 whatever the method.  On smooth data
    the orders separate, and the one-step ADER scheme and the flux-limited
    TVD scheme both reach 2.
    """
    resolutions = np.array([50, 100, 200, 400, 800])
    t_end = 0.15
    ref = solve_field(
        1.4, acoustic_pulse(cell_centres(6400)), t_end,
        scheme="ader2", limiter="none", cfl=0.8,
    )

    configs = [
        (dict(scheme="tvd", second_order=False), "1st order"),
        (dict(scheme="tvd", limiter="none"), "TVD, 2nd order"),
        (dict(scheme="ader2", limiter="none"), "ADER2"),
    ]
    curves, labels = [], []
    for kwargs, label in configs:
        errors = []
        for nx in resolutions:
            x = cell_centres(int(nx))
            sol = solve_field(1.4, acoustic_pulse(x), t_end, cfl=0.8, **kwargs)
            errors.append(np.mean(np.abs(sol.rho - np.interp(x, ref.x, ref.rho))))
        curves.append(errors)
        labels.append(label)

    fig, ax = plot_series(
        resolutions, curves, labels,
        xlabel="number of cells",
        ylabel=r"$L_1$ error in $\rho$",
        title="Order of accuracy on a smooth acoustic pulse",
        logx=True, logy=True,
    )
    for exponent, anchor, text in ((1.0, curves[0][0], "slope $-1$"),
                                   (2.0, curves[2][0], "slope $-2$")):
        ref_line = anchor * (resolutions[0] / resolutions) ** exponent
        ax.plot(resolutions, ref_line, color="#9a9992", lw=1.0,
                ls=(0, (4, 3)), zorder=1)
        ax.annotate(
            text, xy=(resolutions[-2], ref_line[-2]), xytext=(6, 4),
            textcoords="offset points", fontsize=9, color="#52514e",
        )
    # Label only the resolutions actually run; matplotlib's default log
    # minor ticks collide at this aspect ratio.
    ax.set_xticks(resolutions)
    ax.set_xticklabels([str(int(n)) for n in resolutions])
    ax.set_xticks([], minor=True)
    fig.tight_layout()
    return save_figure(fig, os.path.join(FIG_DIR, "order_of_accuracy.png"))


def figure_shallow_water():
    """Dam break on a wet bed and on a dry bed, against the exact solutions."""
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))

    for row, name in enumerate(("dam_break", "dam_break_dry")):
        prob = swe_problem(name)
        exact = swe_solve_exact(
            prob.left, prob.right, prob.t_end, gravity=prob.gravity, n=2000,
            x_diaphragm=prob.x_diaphragm,
        )
        sol = swe_solve(
            prob.left, prob.right, prob.t_end, gravity=prob.gravity, nx=200,
            x_diaphragm=prob.x_diaphragm, cfl=prob.cfl, flux="hll",
        )
        for col, field in enumerate(("h", "u")):
            ax = axes[row, col]
            ax.plot(exact.x, getattr(exact, field), color=PALETTE["exact"],
                    lw=2.0, label="Exact", zorder=2)
            ax.plot(sol.x, getattr(sol, field), ls="none", marker="o", ms=3.5,
                    markerfacecolor="none", markeredgewidth=1.0,
                    color=PALETTE["series"][row], label="HLL, 2nd order",
                    zorder=3)
            ax.set_ylabel("$h$ (m)" if field == "h" else "$u$ (m/s)")
            ax.margins(y=0.12)
        axes[row, 0].set_title(
            "wet bed" if row == 0 else "dry bed", loc="left", color="#52514e"
        )
        axes[row, 1].set_title(
            f"$t = {prob.t_end:g}$ s", loc="left", color="#52514e"
        )
    for ax in axes[1, :]:
        ax.set_xlabel("$x$ (m)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Shallow water: dam break, 200 cells", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    return save_figure(fig, os.path.join(FIG_DIR, "shallow_water_dam_break.png"))


def figure_shallow_water_fluxes():
    """The shallow-water fluxes compared, zoomed on the bore."""
    prob = swe_problem("dam_break")
    exact = swe_solve_exact(prob.left, prob.right, prob.t_end, n=2000)
    pretty = {"roe": "Roe", "hll": "HLL", "rusanov": "Rusanov",
              "lax_friedrichs": "Lax-Friedrichs"}
    runs = []
    for name in ("roe", "hll", "rusanov", "lax_friedrichs"):
        sol = swe_solve(prob.left, prob.right, prob.t_end, nx=100, flux=name)
        runs.append((sol, pretty[name]))
    fig, axes = plot_shallow_water(
        exact=exact, numerical=runs,
        title="Shallow-water fluxes at the bore (dam break, 100 cells)",
    )
    for ax in axes:
        ax.set_xlim(0.6, 0.9)
        _autoscale_y(ax, 0.6, 0.9)
    return save_figure(
        fig, os.path.join(FIG_DIR, "shallow_water_fluxes.png")
    )


FIGURES = [
    figure_sod,
    figure_order_and_limiters,
    figure_entropy_fix,
    figure_toro_suite,
    figure_convergence,
    figure_air_shock_tube,
    figure_flux_comparison,
    figure_toro2_positivity,
    figure_order_of_accuracy,
    figure_shallow_water,
    figure_shallow_water_fluxes,
]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for func in FIGURES:
        path = func()
        plt.close("all")
        print(f"wrote {os.path.relpath(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
