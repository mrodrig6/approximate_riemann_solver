r"""Figure generation for the Riemann solvers.

The solvers return plain data objects and this module draws them, so a run
can be made without producing a figure and any number of runs can be overlaid
without touching solver code.

A note on LaTeX
---------------
Matplotlib can typeset labels with a real LaTeX installation
(``text.usetex=True``), but that silently fails -- or crashes with an opaque
error -- on any machine without LaTeX, dvipng and the right font packages.
Everything here therefore uses matplotlib's **built-in mathtext** renderer,
which needs no external programs and is always available.  ``mathtext.fontset
= "cm"`` gives Computer Modern glyphs, so the labels look like LaTeX without
depending on it.

The practical rules the labels in this module follow, all of them things that
bite when mathtext is treated as if it were full LaTeX:

* Only single-symbol math inside ``$...$``: ``r"$\rho$"``, ``r"$u$"``.
* Units and words stay *outside* the math, as plain text
  (``r"$\rho$ (kg/m$^3$)"``), because mathtext's text-mode support is limited
  and ``\text{}`` is not available.
* Always raw strings, so ``\rho`` is not read as a carriage return plus "ho".
* No ``\begin{...}`` environments, ``\label``, ``\mathbf`` on digits, or
  ``\;``/``\!`` spacing macros -- mathtext does not implement them.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PALETTE",
    "use_paper_style",
    "plot_comparison",
    "plot_series",
    "plot_shallow_water",
    "save_figure",
]

#: Categorical colours, slots assigned in fixed order and never cycled.  The
#: six are validated for colour-vision-deficiency separation as an adjacent
#: pair list; three of them fall below a 3:1 contrast ratio on this surface,
#: which is why every figure using them also carries a legend and distinct
#: marker shapes rather than relying on colour alone.  The reference
#: ("exact") curve deliberately uses neutral ink rather than a series colour,
#: since it is the baseline every series is read against, not one of them.
PALETTE = {
    "exact": "#3f3f3c",
    "series": [
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua
        "#eda100",  # yellow
        "#e87ba4",  # magenta
        "#6250d6",  # violet
    ],
    "grid": "#d8d7d2",
    "text": "#0b0b0b",
}

#: Marker cycle, used as a *secondary* encoding so the series are
#: distinguishable without relying on colour alone.
MARKERS = ["o", "s", "^", "D", "v", "P"]

#: Axis labels, one per field.  Mathtext only -- see the module docstring.
_LABELS = {
    "rho": r"$\rho$",
    "u": r"$u$",
    "p": r"$p$",
    "e": r"$e$",
}

#: Longer descriptions used in panel titles, kept as plain text so there is no
#: chance of a mathtext parse failure in a string that is mostly words.
_TITLES = {
    "rho": "Density",
    "u": "Velocity",
    "p": "Pressure",
    "e": "Specific internal energy",
}


def use_paper_style():
    """Apply a consistent, LaTeX-free rcParam style.

    Call once before creating figures.  Notably sets ``text.usetex`` to
    ``False`` *explicitly*: if a user's own ``matplotlibrc`` turns it on, every
    figure in this package would otherwise fail on a machine without LaTeX.
    """
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "text.usetex": False,          # never shell out to LaTeX
            "mathtext.fontset": "cm",      # Computer Modern look, no LaTeX needed
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
            "axes.edgecolor": "#3f3f3c",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,        # 2px data lines
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )


def _unit_suffix(dimensional):
    """Axis-label unit suffixes, as plain text outside any math.

    ``$^3$`` is the one bit of math in the units; it is a bare superscript,
    which mathtext handles reliably.
    """
    if not dimensional:
        return {"rho": "", "u": "", "p": "", "e": ""}
    return {
        "rho": r" (kg/m$^3$)",
        "u": " (m/s)",
        "p": " (Pa)",
        "e": " (J/kg)",
    }


def plot_comparison(
    exact=None,
    numerical=(),
    fields=("rho", "u", "p", "e"),
    title=None,
    dimensional=False,
    markevery=1,
    xlabel="$x$",
    figsize=(9.0, 6.4),
):
    """Four-panel comparison of an exact solution and one or more Roe runs.

    Parameters
    ----------
    exact : ExactSolution, optional
        Reference solution, drawn as a continuous neutral-ink line.  It is the
        baseline, so it is deliberately *not* given a categorical colour.
    numerical : sequence
        Zero or more :class:`~src.euler.solver.RoeSolution` objects, or
        ``(solution, label)`` pairs to override the automatic label.  Drawn as
        open markers so that cell values are visible as discrete samples --
        which is what they are -- and so the series remain distinguishable
        without colour.
    fields : sequence of str
        Which of ``"rho"``, ``"u"``, ``"p"``, ``"e"`` to draw, in order.
    title : str, optional
        Figure suptitle.
    dimensional : bool
        Append SI units to the axis labels.
    markevery : int
        Plot every ``markevery``-th cell.  Useful above ~400 cells, where
        drawing every marker turns the curve into a solid band.
    xlabel : str
        Label for the shared x axis.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (fig, axes) : tuple
    """
    import matplotlib.pyplot as plt

    use_paper_style()
    units = _unit_suffix(dimensional)

    n_panels = len(fields)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    # Normalise `numerical` to a list of (solution, label) pairs.
    runs = []
    for item in numerical:
        if isinstance(item, tuple):
            runs.append(item)
        else:
            runs.append((item, item.label))

    for ax, field in zip(axes, fields):
        if exact is not None:
            ax.plot(
                exact.x,
                getattr(exact, field),
                color=PALETTE["exact"],
                linewidth=2.0,
                label="Exact",
                zorder=2,
            )
        for idx, (sol, label) in enumerate(runs):
            ax.plot(
                sol.x,
                getattr(sol, field),
                linestyle="none",
                marker=MARKERS[idx % len(MARKERS)],
                markersize=4.0,
                markerfacecolor="none",
                markeredgewidth=1.1,
                color=PALETTE["series"][idx % len(PALETTE["series"])],
                markevery=markevery,
                label=label,
                zorder=3 + idx,
            )
        ax.set_ylabel(_LABELS[field] + units[field])
        ax.set_title(_TITLES[field], loc="left", color="#52514e")
        ax.margins(x=0.02, y=0.08)

    # Label only the bottom row's x axis; the panels share it.
    for ax in axes[len(fields) - n_cols:len(fields)]:
        ax.set_xlabel(xlabel)
    # Hide any unused panel.
    for ax in axes[len(fields):]:
        ax.axis("off")

    # A legend is always present when there are two or more curves; identity is
    # then never carried by colour alone (line vs. markers, and distinct marker
    # shapes between series).
    handles, labels = axes[0].get_legend_handles_labels()
    if len(labels) >= 2:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            bbox_to_anchor=(0.5, -0.02),
        )

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    else:
        fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig, axes


def plot_series(x_values, y_values, labels, xlabel, ylabel, title=None, logx=False,
                logy=False, figsize=(6.0, 4.2)):
    """Generic multi-series line plot, used for the grid-convergence figure.

    Kept separate from :func:`plot_comparison` because a convergence study is a
    different *form*: it answers "what is the trend in the error" rather than
    "what does the solution look like", and so wants log axes, a reference
    slope and direct labelling rather than four stacked state panels.
    """
    import matplotlib.pyplot as plt

    use_paper_style()
    fig, ax = plt.subplots(figsize=figsize)
    for idx, (y, label) in enumerate(zip(y_values, labels)):
        ax.plot(
            x_values,
            y,
            marker=MARKERS[idx % len(MARKERS)],
            markersize=5,
            markerfacecolor="none",
            color=PALETTE["series"][idx % len(PALETTE["series"])],
            label=label,
        )
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if len(labels) >= 2:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_shallow_water(
    exact=None,
    numerical=(),
    fields=("h", "u"),
    title=None,
    markevery=1,
    xlabel="$x$ (m)",
    figsize=(9.0, 3.8),
):
    """Comparison figure for the shallow-water equations.

    The counterpart of :func:`plot_comparison`, with two panels rather than
    four because the system has two conserved variables.  Accepts the
    ``ShallowWaterSolution`` of the exact solver and any number of
    ``ShallowWaterResult`` objects, or ``(result, label)`` pairs.

    Available fields are ``"h"`` (depth), ``"u"`` (velocity) and
    ``"discharge"`` (``h u``, the second conserved variable).
    """
    import matplotlib.pyplot as plt

    use_paper_style()
    labels = {"h": "$h$ (m)", "u": "$u$ (m/s)", "discharge": "$hu$ (m$^2$/s)"}
    titles = {"h": "Depth", "u": "Velocity", "discharge": "Discharge"}

    n_panels = len(fields)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes).ravel()

    runs = []
    for item in numerical:
        runs.append(item if isinstance(item, tuple) else (item, item.label))

    for ax, field in zip(axes, fields):
        if exact is not None:
            ax.plot(
                exact.x, getattr(exact, field), color=PALETTE["exact"],
                linewidth=2.0, label="Exact", zorder=2,
            )
        for idx, (sol, label) in enumerate(runs):
            ax.plot(
                sol.x, getattr(sol, field), linestyle="none",
                marker=MARKERS[idx % len(MARKERS)], markersize=4.0,
                markerfacecolor="none", markeredgewidth=1.1,
                color=PALETTE["series"][idx % len(PALETTE["series"])],
                markevery=markevery, label=label, zorder=3 + idx,
            )
        ax.set_ylabel(labels[field])
        ax.set_xlabel(xlabel)
        ax.set_title(titles[field], loc="left", color="#52514e")
        ax.margins(x=0.02, y=0.1)

    handles, lbls = axes[0].get_legend_handles_labels()
    if len(lbls) >= 2:
        fig.legend(
            handles, lbls, loc="lower center", ncol=min(len(lbls), 4),
            bbox_to_anchor=(0.5, -0.04),
        )
    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    else:
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig, axes


def save_figure(fig, path):
    """Save ``fig`` to ``path``, creating the parent directory if needed."""
    import os

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path)
    return path
