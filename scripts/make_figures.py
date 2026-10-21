#!/usr/bin/env python3
"""Compatible entry point for the figure source beside docs/figures/*.png."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "docs/figures/make_figures.py"),
                   run_name="__main__")
