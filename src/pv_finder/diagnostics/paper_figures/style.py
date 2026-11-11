"""ATLAS-style helpers shared by the paper figures.

Builds on ``gnn.diagnostics.plot_style`` (mplhep ATLAS style, PNG + PDF output)
so the paper figures and the GNN figures share one look.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

from gnn.diagnostics.plot_style import save_figure, use_atlas_style  # noqa: E402

# Figures stay "Internal" until the collaboration approves them.
SIM_STATUS = "Simulation Internal"
DATA_STATUS = "Internal"

# Okabe-Ito colour-blind-safe palette (same as gnn.diagnostics.plot_style).
PVF_COLOR = "#0072B2"
AMVF_COLOR = "#D55E00"
CATEGORY_STYLE: dict[str, dict[str, str]] = {
    "all": {"label": "All reconstructed", "color": "black", "marker": "o"},
    "clean": {"label": "Clean", "color": "#009E73", "marker": "s"},
    "merged": {"label": "Merged", "color": "#E69F00", "marker": "^"},
    "split": {"label": "Split", "color": "#56B4E9", "marker": "v"},
    "fake": {"label": "Fake", "color": "#CC79A7", "marker": "D"},
}
CATEGORIES: tuple[str, ...] = ("clean", "merged", "split", "fake")

__all__ = [
    "AMVF_COLOR",
    "CATEGORIES",
    "CATEGORY_STYLE",
    "DATA_STATUS",
    "PVF_COLOR",
    "SIM_STATUS",
    "label_block",
    "new_figure",
    "save",
]


def new_figure(width: float = 8.0, height: float = 6.0) -> tuple[plt.Figure, plt.Axes]:
    """Return a single-axes figure in the mplhep ATLAS style."""
    use_atlas_style()
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def label_block(
    ax: plt.Axes,
    status: str,
    lines: Sequence[str],
    x: float = 0.05,
    y: float = 0.95,
    line_step: float = 0.055,
    fontsize: float = 13.0,
) -> None:
    """Draw the "ATLAS <status>" label and the sample lines below it.

    Positions are axes fractions. The ATLAS label uses mplhep (inside the
    top-left corner, ``loc=1``) so the font and italic weight follow the
    collaboration convention; ``x`` and ``y`` place the sample lines under it.
    """
    hep.atlas.text(status, ax=ax, loc=1, fontsize=fontsize * 1.25)
    for i, line in enumerate(lines):
        ax.text(
            x,
            y - line_step * (i + 1) - 0.01,
            line,
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
        )


def save(fig: plt.Figure, out_dir: Path, name: str) -> list[Path]:
    """Save as 300-dpi PNG and vector PDF (delegates to the GNN helper)."""
    return save_figure(fig, out_dir, name)
