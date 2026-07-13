"""Shared ATLAS-style plotting helpers for TTVA performance figures.

All GNN TTVA figures go through these helpers so palette, labels, and
output formats stay consistent across diagnostics and publication plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

# ATLAS label status: this analysis is documented in an internal note, so
# figures default to "Simulation Internal". Flip here once approved.
ATLAS_STATUS = "Simulation Internal"

# Okabe-Ito colourblind-safe palette
CATEGORY_COLORS = {
    "Clean": "#009E73",
    "Merged": "#E69F00",
    "Split": "#56B4E9",
    "Fake": "#D55E00",
}
ALGO_COLORS = {
    "pvf_gnn": "#0072B2",
    "amvf": "#D55E00",
    "gnn_truth": "#009E73",
    "zeroshot": "#CC79A7",
    "retrained": "#0072B2",
}
ALGO_LABELS = {
    "pvf_gnn": "PV-Finder + GNN",
    "amvf": "AMVF",
    "gnn_truth": "GNN on truth vertices",
}


def use_atlas_style() -> None:
    """Activate the mplhep ATLAS style globally."""
    plt.style.use(hep.style.ATLAS)


def atlas_label(
    ax: plt.Axes,
    status: str = ATLAS_STATUS,
    desc: str | None = None,
    loc: int = 0,
    desc_xy: tuple[float, float] = (0.05, 0.86),
) -> None:
    """Draw the ATLAS <status> label, with an optional description line.

    Args:
        ax: Target axes.
        status: Text after "ATLAS" (e.g. "Simulation Internal").
        desc: Optional sample/selection description placed below the label.
        loc: mplhep label location code (0 = top-left inside the axes).
        desc_xy: Axes-fraction position of the description text.
    """
    hep.atlas.text(status, ax=ax, loc=loc)
    if desc:
        ax.text(
            desc_xy[0],
            desc_xy[1],
            desc,
            transform=ax.transAxes,
            fontsize="small",
            verticalalignment="top",
        )


def save_figure(fig: plt.Figure, out_dir: str | Path, name: str) -> list[Path]:
    """Save a figure as 300-dpi PNG and vector PDF; return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
    ):
        path = out_dir / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths
