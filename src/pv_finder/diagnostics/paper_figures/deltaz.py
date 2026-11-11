"""Pairwise Delta z between reconstructed vertices and the sigmoid fit (Figures 4, 6).

For every event, Delta z = z_i - z_j over all pairs of reconstructed vertices of
one algorithm (random sign). Close pairs are lost to merging, so the
distribution has a dip at zero. The vertex-vertex resolution sigma_vtx-vtx is
the |Delta z| at which the fitted sigmoid is halfway between its lower and upper
levels (technical note Eq. sigmoid). Each algorithm is fitted to its own pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pv_finder.diagnostics.paper_figures import style
from pv_finder.diagnostics.paper_figures.sources import Resolution, sigmoid

ARMS = {
    "pvf": {"label": "PV-Finder", "color": style.PVF_COLOR, "marker": "o",
            "line": "-"},
    "amvf": {"label": "AMVF", "color": style.AMVF_COLOR, "marker": "s",
             "line": "--"},
}  # fmt: skip


def _hist(res: Resolution) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    edges = np.linspace(-6.0, 6.0, res.n_bins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    cnts, _ = np.histogram(res.dz_mm, bins=edges)
    norm = float(cnts.sum())
    return ctrs, cnts / norm, np.sqrt(cnts) / norm, norm


def plot_deltaz(
    fits: dict[str, Resolution],
    label_lines: Sequence[str],
    status: str,
    out_dir: Path,
    name: str,
    arms: Sequence[str] = ("pvf", "amvf"),
) -> list[Path]:
    """Overlay the normalized pairwise-dz distributions and their fits."""
    fig, ax = style.new_figure(8.0, 6.0)
    ymax = 0.0
    bin_mm = None
    scale = 1.0e3  # plot in units of 10^-3 so no offset text hits the label
    handles, labels = [], []
    for arm in arms:
        res = fits[arm]
        st = ARMS[arm]
        ctrs, y, e, norm = _hist(res)
        y, e, norm = y * scale, e * scale, norm / scale
        bin_mm = 12.0 / res.n_bins
        # Thin the markers for 0.04 mm bins so they stay readable.
        ms = 2.5 if res.n_bins > 100 else 4.5
        pts = ax.errorbar(ctrs, y, yerr=e, fmt=st["marker"], color=st["color"],
                          markersize=ms, elinewidth=0.8)  # fmt: skip
        xf = np.linspace(-6.0, 6.0, 2401)
        (line,) = ax.plot(xf, sigmoid(xf, *res.params) / norm, st["line"],
                          color=st["color"], linewidth=1.8)  # fmt: skip
        handles.append((pts, line))
        labels.append(f"{st['label']}, $\\sigma_{{\\mathrm{{vtx‐vtx}}}}$ = "
                      f"{res.sigma_mm:.3f} $\\pm$ {res.sigma_err_mm:.3f} mm")  # fmt: skip
        ymax = max(ymax, float(y.max()))
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(0, ymax * 1.75)
    ax.set_xlabel("$\\Delta z$ between reconstructed vertices [mm]")
    width = f"{bin_mm:.2f}".rstrip("0").rstrip(".")
    ax.set_ylabel(f"Fraction of vertex pairs / {width} mm [$10^{{-3}}$]")
    style.label_block(ax, status, label_lines)
    top = 0.95 - 0.055 * (len(label_lines) + 1) - 0.02
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.03, top),
              frameon=False, fontsize="small")  # fmt: skip
    return style.save(fig, out_dir, name)
