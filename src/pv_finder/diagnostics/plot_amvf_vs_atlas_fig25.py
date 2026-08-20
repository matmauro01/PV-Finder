"""Plot our AMVF Run 3 MC results next to ATLAS IDTR-2021-01 Figure 25(a,b).

Consumes ``fig25_data.npz`` written by ``amvf_vs_atlas_fig25.py`` and produces

  * ``fig25a_ours.png``      -- average reconstructed vertices per category vs mu
  * ``fig25b_ours.png``      -- pairwise dz between reconstructed vertices
  * ``fig25_comparison.png`` -- ATLAS panels beside ours, if the reference
                                PNGs are supplied with ``--atlas-dir``

The ATLAS reference PNGs (``fig_25a.png``, ``fig_25b.png``) come from
https://atlas.web.cern.ch/Atlas/GROUPS/PHYSICS/PAPERS/IDTR-2021-01/

Usage:
    python -u src/pv_finder/diagnostics/plot_amvf_vs_atlas_fig25.py \
        --input-dir outputs/08_20_2026_output/amvf_vs_atlas_fig25 \
        --atlas-dir outputs/08_20_2026_output/amvf_vs_atlas_fig25/atlas_reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colours picked to match the ATLAS figure so the two panels read together.
STYLE: dict[str, dict[str, Any]] = {
    "All reconstructed": {"color": "black", "marker": "o", "zorder": 5},
    "Matched": {"color": "#2E75B6", "marker": "o", "zorder": 4},
    "Merged": {"color": "#00A170", "marker": "o", "zorder": 3},
    "Split": {"color": "#D87CA6", "marker": "o", "zorder": 2},
    "Fake": {"color": "#6EC5E9", "marker": "o", "zorder": 1},
}
CATEGORIES = ("Matched", "Merged", "Split", "Fake")


def atlas_label(ax: plt.Axes, subtitle: str, x: float = 0.04, y: float = 0.94) -> None:
    """Draw the ATLAS-style two-line label in the corner of an axis.

    Args:
        ax: Target axis.
        subtitle: Second line, e.g. the sample description.
        x: Axis-fraction x position.
        y: Axis-fraction y position.
    """
    ax.text(
        x,
        y,
        "PV-Finder",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        fontstyle="italic",
        va="top",
    )
    ax.text(
        x,
        y - 0.075,
        subtitle,
        transform=ax.transAxes,
        fontsize=10,
        va="top",
    )


def plot_fig25a(
    data: dict[str, np.ndarray], summary: dict[str, Any], out: Path, mu_max: float
) -> None:
    """Plot average reconstructed vertices per category versus pile-up.

    Args:
        data: Arrays loaded from fig25_data.npz.
        summary: Parsed fig25_summary.json.
        out: Output PNG path.
        mu_max: Upper limit of the pile-up axis.
    """
    prof = summary["profile"]
    mu = np.asarray(prof["mu"], dtype=float)
    n_per_bin = np.asarray(prof["n_events_per_bin"], dtype=float)
    keep = n_per_bin >= 20

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    y_top = 60.0
    diag = np.linspace(0, mu_max, 2)
    ax.plot(diag, diag, ls="--", color="#999999", lw=1.4)

    acc = np.asarray(prof["acceptance"], dtype=float)
    ax.plot(mu[keep], acc[keep], ls="--", color="#555555", lw=1.4)

    total = np.zeros_like(mu)
    for c in CATEGORIES:
        total += np.asarray(prof[c], dtype=float)

    for label, series in [("All reconstructed", total)] + [
        (c, np.asarray(prof[c], dtype=float)) for c in CATEGORIES
    ]:
        st = STYLE[label]
        ax.plot(
            mu[keep],
            series[keep],
            marker=st["marker"],
            ms=3,
            lw=1.5,
            color=st["color"],
            label=label,
            zorder=st["zorder"],
        )

    ax.set_xlabel("Number of interactions", fontsize=12)
    ax.set_ylabel("Average number of reconstructed vertices", fontsize=12)
    ax.set_xlim(0, mu_max)
    ax.set_ylim(0, y_top)

    # Anchor the two reference-line labels on their curves, rotated to the
    # slope as it actually renders (data aspect changes with the figure size).
    def slope_deg(x0: float, x1: float, y0: float, y1: float) -> float:
        """Screen-space angle in degrees of a segment in data coordinates."""
        bbox = ax.get_window_extent()
        sx = bbox.width / mu_max
        sy = bbox.height / y_top
        return float(np.degrees(np.arctan2((y1 - y0) * sy, (x1 - x0) * sx)))

    ax.text(
        46,
        46 + 1.2,
        "100% interaction reconstruction efficiency",
        rotation=slope_deg(0, mu_max, 0, mu_max),
        fontsize=7.5,
        color="#777777",
        ha="center",
        va="bottom",
        rotation_mode="anchor",
    )
    a_at = float(np.interp(52, mu[keep], acc[keep]))
    ax.text(
        52,
        a_at + 1.0,
        "Reconstruction acceptance",
        rotation=slope_deg(
            40,
            64,
            float(np.interp(40, mu[keep], acc[keep])),
            float(np.interp(64, mu[keep], acc[keep])),
        ),
        fontsize=7.5,
        color="#555555",
        ha="center",
        va="bottom",
        rotation_mode="anchor",
    )

    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.80), frameon=False, fontsize=10)
    atlas_label(ax, r"AMVF, Run 3 MC ($t\bar{t}$), $\sqrt{s}=13$ TeV")
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def dip_metrics(
    centres: np.ndarray, frac: np.ndarray, plateau: float
) -> dict[str, float]:
    """Quantify the merging dip: depth at zero and the half-recovery width.

    Args:
        centres: Bin centres in mm.
        frac: Fraction of pairs per bin.
        plateau: Plateau level (mean over the flat region).

    Returns:
        Dict with the depth at dz=0 and the |dz| where the level recovers to
        half the plateau.
    """
    core = np.abs(centres) < 0.15
    depth = float(frac[core].mean() / plateau) if plateau > 0 else float("nan")

    pos = centres > 0
    x = centres[pos]
    y = frac[pos] / plateau if plateau > 0 else frac[pos]
    above = np.where(y >= 0.5)[0]
    half = float(x[above[0]]) if above.size else float("nan")
    return {"depth_at_zero_frac_of_plateau": depth, "half_recovery_mm": half}


def plot_fig25b(data: dict[str, np.ndarray], out: Path) -> dict[str, float]:
    """Plot the pairwise dz distribution between reconstructed vertices.

    Args:
        data: Arrays loaded from fig25_data.npz.
        out: Output PNG path.

    Returns:
        Dip metrics for the summary table.
    """
    hist = data["dz_hist"].astype(float)
    edges = data["dz_edges"]
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_pairs = float(data["n_pairs_total"])
    frac = hist / n_pairs

    flat = np.abs(centres) > 4.0
    plateau = float(frac[flat].mean())
    metrics = dip_metrics(centres, frac, plateau)

    sigma_beam = float(data["beam_sigma_z"])
    sigma_dz = np.sqrt(2) * sigma_beam

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(centres, frac, marker="s", ms=3.2, lw=1.2, color="#2E75B6")
    ax.axhline(plateau, ls=":", color="#999999", lw=1.2)
    ax.text(
        -7.7,
        plateau * 1.03,
        f"plateau = {plateau:.2e}   "
        rf"($\sqrt{{2}}\,\sigma_{{\rm beam}}$ = {sigma_dz:.0f} mm)",
        fontsize=8.5,
        color="#666666",
        va="bottom",
    )

    ax.set_xlabel(r"$\Delta z$ [mm]", fontsize=12)
    ax.set_ylabel("Fraction of vertex pairs per 0.2 mm", fontsize=12)
    ax.set_xlim(-8, 8)
    ax.set_ylim(0, plateau * 1.35)
    atlas_label(ax, r"AMVF, Run 3 MC ($t\bar{t}$), $\sqrt{s}=13$ TeV")
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.minorticks_on()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    metrics["plateau"] = plateau
    metrics["sigma_dz_mm"] = float(sigma_dz)
    return metrics


def plot_comparison(atlas_dir: Path, ours_a: Path, ours_b: Path, out: Path) -> None:
    """Assemble the ATLAS reference panels beside ours in one figure.

    Args:
        atlas_dir: Directory holding fig_25a.png and fig_25b.png.
        ours_a: Our Figure 25(a) analogue.
        ours_b: Our Figure 25(b) analogue.
        out: Output PNG path.
    """
    pairs = [
        (
            atlas_dir / "fig_25a.png",
            ours_a,
            "ATLAS Fig. 25(a)",
            "Ours (AMVF, Run 3 MC)",
        ),
        (
            atlas_dir / "fig_25b.png",
            ours_b,
            "ATLAS Fig. 25(b)",
            "Ours (AMVF, Run 3 MC)",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 11.5))
    for row, (atlas_png, our_png, atlas_title, our_title) in enumerate(pairs):
        for col, (png, title) in enumerate(
            [(atlas_png, atlas_title), (our_png, our_title)]
        ):
            ax = axes[row][col]
            ax.axis("off")
            if png.exists():
                ax.imshow(mpimg.imread(str(png)))
                ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
            else:
                ax.text(0.5, 0.5, f"missing:\n{png}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--atlas-dir", default=None)
    p.add_argument("--mu-max", type=float, default=80.0)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    d = Path(args.input_dir)
    data = dict(np.load(d / "fig25_data.npz"))
    with open(d / "fig25_summary.json") as fh:
        summary = json.load(fh)

    a_png = d / "fig25a_ours.png"
    b_png = d / "fig25b_ours.png"
    plot_fig25a(data, summary, a_png, args.mu_max)
    metrics = plot_fig25b(data, b_png)

    summary["dz_dip"] = metrics
    with open(d / "fig25_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {a_png}")
    print(f"Wrote {b_png}")
    print("\ndz dip metrics:")
    for k, v in metrics.items():
        print(f"  {k:<32s} {v:.4g}")

    if args.atlas_dir:
        cmp_png = d / "fig25_comparison.png"
        plot_comparison(Path(args.atlas_dir), a_png, b_png, cmp_png)
        print(f"\nWrote {cmp_png}")


if __name__ == "__main__":
    main()
