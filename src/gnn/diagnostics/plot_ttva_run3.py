"""Plots for the truth-free Run 3 TTVA evaluation.

Reads run3_summary.json + run3_hists.npz (gnn.evaluation.evaluate_ttva_run3)
and produces: GNN score distributions on Run 3 data overlaid with MC
(domain shift), per-vertex track-multiplicity comparison GNN vs AMVF, and
the GNN-AMVF track-assignment agreement vs matching window with the
disagreement breakdown.

Usage:
    python -m gnn.diagnostics.plot_ttva_run3 \\
        --results-dir outputs/<date>_ttva_run3/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.plot_style import (
    ALGO_COLORS,
    atlas_label,
    save_figure,
    use_atlas_style,
)

RUN3_DESC = "Run 3 data"
RUN3_STATUS = "Internal"


def _step(ax: plt.Axes, bins: np.ndarray, hist: np.ndarray, **kwargs) -> None:
    """Draw a density-normalised histogram outline."""
    density = hist / max(hist.sum(), 1) / np.diff(bins)
    ax.stairs(density, bins, **kwargs)


def plot_score_overlay(hists: dict, out_dir: Path, t_star: float) -> None:
    """Run 3 vs MC GNN score distributions (all edges + max per track)."""
    bins = hists["score_bins"]
    panels = (
        ("gnn_all_scores", "mc_all_scores", "GNN edge score (all edges)"),
        ("gnn_max_scores", "mc_max_scores", "Max GNN score per track"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (run3_key, mc_key, xlabel) in zip(axes, panels):
        _step(
            ax,
            bins,
            hists[run3_key],
            color=ALGO_COLORS["pvf_gnn"],
            lw=2,
            label="Run 3 data",
        )
        if mc_key in hists:
            _step(
                ax,
                bins,
                hists[mc_key],
                color="gray",
                lw=2,
                ls="--",
                label="$t\\bar{t}$ MC, $\\langle\\mu\\rangle \\approx 60$",
            )
        ax.axvline(t_star, color="black", ls=":", lw=1.2, label=f"$t^*$ = {t_star}")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalised density")
        ax.legend(fontsize="small", loc="upper center")
        atlas_label(ax, status=RUN3_STATUS, desc=RUN3_DESC)
    save_figure(fig, out_dir, "run3_score_overlay")


def plot_multiplicity(hists: dict, out_dir: Path) -> None:
    """Per-vertex assigned-track multiplicity, GNN vs AMVF."""
    bins = hists["mult_bins"]
    fig, ax = plt.subplots(figsize=(8, 6))
    _step(
        ax,
        bins,
        hists["gnn_multiplicity"],
        color=ALGO_COLORS["pvf_gnn"],
        lw=2,
        label="PV-Finder + GNN",
    )
    _step(
        ax,
        bins,
        hists["amvf_multiplicity"],
        color=ALGO_COLORS["amvf"],
        lw=2,
        ls="--",
        label="AMVF",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Tracks assigned per vertex")
    ax.set_ylabel("Normalised density")
    ax.legend(fontsize="small")
    atlas_label(ax, status=RUN3_STATUS, desc=RUN3_DESC)
    save_figure(fig, out_dir, "run3_multiplicity")


def plot_agreement(summary: dict, out_dir: Path) -> None:
    """Agreement vs matching window + disagreement breakdown."""
    agreement = summary["agreement_by_window"]
    windows = sorted(agreement.keys(), key=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x = np.arange(len(windows))
    vs_amvf = [agreement[w]["agreement_vs_amvf_assigned"] for w in windows]
    vs_both = [agreement[w]["agreement_vs_both_assigned"] for w in windows]
    ax1.bar(x - 0.18, vs_amvf, 0.36, color=ALGO_COLORS["pvf_gnn"],
            label="vs all AMVF-assigned tracks")  # fmt: skip
    ax1.bar(x + 0.18, vs_both, 0.36, color="#56B4E9",
            label="vs tracks assigned by both")  # fmt: skip
    for xi, (a, b) in enumerate(zip(vs_amvf, vs_both)):
        ax1.text(
            xi - 0.18, a + 0.01, f"{100 * a:.1f}%", ha="center", fontsize="x-small"
        )
        ax1.text(
            xi + 0.18, b + 0.01, f"{100 * b:.1f}%", ha="center", fontsize="x-small"
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$|\\Delta z| <$ {w} mm" for w in windows])
    ax1.set_ylabel("Track-assignment agreement with AMVF")
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize="small", loc="lower right")
    atlas_label(ax1, status=RUN3_STATUS, desc=RUN3_DESC)

    main_w = str(summary["config"]["match_window"])
    counts = agreement[main_w]
    n_ref = counts["n_amvf_assigned"]
    parts = {
        "Agree": counts["n_agree"],
        "GNN unassigned": counts["n_gnn_unassigned"],
        "GNN vertex unmatched": counts["n_gnn_vertex_unmatched"],
        "Different vertex": counts["n_different_vertex"],
    }
    colors = ["#009E73", "#E69F00", "#56B4E9", "#D55E00"]
    bottom = 0.0
    for (label, val), color in zip(parts.items(), colors):
        frac = val / n_ref
        ax2.bar([0], [frac], 0.4, bottom=bottom, color=color, label=label)
        if frac > 0.02:
            ax2.text(0, bottom + frac / 2, f"{100 * frac:.1f}%", ha="center",
                     va="center", fontsize="small")  # fmt: skip
        bottom += frac
    ax2.set_xlim(-0.8, 0.8)
    ax2.set_xticks([0])
    ax2.set_xticklabels([f"AMVF-assigned tracks ($|\\Delta z| <$ {main_w} mm)"])
    ax2.set_ylabel("Fraction")
    ax2.set_ylim(0, 1.25)
    ax2.legend(fontsize="small", loc="upper right")
    atlas_label(ax2, status=RUN3_STATUS, desc=RUN3_DESC)
    save_figure(fig, out_dir, "run3_agreement")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot Run 3 TTVA eval results")
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"

    use_atlas_style()
    with open(results_dir / "run3_summary.json") as f:
        summary = json.load(f)
    hists = dict(np.load(results_dir / "run3_hists.npz"))

    plot_score_overlay(hists, out_dir, summary["config"]["threshold"])
    plot_multiplicity(hists, out_dir)
    plot_agreement(summary, out_dir)
    print(f"Saved Run 3 plots to {out_dir}")


if __name__ == "__main__":
    main()
