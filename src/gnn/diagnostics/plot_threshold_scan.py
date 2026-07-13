"""Plots for the TTVA threshold scan and edge-level metrics.

Reads scan_results.json (gnn.evaluation.threshold_scan) and the edge-level
curves (gnn.evaluation.edge_metrics) and produces ATLAS-style figures:
ROC, score distributions, vertex rates vs threshold, clean-vertex
efficiency vs threshold, and track-level metrics vs threshold, each with
the AMVF reference and the chosen working point t*.

Usage:
    python -m gnn.diagnostics.plot_threshold_scan \\
        --scan-dir outputs/<date>_ttva_metrics/ \\
        --edge-dir outputs/<date>_ttva_metrics/edge_level/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.plot_style import (
    ALGO_COLORS,
    ALGO_LABELS,
    CATEGORY_COLORS,
    atlas_label,
    save_figure,
    use_atlas_style,
)

MC_DESC = "$t\\bar{t}$ MC, $\\langle\\mu\\rangle \\approx 60$"


def _scan_arrays(points: list[dict]) -> dict[str, np.ndarray]:
    """Columnar arrays from the per-threshold scan points."""
    return {
        "t": np.array([p["threshold"] for p in points]),
        "clean": np.array([p["rates"]["clean"] for p in points]),
        "merged": np.array([p["rates"]["merged"] for p in points]),
        "split": np.array([p["rates"]["split"] for p in points]),
        "fake": np.array([p["rates"]["fake"] for p in points]),
        "clean_per_truth": np.array([p["clean_per_truth"] for p in points]),
        "trk_eff": np.array([p["track"]["efficiency"] for p in points]),
        "trk_pur": np.array([p["track"]["purity"] for p in points]),
        "trk_f1": np.array([p["track"]["f1"] for p in points]),
    }


def plot_roc(edge_dir: Path, out_dir: Path) -> None:
    """ROC curve (log-x) with AUC annotation."""
    curves = np.load(edge_dir / "edge_curves.npz")
    with open(edge_dir / "edge_metrics.json") as f:
        meta = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        curves["fpr"],
        curves["tpr"],
        color=ALGO_COLORS["pvf_gnn"],
        label=f"GNN edge classifier (AUC = {meta['auc']:.4f})",
    )
    ax.plot([1e-6, 1], [1e-6, 1], ls=":", color="gray", label="Random")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    atlas_label(ax, desc=MC_DESC)
    save_figure(fig, out_dir, "edge_roc")


def plot_score_distributions(edge_dir: Path, out_dir: Path, t_star: float) -> None:
    """Edge-score distributions split by truth label (log-y)."""
    curves = np.load(edge_dir / "edge_curves.npz")
    bins = curves["bins"]
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        centers,
        curves["hist_false"],
        width=width,
        color=CATEGORY_COLORS["Fake"],
        alpha=0.6,
        label="False edges",
    )
    ax.bar(
        centers,
        curves["hist_true"],
        width=width,
        color=CATEGORY_COLORS["Clean"],
        alpha=0.6,
        label="True edges",
    )
    ax.axvline(t_star, color="black", ls="--", lw=1.5, label=f"$t^*$ = {t_star}")
    ax.set_yscale("log")
    ax.set_xlabel("GNN edge score")
    ax.set_ylabel("Edges / bin")
    ax.legend(loc="upper center")
    atlas_label(ax, desc=MC_DESC)
    save_figure(fig, out_dir, "edge_score_distributions")


def plot_vertex_rates(scan: dict, out_dir: Path) -> None:
    """Clean/Merged/Split/Fake rates vs threshold with AMVF references."""
    arrays = _scan_arrays(scan["maxscore"])
    amvf = scan["amvf"]["rates"]
    t_star = scan["working_point"]["t_star"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for cat in ("clean", "merged", "split", "fake"):
        color = CATEGORY_COLORS[cat.capitalize()]
        ax.plot(
            arrays["t"], arrays[cat], "o-", ms=4, color=color, label=cat.capitalize()
        )
        ax.axhline(amvf[cat], color=color, ls=":", lw=1.2)
    ax.axvline(t_star, color="black", ls="--", lw=1.2, label=f"$t^*$ = {t_star}")
    ax.plot([], [], ls=":", color="gray", label="AMVF (dotted)")
    ax.set_xlabel("MaxScore threshold")
    ax.set_ylabel("Fraction of reconstructed vertices")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", fontsize="small")
    atlas_label(ax, desc=MC_DESC)
    save_figure(fig, out_dir, "vertex_rates_vs_threshold")


def plot_clean_efficiency(scan: dict, out_dir: Path, fake_budget: float) -> None:
    """Clean-vertex efficiency and fake rate vs threshold."""
    arrays = _scan_arrays(scan["maxscore"])
    amvf = scan["amvf"]
    t_star = scan["working_point"]["t_star"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        arrays["t"],
        arrays["clean_per_truth"],
        "o-",
        ms=4,
        color=ALGO_COLORS["pvf_gnn"],
        label=ALGO_LABELS["pvf_gnn"],
    )
    ax.axhline(
        amvf["clean_per_truth"],
        color=ALGO_COLORS["amvf"],
        ls=":",
        label=ALGO_LABELS["amvf"],
    )
    ax.axvline(t_star, color="black", ls="--", lw=1.2, label=f"$t^*$ = {t_star}")
    ax.set_xlabel("MaxScore threshold")
    ax.set_ylabel("Clean vertices / truth PVs")
    ax.set_ylim(0.5, 0.85)
    ax.legend(loc="center left", fontsize="small")

    ax2 = ax.twinx()
    ax2.plot(
        arrays["t"],
        arrays["fake"],
        "s--",
        ms=4,
        color=CATEGORY_COLORS["Fake"],
        label="Fake rate (right)",
    )
    ax2.axhline(fake_budget, color=CATEGORY_COLORS["Fake"], ls=":", lw=1.0)
    ax2.set_ylabel("Fake-vertex rate", color=CATEGORY_COLORS["Fake"])
    ax2.set_ylim(0, 0.05)
    ax2.legend(loc="lower left", fontsize="small")
    atlas_label(ax, desc=MC_DESC)
    save_figure(fig, out_dir, "clean_efficiency_vs_threshold")


def plot_track_metrics(scan: dict, out_dir: Path) -> None:
    """Track-level assignment efficiency/purity/F1 vs threshold."""
    arrays = _scan_arrays(scan["maxscore"])
    amvf_track = scan["amvf"]["track"]
    t_star = scan["working_point"]["t_star"]

    fig, ax = plt.subplots(figsize=(8, 6))
    series = (
        ("trk_eff", "efficiency", "Efficiency", "o-", ALGO_COLORS["pvf_gnn"]),
        ("trk_pur", "purity", "Purity", "s-", "#CC79A7"),
        ("trk_f1", "f1", "F1", "^-", "#009E73"),
    )
    for key, amvf_key, label, fmt, color in series:
        ax.plot(arrays["t"], arrays[key], fmt, ms=4, color=color, label=f"GNN {label}")
        ax.axhline(amvf_track[amvf_key], color=color, ls=":", lw=1.2)
    ax.axvline(t_star, color="black", ls="--", lw=1.2, label=f"$t^*$ = {t_star}")
    ax.plot([], [], ls=":", color="gray", label="AMVF (dotted)")
    ax.set_xlabel("MaxScore threshold")
    ax.set_ylabel("Track-assignment metric")
    ax.legend(loc="lower left", fontsize="small")
    atlas_label(ax, desc=MC_DESC)
    save_figure(fig, out_dir, "track_metrics_vs_threshold")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot TTVA threshold-scan results")
    parser.add_argument("--scan-dir", required=True, type=str)
    parser.add_argument("--edge-dir", required=True, type=str)
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--fake-budget", default=0.01, type=float)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    scan_dir = Path(args.scan_dir)
    edge_dir = Path(args.edge_dir)
    out_dir = Path(args.output_dir) if args.output_dir else scan_dir / "plots"

    use_atlas_style()
    with open(scan_dir / "scan_results.json") as f:
        scan = json.load(f)
    t_star = scan["working_point"]["t_star"]

    plot_roc(edge_dir, out_dir)
    plot_score_distributions(edge_dir, out_dir, t_star)
    plot_vertex_rates(scan, out_dir)
    plot_clean_efficiency(scan, out_dir, args.fake_budget)
    plot_track_metrics(scan, out_dir)
    print(f"Saved threshold-scan plots to {out_dir}")


if __name__ == "__main__":
    main()
