"""HL-LHC PU200 TTVA plots: retraining learning curve + zero-shot comparison.

Reads learning_curve.json (gnn.evaluation.evaluate_checkpoints) and makes:
1. Learning curve — clean-vertex rate/efficiency and edge purity/efficiency
   vs training epoch, with the μ≈60 zero-shot baseline and the μ≈60
   in-domain reference as horizontal lines.
2. Category comparison — Clean/Merged/Split/Fake bar chart for the μ≈60
   zero-shot model vs the best PU200-retrained checkpoint.

Usage:
    python -m gnn.diagnostics.plot_ttva_pu200 \\
        --curve outputs/<date>_ttva_hllhc_eval/learning_curve.json \\
        -o outputs/<date>_ttva_hllhc_eval/plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.plot_style import (
    ALGO_COLORS,
    CATEGORY_COLORS,
    atlas_label,
    save_figure,
    use_atlas_style,
)

PU200_DESC = "HL-LHC $t\\bar{t}$ MC, $\\langle\\mu\\rangle = 200$, truth vertices"
# μ≈60 in-domain reference: GNN on truth vertices, clean rate (2026-07-12)
MU60_IN_DOMAIN_CLEAN = 0.7674


def _epoch_entries(curve: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """Sorted (epochs, entries) for epoch_<N> labels only."""
    entries = [e for e in curve if e["label"].startswith("epoch_")]
    epochs = np.array([int(e["label"].split("_")[1]) for e in entries])
    order = np.argsort(epochs)
    return epochs[order], [entries[i] for i in order]


def plot_learning_curve(curve: list[dict], out_dir: Path) -> dict:
    """Vertex- and edge-level metrics vs epoch; returns the best entry."""
    epochs, entries = _epoch_entries(curve)
    zeroshot = next((e for e in curve if "zeroshot" in e["label"]), None)
    best = max(entries, key=lambda e: e["rates"]["clean"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    clean = [e["rates"]["clean"] for e in entries]
    fake = [e["rates"]["fake"] for e in entries]
    ax1.plot(epochs, clean, "o-", color=CATEGORY_COLORS["Clean"], label="Clean rate")
    ax1.plot(epochs, fake, "s-", color=CATEGORY_COLORS["Fake"], label="Fake rate")
    if zeroshot:
        ax1.axhline(
            zeroshot["rates"]["clean"],
            color=ALGO_COLORS["zeroshot"],
            ls="--",
            label="Zero-shot $\\mu{\\approx}60$ model (clean)",
        )
    ax1.axhline(
        MU60_IN_DOMAIN_CLEAN,
        color="gray",
        ls=":",
        label="$\\mu{\\approx}60$ in-domain (clean)",
    )
    ax1.set_xlabel("Training epoch")
    ax1.set_ylabel("Fraction of reconstructed vertices")
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize="small", loc="center right")
    atlas_label(ax1, desc=PU200_DESC)

    purity = [e["edge_purity"] for e in entries]
    efficiency = [e["edge_efficiency"] for e in entries]
    ax2.plot(epochs, purity, "o-", color=ALGO_COLORS["retrained"], label="Edge purity")
    ax2.plot(epochs, efficiency, "s-", color="#009E73", label="Edge efficiency")
    if zeroshot:
        ax2.axhline(
            zeroshot["edge_purity"],
            color=ALGO_COLORS["zeroshot"],
            ls="--",
            label="Zero-shot purity",
        )
    ax2.set_xlabel("Training epoch")
    ax2.set_ylabel("Edge-level metric")
    ax2.set_ylim(0.5, 0.9)
    ax2.legend(fontsize="small", loc="lower right")
    atlas_label(ax2, desc=PU200_DESC)

    save_figure(fig, out_dir, "pu200_learning_curve")
    return best


def plot_category_comparison(curve: list[dict], best: dict, out_dir: Path) -> None:
    """Zero-shot vs best retrained checkpoint category rates."""
    zeroshot = next((e for e in curve if "zeroshot" in e["label"]), None)
    if zeroshot is None:
        return
    entries = {
        "Zero-shot $\\mu{\\approx}60$ model": zeroshot,
        f"PU200 retrained ({best['label'].replace('_', ' ')})": best,
    }
    categories = ("Clean", "Merged", "Split", "Fake")
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    for j, (label, entry) in enumerate(entries.items()):
        rates = [entry["rates"][c.lower()] for c in categories]
        n = entry["reco_pvs"]
        errors = [np.sqrt(r * (1 - r) / n) for r in rates]
        x = np.arange(len(categories)) + (j - 0.5) * width
        color = ALGO_COLORS["zeroshot"] if j == 0 else ALGO_COLORS["retrained"]
        ax.bar(x, rates, width, yerr=errors, color=color, label=label, capsize=2)
        for xi, r in zip(x, rates):
            ax.text(xi, r + 0.015, f"{100 * r:.1f}%", ha="center", fontsize="x-small")
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Fraction of reconstructed vertices")
    ax.set_ylim(0, 0.9)
    ax.legend(fontsize="small")
    atlas_label(ax, desc=PU200_DESC)
    save_figure(fig, out_dir, "pu200_zeroshot_vs_retrained")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PU200 TTVA learning-curve plots")
    parser.add_argument("--curve", required=True, type=str)
    parser.add_argument("-o", "--output-dir", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    use_atlas_style()
    with open(args.curve) as f:
        curve = json.load(f)
    out_dir = Path(args.output_dir)
    best = plot_learning_curve(curve, out_dir)
    plot_category_comparison(curve, best, out_dir)
    print(
        f"Best checkpoint: {best['label']} "
        f"(clean {best['rates']['clean']:.4f}, "
        f"clean/truth {best['clean_per_truth']:.4f}, "
        f"edge purity {best['edge_purity']:.4f})"
    )
    print(f"Saved PU200 plots to {out_dir}")


if __name__ == "__main__":
    main()
