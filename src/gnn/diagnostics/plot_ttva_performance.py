"""Publication-quality MC performance plots for the TTVA GNN at t*.

Compares three chains on the same 2,550 MC test events with the identical
classification core:
  - PV-Finder peaks + GNN association (at the tuned working point t*)
  - AMVF (its own finding + association)
  - Ground-truth vertices + GNN association (upper bound)

Figures: (a) Clean/Merged/Split/Fake grouped bars with binomial errors,
(b) clean-vertex efficiency summary, (c) clean-vertex efficiency vs
truth-PV track multiplicity (PVF+GNN vs AMVF), (d) edge-score
distributions (from gnn.evaluation.edge_metrics output).

Usage:
    python -m gnn.diagnostics.plot_ttva_performance \\
        --pvf-results outputs/<date>_ttva_metrics/t_star \\
        --amvf-results outputs/07_12_2026_ttva_reproduction/amvf \\
        --truth-results outputs/<date>_ttva_publication/gnn_on_truth \\
        --edge-dir outputs/<date>_ttva_metrics/edge_level \\
        -f /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \\
        -i configs/qibin_test_main_indices_v2.p \\
        --t-star 0.98 -o outputs/<date>_ttva_publication/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from gnn.data.graph_construction import load_event_indices
from gnn.diagnostics.plot_style import (
    ALGO_COLORS,
    ALGO_LABELS,
    atlas_label,
    save_figure,
    use_atlas_style,
)
from gnn.diagnostics.plot_threshold_scan import MC_DESC, plot_score_distributions

CATEGORIES = ("Clean", "Merged", "Split", "Fake")
NTRK_BINS = np.array([2, 3, 4, 6, 10, 20, 50, 1000])


def load_results(results_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (rows, info) .npy pair from an eval output directory."""
    results_dir = Path(results_dir)
    rows_path = next(results_dir.glob("total_results_*.npy"))
    info_path = next(results_dir.glob("total_reco_pv_info_list_*.npy"))
    rows = np.array(np.load(rows_path, allow_pickle=True).tolist(), dtype=np.int64)
    info = np.load(info_path, allow_pickle=True)
    return rows, info


def plot_category_rates(
    all_rows: dict[str, np.ndarray], out_dir: Path, t_star: float
) -> None:
    """Grouped Clean/Merged/Split/Fake bars for the three chains."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    width = 0.26
    for j, (algo, rows) in enumerate(all_rows.items()):
        totals = rows.sum(axis=0)
        n_reco = totals[4]
        rates = totals[:4] / n_reco
        errors = np.sqrt(rates * (1 - rates) / n_reco)
        x = np.arange(len(CATEGORIES)) + (j - 1) * width
        label = ALGO_LABELS[algo]
        if algo == "pvf_gnn":
            label += f" ($t^*$ = {t_star})"
        ax.bar(
            x,
            rates,
            width,
            yerr=errors,
            color=ALGO_COLORS[algo],
            label=label,
            capsize=2,
        )
        for xi, r in zip(x, rates):
            ax.text(xi, r + 0.012, f"{100 * r:.1f}", ha="center", fontsize="x-small")
    ax.set_xticks(np.arange(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("Fraction of reconstructed vertices")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize="small")
    atlas_label(ax, desc=MC_DESC, desc_xy=(0.30, 0.86))
    save_figure(fig, out_dir, "category_rates")


def plot_clean_efficiency(
    all_rows: dict[str, np.ndarray], out_dir: Path, t_star: float
) -> None:
    """Clean vertices / truth PVs summary bar for the three chains."""
    fig, ax = plt.subplots(figsize=(8, 6))
    labels, values, errors, colors = [], [], [], []
    for algo, rows in all_rows.items():
        totals = rows.sum(axis=0)
        n_truth = totals[5]
        eff = totals[0] / n_truth
        labels.append(ALGO_LABELS[algo].replace(" + ", "\n+ "))
        values.append(eff)
        errors.append(np.sqrt(eff * (1 - eff) / n_truth))
        colors.append(ALGO_COLORS[algo])
    x = np.arange(len(labels))
    ax.bar(x, values, 0.55, yerr=errors, color=colors, capsize=3)
    for xi, v in zip(x, values):
        ax.text(xi, v + 0.015, f"{100 * v:.1f}%", ha="center", fontsize="small")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize="small")
    ax.set_ylabel("Clean vertices / truth PVs")
    ax.set_ylim(0, 1.0)
    atlas_label(ax, desc=f"{MC_DESC}, $t^*$ = {t_star}")
    save_figure(fig, out_dir, "clean_vertex_efficiency")


def per_truth_outcomes(
    info: np.ndarray,
    h5_path: str,
    event_keys: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-truth-PV (nTrk >= 2) outcome across all events.

    Returns (ntracks, found_clean, found_any) flat arrays over truth PVs.
    """
    ntrk_out: list[np.ndarray] = []
    clean_out: list[np.ndarray] = []
    any_out: list[np.ndarray] = []
    with h5py.File(h5_path, "r") as f:
        pv_ntracks = f["pv_ntracks"]
        for event_key, event_info in zip(event_keys, info):
            ntrk = pv_ntracks[event_key][:].astype(np.int64)
            keep = ntrk >= 2
            found_clean = np.zeros(len(ntrk), dtype=bool)
            found_any = np.zeros(len(ntrk), dtype=bool)
            for pv_info in event_info:
                primary = pv_info["primary_truth_pv"]
                if primary is None or primary == "Fake":
                    continue
                idx = int(primary.rsplit("_", 1)[1])
                found_any[idx] = True
                if pv_info["classification"] == "Clean":
                    found_clean[idx] = True
            ntrk_out.append(ntrk[keep])
            clean_out.append(found_clean[keep])
            any_out.append(found_any[keep])
    return (
        np.concatenate(ntrk_out),
        np.concatenate(clean_out),
        np.concatenate(any_out),
    )


def plot_efficiency_vs_ntracks(
    outcomes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_dir: Path,
    t_star: float,
) -> None:
    """Clean-vertex efficiency vs truth-PV track multiplicity."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    centers = 0.5 * (NTRK_BINS[:-1] + np.minimum(NTRK_BINS[1:], 100))
    for algo, (ntrk, found_clean, _found_any) in outcomes.items():
        effs, errs = [], []
        for lo, hi in zip(NTRK_BINS[:-1], NTRK_BINS[1:]):
            sel = (ntrk >= lo) & (ntrk < hi)
            n = int(sel.sum())
            eff = found_clean[sel].mean() if n else np.nan
            effs.append(eff)
            errs.append(np.sqrt(eff * (1 - eff) / n) if n else np.nan)
        label = ALGO_LABELS[algo]
        if algo == "pvf_gnn":
            label += f" ($t^*$ = {t_star})"
        ax.errorbar(
            centers,
            effs,
            yerr=errs,
            fmt="o-",
            ms=5,
            capsize=2,
            color=ALGO_COLORS[algo],
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Truth PV track multiplicity")
    ax.set_ylabel("Clean-vertex efficiency")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize="small", loc="lower right")
    atlas_label(ax, desc=MC_DESC)
    save_figure(fig, out_dir, "clean_efficiency_vs_ntracks")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Publication TTVA performance plots")
    parser.add_argument("--pvf-results", required=True, type=str)
    parser.add_argument("--amvf-results", required=True, type=str)
    parser.add_argument("--truth-results", required=True, type=str)
    parser.add_argument("--edge-dir", required=True, type=str)
    parser.add_argument("-f", "--filepath", required=True, type=str)
    parser.add_argument("-i", "--indices", required=True, type=str)
    parser.add_argument("--t-star", required=True, type=float)
    parser.add_argument("-o", "--output-dir", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    use_atlas_style()
    out_dir = Path(args.output_dir)

    rows_pvf, info_pvf = load_results(args.pvf_results)
    rows_amvf, info_amvf = load_results(args.amvf_results)
    rows_truth, _info_truth = load_results(args.truth_results)
    all_rows = {"pvf_gnn": rows_pvf, "amvf": rows_amvf, "gnn_truth": rows_truth}

    plot_category_rates(all_rows, out_dir, args.t_star)
    plot_clean_efficiency(all_rows, out_dir, args.t_star)

    indices = load_event_indices(args.indices)
    event_keys = [f"Event{int(idx)}" for idx in indices][: len(info_pvf)]
    outcomes = {
        "pvf_gnn": per_truth_outcomes(info_pvf, args.filepath, event_keys),
        "amvf": per_truth_outcomes(info_amvf, args.filepath, event_keys),
    }
    plot_efficiency_vs_ntracks(outcomes, out_dir, args.t_star)

    plot_score_distributions(Path(args.edge_dir), out_dir, args.t_star)

    # Numbers table for docs — exact sums from the same .npy the bars use
    table = {}
    for algo, rows in all_rows.items():
        totals = rows.sum(axis=0)
        table[algo] = {
            "clean": int(totals[0]),
            "merged": int(totals[1]),
            "split": int(totals[2]),
            "fake": int(totals[3]),
            "n_reco": int(totals[4]),
            "n_truth": int(totals[5]),
            "clean_rate": float(totals[0] / totals[4]),
            "clean_per_truth": float(totals[0] / totals[5]),
        }
    with open(out_dir / "publication_numbers.json", "w") as f:
        json.dump({"t_star": args.t_star, "chains": table}, f, indent=2)
    print(json.dumps(table, indent=2))
    print(f"Saved publication plots to {out_dir}")


if __name__ == "__main__":
    main()
