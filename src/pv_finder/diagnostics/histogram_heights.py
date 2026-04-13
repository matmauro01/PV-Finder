"""Distribution of e2e model histogram output heights for MC and Run 3 data.

Runs the e2e vertex-finding model on both datasets, then plots the
distribution of all non-zero bin values and peak heights (max bin value
per detected PV) as overlaid MC vs Run 3 histograms.

Usage:
    python -m pv_finder.diagnostics.histogram_heights \
        --model-path model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130.pyt \
        --mc-h5 data/monte_carlo/training_data.h5 \
        --run3-cache data/run3/cache_file3_2000ev_seed42.npz \
        --nevents 200 --output-dir outputs/histogram_heights --device-id -1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pv_finder.data.feature_loading import load_mc_data, load_run3_data
from pv_finder.diagnostics.per_vertex_visualization.inference import (
    load_e2e_model,
    run_e2e_on_events,
)
from pv_finder.utils.peak_finding import pv_locations_updated_res

# Threshold below which bins are considered inactive (softplus background)
_ACTIVE_BIN_THRESHOLD = 1e-4


def _stats_dict(arr: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a 1-D array."""
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "count": int(len(arr)),
    }


def _stats_text(arr: np.ndarray) -> str:
    """Format a multi-line stats annotation string."""
    s = _stats_dict(arr)
    return (
        f"N = {s['count']:,}\nmean = {s['mean']:.4f}\n"
        f"median = {s['median']:.4f}\nstd = {s['std']:.4f}"
    )


def _collect_bin_values(
    histograms: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract active bin values and peak heights from model output.

    Returns (active_bins, peak_heights) -- both 1-D float arrays.
    """
    active_list: list[np.ndarray] = []
    peak_list: list[np.ndarray] = []
    for hist_2d in histograms:
        flat = hist_2d.flatten()  # (12000,)
        active_list.append(flat[flat > _ACTIVE_BIN_THRESHOLD])
        _, heights, _, _ = pv_locations_updated_res(flat)
        if len(heights) > 0:
            peak_list.append(heights)
    active_bins = np.concatenate(active_list) if active_list else np.array([])
    peak_heights = np.concatenate(peak_list) if peak_list else np.array([])
    return active_bins, peak_heights


# ── Plotting ─────────────────────────────────────────────────────────────────


def _plot(
    mc_bins: np.ndarray,
    r3_bins: np.ndarray,
    mc_peaks: np.ndarray,
    r3_peaks: np.ndarray,
    output_dir: str,
    n_events: int,
) -> None:
    """Create and save the 2x2 histogram-heights comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: all non-zero bin values (full range)
    ax = axes[0, 0]
    hi = max(mc_bins.max(), r3_bins.max()) * 1.05
    bins_all = np.linspace(0, hi, 150)
    ax.hist(
        mc_bins,
        bins=bins_all,
        color="steelblue",
        edgecolor="none",
        alpha=0.7,
        label=f"MC ({len(mc_bins):,})",
    )
    ax.hist(
        r3_bins,
        bins=bins_all,
        color="darkorange",
        edgecolor="none",
        alpha=0.7,
        label=f"Run 3 ({len(r3_bins):,})",
    )
    ax.set_xlabel("Bin value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"All active bin values  [{n_events} events each]", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")

    # Top-right: tail zoom (bin value > 1.0)
    ax = axes[0, 1]
    mc_tail = mc_bins[mc_bins > 1.0]
    r3_tail = r3_bins[r3_bins > 1.0]
    if len(mc_tail) > 0 or len(r3_tail) > 0:
        upper = max(
            mc_tail.max() if len(mc_tail) > 0 else 0,
            r3_tail.max() if len(r3_tail) > 0 else 0,
        )
        bins_tail = np.linspace(1.0, upper * 1.05, 100)
        if len(mc_tail) > 0:
            ax.hist(
                mc_tail,
                bins=bins_tail,
                color="steelblue",
                edgecolor="none",
                alpha=0.7,
                label=f"MC ({len(mc_tail):,})",
            )
        if len(r3_tail) > 0:
            ax.hist(
                r3_tail,
                bins=bins_tail,
                color="darkorange",
                edgecolor="none",
                alpha=0.7,
                label=f"Run 3 ({len(r3_tail):,})",
            )
    ax.set_xlabel("Bin value", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("High-value tail (bin value > 1.0)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")
    ax.text(
        0.97,
        0.97,
        f"MC: {_stats_text(mc_tail)}\n\nRun3: {_stats_text(r3_tail)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Bottom-left: peak heights (raw counts)
    ax = axes[1, 0]
    upper_pk = max(
        mc_peaks.max() if len(mc_peaks) > 0 else 0,
        r3_peaks.max() if len(r3_peaks) > 0 else 0,
    )
    bins_pk = np.linspace(0, upper_pk * 1.05, 100)
    ax.hist(
        mc_peaks,
        bins=bins_pk,
        color="steelblue",
        edgecolor="none",
        alpha=0.7,
        label=f"MC ({len(mc_peaks):,} PVs)",
    )
    ax.hist(
        r3_peaks,
        bins=bins_pk,
        color="darkorange",
        edgecolor="none",
        alpha=0.7,
        label=f"Run 3 ({len(r3_peaks):,} PVs)",
    )
    ax.set_xlabel("Peak height (max bin per PV)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Peak heights -- raw counts", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")
    ax.text(
        0.97,
        0.97,
        f"MC: {_stats_text(mc_peaks)}\n\nRun3: {_stats_text(r3_peaks)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # Bottom-right: peak heights (density-normalized)
    ax = axes[1, 1]
    if len(mc_peaks) > 0:
        ax.hist(
            mc_peaks,
            bins=bins_pk,
            density=True,
            color="steelblue",
            edgecolor="none",
            alpha=0.7,
            label="MC",
        )
    if len(r3_peaks) > 0:
        ax.hist(
            r3_peaks,
            bins=bins_pk,
            density=True,
            color="darkorange",
            edgecolor="none",
            alpha=0.7,
            label="Run 3",
        )
    ax.set_xlabel("Peak height (max bin per PV)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Peak heights -- normalized (density)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")

    fig.suptitle(
        "E2E model histogram output heights: MC vs Run 3",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out / f"histogram_heights.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="E2E model histogram height distributions (MC vs Run 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130.pyt",
        help="Path to e2e model weights (.pyt, pickled full model)",
    )
    parser.add_argument(
        "--mc-h5",
        default="data/monte_carlo/training_data.h5",
        help="Path to MC training HDF5 file",
    )
    parser.add_argument(
        "--run3-cache",
        default="data/run3/cache_file3_2000ev_seed42.npz",
        help="Path to Run 3 NPZ cache",
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=200,
        help="Number of events to process per dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/histogram_heights",
        help="Directory for output plots and JSON",
    )
    parser.add_argument(
        "--device-id", type=int, default=-1, help="CUDA device id; -1 for CPU"
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    if args.device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Legacy module alias so torch.load can unpickle the model class
    sys.modules["model.autoencoder_models"] = __import__(
        "pv_finder.models.autoencoder_models"
    )

    print(f"\nLoading e2e model: {args.model_path}")
    model = load_e2e_model(args.model_path, device=str(device))

    print(f"\nLoading MC data: {args.mc_h5}")
    mc_events = load_mc_data(args.mc_h5, n_events=args.nevents)
    print(f"Loading Run 3 data: {args.run3_cache}")
    r3_events = load_run3_data(
        args.run3_cache,
        min_pileup=3,
        max_events=args.nevents,
    )

    print("\nRunning e2e inference on MC events ...")
    mc_hists = run_e2e_on_events(model, mc_events, "mc", device=str(device))
    print(f"  {len(mc_hists)} MC histograms produced")

    print("Running e2e inference on Run 3 events ...")
    r3_hists = run_e2e_on_events(model, r3_events, "run3", device=str(device))
    print(f"  {len(r3_hists)} Run 3 histograms produced")

    print("\nExtracting bin values and peak heights ...")
    mc_bins, mc_peaks = _collect_bin_values(mc_hists)
    r3_bins, r3_peaks = _collect_bin_values(r3_hists)
    print(f"  MC:    {len(mc_bins):,} active bins, {len(mc_peaks):,} peaks")
    print(f"  Run 3: {len(r3_bins):,} active bins, {len(r3_peaks):,} peaks")

    print("\nPlotting ...")
    _plot(mc_bins, r3_bins, mc_peaks, r3_peaks, args.output_dir, args.nevents)

    # JSON summary
    summary = {
        "n_events_mc": len(mc_hists),
        "n_events_run3": len(r3_hists),
        "active_bins": {"mc": _stats_dict(mc_bins), "run3": _stats_dict(r3_bins)},
        "peak_heights": {"mc": _stats_dict(mc_peaks), "run3": _stats_dict(r3_peaks)},
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "histogram_heights_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
