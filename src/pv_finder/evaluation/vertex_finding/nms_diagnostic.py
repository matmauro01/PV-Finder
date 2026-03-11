#!/usr/bin/env python3
"""NMS diagnostic -- analyze which peaks are removed by Non-Maximum Suppression.

Re-runs E2E model inference on Run 3 events, applies Gaussian pre-smoothing
and NMS, identifies exactly which peaks were removed, generates statistics,
and creates per-vertex zoom visualizations of removed peaks.

Outputs:
  - removal_stats.png   : 4-panel summary figure
  - zoom_plots/         : per-vertex zoom visualizations of removed peaks
  - removed_peaks_summary.pkl : serialized list of RemovedPeak records
"""

from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from pv_finder.data.run3_io import load_run3_from_root  # noqa: E402
from pv_finder.diagnostics.domain_shift_investigation.kde_study.analytical_kde import (  # noqa: E402
    compute_analytical_kde_event_run3,
)
from pv_finder.diagnostics.per_vertex_visualization.vertex_plots import (  # noqa: E402
    plot_vertex_zoom,
)
from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from efficiency_res_optimized_atlas import (  # noqa: E402
    pv_locations_updated_res,
    suppress_neighbor_peaks,
)
from run_eval_pvf_run3 import (  # noqa: E402
    E2E_CONFIG,
    INTEGRAL_THRESHOLD,
    MIN_WIDTH,
    N_BINS_FULL,
    THRESHOLD,
    build_subevent_inputs,
    load_ckpt,
    run_e2e_inference,
)


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------
class RemovedPeak(NamedTuple):
    event_idx: int
    z_mm: float
    height: float
    survivor_z_mm: float
    survivor_height: float
    distance_mm: float
    height_ratio: float
    is_truth_matched: bool
    truth_distance_mm: float
    in_bump_region: bool


# ---------------------------------------------------------------------------
# Phase 1: Inference + NMS diagnostic
# ---------------------------------------------------------------------------
def run_diagnostic(
    events: list,
    model: torch.nn.Module,
    device: torch.device,
    smooth_sigma: float,
    nms_min_sep: float,
    nms_max_ratio: float,
    match_window: float,
    no_correct_beam: bool = False,
) -> tuple[list[RemovedPeak], list[dict]]:
    """Run inference on events and collect NMS removal diagnostics."""
    all_removed: list[RemovedPeak] = []
    event_data_list: list[dict] = []
    n_events = len(events)

    for i, event in enumerate(events):
        subevents = build_subevent_inputs(event)
        ph = run_e2e_inference(subevents, model, device)

        ph_smooth = (
            gaussian_filter1d(ph, sigma=smooth_sigma) if smooth_sigma > 0 else ph.copy()
        )
        positions, heights, *_ = pv_locations_updated_res(
            ph_smooth, THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH
        )
        keep = suppress_neighbor_peaks(positions, heights, nms_min_sep, nms_max_ratio)

        # Beam-corrected AMVF truth vertices
        t_pvs = event.amvf_z.copy()
        if not no_correct_beam:
            t_pvs = t_pvs - event.beam_z

        event_data_list.append(
            {
                "event": event,
                "ph": ph,
                "positions": positions,
                "heights": heights,
                "keep": keep,
                "t_pvs": t_pvs,
                "all_peaks": list(zip(positions.tolist(), heights.tolist())),
            }
        )

        # Analyze removed peaks
        removed_mask = ~keep
        surviving_pos = positions[keep]
        surviving_hts = heights[keep]

        for j in np.where(removed_mask)[0]:
            rz = float(positions[j])
            rh = float(heights[j])

            # Find nearest surviving peak that is taller
            taller = surviving_hts > rh
            if np.any(taller):
                dists = np.abs(surviving_pos[taller] - rz)
                best = np.argmin(dists)
                sv_z = float(surviving_pos[taller][best])
                sv_h = float(surviving_hts[taller][best])
            elif len(surviving_pos) > 0:
                dists = np.abs(surviving_pos - rz)
                best = np.argmin(dists)
                sv_z = float(surviving_pos[best])
                sv_h = float(surviving_hts[best])
            else:
                sv_z, sv_h = rz, rh

            dist_mm = abs(rz - sv_z)
            ratio = rh / sv_h if sv_h > 0 else 0.0

            # Truth matching
            if len(t_pvs) > 0:
                truth_dists = np.abs(t_pvs - rz)
                nearest_truth = float(np.min(truth_dists))
                is_matched = nearest_truth <= match_window
            else:
                nearest_truth = float("inf")
                is_matched = False

            in_bump = 0.55 <= dist_mm <= 0.80

            all_removed.append(
                RemovedPeak(
                    event_idx=event.event_idx,
                    z_mm=rz,
                    height=rh,
                    survivor_z_mm=sv_z,
                    survivor_height=sv_h,
                    distance_mm=dist_mm,
                    height_ratio=ratio,
                    is_truth_matched=is_matched,
                    truth_distance_mm=nearest_truth,
                    in_bump_region=in_bump,
                )
            )

        if i < 5 or i % 50 == 0:
            n_rem = int(np.sum(~keep))
            print(
                f"  evt {i:3d}/{n_events}: peaks={len(positions)} "
                f"kept={int(np.sum(keep))} removed={n_rem}"
            )

    return all_removed, event_data_list


# ---------------------------------------------------------------------------
# Phase 2: Statistics
# ---------------------------------------------------------------------------
def print_and_save_statistics(
    removed: list[RemovedPeak],
    output_dir: Path,
) -> None:
    """Print summary statistics and generate a 4-panel figure."""
    n_total = len(removed)
    if n_total == 0:
        print("  No peaks were removed by NMS.")
        return

    # Collect per-event counts
    evt_counts: dict[int, int] = {}
    for rp in removed:
        evt_counts[rp.event_idx] = evt_counts.get(rp.event_idx, 0) + 1
    counts_per_evt = list(evt_counts.values())

    distances = np.array([rp.distance_mm for rp in removed])
    ratios = np.array([rp.height_ratio for rp in removed])
    bump_mask = np.array([rp.in_bump_region for rp in removed])
    matched_mask = np.array([rp.is_truth_matched for rp in removed])

    n_bump = int(np.sum(bump_mask))
    n_outside = n_total - n_bump
    bump_real = int(np.sum(bump_mask & matched_mask))
    bump_fake = n_bump - bump_real
    out_real = int(np.sum(~bump_mask & matched_mask))
    out_fake = n_outside - out_real
    total_real = int(np.sum(matched_mask))
    total_fake = n_total - total_real

    # Print text summary
    print(f"\n{'=' * 60}")
    print("  NMS Removal Summary")
    print(f"{'=' * 60}")
    print(f"  Total removed peaks:  {n_total}")
    print(f"  Avg per event:        {np.mean(counts_per_evt):.2f}")
    print(f"  Overall real/fake:    {total_real} real "
          f"({100*total_real/n_total:.1f}%) / "
          f"{total_fake} fake ({100*total_fake/n_total:.1f}%)")  # fmt: skip
    print("\n  Bump region (0.55-0.80 mm):")
    print(f"    Count: {n_bump} ({100 * n_bump / n_total:.1f}% of total)")
    print(f"    Real:  {bump_real}  |  Fake: {bump_fake}")
    print("\n  Outside bump:")
    print(f"    Count: {n_outside} ({100 * n_outside / n_total:.1f}% of total)")
    print(f"    Real:  {out_real}  |  Fake: {out_fake}")
    print(f"{'=' * 60}")

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("NMS Removal Diagnostics", fontsize=14, fontweight="bold")

    # Top-left: removals per event
    ax = axes[0, 0]
    ax.hist(
        counts_per_evt,
        bins=range(0, max(counts_per_evt) + 2),
        edgecolor="k",
        alpha=0.7,
        color="#4c72b0",
    )
    ax.set_xlabel("Removals per event")
    ax.set_ylabel("Number of events")
    ax.set_title("NMS removals per event")

    # Top-right: distance to survivor
    ax = axes[0, 1]
    ax.hist(distances, bins=40, edgecolor="k", alpha=0.7, color="#4c72b0")
    ax.axvspan(0.55, 0.80, alpha=0.25, color="orange", label="Bump 0.55-0.80 mm")
    ax.set_xlabel("Distance to survivor (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Distance to suppressing neighbor")
    ax.legend(fontsize=9)

    # Bottom-left: height ratio
    ax = axes[1, 0]
    ax.hist(ratios, bins=40, edgecolor="k", alpha=0.7, color="#4c72b0")
    ax.set_xlabel("Height ratio (removed / survivor)")
    ax.set_ylabel("Count")
    ax.set_title("Height ratio distribution")

    # Bottom-right: real vs fake, bump vs non-bump
    ax = axes[1, 1]
    x_pos = np.array([0, 1, 2, 3])
    bar_vals = [bump_real, bump_fake, out_real, out_fake]
    colors = ["#2ca02c", "#d62728", "#2ca02c", "#d62728"]
    labels = ["Bump\nreal", "Bump\nfake", "Outside\nreal", "Outside\nfake"]
    ax.bar(x_pos, bar_vals, color=colors, edgecolor="k", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("Real vs fake removed peaks")
    for xi, v in zip(x_pos, bar_vals):
        if v > 0:
            ax.text(xi, v + 0.5, str(v), ha="center", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "removal_stats.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'removal_stats.png'}")


# ---------------------------------------------------------------------------
# Phase 3: Zoom visualizations
# ---------------------------------------------------------------------------
def generate_zoom_plots(
    removed: list[RemovedPeak],
    event_data: list[dict],
    output_dir: Path,
    match_window: float,
    n_viz: int = 40,
) -> None:
    """Generate per-vertex zoom plots for a sample of removed peaks."""
    if not removed:
        print("  No removed peaks to visualize.")
        return

    # Build lookup from event_idx to event_data entry
    data_by_idx: dict[int, dict] = {}
    for ed in event_data:
        data_by_idx[ed["event"].event_idx] = ed

    # Split into bump / non-bump
    bump = [rp for rp in removed if rp.in_bump_region]
    outside = [rp for rp in removed if not rp.in_bump_region]

    n_bump_viz = min(25, len(bump))
    n_out_viz = min(n_viz - n_bump_viz, len(outside))

    sample = random.sample(bump, n_bump_viz) + random.sample(outside, n_out_viz)
    zoom_dir = output_dir / "zoom_plots"
    zoom_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute analytical KDE only for events that appear in the sample
    needed_idxs = {rp.event_idx for rp in sample}
    ana_cache: dict[int, np.ndarray] = {}
    print(f"\n  Computing analytical KDE for {len(needed_idxs)} events ...")
    for eidx in needed_idxs:
        ed = data_by_idx.get(eidx)
        if ed is None:
            continue
        event = ed["event"]
        ana_cache[eidx] = compute_analytical_kde_event_run3(event._asdict())

    print(f"  Generating {len(sample)} zoom plots ...")
    for vi, rp in enumerate(sample):
        ed = data_by_idx.get(rp.event_idx)
        if ed is None:
            continue
        event = ed["event"]
        region = "bump" if rp.in_bump_region else "other"
        label = (
            f"NMS-removed ({region}, d={rp.distance_mm:.2f}mm, r={rp.height_ratio:.2f})"
        )
        hist_ana = ana_cache.get(rp.event_idx, np.zeros(N_BINS_FULL, dtype=np.float32))
        plot_vertex_zoom(
            hist_e2e=ed["ph"],
            hist_analytical=hist_ana,
            truth_z=rp.z_mm,
            pred_peaks=ed["all_peaks"],
            event_idx=rp.event_idx,
            vtx_idx=vi,
            dataset_label="run3",
            output_dir=str(zoom_dir),
            window_mm=4.0,
            tracks_z0=event.z0,
            tracks_d0=event.d0,
            tracks_d0_err=event.d0_err,
            all_truth_vertices=list(ed["t_pvs"]),
            vertex_label=label,
            match_window_mm=match_window,
        )

    print(f"  Saved {len(sample)} plots to {zoom_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    """Run the full NMS diagnostic pipeline."""
    print("=" * 60)
    print("  NMS Diagnostic")
    print("=" * 60)

    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print(f"\nDevice: GPU {args.device} -- "
              f"{torch.cuda.get_device_name(args.device)}")  # fmt: skip
    else:
        device = torch.device("cpu")
        print("\nDevice: CPU")

    print("\n--- Loading Model ---")
    model = trackstoHists_UNet_1000(**E2E_CONFIG)
    load_ckpt(args.e2e_model, model, device)

    print("\n--- Loading Data ---")
    events = load_run3_from_root(
        args.root,
        max_events=args.max_events,
        entry_start=args.entry_start,
        entry_stop=args.entry_stop,
    )
    if not events:
        print("ERROR: no events loaded.")
        sys.exit(1)
    print(f"  Loaded {len(events)} events")
    print(f"  NMS params: min_sep={args.nms_min_sep} mm, "
          f"max_ratio={args.nms_max_ratio}")  # fmt: skip
    print(f"  Smooth sigma: {args.smooth_sigma} bins")
    print(f"  Match window: {args.match_window} mm")

    print(f"\n--- Inference ({len(events)} events) ---")
    removed, event_data = run_diagnostic(
        events,
        model,
        device,
        smooth_sigma=args.smooth_sigma,
        nms_min_sep=args.nms_min_sep,
        nms_max_ratio=args.nms_max_ratio,
        match_window=args.match_window,
    )
    print(f"\n  Total removed peaks: {len(removed)}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n--- Statistics ---")
    print_and_save_statistics(removed, outdir)

    print("\n--- Zoom Plots ---")
    generate_zoom_plots(removed, event_data, outdir, args.match_window, args.n_viz)

    pkl_path = outdir / "removed_peaks_summary.pkl"
    with open(pkl_path, "wb") as fp:
        pickle.dump(removed, fp)
    print(f"\n  Saved: {pkl_path}")
    print(f"\n=== Done ===  (output: {args.output_dir})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NMS diagnostic for PV-Finder Run 3 evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="ROOT file path")
    parser.add_argument(
        "--e2e-model", required=True, dest="e2e_model", help="E2E model checkpoint"
    )
    parser.add_argument(
        "--entry-start",
        type=int,
        default=0,
        dest="entry_start",
        help="First TTree entry to read",
    )
    parser.add_argument(
        "--entry-stop",
        type=int,
        default=None,
        dest="entry_stop",
        help="One-past-last TTree entry",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        dest="max_events",
        help="Max events to process (0=all)",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        dest="smooth_sigma",
        help="Gaussian sigma (bins) for pre-smoothing (0=off)",
    )
    parser.add_argument(
        "--nms-min-sep",
        type=float,
        default=0.85,
        dest="nms_min_sep",
        help="NMS min separation (mm)",
    )
    parser.add_argument(
        "--nms-max-ratio",
        type=float,
        default=0.5,
        dest="nms_max_ratio",
        help="NMS max height ratio",
    )
    parser.add_argument(
        "--match-window",
        type=float,
        default=0.415,
        dest="match_window",
        help="Truth matching window (mm)",
    )
    parser.add_argument(
        "--n-viz",
        type=int,
        default=40,
        dest="n_viz",
        help="Number of zoom plots to generate",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/nms_diagnostic",
        dest="output_dir",
        help="Output directory",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1=CPU)")
    main(parser.parse_args())
