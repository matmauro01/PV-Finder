#!/usr/bin/env python3
"""
Run3 AMVF Evaluation Script

Evaluates a PV-Finder model on Run3 ATLAS data, using AMVF reconstructed vertices
as the reference ("ground truth"). Generates resolution plots and category bar charts.

Reproduces the exact working version that generated run3_results_highpileup_2000evt
"""

import argparse
import json
import os
import pickle
import random

import matplotlib
import numpy as np
import torch
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from pv_finder.evaluation.vertex_matching import (
    _pv_locations_updated_res as pv_locations_updated_res,
)
from pv_finder.evaluation.vertex_matching import (
    compare_res_reco,
)


def select_gpu(device_id):
    """Select GPU or CPU device."""
    if device_id >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


# Global binning info (must match model training)
TOTAL_NUM_BINS = 12000
Z_MAX = 240
Z_MIN = -240
BIN_WIDTH = (Z_MAX - Z_MIN) / TOTAL_NUM_BINS  # 0.04 mm

# Sub-event parameters (12 overlapping 40mm windows)
N_SUBEVENTS = 12
SUBEVENT_WIDTH = 40.0  # mm
SUBEVENT_BINS = 1000
SUBEVENT_STARTS = np.array([-240 + i * 40 for i in range(12)])  # [-240, -200, ..., 200]

# Model input parameters
N_FEATURES = 7
N_TRACKS_PER_SUBEVENT = 100
MASK_VAL = -999999.0  # Padding value from training


def build_subevent_tensors(
    tracks_z0, tracks_d0, tracks_theta, tracks_phi, tracks_sig_d0_z0, tracks_err_z0=None
):
    """
    Build sub-event tensors from track arrays.

    Uses STRICT boundaries (track z0 must be in [z_start, z_end)) to match training.
    Tracks are SORTED by z0 before chunking.
    If more than 100 tracks, creates MULTIPLE chunks and outputs are summed.

    Feature order (from training H5 analysis):
    - 0: d0 / 2
    - 1: z0 RAW (mm)
    - 2: theta / 3
    - 3: (phi + π) / 3
    - 4: sig_d0_z0 (raw)
    - 5: interval_start (raw mm)
    - 6: interval_end (raw mm)

    Returns:
        List of (sub_idx, tensor) tuples. Multiple tensors per sub_idx if > 100 tracks.
    """
    tensors_with_idx = []

    for sub_idx in range(N_SUBEVENTS):
        z_start = SUBEVENT_STARTS[sub_idx]
        z_end = z_start + SUBEVENT_WIDTH

        # STRICT boundaries (matches training data creation)
        mask = (tracks_z0 >= z_start) & (tracks_z0 < z_end)
        n_tracks = np.sum(mask)

        if n_tracks == 0:
            # Empty tensor
            tensor = np.full(
                (N_FEATURES, N_TRACKS_PER_SUBEVENT), MASK_VAL, dtype=np.float32
            )
            tensors_with_idx.append((sub_idx, tensor))
        else:
            # Get all tracks for this sub-event
            z0_sub = tracks_z0[mask]
            d0_sub = tracks_d0[mask]
            theta_sub = tracks_theta[mask]
            phi_sub = tracks_phi[mask]
            sig_d0_z0_sub = tracks_sig_d0_z0[mask]

            # Sort by z0 (matches training data ordering)
            sort_idx = np.argsort(z0_sub)
            z0_sorted = z0_sub[sort_idx]
            d0_sorted = d0_sub[sort_idx]
            theta_sorted = theta_sub[sort_idx]
            phi_sorted = phi_sub[sort_idx]
            sig_d0_z0_sorted = sig_d0_z0_sub[sort_idx]

            # Process in chunks of 100 tracks
            n_chunks = (n_tracks + N_TRACKS_PER_SUBEVENT - 1) // N_TRACKS_PER_SUBEVENT

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * N_TRACKS_PER_SUBEVENT
                end_idx = min(start_idx + N_TRACKS_PER_SUBEVENT, n_tracks)
                n_fill = end_idx - start_idx

                tensor = np.full(
                    (N_FEATURES, N_TRACKS_PER_SUBEVENT), MASK_VAL, dtype=np.float32
                )

                tensor[0, :n_fill] = d0_sorted[start_idx:end_idx] / 2.0
                tensor[1, :n_fill] = z0_sorted[start_idx:end_idx]
                tensor[2, :n_fill] = theta_sorted[start_idx:end_idx] / 3.0
                tensor[3, :n_fill] = (phi_sorted[start_idx:end_idx] + np.pi) / 3.0
                tensor[4, :n_fill] = sig_d0_z0_sorted[start_idx:end_idx]
                tensor[5, :n_fill] = z_start
                tensor[6, :n_fill] = z_end

                tensors_with_idx.append((sub_idx, tensor))

    return tensors_with_idx


def stitch_histograms(sub_histograms):
    """Stitch 12 sub-histograms (each 1000 bins) into one 12000-bin histogram.

    Args:
        sub_histograms: Either shape (12, 1000) or flattened (12000,)
    """
    # Handle both 2D and flattened input
    if sub_histograms.ndim == 2:
        # Shape is (12, 1000) - concatenate
        return sub_histograms.flatten()
    elif len(sub_histograms) == TOTAL_NUM_BINS:
        # Already flattened to 12000
        return sub_histograms.astype(np.float32)
    else:
        # Reshape from flat (12*1000,) assuming correct order
        full_hist = np.zeros(TOTAL_NUM_BINS, dtype=np.float32)
        for sub_idx in range(N_SUBEVENTS):
            start_bin = sub_idx * SUBEVENT_BINS
            end_bin = start_bin + SUBEVENT_BINS
            src_start = sub_idx * SUBEVENT_BINS
            src_end = src_start + SUBEVENT_BINS
            full_hist[start_bin:end_bin] = sub_histograms[src_start:src_end]
        return full_hist


def fit_func_resolution(x, a, b, c, rcc):
    """Fit function for resolution plot (same as ground truth)
    Sigmoid-like function with notch at x=0
    rcc is the resolution (sigma)
    """
    return a / (1 + np.exp(b * (rcc - np.abs(x)))) + c


def make_resolution_plot(
    all_pv_distances, output_dir, label="PV-Finder (Run3)", amvf_distances=None
):
    """Create vertex-vertex resolution plot comparing PV-Finder and AMVF."""
    if len(all_pv_distances) < 10:
        print("Warning: Not enough distances for resolution plot")
        return None, None

    distances = np.array(all_pv_distances)
    print(f"Number of PV-Finder vertex pairs: {len(distances)}")

    if amvf_distances is not None:
        amvf_dist = np.array(amvf_distances)
        print(f"Number of AMVF vertex pairs: {len(amvf_dist)}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot PV-Finder histogram
    counts_pvf, bins, _ = ax.hist(
        distances,
        bins=61,
        range=(-6, 6),
        color="steelblue",
        alpha=0.6,
        edgecolor="darkblue",
        linewidth=0.8,
        label=f"{label}",
    )

    # Bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # PV-Finder errors
    errors_pvf = np.sqrt(counts_pvf + 1)
    ax.errorbar(
        bin_centers,
        counts_pvf,
        yerr=errors_pvf,
        fmt="none",
        ecolor="darkblue",
        elinewidth=1.5,
        capsize=2,
        alpha=0.6,
    )

    # Plot AMVF histogram if provided
    sigma_amvf = None
    if amvf_distances is not None and len(amvf_distances) > 10:
        counts_amvf, _, _ = ax.hist(
            amvf_dist,
            bins=61,
            range=(-6, 6),
            color="orange",
            alpha=0.5,
            edgecolor="darkorange",
            linewidth=0.8,
            label="AMVF",
        )
        errors_amvf = np.sqrt(counts_amvf + 1)
        ax.errorbar(
            bin_centers,
            counts_amvf,
            yerr=errors_amvf,
            fmt="none",
            ecolor="darkorange",
            elinewidth=1.5,
            capsize=2,
            alpha=0.6,
        )

        # Fit AMVF
        try:
            p0_amvf = [max(counts_amvf), 10, min(counts_amvf), 0.5]
            popt_amvf, pcov_amvf = curve_fit(
                fit_func_resolution,
                bin_centers[1:],
                counts_amvf[1:],
                p0=p0_amvf,
                maxfev=10000,
            )
            perr_amvf = np.sqrt(np.diag(pcov_amvf))
            sigma_amvf = abs(popt_amvf[3])

            print("\nAMVF Fit Results:")
            print(f"  Resolution: σ = {sigma_amvf:.3f} ± {perr_amvf[3]:.3f} mm")

            # Plot AMVF fit
            x_fit = np.linspace(-6, 6, 1000)
            y_fit_amvf = fit_func_resolution(x_fit, *popt_amvf)
            ax.plot(
                x_fit,
                y_fit_amvf,
                "darkorange",
                linewidth=2.5,
                linestyle="--",
                label=f"AMVF Fit: σ = {sigma_amvf:.2f} mm",
                zorder=9,
            )
        except Exception as e:
            print(f"Warning: Could not fit AMVF resolution curve: {e}")

    # Fit PV-Finder
    sigma_fit = None
    sigma_err = None
    try:
        p0 = [max(counts_pvf), 10, min(counts_pvf), 0.5]
        popt, pcov = curve_fit(
            fit_func_resolution, bin_centers[1:], counts_pvf[1:], p0=p0, maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        sigma_fit = abs(popt[3])
        sigma_err = perr[3]

        print("\nPV-Finder Fit Results:")
        print(f"  Resolution: σ = {sigma_fit:.3f} ± {sigma_err:.3f} mm")

        # Plot PV-Finder fit
        x_fit = np.linspace(-6, 6, 1000)
        y_fit = fit_func_resolution(x_fit, *popt)
        ax.plot(
            x_fit,
            y_fit,
            "r-",
            linewidth=2.5,
            label=f"PV-Finder Fit: σ = {sigma_fit:.2f} mm",
            zorder=10,
        )

    except Exception as e:
        print(f"Warning: Could not fit PV-Finder resolution curve: {e}")
        close_distances = distances[np.abs(distances) < 2]
        if len(close_distances) > 0:
            sigma_fit = np.std(close_distances)
            sigma_err = 0

    # Formatting
    ax.set_xlabel(r"$\Delta z_{\mathrm{vtx-vtx}}$ [mm]", fontsize=18)
    ax.set_ylabel("Counts", fontsize=18)
    ax.set_title(
        "Distance Between Pairs of Nearby Reconstructed Vertices", fontsize=16, pad=15
    )
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=-50)

    plt.tight_layout()

    for fmt in ["png", "pdf"]:
        outpath = os.path.join(output_dir, f"deltaz_resolution.{fmt}")
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    plt.close(fig)

    return sigma_fit, sigma_err


def plot_category_bar_chart(clean, merged, split, fake, missed, output_dir):
    """Create bar chart of PV categories."""
    categories = ["Clean", "Merged", "Split", "Fake", "Missed"]
    values = [clean, merged, split, fake, missed]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            f"{val:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Vertices", fontsize=14)
    ax.set_title("PV-Finder vs AMVF: Vertex Categories", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=11)

    for fmt in ["png", "pdf"]:
        outpath = os.path.join(output_dir, f"pvfinder_vs_amvf_bar.{fmt}")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PV-Finder on Run3 data against AMVF"
    )
    parser.add_argument("--model", required=True, help="Path to trained model (.pyt)")
    parser.add_argument(
        "--input", required=True, help="Path to input file (.pkl or .root)"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--nevents", type=int, default=1000, help="Number of events to process"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--sigma-vtx-vtx", type=float, default=0.34, help="Resolution for matching (mm)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02, help="Threshold for peak finding"
    )
    parser.add_argument(
        "--integral-threshold", type=float, default=0.4, help="Integral threshold"
    )
    parser.add_argument(
        "--min-width", type=int, default=2, help="Minimum peak width in bins"
    )
    parser.add_argument("--device", type=int, default=0, help="GPU device (-1 for CPU)")
    parser.add_argument(
        "--batch-size", type=int, default=600, help="GPU batch size (sub-events)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Select device
    device = select_gpu(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model: {args.model}")
    model = torch.load(args.model, map_location="cpu")
    model = model.to(device)
    model.eval()

    # Load data
    input_path = args.input
    print(f"Opening data file: {input_path}")

    if input_path.endswith(".pkl"):
        print("Loading from pickle (fast mode)...")
        with open(input_path, "rb") as f:
            events_data = pickle.load(f)
        # Try both branch name conventions
        if "Track_z0" in events_data:
            n_available = len(events_data["Track_z0"])
            track_prefix = "Track_"
        else:
            n_available = len(events_data["RecoTrack_z0"])
            track_prefix = "RecoTrack_"
    else:
        # ROOT file handling
        import uproot

        print("Loading from ROOT file...")
        root_file = uproot.open(input_path)
        tree = root_file["ntuple;1"]
        events_data = tree.arrays(library="np")
        if "Track_z0" in events_data:
            n_available = len(events_data["Track_z0"])
            track_prefix = "Track_"
        else:
            n_available = len(events_data["RecoTrack_z0"])
            track_prefix = "RecoTrack_"

    print(
        f"Loaded {n_available} events from {'pickle' if input_path.endswith('.pkl') else 'ROOT'}"
    )

    n_events = min(args.nevents, n_available)

    # Print transformation info
    print("\n--- TRANSFORMATIONS (from training H5 analysis) ---")
    print("  Feature 0: d0 / 2")
    print("  Feature 1: z0 RAW (mm)")
    print("  Feature 2: theta / 3")
    print("  Feature 3: (phi + π) / 3")
    print("  Feature 4: sig_d0_z0 (raw)")
    print("  Feature 5: interval_start (raw mm)")
    print("  Feature 6: interval_end (raw mm)")
    print("  7 features, padding = -999999, pT NOT used")
    print("  Track selection: 3*sigma_z extension")
    print(f"Resolution: {args.sigma_vtx_vtx:.3f} mm\n")

    # Select random events
    if n_events < n_available:
        selected_indices = sorted(random.sample(range(n_available), n_events))
    else:
        selected_indices = list(range(n_available))

    # Resolution in bins for matching
    sigma_bins = args.sigma_vtx_vtx / BIN_WIDTH

    # Accumulators
    all_clean = []
    all_merged = []
    all_split = []
    all_fake = []
    all_missed = []
    all_pv_distances = []
    all_amvf_distances = []  # AMVF pairwise distances for comparison
    all_reco_z = []
    all_truth_correct = []
    all_truth_ntrks = []
    all_pred_hists = []
    pileup_values = []
    n_vertices_per_event = []

    # Batch processing
    batch_tensors = []
    batch_event_info = []

    for i, evt_idx in enumerate(tqdm(selected_indices, desc="Processing events")):
        # Extract event data
        tracks_z0 = np.asarray(events_data[f"{track_prefix}z0"][evt_idx])
        tracks_d0 = np.asarray(events_data[f"{track_prefix}d0"][evt_idx])
        tracks_theta = np.asarray(events_data[f"{track_prefix}theta"][evt_idx])
        tracks_phi = np.asarray(events_data[f"{track_prefix}phi"][evt_idx])

        # sig_d0_z0
        if f"{track_prefix}ErrD0Z0" in events_data:
            tracks_sig_d0_z0 = np.asarray(
                events_data[f"{track_prefix}ErrD0Z0"][evt_idx]
            )
        elif f"{track_prefix}sig_d0_z0" in events_data:
            tracks_sig_d0_z0 = np.asarray(
                events_data[f"{track_prefix}sig_d0_z0"][evt_idx]
            )
        else:
            tracks_sig_d0_z0 = np.zeros_like(tracks_z0)

        # ErrZ0 for 3-sigma track selection
        if f"{track_prefix}ErrZ0" in events_data:
            tracks_err_z0 = np.asarray(events_data[f"{track_prefix}ErrZ0"][evt_idx])
        else:
            tracks_err_z0 = np.full_like(tracks_z0, 0.5)  # Default 0.5mm

        # AMVF vertices
        amvf_z = np.asarray(events_data["RecoVertex_z"][evt_idx])
        amvf_ntrks = np.asarray(events_data["RecoVertex_nTracks"][evt_idx])

        # Beam spot correction (AMVF vertices are relative to beam spot)
        if "BeamPosZ" in events_data:
            beam_z = events_data["BeamPosZ"][evt_idx]
            if hasattr(beam_z, "__len__"):
                beam_z = float(beam_z[0]) if len(beam_z) > 0 else 0.0
            else:
                beam_z = float(beam_z)
        else:
            beam_z = 0.0
        amvf_z = amvf_z - beam_z  # Convert to detector coordinates

        # Filter AMVF vertices with nTracks >= 2
        valid_amvf = amvf_ntrks >= 2
        amvf_z = amvf_z[valid_amvf]
        amvf_ntrks = amvf_ntrks[valid_amvf]

        # Pileup
        if "ActualNumOfInt" in events_data:
            mu = events_data["ActualNumOfInt"][evt_idx]
            if hasattr(mu, "__len__"):
                mu = float(mu[0]) if len(mu) > 0 else len(amvf_z)
            else:
                mu = float(mu)
        else:
            mu = len(amvf_z)
        pileup_values.append(mu)
        n_vertices_per_event.append(len(amvf_z))

        # Build sub-event tensors (with 3*sigma_z0 track selection)
        # Returns list of (sub_idx, tensor) - may have multiple chunks per sub-event
        subevent_data = build_subevent_tensors(
            tracks_z0,
            tracks_d0,
            tracks_theta,
            tracks_phi,
            tracks_sig_d0_z0,
            tracks_err_z0,
        )

        # Add to batch - track which sub-event each tensor belongs to
        n_tensors = len(subevent_data)
        for sub_idx, tensor in subevent_data:
            batch_tensors.append(tensor)
        batch_event_info.append((i, subevent_data, amvf_z, amvf_ntrks))

        # Process batch when full or at end
        if len(batch_tensors) >= args.batch_size or i == len(selected_indices) - 1:
            # Run inference
            batch_np = np.stack(batch_tensors, axis=0)
            batch_tensor = torch.from_numpy(batch_np).float().to(device)

            with torch.no_grad():
                pred_batch = model(batch_tensor)
                # Handle squeeze issue for batch_size=1
                if pred_batch.dim() == 1:
                    pred_batch = pred_batch.unsqueeze(0)
                pred_batch = pred_batch.cpu().numpy()

            # Process results per event
            tensor_idx = 0
            for evt_i, subevent_data, evt_amvf_z, evt_amvf_ntrks in batch_event_info:
                n_tensors = len(subevent_data)

                # Accumulate outputs per sub-event (sum multiple chunks)
                subevent_outputs = {
                    i: np.zeros(SUBEVENT_BINS, dtype=np.float32)
                    for i in range(N_SUBEVENTS)
                }

                for chunk_i, (sub_idx, _) in enumerate(subevent_data):
                    subevent_outputs[sub_idx] += pred_batch[
                        tensor_idx + chunk_i
                    ].flatten()

                tensor_idx += n_tensors

                # Stitch the summed sub-event outputs
                sub_hists = np.array([subevent_outputs[i] for i in range(N_SUBEVENTS)])
                full_hist = stitch_histograms(sub_hists)
                all_pred_hists.append(full_hist)

                # Find predicted PVs
                pv_result = pv_locations_updated_res(
                    full_hist,
                    threshold=args.threshold,
                    integral_threshold=args.integral_threshold,
                    min_width=args.min_width,
                )
                pred_z = pv_result[0]  # z positions in mm

                all_reco_z.extend(pred_z.tolist())

                # Convert to bins for matching
                pred_bins = (pred_z - Z_MIN) / BIN_WIDTH
                amvf_bins = (evt_amvf_z - Z_MIN) / BIN_WIDTH

                # Get resolution for each prediction
                pred_res = np.full(len(pred_bins), sigma_bins)

                # Compare
                if len(amvf_bins) > 0 and len(pred_bins) > 0:
                    perf_info, truth_class, _ = compare_res_reco(
                        amvf_bins, pred_bins, pred_res, debug=False
                    )

                    n_clean = perf_info.reco_clean
                    n_merged = perf_info.reco_merged
                    n_split = perf_info.reco_split
                    n_fake = perf_info.reco_fake
                    n_missed = sum(1 for tc in truth_class if tc == [])

                    all_clean.append(n_clean)
                    all_merged.append(n_merged)
                    all_split.append(n_split)
                    all_fake.append(n_fake)
                    all_missed.append(n_missed)

                    # Truth tracking
                    for t_idx, t_class in enumerate(truth_class):
                        matched = 1 if t_class != [] else 0
                        all_truth_correct.append(matched)
                        all_truth_ntrks.append(
                            evt_amvf_ntrks[t_idx] if t_idx < len(evt_amvf_ntrks) else 0
                        )
                elif len(pred_bins) > 0:
                    all_clean.append(0)
                    all_merged.append(0)
                    all_split.append(0)
                    all_fake.append(len(pred_bins))
                    all_missed.append(0)
                else:
                    all_clean.append(0)
                    all_merged.append(0)
                    all_split.append(0)
                    all_fake.append(0)
                    all_missed.append(len(amvf_bins))

                # Calculate pairwise distances for resolution plot (same as ground truth eval)
                # All pairs, no filtering - histogram range handles display
                if len(pred_z) >= 2:
                    pred_z_shuffled = pred_z.copy()
                    np.random.shuffle(pred_z_shuffled)
                    for ii in range(len(pred_z_shuffled) - 1):
                        for jj in range(ii + 1, len(pred_z_shuffled)):
                            dist = pred_z_shuffled[ii] - pred_z_shuffled[jj]
                            all_pv_distances.append(dist)

                # Calculate AMVF pairwise distances for comparison
                if len(evt_amvf_z) >= 2:
                    amvf_z_shuffled = evt_amvf_z.copy()
                    np.random.shuffle(amvf_z_shuffled)
                    for ii in range(len(amvf_z_shuffled) - 1):
                        for jj in range(ii + 1, len(amvf_z_shuffled)):
                            dist = amvf_z_shuffled[ii] - amvf_z_shuffled[jj]
                            all_amvf_distances.append(dist)

            # Clear batch
            batch_tensors = []
            batch_event_info = []

    # Summary statistics
    total_clean = sum(all_clean)
    total_merged = sum(all_merged)
    total_split = sum(all_split)
    total_fake = sum(all_fake)
    total_missed = sum(all_missed)
    total_amvf = total_clean + total_merged + total_missed

    efficiency = (total_clean + total_merged) / total_amvf if total_amvf > 0 else 0
    fpr = total_fake / n_events

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Events processed: {n_events}")
    print(f"Total AMVF vertices (nTracks >= 2): {total_amvf}")
    print(f"  Clean matches:  {total_clean} ({100 * total_clean / total_amvf:.1f}%)")
    print(f"  Merged:         {total_merged} ({100 * total_merged / total_amvf:.1f}%)")
    print(f"  Split:          {total_split} ({100 * total_split / total_amvf:.2f}%)")
    print(f"  Fake:           {total_fake}")
    print(f"  Missed:         {total_missed}")
    print(f"\nEfficiency: {100 * efficiency:.2f}%")
    print(f"False Positive Rate: {fpr:.4f} ({100 * fpr:.2f}%)")
    print(f"Avg pileup: {np.mean(pileup_values):.2f}")
    print(f"Avg vertices/event: {np.mean(n_vertices_per_event):.2f}")

    # Save results
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    summary = {
        "n_events": n_events,
        "n_amvf_total": int(total_amvf),
        "n_clean": int(total_clean),
        "n_merged": int(total_merged),
        "n_split": int(total_split),
        "n_fake": int(total_fake),
        "n_missed": int(total_missed),
        "efficiency": float(efficiency),
        "false_positive_rate": float(fpr),
        "avg_pileup": float(np.mean(pileup_values)),
        "avg_vertices_per_event": float(np.mean(n_vertices_per_event)),
    }

    with open(os.path.join(args.output_dir, "run3_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print("  Saved: run3_eval_summary.json")

    pickle.dump(
        all_clean,
        open(os.path.join(args.output_dir, "run3_eval_separated_clean.p"), "wb"),
    )
    print("  Saved: run3_eval_separated_clean.p")
    pickle.dump(
        all_merged,
        open(os.path.join(args.output_dir, "run3_eval_separated_merged.p"), "wb"),
    )
    print("  Saved: run3_eval_separated_merged.p")
    pickle.dump(
        all_split,
        open(os.path.join(args.output_dir, "run3_eval_separated_split.p"), "wb"),
    )
    print("  Saved: run3_eval_separated_split.p")
    pickle.dump(
        all_fake,
        open(os.path.join(args.output_dir, "run3_eval_separated_fake.p"), "wb"),
    )
    print("  Saved: run3_eval_separated_fake.p")
    pickle.dump(
        all_truth_ntrks,
        open(os.path.join(args.output_dir, "run3_eval_truth_ntrks.p"), "wb"),
    )
    print("  Saved: run3_eval_truth_ntrks.p")
    pickle.dump(
        all_truth_correct,
        open(os.path.join(args.output_dir, "run3_eval_truth_correct.p"), "wb"),
    )
    print("  Saved: run3_eval_truth_correct.p")
    pickle.dump(
        all_reco_z,
        open(os.path.join(args.output_dir, "run3_eval_total_reco_z.p"), "wb"),
    )
    print("  Saved: run3_eval_total_reco_z.p")
    pickle.dump(
        all_pred_hists,
        open(os.path.join(args.output_dir, "run3_eval_all_pred_hists.p"), "wb"),
    )
    print("  Saved: run3_eval_all_pred_hists.p")

    print(f"\nPairwise distances collected: {len(all_pv_distances)}")

    # Generate resolution plot
    print("\n" + "=" * 60)
    print("Creating Resolution Plot")
    print("=" * 60)
    print(f"PV-Finder pairwise distances: {len(all_pv_distances)}")
    print(f"AMVF pairwise distances: {len(all_amvf_distances)}")
    sigma_fit, sigma_err = make_resolution_plot(
        all_pv_distances, args.output_dir, amvf_distances=all_amvf_distances
    )

    if sigma_fit is not None:
        summary["sigma_vtx_vtx_fitted"] = float(sigma_fit)
        summary["sigma_vtx_vtx_error"] = float(sigma_err)
        with open(os.path.join(args.output_dir, "run3_eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

    # Generate category bar chart
    print("\n" + "=" * 60)
    print("Creating Category Bar Chart")
    print("=" * 60)
    plot_category_bar_chart(
        total_clean,
        total_merged,
        total_split,
        total_fake,
        total_missed,
        args.output_dir,
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
