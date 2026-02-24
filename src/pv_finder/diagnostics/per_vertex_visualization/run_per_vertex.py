#!/usr/bin/env python3
"""
Per-vertex histogram visualization for MC and Run 3 data.

Runs the e2e trackstoHists_UNet_1000 model on MC and Run 3 events and
produces per-vertex zoom plots plus full-event overview figures.

Usage:
    PYTHONPATH=src venv/bin/python3 -m \\
      pv_finder.diagnostics.per_vertex_visualization.run_per_vertex \\
      --n-events 3 --output-dir outputs/per_vertex [--device cpu] [--window-mm 8]
"""

from __future__ import annotations

import argparse
import os
import warnings

import h5py
import numpy as np

from pv_finder.data.feature_loading import (
    MASK_VAL,
    N_SUBEVENTS,
    load_mc_data,
    load_run3_data,
)
from pv_finder.diagnostics.domain_shift_investigation.kde_study.analytical_kde import (
    compute_analytical_kdes_batch,
)
from pv_finder.diagnostics.per_vertex_visualization.inference import (
    load_e2e_model,
    run_e2e_on_events,
)
from pv_finder.diagnostics.per_vertex_visualization.peak_matching import (
    find_histogram_peaks,
    load_mc_truth_vertices,
    load_run3_amvf_vertices,
    peaks_in_vertex_window,
)
from pv_finder.diagnostics.per_vertex_visualization.vertex_plots import (
    plot_event_overview,
    plot_vertex_zoom,
)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Default paths (relative to project root)
# ---------------------------------------------------------------------------
_DEFAULT_MC_H5 = "data/monte_carlo/training_data.h5"
_DEFAULT_RUN3_NPZ = "data/run3/cache_file3_2000ev_seed42.npz"
_DEFAULT_MODEL = "model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130.pyt"
_VAL_START_SUB = 428400  # 70% of 612000 subevents = validation split start
_VAL_START_EVENT = _VAL_START_SUB // N_SUBEVENTS  # 35700


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_mc_truth_hists(h5_path: str, n_events: int) -> np.ndarray:
    """Load MC truth histograms from H5 target_y_split, channel 0.

    Returns shape (n_events, 12, 1000) float32.
    """
    n_sub = n_events * N_SUBEVENTS
    with h5py.File(h5_path, "r") as f:
        raw = f["target_y_split"][_VAL_START_SUB : _VAL_START_SUB + n_sub, 0, :]
    return raw.reshape(n_events, N_SUBEVENTS, 1000).astype(np.float32)


def _mc_track_arrays(
    event: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (z0, d0, d0_err) flat arrays for all active MC tracks.

    MC H5 format: tracks shape (12, 7, 695); channels 0=d0, 1=z0, 2=d0_err.
    Active tracks have z0 > MASK_VAL + 1.0 (filters out -999999 padding).
    """
    tracks = event["tracks"]  # (12, 7, 695)
    z0_all = tracks[:, 1, :].ravel()
    d0_all = tracks[:, 0, :].ravel()
    d0_err_all = tracks[:, 2, :].ravel()
    active = z0_all > MASK_VAL + 1.0
    return z0_all[active], d0_all[active], d0_err_all[active]


def _process_event(
    event_idx: int,
    hist_e2e: np.ndarray,
    hist_ana: np.ndarray,
    truth_vertices: list[float],
    tracks_z0: np.ndarray,
    tracks_d0: np.ndarray,
    tracks_d0_err: np.ndarray,
    dataset_label: str,
    output_dir: str,
    window_mm: float,
    hist_truth: np.ndarray | None = None,
) -> None:
    """Generate overview + per-vertex zoom plots for one event and print summary."""
    pred_peaks = find_histogram_peaks(hist_e2e.reshape(-1))

    plot_event_overview(
        hist_e2e,
        hist_ana,
        truth_vertices,
        pred_peaks,
        event_idx,
        dataset_label,
        output_dir,
        hist_truth=hist_truth,
    )

    for vi, vz in enumerate(truth_vertices):
        plot_vertex_zoom(
            hist_e2e,
            hist_ana,
            vz,
            pred_peaks,
            event_idx,
            vi,
            dataset_label,
            output_dir,
            window_mm=window_mm,
            tracks_z0=tracks_z0,
            tracks_d0=tracks_d0,
            tracks_d0_err=tracks_d0_err,
            hist_truth=hist_truth,
        )

    # Compact per-event summary
    print(
        f"  Event {event_idx}: {len(truth_vertices)} truth vertices, "
        f"{len(pred_peaks)} histogram peaks total"
    )
    for vi, vz in enumerate(truth_vertices):
        matched = peaks_in_vertex_window(pred_peaks, vz, window_mm=0.5)
        tag = (
            f"{len(matched)} peak(s) in +/-0.5mm" if matched else "no peak in +/-0.5mm"
        )
        print(f"    vtx {vi:02d}: z={vz:+.2f} mm  {tag}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, run inference, and produce all plots."""
    parser = argparse.ArgumentParser(
        description="Per-vertex histogram visualization (e2e model + analytical KDE)"
    )
    parser.add_argument(
        "--n-events", type=int, default=3, help="Number of events to process"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/per_vertex",
        help="Root output directory",
    )
    parser.add_argument(
        "--device", default="cpu", help="PyTorch device string (cpu/cuda)"
    )
    parser.add_argument(
        "--window-mm",
        type=float,
        default=8.0,
        help="Half-width of the zoom window around each truth vertex (mm)",
    )
    parser.add_argument("--mc-h5", default=_DEFAULT_MC_H5, help="MC HDF5 file")
    parser.add_argument(
        "--run3-npz", default=_DEFAULT_RUN3_NPZ, help="Run 3 NPZ cache file"
    )
    parser.add_argument(
        "--model-path", default=_DEFAULT_MODEL, help="e2e model weights (.pyt)"
    )
    args = parser.parse_args()

    n_ev = args.n_events

    # ---- Load tracks ----
    mc_events = load_mc_data(args.mc_h5, n_events=n_ev)
    run3_events = load_run3_data(args.run3_npz, max_events=n_ev)

    # Original NPZ indices are needed by load_run3_amvf_vertices
    run3_indices = [evt["event_idx"] for evt in run3_events]

    # ---- Load truth vertices ----
    mc_truth_vtx = load_mc_truth_vertices(args.mc_h5, n_ev)
    run3_amvf_vtx = load_run3_amvf_vertices(args.run3_npz, run3_indices)

    # ---- e2e model ----
    print(f"Loading e2e model: {args.model_path}")
    model = load_e2e_model(args.model_path, device=args.device)

    # ---- Inference ----
    print("Running e2e model on MC events...")
    mc_e2e = run_e2e_on_events(model, mc_events, "mc", device=args.device)
    print("Running e2e model on Run 3 events...")
    run3_e2e = run_e2e_on_events(model, run3_events, "run3", device=args.device)

    # ---- Analytical KDE (CPU, may take a minute) ----
    mc_ana = compute_analytical_kdes_batch(mc_events, "mc")
    run3_ana = compute_analytical_kdes_batch(run3_events, "run3")

    # ---- MC truth histograms (training targets) ----
    mc_truth_hists = _load_mc_truth_hists(args.mc_h5, n_ev)

    # ---- Output directories ----
    mc_out = os.path.join(args.output_dir, "mc")
    run3_out = os.path.join(args.output_dir, "run3")

    # ---- Process MC ----
    print(f"\n{'=' * 50}")
    print("MC events:")
    for i, evt in enumerate(mc_events):
        z0, d0, d0_err = _mc_track_arrays(evt)
        _process_event(
            event_idx=i,
            hist_e2e=mc_e2e[i],
            hist_ana=mc_ana[i],
            truth_vertices=mc_truth_vtx[i],
            tracks_z0=z0,
            tracks_d0=d0,
            tracks_d0_err=d0_err,
            dataset_label="mc",
            output_dir=mc_out,
            window_mm=args.window_mm,
            hist_truth=mc_truth_hists[i],
        )

    # ---- Process Run 3 ----
    print(f"\n{'=' * 50}")
    print("Run 3 events:")
    for i, evt in enumerate(run3_events):
        _process_event(
            event_idx=i,
            hist_e2e=run3_e2e[i],
            hist_ana=run3_ana[i],
            truth_vertices=run3_amvf_vtx[i],
            tracks_z0=evt["z0"],
            tracks_d0=evt["d0"],
            tracks_d0_err=evt["d0_err"],
            dataset_label="run3",
            output_dir=run3_out,
            window_mm=args.window_mm,
        )

    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
