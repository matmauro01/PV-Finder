#!/usr/bin/env python3
"""
Per-vertex histogram visualization for HL-LHC (Run 4) data.

Runs two models — Run 2 (trained on MC) and HL-LHC (trained on PU200) — on
HL-LHC ROOT events and produces per-vertex zoom plots and event overviews.

Output layout::

    <output-dir>/
        run2_model/event{idx}/    -- overview + per-vertex plots (Run 2 model)
        hllhc_model/event{idx}/   -- overview + per-vertex plots (HL-LHC model)

Usage::

    PYTHONPATH=src python -m \
      pv_finder.diagnostics.per_vertex_visualization.run_per_vertex_run4 \
      --n-events 3 --output-dir outputs/per_vertex_run4 [--device cpu]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch

from pv_finder.data.run3_io import Run3Event, load_run3_from_root
from pv_finder.diagnostics.domain_shift_investigation.kde_study.analytical_kde import (
    compute_analytical_kde_event_run3,
)
from pv_finder.diagnostics.per_vertex_visualization.inference import (
    run_e2e_on_events,
)
from pv_finder.diagnostics.per_vertex_visualization.peak_matching import (
    classify_vertices,
    find_histogram_peaks,
    peaks_in_vertex_window,
)
from pv_finder.diagnostics.per_vertex_visualization.vertex_plots import (
    plot_event_overview,
    plot_vertex_zoom,
)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = (
    "data/run4/Run4_MC21_ITk/"
    "ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root"
)
_DEFAULT_RUN2_MODEL = (
    "model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130.pyt"
)
_DEFAULT_HLLHC_MODEL = (
    "model_weights/hllhc_pu200_mlp50_e2e400_v2_phase2_epoch_100_fullstate.pth"
)

# E2E v1 model configuration — matches run_eval_pvf_run3.py
_N_BINS_SUB = 1000
# fmt: off
_E2E_CONFIG = dict(
    n_InputFeatures=7, n_OutputFeatures=_N_BINS_SUB,
    l_HiddenNodes=[100] * 5, n_LatentChannels=1, n_UNetChannels=64,
    sc_mode="concat", dropout=0.25, LeakyReLU_param=0.01,
    predScaleFactor=0.001, maskVal=-240.0, d_selection="ConvBNrelu",
    u_selection="Up",
)
_E2E_WIDE_OVERRIDES = dict(n_UNetChannels=96, l_HiddenNodes=[128] * 5)
# fmt: on


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _register_legacy_module() -> None:
    """Register the legacy ``model.autoencoder_models`` module alias.

    Needed to unpickle ``.pyt`` checkpoints saved with ``torch.save(model)``,
    where the class is referenced as ``model.autoencoder_models.<Class>``.
    """
    import pv_finder.models.autoencoder_models as _am

    if "model" not in sys.modules:
        sys.modules["model"] = types.ModuleType("model")
    sys.modules["model.autoencoder_models"] = _am


def load_run2_model(path: str, device: str = "cpu") -> torch.nn.Module:
    """Load the Run 2 e2e model from a ``.pyt`` full-object checkpoint."""
    _register_legacy_module()
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Run 2 model: {Path(path).name} ({n:,} params)")
    return model


def load_hllhc_model(path: str, device: str = "cpu") -> torch.nn.Module:
    """Load the HL-LHC wide e2e model from a fullstate ``.pth`` checkpoint.

    Uses the v1 architecture (``trackstoHists_UNet_1000``) with the wide
    config (96 UNet channels, [128]*5 MLP hidden nodes).
    """
    from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000

    cfg = dict(_E2E_CONFIG)
    cfg.update(_E2E_WIDE_OVERRIDES)
    model = trackstoHists_UNet_1000(**cfg)

    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        ep = ckpt.get("epoch", "?")
        loss = ckpt.get("loss")
        loss_s = f"{loss:.6f}" if loss is not None else "N/A"
        print(f"  HL-LHC model: {Path(path).name} epoch={ep} loss={loss_s}")
    elif hasattr(ckpt, "state_dict"):
        model.load_state_dict(ckpt.state_dict())
    else:
        model.load_state_dict(ckpt)

    n = sum(p.numel() for p in model.parameters())
    print(f"    ({n:,} params, wide v1)")
    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Run3Event -> dict adapters (for inference and analytical KDE)
# ---------------------------------------------------------------------------


def _run3event_to_dict(evt: Run3Event) -> dict:
    """Convert a Run3Event named tuple to the dict format used by inference
    and analytical KDE functions."""
    return {
        "z0": evt.z0,
        "d0": evt.d0,
        "d0_err": evt.d0_err,
        "z0_err": evt.z0_err,
        "d0_z0_cov": evt.d0_z0_cov,
        "event_idx": evt.event_idx,
    }


def _truth_vertices_from_event(evt: Run3Event) -> list[float]:
    """Extract beam-corrected AMVF truth vertex positions from a Run3Event.

    Run3Event.amvf_z already has beam correction applied (see run3_io.py),
    but we need to verify: load_run3_from_root stores raw RecoVertex_z
    filtered by nTracks>=2. Beam correction is NOT applied in run3_io.py
    -- the user's task description says it is, but checking the code shows
    it stores raw values.

    For consistency with the eval pipeline (which does apply beam correction),
    we subtract beam_z here.
    """
    # amvf_z is raw RecoVertex_z (filtered to nTracks>=2) -- subtract beam_z
    corrected = evt.amvf_z - evt.beam_z
    return sorted(float(z) for z in corrected)


# ---------------------------------------------------------------------------
# Inference on Run3Event list
# ---------------------------------------------------------------------------


def run_e2e_on_run4_events(
    model: torch.nn.Module,
    events: list[Run3Event],
    device: str = "cpu",
    batch_size: int = 32,
) -> list[np.ndarray]:
    """Run e2e model on HL-LHC events via the run3 inference path.

    Converts Run3Event named tuples to the dict format expected by
    ``run_e2e_on_events``, then delegates to the existing inference code.

    Returns list of (12, 1000) arrays, one per event.
    """
    event_dicts = [_run3event_to_dict(evt) for evt in events]
    return run_e2e_on_events(
        model, event_dicts, "run3", device=device, batch_size=batch_size
    )


# ---------------------------------------------------------------------------
# Analytical KDE on Run3Event list
# ---------------------------------------------------------------------------


def compute_analytical_kdes_run4(
    events: list[Run3Event],
) -> list[np.ndarray]:
    """Compute analytical KDE for each HL-LHC event.

    Converts Run3Event to the dict format expected by the analytical KDE.
    """
    from tqdm import tqdm

    results: list[np.ndarray] = []
    for evt in tqdm(events, desc="Analytical KDE (HL-LHC)"):
        evt_dict = _run3event_to_dict(evt)
        kde = compute_analytical_kde_event_run3(evt_dict)
        results.append(kde)
    return results


# ---------------------------------------------------------------------------
# Per-event processing (shared across models)
# ---------------------------------------------------------------------------


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
) -> None:
    """Generate overview + per-vertex zoom plots for one event."""
    evt_dir = os.path.join(output_dir, f"event{event_idx:04d}")
    pred_peaks = find_histogram_peaks(
        hist_e2e.reshape(-1),
        integral_threshold=0.2,  # HL-LHC PU200 needs lower threshold
    )
    truth_labels, _ = classify_vertices(truth_vertices, pred_peaks)

    plot_event_overview(
        hist_e2e,
        hist_ana,
        truth_vertices,
        pred_peaks,
        event_idx,
        dataset_label,
        evt_dir,
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
            evt_dir,
            window_mm=window_mm,
            tracks_z0=tracks_z0,
            tracks_d0=tracks_d0,
            tracks_d0_err=tracks_d0_err,
            all_truth_vertices=truth_vertices,
            vertex_label=truth_labels[vi],
        )

    # Compact per-event summary
    n_matched = sum(
        1
        for vz in truth_vertices
        if peaks_in_vertex_window(pred_peaks, vz, window_mm=0.5)
    )
    print(
        f"  Event {event_idx}: {len(truth_vertices)} AMVF vtx, "
        f"{len(pred_peaks)} peaks, {n_matched} matched"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments, load data and models, produce per-vertex plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Per-vertex histogram visualization for HL-LHC (Run 4) data. "
            "Compares Run 2 and HL-LHC trained models."
        )
    )
    parser.add_argument("--root", default=_DEFAULT_ROOT, help="HL-LHC ROOT file path")
    parser.add_argument(
        "--run2-model",
        default=_DEFAULT_RUN2_MODEL,
        help="Run 2 e2e model (.pyt, full object)",
    )
    parser.add_argument(
        "--hllhc-model",
        default=_DEFAULT_HLLHC_MODEL,
        help="HL-LHC e2e model (.pth, fullstate dict)",
    )
    parser.add_argument("--n-events", type=int, default=5, help="Number of events")
    parser.add_argument(
        "--output-dir",
        default="outputs/per_vertex_run4",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="GPU index (e.g. 0); use -1 or omit for CPU",
    )
    parser.add_argument(
        "--window-mm",
        type=float,
        default=8.0,
        help="Zoom window half-width around each truth vertex (mm)",
    )
    parser.add_argument(
        "--entry-start",
        type=int,
        default=0,
        help="First ROOT entry to read",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for event selection order",
    )
    args = parser.parse_args()

    # ---- Load events from ROOT ----
    print("=" * 60)
    print("  HL-LHC (Run 4) Per-Vertex Visualization")
    print("=" * 60)

    if args.device >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.device}"
        print(f"Device: GPU {args.device} -- {torch.cuda.get_device_name(args.device)}")
    else:
        device = "cpu"
        print("Device: CPU")

    events = load_run3_from_root(
        args.root,
        max_events=args.n_events * 3,  # load extra to allow filtering
        min_tracks=50,  # PU200 events should have many tracks
        min_amvf_vtx=5,
        entry_start=args.entry_start,
    )

    if not events:
        print("No events loaded. Check ROOT file path and filters.")
        return

    # Optionally shuffle and select
    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    print(f"Shuffle seed: {seed}")
    rng = random.Random(seed)
    rng.shuffle(events)
    events = events[: args.n_events]
    print(f"Selected {len(events)} events for visualization")

    for evt in events:
        mu_str = f"mu={evt.mu:.0f}" if evt.mu is not None else "mu=?"
        print(
            f"  Event {evt.event_idx}: {evt.n_tracks} tracks, "
            f"{len(evt.amvf_z)} AMVF vtx, {mu_str}"
        )

    # ---- Load models ----
    print("\n--- Loading Models ---")
    run2_model = load_run2_model(args.run2_model, device=device)
    hllhc_model = load_hllhc_model(args.hllhc_model, device=device)

    # ---- Inference ----
    print("\n--- Running Inference ---")
    print("Run 2 model on HL-LHC events...")
    run2_hists = run_e2e_on_run4_events(run2_model, events, device=device)
    print("HL-LHC model on HL-LHC events...")
    hllhc_hists = run_e2e_on_run4_events(hllhc_model, events, device=device)

    # ---- Analytical KDE ----
    print("\n--- Computing Analytical KDE ---")
    ana_kdes = compute_analytical_kdes_run4(events)

    # ---- Extract truth vertices and track arrays ----
    truth_vtx_list = [_truth_vertices_from_event(evt) for evt in events]

    # ---- Generate plots for each model ----
    run2_out = os.path.join(args.output_dir, "run2_model")
    hllhc_out = os.path.join(args.output_dir, "hllhc_model")

    for model_label, hists, out_dir in [
        ("Run 2 model", run2_hists, run2_out),
        ("HL-LHC model", hllhc_hists, hllhc_out),
    ]:
        print(f"\n{'=' * 50}")
        print(f"{model_label} on HL-LHC data:")
        for i, evt in enumerate(events):
            _process_event(
                event_idx=evt.event_idx,
                hist_e2e=hists[i],
                hist_ana=ana_kdes[i],
                truth_vertices=truth_vtx_list[i],
                tracks_z0=evt.z0,
                tracks_d0=evt.d0,
                tracks_d0_err=evt.d0_err,
                dataset_label="HL-LHC",
                output_dir=out_dir,
                window_mm=args.window_mm,
            )

    print(f"\nOutputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
