#!/usr/bin/env python3
"""Per-vertex FAILURE-MODE visualization for the HL-LHC v4b (v2) model.

Runs the v4b end-to-end model on HL-LHC events, classifies every MC-truth and
reco vertex (clean / merged / split / fake / missed) against MC ``TruthVertex``
(nTracks>=2), and produces per-vertex zoom plots **only for the failure modes**
— merged & missed truth vertices and fake reco peaks — so we can eyeball exactly
what the model gets wrong and discuss where the gains are.

Each event gets a folder with ``merged/``, ``missed/``, ``fake/`` subfolders.

Usage::

    PYTHONPATH=src python -u \
      src/pv_finder/diagnostics/per_vertex_visualization/failure_mode_viz.py \
      --n-events 4 --device 0 \
      --output-dir outputs/06_08_2026_output/failure_mode_viz
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch

from pv_finder.data.run3_io import load_run3_from_root
from pv_finder.diagnostics.domain_shift_investigation.kde_study.analytical_kde import (
    compute_analytical_kde_event_run3,
)
from pv_finder.diagnostics.per_vertex_visualization.inference import run_e2e_on_events
from pv_finder.diagnostics.per_vertex_visualization.peak_matching import (
    classify_vertices,
    find_histogram_peaks,
)
from pv_finder.diagnostics.per_vertex_visualization.vertex_plots import plot_vertex_zoom
from pv_finder.models.autoencoder_models import MaskedDNN
from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2

_DEFAULT_MODEL = (
    "model_weights/"
    "hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3_fullstate.pth"
)
_DEFAULT_ROOT = (
    "data/run4/PU200_withTiming/ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.root"
)
_INTEGRAL_THRESHOLD = 0.2  # HL-LHC PU200 operating point (matches eval)
_KINDS = ("merged", "missed", "fake")


def load_v4b(path: str, device: torch.device, n_ch: int, n_lat: int) -> torch.nn.Module:
    """Load the v4b v2 model (TracksToHist_v2) from a fullstate/.pyt checkpoint."""
    if "model" not in sys.modules:
        sys.modules["model"] = types.ModuleType("model")
    import pv_finder.models.autoencoder_models as _am

    sys.modules["model.autoencoder_models"] = _am

    t2kde = MaskedDNN(
        input_size=7,
        hidden_nodes=[128] * 5,
        output_size=1000 * n_lat,
        leaky_param=0.01,
        maskVal=-240.0,
        predScaleFactor=0.001,
    )
    model = TracksToHist_v2(
        t2kde, UNet_1000_v2(n=n_ch, n_features=n_lat, dropout_p=0.0)
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  v4b model: {Path(path).name} ({n:,} params)")
    return model


def _to_dict(evt) -> dict:
    return {
        "z0": evt.z0,
        "d0": evt.d0,
        "d0_err": evt.d0_err,
        "z0_err": evt.z0_err,
        "d0_z0_cov": evt.d0_z0_cov,
        "event_idx": evt.event_idx,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--root", default=_DEFAULT_ROOT)
    ap.add_argument("--n-events", type=int, default=4)
    ap.add_argument("--per-category", type=int, default=6,
                    help="max plots per failure mode per event")  # fmt: skip
    ap.add_argument("--window-mm", type=float, default=4.0)
    ap.add_argument("--e2e-unet-channels", type=int, default=280)
    ap.add_argument("--e2e-latent-channels", type=int, default=4)
    ap.add_argument("--match-window-mm", type=float, default=0.5)
    ap.add_argument("--entry-start", type=int, default=80000,
                    help="read from late entries to reduce train-set overlap")  # fmt: skip
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-kde", action="store_true", help="skip analytical KDE overlay")
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument(
        "--output-dir", default="outputs/06_08_2026_output/failure_mode_viz"
    )
    args = ap.parse_args()

    device = (
        torch.device(f"cuda:{args.device}")
        if args.device >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    events = load_run3_from_root(
        args.root, max_events=args.n_events * 4, min_tracks=50,
        min_amvf_vtx=5, entry_start=args.entry_start,
    )  # fmt: skip
    if not events or events[0].truth_z is None:
        print("ERROR: no events or no MC TruthVertex in this ROOT file.")
        sys.exit(1)
    rng = random.Random(args.seed)
    rng.shuffle(events)
    events = events[: args.n_events]
    print(f"Selected {len(events)} events (MC TruthVertex as ground truth)")

    model = load_v4b(
        args.model, device, args.e2e_unet_channels, args.e2e_latent_channels
    )
    hists = run_e2e_on_events(
        model, [_to_dict(e) for e in events], "run3", device=str(device)
    )

    totals = {k: 0 for k in _KINDS}
    for evt, hist in zip(events, hists):
        truth = sorted(float(z) for z in evt.truth_z)
        peaks = find_histogram_peaks(
            hist.reshape(-1), integral_threshold=_INTEGRAL_THRESHOLD
        )
        tlab, rlab = classify_vertices(
            truth, peaks, match_window_mm=args.match_window_mm
        )

        if args.no_kde:
            ana = np.zeros_like(hist)
        else:
            try:
                ana = compute_analytical_kde_event_run3(_to_dict(evt))
            except Exception as exc:  # noqa: BLE001
                print(f"  (KDE failed for evt {evt.event_idx}: {exc}; using zeros)")
                ana = np.zeros_like(hist)

        n_clean = sum(t == "clean" for t in tlab)
        n_merged = sum(t == "merged" for t in tlab)
        n_missed = sum(t == "missed" for t in tlab)
        n_fake = sum(r == "fake" for r in rlab)
        eff = (n_clean + n_merged) / max(len(truth), 1)
        print(
            f"\nEvent {evt.event_idx}: truth={len(truth)} peaks={len(peaks)} | "
            f"clean={n_clean} merged={n_merged} missed={n_missed} fake={n_fake} | "
            f"eff={eff:.3f}"
        )

        cases = {
            "merged": [truth[j] for j in range(len(truth)) if tlab[j] == "merged"],
            "missed": [truth[j] for j in range(len(truth)) if tlab[j] == "missed"],
            "fake": [peaks[i][0] for i in range(len(peaks)) if rlab[i] == "fake"],
        }
        evt_dir = os.path.join(args.output_dir, f"event{evt.event_idx:05d}")
        for kind in _KINDS:
            sel = cases[kind][: args.per_category]
            totals[kind] += len(sel)
            for k, zc in enumerate(sel):
                plot_vertex_zoom(
                    hist, ana, float(zc), peaks, evt.event_idx, k, "HL-LHC v4b",
                    os.path.join(evt_dir, kind), window_mm=args.window_mm,
                    tracks_z0=evt.z0, tracks_d0=evt.d0, tracks_d0_err=evt.d0_err,
                    all_truth_vertices=truth, match_window_mm=args.match_window_mm,
                    vertex_label=kind, truth_name="MC truth",
                )  # fmt: skip
            print(f"    {kind}: plotted {len(sel)}/{len(cases[kind])}")

    print(f"\nDone. Plots: {args.output_dir}")
    print("Totals plotted: " + ", ".join(f"{k}={totals[k]}" for k in _KINDS))


if __name__ == "__main__":
    main()
