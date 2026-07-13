"""Full-chain PU200 evaluation graphs: PVF e2e → peaks → kNN inference graphs.

End-to-end "PV-Finder + GNN at HL-LHC pileup" evaluation input: runs the
production PVF model (TracksToHist_v2, e.g. v4b epoch 3) on PU200 ROOT test
events, finds peaks with the production operating point, and builds kNN
inference graphs whose PV nodes are the found peaks. Each graph carries
``track.truth_pv`` from the ROOT truth so gnn.evaluation.evaluate_ttva_graphs
can classify Clean/Merged/Split/Fake without a truth HDF5.

Also produced in the same ROOT pass:
- AMVF baseline rows (RecoVertex_assocTracks through the identical
  classify_assignments core) → amvf_results.npy + summary in chain_info.json
- Per-stage wall-time statistics (PVF inference / peak finding / graph
  construction) → chain_info.json

Usage:
    python -u -m gnn.data.pu200_chain_graphs \\
        --root data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root \\
        --pvf-weights model_weights/hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3_fullstate.pth \\
        --entry-start 28500 --max-events 1500 -d 1 \\
        --output data/run4/ttva_graphs/pu200_chain_v4b_k20_test.pt \\
        --output-dir outputs/<date>_ttva_chain/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from tqdm import tqdm

from gnn.data.graph_construction import create_inference_graph
from gnn.evaluation.classification import classify_assignments
from pv_finder.data.run3_io import Run3Event
from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import (
    T2KDE_CONFIG,
    build_subevent_inputs,
    load_ckpt,
    run_inference,
)
from pv_finder.models.autoencoder_models import MaskedDNN
from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding import pv_locations_updated_res

BRANCHES = [
    "RecoTrack_z0",
    "RecoTrack_d0",
    "RecoTrack_ErrD0",
    "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0",
    "RecoTrack_theta",
    "RecoTrack_phi",
    "RecoTrack_pT",
    "TruthVertex_nTracks",
    "TruthVertex_assocTracks",
    "RecoVertex_nTracks",
    "RecoVertex_assocTracks",
]


def build_pvf_model(
    weights_path: str,
    device: torch.device,
    unet_channels: int,
    latent_channels: int,
    hidden_nodes: list[int],
) -> torch.nn.Module:
    """Construct the production e2e PVF model and load its checkpoint."""
    t2kde_cfg = dict(
        T2KDE_CONFIG,
        hidden_nodes=hidden_nodes,
        output_size=1000 * latent_channels,
    )
    model = TracksToHist_v2(
        MaskedDNN(**t2kde_cfg),
        UNet_1000_v2(n=unet_channels, n_features=latent_channels, dropout_p=0.0),
    )
    load_ckpt(weights_path, model, device)
    return model


def truth_arrays(event: dict, n_tracks: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Flat truth adjacency (tti, tpi) + nTrk>=2 truth-PV count from ROOT."""
    pv_ntracks = ak.to_numpy(event["TruthVertex_nTracks"]).astype(np.int64)
    tti = ak.to_numpy(ak.flatten(event["TruthVertex_assocTracks"])).astype(np.int64)
    tpi = np.repeat(np.arange(len(pv_ntracks)), pv_ntracks)
    valid = (tti >= 0) & (tti < n_tracks)
    return (
        tti[valid].astype(np.float64),
        tpi[valid].astype(np.float64),
        int((pv_ntracks >= 2).sum()),
    )


def amvf_matched_lists(event: dict, n_tracks: int) -> list[np.ndarray]:
    """Per-AMVF-vertex unique track lists from RecoVertex_assocTracks."""
    matched = []
    for vertex_tracks in event["RecoVertex_assocTracks"]:
        tracks = ak.to_numpy(vertex_tracks).astype(np.int64)
        matched.append(np.unique(tracks[(tracks >= 0) & (tracks < n_tracks)]))
    return matched


def main() -> None:  # noqa: PLR0915
    """CLI entry point."""
    args = _parse_args()
    device = torch.device(
        f"cuda:{args.device_id}"
        if args.device_id >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    pvf = build_pvf_model(
        args.pvf_weights, device, args.unet_channels, args.latent_channels,
        [args.hidden_nodes] * 5,
    )  # fmt: skip

    tree = uproot.open(args.root)["PVFinderData"]
    entry_stop = min(args.entry_start + args.max_events, tree.num_entries)
    print(f"Entries [{args.entry_start}, {entry_stop}) of {tree.num_entries}")

    graphs = []
    amvf_rows = []
    amvf_totals = np.zeros(6, dtype=np.int64)
    timing: dict[str, list[float]] = {"pvf_ms": [], "peak_ms": [], "graph_ms": []}
    n_peaks_total = 0

    progress = tqdm(total=entry_stop - args.entry_start, desc="events")
    for chunk in tree.iterate(
        BRANCHES, step_size=200, entry_start=args.entry_start, entry_stop=entry_stop
    ):
        for event in chunk:
            z0 = ak.to_numpy(event["RecoTrack_z0"]).astype(np.float32)
            d0 = ak.to_numpy(event["RecoTrack_d0"]).astype(np.float32)
            err_d0 = ak.to_numpy(event["RecoTrack_ErrD0"]).astype(np.float32)
            err_z0 = ak.to_numpy(event["RecoTrack_ErrZ0"]).astype(np.float32)
            cov = ak.to_numpy(event["RecoTrack_ErrD0Z0"]).astype(np.float32)
            theta = ak.to_numpy(event["RecoTrack_theta"]).astype(np.float32)
            phi = ak.to_numpy(event["RecoTrack_phi"]).astype(np.float32)
            pt = ak.to_numpy(event["RecoTrack_pT"]).astype(np.float32)
            n_tracks = len(z0)

            tti, tpi, truth_count = truth_arrays(event, n_tracks)
            track_truth = np.full(n_tracks, -1, dtype=np.int64)
            track_truth[tti.astype(np.int64)] = tpi.astype(np.int64)

            # --- PVF: tracks → 12k-bin histogram (timed, GPU-synced) ---
            run3_event = Run3Event(
                z0=z0, d0=d0, d0_err=err_d0, z0_err=err_z0, d0_z0_cov=cov,
                amvf_z=np.array([]), amvf_ntrks=np.array([]), beam_z=0.0,
                mu=None, event_idx=0, n_tracks=n_tracks,
            )  # fmt: skip
            t0 = time.perf_counter()
            hist = run_inference(build_subevent_inputs(run3_event), device, e2e=pvf)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            pred_z, pred_heights, _, pred_sigmas = pv_locations_updated_res(
                hist,
                args.peak_threshold,
                args.integral_threshold,
                args.min_width,
                args.min_height,
            )
            t2 = time.perf_counter()

            # --- Graph: peaks as PV nodes, kNN edges, truth on tracks ---
            tracks_stack = np.stack(
                [d0, z0, err_d0, err_z0, cov, theta, phi, pt / PT_SCALE], axis=1
            ).astype(np.float32)
            graph = create_inference_graph(
                z0, d0, err_z0, err_d0, tracks_stack,
                pred_z, pred_heights, pred_sigmas, knn=args.knn,
            )  # fmt: skip
            graph["track"].truth_pv = torch.from_numpy(track_truth)
            t3 = time.perf_counter()

            timing["pvf_ms"].append(1e3 * (t1 - t0))
            timing["peak_ms"].append(1e3 * (t2 - t1))
            timing["graph_ms"].append(1e3 * (t3 - t2))
            n_peaks_total += len(pred_z)
            graphs.append(graph)

            # --- AMVF baseline through the identical classifier ---
            rows, _ = classify_assignments(
                amvf_matched_lists(event, n_tracks), pt, tti, tpi, truth_count
            )
            amvf_rows.append(rows)
            amvf_totals += np.array(rows, dtype=np.int64)
            progress.update(1)
    progress.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, out_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        out_dir / "amvf_results.npy",
        np.array(amvf_rows, dtype=object),
        allow_pickle=True,
    )

    clean, merged, split, fake, n_reco, n_truth = (int(x) for x in amvf_totals)
    info = {
        "config": vars(args),
        "n_events": len(graphs),
        "peaks_per_event": n_peaks_total / len(graphs),
        "amvf": {
            "clean": clean,
            "merged": merged,
            "split": split,
            "fake": fake,
            "n_reco": n_reco,
            "n_truth": n_truth,
            "clean_rate": clean / n_reco,
            "clean_per_truth": clean / n_truth,
        },  # fmt: skip
        "timing_ms": {
            key: {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90)),
            }
            for key, vals in timing.items()
        },
    }
    with open(out_dir / "chain_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Saved {len(graphs)} chain graphs to {out_path}")
    print(f"Peaks/event: {info['peaks_per_event']:.2f}")
    print(f"AMVF: clean_rate {clean / n_reco:.4f}, clean/truth {clean / n_truth:.4f}")
    print(
        f"Timing (median ms): {[(k, round(v['median'], 2)) for k, v in info['timing_ms'].items()]}"
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build full-chain PU200 inference graphs (PVF peaks + truth)"
    )
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--pvf-weights", required=True, type=str)
    parser.add_argument("--output", required=True, type=str, help="Graphs .pt")
    parser.add_argument("--output-dir", required=True, type=str, help="Aux outputs")
    parser.add_argument("--entry-start", default=28500, type=int)
    parser.add_argument("--max-events", default=1500, type=int)
    parser.add_argument("--knn", default=20, type=int)
    parser.add_argument("-d", "--device-id", default=0, type=int)
    # Production PVF operating point (June 2026 v4b evals)
    parser.add_argument("--peak-threshold", default=1e-2, type=float)
    parser.add_argument("--integral-threshold", default=0.40, type=float)
    parser.add_argument("--min-width", default=3, type=int)
    parser.add_argument("--min-height", default=0.03, type=float)
    # v4b architecture
    parser.add_argument("--unet-channels", default=280, type=int)
    parser.add_argument("--latent-channels", default=4, type=int)
    parser.add_argument("--hidden-nodes", default=128, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
