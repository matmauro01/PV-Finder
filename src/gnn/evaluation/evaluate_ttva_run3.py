"""Truth-free TTVA evaluation on Run 3 real data.

Runs the full PV-Finder chain (T2KDE + K2H → 12k-bin histogram → peak
finding) on cached Run 3 events, associates tracks to the found peaks with
the GNN (MaxScore at a chosen threshold), and compares against AMVF's own
track-vertex associations (RecoVertex_assocTracks) without any truth:

- **Assignment agreement**: fraction of AMVF-assigned tracks whose GNN
  vertex maps (|Δz| matching) to their AMVF vertex, with a breakdown of
  the disagreements; reported for several matching windows.
- **Vertex multiplicity**: per-vertex assigned-track counts, GNN vs AMVF.
- **Unassigned-track fractions** for both methods.
- **Leading-vertex agreement**: does the max-sum-pT² GNN vertex map to the
  max-sum-pT² AMVF vertex (hard-scatter proxy)?
- **Score distributions** (all edges + max per track), optionally overlaid
  with the same model's scores on MC inference graphs (domain shift).

A built-in self-check runs the agreement machinery with AMVF playing both
roles, which must give agreement = 1.

Usage:
    python -u -m gnn.evaluation.evaluate_ttva_run3 \\
        --cache data/run3/cache_file3_2000ev_seed42.npz \\
        --assoc data/run3/assoc_cache_file3_2000ev_seed42.npz \\
        --t2kde-model model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth \\
        --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \\
        --gnn-weights model_weights/gnn_ttva_epoch100.pyt \\
        -t 0.98 -d 0 -o outputs/<date>_ttva_run3/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from gnn.data.graph_construction import create_inference_graph
from gnn.evaluation.classification import get_top_k_associations
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.data.run3_io import Run3Event
from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import (
    K2H_CONFIG,
    T2KDE_CONFIG,
    build_subevent_inputs,
    load_ckpt,
    run_inference,
)
from pv_finder.models.autoencoder_models import MaskedDNN, UNet_1000
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding import pv_locations_updated_res

MATCH_WINDOWS = (0.5, 1.0, 2.0)
HIST_BINS = np.linspace(0.0, 1.0, 201)
MULT_BINS = np.arange(0, 121)


# ---------------------------------------------------------------------------
# Per-event helpers
# ---------------------------------------------------------------------------
def gnn_assignments(
    scores: np.ndarray,
    edge_index: np.ndarray,
    n_tracks: int,
    threshold: float,
) -> np.ndarray:
    """Per-track assigned PV index from MaxScore selection (-1 = unassigned)."""
    selected = get_top_k_associations(scores, edge_index, k=1, threshold=threshold)
    assignment = np.full(n_tracks, -1, dtype=np.int64)
    assignment[edge_index[0][selected]] = edge_index[1][selected]
    return assignment


def amvf_assignments(
    assoc_flat: np.ndarray,
    assoc_counts: np.ndarray,
    vtx_mask: np.ndarray,
    n_tracks: int,
) -> np.ndarray:
    """Per-track AMVF vertex index among kept vertices (-1 = unassigned)."""
    assignment = np.full(n_tracks, -1, dtype=np.int64)
    start = 0
    kept = 0
    for vtx, count in enumerate(assoc_counts):
        tracks = assoc_flat[start : start + count]
        start += count
        if not vtx_mask[vtx]:
            continue
        valid = tracks[(tracks >= 0) & (tracks < n_tracks)]
        assignment[valid] = kept
        kept += 1
    return assignment


def match_vertices(pred_z: np.ndarray, amvf_z: np.ndarray, window: float) -> np.ndarray:
    """Greedy one-to-one |Δz| matching; returns per-pred AMVF index (-1=none)."""
    mapping = np.full(len(pred_z), -1, dtype=np.int64)
    if len(pred_z) == 0 or len(amvf_z) == 0:
        return mapping
    dz = np.abs(pred_z[:, None] - amvf_z[None, :])
    pairs = np.argwhere(dz < window)
    order = np.argsort(dz[pairs[:, 0], pairs[:, 1]], kind="stable")
    used_pred: set[int] = set()
    used_amvf: set[int] = set()
    for p, a in pairs[order]:
        if p in used_pred or a in used_amvf:
            continue
        mapping[p] = a
        used_pred.add(int(p))
        used_amvf.add(int(a))
    return mapping


def agreement_counts(
    gnn_asgn: np.ndarray,
    amvf_asgn: np.ndarray,
    peak_to_amvf: np.ndarray,
) -> dict[str, int]:
    """Track-level agreement breakdown over AMVF-assigned tracks."""
    amvf_assigned = amvf_asgn >= 0
    gnn_assigned = gnn_asgn >= 0
    both = amvf_assigned & gnn_assigned
    mapped = np.full(len(gnn_asgn), -1, dtype=np.int64)
    mapped[gnn_assigned] = peak_to_amvf[gnn_asgn[gnn_assigned]]
    agree = both & (mapped == amvf_asgn)
    return {
        "n_amvf_assigned": int(amvf_assigned.sum()),
        "n_both_assigned": int(both.sum()),
        "n_agree": int(agree.sum()),
        "n_gnn_unassigned": int((amvf_assigned & ~gnn_assigned).sum()),
        "n_gnn_vertex_unmatched": int((both & (mapped == -1)).sum()),
        "n_different_vertex": int((both & (mapped >= 0) & (mapped != amvf_asgn)).sum()),
    }


def leading_vertex(assignment: np.ndarray, pt: np.ndarray, n_vtx: int) -> int:
    """Index of the vertex with the largest sum pT² of assigned tracks."""
    if n_vtx == 0:
        return -1
    sums = np.zeros(n_vtx)
    valid = assignment >= 0
    np.add.at(sums, assignment[valid], pt[valid] ** 2)
    return int(np.argmax(sums))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(args: argparse.Namespace) -> None:  # noqa: PLR0915
    """Run the full Run 3 truth-free evaluation."""
    if args.device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    t2kde = MaskedDNN(**T2KDE_CONFIG)
    load_ckpt(args.t2kde_model, t2kde, device)
    k2h = UNet_1000(**K2H_CONFIG)
    load_ckpt(args.k2h_model, k2h, device)
    gnn = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    gnn.load_state_dict(torch.load(args.gnn_weights, map_location=device))
    gnn.to(device).eval()

    cache = np.load(args.cache, allow_pickle=True)
    assoc = np.load(args.assoc, allow_pickle=True)
    n_events = len(cache["RecoTrack_z0"])
    if args.nevents:
        n_events = min(n_events, args.nevents)

    windows_counts: dict[float, dict[str, int]] = {w: {} for w in MATCH_WINDOWS}
    self_check = {"n_amvf_assigned": 0, "n_agree": 0}
    hists = {
        "gnn_all_scores": np.zeros(len(HIST_BINS) - 1, dtype=np.int64),
        "gnn_max_scores": np.zeros(len(HIST_BINS) - 1, dtype=np.int64),
        "gnn_multiplicity": np.zeros(len(MULT_BINS) - 1, dtype=np.int64),
        "amvf_multiplicity": np.zeros(len(MULT_BINS) - 1, dtype=np.int64),
    }
    totals = {
        "n_events": 0,
        "n_skipped": 0,
        "n_tracks": 0,
        "n_gnn_assigned": 0,
        "n_amvf_assigned": 0,
        "n_peaks": 0,
        "n_amvf_vtx": 0,
        "n_lead_events": 0,
        "n_lead_agree": 0,
    }

    for i in tqdm(range(n_events), desc="Run3 events"):
        z0 = np.asarray(cache["RecoTrack_z0"][i], dtype=np.float32)
        n_tracks = len(z0)
        vtx_ntracks = np.asarray(cache["RecoVertex_nTracks"][i], dtype=np.float32)
        vtx_mask = vtx_ntracks >= 2
        if n_tracks == 0 or int(vtx_mask.sum()) == 0:
            totals["n_skipped"] += 1
            continue
        d0 = np.asarray(cache["RecoTrack_d0"][i], dtype=np.float32)
        d0e = np.asarray(cache["RecoTrack_ErrD0"][i], dtype=np.float32)
        z0e = np.asarray(cache["RecoTrack_ErrZ0"][i], dtype=np.float32)
        cov = np.asarray(cache["RecoTrack_ErrD0Z0"][i], dtype=np.float32)
        theta = np.asarray(cache["RecoTrack_theta"][i], dtype=np.float32)
        phi = np.asarray(cache["RecoTrack_phi"][i], dtype=np.float32)
        pt = np.asarray(cache["RecoTrack_pT"][i], dtype=np.float32)
        beam_z = float(np.atleast_1d(cache["BeamPosZ"][i])[0])
        amvf_z = np.asarray(cache["RecoVertex_z"][i], dtype=np.float32)[vtx_mask]
        amvf_z_corr = amvf_z - beam_z

        # PVF: tracks → histogram → peaks (track z0 frame)
        event = Run3Event(
            z0=z0, d0=d0, d0_err=d0e, z0_err=z0e, d0_z0_cov=cov,
            amvf_z=amvf_z, amvf_ntrks=vtx_ntracks[vtx_mask], beam_z=beam_z,
            mu=None, event_idx=i, n_tracks=n_tracks,
        )  # fmt: skip
        hist = run_inference(build_subevent_inputs(event), device, t2kde=t2kde, k2h=k2h)
        pred_z, pred_heights, _, pred_sigmas = pv_locations_updated_res(
            hist,
            args.pvf_threshold,
            args.pvf_integral_threshold,
            args.pvf_min_width,
            args.pvf_min_height,
        )
        if len(pred_z) == 0:
            totals["n_skipped"] += 1
            continue

        # GNN scores on the (track, peak) bipartite graph
        tracks_stack = np.stack(
            [d0, z0, d0e, z0e, cov, theta, phi, pt / PT_SCALE], axis=1
        ).astype(np.float32)
        graph = create_inference_graph(
            z0, d0, z0e, d0e, tracks_stack, pred_z, pred_heights, pred_sigmas
        )
        graph["track"].batch = torch.zeros(graph["track"].num_nodes, dtype=torch.long)
        graph["pv"].batch = torch.zeros(graph["pv"].num_nodes, dtype=torch.long)
        with torch.no_grad():
            logits = gnn(graph.to(device))
        scores = torch.sigmoid(logits).cpu().numpy()
        edge_index = graph[("track", "to", "pv")].edge_index.cpu().numpy()

        gnn_asgn = gnn_assignments(scores, edge_index, n_tracks, args.threshold)
        amvf_asgn = amvf_assignments(
            assoc["assoc_flat"][i], assoc["assoc_counts"][i], vtx_mask, n_tracks
        )

        # Agreement at each matching window
        for w in MATCH_WINDOWS:
            peak_to_amvf = match_vertices(pred_z, amvf_z_corr, w)
            counts = agreement_counts(gnn_asgn, amvf_asgn, peak_to_amvf)
            for key, val in counts.items():
                windows_counts[w][key] = windows_counts[w].get(key, 0) + val
            if w == args.match_window:
                lead_gnn = leading_vertex(gnn_asgn, pt, len(pred_z))
                lead_amvf = leading_vertex(amvf_asgn, pt, len(amvf_z))
                if lead_gnn >= 0 and lead_amvf >= 0:
                    totals["n_lead_events"] += 1
                    totals["n_lead_agree"] += int(peak_to_amvf[lead_gnn] == lead_amvf)

        # Self-check: AMVF against itself must agree 100%
        identity = match_vertices(amvf_z_corr, amvf_z_corr, args.match_window)
        sc = agreement_counts(amvf_asgn, amvf_asgn, identity)
        self_check["n_amvf_assigned"] += sc["n_amvf_assigned"]
        self_check["n_agree"] += sc["n_agree"]

        # Histograms
        hists["gnn_all_scores"] += np.histogram(scores, bins=HIST_BINS)[0]
        max_per_track = np.full(n_tracks, -1.0)
        np.maximum.at(max_per_track, edge_index[0], scores)
        hists["gnn_max_scores"] += np.histogram(
            max_per_track[max_per_track >= 0], bins=HIST_BINS
        )[0]
        gnn_mult = np.bincount(gnn_asgn[gnn_asgn >= 0], minlength=len(pred_z))
        amvf_mult = np.bincount(amvf_asgn[amvf_asgn >= 0], minlength=len(amvf_z))
        hists["gnn_multiplicity"] += np.histogram(gnn_mult, bins=MULT_BINS)[0]
        hists["amvf_multiplicity"] += np.histogram(amvf_mult, bins=MULT_BINS)[0]

        totals["n_events"] += 1
        totals["n_tracks"] += n_tracks
        totals["n_gnn_assigned"] += int((gnn_asgn >= 0).sum())
        totals["n_amvf_assigned"] += int((amvf_asgn >= 0).sum())
        totals["n_peaks"] += len(pred_z)
        totals["n_amvf_vtx"] += len(amvf_z)

    # Optional MC overlay: same GNN on MC inference graphs
    mc_hists = {}
    if args.mc_graphs:
        mc_hists = mc_score_histograms(args.mc_graphs, gnn, device, args.mc_max_events)

    summarize_and_save(args, totals, windows_counts, self_check, hists, mc_hists)


def mc_score_histograms(
    graphs_path: str,
    gnn: TTVAGATModel,
    device: torch.device,
    max_events: int | None,
) -> dict[str, np.ndarray]:
    """Score MC inference graphs with the same model for the overlay plots."""
    graphs = torch.load(graphs_path, weights_only=False)
    if max_events:
        graphs = graphs[:max_events]
    all_hist = np.zeros(len(HIST_BINS) - 1, dtype=np.int64)
    max_hist = np.zeros(len(HIST_BINS) - 1, dtype=np.int64)
    for graph in tqdm(graphs, desc="MC graphs"):
        if graph["pv"].num_nodes == 0:
            continue
        with torch.no_grad():
            logits = gnn(graph.to(device))
        scores = torch.sigmoid(logits).cpu().numpy()
        edge_index = graph[("track", "to", "pv")].edge_index.cpu().numpy()
        all_hist += np.histogram(scores, bins=HIST_BINS)[0]
        max_per_track = np.full(graph["track"].num_nodes, -1.0)
        np.maximum.at(max_per_track, edge_index[0], scores)
        max_hist += np.histogram(max_per_track[max_per_track >= 0], bins=HIST_BINS)[0]
    return {"mc_all_scores": all_hist, "mc_max_scores": max_hist}


def summarize_and_save(
    args: argparse.Namespace,
    totals: dict[str, int],
    windows_counts: dict[float, dict[str, int]],
    self_check: dict[str, int],
    hists: dict[str, np.ndarray],
    mc_hists: dict[str, np.ndarray],
) -> None:
    """Aggregate rates, print, and write summary.json + histograms npz."""
    agreement = {}
    for w, counts in windows_counts.items():
        n_ref = counts.get("n_amvf_assigned", 0)
        n_both = counts.get("n_both_assigned", 0)
        agreement[str(w)] = {
            **counts,
            "agreement_vs_amvf_assigned": counts.get("n_agree", 0) / n_ref
            if n_ref
            else 0.0,
            "agreement_vs_both_assigned": counts.get("n_agree", 0) / n_both
            if n_both
            else 0.0,
        }
    self_check_rate = (
        self_check["n_agree"] / self_check["n_amvf_assigned"]
        if self_check["n_amvf_assigned"]
        else 0.0
    )
    summary: dict[str, Any] = {
        "config": {
            "threshold": args.threshold,
            "match_window": args.match_window,
            "pvf_threshold": args.pvf_threshold,
            "pvf_integral_threshold": args.pvf_integral_threshold,
            "pvf_min_width": args.pvf_min_width,
            "pvf_min_height": args.pvf_min_height,
            "gnn_weights": args.gnn_weights,
            "t2kde_model": args.t2kde_model,
            "k2h_model": args.k2h_model,
        },
        "totals": totals,
        "rates": {
            "gnn_assigned_fraction": totals["n_gnn_assigned"] / totals["n_tracks"],
            "amvf_assigned_fraction": totals["n_amvf_assigned"] / totals["n_tracks"],
            "peaks_per_event": totals["n_peaks"] / totals["n_events"],
            "amvf_vtx_per_event": totals["n_amvf_vtx"] / totals["n_events"],
            "leading_vertex_agreement": totals["n_lead_agree"]
            / totals["n_lead_events"],
        },
        "agreement_by_window": agreement,
        "self_check_agreement": self_check_rate,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_dir / "run3_hists.npz",
        score_bins=HIST_BINS,
        mult_bins=MULT_BINS,
        **hists,
        **mc_hists,
    )

    print(json.dumps(summary["rates"], indent=2))
    main_w = agreement[str(args.match_window)]
    print(
        f"Agreement (|dz|<{args.match_window} mm): "
        f"{main_w['agreement_vs_amvf_assigned']:.4f} of AMVF-assigned, "
        f"{main_w['agreement_vs_both_assigned']:.4f} of both-assigned"
    )
    print(f"Self-check (AMVF vs AMVF): {self_check_rate:.4f} (must be 1.0)")
    print(f"Saved to {out_dir}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Truth-free TTVA eval on Run 3 data")
    parser.add_argument("--cache", required=True, type=str)
    parser.add_argument("--assoc", required=True, type=str)
    parser.add_argument("--t2kde-model", required=True, type=str)
    parser.add_argument("--k2h-model", required=True, type=str)
    parser.add_argument("--gnn-weights", required=True, type=str)
    parser.add_argument("-t", "--threshold", required=True, type=float)
    parser.add_argument("-d", "--device-id", required=True, type=int)
    parser.add_argument("-o", "--output-dir", required=True, type=str)
    parser.add_argument("-n", "--nevents", default=None, type=int)
    parser.add_argument("--match-window", default=1.0, type=float)
    parser.add_argument("--pvf-threshold", default=1e-2, type=float)
    parser.add_argument("--pvf-integral-threshold", default=0.5, type=float)
    parser.add_argument("--pvf-min-width", default=3, type=int)
    parser.add_argument("--pvf-min-height", default=0.0, type=float)
    parser.add_argument("--mc-graphs", default=None, type=str)
    parser.add_argument("--mc-max-events", default=None, type=int)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    evaluate(_parse_args())


if __name__ == "__main__":
    main()
