"""Threshold scan for the TTVA GNN: vertex- and track-level metrics vs cut.

Runs the GNN ONCE per event to cache edge scores, then re-applies the edge
selection over a grid of thresholds, classifying vertices with the shared
``classify_assignments`` core (identical to evaluate_ttva / evaluate_amvf,
so all numbers are apples-to-apples) and computing track-level assignment
efficiency/purity. AMVF's own associations are evaluated once with the same
code as a model-free reference. Picks a recommended working point t*.

Track-level definition (MaxScore): a track is *correctly assigned* iff the
dominant truth PV of its assigned reco PV equals the track's own truth PV.
Efficiency divides by all truth-associated tracks; purity by the
truth-associated tracks the method assigned to some vertex.

Usage:
    python -u -m gnn.evaluation.threshold_scan \\
        -r outputs/07_12_2026_ttva_reproduction/regen/graphs_pvf_e400_regen.pt \\
        -f /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \\
        -i configs/qibin_test_main_indices_v2.p \\
        -w model_weights/gnn_ttva_epoch100.pyt \\
        -d 0 -o outputs/<date>_ttva_metrics/ \\
        --reference-rows outputs/07_12_2026_ttva_reproduction/total_results_MaxScore.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.data.graph_construction import load_event_indices
from gnn.evaluation.classification import (
    build_truth_adjacency,
    classify_assignments,
    get_top_k_associations,
)
from gnn.evaluation.regression_guard import check_regression
from gnn.models.ttva_gat import TTVAGATModel

# Coarse 0.05 grid plus a fine tail: the GNN's max-scores are strongly
# saturated near 1, so the metrics only move above t ~ 0.9.
THRESHOLD_GRID = np.concatenate(
    [
        np.round(np.arange(1, 20) * 0.05, 2),
        np.array([0.96, 0.97, 0.98, 0.99, 0.995, 0.999]),
    ]
)
THRESHOLD_MODE_POINTS = (0.3, 0.5, 0.7, 0.9)
TRACK_COUNT_KEYS = ("n_assigned", "n_assigned_truth", "n_correct")


# ---------------------------------------------------------------------------
# Stage 1: score caching (the only GPU part)
# ---------------------------------------------------------------------------
def cache_scores(
    graphs_path: str | Path,
    weights_path: str | Path,
    device: torch.device,
    max_events: int | None = None,
) -> list[dict[str, Any]]:
    """Run the GNN once per graph; keep (edge_index, sigmoid scores) per event."""
    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    graphs: list[HeteroData] = torch.load(graphs_path, weights_only=False)
    if max_events is not None:
        graphs = graphs[:max_events]

    cache: list[dict[str, Any]] = []
    for graph in tqdm(graphs, desc="scoring"):
        entry: dict[str, Any] = {
            "n_pvs": int(graph["pv"].num_nodes),
            "n_tracks": int(graph["track"].num_nodes),
        }
        if entry["n_pvs"] > 0:
            with torch.no_grad():
                logits = model(graph.to(device))
            entry["scores"] = torch.sigmoid(logits).cpu().numpy()
            entry["edge_index"] = graph[("track", "to", "pv")].edge_index.cpu().numpy()
        cache.append(entry)
    return cache


# ---------------------------------------------------------------------------
# Truth loading (once, from the event-keyed HDF5)
# ---------------------------------------------------------------------------
def load_truth(h5_path: str | Path, event_keys: list[str]) -> list[dict[str, Any]]:
    """Per-event truth adjacency, pT, and per-track truth-PV map."""
    truth: list[dict[str, Any]] = []
    with h5py.File(str(h5_path), "r") as f:
        pt = f["recoTrk_pt"]
        pv_loc_z = f["pv_loc_z"]
        pv_assoctracks = f["pv_assoc_tracks"]
        pv_ntracks = f["pv_ntracks"]
        for event_key in tqdm(event_keys, desc="loading truth"):
            pt_event = pt[event_key][:]
            ntracks_event = pv_ntracks[event_key][:]
            tti, tpi = build_truth_adjacency(
                pv_loc_z[event_key][:], ntracks_event, pv_assoctracks[event_key][:]
            )
            track_truth = np.full(len(pt_event), -1, dtype=np.int64)
            track_truth[tti.astype(np.int64)] = tpi.astype(np.int64)
            truth.append(
                {
                    "pt": pt_event,
                    "tti": tti,
                    "tpi": tpi,
                    "truth_pvs_count": int((ntracks_event >= 2).sum()),
                    "track_truth": track_truth,
                }
            )
    return truth


# ---------------------------------------------------------------------------
# Selection → per-PV track lists → classification + track-level counts
# ---------------------------------------------------------------------------
def matched_tracks_from_selection(
    edge_index: np.ndarray, selected: np.ndarray, n_pvs: int
) -> list[np.ndarray]:
    """Group selected edges by PV; unique sorted track indices per PV."""
    sel_tracks = edge_index[0][selected]
    sel_pvs = edge_index[1][selected]
    order = np.argsort(sel_pvs, kind="stable")
    sel_tracks = sel_tracks[order]
    sel_pvs = sel_pvs[order]
    boundaries = np.searchsorted(sel_pvs, np.arange(n_pvs + 1))
    return [
        np.unique(sel_tracks[boundaries[p] : boundaries[p + 1]]) for p in range(n_pvs)
    ]


def _dominant_index(primary: str | None) -> int:
    """Parse 'Truth_PV_<i>' → i; Fake/None → -1."""
    if primary is None or primary == "Fake":
        return -1
    return int(primary.rsplit("_", 1)[1])


def track_assignment_counts(
    matched_tracks_per_pv: list[np.ndarray],
    info: list[dict[str, Any]],
    track_truth: np.ndarray,
) -> dict[str, int]:
    """Count correctly assigned tracks given per-PV assignments + dominants."""
    counts = dict.fromkeys(TRACK_COUNT_KEYS, 0)
    for tracks, pv_info in zip(matched_tracks_per_pv, info):
        dominant = _dominant_index(pv_info["primary_truth_pv"])
        truth_of_tracks = track_truth[tracks]
        has_truth = truth_of_tracks >= 0
        counts["n_assigned"] += len(tracks)
        counts["n_assigned_truth"] += int(has_truth.sum())
        if dominant >= 0:
            counts["n_correct"] += int((truth_of_tracks == dominant).sum())
    return counts


def evaluate_selection(
    cache_ev: dict[str, Any],
    truth_ev: dict[str, Any],
    method: str,
    threshold: float,
) -> tuple[list[int], list[dict[str, Any]], dict[str, int]]:
    """Apply one edge selection to one event; classify and count."""
    if cache_ev["n_pvs"] == 0:
        return (
            [0, 0, 0, 0, 0, truth_ev["truth_pvs_count"]],
            [],
            dict.fromkeys(TRACK_COUNT_KEYS, 0),
        )
    if method == "MaxScore":
        selected = get_top_k_associations(
            cache_ev["scores"], cache_ev["edge_index"], k=1, threshold=threshold
        )
    else:
        selected = cache_ev["scores"] > threshold
    matched = matched_tracks_from_selection(
        cache_ev["edge_index"], selected, cache_ev["n_pvs"]
    )
    results, info = classify_assignments(
        matched,
        truth_ev["pt"],
        truth_ev["tti"],
        truth_ev["tpi"],
        truth_ev["truth_pvs_count"],
    )
    counts = track_assignment_counts(matched, info, truth_ev["track_truth"])
    return results, info, counts


def scan_point(
    cache: list[dict[str, Any]],
    truth: list[dict[str, Any]],
    method: str,
    threshold: float,
) -> tuple[dict[str, Any], list[list[int]]]:
    """Evaluate every event at one (method, threshold) point; aggregate."""
    totals = np.zeros(6, dtype=np.int64)
    track_totals = dict.fromkeys(TRACK_COUNT_KEYS, 0)
    rows: list[list[int]] = []
    for cache_ev, truth_ev in zip(cache, truth):
        results, _info, counts = evaluate_selection(
            cache_ev, truth_ev, method, threshold
        )
        totals += np.array(results, dtype=np.int64)
        rows.append(results)
        for key in TRACK_COUNT_KEYS:
            track_totals[key] += counts[key]

    clean, merged, split, fake, n_reco, n_truth = (int(x) for x in totals)
    n_truth_tracks = int(sum((t["track_truth"] >= 0).sum() for t in truth))
    n_tracks = int(sum(t["track_truth"].size for t in truth))
    point: dict[str, Any] = {
        "method": method,
        "threshold": float(threshold),
        "clean": clean,
        "merged": merged,
        "split": split,
        "fake": fake,
        "n_reco": n_reco,
        "n_truth": n_truth,
        **{k: int(v) for k, v in track_totals.items()},
    }
    if n_reco > 0:
        point["rates"] = {
            "clean": clean / n_reco,
            "merged": merged / n_reco,
            "split": split / n_reco,
            "fake": fake / n_reco,
        }
        point["clean_per_truth"] = clean / n_truth
    if method == "MaxScore":
        eff = track_totals["n_correct"] / n_truth_tracks if n_truth_tracks else 0.0
        pur = (
            track_totals["n_correct"] / track_totals["n_assigned_truth"]
            if track_totals["n_assigned_truth"]
            else 0.0
        )
        point["track"] = {
            "efficiency": eff,
            "purity": pur,
            "f1": 2 * eff * pur / (eff + pur) if eff + pur else 0.0,
            "assigned_fraction": track_totals["n_assigned"] / n_tracks,
        }
    return point, rows


# ---------------------------------------------------------------------------
# AMVF reference (model-free, threshold-independent)
# ---------------------------------------------------------------------------
def amvf_reference(
    h5_path: str | Path, event_keys: list[str], truth: list[dict[str, Any]]
) -> dict[str, Any]:
    """Classify AMVF's own associations + track-level counts, same code path."""
    totals = np.zeros(6, dtype=np.int64)
    track_totals = dict.fromkeys(TRACK_COUNT_KEYS, 0)
    with h5py.File(str(h5_path), "r") as f:
        amvf_ntracks = f["reco_pv_ntracks"]
        amvf_assoctracks = f["reco_pv_assoc_tracks"]
        for event_key, truth_ev in zip(tqdm(event_keys, desc="AMVF"), truth):
            amvf_nt = amvf_ntracks[event_key][:].astype(int)
            amvf_at = amvf_assoctracks[event_key][:].astype(int)
            matched: list[np.ndarray] = []
            counter = 0
            for n in amvf_nt:
                matched.append(np.unique(amvf_at[counter : counter + n]))
                counter += n
            results, info = classify_assignments(
                matched,
                truth_ev["pt"],
                truth_ev["tti"],
                truth_ev["tpi"],
                truth_ev["truth_pvs_count"],
            )
            totals += np.array(results, dtype=np.int64)
            counts = track_assignment_counts(matched, info, truth_ev["track_truth"])
            for key in TRACK_COUNT_KEYS:
                track_totals[key] += counts[key]

    clean, merged, split, fake, n_reco, n_truth = (int(x) for x in totals)
    n_truth_tracks = int(sum((t["track_truth"] >= 0).sum() for t in truth))
    n_tracks = int(sum(t["track_truth"].size for t in truth))
    eff = track_totals["n_correct"] / n_truth_tracks
    pur = track_totals["n_correct"] / track_totals["n_assigned_truth"]
    return {
        "clean": clean,
        "merged": merged,
        "split": split,
        "fake": fake,
        "n_reco": n_reco,
        "n_truth": n_truth,
        "rates": {
            "clean": clean / n_reco,
            "merged": merged / n_reco,
            "split": split / n_reco,
            "fake": fake / n_reco,
        },
        "clean_per_truth": clean / n_truth,
        "track": {
            "efficiency": eff,
            "purity": pur,
            "f1": 2 * eff * pur / (eff + pur),
            "assigned_fraction": track_totals["n_assigned"] / n_tracks,
        },
        **{k: int(v) for k, v in track_totals.items()},
    }


# ---------------------------------------------------------------------------
# Working point + regression guard
# ---------------------------------------------------------------------------
def choose_working_point(
    points: list[dict[str, Any]], fake_budget: float
) -> dict[str, Any]:
    """t* = max clean-vertex efficiency subject to fake rate <= budget."""
    feasible = [p for p in points if p.get("rates", {}).get("fake", 1.0) <= fake_budget]
    pool = feasible if feasible else points
    best = max(pool, key=lambda p: p.get("clean_per_truth", 0.0))
    best_f1 = max(points, key=lambda p: p.get("track", {}).get("f1", 0.0))
    return {
        "t_star": best["threshold"],
        "criterion": f"max clean/truth with fake rate <= {fake_budget}",
        "feasible": bool(feasible),
        "clean_per_truth": best["clean_per_truth"],
        "fake_rate": best["rates"]["fake"],
        "t_max_track_f1": best_f1["threshold"],
        "max_track_f1": best_f1.get("track", {}).get("f1", 0.0),
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="TTVA GNN threshold scan")
    parser.add_argument("-r", "--reco-graph-path", required=True, type=str)
    parser.add_argument("-f", "--filepath", required=True, type=str)
    parser.add_argument("-i", "--indices", required=True, type=str)
    parser.add_argument("-w", "--model-weights", required=True, type=str)
    parser.add_argument("-d", "--device-id", required=True, type=int)
    parser.add_argument("-o", "--output-dir", required=True, type=str)
    parser.add_argument("-n", "--max-events", default=None, type=int)
    parser.add_argument(
        "--reference-rows",
        default=None,
        type=str,
        help="Baseline total_results_MaxScore.npy for the t=0.5 bit-exact guard",
    )
    parser.add_argument("--fake-budget", default=0.01, type=float)
    return parser.parse_args()


def main() -> None:  # noqa: PLR0915
    """CLI entry point."""
    args = _parse_args()
    if args.device_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    indices = load_event_indices(args.indices)
    event_keys = [f"Event{int(idx)}" for idx in indices]
    if args.max_events is not None:
        event_keys = event_keys[: args.max_events]

    cache = cache_scores(
        args.reco_graph_path, args.model_weights, device, args.max_events
    )
    if len(cache) != len(event_keys):
        msg = f"Graphs ({len(cache)}) vs event keys ({len(event_keys)}) mismatch"
        raise ValueError(msg)
    truth = load_truth(args.filepath, event_keys)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    regression_ok = None
    maxscore_points: list[dict[str, Any]] = []
    for t in tqdm(THRESHOLD_GRID, desc="MaxScore scan"):
        point, rows = scan_point(cache, truth, "MaxScore", t)
        maxscore_points.append(point)
        if args.reference_rows and abs(t - 0.5) < 1e-9:
            regression_ok = check_regression(rows, args.reference_rows, cache, t)

    threshold_points = [
        scan_point(cache, truth, "Threshold", t)[0] for t in THRESHOLD_MODE_POINTS
    ]
    amvf = amvf_reference(args.filepath, event_keys, truth)
    working_point = choose_working_point(maxscore_points, args.fake_budget)

    results = {
        "n_events": len(event_keys),
        "weights": args.model_weights,
        "graphs": args.reco_graph_path,
        "maxscore": maxscore_points,
        "threshold_mode": threshold_points,
        "amvf": amvf,
        "working_point": working_point,
        "regression_ok": regression_ok,
    }
    with open(out_dir / "scan_results.json", "w") as f:
        json.dump(results, f, indent=2)

    for p in maxscore_points:
        track = p.get("track", {})
        print(
            f"t={p['threshold']:.2f} clean/truth={p.get('clean_per_truth', 0):.4f} "
            f"fake={p.get('rates', {}).get('fake', 0):.4f} "
            f"trk eff={track.get('efficiency', 0):.4f} "
            f"pur={track.get('purity', 0):.4f}"
        )
    print(f"AMVF: clean/truth={amvf['clean_per_truth']:.4f} track={amvf['track']}")
    print(f"Working point: {working_point}")

    # Full per-event outputs at t* for downstream (publication) plots
    t_star = working_point["t_star"]
    star_rows, star_info = [], []
    for cache_ev, truth_ev in zip(cache, truth):
        results_ev, info_ev, _ = evaluate_selection(
            cache_ev, truth_ev, "MaxScore", t_star
        )
        star_rows.append(results_ev)
        star_info.append(info_ev)
    star_dir = out_dir / "t_star"
    star_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        star_dir / "total_results_MaxScore.npy",
        np.array(star_rows, dtype=object),
        allow_pickle=True,
    )
    np.save(
        star_dir / "total_reco_pv_info_list_MaxScore.npy",
        np.array(star_info, dtype=object),
        allow_pickle=True,
    )
    print(f"Saved scan_results.json and t*={t_star} outputs to {out_dir}")


if __name__ == "__main__":
    main()
