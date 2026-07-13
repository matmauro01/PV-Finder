"""Evaluate GNN TTVA on graphs that carry their own truth labels.

For samples without an event-keyed truth HDF5 (e.g. HL-LHC PU200 graphs
built by gnn.data.root_to_graphs), truth comes from the graph itself:
data['track'].truth_pv (per-track true PV index, -1 = none), with the
y=1 edges as fallback for older graph files.

Runs the same MaxScore/Threshold edge selection and the shared
classify_assignments core as evaluate_ttva, and additionally reports
edge-level metrics (AUC, per-track top-1 accuracy).

Usage:
    python -m gnn.evaluation.evaluate_ttva_graphs \\
        -r data/run4/ttva_graphs/pu200_truth_k20_30k.pt \\
        -w model_weights/gnn_ttva_epoch100.pyt \\
        -e MaxScore -t 0.5 -d 0 \\
        --first-event 28500 \\
        -o outputs/<date>/pu200_zeroshot/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.evaluation.classification import classify_assignments, get_top_k_associations
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.utils.constants import GNN_SCORE_THRESHOLD


def _truth_from_graph(graph: HeteroData) -> tuple[np.ndarray, np.ndarray]:
    """Per-track truth PV as flat (track_indices, pv_indices) pairs."""
    if hasattr(graph["track"], "truth_pv"):
        truth_pv = graph["track"].truth_pv.cpu().numpy()
        tracks = np.nonzero(truth_pv >= 0)[0]
        return tracks.astype(np.float64), truth_pv[tracks].astype(np.float64)
    # Fallback: derive from y=1 edges (loses edges dropped by kNN selection)
    edge_index = graph[("track", "to", "pv")].edge_index.cpu().numpy()
    y = graph[("track", "to", "pv")].y.cpu().numpy()
    sel = y > 0.5
    return edge_index[0][sel].astype(np.float64), edge_index[1][sel].astype(np.float64)


def evaluate_graph(
    model: TTVAGATModel,
    graph: HeteroData,
    eval_method: str,
    threshold: float,
    device: torch.device,
) -> tuple[list[int], list[dict[str, Any]], dict[str, float]]:
    """Run the GNN on one labelled graph; classify PVs and score edges."""
    truth_track_indices, truth_pv_indices = _truth_from_graph(graph)
    pv_ntracks = np.bincount(
        truth_pv_indices.astype(int), minlength=graph["pv"].num_nodes
    )
    truth_pvs_count = int((pv_ntracks >= 2).sum())

    if graph["pv"].num_nodes == 0:
        return [0, 0, 0, 0, 0, truth_pvs_count], [], {}

    model.eval()
    with torch.no_grad():
        logits = model(graph.to(device))
    scores = torch.sigmoid(logits).cpu().numpy()

    edge_index = graph[("track", "to", "pv")].edge_index.cpu().numpy()
    # Inference graphs (PV nodes = PVF peaks) carry no edge labels: edge-level
    # metrics are undefined there, only the vertex classification applies.
    edge_store = graph[("track", "to", "pv")]
    y = edge_store.y.cpu().numpy() if "y" in edge_store else None

    if eval_method == "MaxScore":
        selected = get_top_k_associations(scores, edge_index, k=1, threshold=threshold)
    elif eval_method == "Threshold":
        selected = scores > threshold
    else:
        msg = "Evaluation method must be MaxScore or Threshold!"
        raise ValueError(msg)

    # Assigned tracks per PV
    matched_tracks_per_pv: list[np.ndarray] = []
    order = np.argsort(edge_index[1][selected], kind="stable")
    sel_tracks = edge_index[0][selected][order]
    sel_pvs = edge_index[1][selected][order]
    boundaries = np.searchsorted(sel_pvs, np.arange(graph["pv"].num_nodes + 1))
    for p in range(graph["pv"].num_nodes):
        matched_tracks_per_pv.append(
            np.unique(sel_tracks[boundaries[p] : boundaries[p + 1]])
        )

    # pT (scaled) is track feature 7; monotonic, so fine for sum-pT^2 ranking
    pt_event = graph["track"].x[:, 7].cpu().numpy()

    results, info = classify_assignments(
        matched_tracks_per_pv,
        pt_event,
        truth_track_indices,
        truth_pv_indices,
        truth_pvs_count,
    )

    # Edge-level metrics (only when the graph carries edge labels)
    edge_metrics = {}
    if y is not None:
        edge_metrics = {
            "n_edges": float(len(y)),
            "n_true_edges": float((y > 0.5).sum()),
            "sum_scores_true": float(scores[y > 0.5].sum()),
            "sum_scores_false": float(scores[y <= 0.5].sum()),
            "n_correct_selected": float((selected & (y > 0.5)).sum()),
            "n_selected": float(selected.sum()),
        }
    return results, info, edge_metrics


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate GNN TTVA on self-labelled graphs (no truth H5)"
    )
    parser.add_argument(
        "-r", "--graphs", required=True, type=str, help="Labelled graphs (.pt)"
    )
    parser.add_argument(
        "-w", "--model-weights", required=True, type=str, help="Model weights (.pyt)"
    )
    parser.add_argument(
        "-e",
        "--eval-method",
        default="MaxScore",
        choices=["MaxScore", "Threshold"],
        help="Edge selection method (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=GNN_SCORE_THRESHOLD,
        type=float,
        help="Score threshold (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--device-id", required=True, type=int, help="CUDA device (-1 = CPU)"
    )
    parser.add_argument(
        "-o", "--output-dir", default=".", type=str, help="Output directory"
    )
    parser.add_argument(
        "--first-event",
        default=0,
        type=int,
        help="Start at this graph index (e.g. the test-split boundary)",
    )
    parser.add_argument(
        "-n", "--max-events", default=None, type=int, help="Max events to evaluate"
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    if args.device_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)

    graphs: list[HeteroData] = torch.load(args.graphs, weights_only=False)
    graphs = graphs[args.first_event :]
    if args.max_events is not None:
        graphs = graphs[: args.max_events]
    print(f"Evaluating {len(graphs)} graphs from index {args.first_event}")

    total_results: list[list[int]] = []
    edge_totals: dict[str, float] = {}
    totals = np.zeros(6, dtype=np.int64)

    for graph in tqdm(graphs):
        results, _info, edge_metrics = evaluate_graph(
            model, graph, args.eval_method, args.threshold, device
        )
        totals += np.array(results, dtype=np.int64)
        total_results.append(results)
        for k, v in edge_metrics.items():
            edge_totals[k] = edge_totals.get(k, 0.0) + v

    clean, merged, split, fake, n_reco, n_truth = totals
    print(f"Total Reconstructed PVs: {n_reco}")
    print(f"Total Truth PVs (>=2 trk): {n_truth}")
    if n_reco > 0:
        print(f"Total Clean: {clean}, Rate: {clean / n_reco:.4f}")
        print(f"Total Merged: {merged}, Rate: {merged / n_reco:.4f}")
        print(f"Total Split: {split}, Rate: {split / n_reco:.4f}")
        print(f"Total Fake: {fake}, Rate: {fake / n_reco:.4f}")
    if edge_totals.get("n_selected"):
        acc = edge_totals["n_correct_selected"] / edge_totals["n_selected"]
        eff = edge_totals["n_correct_selected"] / edge_totals["n_true_edges"]
        print(
            f"Edge level: selected-edge purity {acc:.4f}, "
            f"true-edge efficiency {eff:.4f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        out_dir / f"total_results_{args.eval_method}.npy",
        np.array(total_results, dtype=object),
        allow_pickle=True,
    )
    summary = {
        "clean": int(clean),
        "merged": int(merged),
        "split": int(split),
        "fake": int(fake),
        "reco_pvs": int(n_reco),
        "truth_pvs": int(n_truth),
        "edge_totals": edge_totals,
        "eval_method": args.eval_method,
        "threshold": args.threshold,
        "weights": args.model_weights,
        "graphs": args.graphs,
        "first_event": args.first_event,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
