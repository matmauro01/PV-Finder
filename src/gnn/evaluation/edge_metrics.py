"""Edge-level metrics for the TTVA GNN on truth-labelled graphs.

Runs the model over graphs that carry edge labels ``y`` (e.g. the μ≈60
ground-truth graph set or PU200 truth graphs), collects per-edge sigmoid
scores, and computes ROC/AUC, a precision curve, and score distributions
split by truth. This is the pure edge-classifier view of the GNN,
independent of the vertex-level Clean/Merged/Split/Fake taxonomy.

Usage:
    python -u -m gnn.evaluation.edge_metrics \\
        -r /path/to/labelled_graphs.pyt \\
        -w model_weights/gnn_ttva_epoch100.pyt \\
        -d 0 -o outputs/<date>_ttva_metrics/edge_level/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.models.ttva_gat import TTVAGATModel

N_CURVE_POINTS = 5001
N_HIST_BINS = 200


def collect_edge_scores(
    graphs_path: str | Path,
    weights_path: str | Path,
    device: torch.device,
    max_events: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over all labelled graphs; return (scores, y) flat arrays."""
    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    graphs: list[HeteroData] = torch.load(graphs_path, weights_only=False)
    if max_events is not None:
        graphs = graphs[:max_events]

    all_scores: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for graph in tqdm(graphs, desc="scoring"):
        if graph["pv"].num_nodes == 0:
            continue
        with torch.no_grad():
            logits = model(graph.to(device))
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(graph[("track", "to", "pv")].y.cpu().numpy())

    scores = np.concatenate(all_scores)
    y = np.concatenate(all_y) > 0.5
    return scores, y


def exact_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Exact ROC AUC via the Mann-Whitney rank statistic (tie-aware)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_curves(scores: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """ROC and precision curves, downsampled to N_CURVE_POINTS."""
    order = np.argsort(-scores, kind="stable")
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted, dtype=np.int64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    idx = np.unique(np.linspace(0, len(y) - 1, N_CURVE_POINTS).astype(np.int64))
    n_sel = idx + 1
    tp = cum_tp[idx]
    fp = n_sel - tp
    return {
        "threshold": scores[order][idx],
        "tpr": tp / n_pos,
        "fpr": fp / n_neg,
        "precision": tp / n_sel,
    }


def score_histograms(scores: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Score histograms for true and false edges on a fixed [0, 1] binning."""
    bins = np.linspace(0.0, 1.0, N_HIST_BINS + 1)
    hist_true, _ = np.histogram(scores[y], bins=bins)
    hist_false, _ = np.histogram(scores[~y], bins=bins)
    return {"bins": bins, "hist_true": hist_true, "hist_false": hist_false}


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Edge-level ROC/AUC and score distributions on labelled graphs"
    )
    parser.add_argument(
        "-r", "--graphs", required=True, type=str, help="Labelled graphs (.pt/.pyt)"
    )
    parser.add_argument(
        "-w", "--model-weights", required=True, type=str, help="Model weights (.pyt)"
    )
    parser.add_argument(
        "-d", "--device-id", required=True, type=int, help="CUDA device (-1 = CPU)"
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=str, help="Output directory"
    )
    parser.add_argument(
        "-n", "--max-events", default=None, type=int, help="Max events to score"
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

    scores, y = collect_edge_scores(
        args.graphs, args.model_weights, device, args.max_events
    )
    print(f"Scored {len(scores)} edges ({int(y.sum())} true)")

    auc = exact_auc(scores, y)
    curves = compute_curves(scores, y)
    hists = score_histograms(scores, y)
    print(f"ROC AUC: {auc:.6f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "edge_curves.npz", **curves, **hists)
    summary = {
        "auc": auc,
        "n_edges": int(len(y)),
        "n_true_edges": int(y.sum()),
        "mean_score_true": float(scores[y].mean()),
        "mean_score_false": float(scores[~y].mean()),
        "graphs": str(args.graphs),
        "weights": str(args.model_weights),
    }
    with open(out_dir / "edge_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved edge metrics to {out_dir}")


if __name__ == "__main__":
    main()
