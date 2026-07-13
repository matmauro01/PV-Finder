"""Per-event inference timing of the TTVA GNN on saved graphs.

Measures, per event: GNN forward (GPU-synchronized), sigmoid + MaxScore
selection (CPU), and reports distribution statistics plus graph-size
context. Use together with the chain builder's chain_info.json (PVF
inference / peak finding / graph construction) for the full-chain latency
budget.

Usage:
    python -u -m gnn.evaluation.benchmark_ttva \\
        -r data/run4/ttva_graphs/pu200_chain_v4b_k20_test.pt \\
        -w model_weights/ttva_gnn_hllhc/ttva_gat_pu200_k20_epoch_175.pyt \\
        -d 1 -n 300 -o outputs/<date>_ttva_chain/gnn_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from gnn.evaluation.classification import get_top_k_associations
from gnn.models.ttva_gat import TTVAGATModel

N_WARMUP = 20


def _stats(values: list[float]) -> dict[str, float]:
    """Mean/median/p90 of a list."""
    arr = np.asarray(values)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def benchmark(
    graphs: list[HeteroData],
    model: TTVAGATModel,
    device: torch.device,
    threshold: float,
) -> dict:
    """Time forward + selection per event; return summary dict."""
    model.eval()
    graphs = [g for g in graphs if g["pv"].num_nodes > 0]

    # Move to device once (deployment would hold the event on-device too);
    # warm up kernels on the first few graphs.
    for graph in graphs[:N_WARMUP]:
        with torch.no_grad():
            model(graph.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    forward_ms: list[float] = []
    select_ms: list[float] = []
    n_edges: list[int] = []
    n_tracks: list[int] = []
    for graph in graphs:
        graph = graph.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(graph)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        scores = torch.sigmoid(logits).cpu().numpy()
        edge_index = graph[("track", "to", "pv")].edge_index.cpu().numpy()
        get_top_k_associations(scores, edge_index, k=1, threshold=threshold)
        t2 = time.perf_counter()

        forward_ms.append(1e3 * (t1 - t0))
        select_ms.append(1e3 * (t2 - t1))
        n_edges.append(len(scores))
        n_tracks.append(int(graph["track"].num_nodes))

    return {
        "n_events": len(forward_ms),
        "device": str(device),
        "threshold": threshold,
        "parameters": sum(p.numel() for p in model.parameters()),
        "edges_per_event_mean": float(np.mean(n_edges)),
        "tracks_per_event_mean": float(np.mean(n_tracks)),
        "gnn_forward_ms": _stats(forward_ms),
        "selection_ms": _stats(select_ms),
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="TTVA GNN inference benchmark")
    parser.add_argument("-r", "--graphs", required=True, type=str)
    parser.add_argument("-w", "--model-weights", required=True, type=str)
    parser.add_argument("-d", "--device-id", required=True, type=int)
    parser.add_argument("-n", "--max-events", default=None, type=int)
    parser.add_argument("-t", "--threshold", default=0.5, type=float)
    parser.add_argument("-o", "--output", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    device = torch.device(
        f"cuda:{args.device_id}"
        if args.device_id >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.to(device)

    graphs = torch.load(args.graphs, weights_only=False)
    if args.max_events:
        graphs = graphs[: args.max_events]

    summary = benchmark(graphs, model, device, args.threshold)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
