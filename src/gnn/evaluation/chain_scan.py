"""Threshold scan for TTVA on self-labelled graphs (chain or truth PU200).

Runs the GNN forward pass ONCE per graph, caches scores, then re-applies
MaxScore selection over a threshold grid. Each operating point is
classified under two vertex conventions:

- ``all_peaks``: every PV node is a reconstructed vertex (historical
  convention; zero-track vertices are booked as Fake).
- ``drop_empty``: vertices with no assigned tracks are removed from the
  reco list before classification. No vertexing chain outputs a trackless
  vertex, so this is the deployment-faithful convention; clean/truth is
  identical between the two by construction (an empty vertex is never
  Clean), only fake/split accounting moves.

Usage:
    python -u -m gnn.evaluation.chain_scan \\
        -r data/run4/ttva_graphs/pu200_chain_v4b_k20_test.pt \\
        -w model_weights/ttva_gnn_hllhc/ttva_gat_pu200_k20_epoch_175.pyt \\
        --first-event 0 -d 0 -o outputs/<date>/chain_scan_v1/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.evaluation.classification import classify_assignments, get_top_k_associations
from gnn.evaluation.evaluate_ttva_graphs import _truth_from_graph
from gnn.models.ttva_gat import TTVAGATModel

THRESHOLD_GRID = (0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def cache_event(
    model: TTVAGATModel, graph: HeteroData, device: torch.device
) -> dict[str, np.ndarray]:
    """One GNN forward pass; keep everything selection needs on CPU."""
    tti, tpi = _truth_from_graph(graph)
    pv_ntracks = np.bincount(tpi.astype(int), minlength=graph["pv"].num_nodes)
    model.eval()
    with torch.no_grad():
        logits = model(graph.to(device))
    return {
        "scores": torch.sigmoid(logits).cpu().numpy(),
        "edge_index": graph[("track", "to", "pv")].edge_index.cpu().numpy(),
        "n_pvs": graph["pv"].num_nodes,
        "pt": graph["track"].x[:, 7].cpu().numpy(),
        "tti": tti,
        "tpi": tpi,
        "truth_count": int((pv_ntracks >= 2).sum()),
    }


def matched_lists(cached: dict, threshold: float) -> list[np.ndarray]:
    """MaxScore selection at *threshold* -> per-PV unique track arrays."""
    selected = get_top_k_associations(
        cached["scores"], cached["edge_index"], k=1, threshold=threshold
    )
    edge_index = cached["edge_index"]
    order = np.argsort(edge_index[1][selected], kind="stable")
    sel_tracks = edge_index[0][selected][order]
    sel_pvs = edge_index[1][selected][order]
    boundaries = np.searchsorted(sel_pvs, np.arange(cached["n_pvs"] + 1))
    return [
        np.unique(sel_tracks[boundaries[p] : boundaries[p + 1]])
        for p in range(cached["n_pvs"])
    ]


def scan_point(cache: list[dict], threshold: float) -> dict:
    """Classify all events at one threshold under both vertex conventions."""
    totals = np.zeros(6, dtype=np.int64)
    totals_drop = np.zeros(6, dtype=np.int64)
    n_empty = 0
    for cached in cache:
        lists = matched_lists(cached, threshold)
        args = (cached["pt"], cached["tti"], cached["tpi"], cached["truth_count"])
        rows, _ = classify_assignments(lists, *args)
        totals += np.array(rows, dtype=np.int64)
        nonempty = [m for m in lists if len(m)]
        n_empty += len(lists) - len(nonempty)
        rows_d, _ = classify_assignments(nonempty, *args)
        totals_drop += np.array(rows_d, dtype=np.int64)

    def _rates(t: np.ndarray) -> dict:
        clean, merged, split, fake, n_reco, n_truth = (int(x) for x in t)
        return {
            "clean": clean, "merged": merged, "split": split, "fake": fake,
            "n_reco": n_reco, "n_truth": n_truth,
            "clean_rate": clean / max(n_reco, 1),
            "merged_rate": merged / max(n_reco, 1),
            "split_rate": split / max(n_reco, 1),
            "fake_rate": fake / max(n_reco, 1),
            "clean_per_truth": clean / max(n_truth, 1),
        }  # fmt: skip

    return {
        "t": threshold,
        "n_empty_vertices": int(n_empty),
        "all_peaks": _rates(totals),
        "drop_empty": _rates(totals_drop),
    }


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

    graphs: list[HeteroData] = torch.load(args.graphs, weights_only=False)
    graphs = graphs[args.first_event :]
    if args.max_events is not None:
        graphs = graphs[: args.max_events]
    print(f"Scanning {len(graphs)} graphs, {len(THRESHOLD_GRID)} thresholds")

    cache = [cache_event(model, g, device) for g in tqdm(graphs, desc="forward")]
    results = [scan_point(cache, t) for t in tqdm(THRESHOLD_GRID, desc="scan")]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": args.model_weights,
        "graphs": args.graphs,
        "first_event": args.first_event,
        "n_events": len(graphs),
        "results": results,
    }
    with open(out_dir / "chain_scan.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n{'t':>6} {'clean/truth':>12} {'fake(all)':>10} {'fake(drop)':>11} "
          f"{'empty':>7}")  # fmt: skip
    for r in results:
        print(f"{r['t']:>6} {r['all_peaks']['clean_per_truth']:>12.4f} "
              f"{r['all_peaks']['fake_rate']:>10.4f} "
              f"{r['drop_empty']['fake_rate']:>11.4f} "
              f"{r['n_empty_vertices']:>7}")  # fmt: skip
    print(f"Saved to {out_dir / 'chain_scan.json'}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-r", "--graphs", required=True, type=str)
    p.add_argument("-w", "--model-weights", required=True, type=str)
    p.add_argument("-d", "--device-id", required=True, type=int)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    p.add_argument("--first-event", default=0, type=int)
    p.add_argument("-n", "--max-events", default=None, type=int)
    return p.parse_args()


if __name__ == "__main__":
    main()
