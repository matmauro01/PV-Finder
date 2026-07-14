"""Hard-scatter vertex identification at PU200 through the full chain.

The hard-scatter (HS) vertex is conventionally the reconstructed vertex
with the largest sum of squared track transverse momenta. This script
measures how often that vertex truly is the HS interaction
(TruthVertex_type == 1) for:

- PVF peaks + GNN track assignment (MaxScore at one or more thresholds),
- AMVF with its own track assignment (RecoVertex_assocTracks),

on the same events, judged identically: the selected reco vertex is
correct if the plurality of its assigned tracks with truth come from the
true HS vertex.

Usage:
    python -u -m gnn.evaluation.hs_id_pu200 \\
        --graphs data/run4/ttva_graphs/pu200_chain_v4b_k20_test.pt \\
        --root data/run4/Run4_MC21_ITk/..._PU200.root --entry-start 28500 \\
        -w model_weights/ttva_gnn_hllhc/ttva_gat_pu200_k20_epoch_175.pyt \\
        -d 0 -o outputs/<date>/hs_id/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from tqdm import tqdm

from gnn.evaluation.classification import get_top_k_associations
from gnn.models.ttva_gat import TTVAGATModel

THRESHOLDS = (0.5, 0.95, 0.995)


def dominant_truth(tracks: np.ndarray, track_truth: np.ndarray) -> int:
    """Plurality truth-PV index among assigned tracks (-1 if none)."""
    truths = track_truth[tracks]
    truths = truths[truths >= 0]
    if len(truths) == 0:
        return -1
    vals, counts = np.unique(truths, return_counts=True)
    return int(vals[np.argmax(counts)])


def leading_vertex(matched: list[np.ndarray], pt: np.ndarray) -> int:
    """Index of the vertex with max sum(pT^2) (-1 if all empty)."""
    best, best_val = -1, -1.0
    for i, tracks in enumerate(matched):
        if len(tracks) == 0:
            continue
        val = float((pt[tracks] ** 2).sum())
        if val > best_val:
            best, best_val = i, val
    return best


def main() -> None:  # noqa: PLR0912
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
    model.eval()

    graphs = torch.load(args.graphs, weights_only=False)
    tree = uproot.open(args.root)["PVFinderData"]
    stop = args.entry_start + len(graphs)

    n_events = 0
    gnn_correct = {t: 0 for t in THRESHOLDS}
    amvf_correct = 0
    idx = 0
    for chunk in tree.iterate(
        ["TruthVertex_type", "RecoVertex_assocTracks", "RecoTrack_pT"],
        step_size=500, entry_start=args.entry_start, entry_stop=stop,
    ):  # fmt: skip
        for event in tqdm(chunk, leave=False):
            graph = graphs[idx]
            idx += 1
            pv_type = ak.to_numpy(event["TruthVertex_type"])
            hs_candidates = np.nonzero(pv_type == 1)[0]
            if len(hs_candidates) != 1:
                continue
            hs_idx = int(hs_candidates[0])
            n_events += 1

            track_truth = graph["track"].truth_pv.numpy()
            pt = graph["track"].x[:, 7].numpy()
            n_tracks = len(pt)

            # --- GNN chain ---
            with torch.no_grad():
                logits = model(graph.to(device))
            scores = torch.sigmoid(logits).cpu().numpy()
            edge_index = graph[("track", "to", "pv")].edge_index.cpu().numpy()
            for t in THRESHOLDS:
                sel = get_top_k_associations(scores, edge_index, k=1, threshold=t)
                matched: list[np.ndarray] = [
                    np.empty(0, np.int64) for _ in range(graph["pv"].num_nodes)
                ]
                sel_tracks = edge_index[0][sel]
                sel_pvs = edge_index[1][sel]
                order = np.argsort(sel_pvs, kind="stable")
                bounds = np.searchsorted(
                    sel_pvs[order], np.arange(graph["pv"].num_nodes + 1)
                )
                for p in range(graph["pv"].num_nodes):
                    matched[p] = np.unique(sel_tracks[order[bounds[p] : bounds[p + 1]]])
                lead = leading_vertex(matched, pt)
                if lead >= 0 and dominant_truth(matched[lead], track_truth) == hs_idx:
                    gnn_correct[t] += 1

            # --- AMVF ---
            pt_full = ak.to_numpy(event["RecoTrack_pT"]).astype(np.float64)
            amvf_matched = []
            for vertex_tracks in event["RecoVertex_assocTracks"]:
                tr = ak.to_numpy(vertex_tracks).astype(np.int64)
                amvf_matched.append(np.unique(tr[(tr >= 0) & (tr < n_tracks)]))
            lead = leading_vertex(amvf_matched, pt_full)
            if lead >= 0 and dominant_truth(amvf_matched[lead], track_truth) == hs_idx:
                amvf_correct += 1

    out = {
        "n_events": n_events,
        "amvf_hs_id": amvf_correct / n_events,
        "gnn_hs_id": {str(t): gnn_correct[t] / n_events for t in THRESHOLDS},
        "weights": args.model_weights,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "hs_id.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--graphs", required=True, type=str)
    p.add_argument("--root", required=True, type=str)
    p.add_argument("--entry-start", default=28500, type=int)
    p.add_argument("-w", "--model-weights", required=True, type=str)
    p.add_argument("-d", "--device-id", required=True, type=int)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    main()
