#!/usr/bin/env python3
"""Is the low-mu score deficit of small vertices intrinsic to the GNN or chain-specific?

Runs a GNN checkpoint on the 1,500 truth-candidate validation-monitor graphs
(the candidates the GNN was trained on) and reports, in mu bins and truth-vertex
multiplicity bins, the fraction of tracks whose top-1 edge is their own truth
vertex and has score < 0.99. Compare with the chain numbers in
``lowmu/lowmu_val_v6_v2e527.json`` (``right_below_0.99_by_ntrk``).

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_lowmu_truthgraph.py \
        --out outputs/11_12_2026_output/gnn_v2_freeze/lowmu -d 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn.evaluation.chain_inference import best_edges, load_gnn  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
D = REPO / "data/run4_all_etas/ttva_graphs/fullmu_v1_20260924"
ROOT = REPO / "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"  # fmt: skip
CKPT = REPO / "model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt"  # fmt: skip
MU_EDGES = (0, 10, 20, 50, 100, 150, 201)
NTRK_EDGES = (2, 3, 5, 10, 20, 1000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-d", "--device", type=int, default=0)
    a = ap.parse_args()
    device = torch.device(f"cuda:{a.device}")
    model = load_gnn(str(CKPT), device)
    graphs = torch.load(D / "validation_monitor.pt", weights_only=False)
    entries = np.load(D / "validation_monitor.entry_indices.npy")
    mu_all = uproot.open(ROOT)["PVFinderData"]["ActualNumOfInt"].array(library="np")
    rows = []
    for g, e in zip(graphs, entries):
        g = g.to(device)
        with torch.no_grad():
            sc = torch.sigmoid(model(g)).cpu().numpy()
        ei = g[("track", "to", "pv")].edge_index.cpu().numpy()
        truth = g["track"].truth_pv.cpu().numpy()
        n = len(truth)
        tg, bs = best_edges(sc, ei, n)
        lab = truth >= 0
        ntrk = np.bincount(truth[lab], minlength=g["pv"].num_nodes)
        right = lab & (tg == truth)
        idx = np.flatnonzero(right)
        rows.append(np.stack([np.full(len(idx), mu_all[e]), bs[idx], ntrk[truth[idx]], np.full(len(idx), g["pv"].num_nodes)], 1))  # fmt: skip
    r = np.concatenate(rows)
    res: dict = {"checkpoint": str(CKPT), "graphs": len(graphs), "bins": {}}
    for lo, hi in zip(MU_EDGES[:-1], MU_EDGES[1:]):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        res["bins"][f"{lo}-{hi}"] = {
            "right_tracks": int(m.sum()),
            "mean_candidates": float(r[m, 3].mean()),
            "right_below_0.99_by_ntrk": {
                f"{nl}-{nh}": float((r[m & (r[:, 2] >= nl) & (r[:, 2] < nh), 1] < 0.99).mean())
                for nl, nh in zip(NTRK_EDGES[:-1], NTRK_EDGES[1:])
            },
        }  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "lowmu_truthgraph_v2e527.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
