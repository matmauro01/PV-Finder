#!/usr/bin/env python3
"""GNN top-1 edges on truth-vertex candidates, in the ``chain_inference`` chunk format.

Diagnostic arm of the GNN v3 campaign: the same events as a full-chain
inference directory, but the candidate list is the truth vertex list and the
graphs are the exact truth-training graphs (``root_to_graphs.build_graph_from_event``,
kNN k = 20, ``hllhc_alleta`` sigmas, recipe heights). ``peak_z`` holds the truth
z, so ``gnn.evaluation.chain_t_metrics`` replays it unchanged and the Clean HS /
HS purity of "GNN on truth candidates" compare directly with "GNN on s8 peaks".

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_truthcand_inference.py \
        --like outputs/11_12_2026_output/gnn_v2_freeze/infer/val_s8 \
        --gnn v2e527=model_weights/.../ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt \
        --out outputs/11_13_2026_output/gnn_v3_campaign/diag/infer/val_truth -d 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.data.event_selection import iterate_entries  # noqa: E402
from gnn.data.root_to_graphs import (  # noqa: E402
    TRACK_BRANCHES,
    TRUTH_BRANCHES,
    build_graph_from_event,
)
from gnn.evaluation.chain_inference import (  # noqa: E402
    best_edges,
    git_state,
    load_gnn,
    sha256,
)
from pv_finder.data.resolution_presets import RESOLUTION_PRESETS  # noqa: E402

KNN = 20


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--like", required=True, type=Path, help="chain_inference dir")
    ap.add_argument("--gnn", required=True, nargs="+", help="name=path.pyt")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("-d", "--device", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{a.device}")
    like = json.loads((a.like / "metadata.json").read_text())
    gnn_paths = dict(s.split("=", 1) for s in a.gnn)
    gnns = {k: load_gnn(v, device) for k, v in gnn_paths.items()}
    res = RESOLUTION_PRESETS["hllhc_alleta"]
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {
        **{k: like[k] for k in ("root", "entries_spec", "n_entries", "entries_sha256")},
        "candidates": "truth vertices (TruthVertex_*), truth-training graph builder",
        "resolution_preset": "hllhc_alleta", "knn": KNN, "chunk": like["chunk"],
        "density": {"name": "truth", "peaks": None, "sha256": None},
        "gnn": {k: {"path": v, "sha256": sha256(Path(v))} for k, v in gnn_paths.items()},
        "git": git_state(), "script_sha256": sha256(Path(__file__)),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }  # fmt: skip
    (a.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    tree = uproot.open(like["root"])["PVFinderData"]
    branches = [*TRACK_BRANCHES, *TRUTH_BRANCHES, "ActualNumOfInt"]
    t0 = time.monotonic()
    for path in sorted(a.like.glob("chunk_*.npz")):
        out_path = a.out / path.name
        if out_path.exists():
            continue
        with np.load(path) as f:
            entries, mu_ref = f["entries"], f["mu"]
        events = list(iterate_entries(tree, branches, entries, step_size=500))
        cols: dict[str, list] = {k: [] for k in ("mu", "n", "pz", "ph", "ps")}
        edges: dict[str, tuple[list, list]] = {g: ([], []) for g in gnns}
        for i, ev in enumerate(events):
            mu = float(ev["ActualNumOfInt"])
            if mu != mu_ref[i]:
                raise RuntimeError(f"entry {entries[i]}: mu mismatch")
            g = build_graph_from_event(ev, KNN, res)
            n = int(g["track"].x.shape[0])
            ei = g[("track", "to", "pv")].edge_index.numpy().copy()
            heights = g["pv"].x[:, 1].numpy().astype(np.float32)
            gd = g.to(device)  # in place
            for name, model in gnns.items():
                with torch.no_grad():
                    sc = torch.sigmoid(model(gd)).cpu().numpy()
                tg, bs = best_edges(sc, ei, n)
                edges[name][0].append(tg)
                edges[name][1].append(bs)
            cols["mu"].append(mu)
            cols["n"].append(n)
            cols["pz"].append(ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64))
            cols["ph"].append(heights)
            cols["ps"].append(np.zeros(len(cols["pz"][-1]), np.float32))
        out = {
            "entries": entries,
            "mu": np.array(cols["mu"]),
            "n_tracks": np.array(cols["n"], dtype=np.int64),
            "track_offsets": np.r_[0, np.cumsum(cols["n"])].astype(np.int64),
            "peak_offsets": np.r_[0, np.cumsum([len(x) for x in cols["pz"]])].astype(
                np.int64
            ),
            "peak_z": np.concatenate(cols["pz"]),
            "peak_height": np.concatenate(cols["ph"]),
            "peak_sigma": np.concatenate(cols["ps"]),
        }
        for name, (tg, bs) in edges.items():
            out[f"targets_{name}"] = np.concatenate(tg)
            out[f"scores_{name}"] = np.concatenate(bs)
        tmp = out_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **out)
        tmp.replace(out_path)
        print(f"{path.name}: {len(entries)} events, {time.monotonic() - t0:.0f} s", flush=True)  # fmt: skip
    done = sorted(p.name for p in a.out.glob("chunk_*.npz"))
    if len(done) != len(list(a.like.glob("chunk_*.npz"))):
        raise RuntimeError("missing chunks")
    h = hashlib.sha256(
        np.concatenate([np.load(a.out / p)["entries"] for p in done]).tobytes()
    ).hexdigest()
    if h != like["entries_sha256"]:
        raise RuntimeError("entry hash differs from the reference inference")
    (a.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


if __name__ == "__main__":
    main()
