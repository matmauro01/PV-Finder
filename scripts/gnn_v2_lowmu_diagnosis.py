#!/usr/bin/env python3
"""Low pile-up diagnosis of the GNN score cut (step 2b, 12 November 2026).

Per truth-labelled track (top-1 GNN edge on PV-Finder peaks): its score, whether
the top-1 peak is "right" (the peak's dominant truth at t = 0 is the track's
own truth vertex), the multiplicity of its truth vertex, the number of peaks in
the event, and the distance from its target peak to the nearest other peak.
Per truth vertex (nTrk >= 2): whether a Clean vertex has it as dominant at each
t. Aggregated in mu bins; a few low-mu event displays are drawn.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_lowmu_diagnosis.py \
        --infer outputs/11_12_2026_output/gnn_v2_freeze/infer/val_v6 --gnn v2e527 \
        --out outputs/11_12_2026_output/gnn_v2_freeze/lowmu
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn.evaluation import hs_metrics as hm  # noqa: E402

TS = (0.0, 0.9, 0.95, 0.98, 0.99)
MU_EDGES = (0, 10, 20, 50, 100, 150, 201)
NTRK_EDGES = (2, 3, 5, 10, 20, 1000)
BR = ["ActualNumOfInt", "RecoTrack_z0", "RecoTrack_pT", "TruthVertex_z",
      "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type"]  # fmt: skip


def chunk(path: str, root: str, gnn: str) -> dict:
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    e = z["entries"]
    data = uproot.open(root)["PVFinderData"].arrays(BR, entry_start=int(e[0]), entry_stop=int(e[-1]) + 1, library="ak")  # fmt: skip
    trk_rows, vtx_rows, split_rows = [], [], []
    for i, ent in enumerate(e):
        ev = data[int(ent) - int(e[0])]
        pt = ak.to_numpy(ev["RecoTrack_pT"]).astype(np.float64)
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        truth = hm.track_truth_labels(ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, len(pt))  # fmt: skip
        types = ak.to_numpy(ev["TruthVertex_type"])
        tz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
        pk = z["peak_z"][z["peak_offsets"][i] : z["peak_offsets"][i + 1]].astype(
            np.float64
        )
        sl = slice(z["track_offsets"][i], z["track_offsets"][i + 1])
        tg, sc = z[f"targets_{gnn}"][sl], z[f"scores_{gnn}"][sl]
        mu = z["mu"][i]
        if len(pk) == 0:
            continue
        lists0 = hm.lists_from_targets(hm.assign_from_scores(tg, sc, 0.0), len(pk))
        dom = hm.classify_vertices(lists0, pt, truth).dominant
        gap = np.full(len(pk), 99.0)
        if len(pk) > 1:
            d = np.abs(pk[:, None] - pk[None, :])
            np.fill_diagonal(d, np.inf)
            gap = d.min(1)
        lab = np.flatnonzero((truth >= 0) & (tg >= 0))
        right = dom[tg[lab]] == truth[lab]
        trk_rows.append(np.stack([np.full(len(lab), mu), sc[lab], right,
                                  tn[truth[lab]], np.full(len(lab), len(pk)),
                                  gap[tg[lab]], types[truth[lab]] == 1], 1))  # fmt: skip
        clean_at = []
        for t in TS:
            lists = hm.lists_from_targets(hm.assign_from_scores(tg, sc, t), len(pk))
            kidx = [j for j, x in enumerate(lists) if len(x)]
            keep = [lists[j] for j in kidx]
            cls = hm.classify_vertices(keep, pt, truth)
            for v in np.flatnonzero(cls.labels == hm.SPLIT):
                dm = cls.dominant[v]
                u = next(
                    w
                    for w in range(len(keep))
                    if cls.dominant[w] == dm and cls.labels[w] != hm.SPLIT
                )
                split_rows.append([mu, t, cls.n_tracks[v], tn[dm], types[dm] == 1,
                                   abs(pk[kidx[v]] - pk[kidx[u]]), cls.n_tracks[u]])  # fmt: skip
            c = np.zeros(len(tn), bool)
            c[cls.dominant[(cls.labels == hm.CLEAN) & (cls.dominant >= 0)]] = True
            clean_at.append(c)
        sel = np.flatnonzero(tn >= 2)
        near = np.abs(tz[sel, None] - pk[None, :]).min(1) < 0.17
        vtx_rows.append(np.column_stack([np.full(len(sel), mu), tn[sel], near, types[sel] == 1,
                                         *[c[sel] for c in clean_at]]))  # fmt: skip
    return {"trk": np.concatenate(trk_rows), "vtx": np.concatenate(vtx_rows),
            "split": np.array(split_rows, dtype=np.float64).reshape(-1, 7)}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--infer", type=Path, required=True)
    ap.add_argument("--gnn", default="v2e527")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    root = json.loads((a.infer / "metadata.json").read_text())["root"]
    paths = sorted(str(p) for p in a.infer.glob("chunk_*.npz"))
    with ProcessPoolExecutor(a.workers) as pool:
        parts = list(pool.map(chunk, paths, [root] * len(paths), [a.gnn] * len(paths)))
    trk = np.concatenate([p["trk"] for p in parts])
    vtx = np.concatenate([p["vtx"] for p in parts])
    tag = f"{a.infer.name}_{a.gnn}"
    spl = np.concatenate([p["split"] for p in parts])
    np.savez_compressed(a.out / f"lowmu_{tag}.npz", trk=trk, vtx=vtx, split=spl)
    res: dict = {"t": TS, "mu_edges": MU_EDGES, "ntrk_edges": NTRK_EDGES, "bins": {}}
    for lo, hi in zip(MU_EDGES[:-1], MU_EDGES[1:]):
        m = (trk[:, 0] >= lo) & (trk[:, 0] < hi)
        r = m & (trk[:, 2] > 0)
        mv = (vtx[:, 0] >= lo) & (vtx[:, 0] < hi)
        b = {
            "tracks": int(m.sum()),
            "right_top1_frac": float(r.sum() / m.sum()),
            "right_score_quantiles_10_25_50": np.quantile(trk[r, 1], [0.1, 0.25, 0.5]).tolist(),
            "right_frac_below": {f"{t}": float((trk[r, 1] < t).mean()) for t in TS[1:]},
            "wrong_frac_below": {f"{t}": float((trk[m & (trk[:, 2] == 0), 1] < t).mean()) for t in TS[1:]},
            "right_below_0.99_by_gap": {},
            "right_below_0.99_by_ntrk": {},
            "mean_peaks": float(trk[m, 4].mean()),
            "truth_vertices": int(mv.sum()),
            "finder_found_frac": float(vtx[mv, 2].mean()),
            "clean_frac_by_ntrk": {},
            "hs_clean_frac": [float(vtx[mv & (vtx[:, 3] > 0), 4 + k].mean()) for k in range(len(TS))],
        }  # fmt: skip
        for glo, ghi in ((0, 0.2), (0.2, 0.4), (0.4, 1.0), (1.0, 100)):
            g = r & (trk[:, 5] >= glo) & (trk[:, 5] < ghi)
            b["right_below_0.99_by_gap"][f"{glo}-{ghi}mm"] = [float((trk[g, 1] < 0.99).mean()) if g.any() else None, int(g.sum())]  # fmt: skip
        for nlo, nhi in zip(NTRK_EDGES[:-1], NTRK_EDGES[1:]):
            g = r & (trk[:, 3] >= nlo) & (trk[:, 3] < nhi)
            b["right_below_0.99_by_ntrk"][f"{nlo}-{nhi}"] = float(
                (trk[g, 1] < 0.99).mean()
            )
            gv = mv & (vtx[:, 1] >= nlo) & (vtx[:, 1] < nhi)
            b["clean_frac_by_ntrk"][f"{nlo}-{nhi}"] = {"n": int(gv.sum()), "frac_per_t": [float(vtx[gv, 4 + k].mean()) for k in range(len(TS))]}  # fmt: skip
        b["split_vertices"] = {}
        for k, t in enumerate(TS):
            x = spl[(spl[:, 0] >= lo) & (spl[:, 0] < hi) & (spl[:, 1] == t)]
            b["split_vertices"][f"{t}"] = {
                "n": int(len(x)),
                "frac_one_track": float((x[:, 2] == 1).mean()) if len(x) else None,
                "frac_hs_dominant": float(x[:, 4].mean()) if len(x) else None,
                "median_dz_to_kept_mm": float(np.median(x[:, 5])) if len(x) else None,
                "median_dominant_truth_ntrk": float(np.median(x[:, 3]))
                if len(x)
                else None,
            }
        res["bins"][f"{lo}-{hi}"] = b
    (a.out / f"lowmu_{tag}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
