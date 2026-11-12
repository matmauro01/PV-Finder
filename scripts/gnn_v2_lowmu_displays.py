#!/usr/bin/env python3
"""Low-mu event displays for the GNN v2 freeze (two patterns found in the diagnosis).

(a) A truth vertex that is Clean at t = 0.9 and lost at t = 0.99: its tracks'
    top-1 scores are shown against the cut.
(b) A one-track Split vertex whose only track is a hard-scatter track (an HS
    satellite peak).
Each panel: PV-Finder density, peaks, truth vertices (height = nTrk) and the
z0 of the relevant truth vertex's tracks, coloured by top-1 score and target.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_lowmu_displays.py \
        --infer outputs/11_12_2026_output/gnn_v2_freeze/infer/val_v6 \
        --out outputs/11_12_2026_output/gnn_v2_freeze/lowmu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn.diagnostics.plot_style import save_figure  # noqa: E402
from gnn.evaluation import hs_metrics as hm  # noqa: E402

Z_MIN, BW = -240.0, 0.04
BR = ["ActualNumOfInt", "RecoTrack_z0", "RecoTrack_pT", "TruthVertex_z",
      "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type"]  # fmt: skip


def clean_set(tg, sc, t, npk, pt, truth):  # noqa: ANN001, ANN201
    lists = hm.lists_from_targets(hm.assign_from_scores(tg, sc, t), npk)
    idx = [j for j, x in enumerate(lists) if len(x)]
    cls = hm.classify_vertices([lists[j] for j in idx], pt, truth)
    return cls, idx


def find_examples(infer: Path, gnn: str, root: str, n_each: int) -> list[dict]:
    out: dict[str, list] = {"lost": [], "satellite": []}
    for p in sorted(infer.glob("chunk_*.npz")):
        z = np.load(p)
        sel = np.flatnonzero(z["mu"] < 20)
        if not len(sel):
            continue
        e = z["entries"]
        d = uproot.open(root)["PVFinderData"].arrays(BR, entry_start=int(e[0]), entry_stop=int(e[-1]) + 1, library="ak")  # fmt: skip
        for i in sel:
            ev = d[int(e[i]) - int(e[0])]
            pt = ak.to_numpy(ev["RecoTrack_pT"]).astype(float)
            tn = ak.to_numpy(ev["TruthVertex_nTracks"])
            truth = hm.track_truth_labels(ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, len(pt))  # fmt: skip
            ty = ak.to_numpy(ev["TruthVertex_type"])
            po = z["peak_offsets"]
            npk = po[i + 1] - po[i]
            s = slice(z["track_offsets"][i], z["track_offsets"][i + 1])
            tg, sc = z[f"targets_{gnn}"][s], z[f"scores_{gnn}"][s]
            c90, _ = clean_set(tg, sc, 0.9, npk, pt, truth)
            c99, _ = clean_set(tg, sc, 0.99, npk, pt, truth)
            a = set(c90.dominant[c90.labels == hm.CLEAN]) - set(c99.dominant[c99.labels == hm.CLEAN])  # fmt: skip
            lost = [v for v in a if v >= 0 and 5 <= tn[v] <= 12]
            if lost and len(out["lost"]) < n_each:
                out["lost"].append({"chunk": str(p), "i": int(i), "truth": int(lost[0]), "kind": "lost"})  # fmt: skip
            c0, idx0 = clean_set(tg, sc, 0.0, npk, pt, truth)
            sat = [v for v in np.flatnonzero(c0.labels == hm.SPLIT) if c0.n_tracks[v] == 1 and ty[c0.dominant[v]] == 1]  # fmt: skip
            if sat and len(out["satellite"]) < n_each:
                out["satellite"].append({"chunk": str(p), "i": int(i), "truth": int(c0.dominant[sat[0]]), "kind": "satellite", "peak": int(idx0[sat[0]])})  # fmt: skip
            if all(len(v) >= n_each for v in out.values()):
                return out["lost"] + out["satellite"]
    return out["lost"] + out["satellite"]


def draw(ax, ex: dict, gnn: str, root: str) -> None:  # noqa: ANN001
    z = np.load(ex["chunk"])
    i = ex["i"]
    ent = int(z["entries"][i])
    ev = uproot.open(root)["PVFinderData"].arrays(BR, entry_start=ent, entry_stop=ent + 1, library="ak")[0]  # fmt: skip
    z0 = ak.to_numpy(ev["RecoTrack_z0"]).astype(float)
    tn = ak.to_numpy(ev["TruthVertex_nTracks"])
    tz = ak.to_numpy(ev["TruthVertex_z"]).astype(float)
    truth = hm.track_truth_labels(ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, len(z0))  # fmt: skip
    po = z["peak_offsets"]
    pk = z["peak_z"][po[i] : po[i + 1]]
    s = slice(z["track_offsets"][i], z["track_offsets"][i + 1])
    sc = z[f"scores_{gnn}"][s]
    v = ex["truth"]
    zc = tz[v]
    lo, hi = zc - 2.0, zc + 2.0
    h = z["histograms"][i]
    x = Z_MIN + BW * (np.arange(len(h)) + 0.5)
    m = (x > lo) & (x < hi)
    ax.plot(x[m], h[m], color="#0072B2", lw=1.2, label="PV-Finder density")
    top = max(h[m].max(), 1e-3)
    for zz, n in zip(tz, tn):
        if lo < zz < hi:
            ax.vlines(zz, 0, top * min(1.0, n / 30), color="#999999", lw=2)
            ax.text(
                zz, top * min(1.0, n / 30), f"{n}", fontsize=7, ha="center", va="bottom"
            )
    for p in pk:
        if lo < p < hi:
            ax.plot(p, -0.05 * top, "v", color="#000000", ms=6)
    own = np.flatnonzero(truth == v)
    for k in own:
        c = "#009E73" if sc[k] >= 0.99 else ("#E69F00" if sc[k] >= 0.9 else "#D55E00")
        ax.plot(z0[k], -0.15 * top, "|", color=c, ms=12, mew=1.5)
    hit = np.mean(sc[own] >= 0.99) if len(own) else np.nan
    ax.set_xlim(lo, hi)
    ax.set_xlabel("z [mm]")
    title = (f"entry {ent}, mu {z['mu'][i]:.0f}: truth vertex nTrk {tn[v]}"
             + (" (HS)" if ex["kind"] == "satellite" else "")
             + f"; {100 * hit:.0f}% of its tracks have score >= 0.99")  # fmt: skip
    ax.set_title(title, fontsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--infer", type=Path, required=True)
    ap.add_argument("--gnn", default="v2e527")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-each", type=int, default=2)
    a = ap.parse_args()
    import matplotlib.pyplot as plt

    root = json.loads((a.infer / "metadata.json").read_text())["root"]
    exs = find_examples(a.infer, a.gnn, root, a.n_each)
    fig, axes = plt.subplots(len(exs), 1, figsize=(9, 2.6 * len(exs)))
    for ax, ex in zip(np.atleast_1d(axes), exs):
        draw(ax, ex, a.gnn, root)
    np.atleast_1d(axes)[0].text(
        0.01, 0.95, "ticks: tracks of the labelled truth vertex; green >= 0.99, "
        "orange 0.9-0.99, red < 0.9; triangles: peaks; grey: truth (label nTrk)",
        transform=np.atleast_1d(axes)[0].transAxes, fontsize=7, va="top",
    )  # fmt: skip
    fig.tight_layout()
    save_figure(fig, a.out, f"lowmu_displays_{a.infer.name}_{a.gnn}")
    (a.out / f"lowmu_displays_{a.infer.name}_{a.gnn}.json").write_text(
        json.dumps(exs, indent=1)
    )


if __name__ == "__main__":
    main()
