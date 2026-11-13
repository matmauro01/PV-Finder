"""EB question 15: does the Athena sum-pT^2 rule on AMVF's own fit tracks reproduce its stored primary?

For r16443 entries 0-24,999 (Trey's supplementary ntuple), rank every AMVF vertex by
sum(pT^2) over (a) RecoVertex_fitTracks (the vertex trackParticleLinks Athena sorts),
(b) the same restricted to fit weight > 0.5 (no-op test), (c) RecoVertex_assocTracks (TTVA
lists, which our same-rule AMVF number uses), and report how often the argmax equals the
stored primary (RecoVertex_type == 1). PU200 = 185 <= mu <= 215.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import uproot

BR = [
    "ActualNumOfInt",
    "RecoTrack_pT",
    "RecoVertex_type",
    "RecoVertex_fitTracks",
    "RecoVertex_fitWeights",
    "RecoVertex_assocTracks",
    "RecoVertex_nTracks",
]


def argmax_sumpt2(
    lists: list, pt2: np.ndarray, weights: list | None = None, wmin: float = -1.0
) -> int:
    """Index of the vertex with the largest sum of pT^2 over its listed tracks (-1 if none)."""
    best, arg = -1.0, -1
    for v, trk in enumerate(lists):
        trk = np.asarray(trk, dtype=np.int64)
        if weights is not None:
            trk = trk[np.asarray(weights[v]) > wmin]
        s = float(pt2[trk].sum()) if trk.size else 0.0
        if s > best:
            best, arg = s, v
    return arg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/share/lazy/pv-finder/Paper_2026Nov12/"
        "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root",
    )
    ap.add_argument("--n", type=int, default=25000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = uproot.open(a.root)["PVFinderData"]
    arr = t.arrays(BR, entry_stop=a.n, library="np")
    keys = ("fit", "fit_w05", "assoc")
    agree = {k: {"all": 0, "pu200": 0} for k in keys}
    n = {"all": 0, "pu200": 0}
    for e in range(a.n):
        typ = np.asarray(arr["RecoVertex_type"][e])
        stored = np.flatnonzero(typ == 1)
        if stored.size != 1:
            continue
        stored = int(stored[0])
        pt2 = np.asarray(arr["RecoTrack_pT"][e], dtype=np.float64) ** 2
        fit, w = (
            list(arr["RecoVertex_fitTracks"][e]),
            list(arr["RecoVertex_fitWeights"][e]),
        )
        assoc = list(arr["RecoVertex_assocTracks"][e])
        res = {
            "fit": argmax_sumpt2(fit, pt2),
            "fit_w05": argmax_sumpt2(fit, pt2, w, 0.5),
            "assoc": argmax_sumpt2(assoc, pt2),
        }
        pu = 185 <= float(arr["ActualNumOfInt"][e]) <= 215
        n["all"] += 1
        n["pu200"] += pu
        for k in keys:
            ok = res[k] == stored
            agree[k]["all"] += ok
            agree[k]["pu200"] += ok and pu
    out = {
        k: {
            s: {"agree": agree[k][s], "n": n[s], "frac": agree[k][s] / max(n[s], 1)}
            for s in ("all", "pu200")
        }
        for k in keys
    }
    with open(f"{a.out}/amvf_primary_from_fit_tracks.json", "w") as f:
        json.dump({"root": a.root, "n_entries": a.n, "result": out}, f, indent=1)
    for k in keys:
        print(
            k,
            {
                s: f"{100 * out[k][s]['frac']:.2f}% ({out[k][s]['agree']}/{out[k][s]['n']})"
                for s in out[k]
            },
        )


if __name__ == "__main__":
    main()
