"""Track multiplicity of truth vertices without a one-to-one PV-Finder peak (note Sec. 7.5).

PU200 events of r16443 (the frozen s8 evaluation cache), truth vertices with
>= 2 tracks, greedy closest-first one-to-one matching of all peaks to truth
within ``--window`` (the "finder cap" matching of the chain studies). Reports
the median n_trk of unmatched and matched truth vertices and the fraction of
unmatched ones with exactly two tracks. CPU only, no inference.

Usage (repo root, venv, PYTHONPATH=src):
    python scripts/note_unmatched_truth_ntrk.py \
        --cache outputs/11_12_2026_output/hllhc_final/h1_vertex_finding/fair_r16443/cache_r16443.npz \
        --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
        --out outputs/11_13_2026_output/note_edits/unmatched_ntrk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from gnn.diagnostics.chain_gap_decomposition import greedy_match


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--window", type=float, default=0.17)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    c = np.load(args.cache, allow_pickle=True)
    ev = c["event_idx"]
    tree = uproot.open(args.root)["PVFinderData"]
    arr = tree.arrays(
        ["TruthVertex_z", "TruthVertex_nTracks"],
        entry_start=0,
        entry_stop=int(ev.max()) + 1,
        library="np",
    )
    unmatched, matched = [], []
    for i, e in enumerate(ev):
        z = np.asarray(arr["TruthVertex_z"][e], dtype=np.float64)
        n = np.asarray(arr["TruthVertex_nTracks"][e])
        zt, nt = z[n >= 2], n[n >= 2]
        zc = c["truth_ge2_z"][c["truth_ge2_off"][i] : c["truth_ge2_off"][i + 1]]
        if len(zc) != len(zt) or not np.allclose(np.sort(zc), np.sort(zt), atol=1e-4):
            raise AssertionError(f"event {e}: ROOT truth differs from the cache")
        peaks = c["pvf_z"][c["pvf_off"][i] : c["pvf_off"][i + 1]].astype(np.float64)
        _, ti = greedy_match(peaks, zt, args.window)
        m = np.zeros(len(zt), dtype=bool)
        m[ti] = True
        unmatched.append(nt[~m])
        matched.append(nt[m])
    um, mt = np.concatenate(unmatched), np.concatenate(matched)
    res = {
        "n_events": int(len(ev)),
        "window_mm": args.window,
        "n_truth": int(len(um) + len(mt)),
        "one_to_one_eff": len(mt) / (len(um) + len(mt)),
        "n_unmatched": int(len(um)),
        "median_ntrk_unmatched": float(np.median(um)),
        "median_ntrk_matched": float(np.median(mt)),
        "frac_unmatched_ntrk2": float(np.mean(um == 2)),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
