#!/usr/bin/env python3
"""Inventory of candidate ATLAS PVFinderData ROOT files: is there Run 3 MC?

For every file given on the command line this prints and stores (JSON):
size, tree entries, how many events carry truth vertices (``NumTruthVtx > 0``),
the pile-up range, beam-spot z and its length (``BeamPosSigmaZ``), the spread
of the stored vertex z positions, and every non-tree object in the file (the
only place a campaign / sqrt(s) tag could live, since the ntuple has no
run-number or channel branch).  Only scalar branches are read in full; the
jagged vertex branches are read on the first ``--n-jagged`` entries.

Usage:
    python -u scripts/data_mc_inventory.py --out inventory.json FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import uproot

_SCALARS = ("NumTruthVtx", "ActualNumOfInt", "BeamPosZ", "BeamPosSigmaZ")


def _stats(x: np.ndarray) -> dict[str, float]:
    """Min / mean / median / max of a 1-D array (NaN-safe, empty-safe)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return {}
    return {
        "min": float(np.nanmin(x)),
        "mean": float(np.nanmean(x)),
        "median": float(np.nanmedian(x)),
        "max": float(np.nanmax(x)),
    }


def _flat_scalar(arr: np.ndarray) -> np.ndarray:
    """Branches such as BeamPosZ are stored as length-1 vectors in some files."""
    if arr.dtype == object:
        return np.array([float(np.asarray(v).ravel()[0]) if len(v) else np.nan
                         for v in arr])  # fmt: skip
    return arr.astype(np.float64)


def inspect_file(path: str, n_jagged: int) -> dict:
    """Return the inventory record for one ROOT file."""
    real = os.path.realpath(path)
    rec: dict = {"path": path, "realpath": real,
                 "size_GB": round(os.path.getsize(real) / 1e9, 3)}  # fmt: skip
    f = uproot.open(path)
    classes = f.classnames()
    rec["objects"] = {k: c for k, c in classes.items() if c != "TTree"}
    trees = [k for k, c in classes.items() if c == "TTree"]
    rec["trees"] = trees
    if not trees:
        return rec
    t = f[trees[0]]
    rec["entries"] = int(t.num_entries)
    keys = set(t.keys())
    rec["has_truth_branches"] = "TruthVertex_z" in keys
    for b in _SCALARS:
        if b not in keys:
            continue
        v = _flat_scalar(t[b].array(library="np"))
        rec[b] = _stats(v)
        if b == "NumTruthVtx":
            rec["n_events_truth_gt0"] = int(np.sum(v > 0))
        if b == "ActualNumOfInt":
            rec["n_events_mu_gt1"] = int(np.sum(v > 1))
    jag = [b for b in ("TruthVertex_z", "TruthVertex_nTracks", "RecoVertex_z",
                       "RecoVertex_nTracks") if b in keys]  # fmt: skip
    a = t.arrays(jag, entry_stop=min(n_jagged, t.num_entries), library="np")
    if "TruthVertex_z" in a:
        tz = np.concatenate([np.asarray(x) for x in a["TruthVertex_z"]] or [[]])
        rec["truth_z_first_n"] = {
            "n_vtx": int(tz.size),
            **_stats(tz),
            "std": float(np.std(tz)) if tz.size else None,
        }
    if "RecoVertex_z" in a:
        rz = [np.asarray(z)[np.asarray(n) >= 2]
              for z, n in zip(a["RecoVertex_z"], a["RecoVertex_nTracks"])]  # fmt: skip
        rz_all = np.concatenate(rz) if rz else np.zeros(0)
        rec["amvf_ge2_z_first_n"] = {
            "n_vtx": int(rz_all.size),
            "std": float(np.std(rz_all)) if rz_all.size else None,
        }
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--n-jagged", type=int, default=3000)
    args = p.parse_args()
    out = []
    for fp in args.files:
        try:
            rec = inspect_file(fp, args.n_jagged)
        except Exception as exc:  # report unreadable files, do not abort
            rec = {"path": fp, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(rec, indent=None, default=str), flush=True)
        out.append(rec)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
