"""Truth-guided (oracle) track assignment on the frozen low pile-up peaks (note Sec. 7.6).

Each truth-labelled track goes to the PV-Finder peak nearest its own truth
vertex, if that peak is within ``--window`` of the truth vertex; other tracks
stay unassigned. The resulting per-peak track lists are classified with the
same code and vertex rule as the GNN chain (``gnn.evaluation.lowpu_chain``,
vertices with >= ``--min-tracks`` tracks). This is the low pile-up analogue of
the HL-LHC "truth-guided" row. MC truth only; no inference.

Peaks: the frozen e400 vertex-finding evaluation (``eval_results.pkl``,
integral 0.50, local centroid). Truth and tracks: the TTVA h5, event keys from
``configs/qibin_test_main_indices_v2.p`` (same order as the pkl; checked).

Usage (repo root, venv, PYTHONPATH=src):
    python scripts/note_lowpu_oracle.py \
        --vf-pkl outputs/11_12_2026_output/lowpu_final/vf_mc/eval_results.pkl \
        --out outputs/11_13_2026_output/note_edits/lowpu_oracle
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np

from gnn.evaluation.lowpu_chain import (
    QIBIN,
    TTVA_H5,
    check_bookkeeping,
    event_record,
    load_hs_and_amvf,
    ratios,
)
from gnn.evaluation.threshold_scan import load_truth


def oracle_lists(
    peaks: np.ndarray, truth_z: np.ndarray, track_truth: np.ndarray, window: float
) -> list[np.ndarray]:
    """Per-peak track lists: each labelled track to the peak nearest its truth vertex."""
    lists: list[list[int]] = [[] for _ in range(len(peaks))]
    if len(peaks) == 0:
        return []
    for v, zv in enumerate(truth_z):
        d = np.abs(peaks - zv)
        p = int(np.argmin(d))
        if d[p] < window:
            lists[p].extend(np.nonzero(track_truth == v)[0].tolist())
    return [np.unique(np.asarray(m, dtype=np.int64)) for m in lists]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vf-pkl", required=True)
    ap.add_argument("--h5", default=TTVA_H5)
    ap.add_argument("--indices", default=QIBIN)
    ap.add_argument("--window", type=float, default=0.17)
    ap.add_argument("--min-tracks", type=int, default=2)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.indices, "rb") as fh:
        keys = [f"Event{int(i)}" for i in pickle.load(fh)]
    with open(args.vf_pkl, "rb") as fh:
        vf = pickle.load(fh)
    if len(vf["per_event"]) != len(keys):
        raise ValueError("pkl and index file differ in length")
    truth = load_truth(args.h5, keys)
    hs, _, _ = load_hs_and_amvf(args.h5, keys)

    recs, n_frame_bad = [], 0
    with h5py.File(args.h5, "r") as f:
        for i, key in enumerate(keys):
            tz = f["pv_loc_z"][key][:].astype(np.float64)
            # Alignment check: the pkl truth list equals the h5 truth list.
            pkl_truth = np.sort(np.asarray(vf["truth_pvs_mm"][i], dtype=np.float64))
            if not np.allclose(pkl_truth, np.sort(tz), atol=1e-3):
                n_frame_bad += 1
            peaks = np.asarray(vf["pred_pvs_mm"][i], dtype=np.float64)
            lists = oracle_lists(peaks, tz, truth[i]["track_truth"], args.window)
            recs.append(
                event_record(lists, truth[i], hs[i], min_tracks=args.min_tracks)[0]
            )
    if n_frame_bad:
        raise AssertionError(f"{n_frame_bad} events: pkl truth differs from the h5")
    rec = np.stack(recs)
    check_bookkeeping(rec, "oracle")
    rng = np.random.default_rng(12)
    n = len(rec)
    boot = [ratios(rec[rng.integers(0, n, n)]) for _ in range(1000)]
    central = ratios(rec)
    res = {
        "window_mm": args.window,
        "min_tracks": args.min_tracks,
        "n_events": n,
        "totals": dict(
            zip(
                ("clean", "merged", "split", "fake", "n_reco", "n_truth"),
                rec[:, :6].sum(axis=0).tolist(),
            )
        ),  # fmt: skip
        "ratios": {
            k: {"value": v, "err": float(np.std([b[k] for b in boot]))}
            for k, v in central.items()
        },  # fmt: skip
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "oracle.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
