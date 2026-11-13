"""Low pile-up (Run 2 MC) GNN chain: per-event metrics versus t, and AMVF on the same events.

Inputs are the inference graphs built by ``gnn.data.pvf_to_graphs --regime lowpu``
from the PVF-e400 histograms of the 2,550 Run 2 MC test events, and the
event-keyed TTVA h5 (``recoTracks_incamvfassoc.h5``) for truth and AMVF.
The GNN runs once per event; every t is a replay of the top-1 edges.

Per event, per t (and once for AMVF) one integer record ``FIELDS``:

* track-purity categories (``classify_assignments``, the code behind every
  earlier low pile-up table) on the vertices with >= ``--min-tracks``
  assigned tracks (default 2, decided 12 Nov 2026; 1 is the July drop-empty
  convention); tracks on dropped vertices count as unassigned: Clean / Merged / Split / Fake,
  n_reco, n_truth (truth vertices with >= 2 tracks);
* track counts (``threshold_scan.track_assignment_counts``): correct,
  assigned with a truth label, assigned, and all truth-labelled tracks;
* HS identification (the July 2026 / ``hs_id_pu200`` convention): the vertex
  with the largest sum pT^2 of its assigned tracks is correct when the
  plurality truth vertex of its truth-labelled tracks is the HS
  (``pv_type == 1``). AMVF is judged the same way on its own associations
  (``hs_ok``) and, separately, through its stored primary (``hs_ok_primary``).

Run (repo root, inside tmux):
    PYTHONPATH=src python -u -m gnn.evaluation.lowpu_chain \\
        --graphs outputs/.../graphs_e400_lowpu.pt \\
        --weights model_weights/ttva_gnn_mu60_v2/ttva_gat_mu60_v2_aug_epoch_115.pyt \\
        --vf-pkl outputs/.../vf_mc/eval_results.pkl -d 2 --out outputs/.../gnn_mc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from gnn.evaluation.classification import classify_assignments, get_top_k_associations
from gnn.evaluation.hs_id_pu200 import dominant_truth, leading_vertex
from gnn.evaluation.threshold_scan import (
    _dominant_index,
    cache_scores,
    load_truth,
    matched_tracks_from_selection,
    track_assignment_counts,
)
from pv_finder.utils.working_points import get_regime

TTVA_H5 = "/share/lazy/qibinlei/recoTracks_incamvfassoc.h5"
QIBIN = "configs/qibin_test_main_indices_v2.p"
T_GRID: tuple[float, ...] = (
    0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96,
    0.97, 0.98, 0.985, 0.99, 0.995, 0.999,
)  # fmt: skip
FIELDS: tuple[str, ...] = (
    "clean", "merged", "split", "fake", "n_reco", "n_truth",
    "n_correct", "n_assigned_truth", "n_assigned", "n_labelled",
    "hs_ok", "hs_ok_primary", "n_empty", "n_small",
)  # fmt: skip
# Decided 12 Nov 2026: a reconstructed vertex counts only with >= 2 tracks,
# for PV-Finder + GNN and AMVF alike. --min-tracks 1 gives the July convention.
MIN_TRACKS = 2
NTRK_EDGES = np.array([2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 70, 100, 1000])
MU_BINS: dict[str, tuple[float, float]] = {
    "inclusive": (-1.0, 1e9), "0-20": (-1.0, 20.0), "20-40": (20.0, 40.0),
    "40-60": (40.0, 60.0), "60-80": (60.0, 1e9),
}  # fmt: skip


def sha256(path: str | Path) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_hs_and_amvf(
    h5_path: str, event_keys: list[str]
) -> tuple[list[int], list[list[np.ndarray]], list[int]]:
    """Per event: HS truth index, AMVF per-vertex track lists, AMVF primary index."""
    hs, amvf, primary = [], [], []
    with h5py.File(h5_path, "r") as f:
        for key in event_keys:
            pv_type = f["pv_type"][key][:]
            cand = np.nonzero(pv_type == 1)[0]
            if len(cand) != 1:
                raise ValueError(f"{key}: {len(cand)} HS truth vertices")
            hs.append(int(cand[0]))
            nt = f["reco_pv_ntracks"][key][:].astype(int)
            at = f["reco_pv_assoc_tracks"][key][:].astype(int)
            bounds = np.concatenate([[0], np.cumsum(nt)])
            amvf.append(
                [np.unique(at[bounds[v] : bounds[v + 1]]) for v in range(len(nt))]
            )
            rtype = f["reco_pv_type"][key][:]
            prim = np.nonzero(rtype == 1)[0]
            primary.append(int(prim[0]) if len(prim) else -1)
    return hs, amvf, primary


def event_record(
    lists: list[np.ndarray], truth_ev: dict[str, Any], hs: int, primary: int = -2,
    min_tracks: int = MIN_TRACKS,
) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """One integer record (``FIELDS``) for one event and one vertex list.

    Vertices with fewer than ``min_tracks`` tracks are dropped before
    classification; their tracks count as unassigned. ``n_empty`` counts the
    dropped vertices with no track, ``n_small`` those with 1 .. min_tracks - 1.
    ``primary`` (AMVF only) is the index, in ``lists``, of the stored primary
    vertex; -2 means "not applicable".
    Also returns the histogram over ``NTRK_EDGES`` of the track counts of the
    truth vertices (>= 2 tracks) that are the dominant truth of a Clean vertex.
    """
    keep = [i for i, m in enumerate(lists) if len(m) >= max(min_tracks, 1)]
    nonempty = [lists[i] for i in keep]
    n_empty = sum(1 for m in lists if len(m) == 0)
    rows, info = classify_assignments(
        nonempty, truth_ev["pt"], truth_ev["tti"], truth_ev["tpi"],
        truth_ev["truth_pvs_count"],
    )  # fmt: skip
    counts = track_assignment_counts(nonempty, info, truth_ev["track_truth"])
    lead = leading_vertex(nonempty, truth_ev["pt"])
    tt = truth_ev["track_truth"]
    hs_ok = int(lead >= 0 and dominant_truth(nonempty[lead], tt) == hs)
    hs_ok_primary = 0
    if primary >= 0 and len(lists[primary]) >= max(min_tracks, 1):
        hs_ok_primary = int(dominant_truth(lists[primary], tt) == hs)
    rec = [
        *rows, counts["n_correct"], counts["n_assigned_truth"], counts["n_assigned"],
        int((tt >= 0).sum()), hs_ok, hs_ok_primary, n_empty,
        len(lists) - len(nonempty) - n_empty,
    ]  # fmt: skip
    ntrk = np.bincount(truth_ev["tpi"].astype(np.int64))
    clean_truth = [
        _dominant_index(v["primary_truth_pv"])
        for v in info
        if v["classification"] == "Clean"
    ]
    matched = ntrk[clean_truth] if clean_truth else np.zeros(0, dtype=np.int64)
    hist = np.histogram(matched[matched >= 2], bins=NTRK_EDGES)[0]
    return np.asarray(rec, dtype=np.int64), hist


def truth_ntrk_hist(truth_ev: dict[str, Any]) -> np.ndarray:
    """Histogram over ``NTRK_EDGES`` of all truth vertices with >= 2 tracks."""
    ntrk = np.bincount(truth_ev["tpi"].astype(np.int64))
    return np.histogram(ntrk[ntrk >= 2], bins=NTRK_EDGES)[0]


def gnn_lists(cache_ev: dict[str, Any], t: float) -> list[np.ndarray]:
    """Top-1 edge per track kept if its score >= t, grouped per PV node."""
    if cache_ev["n_pvs"] == 0:
        return []
    sel = get_top_k_associations(
        cache_ev["scores"], cache_ev["edge_index"], k=1, threshold=t
    )
    return matched_tracks_from_selection(cache_ev["edge_index"], sel, cache_ev["n_pvs"])


def ratios(rec: np.ndarray) -> dict[str, float]:
    """Ratios of sums over the events of ``rec`` (n_events, len(FIELDS))."""
    s = dict(zip(FIELDS, rec.sum(axis=0).astype(float)))
    n_ev = float(len(rec))
    eff = s["n_correct"] / s["n_labelled"] if s["n_labelled"] else 0.0
    pur = s["n_correct"] / s["n_assigned_truth"] if s["n_assigned_truth"] else 0.0
    nr = s["n_reco"] or 1.0
    return {
        "clean_per_truth": s["clean"] / s["n_truth"],
        "clean_per_reco": s["clean"] / nr,
        "merged_per_reco": s["merged"] / nr,
        "split_per_reco": s["split"] / nr,
        "fake_per_reco": s["fake"] / nr,
        "reco_per_truth": s["n_reco"] / s["n_truth"],
        "trk_eff": eff,
        "trk_pur": pur,
        "trk_f1": 2 * eff * pur / (eff + pur) if eff + pur else 0.0,
        "hs_id": s["hs_ok"] / n_ev,
        "hs_id_primary": s["hs_ok_primary"] / n_ev,
    }


def summarize(
    recs: dict[str, np.ndarray], mu: np.ndarray, n_boot: int, seed: int = 12
) -> dict[str, Any]:
    """Ratios per arm and mu bin, with paired event-bootstrap errors."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for bname, (lo, hi) in MU_BINS.items():
        idx = np.nonzero((mu >= lo) & (mu < hi))[0]
        reps = [rng.choice(idx, size=len(idx)) for _ in range(n_boot)]
        out[bname] = {"n_events": int(len(idx))}
        for arm, rec in recs.items():
            central = ratios(rec[idx])
            boot = [ratios(rec[r]) for r in reps]
            out[bname][arm] = {
                k: {"value": v, "err": float(np.std([b[k] for b in boot]))}
                for k, v in central.items()
            }
            out[bname][arm]["totals"] = {
                f: int(x) for f, x in zip(FIELDS, rec[idx].sum(axis=0))
            }
    return out


def check_bookkeeping(rec: np.ndarray, name: str) -> None:
    """Clean+Merged+Split+Fake = n_reco and track-count ordering, per event."""
    f = {k: rec[:, i] for i, k in enumerate(FIELDS)}
    if np.any(f["clean"] + f["merged"] + f["split"] + f["fake"] != f["n_reco"]):
        raise AssertionError(f"{name}: categories do not sum to n_reco")
    if np.any(f["n_correct"] > f["n_assigned_truth"]) or np.any(
        f["n_assigned_truth"] > f["n_assigned"]
    ):
        raise AssertionError(f"{name}: correct <= assigned_truth <= assigned violated")
    if np.any(f["n_correct"] > f["n_labelled"]):
        raise AssertionError(f"{name}: correct > labelled")


def _mu_from_pkl(path: str, n: int) -> tuple[np.ndarray, list[int]]:
    with open(path, "rb") as fh:
        r = pickle.load(fh)
    pe = r["per_event"]
    if len(pe) != n:
        raise ValueError(f"{path}: {len(pe)} events, graphs have {n}")
    return np.array([e["mu"] for e in pe], dtype=float), [
        int(e["event_idx"]) for e in pe
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument(
        "--vf-pkl", required=True, help="run_eval_pvf eval_results.pkl (mu)"
    )
    ap.add_argument("--h5", default=TTVA_H5)
    ap.add_argument("--indices", default=QIBIN)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("-d", "--device", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("-n", "--max-events", type=int, default=None)
    ap.add_argument("--min-tracks", type=int, default=MIN_TRACKS)
    args = ap.parse_args()
    dev = torch.device(
        f"cuda:{args.device}"
        if args.device >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    with open(args.indices, "rb") as fh:
        keys = [f"Event{int(i)}" for i in pickle.load(fh)]
    cache = cache_scores(args.graphs, args.weights, dev, args.max_events)
    keys = keys[: len(cache)]
    truth = load_truth(args.h5, keys)
    hs, amvf, primary = load_hs_and_amvf(args.h5, keys)
    for c, t in zip(cache, truth):
        if c["n_tracks"] != len(t["pt"]):
            raise ValueError("graph and h5 track counts differ: misaligned events")
    if args.max_events is None:
        mu, h5_events = _mu_from_pkl(args.vf_pkl, len(cache))
    else:
        mu, h5_events = np.zeros(len(cache)), []

    recs: dict[str, np.ndarray] = {}
    hists: dict[str, np.ndarray] = {
        "truth": np.sum([truth_ntrk_hist(tr) for tr in truth], axis=0)
    }
    for t in T_GRID:
        out = [
            event_record(gnn_lists(c, t), tr, h, min_tracks=args.min_tracks)
            for c, tr, h in zip(cache, truth, hs)
        ]
        recs[f"t{t:g}"] = np.stack([o[0] for o in out])
        hists[f"t{t:g}"] = np.sum([o[1] for o in out], axis=0)
    out = [
        event_record(a, tr, h, p, args.min_tracks)
        for a, tr, h, p in zip(amvf, truth, hs, primary)
    ]
    recs["amvf"] = np.stack([o[0] for o in out])
    hists["amvf"] = np.sum([o[1] for o in out], axis=0)
    for name, rec in recs.items():
        check_bookkeeping(rec, name)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "records.npz", fields=np.array(FIELDS), mu=mu,
        event_keys=np.array(keys), ntrk_edges=NTRK_EDGES, **recs,
        **{f"ntrk_matched_{k}": v for k, v in hists.items()},
    )  # fmt: skip
    regime = get_regime("lowpu")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
    )
    summary = {
        "provenance": {
            "git_hash": git.stdout.strip(),
            "src_dirty": bool(dirty.stdout.strip()),
            "argv": sys.argv,
            "regime": regime.to_dict(),
            "gnn_weights": {"path": args.weights, "sha256": sha256(args.weights)},
            "graphs": {"path": args.graphs, "sha256": sha256(args.graphs)},
            "density_checkpoint_sha256": sha256(regime.density.checkpoint),
            "event_keys_sha256": hashlib.sha256("|".join(keys).encode()).hexdigest(),
            "vf_h5_events_first_last": h5_events[:1] + h5_events[-1:],
            "convention": f"vertices with >= {args.min_tracks} tracks, MaxScore "
            "top-1, track-purity 70%",
            "min_tracks": args.min_tracks,
        },
        "t_grid": list(T_GRID),
        "fields": list(FIELDS),
        "bins": summarize(recs, mu, args.n_boot),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    inc = summary["bins"]["inclusive"]
    print(f"{'arm':>8} {'C/truth':>8} {'C/reco':>7} {'M/reco':>7} {'S/reco':>7} "
          f"{'F/reco':>7} {'trkF1':>7} {'HS-ID':>7}")  # fmt: skip
    for arm in recs:
        r = inc[arm]
        print(f"{arm:>8} " + " ".join(f"{100 * r[k]['value']:7.2f}" for k in (
            "clean_per_truth", "clean_per_reco", "merged_per_reco", "split_per_reco",
            "fake_per_reco", "trk_f1", "hs_id")))  # fmt: skip
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
