#!/usr/bin/env python3
"""Full-chain extras at the frozen HL-LHC working point (H3, H7): per-event replay.

One pass over the ``chain_inference`` chunks of r16443 entries 0-24,999 (s8
peaks, GNN v2 e527 top-1 edges) with the ROOT truth and AMVF branches. No
inference. Every arm is judged by ``gnn.evaluation.hs_metrics`` exactly as in
``gnn.evaluation.chain_t_metrics`` (labels, HS handling, drop-empty lists,
AMVF stored lists through ``np.unique``).

Arms (vertex lists with >= 1 and >= 2 assigned tracks):
  gnn_t         GNN top-1 edge kept if score >= t (t from the regime, 0.97)
  gnn_t0        GNN top-1 edge, no score cut
  z0_nearest    nearest peak in |z0 - z|
  z0_3sigma     nearest peak if |z0 - z| < 3 sigma(z0)
  eq4_s<smax>   Eq. (4): smallest s = |z0 - z| / sqrt(sigma(z0)^2 + sigma_PV^2)
                among the track's k = 20 graph candidates, kept if s < smax;
                sigma_PV = the peak width the chain graph gives the GNN
  truth_guided  each labelled track to the peak nearest its truth vertex if
                within the window (MC-only diagnostic; = the chain-gap oracle)
  amvf          stored AMVF track lists

Outputs (``--out``): ``chain_extras_events.npz`` (per event and arm counts,
event fields, per truth vertex with n_trk >= 2 records) and ``.json``
metadata with the validation-gate results. Gates are asserted.

    PYTHONPATH=src python -u scripts/hllhc_final_chain_extras.py \
        --infer outputs/11_12_2026_output/gnn_v2_freeze/infer/test_s8 \
        --ref outputs/11_12_2026_output/gnn_v2_freeze/metrics/test_s8_v2e527.npz \
        --out outputs/11_12_2026_output/hllhc_final/chain_extras --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.diagnostics.chain_gap_decomposition import greedy_match  # noqa: E402
from gnn.evaluation import hs_metrics as hm  # noqa: E402
from gnn.evaluation.chain_inference import KNN, git_state, sha256  # noqa: E402
from pv_finder.diagnostics.amvf_fair_comparison.matching import (  # noqa: E402
    match_vertices,
)
from pv_finder.utils.working_points import get_regime  # noqa: E402

REGIME = get_regime("hllhc")
WINDOW = REGIME.match_window_mm  # 0.17 mm
T_CUT = float(REGIME.gnn.t)  # 0.97
GNN_KEY = "v2e527"
S_MAX = (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, np.inf)
Z0_NSIGMA = 3.0  # z0_3sigma baseline (hs_chain_study)
DENSITY_HALF_MM = 2.0  # local density: truth vertices within +-2 mm, per 4 mm
ARMS = (
    "gnn_t", "gnn_t0", "z0_nearest", "z0_3sigma",
    *(f"eq4_s{s:g}" for s in S_MAX), "truth_guided", "amvf",
)  # fmt: skip
MIN_TRACKS = (1, 2)
COUNT_FIELDS = (
    "clean", "merged", "split", "fake", "n_reco", "n_correct",
    "n_assigned_truth", "n_assigned", "wrong", "n_assigned_unique",
    "n_labelled_unassigned", "n_unlabelled_unassigned", "n_lost_unassigned",
    "n_lost_unassigned_out", "clean_dom_ge2",
)  # fmt: skip
EVENT_FIELDS = (
    "entry", "mu", "n_tracks", "n_labelled", "n_unlabelled", "n_truth_all",
    "n_truth_ge2", "n_peaks", "n_amvf", "n_amvf2", "has_hs", "mv_found_pvf",
    "mv_found_amvf", "o2o_pvf", "o2o_amvf", "o2o_gnn", "n_amvf_multi",
    "knn_violations", "n_zero_sigma_edges",
)  # fmt: skip
TRUTH_FIELDS = (
    "row", "entry", "mu", "n_trk", "is_hs", "density", "dz_nearest",
    "found_pvf", "found_amvf", "o2o_pvf", "o2o_amvf", "o2o_gnn", "o2o_peak_sigma",
)  # fmt: skip
BRANCHES = [
    "ActualNumOfInt", "RecoTrack_z0", "RecoTrack_ErrZ0", "RecoTrack_pT",
    "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_assocTracks",
    "TruthVertex_type", "RecoVertex_z", "RecoVertex_assocTracks",
]  # fmt: skip
GATE1_FIELDS = (
    "clean", "merged", "split", "fake", "n_reco", "n_correct", "n_assigned_truth",
    "n_assigned",
)  # fmt: skip


def best_significance(
    z0: np.ndarray, z0_err: np.ndarray, peak_z: np.ndarray, peak_sigma: np.ndarray,
    knn: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:  # fmt: skip
    """Eq. (4) on the chain graph: per track, the candidate with the smallest s.

    Candidates are the track's ``knn`` nearest peaks in |z0 - z|, chosen with
    the same ``argpartition`` call as ``create_inference_graph`` (all peaks
    when there are at most ``knn``); ``s`` is the absolute value of the GNN's
    first edge feature (``_compute_edge_attributes``: 0 where the combined
    sigma is 0). Pass the graph's dtypes (float32) for an identical graph.
    Ties go to the lowest peak index.

    Returns (best peak, its s, the candidate matrix, edges with zero sigma);
    tracks get peak -1 and s = inf when there are no peaks.
    """
    n, m = len(z0), len(peak_z)
    if m == 0:
        empty = np.zeros((n, 0), np.int64)
        return np.full(n, -1, np.int64), np.full(n, np.inf), empty, 0
    dz = np.abs(z0[:, None] - peak_z[None, :])
    if 0 < knn < m:
        cand = np.sort(np.argpartition(dz, knn - 1, axis=1)[:, :knn], axis=1)
    else:
        cand = np.broadcast_to(np.arange(m), (n, m))
    diff = z0[:, None] - peak_z[cand]
    sig = np.sqrt(z0_err[:, None] ** 2 + peak_sigma[cand] ** 2)
    s = np.abs(np.divide(diff, sig, out=np.zeros_like(diff), where=sig != 0))
    j = s.argmin(axis=1)
    rows = np.arange(n)
    return cand[rows, j].astype(np.int64), s[rows, j], cand, int((sig == 0).sum())


def no_vertex_near(
    track_truth: np.ndarray, truth_z: np.ndarray, vertex_z: np.ndarray, window: float
) -> np.ndarray:
    """Per track: labelled, and no vertex within ``window`` of its truth vertex.

    "Within" is strict (< window), as in ``assign_truth_guided``; unlabelled
    tracks are False.
    """
    track_truth = np.asarray(track_truth, dtype=np.int64)
    out = np.zeros(len(track_truth), dtype=bool)
    lab = np.flatnonzero(track_truth >= 0)
    if len(vertex_z) == 0:
        out[lab] = True
        return out
    tz = np.asarray(truth_z, dtype=np.float64)[track_truth[lab]]
    d = np.abs(tz[:, None] - np.asarray(vertex_z, dtype=np.float64)[None, :]).min(1)
    out[lab] = d >= window
    return out


def arm_counts(
    lists: list[np.ndarray], vz: np.ndarray, pt: np.ndarray, truth: np.ndarray,
    tz: np.ndarray, tn: np.ndarray, lost_cand: np.ndarray,
) -> tuple[np.ndarray, hm.VertexClassification]:  # fmt: skip
    """COUNT_FIELDS for one vertex list, and its classification."""
    cls = hm.classify_vertices(lists, pt, truth)
    nt = len(truth)
    n_correct = n_at = n_a = 0
    assigned = np.zeros(nt, dtype=bool)
    for v, x in enumerate(lists):
        tt = truth[x]
        n_a += len(x)
        n_at += int((tt >= 0).sum())
        assigned[x] = True
        if cls.dominant[v] >= 0:
            n_correct += int((tt == cls.dominant[v]).sum())
    lab = truth >= 0
    free = ~assigned
    lost_out = no_vertex_near(truth, tz, vz, WINDOW)
    clean_dom = cls.dominant[cls.labels == hm.CLEAN]
    return np.array(
        [*cls.counts(), len(lists), n_correct, n_at, n_a, n_at - n_correct,
         int(assigned.sum()), int((lab & free).sum()), int((~lab & free).sum()),
         int((lost_cand & free).sum()), int((lost_out & free).sum()),
         int((tn[clean_dom] >= 2).sum())],
        dtype=np.int64,
    ), cls  # fmt: skip


def truth_records(
    tz: np.ndarray, tn: np.ndarray, hs: int, peaks: np.ndarray, psig: np.ndarray,
    amvf2: np.ndarray, gnn_z: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:  # fmt: skip
    """Per truth vertex with n_trk >= 2: TRUTH_FIELDS[3:] and the 1-1 counts."""
    idx = np.flatnonzero(tn >= 2)
    z2 = tz[idx]
    d_all = np.abs(z2[:, None] - tz[None, :])
    density = ((d_all < DENSITY_HALF_MM).sum(1) - 1) / (2 * DENSITY_HALF_MM)
    d2 = np.abs(z2[:, None] - z2[None, :])
    np.fill_diagonal(d2, np.inf)
    dz_near = d2.min(1) if len(z2) > 1 else np.full(len(z2), np.inf)

    def found(vz: np.ndarray) -> np.ndarray:  # <=, as match_vertices
        if len(vz) == 0 or len(z2) == 0:
            return np.zeros(len(z2), bool)
        return (np.abs(z2[:, None] - vz[None, :]) <= WINDOW).any(1)

    o2o, counts = {}, {}
    for key, vz in (("pvf", peaks), ("amvf", amvf2), ("gnn", gnn_z)):
        p_i, t_i = greedy_match(vz, z2, WINDOW)
        flag = np.zeros(len(z2), bool)
        flag[t_i] = True
        if len(t_i) != hm.greedy_one_to_one(vz, z2, WINDOW):
            raise AssertionError("greedy_match and greedy_one_to_one disagree")
        o2o[key], counts[key] = flag, len(t_i)
        if key == "pvf":
            sig = np.full(len(z2), np.nan)
            sig[t_i] = psig[p_i]
    rec = np.stack(
        [tn[idx], idx == hs, density, dz_near, found(peaks), found(amvf2),
         o2o["pvf"], o2o["amvf"], o2o["gnn"], sig], axis=1,
    ).astype(np.float64)  # fmt: skip
    return rec, counts


def process_chunk(path: str, root: str) -> dict:
    """Per-event and per-truth-vertex records for one inference chunk."""
    keys = (
        "entries", "mu", "n_tracks", "track_offsets", "peak_offsets", "peak_z",
        "peak_sigma", f"targets_{GNN_KEY}", f"scores_{GNN_KEY}",
    )  # fmt: skip
    with np.load(path) as f:
        z = {k: f[k] for k in keys}
    entries = z["entries"]
    lo_e, hi_e = int(entries[0]), int(entries[-1]) + 1
    if not np.array_equal(entries, np.arange(lo_e, hi_e)):
        raise ValueError(f"{path}: entries are not contiguous")
    toff, poff = z["track_offsets"], z["peak_offsets"]
    tg_all, sc_all = z[f"targets_{GNN_KEY}"], z[f"scores_{GNN_KEY}"]
    tree = uproot.open(root)["PVFinderData"]
    data = tree.arrays(BRANCHES, entry_start=lo_e, entry_stop=hi_e, library="ak")
    n = len(entries)
    counts = np.zeros((n, len(ARMS), len(MIN_TRACKS), len(COUNT_FIELDS)), np.int64)
    ev_out = np.zeros((n, len(EVENT_FIELDS)))
    t_rows, t_clean = [], []
    for i, e in enumerate(entries):
        ev = data[int(e) - lo_e]
        if float(ev["ActualNumOfInt"]) != z["mu"][i]:
            raise ValueError(f"entry {e}: mu differs from the inference record")
        z0_32 = ak.to_numpy(ev["RecoTrack_z0"]).astype(np.float32)
        ez0_32 = ak.to_numpy(ev["RecoTrack_ErrZ0"]).astype(np.float32)
        z0, ez0 = z0_32.astype(np.float64), ez0_32.astype(np.float64)
        pt = ak.to_numpy(ev["RecoTrack_pT"]).astype(np.float64)
        nt = len(z0)
        if nt != z["n_tracks"][i]:
            raise ValueError(f"entry {e}: track count differs")
        tz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        truth = hm.track_truth_labels(
            ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, nt
        )
        hs_c = np.flatnonzero(ak.to_numpy(ev["TruthVertex_type"]) == 1)
        if len(hs_c) > 1:
            raise ValueError(f"entry {e}: {len(hs_c)} HS truth vertices")
        h = int(hs_c[0]) if len(hs_c) == 1 else -2  # as chain_t_metrics
        pz32 = z["peak_z"][poff[i] : poff[i + 1]]
        ps32 = z["peak_sigma"][poff[i] : poff[i + 1]]
        peaks = pz32.astype(np.float64)
        tg, sc = tg_all[toff[i] : toff[i + 1]], sc_all[toff[i] : toff[i + 1]]
        best, s_best, cand, n_zero = best_significance(z0_32, ez0_32, pz32, ps32, KNN)
        # Gate: every GNN top-1 target is one of the track's graph candidates
        if len(peaks):
            ok = (cand == tg[:, None]).any(1)
        else:
            ok = tg < 0
        knn_bad = int((~ok).sum())
        alists = [np.unique(ak.to_numpy(x).astype(np.int64)) for x in ev["RecoVertex_assocTracks"]]  # fmt: skip
        if any(len(x) and (x.min() < 0 or x.max() >= nt) for x in alists):
            raise ValueError(f"entry {e}: AMVF track index out of range")
        az = ak.to_numpy(ev["RecoVertex_z"]).astype(np.float64)
        an = np.array([len(x) for x in alists], dtype=np.int64)
        multi = np.bincount(np.concatenate([*alists, np.zeros(0, np.int64)]), minlength=nt)  # fmt: skip
        lost_pvf = no_vertex_near(truth, tz, peaks, WINDOW)
        lost_amvf = no_vertex_near(truth, tz, az[an >= 1], WINDOW)
        clean_flags = np.zeros((len(MIN_TRACKS), int((tn >= 2).sum()), len(ARMS)), dtype=bool)  # fmt: skip
        idx2 = np.flatnonzero(tn >= 2)
        gnn_keep_z = peaks[:0]
        for a, arm in enumerate(ARMS):
            if arm == "amvf":
                all_lists, all_z, lost = alists, az, lost_amvf
            else:
                if arm == "gnn_t":
                    tgt = hm.assign_from_scores(tg, sc, T_CUT)
                elif arm == "gnn_t0":
                    tgt = hm.assign_from_scores(tg, sc, 0.0)
                elif arm == "z0_nearest":
                    tgt = hm.assign_nearest_z0(z0, peaks)
                elif arm == "z0_3sigma":
                    tgt = hm.assign_nearest_z0(z0, peaks, ez0, Z0_NSIGMA)
                elif arm == "truth_guided":
                    tgt = hm.assign_truth_guided(truth, tz, peaks, WINDOW)
                else:
                    smax = S_MAX[ARMS.index(arm) - ARMS.index("eq4_s1")]
                    tgt = np.where(s_best < smax, best, -1)
                all_lists, all_z, lost = hm.lists_from_targets(tgt, len(peaks)), peaks, lost_pvf  # fmt: skip
            for m, mt in enumerate(MIN_TRACKS):
                keep = np.flatnonzero([len(x) >= mt for x in all_lists])
                lists = [all_lists[j] for j in keep]
                vz = all_z[keep]
                counts[i, a, m], cls = arm_counts(lists, vz, pt, truth, tz, tn, lost)
                dom = np.zeros(len(tz), dtype=bool)
                dom[cls.dominant[cls.labels == hm.CLEAN]] = True
                clean_flags[m, :, a] = dom[idx2]
                if m == 0 and arm == "gnn_t":
                    gnn_keep_z = vz
        rec, o2o = truth_records(tz, tn, h, peaks, ps32.astype(np.float64), az[an >= 2], gnn_keep_z)  # fmt: skip
        mv_p = match_vertices(tz[tn >= 2], peaks, WINDOW).n_truth_matched
        mv_a = match_vertices(tz[tn >= 2], az[an >= 2], WINDOW).n_truth_matched
        n_lab = int((truth >= 0).sum())
        ev_out[i] = [
            e, z["mu"][i], nt, n_lab, nt - n_lab, len(tz), len(idx2), len(peaks),
            int((an >= 1).sum()), int((an >= 2).sum()), int(h >= 0), mv_p, mv_a,
            o2o["pvf"], o2o["amvf"], o2o["gnn"], int((multi > 1).sum()), knn_bad,
            n_zero,
        ]  # fmt: skip
        head = np.column_stack([np.full(len(idx2), i), np.full(len(idx2), e), np.full(len(idx2), z["mu"][i])])  # fmt: skip
        t_rows.append(np.hstack([head, rec]))
        t_clean.append(clean_flags)
    return {
        "path": path, "counts": counts, "events": ev_out,
        "truth": np.concatenate(t_rows),
        "truth_clean": np.concatenate([c[0] for c in t_clean]),
        "truth_clean2": np.concatenate([c[1] for c in t_clean]),
    }  # fmt: skip


def check_inputs(meta: dict) -> dict:
    """The inference used the frozen regime; re-hash the checkpoints."""
    dm, g = meta["density"], meta["gnn"][GNN_KEY]
    if dm["checkpoint"] != REGIME.density.checkpoint or g["path"] != REGIME.gnn.checkpoint:  # fmt: skip
        raise RuntimeError("inference checkpoints differ from the frozen regime")
    if dm["peaks"] != asdict(REGIME.peaks):
        raise RuntimeError(f"peak cuts {dm['peaks']} differ from the regime")
    if meta["knn"] != KNN or meta["regime_window_mm"] != WINDOW:
        raise RuntimeError("kNN or window differs from the regime")
    if meta["entries_spec"] != "0:25000" or "r16443" not in meta["root"]:
        raise RuntimeError("not r16443 entries 0-24,999 (GNN-held-out events)")
    out = {}
    for key, rel, ref in (("density", dm["checkpoint"], dm["sha256"]), ("gnn", g["path"], g["sha256"])):  # fmt: skip
        h = sha256(REPO / rel)
        if h != ref:
            raise RuntimeError(f"{rel}: sha256 {h} != inference record {ref}")
        out[key] = {"path": rel, "sha256": h, "verified": True}
    return out


def gate1(counts: np.ndarray, events: np.ndarray, ref_path: Path) -> dict:
    """Shared per-event counts equal chain_t_metrics, event by event."""
    with np.load(ref_path) as f:
        rc, rev = f["counts"], f["events"]
    rmeta = json.loads(ref_path.with_suffix(".json").read_text())
    if rmeta["gnn"] != GNN_KEY:
        raise RuntimeError("reference metrics are for another GNN")
    r_entry = rev[:, rmeta["event_fields"].index("entry")].astype(np.int64)
    pos = {int(x): k for k, x in enumerate(r_entry)}
    rows = np.array([pos[int(x)] for x in events[:, EVENT_FIELDS.index("entry")]])
    t_grid = rmeta["t_grid"]
    pairs = {"gnn_t": t_grid.index(T_CUT), "gnn_t0": t_grid.index(0.0), "amvf": len(t_grid)}  # fmt: skip
    out = {}
    for arm, ra in pairs.items():
        for m in range(len(MIN_TRACKS)):
            for fld in GATE1_FIELDS:
                mine = counts[:, ARMS.index(arm), m, COUNT_FIELDS.index(fld)]
                ref = rc[rows, ra, m, rmeta["count_fields"].index(fld)]
                out[f"{arm}/min{MIN_TRACKS[m]}/{fld}"] = int((mine != ref).sum())
    return {"n_events": len(rows), "mismatched_events": out, "pass": not any(out.values())}  # fmt: skip


def gates_2_to_5(counts: np.ndarray, events: np.ndarray, truth: np.ndarray, clean: np.ndarray, clean2: np.ndarray) -> dict:  # fmt: skip
    """Bookkeeping (2), Eq. (1) identity (3), Clean per truth vertex (4), kNN (5)."""
    c = {f: counts[..., COUNT_FIELDS.index(f)] for f in COUNT_FIELDS}
    ev = {f: events[:, EVENT_FIELDS.index(f)].astype(np.int64) for f in EVENT_FIELDS}
    g2 = {
        "categories_sum_to_n_reco": bool(
            np.array_equal(
                c["clean"] + c["merged"] + c["split"] + c["fake"], c["n_reco"]
            )
        ),  # fmt: skip
        "correct_le_assigned_truth": bool(
            np.all(c["n_correct"] <= c["n_assigned_truth"])
        ),  # fmt: skip
        "assigned_truth_le_labelled": bool(
            np.all(c["n_assigned_truth"] <= ev["n_labelled"][:, None, None])
        ),  # fmt: skip
        "assigned_le_tracks": bool(
            np.all(c["n_assigned"] <= ev["n_tracks"][:, None, None])
        ),  # fmt: skip
        "unique_le_assigned": bool(np.all(c["n_assigned_unique"] <= c["n_assigned"])),
        "lost_le_labelled_unassigned": bool(
            np.all(c["n_lost_unassigned"] <= c["n_labelled_unassigned"])
        ),  # fmt: skip
    }
    tg = ARMS.index("truth_guided")
    g2["truth_guided_unassigned_all_lost"] = bool(np.array_equal(c["n_lost_unassigned"][:, tg, 0], c["n_labelled_unassigned"][:, tg, 0]))  # fmt: skip
    row = truth[:, TRUTH_FIELDS.index("row")].astype(np.int64)
    n_ev = len(events)

    def per_event(col: str) -> np.ndarray:
        return np.bincount(row, weights=truth[:, TRUTH_FIELDS.index(col)], minlength=n_ev).astype(np.int64)  # fmt: skip

    g3 = {
        "pvf_mismatched_events": int(
            (per_event("found_pvf") != ev["mv_found_pvf"]).sum()
        ),  # fmt: skip
        "amvf_mismatched_events": int(
            (per_event("found_amvf") != ev["mv_found_amvf"]).sum()
        ),  # fmt: skip
        "n_events": n_ev,
    }
    g3["pass"] = g3["pvf_mismatched_events"] == 0 and g3["amvf_mismatched_events"] == 0
    g4 = {}
    for a, arm in enumerate(ARMS):
        per = np.bincount(row, weights=clean[:, a], minlength=n_ev).astype(np.int64)
        per2 = np.bincount(row, weights=clean2[:, a], minlength=n_ev).astype(np.int64)
        g4[arm] = {
            "mismatched_events": int(
                (per != c["clean_dom_ge2"][:, a, 0]).sum()
                + (per2 != c["clean_dom_ge2"][:, a, 1]).sum()
            ),  # fmt: skip
            "clean_total": int(c["clean"][:, a, 0].sum()),
            "clean_dominant_ntrk_lt2": int(
                (c["clean"][:, a, 0] - c["clean_dom_ge2"][:, a, 0]).sum()
            ),  # fmt: skip
            "clean_dominant_ntrk_lt2_min2": int(
                (c["clean"][:, a, 1] - c["clean_dom_ge2"][:, a, 1]).sum()
            ),  # fmt: skip
        }
    g5 = {"knn_violations": int(ev["knn_violations"].sum()), "zero_sigma_edges": int(ev["n_zero_sigma_edges"].sum())}  # fmt: skip
    return {
        "gate2_bookkeeping": {**g2, "pass": all(g2.values())},
        "gate3_eq1_identity": g3,
        "gate4_clean_per_truth": {
            **g4,
            "pass": all(v["mismatched_events"] == 0 for v in g4.values()),
        },  # fmt: skip
        "gate5_same_graph": {**g5, "pass": g5["knn_violations"] == 0},
    }


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--infer", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path, help="chain_t_metrics npz")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-chunks", type=int, default=0, help="test runs only")
    a = ap.parse_args()
    meta_in = json.loads((a.infer / "metadata.json").read_text())
    ckpts = check_inputs(meta_in)
    paths = sorted(str(p) for p in a.infer.glob("chunk_*.npz"))
    if a.max_chunks:
        paths = paths[: a.max_chunks]
    t0 = time.monotonic()
    with ProcessPoolExecutor(min(a.workers, 12)) as pool:
        futs = [pool.submit(process_chunk, p, meta_in["root"]) for p in paths]
        parts = []
        for k, f in enumerate(futs):
            parts.append(f.result())
            print(f"{k + 1}/{len(paths)} chunks, {time.monotonic() - t0:.0f} s", flush=True)  # fmt: skip
    offs = np.cumsum([0] + [len(p["events"]) for p in parts])
    for p, o in zip(parts, offs):
        p["truth"][:, 0] += o  # chunk-local event row -> global row
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("counts", "events", "truth", "truth_clean", "truth_clean2")}  # fmt: skip
    if not a.max_chunks and len(cat["events"]) != meta_in["n_entries"]:
        raise RuntimeError("event count differs from the inference metadata")
    gates = {"gate1_vs_chain_t_metrics": gate1(cat["counts"], cat["events"], a.ref)}
    gates.update(gates_2_to_5(cat["counts"], cat["events"], cat["truth"], cat["truth_clean"], cat["truth_clean2"]))  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    npz = a.out / ("chain_extras_events.npz" if not a.max_chunks else "chain_extras_events_test.npz")  # fmt: skip
    np.savez_compressed(npz, **cat)
    meta = {
        "infer": str(a.infer), "ref_metrics": str(a.ref), "gnn": GNN_KEY,
        "checkpoints": ckpts, "regime": REGIME.to_dict(), "window_mm": WINDOW,
        "t": T_CUT, "s_max": [float(s) for s in S_MAX], "z0_nsigma": Z0_NSIGMA,
        "knn": KNN, "density_half_mm": DENSITY_HALF_MM, "arms": ARMS,
        "min_tracks": MIN_TRACKS, "count_fields": COUNT_FIELDS,
        "event_fields": EVENT_FIELDS, "truth_fields": TRUTH_FIELDS,
        "gates": gates, "git": git_state(), "script_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(Path(hm.__file__)),
        "events_sha256": hashlib.sha256(cat["events"][:, 0].astype(np.int64).tobytes()).hexdigest(),  # fmt: skip
        "runtime_s": time.monotonic() - t0,
    }  # fmt: skip
    npz.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float) + "\n")  # fmt: skip
    print(json.dumps(gates, indent=1, default=float), flush=True)
    failed = [k for k, v in gates.items() if not v["pass"]]
    if failed:
        raise AssertionError(f"validation gates failed: {failed}")
    print("done, all gates pass", flush=True)


if __name__ == "__main__":
    main()
