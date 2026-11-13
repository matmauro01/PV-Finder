#!/usr/bin/env python3
"""Supplementary HL-LHC studies with Trey's 12 Nov 2026 r16443 side ntuple (compute pass).

The old r16443 file stays the primary input (it is what the s8 + GNN v2 e527
chain ran on); the new file is joined to it by entry index as a side table.
For entries 0-24,999 this script

1. checks that the two files hold the same events, entry by entry (tracks,
   AMVF vertices and their TTVA links, mu, truth vertices with >= 2 tracks,
   the hard scatter), and fails on the first mismatching entry and branch;
2. computes, per event, the ITk-paper local pile-up density (ALL truth
   vertices of the new file within +-2 mm of the HS, HS excluded, / 4 mm)
   next to the old stored-truth density, and the HS-reconstruction flags;
3. classifies the vertices Clean / Merged / Split / Fake at the frozen t
   (``get_regime("hllhc")``) for PV-Finder + GNN and AMVF, keeping vertices
   with >= 1 and with >= 2 tracks (the same rule for both algorithms), with
   the unweighted same-rule taxonomy (``hs_metrics.classify_vertices``,
   asserted equal to ``chain_t_metrics``) and the weighted cross-checks
   (``weighted_classification``: AMVF fitTracks + fitWeights, GNN scores).

Output: ``per_event.npz`` + ``identity_check.json`` under ``--out``. The
tables and plots are made by ``hllhc_final_supplementary_plots.py``. No
inference; CPU only.

    PYTHONPATH=src venv/bin/python -u scripts/hllhc_final_supplementary.py \
        --out outputs/11_12_2026_output/hllhc_final/supplementary --workers 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.evaluation import hs_metrics as hm  # noqa: E402
from gnn.evaluation.chain_inference import git_state, sha256  # noqa: E402
from gnn.evaluation.weighted_classification import (  # noqa: E402
    classify_vertices_weighted,
)
from pv_finder.utils.working_points import get_regime  # noqa: E402

REGIME = get_regime("hllhc")
T = REGIME.gnn.t  # from the regime
HS_WIN = REGIME.hs_match_window_mm  # 0.1 mm
DENS_HALF = 2.0  # mm, ITk paper / Athena local pile-up density
GNN_KEY = "v2e527"
OLD_ROOT = REPO / (
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"
)
NEW_ROOT = Path(
    "/share/lazy/pv-finder/Paper_2026Nov12/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"
)
INFER = REPO / "outputs/11_12_2026_output/gnn_v2_freeze/infer/test_s8"
METRICS = REPO / "outputs/11_12_2026_output/gnn_v2_freeze/metrics/test_s8_v2e527.npz"
# Branches compared entry by entry (exact equality, order included).
SAME_SCALAR = ("ActualNumOfInt", "NumRecoTrks", "NumRecoVtx", "BeamPosX", "BeamPosY",
               "BeamPosZ", "BeamPosSigmaZ")  # fmt: skip
SAME_JAGGED = ("RecoTrack_z0", "RecoTrack_pT", "RecoTrack_d0", "RecoTrack_phi",
               "RecoTrack_theta", "RecoTrack_eta", "RecoTrack_ErrD0", "RecoTrack_ErrZ0",
               "RecoTrack_ErrD0Z0", "RecoVertex_x", "RecoVertex_y", "RecoVertex_z",
               "RecoVertex_nTracks", "RecoVertex_type")  # fmt: skip
# Empty in the old file (filled in the new one): counted, not compared.
OLD_EMPTY = ("RecoVertex_ErrX", "RecoVertex_ErrY", "RecoVertex_ErrZ")
TRUTH = ("TruthVertex_x", "TruthVertex_y", "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_type",
         "TruthVertex_assocTracks", "NumTruthVtx")  # fmt: skip
COMMON = (*SAME_SCALAR, *SAME_JAGGED, *OLD_EMPTY, "RecoVertex_assocTracks", *TRUTH)
NEW_ONLY = ("RecoVertex_fitTracks", "RecoVertex_fitWeights", "RecoVertex_nTracksAll",
            "RecoVertex_sumWeightsAll", "RecoTrack_ttvaVertex", "RecoTrack_truthVertex")  # fmt: skip
ARMS = ("pvf_unit", "pvf_score", "amvf_assoc", "amvf_fit_unit", "amvf_fitw",
        "amvf_fitw_unwpt2")  # fmt: skip
CLS_FIELDS = ("clean", "merged", "split", "fake", "n_reco", "n_zero_weight")
# Vertex track-multiplicity rules (same rule for both algorithms): a vertex
# counts if it has >= mt tracks (AMVF: assocTracks or fitTracks; PV-Finder:
# GNN-assigned tracks at t). 1 = drop-empty, 2 = the headline rule.
MIN_TRACKS = (1, 2)
EV_FIELDS = (
    "entry", "mu", "has_hs", "hs_z", "dens_itk", "dens_old", "dens_new_ge1",
    "n_truth_old", "n_truth_new", "n_truth_new_tn0", "pvf_any", "pvf_ge1", "pvf_ge2",
    "amvf_ge1", "amvf_ge2", "n_trk", "n_fit_pairs", "n_fit_not_assoc",
    "w_fit_not_assoc", "w_fit", "n_assoc_pairs", "n_assoc_not_fit",
    "n_fit_tracks", "n_fit_tracks_shared", "n_assoc_tracks_shared",
    "n_vtx_fit_empty", "n_vtx_assoc_empty",
)  # fmt: skip
W_BINS = np.linspace(0.0, 1.0, 1001)


def _np(x: ak.Array, dtype: type = np.float64) -> np.ndarray:
    return ak.to_numpy(x).astype(dtype)


def _lists(x: ak.Array, dtype: type = np.int64) -> list[np.ndarray]:
    """Jagged per-event branch -> list of arrays (fast: one flatten, one split)."""
    cnt = ak.to_numpy(ak.num(x))
    if len(cnt) == 0:
        return []
    flat = ak.to_numpy(ak.flatten(x)).astype(dtype)
    return np.split(flat, np.cumsum(cnt)[:-1])


def _fail(e: int, what: str) -> None:
    raise ValueError(f"IDENTITY FAILURE at entry {e}: {what}")


def check_identity(e: int, o: ak.Record, n: ak.Record, stats: Counter) -> None:
    """Exact old/new equality of the common reco branches (raises on mismatch)."""
    for b in SAME_SCALAR:
        if not float(o[b]) == float(n[b]):
            _fail(e, f"{b} old={o[b]} new={n[b]}")
    for b in SAME_JAGGED:
        if not np.array_equal(_np(o[b]), _np(n[b]), equal_nan=True):
            _fail(e, b)
    oa, na = _lists(o["RecoVertex_assocTracks"]), _lists(n["RecoVertex_assocTracks"])
    if len(oa) != len(na) or any(not np.array_equal(a, b) for a, b in zip(oa, na)):
        _fail(e, "RecoVertex_assocTracks (ordered lists)")
    for b in OLD_EMPTY:
        stats[f"ev_old_{b}_empty"] += int(len(o[b]) == 0)
        stats[f"ev_new_{b}_filled"] += int(len(n[b]) == len(n["RecoVertex_z"]))
    nvtx = _np(o["RecoVertex_nTracks"])
    stats["ev_reco_ntracks_eq_len_assoc"] += int(
        np.array_equal(nvtx, [len(a) for a in oa])
    )


def check_truth(
    e: int, o: ak.Record, n: ak.Record, stats: Counter
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Truth identity (nTracks >= 2 set, HS) and old-vs-new list relations.

    Returns old z, new z, old HS index, new HS index (-1 if none).
    """
    oz, nz = _np(o["TruthVertex_z"]), _np(n["TruthVertex_z"])
    otn, ntn = _np(o["TruthVertex_nTracks"], np.int64), _np(n["TruthVertex_nTracks"], np.int64)  # fmt: skip
    oa, na = _lists(o["TruthVertex_assocTracks"]), _lists(n["TruthVertex_assocTracks"])
    if int(n["NumTruthVtx"]) != len(nz) or int(o["NumTruthVtx"]) != len(oz):
        _fail(e, "NumTruthVtx differs from the TruthVertex_z length")
    o2, n2 = np.flatnonzero(otn >= 2), np.flatnonzero(ntn >= 2)
    if len(o2) != len(n2):
        _fail(e, f"number of truth vertices with nTracks >= 2: {len(o2)} vs {len(n2)}")
    stats["ev_truth_ge2_same_order"] += int(np.array_equal(oz[o2], nz[n2]))
    # Pair by (z, x, y): distinct vertices can share a float32 z exactly.
    ox, oy, nx, ny = (_np(r[f"TruthVertex_{c}"]) for r in (o, n) for c in "xy")
    so = o2[np.lexsort((oy[o2], ox[o2], oz[o2]))]
    sn = n2[np.lexsort((ny[n2], nx[n2], nz[n2]))]
    stats["truth_ge2_z_ties"] += len(o2) - len(np.unique(oz[o2]))
    if not (np.array_equal(oz[so], nz[sn]) and np.array_equal(ox[so], nx[sn]) and np.array_equal(oy[so], ny[sn])):  # fmt: skip
        _fail(e, "TruthVertex_z/x/y of the nTracks >= 2 vertices (as a set)")
    if not np.array_equal(otn[so], ntn[sn]):
        _fail(e, "TruthVertex_nTracks of the nTracks >= 2 vertices")
    for a, b in zip(so, sn):
        if not np.array_equal(np.sort(oa[a]), np.sort(na[b])):
            _fail(e, f"TruthVertex_assocTracks (as sets) of the vertex at z={oz[a]}")
        stats["truth_ge2_assoc_same_track_order"] += int(np.array_equal(oa[a], na[b]))
    stats["truth_ge2_vertices"] += len(o2)
    stats["ev_new_z_unique"] += int(len(np.unique(nz)) == len(nz))
    stats["truth_new_z_ties"] += len(nz) - len(np.unique(nz))
    oh = np.flatnonzero(_np(o["TruthVertex_type"], np.int64) == 1)
    nh = np.flatnonzero(_np(n["TruthVertex_type"], np.int64) == 1)
    if len(oh) > 1 or len(nh) > 1 or len(oh) > len(nh):
        _fail(e, f"HS count old {len(oh)} new {len(nh)}")
    h_o = int(oh[0]) if len(oh) else -1
    h_n = int(nh[0]) if len(nh) else -1
    if h_o < 0 and h_n >= 0:  # HS without selected tracks, kept only in the new file
        stats["ev_hs_only_in_new"] += 1
        stats["hs_only_in_new_ntracks"] += int(ntn[h_n])
    if h_o >= 0:
        if oz[h_o] != nz[h_n] or otn[h_o] != ntn[h_n]:
            _fail(e, "HS (TruthVertex_type == 1) z or nTracks")
        if not np.array_equal(np.sort(oa[h_o]), np.sort(na[h_n])):
            _fail(e, "HS TruthVertex_assocTracks")
    in_new = np.isin(oz, nz)
    stats["ev_old_subset_of_new"] += int(in_new.all())
    stats["old_not_in_new"] += int((~in_new).sum())
    new_only = ~np.isin(nz, oz)
    stats["new_only_vertices"] += int(new_only.sum())
    stats["new_only_with_tracks"] += int((ntn[new_only] > 0).sum())
    stats["ev_old_equals_new_ntrk_ge1"] += int(
        np.array_equal(np.sort(oz), np.sort(nz[ntn >= 1]))
    )
    stats["old_vertices_ntrk0"] += int((otn == 0).sum())
    return oz, nz, h_o, h_n


def density(z: np.ndarray, h: int) -> float:
    """Other truth vertices within +-2 mm of the HS, per mm (NaN without HS)."""
    if h < 0:
        return float("nan")
    hz = z[h]
    return float((np.abs(np.delete(z, h) - hz) < DENS_HALF).sum() / (2 * DENS_HALF))


def density_subset(z: np.ndarray, keep: np.ndarray, h: int) -> float:
    """``density`` over the vertices ``keep`` (NaN if the HS is not among them)."""
    if h < 0 or not keep[h]:
        return float("nan")
    return density(z[keep], int(np.cumsum(keep)[h] - 1))


def counts_of(c: hm.VertexClassification, w: list[np.ndarray] | None) -> list[int]:
    """CLS_FIELDS of one classification."""
    nz = 0 if w is None else sum(int(len(x) > 0 and x.sum() == 0) for x in w)
    return [*c.counts().tolist(), len(c.labels), nz]


def classify_event(
    e: int, lists: dict[str, tuple[list[np.ndarray], list[np.ndarray] | None]],
    pt: np.ndarray, truth: np.ndarray,
) -> np.ndarray:  # fmt: skip
    """(ARMS, CLS_FIELDS) counts; unit-weight runs must equal the unweighted rule."""
    out = np.zeros((len(ARMS), len(CLS_FIELDS)), np.int64)
    pv, pw = lists["pvf"]
    av, _ = lists["amvf"]
    fv, fw = lists["fit"]
    unit = lambda ls: [np.ones(len(x)) for x in ls]  # noqa: E731
    for arm, (ls, w) in (("pvf_unit", (pv, None)), ("amvf_assoc", (av, None))):
        ref = hm.classify_vertices(ls, pt, truth)
        got = classify_vertices_weighted(ls, unit(ls), pt, truth)
        if not np.array_equal(ref.labels, got.labels):
            raise AssertionError(f"entry {e}: weighted rule with unit weights != unweighted ({arm})")  # fmt: skip
        out[ARMS.index(arm)] = counts_of(ref, None)
    out[ARMS.index("pvf_score")] = counts_of(classify_vertices_weighted(pv, pw, pt, truth), pw)  # fmt: skip
    out[ARMS.index("amvf_fit_unit")] = counts_of(hm.classify_vertices(fv, pt, truth), None)  # fmt: skip
    out[ARMS.index("amvf_fitw")] = counts_of(classify_vertices_weighted(fv, fw, pt, truth), fw)  # fmt: skip
    out[ARMS.index("amvf_fitw_unwpt2")] = counts_of(
        classify_vertices_weighted(fv, fw, pt, truth, weighted_sumpt2=False), fw
    )
    return out


def process_chunk(path: str) -> dict:
    """Identity checks, densities, HS flags and classifications for one chunk."""
    with np.load(path) as f:
        z = {k: f[k] for k in ("entries", "mu", "n_tracks", "track_offsets", "peak_offsets",
                               "peak_z", f"targets_{GNN_KEY}", f"scores_{GNN_KEY}")}  # fmt: skip
    entries = z["entries"]
    lo, hi = int(entries[0]), int(entries[-1]) + 1
    if not np.array_equal(entries, np.arange(lo, hi)):
        raise ValueError(f"{path}: entries are not contiguous")
    old = uproot.open(OLD_ROOT)["PVFinderData"].arrays(list(COMMON), entry_start=lo, entry_stop=hi)  # fmt: skip
    new = uproot.open(NEW_ROOT)["PVFinderData"].arrays(
        [*COMMON, *NEW_ONLY], entry_start=lo, entry_stop=hi
    )
    toff, poff = z["track_offsets"], z["peak_offsets"]
    stats: Counter = Counter()
    whist = np.zeros(len(W_BINS) - 1, np.int64)
    w_min, w_max = np.inf, -np.inf
    ev = np.zeros((len(entries), len(EV_FIELDS)))
    cls = np.zeros(
        (len(entries), len(MIN_TRACKS), len(ARMS), len(CLS_FIELDS)), np.int64
    )
    for i, e in enumerate(entries.tolist()):
        o, n = old[e - lo], new[e - lo]
        check_identity(e, o, n, stats)
        oz, nz, h_o, h_n = check_truth(e, o, n, stats)
        if float(o["ActualNumOfInt"]) != z["mu"][i]:
            raise ValueError(f"entry {e}: mu differs from the inference record")
        pt = _np(o["RecoTrack_pT"])
        nt = len(pt)
        if nt != z["n_tracks"][i]:
            raise ValueError(
                f"entry {e}: track count differs from the inference record"
            )
        truth = hm.track_truth_labels(
            _np(ak.flatten(o["TruthVertex_assocTracks"]), np.int64),
            _np(o["TruthVertex_nTracks"], np.int64), nt,
        )  # fmt: skip
        ntn = _np(n["TruthVertex_nTracks"], np.int64)
        truth_new = hm.track_truth_labels(
            _np(ak.flatten(n["TruthVertex_assocTracks"]), np.int64), ntn, nt
        )  # fmt: skip
        stats["ev_truthVertex_eq_assoc_inverse"] += int(
            np.array_equal(truth_new, _np(n["RecoTrack_truthVertex"], np.int64))
        )
        # ---- AMVF TTVA links vs RecoTrack_ttvaVertex
        alists_raw = _lists(o["RecoVertex_assocTracks"])
        ttva = _np(n["RecoTrack_ttvaVertex"], np.int64)
        inv = np.full(nt, -1, np.int64)
        mult = np.zeros(nt, np.int64)
        for v, x in enumerate(alists_raw):
            inv[x] = v
            mult[x] += 1
        stats["ev_ttva_eq_assoc_inverse"] += int(np.array_equal(inv, ttva))
        stats["tracks_ttva_unassigned"] += int((ttva < 0).sum())
        stats["tracks"] += nt
        # ---- AMVF fit tracks and weights
        flists = _lists(n["RecoVertex_fitTracks"])
        fweights = _lists(n["RecoVertex_fitWeights"], np.float64)
        if len(flists) != len(alists_raw):
            raise ValueError(f"entry {e}: fitTracks and assocTracks have different vertex counts")  # fmt: skip
        stats["ev_nTracksAll_eq_len_fit"] += int(np.array_equal(_np(n["RecoVertex_nTracksAll"], np.int64), [len(x) for x in flists]))  # fmt: skip
        sw = np.array([w.sum() for w in fweights])
        dsw = np.abs(sw - _np(n["RecoVertex_sumWeightsAll"])).max(initial=0.0)
        stats["ev_sumWeightsAll_close"] += int(dsw < 1e-3)
        stats["max_sumWeightsAll_absdiff_e6"] = max(stats["max_sumWeightsAll_absdiff_e6"], int(dsw * 1e6))  # fmt: skip
        fit_cat = np.concatenate(flists + [np.zeros(0, np.int64)])
        w_cat = np.concatenate(fweights + [np.zeros(0)])
        if len(fit_cat) and (fit_cat.min() < 0 or fit_cat.max() >= nt):
            raise ValueError(f"entry {e}: fit track index out of range")
        stats["fit_vertices_with_duplicate_tracks"] += sum(len(np.unique(x)) != len(x) for x in flists)  # fmt: skip
        stats["fit_weights_zero"] += int((w_cat == 0).sum())
        stats["fit_weights_gt1"] += int((w_cat > 1).sum())
        whist += np.histogram(np.clip(w_cat, 0, 1), W_BINS)[0]
        if len(w_cat):
            w_min, w_max = min(w_min, w_cat.min()), max(w_max, w_cat.max())
        not_assoc = [~np.isin(f, a) for f, a in zip(flists, alists_raw)]
        n_not_fit = sum(int((~np.isin(a, f)).sum()) for f, a in zip(flists, alists_raw))
        fmult = np.bincount(fit_cat, minlength=nt)
        # ---- PV-Finder + GNN at the frozen t, drop-empty
        peaks = z["peak_z"][poff[i] : poff[i + 1]].astype(np.float64)
        tg = z[f"targets_{GNN_KEY}"][toff[i] : toff[i + 1]]
        sc32 = z[f"scores_{GNN_KEY}"][toff[i] : toff[i + 1]]
        # cut on the stored float32 scores, exactly as chain_t_metrics does
        # (a float64 cast flips scores that equal float32(t), e.g. t = 0.94)
        all_p = hm.lists_from_targets(hm.assign_from_scores(tg, sc32, T), len(peaks))
        sc = sc32.astype(np.float64)
        alists = [np.unique(x) for x in alists_raw]
        for m, mt in enumerate(MIN_TRACKS):
            pv = [x for x in all_p if len(x) >= mt]
            fkeep = [k for k, x in enumerate(flists) if len(x) >= mt]
            lists = {
                "pvf": (pv, [sc[x] for x in pv]),
                "amvf": ([x for x in alists if len(x) >= mt], None),
                "fit": ([flists[k] for k in fkeep], [fweights[k] for k in fkeep]),
            }
            cls[i, m] = classify_event(e, lists, pt, truth)
        # ---- HS reconstruction flags and densities
        hz = oz[h_o] if h_o >= 0 else np.nan
        npk = np.array([len(x) for x in all_p], np.int64)
        near = np.abs(peaks - hz) < HS_WIN
        az = _np(o["RecoVertex_z"])
        rn = np.array([len(x) for x in alists_raw], np.int64)
        anear = np.abs(az - hz) < HS_WIN
        ev[i] = [
            e, z["mu"][i], int(h_o >= 0), hz, density(nz, h_n), density(oz, h_o),
            density_subset(nz, ntn >= 1, h_n),
            len(oz), len(nz), int((ntn == 0).sum()), int(near.any()),
            int((near & (npk >= 1)).any()), int((near & (npk >= 2)).any()),
            int((anear & (rn >= 1)).any()), int((anear & (rn >= 2)).any()), nt,
            len(fit_cat), sum(int(x.sum()) for x in not_assoc),
            sum(float(w[x].sum()) for w, x in zip(fweights, not_assoc)), float(w_cat.sum()),
            int(rn.sum()), n_not_fit, int((fmult > 0).sum()), int((fmult > 1).sum()),
            int((mult > 1).sum()), sum(len(x) == 0 for x in flists), int((rn == 0).sum()),
        ]  # fmt: skip
    return {"ev": ev, "cls": cls, "stats": stats, "whist": whist,
            "w_min": w_min, "w_max": w_max}  # fmt: skip


def verify_frozen() -> dict:
    """Inference/metrics are at the frozen regime; checkpoints re-hashed from model_weights/."""
    REGIME.require_frozen()
    inf = json.loads((INFER / "metadata.json").read_text())
    meta = json.loads(METRICS.with_suffix(".json").read_text())
    dens, gnn = inf["density"], inf["gnn"][GNN_KEY]
    d_sha, g_sha = sha256(REPO / REGIME.density.checkpoint), sha256(REPO / REGIME.gnn.checkpoint)  # fmt: skip
    checks = {
        "density_checkpoint": dens["checkpoint"] == REGIME.density.checkpoint,
        "density_peaks": dens["peaks"] == asdict(REGIME.peaks),
        "density_sha256": d_sha == dens["sha256"],
        "gnn_checkpoint": gnn["path"] == REGIME.gnn.checkpoint,
        "gnn_sha256": g_sha == gnn["sha256"],
        "metrics_gnn_key": meta["gnn"] == GNN_KEY,
        "t_in_metrics_grid": T in meta["t_grid"],
        "windows": meta["window_mm"] == REGIME.match_window_mm and meta["hs_window_mm"] == HS_WIN,
        "entries": inf["entries_spec"] == "0:25000"
        and Path(inf["root"]).resolve() == OLD_ROOT.resolve(),
        "metrics_events": meta["events_sha256"] == inf["entries_sha256"],
        "inference_done": (INFER / "DONE").exists(),
    }  # fmt: skip
    if not all(checks.values()):
        raise RuntimeError(f"inputs are not at the frozen working point: {checks}")
    return {
        "git": git_state(), "regime": REGIME.to_dict(), "frozen_checks": checks,
        "density_checkpoint": REGIME.density.checkpoint, "density_sha256": d_sha,
        "gnn_checkpoint": REGIME.gnn.checkpoint, "gnn_sha256": g_sha,
        "inference": str(INFER.relative_to(REPO)), "inference_git": inf["git"],
        "metrics": str(METRICS.relative_to(REPO)), "metrics_git": meta["git"],
        "entries": "0:25000", "t": T, "hs_window_mm": HS_WIN,
    }  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-chunks", type=int, default=0, help="smoke test only")
    a = ap.parse_args()
    t0 = time.monotonic()
    prov = verify_frozen()
    paths = sorted(str(p) for p in INFER.glob("chunk_*.npz"))
    if a.max_chunks:
        paths = paths[: a.max_chunks]
    with ProcessPoolExecutor(min(a.workers, 12)) as pool:
        h_old, h_new = pool.submit(sha256, OLD_ROOT), pool.submit(sha256, NEW_ROOT)
        futs = [pool.submit(process_chunk, p) for p in paths]
        parts = []
        for k, f in enumerate(futs):  # in entry order: the first failure is reported
            parts.append(f.result())
            print(f"{k + 1}/{len(paths)} chunks, {time.monotonic() - t0:.0f} s", flush=True)  # fmt: skip
        prov["old_root"], prov["new_root"] = str(OLD_ROOT), str(NEW_ROOT)
        prov["old_root_sha256"], prov["new_root_sha256"] = (
            h_old.result(),
            h_new.result(),
        )
    ev = np.concatenate([p["ev"] for p in parts])
    cls = np.concatenate([p["cls"] for p in parts])
    stats: Counter = Counter()
    for p in parts:
        mx = max(stats["max_sumWeightsAll_absdiff_e6"], p["stats"]["max_sumWeightsAll_absdiff_e6"])  # fmt: skip
        stats.update(p["stats"])
        stats["max_sumWeightsAll_absdiff_e6"] = mx
    gates = cross_check_metrics(ev, cls)
    whist = sum(p["whist"] for p in parts)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "per_event.npz", ev=ev, cls=cls, whist=whist, w_bins=W_BINS)  # fmt: skip
    meta = {"ev_fields": EV_FIELDS, "arms": ARMS, "cls_fields": CLS_FIELDS, "min_tracks": MIN_TRACKS,
            "provenance": prov, "n_events": len(ev)}  # fmt: skip
    (a.out / "per_event.json").write_text(json.dumps(meta, indent=2) + "\n")
    cdf = np.cumsum(whist) / whist.sum()
    identity = {
        "provenance": prov,
        "result": "PASS: entries 0-24,999 identical in every compared branch",
        "n_events": len(ev), "compared_exactly": [*SAME_SCALAR, *SAME_JAGGED, "RecoVertex_assocTracks"],
        "truth_compared": "nTracks >= 2 vertices as a set paired by (z, x, y): z, x, y, nTracks, "
                          "assocTracks as sorted sets; HS z, nTracks, assocTracks",
        "counters": dict(stats), "gates_vs_chain_t_metrics": gates,
        "mean_truth_per_event": {"old": float(ev[:, EV_FIELDS.index("n_truth_old")].mean()),
                                 "new": float(ev[:, EV_FIELDS.index("n_truth_new")].mean()),
                                 "new_with_0_tracks": float(ev[:, EV_FIELDS.index("n_truth_new_tn0")].mean())},
        "fit_weights": {"min": float(min(p["w_min"] for p in parts)),
                        "max": float(max(p["w_max"] for p in parts)),
                        "median": float(W_BINS[1:][np.searchsorted(cdf, 0.5)]),
                        "frac_below_0p5": float(cdf[499])},
        "runtime_s": time.monotonic() - t0,
    }  # fmt: skip
    (a.out / "identity_check.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({k: identity[k] for k in ("counters", "gates_vs_chain_t_metrics")}, indent=1))  # fmt: skip
    print("done", flush=True)


def cross_check_metrics(ev: np.ndarray, cls: np.ndarray) -> dict:
    """Assert the recomputed quantities equal the chain_t_metrics per-event record."""
    with np.load(METRICS) as f:
        counts, events = f["counts"], f["events"]
    meta = json.loads(METRICS.with_suffix(".json").read_text())
    n = len(ev)
    counts, events = counts[:n], events[:n]
    ef, cf = meta["event_fields"], meta["count_fields"]
    a_t = meta["t_grid"].index(T)
    g = {
        "entries": np.array_equal(ev[:, 0], events[:, ef.index("entry")]),
        "has_hs": np.array_equal(ev[:, 2], events[:, ef.index("has_hs")]),
        "dens_old_eq_metrics_local_density": np.array_equal(
            np.nan_to_num(ev[:, EV_FIELDS.index("dens_old")]), events[:, ef.index("local_density")]),
        "dens_new_ge1_eq_dens_old": np.array_equal(
            ev[:, EV_FIELDS.index("dens_new_ge1")], ev[:, EV_FIELDS.index("dens_old")], equal_nan=True),
        "pvf_any_eq_pvf_hs_within": np.array_equal(ev[:, EV_FIELDS.index("pvf_any")], events[:, ef.index("pvf_hs_within")]),
        "amvf_ge2_eq_amvf_hs_within": np.array_equal(ev[:, EV_FIELDS.index("amvf_ge2")], events[:, ef.index("amvf_hs_within")]),
    }  # fmt: skip
    if tuple(meta["min_tracks"]) != MIN_TRACKS:
        raise AssertionError("chain_t_metrics min-track rules differ")
    for m, mt in enumerate(MIN_TRACKS):
        for arm, a in (("pvf_unit", a_t), ("amvf_assoc", -1)):
            for k in ("clean", "merged", "split", "fake", "n_reco"):
                ref = counts[:, a, m, cf.index(k)]
                got = cls[:, m, ARMS.index(arm), CLS_FIELDS.index(k)]
                g[f"min{mt}_{arm}_{k}_eq_metrics"] = np.array_equal(got, ref)
    g = {k: bool(v) for k, v in g.items()}
    if not all(g.values()):
        raise AssertionError(f"recomputation differs from chain_t_metrics: {g}")
    return g


if __name__ == "__main__":
    main()
