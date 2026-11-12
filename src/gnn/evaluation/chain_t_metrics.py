"""Per-event full-chain metrics versus the GNN score cut t, and AMVF on the same events.

Replays ``gnn.evaluation.chain_inference`` outputs (peaks and top-1 GNN edges)
together with the ROOT truth and AMVF branches. No inference. For each event,
each t in ``T_GRID`` (PV-Finder + GNN) and AMVF, it stores:

* **track-purity categories** (``hs_metrics.classify_vertices``, identical to
  ``classification.classify_assignments``): Clean / Merged / Split / Fake and
  the number of vertices, for vertices with >= 1 and >= 2 assigned tracks;
* **positional categories** of the same vertex list (greedy closest-first
  ``match_vertices`` at the regime window, 0.17 mm): clean, merged, split,
  fake and truth vertices found;
* **track counts**: correct (track truth = dominant truth of its vertex),
  assigned truth-labelled, assigned;
* **HS composition** (``hs_metrics.hs_event_record`` subset);
* **HS selection, Athena 24.0 rule** on populated vertices
  (``hs_selection_identical_rule.lazy_rule_selection``; PV-Finder vertex
  positions from the 3D adaptive fit), plus the AMVF stored primary;
* **HS longitudinal residual**: PV-Finder = 3D adaptive beam-spot fit of the
  tracks the GNN assigns at t to the peak closest to the HS (within 0.1 mm);
  AMVF = its closest nTrk >= 2 vertex within 0.1 mm.

Run (repo root, inside tmux):
    PYTHONPATH=src python -u -m gnn.evaluation.chain_t_metrics \\
        --infer outputs/.../infer/test_v6 --gnn v2e527 --out outputs/.../metrics/test_v6_v2e527.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from gnn.diagnostics.hs_selection_identical_rule import Tracks, lazy_rule_selection
from gnn.evaluation import hs_metrics as hm
from gnn.evaluation import pv_selection as ps
from gnn.evaluation.chain_inference import git_state, sha256
from pv_finder.diagnostics.amvf_fair_comparison.matching import match_vertices
from pv_finder.utils.working_points import HLLHC

T_GRID = (
    0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97,
    0.98, 0.985, 0.99, 0.995,
)  # fmt: skip
MIN_TRACKS = (1, 2)
WINDOW = HLLHC.match_window_mm  # 0.17 mm
HS_WINDOW = HLLHC.hs_match_window_mm  # 0.1 mm
COUNT_FIELDS = (
    "clean", "merged", "split", "fake", "n_reco",
    "p_clean", "p_merged", "p_split", "p_fake", "p_truth_found",
    "n_correct", "n_assigned_truth", "n_assigned",
    "hs_has_match", "hs_match_label", "hs_clean", "hs_match_hs_tracks", "hs_match_tracks",
)  # fmt: skip
SEL_FIELDS = ("has_match", "sel_is_match", "sel_near", "reco_sel")
EVENT_FIELDS = (
    "entry", "mu", "n_truth_ge2", "n_truth_all", "n_tracks", "n_labelled",
    "n_hs_tracks", "hs_truth_ntrk", "local_density", "n_peaks", "pvf_hs_within",
    "amvf_hs_within", "has_hs",
)  # fmt: skip
BRANCHES = [
    "ActualNumOfInt", "BeamPosX", "BeamPosY", "BeamPosSigmaX",
    "RecoTrack_pT", "RecoTrack_d0", "RecoTrack_z0", "RecoTrack_phi", "RecoTrack_theta",
    "RecoTrack_ErrD0", "RecoTrack_ErrZ0", "RecoTrack_ErrD0Z0",
    "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type",
    "RecoVertex_x", "RecoVertex_y", "RecoVertex_z", "RecoVertex_type",
    "RecoVertex_assocTracks",
]  # fmt: skip


def list_counts(
    lists: list[np.ndarray], vz: np.ndarray, pt: np.ndarray, truth: np.ndarray,
    truth_z2: np.ndarray, hs: int, hz: float, order: np.ndarray | None,
) -> np.ndarray:  # fmt: skip
    """COUNT_FIELDS for one vertex list (already filtered by track multiplicity)."""
    cls = hm.classify_vertices(lists, pt, truth)
    c = cls.counts()
    m = match_vertices(truth_z2, np.asarray(vz, dtype=np.float64), WINDOW)
    n_correct = n_at = n_a = 0
    for v, x in enumerate(lists):
        tt = truth[x]
        n_a += len(x)
        n_at += int((tt >= 0).sum())
        if cls.dominant[v] >= 0:
            n_correct += int((tt == cls.dominant[v]).sum())
    rec = hm.hs_event_record(lists, cls, truth, hs, vz, hz, HS_WINDOW, order)
    f = dict(zip(hm.HS_FIELDS, rec))
    return np.array(
        [*c, len(lists), m.reco_clean, m.reco_merged, m.reco_split, m.reco_fake,
         m.n_truth_matched, n_correct, n_at, n_a, f["has_match"], f["match_label"],
         f["clean_hs"], f["match_hs_tracks"], f["match_tracks"]],
        dtype=np.int64,
    )  # fmt: skip


def hs_selection(
    trk: Tracks, lists: list[np.ndarray], vz: np.ndarray, truth: np.ndarray,
    hs: int, hz: float, beam: tuple[float, float], sig: float,
) -> list[int]:  # fmt: skip
    """SEL_FIELDS under the Athena rule, positions from the 3D adaptive fit."""
    if not lists:
        return [0, 0, 0, 0]
    cls = hm.classify_vertices(lists, trk.pt, truth)
    nhs = hm.hs_track_counts(lists, cls, truth, hs)
    plain = np.array([ps.sumpt2_score(trk.pt[x] / 1000.0) for x in lists])
    fits: dict[int, tuple[float, float, float]] = {}

    def fpos(v: int) -> tuple[float, float, float]:
        if v not in fits:
            fits[v] = trk.fit(lists[v], vz[v], beam, sig)
        return fits[v]

    sel, match = lazy_rule_selection(trk, lists, plain, fpos, nhs)
    near = sel >= 0 and abs(fpos(sel)[2] - hz) < HS_WINDOW
    ok = match >= 0 and sel == match
    return [int(match >= 0), int(ok), int(near), int(ok and near)]


def process_chunk(path: str, root: str, gnn: str, do_hs: bool) -> dict:
    """All metrics for one inference chunk."""
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    entries = z["entries"]
    toff, poff = z["track_offsets"], z["peak_offsets"]
    tg_all, sc_all = z[f"targets_{gnn}"], z[f"scores_{gnn}"]
    tree = uproot.open(root)["PVFinderData"]
    lo_e, hi_e = int(entries[0]), int(entries[-1]) + 1
    data = tree.arrays(BRANCHES, entry_start=lo_e, entry_stop=hi_e, library="ak")
    n, na = len(entries), len(T_GRID) + 1
    counts = np.zeros((n, na, len(MIN_TRACKS), len(COUNT_FIELDS)), np.int64)
    sel = np.zeros((n, na + 1, len(SEL_FIELDS)), np.int64)
    res = np.full((n, na), np.nan)
    fallback = np.zeros((n, len(T_GRID)), np.int64)
    ev_out = np.zeros((n, len(EVENT_FIELDS)))
    for i, e in enumerate(entries):
        ev = data[int(e) - lo_e]
        if float(ev["ActualNumOfInt"]) != z["mu"][i]:
            raise ValueError(f"entry {e}: mu differs from the inference record")
        trk = Tracks(ev)
        nt = len(trk.pt)
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
        # No stored HS (no selected HS track; 2 of 100,000 r16443 entries):
        # label -2 matches no track and NaN z matches no vertex; the event is
        # flagged has_hs = 0 and excluded from every HS denominator.
        has_hs = len(hs_c) == 1
        h = int(hs_c[0]) if has_hs else -2
        hz = float(tz[h]) if has_hs else float("nan")
        tz2 = tz[tn >= 2]
        beam = (float(ev["BeamPosX"]), float(ev["BeamPosY"]))
        sig = float(ev["BeamPosSigmaX"])
        peaks = z["peak_z"][poff[i] : poff[i + 1]].astype(np.float64)
        tg, sc = tg_all[toff[i] : toff[i + 1]], sc_all[toff[i] : toff[i + 1]]
        if nt and len(peaks) and tg.max() >= len(peaks):
            raise ValueError(f"entry {e}: edge target beyond the peak list")
        k_hs = int(np.argmin(np.abs(peaks - hz))) if len(peaks) else -1
        if k_hs >= 0 and not abs(peaks[k_hs] - hz) < HS_WINDOW:
            k_hs = -1
        for a, t in enumerate(T_GRID):
            all_lists = hm.lists_from_targets(
                hm.assign_from_scores(tg, sc, t), len(peaks)
            )
            for m, mt in enumerate(MIN_TRACKS):
                keep = np.flatnonzero([len(x) >= mt for x in all_lists])
                lists = [all_lists[j] for j in keep]
                counts[i, a, m] = list_counts(
                    lists, peaks[keep], trk.pt, truth, tz2, h, hz, None
                )
                if m == 0 and do_hs:
                    sel[i, a] = hs_selection(
                        trk, lists, peaks[keep], truth, h, hz, beam, sig
                    )
            if do_hs and k_hs >= 0:
                idx = all_lists[k_hs]
                if len(idx):
                    res[i, a] = trk.fit(idx, peaks[k_hs], beam, sig)[2] - hz
                else:
                    res[i, a] = peaks[k_hs] - hz
                    fallback[i, a] = 1
        # ---- AMVF: stored lists and positions, Athena container order
        alists = [np.unique(ak.to_numpy(x).astype(np.int64)) for x in ev["RecoVertex_assocTracks"]]  # fmt: skip
        if any(len(x) and (x.min() < 0 or x.max() >= nt) for x in alists):
            raise ValueError(f"entry {e}: AMVF track index out of range")
        axyz = np.stack(
            [ak.to_numpy(ev[c]).astype(np.float64) for c in ("RecoVertex_x", "RecoVertex_y", "RecoVertex_z")], 1
        )  # fmt: skip
        native = np.flatnonzero(ak.to_numpy(ev["RecoVertex_type"]) == 1)
        if len(native) != 1:
            raise ValueError(f"entry {e}: {len(native)} stored AMVF primaries")
        for m, mt in enumerate(MIN_TRACKS):
            keep = np.flatnonzero([len(x) >= mt for x in alists])
            lists = [alists[j] for j in keep]
            counts[i, -1, m] = list_counts(
                lists, axyz[keep, 2], trk.pt, truth, tz2, h, hz, np.arange(len(lists))
            )
        if do_hs:
            keep = np.flatnonzero([len(x) > 0 for x in alists])
            lists = [alists[j] for j in keep]
            pos = axyz[keep]
            acls = hm.classify_vertices(lists, trk.pt, truth)
            anhs = hm.hs_track_counts(lists, acls, truth, h)
            nat = int(np.flatnonzero(keep == native[0])[0])
            smatch = ps.best_hs_match(anhs, np.arange(len(lists)))
            near = abs(pos[nat, 2] - hz) < HS_WINDOW
            ok = smatch >= 0 and nat == smatch
            sel[i, na - 1] = [int(smatch >= 0), int(ok), int(near), int(ok and near)]
            aplain = np.array([ps.sumpt2_score(trk.pt[x] / 1000.0) for x in lists])
            s_r, m_r = lazy_rule_selection(
                trk, lists, aplain, lambda v: tuple(pos[v]), anhs
            )
            near = s_r >= 0 and abs(pos[s_r, 2] - hz) < HS_WINDOW
            ok = m_r >= 0 and s_r == m_r
            sel[i, na] = [int(m_r >= 0), int(ok), int(near), int(ok and near)]
        rn = np.array([len(x) for x in alists])
        a2 = axyz[rn >= 2, 2]
        ka = int(np.argmin(np.abs(a2 - hz))) if len(a2) else -1
        if ka >= 0 and abs(a2[ka] - hz) < HS_WINDOW:
            res[i, -1] = a2[ka] - hz
        others = np.abs(np.delete(tz, h) - hz) < 2.0 if has_hs else np.zeros(0)
        ev_out[i] = [
            e, z["mu"][i], len(tz2), len(tz), nt, int((truth >= 0).sum()),
            int((truth == h).sum()), tn[h] if has_hs else 0, others.sum() / 4.0,
            len(peaks), int(k_hs >= 0), int(ka >= 0 and abs(a2[ka] - hz) < HS_WINDOW),
            int(has_hs),
        ]  # fmt: skip
    return {"path": path, "counts": counts, "sel": sel, "res": res,
            "fallback": fallback, "events": ev_out}  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--infer", required=True, type=Path)
    ap.add_argument("--gnn", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no-hs", action="store_true", help="skip HS selection/fits")
    ap.add_argument("--max-chunks", type=int, default=0)
    a = ap.parse_args()
    meta_in = json.loads((a.infer / "metadata.json").read_text())
    if not (a.infer / "DONE").exists() and not a.max_chunks:
        raise RuntimeError(f"{a.infer} inference not finished")
    paths = sorted(str(p) for p in a.infer.glob("chunk_*.npz"))
    if a.max_chunks:
        paths = paths[: a.max_chunks]
    t0 = time.monotonic()
    with ProcessPoolExecutor(a.workers) as pool:
        futs = [pool.submit(process_chunk, p, meta_in["root"], a.gnn, not a.no_hs) for p in paths]  # fmt: skip
        parts = []
        for k, f in enumerate(futs):
            parts.append(f.result())
            print(f"{k + 1}/{len(paths)} chunks, {time.monotonic() - t0:.0f} s", flush=True)  # fmt: skip
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("counts", "sel", "res", "fallback", "events")}  # fmt: skip
    if not a.max_chunks and len(cat["events"]) != meta_in["n_entries"]:
        raise RuntimeError("event count differs from the inference metadata")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **cat)
    meta = {
        "infer": str(a.infer), "infer_metadata": meta_in, "gnn": a.gnn,
        "t_grid": T_GRID, "min_tracks": MIN_TRACKS, "count_fields": COUNT_FIELDS,
        "sel_fields": SEL_FIELDS, "sel_arms": [*(f"t{t}" for t in T_GRID), "amvf_stored", "amvf_rule"],
        "event_fields": EVENT_FIELDS, "window_mm": WINDOW, "hs_window_mm": HS_WINDOW,
        "max_vertex_chi2": ps.MAX_VERTEX_CHI2, "hs_computed": not a.no_hs,
        "git": git_state(), "script_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(Path(hm.__file__)),
        "events_sha256": hashlib.sha256(cat["events"][:, 0].astype(np.int64).tobytes()).hexdigest(),
        "runtime_s": time.monotonic() - t0,
    }  # fmt: skip
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
