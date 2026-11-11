"""Per-event HS and vertex-composition study for PV-Finder(+GNN) vs AMVF.

Replays saved inputs only (no network inference): the PV-Finder histograms of
the 25,000 r16443 validation entries, the saved top-1 GNN edges (best target
and score per track) for two checkpoints, and the ROOT truth/AMVF branches.
Every association arm is judged by ``gnn.evaluation.hs_metrics``.

Arms (vertex lists), each evaluated with reco >= 1 and >= 2 tracks:
  gnn_v4_t098, gnn_v4_t095, gnn_v4_t000   v4 noaug e107 (note checkpoint)
  gnn_fullmu_t098, gnn_fullmu_t000        fullmu v1 e213 (deck checkpoint)
  z0_nearest, z0_3sigma                   baselines on the same peaks
  truth_0p5, truth_0p1694                 truth-guided reference (MC only)
  amvf                                    stored AMVF track lists

Usage (from the repo root, inside tmux):
    PYTHONPATH=src python -u -m gnn.diagnostics.hs_chain_study \\
        --out outputs/11_09_2026_output/hs_chain_study --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from gnn.evaluation import hs_metrics as hm
from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / (
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"
)
HIST_DIR = REPO / "outputs/09_23_2026_output/hs_fullrange_050"
PKL = REPO / "outputs/08_05_2026_output/eval_v6_operating_point/r16443/eval_results.pkl"
EDGES = {
    "v4": REPO
    / (
        "presentations/mattia/09_25_2026/opening_slides/review/"
        "archive_v4_noaug_epoch107_20260925/questions_audit"
    ),
    "fullmu": REPO
    / "outputs/09_25_2026_output/gnn_fullmu_deck_eval/review/questions_audit",
}
CHECKPOINTS = {
    "v4": "model_weights/ttva_gnn_hllhc_v4_noaug/ttva_gat_alleta_k20_v4_noaug180k_epoch_107.pyt",
    "fullmu": "model_weights/ttva_gnn_hllhc_fullmu_v1_20260924/"
    "ttva_gat_alleta_k20_fullmu_v1_60k_epoch_213.pyt",
}
# Production peak settings of the note (Sec. 4.1): threshold, integral,
# min width, min height, centroid half-width.
PEAK_ARGS = (0.01, 0.30, 3, 0.03, 3)
ARMS = (
    "gnn_v4_t098", "gnn_v4_t095", "gnn_v4_t000", "gnn_fullmu_t098",
    "gnn_fullmu_t000", "z0_nearest", "z0_3sigma", "truth_0p5",
    "truth_0p1694", "amvf",
)  # fmt: skip
MIN_TRACKS = (1, 2)
HS_WINDOW = 0.1  # mm, ITk paper Sec. 5.5
WINDOW = 0.1694  # mm, note-wide vertex matching window (PI decision)
EVENT_FIELDS = (
    "entry", "mu_x100", "n_truth_ge2", "n_tracks", "n_peaks", "n_amvf",
    "hs_truth_ntracks", "density_x1000", "peaks_hs_within", "amvf_hs_within",
    "peaks_1to1_0p5", "peaks_1to1_0p1694", "amvf_1to1_0p5", "amvf_1to1_0p1694",
    "native_is_match", "native_dominant_hs", "native_within", "native_is_leader",
)  # fmt: skip
BRANCHES = [
    "ActualNumOfInt", "RecoTrack_z0", "RecoTrack_ErrZ0", "RecoTrack_pT",
    "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_assocTracks",
    "TruthVertex_type", "RecoVertex_z", "RecoVertex_type", "RecoVertex_assocTracks",
]  # fmt: skip


def _load_edges(
    directory: Path, start: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(directory / f"best_edges_{start:05d}.npz") as f:
        return f["offsets"], f["scores"], f["targets"]


def _arm_targets(
    arm: str,
    edges: dict[str, tuple[np.ndarray, np.ndarray]],
    z0: np.ndarray,
    ez0: np.ndarray,
    peaks: np.ndarray,
    truth: np.ndarray,
    truth_z: np.ndarray,
) -> np.ndarray:
    """Per-track target peak (or -1) for the PV-Finder-based arms."""
    if arm.startswith("gnn_"):
        _, ckpt, cut = arm.split("_")
        targets, scores = edges[ckpt]
        return hm.assign_from_scores(targets, scores, int(cut[1:]) / 100)
    if arm == "z0_nearest":
        return hm.assign_nearest_z0(z0, peaks)
    if arm == "z0_3sigma":
        return hm.assign_nearest_z0(z0, peaks, ez0, 3.0)
    window = {"truth_0p5": 0.5, "truth_0p1694": WINDOW}[arm]
    return hm.assign_truth_guided(truth, truth_z, peaks, window)


def process_chunk(start: int, stop: int, peaks_ref: dict[int, np.ndarray]) -> dict:
    """Evaluate entries [start, stop); returns per-event arrays."""
    tree = uproot.open(ROOT)["PVFinderData"]
    data = tree.arrays(BRANCHES, entry_start=start, entry_stop=stop, library="ak")
    with np.load(HIST_DIR / f"chunk_{start:05d}_{start + 500:05d}.npz") as f:
        hists = f["histograms"]
        np.testing.assert_array_equal(
            f["event_idx"][: stop - start], np.arange(start, stop)
        )
    edge_files = {k: _load_edges(d, start) for k, d in EDGES.items()}
    n = stop - start
    counts = np.zeros((n, len(ARMS), len(MIN_TRACKS), 6), dtype=np.int64)
    hs = np.zeros((n, len(ARMS), len(MIN_TRACKS), len(hm.HS_FIELDS)), dtype=np.int64)
    ev_out = np.zeros((n, len(EVENT_FIELDS)), dtype=np.int64)
    for loc in range(n):
        entry = start + loc
        ev = data[loc]
        z0 = ak.to_numpy(ev["RecoTrack_z0"]).astype(np.float64)
        ez0 = ak.to_numpy(ev["RecoTrack_ErrZ0"]).astype(np.float64)
        pt = ak.to_numpy(ev["RecoTrack_pT"]).astype(np.float64)
        n_trk = len(z0)
        tz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        assoc = ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"]))
        truth = hm.track_truth_labels(assoc, tn, n_trk)
        hs_cand = np.flatnonzero(ak.to_numpy(ev["TruthVertex_type"]) == 1)
        if len(hs_cand) != 1:
            raise ValueError(f"entry {entry}: {len(hs_cand)} HS truth vertices")
        h = int(hs_cand[0])
        hz = float(tz[h])
        n_truth2 = int((tn >= 2).sum())
        peaks = pv_locations_updated_res_fast(hists[loc], *PEAK_ARGS)[0].astype(
            np.float64
        )
        np.testing.assert_array_equal(peaks, peaks_ref[entry])
        edges = {}
        for key, (off, sc, tg) in edge_files.items():
            lo, hi = off[loc], off[loc + 1]
            if hi - lo != n_trk:
                raise ValueError(f"entry {entry}: edge count mismatch ({key})")
            if n_trk and tg[lo:hi].max() >= len(peaks):
                raise ValueError(f"entry {entry}: edge target beyond peak list ({key})")
            edges[key] = (tg[lo:hi], sc[lo:hi])
        amvf_raw = [
            ak.to_numpy(x).astype(np.int64) for x in ev["RecoVertex_assocTracks"]
        ]
        amvf_lists = [np.unique(x) for x in amvf_raw]
        if any(len(x) and (x.min() < 0 or x.max() >= n_trk) for x in amvf_lists):
            raise ValueError(f"entry {entry}: AMVF track index out of range")
        amvf_z = ak.to_numpy(ev["RecoVertex_z"]).astype(np.float64)
        for a, arm in enumerate(ARMS):
            if arm == "amvf":
                lists_all, z_all = amvf_lists, amvf_z
            else:
                tgt = _arm_targets(arm, edges, z0, ez0, peaks, truth, tz)
                lists_all, z_all = hm.lists_from_targets(tgt, len(peaks)), peaks
            for m, min_trk in enumerate(MIN_TRACKS):
                keep = np.array([len(x) >= min_trk for x in lists_all], dtype=bool)
                lists = [x for x, k in zip(lists_all, keep) if k]
                zs = z_all[keep] if len(z_all) else z_all
                cls = hm.classify_vertices(lists, pt, truth)
                counts[loc, a, m] = [*cls.counts(), len(lists), n_truth2]
                hs[loc, a, m] = hm.hs_event_record(
                    lists, cls, truth, h, zs, hz, HS_WINDOW
                )
        # AMVF native primary (RecoVertex_type == 1), full stored list
        native = np.flatnonzero(ak.to_numpy(ev["RecoVertex_type"]) == 1)
        if len(native) != 1:
            raise ValueError(f"entry {entry}: {len(native)} stored AMVF primaries")
        nv = int(native[0])
        acls = hm.classify_vertices(amvf_lists, pt, truth)
        amatch = hm.hs_matched_index(
            hm.hs_track_counts(amvf_lists, acls, truth, h), acls
        )
        amvf2 = amvf_z[[len(x) >= 2 for x in amvf_lists]]
        t2 = tz[tn >= 2]
        others = np.abs(np.delete(tz, h) - hz) < 2.0
        ev_out[loc] = [
            entry, round(float(ev["ActualNumOfInt"]) * 100), n_truth2, n_trk, len(peaks),
            len(amvf2), tn[h], round(1000 * others.sum() / 4.0),
            int(len(peaks) > 0 and np.abs(peaks - hz).min() < HS_WINDOW),
            int(len(amvf2) > 0 and np.abs(amvf2 - hz).min() < HS_WINDOW),
            hm.greedy_one_to_one(peaks, t2, 0.5), hm.greedy_one_to_one(peaks, t2, WINDOW),
            hm.greedy_one_to_one(amvf2, t2, 0.5), hm.greedy_one_to_one(amvf2, t2, WINDOW),
            int(nv == amatch), int(acls.dominant[nv] == h), int(abs(amvf_z[nv] - hz) < HS_WINDOW),
            int(nv == hm.selected_index(acls)),
        ]  # fmt: skip
    return {"start": start, "counts": counts, "hs": hs, "events": ev_out}


def _reference_peaks(n: int) -> dict[int, np.ndarray]:
    """Peak positions saved by the note's PV-Finder evaluation (cross-check)."""
    with PKL.open("rb") as f:
        saved = pickle.load(f)
    entries = [int(r["event_idx"]) for r in saved["per_event"][:n]]
    if entries != list(range(n)):
        raise ValueError("Saved evaluation is not entries 0..n-1 in order")
    return {i: np.asarray(saved["pred_pvs_mm"][i], dtype=np.float64) for i in range(n)}


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-events", type=int, default=25000)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    if args.n_events % 500:
        raise ValueError("--n-events must be a multiple of 500 (chunk size)")
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    ref = _reference_peaks(args.n_events)
    print(f"loaded reference peaks in {time.monotonic() - t0:.0f} s", flush=True)
    starts = list(range(0, args.n_events, 500))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                process_chunk, s, s + 500, {i: ref[i] for i in range(s, s + 500)}
            )
            for s in starts
        ]
        for k, fut in enumerate(futures):
            results.append(fut.result())
            print(
                f"{k + 1}/{len(starts)} chunks; {time.monotonic() - t0:.0f} s",
                flush=True,
            )
    results.sort(key=lambda r: r["start"])
    np.savez_compressed(
        args.out / "hs_chain_events.npz",
        counts=np.concatenate([r["counts"] for r in results]),
        hs=np.concatenate([r["hs"] for r in results]),
        events=np.concatenate([r["events"] for r in results]),
    )
    meta = {
        "arms": ARMS, "min_tracks": MIN_TRACKS, "hs_fields": hm.HS_FIELDS,
        "event_fields": EVENT_FIELDS,
        "count_fields": ["clean", "merged", "split", "fake", "n_reco", "n_truth_ge2"],
        "peak_args": PEAK_ARGS, "hs_window_mm": HS_WINDOW, "window_mm": WINDOW,
        "root": str(ROOT), "histograms": str(HIST_DIR), "edges": {k: str(v) for k, v in EDGES.items()},
        "checkpoints": CHECKPOINTS, "reference_peaks": str(PKL),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metrics_sha256": hashlib.sha256(Path(hm.__file__).read_bytes()).hexdigest(),
        "runtime_s": time.monotonic() - t0,
    }  # fmt: skip
    (args.out / "hs_chain_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
