#!/usr/bin/env python3
"""HS longitudinal resolution with alternative PV-Finder position estimators.

Same definition as ``scripts/itk_hs_resolution.py`` (closest vertex within
0.1 mm of the true HS, iterative 3-RMS starting at +-0.1 mm), on the note's
1,920 PU200 events of r16443, restricted to the events in which BOTH
PV-Finder and AMVF (nTrk >= 2) reconstruct the HS within 0.1 mm. No inference:
saved PV-Finder histograms and saved top-1 GNN edges (v4 noaug e107, the
note's checkpoint), as in ``gnn.diagnostics.hs_chain_study``.

Estimators for the PV-Finder HS vertex (the peak closest to the HS):
  centroid      production local centroid (+-3 bins)
  parab_lin/log 3-point parabola on heights / log heights
  refit_tXX     inverse-variance mean of z0 of the tracks the GNN assigns to
                that peak at score >= 0.XX; robust_tXX drops > 3 sigma tracks
  refit_truth   same with the true HS tracks (ideal track-based precision)
  target_*      the training target histogram of the event read out the same
                way (floor of the 0.04 mm encoding)
A refit with no assigned track falls back to the centroid (counted).

    PYTHONPATH=src venv/bin/python -u scripts/hs_position_estimators.py \
        --out outputs/11_11_2026_output/hs_position_estimators
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pv_finder.data.root_to_h5 as r2h  # noqa: E402
from gnn.diagnostics.hs_chain_study import (  # noqa: E402
    EDGES,
    HIST_DIR,
    PEAK_ARGS,
    PKL,
    ROOT,
)
from pv_finder.evaluation.vertex_finding.hs_position_estimators import (  # noqa: E402
    adaptive_beamline_fit,
    adaptive_z_fit,
    parabolic_peak,
    robust_weighted_z0_mean,
    weighted_z0_mean,
)
from pv_finder.evaluation.vertex_finding.itk_resolution import (  # noqa: E402
    binned_iterative_rms,
    iterative_rms,
)
from pv_finder.utils.peak_finding_fast import (
    pv_locations_updated_res_fast,  # noqa: E402
)

HS_R = 0.1  # mm
RES_PRESET = (0.14318, 0.362, -0.01797)  # training h5 attrs (r16438 PU200)
CUTS = (98, 95, 0)
NTRK_EDGES = np.array([2, 20, 30, 40, 50, 60, 80, 130])
BRANCHES = [
    "ActualNumOfInt", "RecoTrack_z0", "RecoTrack_ErrZ0", "RecoTrack_d0",
    "RecoTrack_ErrD0", "TruthVertex_z",
    "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type",
    "RecoVertex_z", "RecoVertex_nTracks", "RecoVertex_assocTracks",
    "RecoTrack_phi", "RecoTrack_theta", "RecoTrack_ErrD0Z0", "BeamPosX",
    "BeamPosY", "BeamPosSigmaX",
]  # fmt: skip
METHODS = ("refit", "robust", "robseed", "robseed_d0")
SETS = tuple(f"t{c:03d}" for c in CUTS) + ("truth",)
EST = (
    ["centroid", "parab_lin", "parab_log"]
    + [f"{m}_{t}" for t in SETS for m in METHODS]
    + ["ada_t098", "ada_t000", "ada_score", "ada_win1mm", "ada_truth"]
    + [f"ada3d_{v}" for v in ("t098", "t095", "t000", "score", "win1mm", "truth")]
    + ["ada3d_amvftrk"]
    + ["target_centroid", "target_parab_log", "amvf"]
)
REFIT_REF = "robseed_t098"  # 2026-11-11 headline simple refit (12.02 um)
D0_SIG_MAX = 3.0


def _closest(z: np.ndarray, hz: float) -> int:
    if len(z) == 0:
        return -1
    k = int(np.argmin(np.abs(z - hz)))
    return k if abs(z[k] - hz) <= HS_R else -1


def process_chunk(start: int, entries: list[int], ref: dict[int, np.ndarray]) -> dict:
    """Per-event residuals [mm] (NaN = not reconstructed) for one 500-entry chunk."""
    r2h.set_resolution(*RES_PRESET)
    data = uproot.open(ROOT)["PVFinderData"].arrays(
        BRANCHES, entry_start=start, entry_stop=start + 500, library="ak"
    )
    with np.load(HIST_DIR / f"chunk_{start:05d}_{start + 500:05d}.npz") as f:
        hists = f["histograms"]
    with np.load(EDGES["v4"] / f"best_edges_{start:05d}.npz") as f:
        off, scores, targets = f["offsets"], f["scores"], f["targets"]
    rows, info, ovl = [], [], []
    for e in entries:
        loc = e - start
        ev = data[loc]
        z0 = ak.to_numpy(ev["RecoTrack_z0"]).astype(np.float64)
        ez0 = ak.to_numpy(ev["RecoTrack_ErrZ0"]).astype(np.float64)
        tz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        h = int(np.flatnonzero(ak.to_numpy(ev["TruthVertex_type"]) == 1)[0])
        hz = tz[h]
        hist = hists[loc]
        peaks, _, pbin, _ = pv_locations_updated_res_fast(hist, *PEAK_ARGS)
        peaks = peaks.astype(np.float64)
        np.testing.assert_array_equal(peaks, ref[e])
        lo, hi = off[loc], off[loc + 1]
        if hi - lo != len(z0):
            raise ValueError(f"entry {e}: edge count mismatch")
        r = dict.fromkeys(EST, np.nan)
        k = _closest(peaks, hz)
        n_fallback = {c: 0 for c in CUTS}
        if k >= 0:
            r["centroid"] = peaks[k] - hz
            r["parab_lin"] = parabolic_peak(hist, pbin[k]) - hz
            r["parab_log"] = parabolic_peak(hist, pbin[k], log=True) - hz
            d0 = ak.to_numpy(ev["RecoTrack_d0"]).astype(np.float64)
            ed0 = ak.to_numpy(ev["RecoTrack_ErrD0"]).astype(np.float64)
            prompt = np.abs(d0) <= D0_SIG_MAX * ed0
            hs_trk = ak.to_numpy(ev["TruthVertex_assocTracks"][h]).astype(np.int64)
            hs_trk = hs_trk[(hs_trk >= 0) & (hs_trk < len(z0))]
            truth_sel = np.zeros(len(z0), dtype=bool)
            truth_sel[hs_trk] = True
            sels = {
                f"t{c:03d}": (targets[lo:hi] == k) & (scores[lo:hi] >= c / 100)
                for c in CUTS
            }
            sels["truth"] = truth_sel
            for t, sel in sels.items():
                if not sel.any():
                    if t != "truth":
                        n_fallback[int(t[1:])] = 1
                    for m in METHODS:
                        r[f"{m}_{t}"] = r["centroid"]
                    continue
                zs, es = z0[sel], ez0[sel]
                r[f"refit_{t}"] = weighted_z0_mean(zs, es) - hz
                r[f"robust_{t}"] = robust_weighted_z0_mean(zs, es) - hz
                r[f"robseed_{t}"] = robust_weighted_z0_mean(zs, es, seed=peaks[k]) - hz
                sp = sel & prompt
                r[f"robseed_d0_{t}"] = (
                    robust_weighted_z0_mean(z0[sp], ez0[sp], seed=peaks[k]) - hz
                    if sp.any()
                    else r["centroid"]
                )
        if k >= 0:
            zk, sel_k = peaks[k], targets[lo:hi] == k
            for name, sel, pr in (
                ("ada_t098", sel_k & (scores[lo:hi] >= 0.98), None),
                ("ada_t000", sel_k, None),
                ("ada_score", sel_k, scores[lo:hi]),
                ("ada_win1mm", np.abs(z0 - zk) <= 1.0, None),
                ("ada_truth", truth_sel, None),
            ):
                prior = None if pr is None else pr[sel]
                r[name] = adaptive_z_fit(z0[sel], ez0[sel], zk, prior=prior) - hz
            trk = {
                f: ak.to_numpy(ev[f"RecoTrack_{f}"]).astype(np.float64)
                for f in ("d0", "phi", "theta", "ErrD0", "ErrD0Z0")
            }
            beam = (float(ev["BeamPosX"]), float(ev["BeamPosY"]))
            sig_b = float(ev["BeamPosSigmaX"])
            amvf_sel = np.zeros(len(z0), dtype=bool)
            rz_all = ak.to_numpy(ev["RecoVertex_z"]).astype(np.float64)
            rn_all = ak.to_numpy(ev["RecoVertex_nTracks"])
            cand = np.flatnonzero(rn_all >= 2)
            if len(cand):
                ka3 = cand[np.argmin(np.abs(rz_all[cand] - hz))]
                at = ak.to_numpy(ev["RecoVertex_assocTracks"][ka3]).astype(np.int64)
                amvf_sel[at[(at >= 0) & (at < len(z0))]] = True
            sc = scores[lo:hi]
            for name, sel, pr in (
                ("ada3d_t098", sel_k & (sc >= 0.98), None),
                ("ada3d_t095", sel_k & (sc >= 0.95), None),
                ("ada3d_t000", sel_k, None),
                ("ada3d_score", sel_k, sc),
                ("ada3d_win1mm", np.abs(z0 - zk) <= 1.0, None),
                ("ada3d_truth", truth_sel, None),
                ("ada3d_amvftrk", amvf_sel, None),
            ):
                if not sel.any():
                    r[name] = r["centroid"]
                    continue
                fit = adaptive_beamline_fit(
                    trk["d0"][sel], z0[sel], trk["phi"][sel], trk["theta"][sel],
                    trk["ErrD0"][sel] ** 2, ez0[sel] ** 2, trk["ErrD0Z0"][sel], zk,
                    beam_xy=beam, beam_sigma_xy=sig_b,
                    prior=None if pr is None else pr[sel],
                )  # fmt: skip
                r[name] = fit[2] - hz
            ovl.append(
                [
                    int((sel_k & (sc >= 0.98) & amvf_sel).sum()),
                    int((sel_k & (sc >= 0.98) & ~amvf_sel).sum()),
                    int((~(sel_k & (sc >= 0.98)) & amvf_sel).sum()),
                    int((sel_k & (sc >= 0.98) & truth_sel).sum()),
                    int((amvf_sel & truth_sel).sum()),
                    int(truth_sel.sum()),
                    int(amvf_sel.sum()),
                    int((sel_k & (sc >= 0.98)).sum()),
                ]
            )
        else:
            ovl.append([0] * 8)
        target = r2h._build_truth_histogram(tz, tn)[0].astype(np.float32)
        tpk, _, tbin, _ = pv_locations_updated_res_fast(target, *PEAK_ARGS)
        kt = _closest(tpk.astype(np.float64), hz)
        if kt >= 0:
            r["target_centroid"] = float(tpk[kt]) - hz
            r["target_parab_log"] = parabolic_peak(target, tbin[kt], log=True) - hz
        rz = ak.to_numpy(ev["RecoVertex_z"]).astype(np.float64)
        rn = ak.to_numpy(ev["RecoVertex_nTracks"])
        ka = _closest(rz[rn >= 2], hz)
        if ka >= 0:
            r["amvf"] = rz[rn >= 2][ka] - hz
        rows.append([r[x] for x in EST])
        info.append(
            [e, float(ev["ActualNumOfInt"]), tn[h], *[n_fallback[c] for c in CUTS]]
        )
    return {
        "res": np.array(rows),
        "info": np.array(info, dtype=np.float64),
        "ovl": np.array(ovl, dtype=np.int64),
    }


def bootstrap(res: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_est) iterative RMS on paired event resamples."""
    rng = np.random.default_rng(seed)
    n = len(res)
    out = np.empty((n_boot, res.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        out[b] = [iterative_rms(res[pick, j], HS_R).rms for j in range(res.shape[1])]
    return out


def time_refit(n: int = 20000) -> float:
    """Mean cost [us] of the robust refit on a 60-track vertex."""
    rng = np.random.default_rng(0)
    z, e = rng.normal(0, 0.03, 60), rng.uniform(0.01, 0.1, 60)
    t0 = time.perf_counter()
    for _ in range(n):
        robust_weighted_z0_mean(z, e, seed=0.0)
    return 1e6 * (time.perf_counter() - t0) / n


def time_adaptive(n: int = 5000) -> float:
    """Mean cost [us] of the adaptive fit on a 60-track vertex."""
    rng = np.random.default_rng(0)
    z, e = rng.normal(0, 0.03, 60), rng.uniform(0.01, 0.1, 60)
    t0 = time.perf_counter()
    for _ in range(n):
        adaptive_z_fit(z, e, 0.0)
    return 1e6 * (time.perf_counter() - t0) / n


def analyse(res: np.ndarray, info: np.ndarray, n_boot: int) -> dict:
    """Common-set numbers, paired differences and vs-ntrk curves."""
    j = {name: i for i, name in enumerate(EST)}
    common = np.isfinite(res[:, j["centroid"]]) & np.isfinite(res[:, j["amvf"]])
    rc, ic = res[common], info[common]
    boot = bootstrap(rc, n_boot, 20261111)
    out: dict = {
        "n_events_pu200": int(len(res)),
        "n_pvf_hs_reco": int(np.isfinite(res[:, j["centroid"]]).sum()),
        "n_amvf_hs_reco": int(np.isfinite(res[:, j["amvf"]]).sum()),
        "n_common": int(common.sum()),
        "refit_fallbacks": {
            f"t{c:03d}": int(ic[:, 3 + m].sum()) for m, c in enumerate(CUTS)
        },
        "estimators": {},
    }
    for name in EST:
        k = j[name]
        it = iterative_rms(rc[:, k], HS_R)
        out["estimators"][name] = {
            "rms_um": 1e3 * it.rms,
            "boot_err_um": 1e3 * float(np.std(boot[:, k], ddof=1)),
            "mean_um": 1e3 * it.mean,
            "n_in": it.n_in,
            "n_total": it.n_total,
            "minus_amvf_um": 1e3 * (it.rms - iterative_rms(rc[:, j["amvf"]], HS_R).rms),
            "minus_amvf_err_um": 1e3
            * float(np.std(boot[:, k] - boot[:, j["amvf"]], ddof=1)),
            "minus_refit_um": 1e3
            * (it.rms - iterative_rms(rc[:, j[REFIT_REF]], HS_R).rms),
            "minus_refit_err_um": 1e3
            * float(np.std(boot[:, k] - boot[:, j[REFIT_REF]], ddof=1)),
            "minus_centroid_um": 1e3
            * (it.rms - iterative_rms(rc[:, j["centroid"]], HS_R).rms),
            "minus_centroid_err_um": 1e3
            * float(np.std(boot[:, k] - boot[:, j["centroid"]], ddof=1)),
        }
    out["vs_ntrk"] = {
        name: binned_iterative_rms(ic[:, 2], rc[:, j[name]], NTRK_EDGES, HS_R)
        for name in EST
    }
    out["refit_cost_us"] = time_refit()
    out["adaptive_cost_us"] = time_adaptive()
    return out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with PKL.open("rb") as f:
        saved = pickle.load(f)
    pu = [
        (int(r["event_idx"]), i)
        for i, r in enumerate(saved["per_event"])
        if 185 <= r["mu"] <= 215
    ]
    ref = {e: np.asarray(saved["pred_pvs_mm"][i], dtype=np.float64) for e, i in pu}
    chunks: dict[int, list[int]] = {}
    for e, _ in pu:
        chunks.setdefault(500 * (e // 500), []).append(e)
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(process_chunk, s, es, {e: ref[e] for e in es})
            for s, es in sorted(chunks.items())
        ]
        parts = [f.result() for f in futs]
    res = np.concatenate([x["res"] for x in parts])
    info = np.concatenate([x["info"] for x in parts])
    ovl = np.concatenate([x["ovl"] for x in parts])
    print(f"{len(res)} events extracted in {time.monotonic() - t0:.0f} s", flush=True)
    np.savez_compressed(
        args.out / "events.npz", res=res, info=info, ovl=ovl, est=np.array(EST)
    )
    out = analyse(res, info, args.n_boot)
    o = ovl[np.isfinite(res[:, EST.index("centroid")])].sum(axis=0)
    out["hs_track_lists_t098"] = {
        "gnn_and_amvf": int(o[0]), "gnn_not_amvf": int(o[1]),
        "amvf_not_gnn": int(o[2]), "gnn_true_hs": int(o[3]),
        "amvf_true_hs": int(o[4]), "true_hs": int(o[5]),
        "amvf_total": int(o[6]), "gnn_total": int(o[7]),
    }  # fmt: skip
    (args.out / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, v in out["estimators"].items():
        print(
            f"{name:18s} {v['rms_um']:6.2f} +- {v['boot_err_um']:.2f}  "
            f"vs AMVF {v['minus_amvf_um']:+6.2f} +- {v['minus_amvf_err_um']:.2f}  "
            f"vs refit {v['minus_refit_um']:+6.2f} +- {v['minus_refit_err_um']:.2f}  "
            f"vs centroid {v['minus_centroid_um']:+6.2f} +- "
            f"{v['minus_centroid_err_um']:.2f}  mean {v['mean_um']:+.2f}"
        )
    print(json.dumps({k: out[k] for k in out if k not in ("estimators", "vs_ntrk")}))


if __name__ == "__main__":
    main()
