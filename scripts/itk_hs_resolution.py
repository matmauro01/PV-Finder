#!/usr/bin/env python3
"""ITk-paper-style hard-scatter longitudinal resolution, PV-Finder vs AMVF.

Definition (arXiv:2412.15090 Section 5.5 / Figure 28, with the estimator of
the Athena monitoring that makes such plots; see
``pv_finder.evaluation.vertex_finding.itk_resolution``):

* per event, residual z_reco - z_HS for the reconstructed vertex matched to the
  true HS (TruthVertex_type == 1); events whose HS is not reconstructed are
  not filled;
* width = iterative RMS truncated at mean +- 3 RMS (Athena ``iterRMS_convergence``),
  error RMS / sqrt(2N); also a paired event bootstrap for the PU200 numbers;
* plotted against the local pile-up density (other truth vertices within 2 mm
  of the HS, divided by 4 mm) and against mu.

HS matching (our ntuples have no vertex-fit truth weights):

* ``pos``: positional, the closest reconstructed vertex within the HS
  reconstruction radius.  Primary radius 0.1 mm, the ITk Figure 26(b)
  criterion; 0.5 and 1.0 mm as checks.  Used for both algorithms.
* ``trk`` (AMVF only): the AMVF vertex sharing the most tracks with the HS
  truth vertex (ties: closest in z), closer to Athena's truth-based match.

No inference.  PV-Finder positions are the production peaks saved for the
note's comparison (integral 0.30, height 0.03, local centroid).

    venv/bin/python -u scripts/itk_hs_resolution.py \
        --outdir outputs/11_09_2026_output/itk_hs_resolution
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_finder.evaluation.vertex_finding.itk_resolution import (  # noqa: E402
    binned_iterative_rms,
    hs_residual,
    iterative_rms,
    local_pu_density,
)

CACHE = "outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz"
RADII_MM = (0.1, 0.5, 1.0)
PRIMARY_RADIUS_MM = 0.1
DENSITY_EDGES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 4.0])
MU_EDGES = np.arange(0.0, 201.0, 20.0)
NTRK_EDGES = np.array([2, 20, 30, 40, 50, 60, 80, 130])
ARMS = ("pvf", "amvf_pos", "amvf_trk")


def best_track_match(
    reco_tracks: list[np.ndarray],
    reco_z: np.ndarray,
    hs_tracks: np.ndarray,
    hs_z: float,
) -> tuple[float, int]:
    """Residual of the reco vertex sharing most tracks with the HS; (nan, 0) if none."""
    hs_set = set(np.asarray(hs_tracks).tolist())
    best, best_n = -1, 0
    for k, trks in enumerate(reco_tracks):
        n = len(hs_set.intersection(np.asarray(trks).tolist()))
        if n > best_n or (
            n == best_n and n > 0 and abs(reco_z[k] - hs_z) < abs(reco_z[best] - hs_z)
        ):
            best, best_n = k, n
    if best < 0:
        return float("nan"), 0
    return float(reco_z[best] - hs_z), best_n


def extract(cache_path: str) -> tuple[dict[str, np.ndarray], dict]:
    """Per-event HS residuals with explicit alignment checks against the pkl."""
    import uproot

    cache = np.load(cache_path, allow_pickle=False)
    meta = json.loads(str(cache["meta"]))
    with open(meta["pkl"], "rb") as fh:
        saved = pickle.load(fh)
    recs = saved["per_event"]
    idx = np.array([r["event_idx"] for r in recs], dtype=np.int64)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("Repeated event indices in the saved evaluation")
    branches = [
        "TruthVertex_z", "TruthVertex_type", "TruthVertex_nTracks",
        "TruthVertex_assocTracks", "RecoVertex_z", "RecoVertex_nTracks",
        "RecoVertex_assocTracks", "ActualNumOfInt", "BeamPosZ",
    ]  # fmt: skip
    tree = uproot.open(meta["root"])["PVFinderData"]
    print(f"reading {idx.max() + 1} entries of {meta['root']}", flush=True)
    root = tree.arrays(branches, entry_stop=int(idx.max()) + 1, library="np")
    rows: list[dict[str, float]] = []
    skipped: list[int] = []
    for k, e in enumerate(idx):
        tz = np.asarray(root["TruthVertex_z"][e], dtype=np.float64)
        tn = np.asarray(root["TruthVertex_nTracks"][e])
        tt = np.asarray(root["TruthVertex_type"][e])
        rz = np.asarray(root["RecoVertex_z"][e], dtype=np.float64)
        rn = np.asarray(root["RecoVertex_nTracks"][e])
        pvf = np.asarray(saved["pred_pvs_mm"][k], dtype=np.float64)
        if not np.array_equal(
            np.sort(tz[tn >= 2]), np.sort(np.asarray(saved["truth_pvs_mm"][k], float))
        ):
            raise ValueError(f"truth alignment failed at entry {e}")
        if recs[k]["n_pred"] != len(pvf) or float(root["BeamPosZ"][e]) != 0.0:
            raise ValueError(f"multiplicity or beam alignment failed at entry {e}")
        hs = np.flatnonzero(tt == 1)
        if len(hs) != 1:
            skipped.append(int(e))
            continue
        h = int(hs[0])
        good = rn >= 2
        amvf_z = rz[good]
        amvf_trk = [t for t, g in zip(root["RecoVertex_assocTracks"][e], good) if g]
        row = {
            "event_idx": float(e),
            "mu": float(root["ActualNumOfInt"][e]),
            "hs_z": float(tz[h]),
            "hs_ntrk": float(tn[h]),
            "density": local_pu_density(tz, h),
        }
        for r in RADII_MM:
            row[f"pvf@{r:g}"] = hs_residual(pvf, tz[h], r)
            row[f"amvf_pos@{r:g}"] = hs_residual(amvf_z, tz[h], r)
        dz, nshared = best_track_match(
            amvf_trk, amvf_z, root["TruthVertex_assocTracks"][e][h], tz[h]
        )
        row["amvf_trk"] = dz
        row["amvf_trk_nshared"] = float(nshared)
        rows.append(row)
    ev = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    meta.update(n_input=len(idx), n_events=len(rows), skipped_no_unique_hs=skipped)
    return ev, meta


def residuals(ev: dict[str, np.ndarray], arm: str, radius: float) -> np.ndarray:
    """Residual column for ``arm`` at HS radius ``radius`` (trk: |dz| <= radius)."""
    if arm == "amvf_trk":
        r = ev["amvf_trk"].copy()
        r[~(np.abs(r) <= radius)] = np.nan
        return r
    return ev[f"{arm}@{radius:g}"]


def paired_bootstrap(
    res: dict[str, np.ndarray], start: float, n_boot: int, seed: int
) -> dict[str, dict[str, float]]:
    """Event bootstrap of the iterative RMS for each arm and PVF minus AMVF."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(res.values())))
    draws: dict[str, list[float]] = {a: [] for a in res}
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        for a, r in res.items():
            draws[a].append(iterative_rms(r[pick], start).rms)
    out = {a: {"boot_err": float(np.std(v, ddof=1))} for a, v in draws.items()}
    if "pvf" in res and "amvf_pos" in res:
        d = np.array(draws["pvf"]) - np.array(draws["amvf_pos"])
        out["pvf_minus_amvf_pos"] = {"boot_err": float(np.std(d, ddof=1))}
    return out


def analyse(ev: dict[str, np.ndarray], n_boot: int, seed: int) -> dict:
    """PU200 numbers, sensitivity grid, and binned curves."""
    pu = (ev["mu"] >= 185) & (ev["mu"] <= 215)
    out: dict = {"n_events_all": int(len(ev["mu"])), "n_events_pu200": int(pu.sum())}
    grid = {}
    for radius in RADII_MM:
        for start in sorted({radius, 1.0}):
            key = f"radius{radius:g}_start{start:g}"
            grid[key] = {
                a: iterative_rms(residuals(ev, a, radius)[pu], start)._asdict()
                for a in ARMS
            }
    out["pu200_grid"] = grid
    r0 = PRIMARY_RADIUS_MM
    prim = {a: residuals(ev, a, r0)[pu] for a in ARMS}
    out["pu200_primary"] = {
        "radius_mm": r0,
        "start_halfrange_mm": r0,
        **{a: iterative_rms(prim[a], r0)._asdict() for a in ARMS},
        "bootstrap": paired_bootstrap(prim, r0, n_boot, seed),
        "median_abs_um": {a: float(np.nanmedian(np.abs(prim[a])) * 1e3) for a in ARMS},
        "trk_vs_pos_same_vertex_frac": float(
            np.mean(np.isclose(ev["amvf_trk"][pu], ev[f"amvf_pos@{r0:g}"][pu]))
        ),
    }
    curves = {}
    for name, var, edges, mask in (
        ("density_all", "density", DENSITY_EDGES, np.ones_like(pu)),
        ("density_pu200", "density", DENSITY_EDGES, pu),
        ("mu_all", "mu", MU_EDGES, np.ones_like(pu)),
        ("ntrk_pu200", "hs_ntrk", NTRK_EDGES, pu),
    ):
        curves[name] = {
            a: binned_iterative_rms(
                ev[var][mask], residuals(ev, a, r0)[mask], edges, r0
            )
            for a in ARMS
        }
    out["curves"] = curves
    return out


def main(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    npz = outdir / "events.npz"
    if args.reuse_events and npz.exists():
        ev = dict(np.load(npz))
        meta = json.loads((outdir / "meta.json").read_text())
    else:
        ev, meta = extract(args.cache)
        np.savez_compressed(npz, **ev)
        (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    res = analyse(ev, args.n_boot, args.seed)
    res["meta"] = meta
    (outdir / "results.json").write_text(json.dumps(res, indent=1))
    p = res["pu200_primary"]
    print(
        f"PU200 events: {res['n_events_pu200']}  (radius/start {PRIMARY_RADIUS_MM} mm)"
    )
    for a in ARMS:
        r = p[a]
        print(
            f"  {a:9s} iterRMS = {1e3 * r['rms']:.2f} +- {1e3 * r['rms_err']:.2f} "
            f"(boot {1e3 * p['bootstrap'][a]['boot_err']:.2f}) um, N={r['n_in']}/"
            f"{r['n_total']}, mean {1e3 * r['mean']:+.2f} um, iters {r['n_iter']}"
        )
    print(json.dumps({k: {a: round(1e3 * v[a]["rms"], 2) for a in ARMS}
                      for k, v in res["pu200_grid"].items()}, indent=1))  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--outdir", default="outputs/11_09_2026_output/itk_hs_resolution")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--reuse-events", action="store_true")
    main(ap.parse_args())
