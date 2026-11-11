"""Identical-code measurements on Run 2 MC and HL-LHC (driver: ``run_all``).

Every number goes through one estimator per metric (``metrics``), on event
lists fixed up front:

``run2``        the 2 550 Run 2 PV-Finder test events (ROOT entries via qibin)
``hl_pu200``    HL-LHC r16443 events with 185 <= round(mu) <= 215 (1 920)
``hl_mu80``     HL-LHC events with round(mu) <= 80 (all, ~10 500)
``hl_mu_match`` 2 550 HL-LHC events drawn with Run 2's exact mu distribution
``hl_rho_match`` 2 550 HL-LHC events drawn with Run 2's distribution of event
                truth density rho = N(nTrk>=2) / (sqrt(2 pi) sigma_z(beam))

PV-Finder configurations (HL-LHC; Run 2 has only the first):
``int050_h000_full``  integral 0.5, no height floor, full-region mean = Run 2 config
``int050_h003_cen``   integral 0.5, floor 0.03, local centroid = the note's 0.229
``int030_h003_cen``   integral 0.3, floor 0.03, local centroid = counting point
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from pv_finder.diagnostics.amvf_fair_comparison.cache import load_cache
from pv_finder.diagnostics.amvf_fair_comparison.scan import (
    fit_sigma_vtx_vtx,
    sigmoid_fit,
)
from pv_finder.diagnostics.run2_vs_run4 import metrics as M
from pv_finder.diagnostics.run2_vs_run4 import physics, samples
from pv_finder.utils.pairwise_dz import in_summary_window

FAIR_CACHE = "outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz"
HL_CONFIGS = ("int050_h000_full", "int050_h003_cen", "int030_h003_cen")
N_BOOT = 200


def _t2(d: dict[str, Any]) -> list[np.ndarray]:
    return [z[n >= 2] for z, n in zip(d["truth_z"], d["truth_ntrk"])]


def _amvf(d: dict[str, Any], cut: int = 2, key: str = "amvf") -> list[np.ndarray]:
    return [z[n >= cut] for z, n in zip(d[f"{key}_z"], d[f"{key}_ntrk"])]


def _rho(d: dict[str, Any]) -> np.ndarray:
    """Event truth density at the beam-spot centre, nTrk>=2 vertices per mm."""
    n = np.array([len(t) for t in _t2(d)], dtype=float)
    return n / (np.sqrt(2 * np.pi) * d["beam_sigma_z"])


def selections(r2: dict[str, Any], hl: dict[str, Any]) -> dict[str, np.ndarray]:
    """Fixed event lists (indices into each sample's cache)."""
    nt = [len(t) for t in _t2(hl)]
    mu = hl["mu"]
    pu200 = [k for k in range(len(mu)) if in_summary_window(mu[k], nt[k], 185, 215)]
    mu80 = [k for k in range(len(mu)) if in_summary_window(mu[k], nt[k], 0, 80)]
    mu_edges = np.arange(-0.5, 216.0, 1.0)
    return {
        "run2": np.arange(len(r2["mu"])),
        "hl_pu200": np.asarray(pu200),
        "hl_mu80": np.asarray(mu80),
        "hl_mu_match": M.matched_indices(np.round(r2["mu"]), np.round(mu), mu_edges),
        "hl_rho_match": M.matched_indices(_rho(r2), _rho(hl), np.arange(0, 1.3, 0.02)),
    }


def legacy_pvf_fit(dz: np.ndarray) -> float:
    """run_eval_pvf.py (2026-04-16): 60 bins over +-6 mm, FIT_P0, maxfev 10000."""
    edges = np.linspace(-6.0, 6.0, 61)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    cnts = np.histogram(dz, bins=edges)[0].astype(float)
    popt, _ = curve_fit(sigmoid_fit, ctrs, cnts, p0=[1000.0, 10.0, 30.0, 0.8],
                        maxfev=10000, bounds=([0, 0, 0, 0], [np.inf] * 4))  # fmt: skip
    return float(abs(popt[3]))


def legacy_amvf_fit(z_by_event: list[np.ndarray]) -> float:
    """amvf_run3_performance_plots.py: symmetrised pairs, 80 bins over +-8 mm."""
    from pv_finder.diagnostics.amvf_run3_performance_plots import fit_resolution

    vals = []
    for z in z_by_event:
        z = np.asarray(z, dtype=np.float64)
        if len(z) >= 2:
            up = (z[:, None] - z[None, :])[np.triu_indices(len(z), k=1)]
            vals.extend([up, -up])
    _, _, _, sigma, _ = fit_resolution(np.concatenate(vals), dz_max_mm=8.0, n_bins=80)
    return float(sigma)


def reproduce(r2: dict, hl: dict, sel: dict, pkl_dz: np.ndarray) -> dict[str, Any]:
    """Published numbers from their own code, then the shared code on the same input."""
    fair = load_cache(FAIR_CACHE)
    if not np.array_equal(fair["event_idx"], sel["hl_pu200"]):
        raise ValueError("fair-comparison cache events != hl_pu200 selection")
    a_hl = _amvf(hl)
    for k, e in enumerate(fair["event_idx"]):
        if not np.array_equal(fair["amvf"][k], a_hl[e]):
            raise ValueError(f"AMVF positions differ from the fair cache at event {e}")
    out = {
        "run2_pvf_published_code_60bins": legacy_pvf_fit(pkl_dz),
        "run2_amvf_published_code_entries0-2499": legacy_amvf_fit(
            [z[n >= 2] for z, n in zip(r2["pub_amvf_z"], r2["pub_amvf_ntrk"])]
        ),
        "hl_amvf_fair_cache_fit": fit_sigma_vtx_vtx(fair["amvf"])[0],
        "hl_pvf_note_fit_recovered": fit_sigma_vtx_vtx(
            [hl["pvf_int050_h003_cen"][k] for k in sel["hl_pu200"]]
        )[0],
        "hl_pvf_note_pkl": hl["meta"]["recovery"]["int050_h003_cen"]["sigma_pkl_mm"],
        "run2_pvf_pkl": r2["meta"]["sigma_pkl_mm"],
    }
    return out


def arms_of(d: dict[str, Any], is_hl: bool) -> dict[str, list[np.ndarray]]:
    """PV-Finder configurations, AMVF nTrk>=2, and truth nTrk>=2."""
    a: dict[str, list[np.ndarray]] = {}
    if is_hl:
        for c in HL_CONFIGS:
            a[f"pvf_{c}"] = d[f"pvf_{c}"]
        # Floor 0.03 applied after the fact to the full-mean list: differs from
        # int050_h003_cen only in the position estimator.
        a["pvf_int050_h003_full"] = [
            z[h >= 0.03]
            for z, h in zip(d["pvf_int050_h000_full"], d["h_int050_h000_full"])
        ]
    else:
        a["pvf_int050_h000_full"] = d["pvf_int050_h000_full"]
    a["amvf"] = _amvf(d)
    a["truth"] = _t2(d)
    return a


def headline(
    H: dict[str, np.ndarray], edges: np.ndarray, idx: np.ndarray, pvf_key: str
) -> dict[str, Any]:
    """sigma_vtx-vtx for every arm on one event list, plus the paired AMVF/PVF ratio."""
    out = {a: M.sigma_summary(h, edges, idx, n_boot=N_BOOT) for a, h in H.items()}
    out["ratio_amvf_over_pvf"] = M.paired_ratio(H["amvf"], H[pvf_key], edges, idx,
                                                n_boot=N_BOOT)  # fmt: skip
    return out


def method_scans(
    arms: dict[str, list[np.ndarray]], idx: np.ndarray, pvf_key: str
) -> dict[str, Any]:
    """sigma vs binning and fit range, and the published AMVF method, for PVF and AMVF."""
    out: dict[str, Any] = {
        "bins_pm6": {},
        "range_0p04bins": {},
        "legacy_amvf_method": {},
    }
    for a in (pvf_key, "amvf"):
        z = [arms[a][k] for k in idx]
        out["bins_pm6"][a] = {
            nb: M.fit_counts(*_hist(z, nb, 6.0))[0] for nb in (60, 100, 150, 300, 600)
        }
        out["range_0p04bins"][a] = {
            r: M.fit_counts(*_hist(z, int(round(2 * r / 0.04)), r))[0]
            for r in (3.0, 4.0, 6.0, 8.0, 10.0)
        }
        out["legacy_amvf_method"][a] = legacy_amvf_fit(z)
    return out


def _hist(
    z: list[np.ndarray], n_bins: int, range_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    H, edges = M.event_hists(z, n_bins=n_bins, range_mm=range_mm)
    return edges, H.sum(0)


def pair_stat_test(
    H: dict[str, np.ndarray], edges: np.ndarray, idx: np.ndarray, target_pairs: dict,
    n_rep: int = 100,
) -> dict[str, Any]:  # fmt: skip
    """HL-LHC sigma on random event subsets with Run 2's pair count, per arm."""
    rng = np.random.default_rng(M.SEED)
    out = {}
    for a, n_target in target_pairs.items():
        per_evt = H[a][idx].sum(1)
        n_evt = max(int(round(n_target / per_evt.mean())), 5)
        vals = [
            M.fit_counts(edges, H[a][rng.choice(idx, n_evt, replace=False)].sum(0))[0]
            for _ in range(n_rep)
        ]
        out[a] = {"n_events": n_evt, "target_pairs": int(n_target),
                  "mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)),
                  "median": float(np.nanmedian(vals))}  # fmt: skip
    return out


def ntrk_scan(d: dict[str, Any], idx: np.ndarray) -> dict[int, float]:
    """AMVF sigma_vtx-vtx vs the reco nTracks cut."""
    return {
        c: M.fit_counts(*_hist([_amvf(d, c)[k] for k in idx], 300, 6.0))[0]
        for c in (2, 3, 4, 5, 6, 8)
    }


def hs_exclusion(arms: dict, d: dict, idx: np.ndarray, pvf_key: str) -> dict[str, Any]:
    """Drop every vertex within R of the true HS (removes HS splitting) and refit."""
    hsz = np.array([z[t == 1][0] for z, t in zip(d["truth_z"], d["truth_type"])])
    out = {}
    for R in (2.0, 5.0):
        res = {}
        for a in (pvf_key, "amvf"):
            z = [arms[a][k][np.abs(arms[a][k] - hsz[k]) > R] for k in idx]
            res[a] = M.fit_counts(*_hist(z, 300, 6.0))[0]
        res["ratio"] = res["amvf"] / res[pvf_key]
        out[f"R{R:g}mm"] = res
    return out


def run_all(out_dir: Path) -> None:
    """Compute everything, write results.json and the figures."""
    import pickle

    from pv_finder.diagnostics.run2_vs_run4 import plots

    r2 = samples.load(out_dir / "run2_cache.npz")
    hl = samples.load(out_dir / "hl_cache.npz")
    sel = selections(r2, hl)
    res: dict[str, Any] = {"selections": {k: int(len(v)) for k, v in sel.items()}}
    res["selection_dups"] = {k: int(len(v) - len(np.unique(v))) for k, v in sel.items()}
    with open(r2["meta"]["pkl"], "rb") as fh:
        pkl_dz = pickle.load(fh)["pairwise_dz_mm"]
    res["reproduction"] = reproduce(r2, hl, sel, pkl_dz)
    print("[repro]", res["reproduction"], flush=True)
    del pkl_dz

    A2, AH = arms_of(r2, False), arms_of(hl, True)
    res["verify_scan"] = {
        "run2_pvf": M.verify_against_scan(A2["pvf_int050_h000_full"]),
        "hl_pu200_amvf": M.verify_against_scan(
            [AH["amvf"][k] for k in sel["hl_pu200"]]
        ),
    }
    H2 = {a: M.event_hists(v)[0] for a, v in A2.items()}
    HH = {a: M.event_hists(v)[0] for a, v in AH.items()}
    edges = M.event_hists([np.zeros(2)])[1]
    pk = "pvf_int050_h000_full"
    res["sigma"] = {"run2": headline(H2, edges, sel["run2"], pk)}
    for s in ("hl_pu200", "hl_mu80", "hl_mu_match", "hl_rho_match"):
        res["sigma"][s] = headline(HH, edges, sel[s], pk)
        res["sigma"][s]["ratio_amvf_over_pvf_note"] = M.paired_ratio(
            HH["amvf"], HH["pvf_int050_h003_cen"], edges, sel[s], n_boot=N_BOOT
        )
        print(f"[sigma] {s} done", flush=True)
    res["method_scans"] = {
        "run2": method_scans(A2, sel["run2"], pk),
        "hl_pu200": method_scans(AH, sel["hl_pu200"], pk),
    }
    tp = {a: res["sigma"]["run2"][a]["n_pairs"] for a in (pk, "amvf")}
    res["pair_stat_test_hl_pu200"] = pair_stat_test(HH, edges, sel["hl_pu200"], tp)
    res["ntrk_scan"] = {"run2": ntrk_scan(r2, sel["run2"]),
                        "hl_pu200": ntrk_scan(hl, sel["hl_pu200"]),
                        "hl_rho_match": ntrk_scan(hl, sel["hl_rho_match"])}  # fmt: skip
    res["hs_exclusion"] = {"run2": hs_exclusion(A2, r2, sel["run2"], pk),
                           "hl_pu200": hs_exclusion(AH, hl, sel["hl_pu200"], pk)}  # fmt: skip
    print("[scans] done", flush=True)

    phys = physics.run_physics(r2, hl, sel, A2, AH, H2, HH, edges, out_dir)
    res.update(phys["results"])
    vp = res["vertex_precision"]
    res["sigma_over_ideal_pu_precision"] = {
        s: {a: res["sigma"][s][a]["sigma_mm"] / (1e-3 * vp[t]["pu_median_um"])
            for a in (pk, "amvf")}
        for s, t in (("run2", "run2"), ("hl_pu200", "hl"), ("hl_rho_match", "hl"))
    }  # fmt: skip
    with open(out_dir / "results.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"[done] wrote {out_dir / 'results.json'}", flush=True)
    plots.make_all(res, phys["arrays"], H2, HH, edges, sel, r2, hl, out_dir)
