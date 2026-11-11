"""Position resolutions, density/mu dependence and detector inputs.

Called by ``analysis.run_all`` with the per-event caches and histograms it has
already built, so every curve here uses the same event lists and estimators as
the headline table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pv_finder.data.resolution_presets import RESOLUTION_PRESETS
from pv_finder.diagnostics.run2_vs_run4 import metrics as M
from pv_finder.diagnostics.run2_vs_run4 import targets as T
from pv_finder.evaluation.vertex_finding.itk_resolution import binned_iterative_rms

RHO_EDGES = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.95, 1.1])
MU_EDGES = np.array([0, 20, 40, 60, 80, 100, 130, 160, 185, 216])
LOCAL_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
M_BOOT = 200
H5_RUN2 = "/share/lazy/qibinlei/recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5"
H5_HL = "data/run4_all_etas/PU200_alleta_h5/ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.h5"
ROOT_HL_TRAIN = (
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root"
)


def sigma_in_bins(
    H: dict[str, np.ndarray], edges: np.ndarray, x: np.ndarray, bins: np.ndarray,
    arms: tuple[str, ...], n_boot: int = 100,
) -> list[dict[str, Any]]:  # fmt: skip
    """sigma_vtx-vtx per arm in bins of an event variable, with bootstrap errors."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = np.flatnonzero((x >= lo) & (x < hi))
        if len(idx) < 20:
            continue
        row: dict[str, Any] = {
            "lo": float(lo),
            "hi": float(hi),
            "n_events": int(len(idx)),
        }
        for a in arms:
            row[a] = M.sigma_summary(H[a], edges, idx, n_boot=n_boot)
        if "amvf" in arms:
            pv = [a for a in arms if a.startswith("pvf")][0]
            row["ratio"] = M.paired_ratio(H["amvf"], H[pv], edges, idx, n_boot=n_boot)
        rows.append(row)
    return rows


def position_tables(
    A: dict[str, list[np.ndarray]], sels: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Core p68 (optimal one-to-one, 1 mm) per arm and selection."""
    need = np.unique(np.concatenate(list(sels.values())))
    keep = np.zeros(len(A["truth"]), dtype=bool)
    keep[need] = True
    out: dict[str, dict[str, Any]] = {s: {} for s in sels}
    for a in A:
        if a == "truth":
            continue
        res = [
            M.matched_residuals_optimal(t, r, 1.0)
            if k and len(t) and len(r)
            else np.zeros(0)
            for t, r, k in zip(A["truth"], A[a], keep)
        ]
        for s, idx in sels.items():
            out[s][a] = M.p68_summary(res, idx, n_boot=M_BOOT)
    return out


def hs_tables(
    d: dict[str, Any], A: dict[str, list[np.ndarray]], sels: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """ITk-style HS resolution per arm and selection, plus the per-event table."""
    reco = {a: v for a, v in A.items() if a != "truth"}
    tab = M.hs_table(d["truth_z"], d["truth_type"], reco, max_dz=1.0)
    out: dict[str, Any] = {}
    for s, idx in sels.items():
        out[s] = {
            a: {
                f"half{h:g}": M.hs_summary(tab[a], idx, h, n_boot=M_BOOT)
                for h in (1.0, 0.5)
            }
            for a in reco
        }
    return out, tab


def hs_vs_density(tab: dict[str, np.ndarray], idx: np.ndarray, arms: tuple) -> dict:
    """Iterative RMS (1 mm start) in bins of the local PU density."""
    return {
        a: binned_iterative_rms(tab["density"][idx], tab[a][idx], LOCAL_EDGES, 1.0)
        for a in arms
    }


def track_tables(out_dir: Path) -> dict[str, Any]:
    """Median track z0 error in (|eta|, pT) cells for both ntuples."""
    eta_e = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0]
    pt_e = [0.5, 0.9, 1.5, 3.0, 10.0, 1e4]
    out: dict[str, Any] = {"eta_edges": eta_e, "pt_edges": pt_e}
    for tag in ("run2", "hl"):
        t = dict(np.load(out_dir / f"tracks_{tag}.npz"))
        ae = np.abs(t["eta"])
        grid = np.full((len(eta_e) - 1, len(pt_e) - 1), np.nan)
        for i in range(len(eta_e) - 1):
            for j in range(len(pt_e) - 1):
                s = (ae >= eta_e[i]) & (ae < eta_e[i + 1]) & (t["pt"] >= pt_e[j]) & (
                    t["pt"] < pt_e[j + 1])  # fmt: skip
                if s.sum() >= 50:
                    grid[i, j] = 1e3 * float(np.median(t["errz0"][s]))
        out[tag] = {
            "median_errz0_um": grid.tolist(),
            "eta_fraction": np.histogram(ae, bins=eta_e)[0].tolist(),
            "n_tracks": int(len(ae)), "pt_min_gev": float(t["pt"].min()),
            "abs_eta_max": float(ae.max()),
            "median_errz0_um_all": 1e3 * float(np.median(t["errz0"])),
        }  # fmt: skip
    return out


def precision_tables(roots: dict[str, str], n_entries: int = 1000) -> tuple[dict, dict]:
    """Ideal truth-vertex z precision from associated tracks (both ntuples)."""
    nb = [2, 3, 5, 10, 20, 50, 400]
    out, arrays = {"ntrk_edges": nb}, {}
    for tag, root in roots.items():
        p = T.truth_vertex_precision(root, n_entries)
        arrays[tag] = p
        pu = (p["type"] == 0) & (p["ntrk"] >= 2)
        rows = []
        for lo, hi in zip(nb[:-1], nb[1:]):
            s = pu & (p["ntrk"] >= lo) & (p["ntrk"] < hi)
            rows.append(
                1e3 * float(np.median(p["sigma_ideal_mm"][s])) if s.any() else None
            )
        out[tag] = {
            "pu_median_um": 1e3 * float(np.median(p["sigma_ideal_mm"][pu])),
            "pu_mean_ntrk": float(p["ntrk"][pu].mean()),
            "pu_median_ntrk": float(np.median(p["ntrk"][pu])),
            "hs_median_um": 1e3 * float(np.median(p["sigma_ideal_mm"][p["type"] == 1])),
            "pu_median_um_by_ntrk": rows, "n_pu_vertices": int(pu.sum()),
        }  # fmt: skip
    return out, arrays


def target_tables(r2: dict[str, Any], n_events: int = 150) -> tuple[dict, dict]:
    """Measured training-target widths (isolated vertices) vs truth nTracks."""
    import uproot

    hl_t = uproot.open(ROOT_HL_TRAIN)["PVFinderData"].arrays(
        ["TruthVertex_z", "TruthVertex_nTracks"], entry_stop=n_events // 2, library="np"
    )
    meas = {
        # Run 2: the first test events; truth lists are the qibin-mapped ROOT ones.
        "run2": T.measure_target_widths(
            H5_RUN2, [int(e) for e in r2["event_idx"][:n_events]],
            r2["truth_z"][:n_events], r2["truth_ntrk"][:n_events], split=False),
        "hl": T.measure_target_widths(
            H5_HL, list(range(n_events // 2)), list(hl_t["TruthVertex_z"]),
            list(hl_t["TruthVertex_nTracks"]), split=True),
    }  # fmt: skip
    nb = [2, 3, 5, 10, 20, 50, 400]
    out: dict[str, Any] = {"ntrk_edges": nb}
    for tag, m in meas.items():
        out[tag] = [
            1e3 * float(np.median(m["rms_mm"][(m["ntrk"] >= lo) & (m["ntrk"] < hi)]))
            if np.any((m["ntrk"] >= lo) & (m["ntrk"] < hi)) else None
            for lo, hi in zip(nb[:-1], nb[1:])
        ]  # fmt: skip
    mids = np.array([2, 4, 7, 14, 30, 80])
    out["model_hllhc_alleta_um"] = (
        1e3 * T.model_rms(mids, *RESOLUTION_PRESETS["hllhc_alleta"])
    ).tolist()
    out["model_run3_preset_um"] = (
        1e3 * T.model_rms(mids, *RESOLUTION_PRESETS["run3"])
    ).tolist()
    return out, meas


def run_physics(
    r2: dict, hl: dict, sel: dict, A2: dict, AH: dict, H2: dict, HH: dict,
    edges: np.ndarray, out_dir: Path,
) -> dict[str, Any]:  # fmt: skip
    """All non-headline measurements; returns JSON-able results and plot arrays."""
    from pv_finder.diagnostics.run2_vs_run4.analysis import _rho

    res: dict[str, Any] = {}
    arr: dict[str, Any] = {}
    pk, pn = "pvf_int050_h000_full", "pvf_int050_h003_cen"
    rho2, rhoh = _rho(r2), _rho(hl)
    arr["rho"] = {"run2": rho2, "hl": rhoh}
    res["sigma_vs_rho"] = {
        "run2": sigma_in_bins(H2, edges, rho2, RHO_EDGES, (pk, "amvf")),
        "hl": sigma_in_bins(HH, edges, rhoh, RHO_EDGES, (pk, pn, "amvf")),
    }
    res["sigma_vs_mu"] = {
        "run2": sigma_in_bins(H2, edges, r2["mu"], MU_EDGES, (pk, "amvf")),
        "hl": sigma_in_bins(HH, edges, hl["mu"], MU_EDGES, (pk, pn, "amvf")),
    }
    print("[physics] sigma vs rho / mu done", flush=True)
    res["verify_p68_run2_pvf_mm"] = M.verify_p68(A2["truth"], A2[pk])
    s2 = {"run2": sel["run2"]}
    sh = {k: v for k, v in sel.items() if k != "run2"}
    res["p68"] = {**position_tables(A2, s2), **position_tables(AH, sh)}
    print("[physics] p68 done", flush=True)
    hs2, tab2 = hs_tables(r2, A2, s2)
    hsh, tabh = hs_tables(hl, AH, {**sh, "hl_all": np.arange(len(hl["mu"]))})
    res["hs_resolution"] = {**hs2, **hsh}
    res["hs_vs_local_density"] = {
        "run2": hs_vs_density(tab2, sel["run2"], (pk, "amvf")),
        "hl_all": hs_vs_density(tabh, np.arange(len(hl["mu"])), (pk, pn, "amvf")),
        "hl_pu200": hs_vs_density(tabh, sel["hl_pu200"], (pk, pn, "amvf")),
    }
    arr["hs_tab"] = {"run2": tab2, "hl": tabh}
    print("[physics] HS done", flush=True)
    res["tracks"] = track_tables(out_dir)
    res["vertex_precision"], arr["precision"] = precision_tables(
        {"run2": r2["meta"]["root"], "hl": hl["meta"]["root"]}
    )
    res["target_widths"], arr["targets"] = target_tables(r2)
    print("[physics] detector inputs done", flush=True)
    return {"results": res, "arrays": arr}
