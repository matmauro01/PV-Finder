"""Ideal-target readout: training targets through the production peak finder.

Is PV-Finder using the HL-LHC track precision fully, or does the 0.04 mm grid
or the target definition limit it?  The training target is what a perfect
network would output; it goes through the same peak finder and the same
metrics as PV-Finder and AMVF, on the events of the Run 2 vs HL-LHC study
(``outputs/11_09_2026_output/run2_vs_run4_resolution``).

Run 2 MC   the stored ``target_y`` of the 2,550 test events (h5 events
           48450..50999, ROOT entries via ``configs/qibin_test_main_indices_v2.p``).
HL-LHC     targets rebuilt from the r16443 ROOT truth for the 1,920 PU200
           events with the v6 converter rule (A, B, C from the v6 training h5),
           as float64, as the float16 the v6 training h5 stores, and on a
           0.02 mm grid.

Stages (repo root, venv active, inside tmux)::

    python -u scripts/target_readout.py build
    python -u scripts/target_readout.py analyze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_finder.diagnostics.amvf_fair_comparison import scan  # noqa: E402
from pv_finder.diagnostics.run2_vs_run4 import analysis, samples  # noqa: E402
from pv_finder.diagnostics.run2_vs_run4 import metrics as M  # noqa: E402
from pv_finder.diagnostics.run2_vs_run4.target_readout import (  # noqa: E402
    NOTE_CONFIG,
    NOTE_W1_CONFIG,
    RUN2_CONFIG,
    build_truth_histogram,
    find_peaks,
)
from pv_finder.evaluation.vertex_finding.itk_resolution import (  # noqa: E402
    hs_residual,
    iterative_rms,
)

STUDY = Path("outputs/11_09_2026_output/run2_vs_run4_resolution")
OUT = Path("outputs/11_11_2026_output/target_readout")
H5_RUN2 = "/share/lazy/qibinlei/recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5"
H5_V6 = "data/run4_all_etas/PU200_alleta_h5/ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.h5"
ROOT_V6 = (
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root"
)
N_BOOT = 200
SEED = 20261111


def v6_abc() -> tuple[float, float, float]:
    """(A, B, C) of the v6 training targets, read from the training h5."""
    a = h5py.File(H5_V6, "r").attrs
    return (
        float(a["resolution_a_mm"]),
        float(a["resolution_b"]),
        float(a["resolution_c_mm"]),
    )


def check_against_training_h5(abc: tuple[float, float, float], n: int = 5) -> float:
    """Rebuild the first stored v6 training targets from ROOT; max |difference|."""
    import uproot

    t = uproot.open(ROOT_V6)["PVFinderData"].arrays(
        ["TruthVertex_z", "TruthVertex_nTracks"], entry_stop=n, library="np"
    )
    worst = 0.0
    with h5py.File(H5_V6, "r") as f:
        for e in range(n):
            mine = build_truth_histogram(
                t["TruthVertex_z"][e], t["TruthVertex_nTracks"][e], abc
            )
            ref = f["target_y_split"][12 * e : 12 * e + 12, :, :].transpose(1, 0, 2)
            ref = ref.reshape(2, -1).astype(np.float64)
            worst = max(worst, float(np.max(np.abs(mine.astype(np.float16) - ref))))
    return worst


def _flat(lists: list[np.ndarray]) -> dict[str, np.ndarray]:
    off = np.cumsum([0] + [len(x) for x in lists])
    v = (
        np.concatenate([np.asarray(x, np.float64) for x in lists])
        if lists
        else np.zeros(0)
    )
    return {"v": v, "off": off}


def _unflat(v: np.ndarray, off: np.ndarray) -> list[np.ndarray]:
    return [v[off[k] : off[k + 1]] for k in range(len(off) - 1)]


def build() -> None:
    """Peak lists of the target readouts, saved as ragged arrays."""
    OUT.mkdir(parents=True, exist_ok=True)
    r2 = samples.load(STUDY / "run2_cache.npz")
    hl = samples.load(STUDY / "hl_cache.npz")
    sel = analysis.selections(r2, hl)
    abc = v6_abc()
    meta: dict[str, Any] = {
        "abc_v6": abc,
        "check_vs_training_h5_max_abs": check_against_training_h5(abc),
    }
    print("[build] v6 target rebuild vs training h5 (float16):", meta, flush=True)

    arms: dict[str, list[np.ndarray]] = {}
    ev = [int(e) for e in r2["event_idx"]]
    if ev != sorted(ev):
        raise ValueError("Run 2 event_idx not sorted")
    with h5py.File(H5_RUN2, "r") as f:
        t = f["target_y"][ev[0] : ev[-1] + 1, 0, :]
    t = t[np.asarray(ev) - ev[0]]
    for name, cfg in (
        ("run2cfg", RUN2_CONFIG),
        ("notecfg", NOTE_CONFIG),
        ("notecfg_w1", NOTE_W1_CONFIG),
    ):
        arms[f"run2_target_{name}"] = [find_peaks(h, cfg) for h in t]
    print("[build] Run 2 done", flush=True)
    del t

    idx = sel["hl_pu200"]
    lists = {
        k: []
        for k in (
            "notecfg",
            "notecfg_f16",
            "run2cfg",
            "notecfg_g002",
            "notecfg_w1",
            "notecfg_w1_g002",
        )
    }
    for n, k in enumerate(idx):
        tz, tn = hl["truth_z"][k], hl["truth_ntrk"][k]
        h = build_truth_histogram(tz, tn, abc)[0]
        lists["notecfg"].append(find_peaks(h, NOTE_CONFIG))
        lists["notecfg_f16"].append(
            find_peaks(h.astype(np.float16).astype(np.float64), NOTE_CONFIG)
        )
        lists["run2cfg"].append(find_peaks(h, RUN2_CONFIG))
        g = build_truth_histogram(tz, tn, abc, 0.02)[0]
        lists["notecfg_g002"].append(find_peaks(g, NOTE_CONFIG, 0.02))
        lists["notecfg_w1"].append(find_peaks(h, NOTE_W1_CONFIG))
        lists["notecfg_w1_g002"].append(find_peaks(g, NOTE_W1_CONFIG, 0.02))
        if n % 400 == 0:
            print(f"[build] HL {n}/{len(idx)}", flush=True)
    for k, v in lists.items():
        arms[f"hl_target_{k}"] = v
    payload: dict[str, np.ndarray] = {"hl_idx": idx}
    for a, v in arms.items():
        fl = _flat(v)
        payload[f"{a}__v"], payload[f"{a}__off"] = fl["v"], fl["off"]
    np.savez_compressed(OUT / "target_peaks.npz", **payload)
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=1))
    print("[build] wrote target_peaks.npz", flush=True)


def _eff(truth2: list, reco: list, truth1: list, idx: np.ndarray, w: float) -> dict:
    c = scan.per_event_counts([truth2[k] for k in idx], [reco[k] for k in idx],
                              [truth1[k] for k in idx], w)  # fmt: skip
    m = scan._metrics(c, np.arange(len(idx)))
    return {
        k: m[k]
        for k in ("eff", "eff_strict", "fake_per_evt", "split_per_evt", "reco_per_evt")
    }


def arm_metrics(d: dict, reco: list, idx: np.ndarray, windows: tuple) -> dict[str, Any]:
    """sigma_vtx-vtx, core p68, HS (1 mm study def and ITk 0.1 mm), eff/FP."""
    t2 = [z[n >= 2] for z, n in zip(d["truth_z"], d["truth_ntrk"])]
    t1 = [z[n == 1] for z, n in zip(d["truth_z"], d["truth_ntrk"])]
    H, edges = M.event_hists(reco)
    out: dict[str, Any] = {"sigma": M.sigma_summary(H, edges, idx, n_boot=N_BOOT)}
    res = [M.matched_residuals_optimal(t2[k], reco[k], 1.0) if len(t2[k]) and len(reco[k])
           else np.zeros(0) for k in range(len(reco))]  # fmt: skip
    out["p68"] = M.p68_summary(res, idx, n_boot=N_BOOT)
    tab = M.hs_table(d["truth_z"], d["truth_type"], {"x": reco}, max_dz=1.0)
    out["hs_study_1mm"] = M.hs_summary(tab["x"], idx, 1.0, n_boot=N_BOOT)
    hsz = tab["hs_z"]
    r01 = np.array([hs_residual(reco[k], float(hsz[k]), 0.1) for k in idx])
    it = iterative_rms(r01, 0.1)
    rng = np.random.default_rng(SEED)
    bs = [
        iterative_rms(r01[rng.integers(0, len(r01), len(r01))], 0.1).rms
        for _ in range(N_BOOT)
    ]
    out["hs_itk_0p1"] = {**it._asdict(), "boot_err": float(np.std(bs, ddof=1)),
                         "hs_reco": float(np.mean(np.isfinite(r01)))}  # fmt: skip
    out["eff"] = {f"{w:g}": _eff(t2, reco, t1, idx, w) for w in windows}
    return out


def analyze() -> None:
    """All metrics for target readouts next to PV-Finder and AMVF."""
    r2 = samples.load(STUDY / "run2_cache.npz")
    hl = samples.load(STUDY / "hl_cache.npz")
    sel = analysis.selections(r2, hl)
    P = np.load(OUT / "target_peaks.npz")
    idx_hl = P["hl_idx"]
    if not np.array_equal(idx_hl, sel["hl_pu200"]):
        raise ValueError("HL selection changed since build")

    def arm(name: str, n: int, at: np.ndarray | None = None) -> list[np.ndarray]:
        v = _unflat(P[f"{name}__v"], P[f"{name}__off"])
        if at is None:
            return v
        full = [np.zeros(0)] * n
        for j, k in enumerate(at):
            full[k] = v[j]
        return full

    amvf2 = [z[n >= 2] for z, n in zip(r2["amvf_z"], r2["amvf_ntrk"])]
    amvfh = [z[n >= 2] for z, n in zip(hl["amvf_z"], hl["amvf_ntrk"])]
    nh = len(hl["mu"])
    R: dict[str, Any] = {"run2": {}, "hl_pu200": {}}
    run2_arms = {
        "target_run2cfg": arm("run2_target_run2cfg", 0),
        "target_notecfg": arm("run2_target_notecfg", 0),
        "target_notecfg_w1": arm("run2_target_notecfg_w1", 0),
        "pvf_run2cfg": r2["pvf_int050_h000_full"],
        "amvf": amvf2,
    }
    for a, reco in run2_arms.items():
        R["run2"][a] = arm_metrics(r2, reco, sel["run2"], (0.416, 0.1694))
        print(f"[analyze] run2 {a} done", flush=True)
    hl_arms = {
        "target_notecfg": arm("hl_target_notecfg", nh, idx_hl),
        "target_notecfg_f16": arm("hl_target_notecfg_f16", nh, idx_hl),
        "target_run2cfg": arm("hl_target_run2cfg", nh, idx_hl),
        "target_notecfg_g002": arm("hl_target_notecfg_g002", nh, idx_hl),
        "target_notecfg_w1": arm("hl_target_notecfg_w1", nh, idx_hl),
        "target_notecfg_w1_g002": arm("hl_target_notecfg_w1_g002", nh, idx_hl),
        "pvf_notecfg": hl["pvf_int030_h003_cen"],
        "pvf_run2cfg": hl["pvf_int050_h000_full"],
        "amvf": amvfh,
    }
    for a, reco in hl_arms.items():
        R["hl_pu200"][a] = arm_metrics(hl, reco, idx_hl, (0.1694,))
        print(f"[analyze] hl {a} done", flush=True)
    vp = json.loads((STUDY / "results.json").read_text())["vertex_precision"]
    R["ideal_precision_um"] = {s: {"pu_median": vp[s]["pu_median_um"],
                                   "hs_median": vp[s]["hs_median_um"]} for s in ("run2", "hl")}  # fmt: skip
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=float))
    print(f"[analyze] wrote {OUT / 'results.json'}", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("build", "analyze"))
    {"build": build, "analyze": analyze}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
