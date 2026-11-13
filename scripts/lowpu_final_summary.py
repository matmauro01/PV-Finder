#!/usr/bin/env python3
"""Low pile-up final recompute, vertex finding (R1, R3): numbers and figures.

Reads the ``--regime lowpu`` evaluation pickles in ``--base`` (vf_mc, vf_run2,
vf_run3) and the Run 2 MC ROOT truth/AMVF. Also recomputes, with the same code
and window, the April 2026 ep300 pickles (the numbers the note quotes) and the
AMVF anchors of docs/evaluation/data_mc_validation_2026-11-09.md, so every
change can be attributed. No inference.

Usage (repo root, venv active):
  PYTHONPATH=src python -u scripts/lowpu_final_summary.py \
      --base outputs/11_12_2026_output/lowpu_final
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pv_finder.diagnostics.data_mc_validation import plots as dmc  # noqa: E402
from pv_finder.diagnostics.data_mc_validation import reference, samples  # noqa: E402
from pv_finder.diagnostics.lowpu_final import metrics as M  # noqa: E402
from pv_finder.diagnostics.lowpu_final import plots as P  # noqa: E402
from pv_finder.diagnostics.paper_figures import deltaz, style  # noqa: E402
from pv_finder.utils.working_points import get_regime  # noqa: E402

MC_ROOT = REPO / "data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root"
QIBIN = REPO / "configs/qibin_test_main_indices_v2.p"
APRIL = REPO / "outputs/04_16_2026_output/reproduction_ep300_{}/eval_results.pkl"
MC_LINE = "Run 2 MC, $t\\bar{t}$, $\\sqrt{s}$ = 13 TeV"
DATA_LINE = {"run2": "Run 2 data 2018, $\\sqrt{s}$ = 13 TeV",
             "run3": "Run 3 data, $\\sqrt{s}$ = 13.6 TeV"}  # fmt: skip
WP_LINE = "PV-Finder integral threshold 0.50"


def load_root(cache: Path) -> dict[str, list]:
    """All MC ROOT vertex branches (cached as npz)."""
    if not cache.exists():
        import uproot

        a = uproot.open(MC_ROOT)["PVFinderData"].arrays(
            list(M.MC_BRANCHES), library="np"
        )
        pay = {}
        for b in M.MC_BRANCHES[:5]:
            lens = np.array([len(x) for x in a[b]])
            pay[b] = np.concatenate([np.asarray(x, dtype=np.float64) for x in a[b]])
            pay[b + "_off"] = np.concatenate([[0], np.cumsum(lens)])
        pay["ActualNumOfInt"] = np.asarray(a["ActualNumOfInt"], dtype=np.float64)
        pay["BeamPosZ"] = np.array([float(np.ravel(v)[0]) for v in a["BeamPosZ"]])
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **pay)
    d = np.load(cache)
    out: dict[str, list] = {k: d[k] for k in ("ActualNumOfInt", "BeamPosZ")}
    for b in M.MC_BRANCHES[:5]:
        f, o = d[b], d[b + "_off"]
        out[b] = [f[o[i] : o[i + 1]] for i in range(len(o) - 1)]
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pkl_gate(stored: dict, pvf: dict) -> dict:
    """Eval's own float32 matcher vs the float64 recomputation (knife edge, §5.2).

    A reference vertex at |dz| = 0.17 mm to float32 precision can flip; more
    than one flip per sample would indicate a real disagreement.
    """
    found = stored["total_truth_clean"] + stored["total_truth_merged"]
    d = {
        "pkl_found": int(found),
        "recomputed_found": int(pvf["ref_matched"]),
        "pkl_no_ref": int(stored["total_fake"]),
        "recomputed_no_ref": int(pvf["no_ref"]),
    }
    d["ok"] = (
        abs(d["pkl_found"] - d["recomputed_found"]) <= 1
        and abs(d["pkl_no_ref"] - d["recomputed_no_ref"]) <= 1
    )
    if not d["ok"]:
        raise AssertionError(f"eval pkl and recomputation disagree: {d}")
    return d


def amvf_anchors(mc: M.LowPuSample) -> dict:
    """AMVF vs truth at 0.1694 mm and the 60-bin fixed-start sigma (Nov 9 doc)."""
    c = reference.match_events(mc.ref, mc.arms["amvf"].reco, 0.1694)
    s = c.summary()
    fit60 = samples.fit_sigma(mc.arms["amvf"].dz, "fixed", 60)
    return {"eff_0p1694": s["recovery"], "no_ref_per_evt_0p1694": s["no_ref_per_evt"],
            "sigma_60bins_fixed_p0": fit60.sigma, "expected": {
                "eff_0p1694": 0.7567, "no_ref_per_evt_0p1694": 2.50,
                "sigma": 0.789}}  # fmt: skip


def figures(mc: M.LowPuSample, data: dict, res: dict, regime, out: Path) -> dict:
    """Paper Figs. 8 (Run 2 MC) and 10 (Run 2 / Run 3 data, with Run 2 MC)."""
    figs: dict[str, list] = {}
    nb = regime.pairwise_bins
    fits = {k: M.resolution(mc.arms[k].dz, nb) for k in ("pvf", "amvf")}
    lines = [MC_LINE, f"{len(mc.ref):,} events, $0 \\leq \\mu \\leq 80$", WP_LINE]
    figs["fig8a_mc_deltaz"] = deltaz.plot_deltaz(fits, lines, style.SIM_STATUS, out,
                                                 "fig8a_run2mc_deltaz")  # fmt: skip
    cnt = {"pvf": np.array([len(r) for r in mc.arms["pvf"].reco], float),
           "amvf": np.array([len(r) for r in mc.arms["amvf"].reco], float),
           "truth": np.array([len(r) for r in mc.ref], float)}  # fmt: skip
    figs["fig8b_mc_yield"] = P.yields_vs_mu(
        mc.mu, cnt, lines, out, "fig8b_run2mc_yield_vs_mu"
    )
    mc_fit = samples.fit_sigma(mc.arms["pvf"].dz, "adaptive", nb)
    mc_vals = {"pvf": cnt["pvf"], "amvf": cnt["amvf"]}
    tag = {"run2": "a", "run3": "c"}
    for k, s in data.items():
        lab = dmc.Labels(
            data=f"{DATA_LINE[k]}, $\\langle\\mu\\rangle$ = {s.mu.mean():.1f}",
            mc=f"{MC_LINE}, $\\langle\\mu\\rangle$ = {mc.mu.mean():.1f}",
            caveat=WP_LINE + ("\nRun 2 MC shown (no Run 3 MC)" if k == "run3" else ""),
        )
        dfit = samples.fit_sigma(s.arms["pvf"].dz, "adaptive", nb)
        r = dmc.plot_resolution_overlay(
            dfit, mc_fit, lab, out, f"fig10{tag[k]}_{k}_deltaz"
        )
        figs[f"fig10{tag[k]}_{k}_deltaz"] = r["paths"]
        res[k]["data_mc_deltaz_plateau_chi2_ndf"] = r["plateau_chi2_over_ndf"]
        dv = {"pvf": np.array([len(x) for x in s.arms["pvf"].reco], float),
              "amvf": np.array([len(x) for x in s.ref], float)}  # fmt: skip
        nxt = chr(ord(tag[k]) + 1)
        r = dmc.plot_yield_overlay(s.mu, dv, mc.mu, mc_vals, lab, out,
                                   f"fig10{nxt}_{k}_yield_vs_mu")  # fmt: skip
        figs[f"fig10{nxt}_{k}_yield"] = r["paths"]
    return {
        k: P.paths(v) if v and not isinstance(v[0], str) else v for k, v in figs.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, type=Path)
    args = ap.parse_args()
    regime = get_regime("lowpu")
    out = args.base / "summary"
    root = load_root(out / "cache" / "mc_root_vertices_with_type.npz")
    with open(QIBIN, "rb") as fh:
        qibin = [int(x) for x in pickle.load(fh)]

    res: dict = {"regime": regime.to_dict(), "density_checkpoint_sha256":
                 sha256(REPO / regime.density.checkpoint)}  # fmt: skip
    mc_pkl = args.base / "vf_mc" / "eval_results.pkl"
    M.check_regime(M.load_pkl(mc_pkl), regime)
    mc = M.load_mc(mc_pkl, root, qibin)
    res["mc"] = M.sample_metrics(mc, regime)
    res["mc"]["pvf_vs_amvf_reference"] = M.pvf_vs_amvf_reference(
        mc, regime.match_window_mm
    )
    stored = M.load_pkl(mc_pkl)
    res["mc"]["gate_pkl_vs_recomputed"] = pkl_gate(stored, res["mc"]["pvf"])
    res["mc"]["amvf_anchors"] = amvf_anchors(mc)
    data = {}
    for k in ("run2", "run3"):
        pk = args.base / f"vf_{k}" / "eval_results.pkl"
        M.check_regime(M.load_pkl(pk), regime)
        data[k] = M.load_data(k, pk)
        res[k] = M.sample_metrics(data[k], regime)
        res[k]["gate_pkl_vs_recomputed"] = pkl_gate(M.load_pkl(pk), res[k]["pvf"])
    # April ep300 pickles at the same window: what the model + estimator change did.
    legacy = {"mc": M.load_mc(Path(str(APRIL).format("mc")), root, qibin)}
    for k in ("run2", "run3"):
        legacy[k] = M.load_data(k, Path(str(APRIL).format(k)))
    res["april_ep300_at_0p17"] = {}
    for k, s in legacy.items():
        same = [len(a) == len(b) and np.allclose(a, b) for a, b in
                zip(s.ref, (mc if k == "mc" else data[k]).ref)]  # fmt: skip
        r = M.sample_metrics(s, regime)
        r["identical_reference_events"] = int(np.sum(same))
        res["april_ep300_at_0p17"][k] = r
    res["figures"] = figures(mc, data, res, regime, out / "figures")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    res["git_hash"] = git.stdout.strip()
    res["argv"] = sys.argv
    out.mkdir(parents=True, exist_ok=True)
    (out / "vf_results.json").write_text(json.dumps(res, indent=2, default=float))
    for k in ("mc", "run2", "run3"):
        p = res[k]["pvf"]
        print(f"{k:5s} PVF rec={100 * p['recovery']:.2f}+-{100 * p['recovery_err']:.2f} "
              f"fakefrac={100 * p['fake_fraction']['value']:.2f} "
              f"noref/evt={p['no_ref_per_evt']:.3f} sigma={p['sigma_vtx_vtx_mm']['value']:.4f}")  # fmt: skip
    a = res["mc"]["amvf"]
    print(f"MC AMVF eff={100 * a['recovery']:.2f} fake={100 * a['fake_fraction']['value']:.2f} "
          f"sigma={a['sigma_vtx_vtx_mm']['value']:.4f} p68={a['p68_residual_um']:.1f}um "
          f"HS={100 * a['hs_eff']['eff']:.2f}")  # fmt: skip
    p = res["mc"]["pvf"]
    print(f"MC PVF p68={p['p68_residual_um']:.1f}um window/p68={p['window_over_p68']:.2f} "
          f"HS={100 * p['hs_eff']['eff']:.2f}")  # fmt: skip
    print("anchors", res["mc"]["amvf_anchors"])
    print(f"wrote {out / 'vf_results.json'}")


if __name__ == "__main__":
    main()
