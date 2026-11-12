"""Compare HL-LHC density-model freeze candidates on one held-out file (step 2a).

Reads the saved evaluations of run_eval_pvf_run3.py (``--regime hllhc``,
0.17 mm window, ``--save-histograms``) for every candidate on the same
file, and reports on the PU200 events (185 <= mu <= 215):

- vertex-finding efficiency, the paper's Eq. (1): truth vertices (n_trk >= 2)
  with at least one reconstructed vertex within the window, over all truth
  vertices. Computed independently from positions and asserted equal, event
  by event, to the production matcher's merged-credit numerator;
- fakes per event and the fake fraction (fake / reconstructed);
- hard-scatter reconstruction efficiency at 0.1 mm (TruthVertex_type == 1);
- sigma_vtx-vtx on the same peaks as the counting, and the 68% single-vertex
  residual (optimal one-to-one within 1 mm);
- every candidate at the common HL-LHC cuts and, when a matched-fake-rate
  JSON from scripts/matched_fp_scan.py is given, at its matched cuts.

Differences to the reference come from a paired event bootstrap (the same
resampled events for both arms). AMVF (n_trk >= 2) is reported on the same
events for orientation.

Example (from the repo root):
    python scripts/freeze_candidates_compare.py --tag r16443 \
        --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
        --ref v6 --model v6=outputs/11_12_2026_output/freeze_candidates/v6/r16443 \
        --model s8=outputs/11_12_2026_output/freeze_candidates/s8/r16443 \
        --matched outputs/11_12_2026_output/freeze_candidates/matched_r16443.json \
        --out outputs/11_12_2026_output/freeze_candidates/compare_r16443
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import uproot

ROOT = Path(__file__).resolve().parents[1]
VF = ROOT / "src" / "pv_finder" / "evaluation" / "vertex_finding"
for p in (VF, ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
from efficiency_res_optimized_atlas import compare_res_reco  # noqa: E402
from matched_fp_scan import (  # noqa: E402
    PROD_HEIGHT,
    PROD_INTEGRAL,
    check_replay,
    load,
    peaks,
)

from pv_finder.diagnostics.amvf_fair_comparison.scan import (  # noqa: E402
    fit_sigma_vtx_vtx,
)
from pv_finder.diagnostics.run2_vs_run4.metrics import (  # noqa: E402
    event_hists,
    fit_counts,
    per_event_residuals,
)
from pv_finder.evaluation.vertex_finding.freeze_compare import (  # noqa: E402
    check_bookkeeping,
    clopper_pearson,
    hs_found,
    paired_ratio_diff,
    truth_found_any,
)
from pv_finder.utils.constants import (  # noqa: E402
    HS_MATCH_WINDOW_MM,
    MATCH_WINDOW_MM,
)

Z_MIN, BIN_WIDTH = -240.0, 0.04
N_BOOT, N_BOOT_FIT, SEED = 2000, 200, 20261109


def full_counts(truth: list[np.ndarray], preds: list[np.ndarray]) -> dict:
    """Per-event production-matcher categories plus the independent Eq. (1) count."""
    n = len(truth)
    keys = ("found", "found_any", "truth", "npred", "clean", "merged", "split", "fake")
    out = {k: np.zeros(n) for k in keys}
    w = MATCH_WINDOW_MM / BIN_WIDTH
    for k, (t, p) in enumerate(zip(truth, preds)):
        out["truth"][k], out["npred"][k] = len(t), len(p)
        # Same bin-unit arrays as the matcher, so the identity is exact even
        # for truth vertices on the window edge (float32 rounding).
        tb, pb = (t - Z_MIN) / BIN_WIDTH, (p - Z_MIN) / BIN_WIDTH
        out["found_any"][k] = truth_found_any(tb, pb, w)
        if len(p) == 0 or len(t) == 0:
            out["fake"][k] = len(p)
            continue
        res, tc_arr, *_ = compare_res_reco(
            tb, pb, w * np.ones(len(p)), debug=0, return_reco_cls=True
        )
        out["found"][k] = np.sum(tc_arr == "clean") + np.sum(tc_arr == "merged")
        out["clean"][k], out["merged"][k] = res.reco_clean, res.reco_merged
        out["split"][k], out["fake"][k] = res.reco_split, res.reco_fake
    reco = out["clean"] + out["merged"] + out["split"] + out["fake"]
    if not np.array_equal(reco, out["npred"]):
        raise AssertionError("reco categories do not sum to the number of peaks")
    bad = np.flatnonzero(out["found"] != out["found_any"])
    if len(bad):
        raise AssertionError(f"Eq. (1) != merged credit in {len(bad)} events")
    return out


def read_root(path: str, idx: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """True HS z (exactly one TruthVertex_type == 1) and AMVF z (n_trk >= 2)."""
    br = ["TruthVertex_z", "TruthVertex_type", "RecoVertex_z", "RecoVertex_nTracks"]
    a = uproot.open(path)["PVFinderData"].arrays(
        br, entry_stop=int(idx.max()) + 1, library="np"
    )
    hs_z, amvf = np.zeros(len(idx)), []
    for k, i in enumerate(idx):
        hs = np.flatnonzero(np.asarray(a["TruthVertex_type"][i]) == 1)
        if len(hs) != 1:
            raise ValueError(f"entry {i}: {len(hs)} HS truth vertices")
        hs_z[k] = float(a["TruthVertex_z"][i][hs[0]])
        z, n = np.asarray(a["RecoVertex_z"][i]), np.asarray(a["RecoVertex_nTracks"][i])
        # float32, exactly as run3_io stores AMVF z (matters on the window edge)
        amvf.append(np.asarray(z[n >= 2], dtype=np.float32))
    return hs_z, amvf


def arm(truth: list[np.ndarray], preds: list[np.ndarray], hs_z: np.ndarray) -> dict:
    """Everything one arm (model at one setting, or AMVF) needs for the tables."""
    c = full_counts(truth, preds)
    c["hs"] = np.array([hs_found(p, hs_z[k], HS_MATCH_WINDOW_MM)
                        for k, p in enumerate(preds)], dtype=float)  # fmt: skip
    c["H"], c["edges"] = event_hists(preds)
    c["res"] = per_event_residuals(truth, preds)
    return c


def sigma_of(c: dict, j: np.ndarray) -> float:
    return fit_counts(c["edges"], c["H"][j].sum(0))[0]


def p68_of(c: dict, j: np.ndarray) -> float:
    return 1e3 * float(np.percentile(np.concatenate([c["res"][i] for i in j]), 68.0))


def summarise(c: dict, ref: dict, boots: np.ndarray, fboots: np.ndarray) -> dict:
    """Values and paired differences to the reference arm."""
    n = len(c["truth"])
    allev, ones = np.arange(n), np.ones(n)
    k_hs = int(c["hs"].sum())
    lo, hi = clopper_pearson(k_hs, n)
    out = {
        "eff": c["found"].sum() / c["truth"].sum(),
        "fp_per_evt": c["fake"].mean(),
        "fake_frac": c["fake"].sum() / c["npred"].sum(),
        "split_per_evt": c["split"].mean(),
        "reco_per_evt": c["npred"].mean(),
        "hs_reco_eff": k_hs / n, "hs_reco_cp68": [lo, hi], "hs_reco_n": k_hs,
        "sigma_mm": sigma_of(c, allev),
        "sigma_fit_err_mm": fit_counts(c["edges"], c["H"].sum(0))[1],
        "p68_um": p68_of(c, allev),
    }  # fmt: skip
    s_b = [sigma_of(c, j) for j in fboots]
    p_b = [p68_of(c, j) for j in fboots]
    out["sigma_err_mm"] = float(np.nanstd(s_b, ddof=1))
    out["p68_err_um"] = float(np.std(p_b, ddof=1))
    pairs = {
        "d_eff_pt": (c["found"], c["truth"], ref["found"], ref["truth"], 100.0),
        "d_fp": (c["fake"], ones, ref["fake"], ones, 1.0),
        "d_fake_frac_pt": (c["fake"], c["npred"], ref["fake"], ref["npred"], 100.0),
        "d_hs_pt": (c["hs"], ones, ref["hs"], ones, 100.0),
    }
    for key, (na, da, nb, db, scale) in pairs.items():
        d, e = paired_ratio_diff(na, da, nb, db, boots)
        out[key], out[key + "_err"] = scale * d, scale * e
    ds = [sigma_of(c, j) - sigma_of(ref, j) for j in fboots]
    dp = [p68_of(c, j) - p68_of(ref, j) for j in fboots]
    out["d_sigma_um"] = 1e3 * (out["sigma_mm"] - sigma_of(ref, allev))
    out["d_sigma_um_err"] = 1e3 * float(np.nanstd(ds, ddof=1))
    out["d_p68_um"] = out["p68_um"] - p68_of(ref, allev)
    out["d_p68_um_err"] = float(np.std(dp, ddof=1))
    return {k: float(v) if np.isscalar(v) else v for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref", required=True, help="name of the reference model")
    ap.add_argument("--model", action="append", required=True, help="name=dir")
    ap.add_argument("--matched", help="matched_fp_scan.py JSON for the same file")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--skip-saved-sigma-check", action="store_true",
        help="only for evaluations made before 11 Nov 2026, whose sigma was "
        "fitted on a separate integral-0.50 peak finding",
    )  # fmt: skip
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for spec in args.model:
        name, d = spec.split("=", 1)
        runs[name] = load(Path(d))
        check_replay(name, runs[name])
    names = list(runs)
    idx = runs[args.ref]["idx"]
    for name in names:
        if not np.array_equal(runs[name]["idx"], idx):
            raise ValueError(f"{name}: PU200 events differ from {args.ref}")
        for k, rec in enumerate(runs[name]["saved"]):
            check_bookkeeping(
                rec, len(runs[name]["pvs"][k]), len(runs[name]["truth"][k])
            )
    truth = runs[args.ref]["truth"]
    for name in names:
        if any(not np.array_equal(a, b) for a, b in zip(runs[name]["truth"], truth)):
            raise ValueError(f"{name}: truth vertices differ from {args.ref}")
    print(f"[{args.tag}] {len(idx)} identical PU200 events in {names}; bookkeeping ok")

    hs_z, amvf = read_root(args.root, idx)
    matched = (
        json.loads(Path(args.matched).read_text())["models"] if args.matched else {}
    )
    rng = np.random.default_rng(SEED)
    n = len(idx)
    boots = rng.integers(0, n, size=(N_BOOT, n))
    fboots = rng.integers(0, n, size=(N_BOOT_FIT, n))

    arms: dict[str, dict] = {}
    settings: dict[str, dict] = {}
    for name in names:
        arms[f"{name}@common"] = arm(truth, runs[name]["pvs"], hs_z)
        settings[f"{name}@common"] = {
            "integral": PROD_INTEGRAL,
            "min_height": PROD_HEIGHT,
        }
        if name in matched:
            m = matched[name]["matched_to_ref_fp"]
            pv = [
                peaks(runs[name], k, m["integral"], m["min_height"]) for k in range(n)
            ]
            arms[f"{name}@matched"] = arm(truth, pv, hs_z)
            settings[f"{name}@matched"] = {"integral": m["integral"],
                                           "min_height": m["min_height"]}  # fmt: skip
    arms["AMVF"] = arm(truth, amvf, hs_z)
    settings["AMVF"] = {"vertices": "RecoVertex with nTracks >= 2"}

    # Cross-checks against what run_eval_pvf_run3.py saved. sigma agrees to
    # float summation order (~1e-9 mm), hence a relative tolerance of 1e-6.
    ref_c = arms[f"{args.ref}@common"]
    saved_sigma = runs[args.ref]["sigma"]
    s_chk = fit_sigma_vtx_vtx(runs[args.ref]["pvs"])[0]
    if not args.skip_saved_sigma_check and not np.isclose(
        s_chk, saved_sigma, rtol=1e-6, atol=0
    ):
        raise AssertionError(f"sigma {s_chk} != saved {saved_sigma}")
    amvf_saved = np.array([r["amvf_t_found"] for r in runs[args.ref]["saved"]], float)
    if not np.array_equal(arms["AMVF"]["found"], amvf_saved):
        raise AssertionError("AMVF found count differs from the saved evaluation")
    print(f"[{args.tag}] sigma and AMVF counts reproduce the saved evaluation")

    result = {"tag": args.tag, "ref": f"{args.ref}@common", "n_events": n,
              "window_mm": MATCH_WINDOW_MM, "hs_window_mm": HS_MATCH_WINDOW_MM,
              "n_boot": N_BOOT, "n_boot_fit": N_BOOT_FIT, "seed": SEED,
              "event_idx_sha1": hashlib.sha1(idx.astype(np.int64).tobytes()).hexdigest(), "arms": {}}  # fmt: skip
    for key, c in arms.items():
        result["arms"][key] = {**summarise(c, ref_c, boots, fboots), **settings[key]}
        r = result["arms"][key]
        print(f"  {key:14s} eff {100 * r['eff']:.2f} ({r['d_eff_pt']:+.2f}±"
              f"{r['d_eff_pt_err']:.2f})  FP {r['fp_per_evt']:.2f} ({r['d_fp']:+.2f})  "
              f"HS {100 * r['hs_reco_eff']:.2f}  sigma {r['sigma_mm']:.4f} "
              f"({r['d_sigma_um']:+.1f}±{r['d_sigma_um_err']:.1f} um)  "
              f"p68 {r['p68_um']:.2f} ({r['d_p68_um']:+.2f}±{r['d_p68_um_err']:.2f})")  # fmt: skip
    (out_dir / "compare.json").write_text(json.dumps(result, indent=1))
    np.savez_compressed(
        out_dir / "pair_hists.npz", edges=ref_c["edges"],
        **{k.replace("@", "_"): c["H"].sum(0) for k, c in arms.items()},
    )  # fmt: skip


if __name__ == "__main__":
    main()
