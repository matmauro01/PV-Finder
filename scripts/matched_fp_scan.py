"""Compare PV-Finder models at a matched fake rate on held-out PU200.

Re-runs only peak finding and matching on saved evaluations written by
run_eval_pvf_run3.py (no inference), with the production matcher and the
fixed 0.1694 mm window, on the events with 185 <= mu <= 215.

- A model whose eval was run with --save-histograms can have its integral
  threshold (and height floor) lowered or raised.
- A model without histograms can only have its height floor raised; that
  cut is replayed exactly on the saved peak heights.

Each run's replay at its saved settings is checked event by event against
the stored per-event counts before any number is reported.

Example:
    python scripts/matched_fp_scan.py \
        --ref v6=outputs/11_10_2026_output/small_model_eval/v6/r16443 \
        --model s8=outputs/11_11_2026_output/small_model_matched_fp/s8/r16443 \
        --out outputs/11_11_2026_output/small_model_matched_fp/r16443.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VF = ROOT / "src" / "pv_finder" / "evaluation" / "vertex_finding"
sys.path.insert(0, str(VF))
sys.path.insert(0, str(ROOT / "src"))
from efficiency_res_optimized_atlas import (  # noqa: E402
    compare_res_reco,
    pv_locations_updated_res,
)

from pv_finder.evaluation.vertex_finding.operating_point import (  # noqa: E402
    bisect_to_target,
    filter_by_height,
)
from pv_finder.utils.constants import MATCH_WINDOW_MM  # noqa: E402

Z_MIN, BIN_WIDTH = -240.0, 0.04
PEAK_THRESHOLD, MIN_WIDTH, CENTROID_HW = 0.01, 3, 3
PROD_INTEGRAL, PROD_HEIGHT = 0.30, 0.03
N_BOOT, SEED = 2000, 20261109


def load(path: Path) -> dict:
    """In-window events of one saved eval: truth, peaks, histograms, records."""
    with open(path / "eval_results.pkl", "rb") as f:
        d = pickle.load(f)
    if d["match_window_mm"] != MATCH_WINDOW_MM or d["smooth_sigma"] != 0.0:
        raise ValueError(f"{path}: not the production methodology")
    if d["nms_min_sep"] != 0.0 or d["centroid_halfwidth"] != CENTROID_HW:
        raise ValueError(f"{path}: not the production methodology")
    win = np.asarray(d["in_mu_window"], dtype=bool)
    # per_event skips events without truth; align it with the event list.
    with_truth = [i for i, t in enumerate(d["truth_pvs_mm"]) if len(t) > 0]
    if len(with_truth) != len(d["per_event"]):
        raise ValueError(f"{path}: per_event does not align with events")
    rec = dict(zip(with_truth, d["per_event"]))
    sel = [i for i in range(len(win)) if win[i]]
    hists = d.get("histograms")
    return {
        "idx": np.array([rec[i]["event_idx"] for i in sel]),
        "truth": [np.asarray(d["truth_pvs_mm"][i]) for i in sel],
        "pvs": [np.asarray(d["pred_pvs_mm"][i]) for i in sel],
        "hts": [np.asarray(d["pred_heights"][i]) for i in sel],
        "hist": [np.asarray(hists[i]) for i in sel] if hists else None,
        "saved": [rec[i] for i in sel],
        "sigma": float(d["sigma_vtx_vtx_mm"]),
        "sigma_err": float(d["sigma_vtx_vtx_err_mm"]),
    }


def peaks(run: dict, k: int, integral: float, height: float) -> np.ndarray:
    """Peak positions for event k at the given cuts."""
    if run["hist"] is not None:
        p, *_ = pv_locations_updated_res(
            run["hist"][k], PEAK_THRESHOLD, integral, MIN_WIDTH, height, CENTROID_HW
        )
        return np.asarray(p)
    if integral != PROD_INTEGRAL or height < PROD_HEIGHT:
        raise ValueError("without histograms only a higher height floor works")
    return filter_by_height(run["pvs"][k], run["hts"][k], height)


def counts(run: dict, integral: float, height: float) -> dict:
    """Per-event found / truth / fake / split arrays (production matcher)."""
    w = MATCH_WINDOW_MM / BIN_WIDTH
    out = {
        key: np.zeros(len(run["truth"])) for key in ("found", "truth", "fake", "split")
    }
    for k, t in enumerate(run["truth"]):
        p = peaks(run, k, integral, height)
        out["truth"][k] = len(t)
        if len(p) == 0:
            continue
        res, tc_arr, *_ = compare_res_reco(
            (t - Z_MIN) / BIN_WIDTH, (p - Z_MIN) / BIN_WIDTH, w * np.ones(len(p)),
            debug=0, return_reco_cls=True,
        )  # fmt: skip
        out["found"][k] = np.sum(tc_arr == "clean") + np.sum(tc_arr == "merged")
        out["fake"][k] = res.reco_fake
        out["split"][k] = res.reco_split
    return out


def check_replay(name: str, run: dict) -> dict:
    """Replay the saved settings and require exact agreement per event."""
    c = counts(run, PROD_INTEGRAL, PROD_HEIGHT)
    saved = run["saved"]
    for key, sk in (("fake", "fake"), ("split", "split")):
        ref = np.array([e[sk] for e in saved], dtype=float)
        if not np.array_equal(c[key], ref):
            raise RuntimeError(f"{name}: replay {key} differs from saved eval")
    ref = np.array([e["tc"] + e["tm"] for e in saved], dtype=float)
    if not np.array_equal(c["found"], ref):
        raise RuntimeError(f"{name}: replay efficiency differs from saved eval")
    print(f"  {name}: replay at saved settings matches all {len(saved)} events")
    return c


def summary(c: dict, ref: dict, boots: np.ndarray) -> dict:
    """Efficiency and fakes/event, with paired-bootstrap differences to ref."""

    def m(x: dict, s: np.ndarray) -> np.ndarray:
        return np.array(
            [x["found"][s].sum() / x["truth"][s].sum(), x["fake"][s].mean()]
        )

    allev = np.arange(len(c["truth"]))
    val, dv = m(c, allev), m(c, allev) - m(ref, allev)
    err = np.array([m(c, b) - m(ref, b) for b in boots]).std(axis=0)
    return {
        "eff": val[0], "fp_per_evt": val[1],
        "d_eff_pt": 100 * dv[0], "d_eff_pt_err": 100 * err[0],
        "d_fp": dv[1], "d_fp_err": err[1],
    }  # fmt: skip


def fp_mean(run: dict, integral: float, height: float) -> float:
    return float(counts(run, integral, height)["fake"].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="name=dir of the reference eval")
    ap.add_argument("--model", action="append", required=True, help="name=dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rname, rdir = args.ref.split("=", 1)
    ref = load(Path(rdir))
    ref_c = check_replay(rname, ref)
    ref_fp = float(ref_c["fake"].mean())
    rng = np.random.default_rng(SEED)
    n = len(ref["idx"])
    boots = rng.integers(0, n, size=(N_BOOT, n))
    result: dict = {"ref": rname, "ref_fp": ref_fp, "window_mm": MATCH_WINDOW_MM,
                    "n_events": n, "models": {}}  # fmt: skip

    for spec in args.model:
        name, mdir = spec.split("=", 1)
        run = load(Path(mdir))
        if not np.array_equal(run["idx"], ref["idx"]):
            raise ValueError(f"{name}: events differ from {rname}")
        if run["hist"] is None:
            raise ValueError(f"{name}: needs --save-histograms to lower cuts")
        base_c = check_replay(name, run)
        base = summary(base_c, ref_c, boots)
        # Scale both peak cuts (integral 0.30 k, height 0.03 k) by one factor k
        # until fakes match the reference; one knob keeps the scan monotone.
        k, _ = bisect_to_target(
            lambda x, r=run: fp_mean(r, PROD_INTEGRAL * x, PROD_HEIGHT * x),
            ref_fp,
            0.05,
            1.0,
        )
        thr, hgt = PROD_INTEGRAL * k, PROD_HEIGHT * k
        matched = summary(counts(run, thr, hgt), ref_c, boots)
        # Reverse: raise the reference's height floor to this model's fakes.
        base_fp = float(base_c["fake"].mean())
        h, _ = bisect_to_target(
            lambda x: fp_mean(ref, PROD_INTEGRAL, x), base_fp, PROD_HEIGHT, 1.0
        )
        rev = summary(counts(ref, PROD_INTEGRAL, h), base_c, boots)
        result["models"][name] = {
            "production": {**base, "integral": PROD_INTEGRAL, "min_height": PROD_HEIGHT},
            "matched_to_ref_fp": {**matched, "scale": k, "integral": thr, "min_height": hgt},
            "ref_raised_to_model_fp": {**rev, "integral": PROD_INTEGRAL, "min_height": h,
                                       "note": "ref minus model, at model's fake rate"},
            "sigma_mm": run["sigma"], "sigma_err_mm": run["sigma_err"],
            "ref_sigma_mm": ref["sigma"], "ref_sigma_err_mm": ref["sigma_err"],
        }  # fmt: skip
        print(f"  {name}: production dEff {base['d_eff_pt']:+.2f}±{base['d_eff_pt_err']:.2f} "
              f"pt, FP {base['fp_per_evt']:.2f}")  # fmt: skip
        print(f"  {name}: at {rname} FP ({ref_fp:.2f}) integral={thr:.4f} height={hgt:.4f}: "
              f"FP {matched['fp_per_evt']:.2f}, dEff "
              f"{matched['d_eff_pt']:+.2f}±{matched['d_eff_pt_err']:.2f} pt")  # fmt: skip
        print(f"  {rname} raised to {name} FP ({base_fp:.2f}) min_height={h:.4f}: "
              f"{rname}-{name} dEff {rev['d_eff_pt']:+.2f}±{rev['d_eff_pt_err']:.2f} pt")  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
