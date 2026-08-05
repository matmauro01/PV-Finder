"""Re-classify an archived eval at a different sigma_vtx_vtx matching window.

``run_eval_pvf_run3.py`` feeds its own fitted ``sigma_vtx_vtx`` back in as the
matching window for ``compare_res_reco`` (lines ~492, 518, 527, 551).  A deck
that quotes a category table produced at one sigma next to a sigma taken from a
*different* (finer-binned) fit is therefore internally inconsistent: the
efficiency belongs to the wide window, the resolution to the narrow one.

This tool re-runs the classifier over the stored per-event peak and truth
positions at an arbitrary window, so the size of that inconsistency can be
measured rather than argued.  It reproduces the eval exactly when given the
eval's own sigma, which is the check that it is doing the right thing.

Only PV-Finder can be re-derived this way: ``eval_results.pkl`` keeps the
per-event AMVF *category counts* but not the AMVF vertex positions, so an AMVF
row needs the ROOT file and is out of scope here.

Usage::

    PYTHONPATH=src python -u src/pv_finder/diagnostics/deck_window_sensitivity.py \
        --pkl outputs/07_20_2026_output/eval_v4b_baseline_nofloor/eval_results.pkl \
        --windows 0.2851 0.2263 0.2230 \
        --output-dir outputs/08_05_2026_output/deck_claim_audit
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from pv_finder.evaluation.vertex_finding.efficiency_res_optimized_atlas import (
    compare_res_reco,
)
from pv_finder.utils.constants import BIN_WIDTH_MM, Z_MIN


def mm_to_bins(z_mm: np.ndarray) -> np.ndarray:
    """Positions in mm -> bin units, the convention ``compare_res_reco`` expects."""
    return (np.asarray(z_mm, dtype=float) - Z_MIN) / BIN_WIDTH_MM


def classify_at_window(
    pred_mm: list[np.ndarray],
    truth_mm: list[np.ndarray],
    keep: np.ndarray,
    window_mm: float,
) -> dict[str, float]:
    """Category counts and efficiency over the kept events at ``window_mm``."""
    sig_bins = window_mm / BIN_WIDTH_MM
    tot = dict(clean=0, merged=0, split=0, fake=0, tc=0, tm=0, tmiss=0, truth=0)
    n_used = 0
    for i, take in enumerate(keep):
        if not take:
            continue
        t_mm, p_mm = np.asarray(truth_mm[i]), np.asarray(pred_mm[i])
        nt = len(t_mm)
        if nt == 0:
            continue
        n_used += 1
        tot["truth"] += nt
        if len(p_mm) == 0:
            tot["tmiss"] += nt
            continue
        res, tc_arr, _ = compare_res_reco(
            mm_to_bins(t_mm), mm_to_bins(p_mm), sig_bins * np.ones(len(p_mm)), debug=0
        )
        tot["clean"] += res.reco_clean
        tot["merged"] += res.reco_merged
        tot["split"] += res.reco_split
        tot["fake"] += res.reco_fake
        tot["tc"] += int(np.sum(tc_arr == "clean"))
        tot["tm"] += int(np.sum(tc_arr == "merged"))
        tot["tmiss"] += int(np.sum(tc_arr == "missed"))

    n = max(n_used, 1)
    eff = (tot["tc"] + tot["tm"]) / max(tot["truth"], 1)
    return {
        "window_mm": window_mm,
        "n_events": n_used,
        "truth_per_evt": tot["truth"] / n,
        "clean_per_evt": tot["clean"] / n,
        "merged_per_evt": tot["merged"] / n,
        "split_per_evt": tot["split"] / n,
        "fake_per_evt": tot["fake"] / n,
        "efficiency": eff,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", required=True, help="an eval_results.pkl")
    ap.add_argument(
        "--windows",
        type=float,
        nargs="+",
        required=True,
        help="matching windows in mm; the eval's own sigma should be one of them",
    )
    ap.add_argument("--mu-min", type=float, default=None)
    ap.add_argument("--mu-max", type=float, default=None)
    ap.add_argument(
        "--output-dir", default="outputs/08_05_2026_output/deck_claim_audit"
    )
    ap.add_argument("--tag", default=None, help="name for the output json")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    with pkl_path.open("rb") as fh:
        d = pickle.load(fh)

    pred = d["pred_pvs_mm"]
    truth = d["truth_pvs_mm"]
    per_event = d["per_event"]
    n = len(pred)

    # The pickle holds every event the eval READ, not the mu-window subset it
    # summarised.  Select on the stored per-event mu or the numbers will
    # describe a different population from the eval's own table.
    mu = np.array([e.get("mu", np.nan) for e in per_event], dtype=float)
    keep = np.ones(n, dtype=bool)
    if args.mu_min is not None:
        keep &= mu >= args.mu_min
    if args.mu_max is not None:
        keep &= mu <= args.mu_max

    print(f"{pkl_path}")
    print(f"  events in pickle: {n}; selected: {int(keep.sum())}")
    print(f"  stored sigma_vtx_vtx = {d.get('sigma_vtx_vtx_mm'):.4f} mm")
    print(f"  stored overall efficiency = {d.get('overall_efficiency'):.4f}")

    rows = [classify_at_window(pred, truth, keep, w) for w in args.windows]
    print(
        f"\n  {'window':>8} {'eff':>8} {'clean/e':>9} {'merged/e':>9} "
        f"{'split/e':>8} {'fake/e':>8} {'truth/e':>9}"
    )
    for r in rows:
        print(
            f"  {r['window_mm']:8.4f} {r['efficiency']:8.4f} {r['clean_per_evt']:9.2f} "
            f"{r['merged_per_evt']:9.2f} {r['split_per_evt']:8.2f} "
            f"{r['fake_per_evt']:8.2f} {r['truth_per_evt']:9.2f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or pkl_path.parent.name
    payload: dict[str, Any] = {
        "pkl": str(pkl_path),
        "stored_sigma_mm": float(d.get("sigma_vtx_vtx_mm", float("nan"))),
        "stored_overall_efficiency": float(d.get("overall_efficiency", float("nan"))),
        "mu_min": args.mu_min,
        "mu_max": args.mu_max,
        "n_events_selected": int(keep.sum()),
        "rows": rows,
    }
    out_path = out_dir / f"window_sensitivity_{tag}.json"
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
