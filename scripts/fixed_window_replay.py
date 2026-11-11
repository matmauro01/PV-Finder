#!/usr/bin/env python3
"""Replay the note's fake-filter checks at the fixed 0.1694 mm window.

The note's Table tab:fakefilters (height floor 0 -> 0.03 on 384 r16443 events;
GBT score >= 0.3 after the floor on 268 r16638 events) used each sample's
fitted sigma_vtx-vtx (0.228983 / 0.227832 mm) as the window.  The PI decided
that all non-hard-scatter vertex studies use 0.1694 mm, so this replays the
same stored peak lists and events at 0.1694 mm (and at the old windows, as a
reproduction check).  Same selections and inputs as the note's
``scripts/check_hllhc_results.py``; no inference.

    venv/bin/python -u scripts/fixed_window_replay.py \
        --outdir outputs/11_09_2026_output/matching_window_audit
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_finder.diagnostics.amvf_fair_comparison.matching import (  # noqa: E402
    match_vertices,
)
from pv_finder.utils.constants import MATCH_WINDOW_MM  # noqa: E402

PAIRS = {
    # tag: (baseline pkl, variant pkl, event limit on the baseline)
    "r16443": (
        "outputs/08_05_2026_output/eval_v6_mu_window/r16443/eval_results.pkl",
        "outputs/08_20_2026_output/gbt_v6_train_r16443/eval_results.pkl",
        6000,
    ),
    "r16638": (
        "outputs/08_05_2026_output/eval_v6_mu_window/r16638/eval_results.pkl",
        "outputs/08_20_2026_output/v6_r16638_gbt03/eval_results.pkl",
        None,
    ),
}


def summarize(
    data: dict, window: float, limit: int | None, keep: set[int] | None = None
) -> dict[str, float]:
    """Merged-credit eff, one-to-one eff, fake (excl. split) and surplus per event."""
    tot = dict(n=0, truth=0, found=0, primary=0, fake=0, split=0, reco=0)
    for i, ev in enumerate(data["per_event"]):
        if limit is not None and i >= limit:
            break
        if not 185 <= ev["mu"] <= 215:
            continue
        if keep is not None and int(ev["event_idx"]) not in keep:
            continue
        t = np.asarray(data["truth_pvs_mm"][i], dtype=np.float64)
        r = np.asarray(data["pred_pvs_mm"][i], dtype=np.float64)
        m = match_vertices(t, r, window)
        tot["n"] += 1
        tot["truth"] += m.n_truth
        tot["found"] += m.n_truth_matched
        tot["primary"] += m.n_truth_primary
        tot["fake"] += m.reco_fake
        tot["split"] += m.reco_split
        tot["reco"] += m.n_reco
    n, nt = tot["n"], tot["truth"]
    return {
        "n_events": n,
        "n_truth": nt,
        "eff_merged_credit_pct": 100.0 * tot["found"] / nt,
        "eff_one_to_one_pct": 100.0 * tot["primary"] / nt,
        "fake_per_evt": tot["fake"] / n,
        "split_per_evt": tot["split"] / n,
        "surplus_per_evt": (tot["fake"] + tot["split"]) / n,
        "reco_per_evt": tot["reco"] / n,
    }


def main(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report: dict = {"window_mm": MATCH_WINDOW_MM}
    for tag, (base_p, var_p, limit) in PAIRS.items():
        with open(base_p, "rb") as fh:
            base = pickle.load(fh)
        with open(var_p, "rb") as fh:
            var = pickle.load(fh)
        old_w = float(base["sigma_vtx_vtx_mm"])
        # Compare before/after on the same events: the variant may cover fewer.
        keep = {int(e["event_idx"]) for e in var["per_event"]}
        entry = {"baseline": base_p, "variant": var_p, "old_window_mm": old_w}
        for label, w in (("fixed", MATCH_WINDOW_MM), ("old", old_w)):
            entry[label] = {
                "baseline": summarize(base, w, limit, keep),
                "variant": summarize(var, w, None),
            }
        report[tag] = entry
        print(tag, json.dumps(entry, indent=1), flush=True)
    (outdir / "fake_filter_replay.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir", default="outputs/11_09_2026_output/matching_window_audit"
    )
    main(ap.parse_args())
