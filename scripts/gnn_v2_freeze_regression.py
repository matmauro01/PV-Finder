#!/usr/bin/env python3
"""Regression of the new chain code against the 11 Nov 2026 studies (same inputs).

GNN v1 (full-mu e213) on v6 peaks of r16443 entries 0-24,999 was evaluated on
11 Nov by ``hs_chain_study`` (categories, HS composition) and
``hs_selection_identical_rule`` (Athena rule). The new
``chain_t_metrics`` run ``metrics/test_v6_v1e213`` must reproduce them event
by event (t = 0 and 0.98, AMVF), up to GNN scores within ~1e-5 of t.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_regression.py \
        --dir outputs/11_12_2026_output/gnn_v2_freeze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn.evaluation.chain_t_summary import load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OLD_CHAIN = REPO / "outputs/11_11_2026_output/hs_chain_study"
OLD_RULE = REPO / "outputs/11_11_2026_output/hs_selection_identical_rule"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    new = load(a.dir / "metrics/test_v6_v1e213.npz")
    t_grid = new.meta["t_grid"]
    with np.load(OLD_CHAIN / "hs_chain_events.npz") as f:
        oc, ohs, oev = f["counts"], f["hs"], f["events"]
    cmeta = json.loads((OLD_CHAIN / "hs_chain_metadata.json").read_text())
    with np.load(OLD_RULE / "hs_selection_events.npz") as f:
        orule = f["out"]
    rmeta = json.loads((OLD_RULE / "summary.json").read_text())["meta"]
    if not np.array_equal(oev[:, 0], new.ev("entry").astype(int)):
        raise AssertionError("event sets differ")
    res: dict = {}
    pairs = {"t0.98": ("gnn_fullmu_t098", 0.98), "t0": ("gnn_fullmu_t000", 0.0), "amvf": ("amvf", None)}  # fmt: skip
    for key, (old_arm, t) in pairs.items():
        ia = cmeta["arms"].index(old_arm)
        na = len(t_grid) if t is None else t_grid.index(t)
        for m in (0, 1):
            old = oc[:, ia, m, :5]
            nw = np.stack([new.cf(k, m)[:, na] for k in ("clean", "merged", "split", "fake", "n_reco")], 1)  # fmt: skip
            diff = np.any(old != nw, axis=1)
            fields = cmeta["hs_fields"]
            hs_old = ohs[:, ia, m, fields.index("clean_hs")]
            hs_new = new.cf("hs_clean", m)[:, na]
            res[f"{key}_min{m + 1}"] = {
                "events_with_category_difference": int(diff.sum()),
                "sum_old": old.sum(0).tolist(), "sum_new": nw.sum(0).tolist(),
                "clean_hs_differences": int((hs_old != hs_new).sum()),
            }  # fmt: skip
    rf = rmeta["fields"].index("reco_sel")
    for key, old_arm, sel_arm in (
        ("rule_t0.98", "gnn_fullmu_t098_rule_fit", "t0.98"),
        ("rule_t0", "gnn_fullmu_t000_rule_fit", "t0.0"),
        ("amvf_stored", "amvf_stored", "amvf_stored"),
        ("amvf_rule", "amvf_rule", "amvf_rule"),
    ):
        o = orule[:, rmeta["arms"].index(old_arm), rf]
        n = new.sel[:, new.meta["sel_arms"].index(sel_arm), 3]
        res[key] = {"differences": int((o != n).sum()), "old": int(o.sum()), "new": int(n.sum())}  # fmt: skip
    (a.dir / "regression_vs_11nov.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
