#!/usr/bin/env python3
"""Compare the main pipeline (chain_t_metrics) with the independent recomputation.

The independent code (``outputs/.../crosscheck/crosscheck.py``, written
separately on the original dict-based classifier) ran on test_v6 with GNN
v2e527. This prints both and their differences for every shared headline.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_crosscheck_compare.py \
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
from pv_finder.evaluation.vertex_finding.itk_resolution import (
    iterative_rms,  # noqa: E402
)

KEYS = {"clean": "clean", "merged": "merged", "split": "split", "fake": "fake",
        "n_reco": "n_reco", "trk_correct_assign": "n_correct",
        "trk_assigned_labelled": "n_assigned_truth", "pos_matched": "p_truth_found"}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    x = json.loads((a.dir / "crosscheck/crosscheck.json").read_text())
    rec = load(a.dir / "metrics/test_v6_v2e527.npz")
    tg = rec.meta["t_grid"]
    mu = rec.ev("mu")
    pu = (mu >= 185) & (mu <= 215)
    rows, n_bad = [], 0
    for scope, mask in (("all", np.ones(len(mu), bool)), ("pu200", pu)):
        for arm, ai in (
            ("t0", tg.index(0.0)),
            ("t98", tg.index(0.98)),
            ("amvf", len(tg)),
        ):
            for k, f in KEYS.items():
                mine = int(rec.cf(f, 0)[mask, ai].sum())
                other = int(x[scope][arm][k])
                n_bad += mine != other
                rows.append([scope, arm, k, mine, other, mine - other])
    sf = rec.meta["sel_fields"].index("reco_sel")
    stored = int(rec.sel[:, rec.meta["sel_arms"].index("amvf_stored"), sf].sum())
    rule98 = int(rec.sel[pu, rec.meta["sel_arms"].index("t0.98"), sf].sum())
    rows.append(["all", "amvf", "stored_primary_hs", stored, x["all"]["amvf_stored_primary_hs_eff"]["n_success"], 0])  # fmt: skip
    rows.append(["pu200", "t98", "athena_rule_hs", rule98, x["pu200"]["pvf_t98_athena_hs_eff"]["n_success"], 0])  # fmt: skip
    for r in rows[-2:]:
        r[5] = r[3] - r[4]
        n_bad += r[5] != 0
    res_rows = []
    for arm, ai in (("t0", tg.index(0.0)), ("t98", tg.index(0.98))):
        both = pu & np.isfinite(rec.res[:, ai]) & np.isfinite(rec.res[:, -1])
        mine = 1e3 * iterative_rms(rec.res[both, ai], 0.1).rms
        mine_a = 1e3 * iterative_rms(rec.res[both, -1], 0.1).rms
        o = (
            x["pu200"]["hs_resolution"][arm]
            if "hs_resolution" in x["pu200"]
            else x["all"]["hs_resolution"][arm]
        )
        res_rows.append([arm, int(both.sum()), o["n_both"], mine, o["pvf"]["rms_um"], mine_a, o["amvf"]["rms_um"]])  # fmt: skip
    out = {
        "count_rows": rows,
        "n_count_mismatches": int(n_bad),
        "hs_resolution_rows": res_rows,
    }
    (a.dir / "crosscheck_compare.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(*r)
    for r in res_rows:
        print("res", *r)
    print("count mismatches:", n_bad)


if __name__ == "__main__":
    main()
