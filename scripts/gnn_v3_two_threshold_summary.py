#!/usr/bin/env python3
"""GNN v3: two-threshold working points against AMVF, and the track-efficiency bound.

Input: ``chain_t_metrics --two-threshold`` records (arms: plain t, (t, t_HS,
mode) with mode ``unassign`` or ``fallback``, and the candidate ``oracle``). For
every arm it counts failing (metric, bin) pairs against AMVF exactly as
``gnn_v3_eval_summary.py`` (13 metrics x 22 bins, strict and > 1 paired sigma),
then reports per model:

* the best plain-t arm and the best two-threshold arm of each mode, with their
  failing pairs and the inclusive / PU200 / mu 0-10 rows;
* failure-count grids over (t, t_HS) for each mode;
* the track-efficiency bound per mu bin: labelled tracks of truth vertices with
  < 2 tracks, and of >= 2-track truth vertices that own no s8 candidate (or no
  AMVF vertex) under the one-to-one 0.17 mm matching, as fractions of the
  labelled tracks (the trk_eff denominator); the oracle arm's trk_eff (each
  labelled track on the candidate its truth vertex owns, same metric code).

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_two_threshold_summary.py \
        --metrics c=.../twothr_bc/metrics/c.npz b=.../metrics/b.npz --out .../twothr_bc/summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from gnn_v3_eval_summary import (  # noqa: E402
    BINS,
    METRICS,
    PU,
    failures,
    fmt,
    metric_rows,
)

from gnn.evaluation.chain_t_summary import load, summarize  # noqa: E402

SHOW = ("inclusive", PU, "0-10")


def label(arm: list) -> str:
    """Short arm label."""
    if arm[0] == "t":
        return f"t={arm[1]:g}"
    if arm[0] == "oracle":
        return "oracle"
    return f"t={arm[1]:g}, t_HS={arm[2]:g} ({arm[3]})"


def row(rows: dict, lab: str, j: int) -> list[str]:
    """Cells of one (bin, arm): value with x (worse) / **x** (worse by > 1 sigma)."""
    cells = []
    for key, _, sense, _ in METRICS:
        r = rows[lab]["rows"][key]
        d, e = sense * r["diff"][j], r["diff_err"][j]
        mark = "" if not np.isfinite(d) or d >= 0 else (" **x**" if d < -e else " x")
        cells.append(fmt(key, r["value"][j]) + mark)
    return cells


def amvf_row(rows: dict, lab: str) -> list[str]:
    """AMVF cells of one bin."""
    return [fmt(k, rows[lab]["rows"][k]["amvf"][0]) for k, _, _, _ in METRICS]


def bound_table(rec, rows: dict, arms: list, picks: dict[str, int]) -> list[str]:  # noqa: ANN001
    """Markdown rows of the track-efficiency bound per bin."""
    bf = rec.meta["bound_fields"]
    with np.load(rec.meta["_path"]) as f:
        bound = f["bound"]
    mu, lab_n = rec.ev("mu"), rec.ev("n_labelled")
    j_or = next(k for k, a in enumerate(arms) if a[0] == "oracle")
    head = ["bin", "AMVF trk eff", *[f"{n} trk eff (gap)" for n in picks], "oracle trk eff",
            "tracks of 1-trk truth vtx", "no s8 candidate", "no AMVF vertex", "1 - both s8"]  # fmt: skip
    out = ["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
    for lab, (lo, hi) in BINS.items():
        m = (mu >= lo) & (mu < hi)
        den = lab_n[m].sum()
        f = {k: bound[m, bf.index(k)].sum() / den for k in bf}
        te = rows[lab]["rows"]["trk_eff"]
        cells = [lab, f"{100 * te['amvf'][0]:.1f}"]
        for _, j in picks.items():
            cells.append(f"{100 * te['value'][j]:.1f} ({100 * (te['value'][j] - te['amvf'][0]):+.1f})")  # fmt: skip
        cells += [f"{100 * te['value'][j_or]:.1f}", f"{100 * f['lab_truth_lt2']:.1f}",
                  f"{100 * f['lab_nocand_s8']:.1f}", f"{100 * f['lab_nocand_amvf']:.1f}",
                  f"{100 * (1 - f['lab_truth_lt2'] - f['lab_nocand_s8']):.1f}"]  # fmt: skip
        out.append("| " + " | ".join(cells) + " |")
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", required=True, nargs="+", help="name=metrics.npz")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    md = ["# GNN v3: two-threshold working points vs AMVF (>= 2-track vertices)\n",
          f"Failures: (metric, bin) pairs worse than AMVF, of {len(METRICS) * len(BINS)}; "
          "strict / > 1 paired sigma (200 event replicas).\n"]  # fmt: skip
    res: dict = {}
    heads = [m[1] for m in METRICS]
    for spec in a.metrics:
        name, path = spec.split("=", 1)
        rec = load(path)
        rec.meta["_path"] = path
        arms = rec.meta.get("arms")
        if not arms or rec.meta.get("hs_min_tracks") != 2:
            raise RuntimeError(f"{path}: need --two-threshold and --hs-min-tracks 2")
        rows = metric_rows(rec, summarize(rec, bins=BINS))
        per = []
        for j, arm in enumerate(arms):
            st, sg = failures(rows, j, tuple(BINS))
            per.append({"arm": arm, "label": label(arm), "n_strict": len(st), "n_sig": len(sg),
                        "strict": st, "sig": sg})  # fmt: skip
        key = lambda r: (r["n_strict"], r["n_sig"])  # noqa: E731
        idx = {
            "single t": min(
                (k for k, r in enumerate(per) if r["arm"][0] == "t"),
                key=lambda k: key(per[k]),
            ),  # fmt: skip
            **{
                f"two-t {m}": min(
                    (
                        k
                        for k, r in enumerate(per)
                        if r["arm"][0] == "two" and r["arm"][3] == m
                    ),
                    key=lambda k: key(per[k]),
                )
                for m in ("unassign", "fallback")
            },  # fmt: skip
        }
        res[name] = {
            "metrics": path,
            "per_arm": per,
            "best": {k: per[j]["label"] for k, j in idx.items()},
        }
        md.append(f"\n## {name}\n")
        md.append(
            "| choice | arm | strict | > 1 sigma | failing (metric: bins) |\n|---|---|---:|---:|---|"
        )
        for k, j in idx.items():
            by: dict[str, list[str]] = {}
            for mk, lab in per[j]["strict"]:
                by.setdefault(mk, []).append(lab.split(" ")[0])
            desc = "; ".join(f"{mk}: {', '.join(v)}" if len(v) <= 5 else f"{mk}: {len(v)} bins"
                             for mk, v in by.items())  # fmt: skip
            md.append(f"| {k} | {per[j]['label']} | {per[j]['n_strict']} | {per[j]['n_sig']} | {desc} |")  # fmt: skip
        zero = [r["label"] for r in per if r["n_strict"] == 0]
        md.append(f"\nArms with no failure: {', '.join(zero) or 'none'}. "
                  f"With no failure beyond 1 sigma: "
                  f"{', '.join(r['label'] for r in per if r['n_sig'] == 0) or 'none'}.\n")  # fmt: skip
        t_g = sorted({r["arm"][1] for r in per if r["arm"][0] == "two"})
        th_g = sorted({r["arm"][2] for r in per if r["arm"][0] == "two"})
        for mode in ("unassign", "fallback"):
            md.append(f"\nStrict failures, mode {mode} (rows t, columns t_HS):\n")
            md.append("| t \\ t_HS | " + " | ".join(f"{x:g}" for x in th_g) + " |")
            md.append("|---|" + "---:|" * len(th_g))
            for t in t_g:
                cells = []
                for th in th_g:
                    r = [p for p in per if p["arm"] == ["two", t, th, mode]]
                    cells.append(str(r[0]["n_strict"]) if r else "-")
                md.append(f"| {t:g} | " + " | ".join(cells) + " |")
        picks = {**{k: j for k, j in idx.items()},
                 "oracle": next(k for k, r in enumerate(per) if r["arm"][0] == "oracle")}  # fmt: skip
        for lab in SHOW:
            md.append(f"\n### {name}, {lab} ({rows[lab]['n_events']} events)\n")
            md.append(
                "| arm | " + " | ".join(heads) + " |\n|---|" + "---:|" * len(heads)
            )
            md.append("| AMVF | " + " | ".join(amvf_row(rows, lab)) + " |")
            for k, j in picks.items():
                md.append(
                    f"| {k}: {per[j]['label']} | "
                    + " | ".join(row(rows, lab, j))
                    + " |"
                )
        md.append(
            f"\n### {name}: track efficiency, gap to AMVF and its bound (% of labelled tracks)\n"
        )
        md += bound_table(rec, rows, arms, {k: j for k, j in idx.items()})
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    (a.out / "summary.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
