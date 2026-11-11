#!/usr/bin/env python3
"""Split-inclusive vertex efficiency for PV-Finder and AMVF (reviewer request).

Re-evaluates the note's PU200 comparison (1,920 events, r16443, mu in
[185, 215]) from the saved comparison cache; no inference.  For each window it
reports, with 2,000-replica paired event bootstraps:

* ``eff``                    merged-credit efficiency (note Eq. vertexeff);
* ``eff_incl_split``         ... plus truth inside the window of a split reco;
* ``eff_strict``             one-to-one efficiency;
* ``eff_strict_incl_split``  ... plus truth inside the window of a split reco;
* split, fake and fake+split per event.

Counting rule and the reason the split-inclusive numerators equal the existing
ones are documented in
``efficiency_res_optimized_atlas.truth_found_incl_split``.  The hard-scatter
numbers at 0.1 mm are taken from the hard-scatter study (any reconstructed
vertex counts, so split vertices are already included by definition) and are
re-derived here from the same per-event file as a check.

Run from the repository root:

    venv/bin/python -u scripts/split_inclusive_efficiency.py \
        --outdir outputs/11_09_2026_output/split_inclusive_efficiency
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_finder.diagnostics.amvf_fair_comparison.cache import load_cache  # noqa: E402
from pv_finder.diagnostics.amvf_fair_comparison.scan import (  # noqa: E402
    bootstrap_point,
    per_event_counts,
)
from pv_finder.utils.constants import (  # noqa: E402
    HS_MATCH_WINDOW_MM,
    MATCH_WINDOW_MM,
)

KEYS = (
    "eff",
    "eff_incl_split",
    "eff_strict",
    "eff_strict_incl_split",
    "split_per_evt",
    "fake_per_evt",
    "surplus_per_evt",
)
CACHES = {
    "r16443": "outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz",
    "r16638": "outputs/08_05_2026_output/amvf_fair_comparison/cache_r16638.npz",
}
HS_EVENTS = "outputs/09_17_2026_output/hs_reconstruction/{tag}/events.csv"


def evaluate(cache_path: str, window: float, n_boot: int, seed: int) -> dict:
    """Bootstrap both arms on one cache at one window."""
    data = load_cache(cache_path)
    counts = {
        a: per_event_counts(data["truth_ge2"], data[a], data["truth_eq1"], window)
        for a in ("pvf", "amvf")
    }
    out = bootstrap_point(counts, n_boot=n_boot, seed=seed)
    out["window_mm"] = window
    out["n_events"] = len(data["pvf"])
    out["totals"] = {
        a: {k: int(v.sum()) for k, v in c.items()} for a, c in counts.items()
    }
    return out


def hs_check(tag: str, window: float) -> dict:
    """HS efficiency (any reco vertex within ``window``) on the PU200 subset."""
    path = Path(HS_EVENTS.format(tag=tag))
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    mu = np.array([float(r["mu"]) for r in rows])
    sel = (mu >= 185) & (mu <= 215)
    out = {"source": str(path), "n_events": int(sel.sum()), "window_mm": window}
    for arm in ("pvf", "amvf"):
        dz = np.array([float(r[f"{arm}_dz_mm"]) for r in rows])[sel]
        out[arm] = {"passed": int((dz < window).sum())}
    return out


def fmt(row: dict, arm: str, key: str, scale: float) -> str:
    """Value +- error string."""
    return f"{scale * row[arm][key]:.2f} +- {scale * row[arm][key + '_err']:.2f}"


def main(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    results: dict = {"windows": {}, "hs": {}}
    lines = [
        "# Split-inclusive efficiency, PU200 (merged credit and one-to-one)",
        "",
        "Rule: a truth vertex inside the window of a reco vertex labelled split",
        "also counts as found (once). Errors: paired event bootstrap.",
        "",
    ]
    for tag, path in CACHES.items():
        for w in (MATCH_WINDOW_MM, HS_MATCH_WINDOW_MM, 0.5):
            print(f"[{tag}] window {w} mm", flush=True)
            row = evaluate(path, w, args.n_boot, args.seed)
            results["windows"][f"{tag}@{w:g}"] = row
            lines.append(f"## {tag}, window {w:g} mm, {row['n_events']} events")
            lines.append("")
            lines.append("| quantity | PV-Finder | AMVF | PVF - AMVF |")
            lines.append("|---|---|---|---|")
            for k in KEYS:
                scale = 100.0 if k.startswith("eff") else 1.0
                d = f"{scale * row['diff'][k]:+.2f} +- {scale * row['diff'][k + '_err']:.2f}"
                lines.append(
                    f"| {k} | {fmt(row, 'pvf', k, scale)} | "
                    f"{fmt(row, 'amvf', k, scale)} | {d} |"
                )
            for a in ("pvf", "amvf"):
                t = row["totals"][a]
                lines.append(
                    f"\n{a}: truth {t['n_truth']}, found {t['matched']}, "
                    f"found incl. split {t['matched_incl_split']}, one-to-one "
                    f"{t['primary']}, one-to-one incl. split "
                    f"{t['primary_incl_split']}, split reco {t['split']}"
                )
            lines.append("")
        results["hs"][tag] = hs_check(tag, HS_MATCH_WINDOW_MM)
    (outdir / "results.json").write_text(json.dumps(results, indent=1))
    (outdir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(results["hs"], indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir", default="outputs/11_09_2026_output/split_inclusive_efficiency"
    )
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260805)
    main(p.parse_args())
