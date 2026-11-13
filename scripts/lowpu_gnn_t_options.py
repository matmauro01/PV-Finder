#!/usr/bin/env python3
"""Low pile-up GNN score cut t: options, the decision check, and Figs. 9a/9b.

Reads ``gnn.evaluation.lowpu_chain`` outputs (``records.npz``, ``summary.json``)
of the 2,550 Run 2 MC test events. Decision rule (Mattia, 12 Nov 2026): use the
HL-LHC value t = 0.97 unless it is clearly worse all-round at low pile-up,
i.e. unless one other t beats it in Clean/truth AND HS identification AND
track F1, each by more than 2 paired-bootstrap standard errors.

Option rules, as for HL-LHC (docs/evaluation/gnn_v2_freeze_2026-11-12.md):
A max Clean/truth, B max HS identification, C max track F1, D most mu-robust
(largest worst-case ratio, over the mu bins, of Clean/truth to its best value
in that bin), plus t = 0.99 (the July proposal) and t = 0 (no cut).
There is no GNN-independent validation set at low pile-up (the GNN trained on
all 48,450 non-test events), so the rules run on the test events; their
stability is checked on the even and odd halves.

Usage: PYTHONPATH=src python -u scripts/lowpu_gnn_t_options.py \
    --dir outputs/11_12_2026_output/lowpu_final/gnn_mc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from gnn.evaluation.lowpu_chain import MU_BINS, ratios  # noqa: E402
from pv_finder.diagnostics.lowpu_final import plots as P  # noqa: E402

HLLHC_T = 0.97
KEYS = ("clean_per_truth", "hs_id", "trk_f1")
MU_ONLY = [b for b in MU_BINS if b != "inclusive"]
N_BOOT = 1000


def options(recs: dict, tags: list[str], mu: np.ndarray, idx: np.ndarray) -> dict:
    """Option t values derived on the events ``idx``."""
    r = {t: ratios(recs[t][idx]) for t in tags}
    best = {k: max(tags, key=lambda t: r[t][k]) for k in KEYS}
    per_bin = {}
    for b in MU_ONLY:
        lo, hi = MU_BINS[b]
        sel = idx[(mu[idx] >= lo) & (mu[idx] < hi)]
        per_bin[b] = {t: ratios(recs[t][sel])["clean_per_truth"] for t in tags}
    worst = {
        t: min(per_bin[b][t] / max(per_bin[b].values()) for b in MU_ONLY) for t in tags
    }
    return {"A": best["clean_per_truth"], "B": best["hs_id"], "C": best["trk_f1"],
            "D": max(tags, key=lambda t: worst[t])}  # fmt: skip


def paired(recs: dict, a: str, b: str, rng: np.random.Generator, n: int) -> dict:
    """metric(a) - metric(b) with paired event-bootstrap errors."""
    reps = [rng.integers(0, n, n) for _ in range(N_BOOT)]
    out = {}
    for k in ("clean_per_truth", "clean_per_reco", "split_per_reco", "hs_id", "trk_f1"):
        d0 = ratios(recs[a])[k] - ratios(recs[b])[k]
        ds = [ratios(recs[a][i])[k] - ratios(recs[b][i])[k] for i in reps]
        out[k] = {"delta": d0, "err": float(np.std(ds))}
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, type=Path)
    args = ap.parse_args()
    d = np.load(args.dir / "records.npz")
    summ = json.loads((args.dir / "summary.json").read_text())
    tags = [f"t{t:g}" for t in summ["t_grid"]]
    recs = {k: d[k] for k in [*tags, "amvf"]}
    mu = d["mu"]
    n = len(mu)
    every = np.arange(n)
    opt = options(recs, tags, mu, every)
    halves = {"even": options(recs, tags, mu, every[::2]),
              "odd": options(recs, tags, mu, every[1::2])}  # fmt: skip
    rng = np.random.default_rng(20261112)
    chosen = f"t{HLLHC_T:g}"
    dominates = {}
    for t in tags:
        if t == chosen:
            continue
        p = paired(recs, t, chosen, rng, n)
        dominates[t] = {"paired_vs_0.97": p,
                        "beats_0.97_in_all_three_by_2sigma": all(
                            p[k]["delta"] > 2 * p[k]["err"] for k in KEYS)}  # fmt: skip
    clearly_worse = any(
        v["beats_0.97_in_all_three_by_2sigma"] for v in dominates.values()
    )
    table_t = {"E (HL-LHC, chosen)": chosen, **{f"{k}": v for k, v in opt.items()},
               "0.99 (July)": "t0.99", "no cut": "t0"}  # fmt: skip
    bins = summ["bins"]
    rows = []
    for name, t in table_t.items():
        inc = bins["inclusive"][t]
        rows.append({"option": name, "t": t[1:], **{
            k: [inc[k]["value"], inc[k]["err"]] for k in (
                "clean_per_truth", "clean_per_reco", "merged_per_reco", "split_per_reco",
                "fake_per_reco", "trk_f1", "hs_id")},
            "clean_per_truth_by_mu": {b: bins[b][t]["clean_per_truth"]["value"] for b in MU_ONLY},
            "n_reco": inc["totals"]["n_reco"]})  # fmt: skip
    amvf = bins["inclusive"]["amvf"]
    res = {"options_full": opt, "options_halves": halves, "rule": __doc__.split("\n\n")[1],
           "t_hllhc": HLLHC_T, "t_0p97_clearly_worse_all_round": clearly_worse,
           "chosen_t": HLLHC_T if not clearly_worse else None, "rows": rows,
           "amvf": {k: [amvf[k]["value"], amvf[k]["err"]] for k in amvf if k != "totals"},
           "paired_vs_0p97": dominates}  # fmt: skip
    if clearly_worse:
        print("WARNING: t = 0.97 is dominated; choose by hand and document")
    (args.dir / "t_options.json").write_text(json.dumps(res, indent=2))

    md = ["| Option | t | Clean/truth | Clean/reco | Merged/reco | Split/reco | Fake/reco "
          "| track F1 | HS-ID | Clean/truth mu 0-20 / 20-40 / 40-60 / 60-80 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]  # fmt: skip
    for r in rows:
        c = " / ".join(f"{100 * v:.1f}" for v in r["clean_per_truth_by_mu"].values())
        md.append(f"| {r['option']} | {r['t']} | " + " | ".join(
            f"{100 * r[k][0]:.2f} ± {100 * r[k][1]:.2f}" for k in (
                "clean_per_truth", "clean_per_reco", "merged_per_reco", "split_per_reco",
                "fake_per_reco", "trk_f1", "hs_id")) + f" | {c} |")  # fmt: skip
    md.append("| AMVF | - | " + " | ".join(f"{100 * res['amvf'][k][0]:.2f} ± "
              f"{100 * res['amvf'][k][1]:.2f}" for k in ("clean_per_truth", "clean_per_reco",
              "merged_per_reco", "split_per_reco", "fake_per_reco", "trk_f1", "hs_id"))
              + " | |")  # fmt: skip
    (args.dir / "t_options.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("options (full / even / odd):", opt, halves)
    print("t = 0.97 clearly worse all-round:", clearly_worse)

    # Figures
    fig = args.dir / "figures"
    tv = np.array(summ["t_grid"])
    inc = bins["inclusive"]
    curves = {lab: (np.array([inc[t][k]["value"] for t in tags]),
                    np.array([inc[t][k]["err"] for t in tags]), amvf[k]["value"])
              for lab, k in (("Clean / truth", "clean_per_truth"),
                             ("Clean / reco", "clean_per_reco"),
                             ("Track $F_1$", "trk_f1"), ("HS identification", "hs_id"))}  # fmt: skip
    lines = ["Run 2 MC, $t\\bar{t}$, $\\sqrt{s}$ = 13 TeV, 2,550 events",
             "PV-Finder (integral 0.50) + GNN"]  # fmt: skip
    paths = {"t_scan": P.t_scan(tv, curves, HLLHC_T, lines, fig, "gnn_t_scan_run2mc")}
    c = inc[chosen]
    frac = {"pvf_gnn": {k: (c[f"{k}_per_reco"]["value"], c[f"{k}_per_reco"]["err"])
                        for k in ("clean", "merged", "split", "fake")},
            "amvf": {k: (amvf[f"{k}_per_reco"]["value"], amvf[f"{k}_per_reco"]["err"])
                     for k in ("clean", "merged", "split", "fake")}}  # fmt: skip
    lab9 = [lines[0], f"PV-Finder + GNN ($t$ = {HLLHC_T:g}), AMVF",
            f"Vertices with $\\geq$ {summ['provenance'].get('min_tracks', 1)} tracks"]  # fmt: skip
    paths["fig9b"] = P.category_bars(frac, lab9, fig, "fig9b_run2mc_categories")
    matched = {"pvf_gnn": d[f"ntrk_matched_{chosen}"], "amvf": d["ntrk_matched_amvf"]}
    paths["fig9a"] = P.eff_vs_ntrk(d["ntrk_edges"], d["ntrk_matched_truth"], matched,
                                   lab9, fig, "fig9a_run2mc_eff_vs_ntrk")  # fmt: skip
    res["figures"] = {k: P.paths(v) for k, v in paths.items()}
    (args.dir / "t_options.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
