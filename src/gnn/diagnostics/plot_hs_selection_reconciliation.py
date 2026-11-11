"""Reconciliation table and figure for AMVF HS selection efficiency.

Reads the per-event flags of ``hs_selection_reconciliation.py`` and writes
``reconciliation.json`` (counts and 68.27% Clopper-Pearson intervals for every
flag in every selection) and ``reconciliation_table.md`` (the cumulative
step table), plus ``amvf_hs_selection_vs_mu`` (ITk Fig. 27(b) analogue for
AMVF on the ITk mu 0-200 sample and the ID mu 0-80 sample).

Usage:
    PYTHONPATH=src python -u src/gnn/diagnostics/plot_hs_selection_reconciliation.py \\
        --dir outputs/11_09_2026_output/hs_selection_reconciliation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.hs_chain_summary import clopper_pearson
from gnn.diagnostics.plot_style import atlas_label, save_figure, use_atlas_style

# Cumulative definitional changes, starting from the note's 80.7%.
STEPS = (
    (
        "rerank_plurality",
        "Note Sec. 7.5: re-rank stored links by sum(pT^2); dominant truth = HS",
    ),
    ("rerank_mosths", "+ ITk criterion: selected vertex has the most HS tracks"),
    (
        "rerank_mosths_0p1",
        "+ selected vertex within 0.1 mm of the true HS (reco and selected)",
    ),
    ("cap15_mosths_0p1", "  diagnostic: re-rank with track pT capped at 15 GeV"),
    (
        "native_mosths_0p1",
        "+ stored AMVF primary (RecoVertex_type = 1) as selected vertex",
    ),
)
SELECTIONS = {
    "pu200_185_200": ("itk", 185.0, 215.0),
    "pu200_190_200": ("itk", 190.0, 215.0),
    "itk_all_0_200": ("itk", 0.0, 1e9),
    "itk_0_80": ("itk", 0.0, 80.0),
    "id_lowmu_0_80": ("lowmu", 0.0, 1e9),
}
MU_EDGES = {"itk": np.arange(0, 201, 20), "lowmu": np.arange(0, 81, 10)}


def load(directory: Path, sample: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = np.load(directory / f"flags_{sample}.npz")
    names = json.loads((directory / f"flags_{sample}.json").read_text())["flags"]
    return d["flags"], d["mu"], names


def summarize(directory: Path) -> dict:
    """Counts and intervals for every flag and selection."""
    data = {s: load(directory, s) for s in ("itk", "lowmu")}
    out: dict = {}
    for key, (sample, lo, hi) in SELECTIONS.items():
        flags, mu, names = data[sample]
        m = (mu >= lo) & (mu <= hi)
        out[key] = {
            "events": int(m.sum()),
            "mean_mu": float(mu[m].mean()),
            "flags": {
                n: clopper_pearson(int(flags[m, i].sum()), int(m.sum()))
                for i, n in enumerate(names)
            },
        }
    return out


def table(summary: dict, key: str = "pu200_185_200") -> str:
    """Markdown step table: value, interval, change from the previous step."""
    rows = summary[key]["flags"]
    lines = [
        f"Sample `{key}`: {summary[key]['events']} events, <mu> = {summary[key]['mean_mu']:.1f}",
        "",
        "| Step | Definition | Efficiency [%] | 68.27% CP | Change [points] |",
        "|---|---|---:|---:|---:|",
    ]
    prev = None
    for i, (flag, text) in enumerate(STEPS):
        r = rows[flag]
        diag = text.startswith("  ")
        delta = "" if prev is None or diag else f"{100 * (r['value'] - prev):+.2f}"
        lines.append(
            f"| {'-' if diag else i} | {text.strip()} | {100 * r['value']:.2f} | "
            f"{100 * r['low']:.2f}-{100 * r['high']:.2f} | {delta} |"
        )
        if not diag:
            prev = r["value"]
    return "\n".join(lines) + "\n"


def plot(directory: Path, out: Path) -> None:
    """AMVF reconstruction-and-selection efficiency versus mu, both samples."""
    use_atlas_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    style = {
        ("itk", "native_mosths_0p1"): ("ITk, stored primary", "#D55E00", "s", True),
        ("itk", "rerank_mosths_0p1"): (
            "ITk, re-ranked stored links",
            "#D55E00",
            "s",
            False,
        ),
        ("lowmu", "native_mosths_0p1"): ("ID, stored primary", "#0072B2", "o", True),
        ("lowmu", "rerank_mosths_0p1"): (
            "ID, re-ranked stored links",
            "#0072B2",
            "o",
            False,
        ),
    }
    for (sample, flag), (label, color, marker, filled) in style.items():
        flags, mu, names = load(directory, sample)
        edges = MU_EDGES[sample]
        x, v, lo, hi = [], [], [], []
        for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            m = (mu >= a) & ((mu <= b) if j == len(edges) - 2 else (mu < b))
            r = clopper_pearson(int(flags[m, names.index(flag)].sum()), int(m.sum()))
            x.append(0.5 * (a + b))
            v.append(r["value"])
            lo.append(r["value"] - r["low"])
            hi.append(r["high"] - r["value"])
        ax.errorbar(x, v, yerr=[lo, hi], color=color, marker=marker, ls="none", ms=7,
                    mfc=color if filled else "white", mew=1.4, label=label)  # fmt: skip
    ax.set_xlabel(r"Number of interactions $\mu$")
    ax.set_ylabel("Vertex sel. and reco. efficiency")
    ax.set_xlim(0, 200)
    ax.set_ylim(0.7, 1.12)
    ax.legend(loc="lower left", fontsize=12, frameon=False)
    desc = (
        r"$t\bar{t}$, AMVF; ITk $\sqrt{s}=14$ TeV, ID $0\leq\mu\leq80$"
        + "\nMost-HS-track match, $|\\Delta z|<0.1$ mm"
    )
    atlas_label(ax, desc=desc, desc_xy=(0.05, 0.88))
    save_figure(fig, out, "amvf_hs_selection_vs_mu")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", required=True, type=Path)
    args = p.parse_args()
    summary = summarize(args.dir)
    (args.dir / "reconciliation.json").write_text(json.dumps(summary, indent=1) + "\n")
    md = "".join(
        table(summary, k) + "\n"
        for k in ("pu200_185_200", "pu200_190_200", "id_lowmu_0_80")
    )
    (args.dir / "reconciliation_table.md").write_text(md)
    plot(args.dir, args.dir)
    print(md)


if __name__ == "__main__":
    main()
