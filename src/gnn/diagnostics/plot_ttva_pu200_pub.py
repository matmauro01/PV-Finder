"""Publication plots for the PU200 TTVA full chain.

Produces the HL-LHC counterparts of the mu60 publication set from the
measured JSONs (no re-evaluation):

- pu200_yield_ladder: clean-vertex yield of AMVF, the PVF+GNN chain(s),
  and the three bounds (oracle association, finder cap, GNN on truth).
- pu200_chain_scan: clean/truth and fake rate (drop-empty) vs threshold
  for each provided model scan, with the AMVF reference.
- pu200_miss_ntrk: finder-miss probability vs truth-vertex nTracks.

Usage:
    python -u -m gnn.diagnostics.plot_ttva_pu200_pub \\
        --gap outputs/07_14_2026_ttva_gap/gap_decomposition.json \\
        --scan v1=outputs/07_14_2026_ttva_gap/chain_scan_v1/chain_scan.json \\
        --scan v2=outputs/07_14_2026_ttva_gap/chain_scan_v2/chain_scan.json \\
        --truth-bound 0.9175 -o outputs/<date>_ttva_pu200_publication/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.plot_style import atlas_label, save_figure, use_atlas_style

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
LUMI_DESC = r"HL-LHC $t\bar{t}$, $\langle\mu\rangle=200$, ITk layout"


def _best_point(scan: dict, max_fake: float = 0.01) -> dict:
    """Highest clean/truth at drop-empty fake rate <= max_fake."""
    ok = [r for r in scan["results"] if r["drop_empty"]["fake_rate"] <= max_fake]
    return max(ok, key=lambda r: r["drop_empty"]["clean_per_truth"])


def plot_ladder(gap: dict, scans: dict[str, dict], truth_bound: float, out: Path):
    """Horizontal yield ladder: algorithms and bounds."""
    amvf = None
    rows: list[tuple[str, float, str]] = []
    for label, scan in scans.items():
        best = _best_point(scan)
        rows.append((
            f"PVF + GNN {label} (t={best['t']})",
            best["drop_empty"]["clean_per_truth"],
            OKABE[0],
        ))  # fmt: skip
    # AMVF reference from the gap file context (measured chain_info numbers)
    amvf = gap.get("amvf_clean_per_truth", 0.5729)
    rows.insert(0, ("AMVF", amvf, OKABE[1]))
    cap = gap["finder_cap_per_window"]["0.5"]
    oracle = gap["oracle"]["drop_empty"]["clean_per_truth"]
    bounds = [
        ("Oracle association on peaks", oracle),
        ("Finder cap (0.5 mm)", cap),
        ("GNN on truth vertices", truth_bound),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    for yi, (_label, v, _c) in zip(y, rows):
        ax.text(v + 0.008, yi, f"{v * 100:.1f}%", va="center", fontsize=13)
    for (label, v), ls in zip(bounds, (":", "--", "-.")):
        ax.axvline(v, color="0.35", ls=ls, lw=1.6)
        ax.text(v - 0.008, len(rows) + 0.75, f"{label}  ({v * 100:.1f}%)",
                fontsize=11, color="0.25", ha="right", va="top",
                rotation=90)  # fmt: skip
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.set_xlabel("Clean vertices / truth PVs (nTrk $\\geq$ 2)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, len(rows) + 0.9)
    atlas_label(ax, desc=LUMI_DESC, desc_xy=(0.02, 0.97))
    save_figure(fig, out, "pu200_yield_ladder")
    plt.close(fig)


def plot_scan(scans: dict[str, dict], gap: dict, out: Path):
    """Clean/truth and drop-empty fake rate vs threshold."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7.6), sharex=True, gridspec_kw={"hspace": 0.06}
    )
    amvf = gap.get("amvf_clean_per_truth", 0.5729)
    for (label, scan), color in zip(scans.items(), OKABE):
        t = [r["t"] for r in scan["results"]]
        cpt = [r["drop_empty"]["clean_per_truth"] for r in scan["results"]]
        fr = [r["drop_empty"]["fake_rate"] for r in scan["results"]]
        ax1.plot(t, cpt, "o-", color=color, label=f"PVF + GNN {label}", ms=5)
        ax2.plot(t, fr, "o-", color=color, ms=5)
    ax1.axhline(amvf, color="0.3", ls="--", lw=1.6)
    ax1.text(0.31, amvf + 0.004, f"AMVF  {amvf * 100:.1f}%", fontsize=11, color="0.3")
    ax2.axhline(0.0091, color="0.3", ls="--", lw=1.6)
    ax2.text(0.31, 0.0104, "AMVF fake rate", fontsize=11, color="0.3")
    ax1.set_ylabel("Clean / truth PVs")
    ax2.set_ylabel("Fake rate (drop-empty)")
    ax2.set_yscale("log")
    ax2.set_xlabel("MaxScore threshold $t$")
    ax1.legend(loc="lower right", fontsize=12)
    atlas_label(ax1, desc=LUMI_DESC, desc_xy=(0.03, 0.90))
    save_figure(fig, out, "pu200_chain_scan")
    plt.close(fig)


def plot_miss_ntrk(gap: dict, out: Path):
    """Finder-miss probability vs truth nTracks."""
    edges = np.asarray(gap["ntrk_bin_edges"], dtype=float)
    missed = np.asarray(gap["missed_ntrk_hist"], dtype=float)
    matched = np.asarray(gap["matched_ntrk_hist"], dtype=float)
    total = missed + matched
    prob = np.divide(missed, total, out=np.zeros_like(missed), where=total > 0)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ax.bar(centers, prob, width=widths * 0.92, color=OKABE[0], alpha=0.85)
    err = np.divide(
        np.sqrt(missed * matched / np.maximum(total, 1)),
        np.maximum(total, 1),
    )
    ax.errorbar(centers, prob, yerr=err, fmt="none", ecolor="0.2", capsize=2)
    ax.set_xscale("log")
    ax.set_xlabel("Truth-vertex $n_{\\mathrm{trk}}$")
    ax.set_ylabel("Finder miss probability (no peak within 0.5 mm)")
    ax.set_ylim(0, 1.0)
    atlas_label(ax, desc=LUMI_DESC, desc_xy=(0.35, 0.97))
    save_figure(fig, out, "pu200_miss_ntrk")
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    use_atlas_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.gap) as f:
        gap = json.load(f)
    scans = {}
    for spec in args.scan:
        label, path = spec.split("=", 1)
        with open(path) as f:
            scans[label] = json.load(f)

    plot_ladder(gap, scans, args.truth_bound, out)
    plot_scan(scans, gap, out)
    plot_miss_ntrk(gap, out)
    for label, scan in scans.items():
        best = _best_point(scan)
        print(f"{label}: best t={best['t']} clean/truth "
              f"{best['drop_empty']['clean_per_truth']:.4f} fake "
              f"{best['drop_empty']['fake_rate']:.4f}")  # fmt: skip
    print(f"Saved plots to {out}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--gap", required=True, type=str)
    p.add_argument("--scan", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--truth-bound", default=0.9175, type=float)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    main()
