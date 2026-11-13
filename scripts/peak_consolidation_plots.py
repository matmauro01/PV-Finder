"""Plots and markdown tables for scripts/peak_consolidation.py (13 Nov 2026).

Usage:
    python scripts/peak_consolidation_plots.py --dir outputs/11_13_2026_output/peak_finding
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette (dataviz skill), fixed order.
C = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
INK, MUTED = "#0b0b0b", "#52514e"
PU200 = "PU200 (185-215)"
MU_ORDER = ["0-20", "20-50", "50-100", "100-150", "150-200", PU200]
MU_X = np.arange(len(MU_ORDER))
EST_LABEL = {
    "max_bin": "max bin", "centroid_1": "centroid ±1", "centroid_2": "centroid ±2",
    "centroid_3": "centroid ±3 (current)", "centroid_5": "centroid ±5",
    "full_region": "full-region centroid", "parabola_3pt": "parabola (3 pt)",
    "gauss_3pt": "Gaussian (3 pt, log)", "AMVF": "AMVF",
}  # fmt: skip
FAM_LABEL = {
    "frozen": "current: integral 0.184 + height 0.0184",
    "current_scaled": "integral + height, scaled together",
    "integral_only": "integral only (width 3)",
    "integral_only_w1": "integral only (no width)",
    "height_only": "height only (width 3)",
    "height_only_w1": "height only (no width)",
    "amplitude_only": "amplitude only",
    "amplitude_w3": "amplitude + width 3",
    "height_0p5_literal": "height 0.5 (literal)",
    "AMVF": "AMVF",
}


# One colour per configuration in every figure (colour follows the entity).
FAM_COLOR = {"current_scaled": C[0], "integral_only": C[1], "integral_only_w1": C[2],
             "height_only": C[3], "amplitude_only": C[4], "amplitude_w3": C[5]}  # fmt: skip


def style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)


def plot_estimators(res: dict, s: str, out: Path) -> None:
    ests = [e for e in EST_LABEL if e in res]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    y = np.arange(len(ests))
    v = [res[e][PU200]["p68_um"] for e in ests]
    err = [res[e][PU200].get("d_p68_um_err", 0.0) for e in ests]
    ax.errorbar(v, y, xerr=err, fmt="o", color=C[0], ms=7, capsize=3, lw=1.5)
    ax.set_yticks(y, [EST_LABEL[e] for e in ests])
    ax.invert_yaxis()
    ax.set_xlabel("68% |z_reco - z_true| at PU200 [um]  (error: paired vs centroid ±3)")
    style(ax)
    ax = axes[1]
    show = [
        "max_bin",
        "centroid_1",
        "centroid_3",
        "full_region",
        "parabola_3pt",
        "AMVF",
    ]
    for k, e in enumerate(show):
        ax.plot(MU_X, [res[e][m]["p68_um"] for m in MU_ORDER], "-o", color=C[k], lw=2,
                ms=6, label=EST_LABEL[e])  # fmt: skip
    ax.set_xticks(MU_X, ["0-20", "20-50", "50-100", "100-150", "150-200", "PU200"])
    ax.set_xlabel("mu")
    ax.set_ylabel("68% residual [um]")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.suptitle(f"s8 position estimators at the frozen cuts ({s})", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"estimators_{s}.png", dpi=150)
    plt.close(fig)


def plot_tradeoff(d: dict, s: str, out: Path) -> None:
    cur = d["curves"][s]
    fams = ["current_scaled", "integral_only", "integral_only_w1", "height_only",
            "amplitude_only", "amplitude_w3"]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for k, f in enumerate(fams):
        ax.plot(cur[f]["fp"], 100 * np.asarray(cur[f]["eff"]), "-", color=FAM_COLOR[f], lw=2,
                label=FAM_LABEL[f])  # fmt: skip
    a = cur["AMVF"]
    ax.plot(a["fp"], 100 * a["eff"], "*", color=INK, ms=14, label="AMVF")
    fr = d["cuts"][s]["all"]["frozen"][PU200]  # curves use all PU200 events too
    ax.axvline(fr["fp_per_evt"], color=MUTED, lw=1, ls="--")
    ax.text(fr["fp_per_evt"] + 0.3, 85.5, "current fake rate", color=MUTED, fontsize=8)
    ax.plot(fr["fp_per_evt"], 100 * fr["eff"], "o", color=INK, ms=6, mfc="white",
            label="current cuts")  # fmt: skip
    ax.set_xlim(5, 30)
    ax.set_ylim(76, 86)
    ax.set_xlabel("fakes per event (PU200)")
    ax.set_ylabel("efficiency [%] (truth n_trk >= 2, 0.17 mm)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style(ax)
    ax.set_title(f"Efficiency vs fakes, one knob per curve ({s}, PU200)", color=INK)
    fig.tight_layout()
    fig.savefig(out / f"tradeoff_{s}.png", dpi=150)
    plt.close(fig)


def plot_vs_mu(d: dict, s: str, out: Path) -> None:
    setname = "report" if s == "r16638" else "all"
    r = d["cuts"][s][setname]
    cfgs = [
        "integral_only",
        "integral_only_w1",
        "height_only",
        "amplitude_only",
        "current_scaled",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    xt = ["0-20", "20-50", "50-100", "100-150", "150-200", "PU200"]
    for k, c in enumerate(cfgs):
        de = [r[c][m]["d_eff_pt"] for m in MU_ORDER]
        ee = [r[c][m]["d_eff_pt_err"] for m in MU_ORDER]
        df = [r[c][m]["d_fp"] for m in MU_ORDER]
        fe = [r[c][m]["d_fp_err"] for m in MU_ORDER]
        off = (k - 2) * 0.07
        axes[0].errorbar(MU_X + off, de, yerr=ee, fmt="-o", color=FAM_COLOR[c], ms=5, lw=1.5,
                         capsize=2, label=FAM_LABEL[c])  # fmt: skip
        axes[1].errorbar(
            MU_X + off, df, yerr=fe, fmt="-o", color=C[k], ms=5, lw=1.5, capsize=2
        )
    axes[0].set_ylabel("efficiency minus current [pt]")
    axes[1].set_ylabel("fakes/event minus current")
    axes[2].plot(MU_X, [100 * r["frozen"][m]["eff"] for m in MU_ORDER], "-o", color=MUTED,
                 lw=2, label="PV-Finder s8, current cuts")  # fmt: skip
    axes[2].plot(MU_X, [100 * r["integral_only"][m]["eff"] for m in MU_ORDER], "--s",
                 color=C[1], lw=1.5, label="PV-Finder s8, integral only")  # fmt: skip
    axes[2].plot(MU_X, [100 * r["AMVF"][m]["eff"] for m in MU_ORDER], "-*", color=INK,
                 lw=2, ms=9, label="AMVF")  # fmt: skip
    axes[2].set_ylabel("efficiency [%]")
    for ax in axes:
        ax.set_xticks(MU_X, xt)
        ax.set_xlabel("mu")
        style(ax)
    for ax in axes[:2]:
        ax.axhline(0, color=MUTED, lw=1)
    axes[0].legend(frameon=False, fontsize=7.5)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"One-knob configurations at the PU200-matched fake rate vs mu ({s}, "
                 f"{'odd entries' if s == 'r16638' else 'GNN-validation entries'})",
                 color=INK)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out / f"cuts_vs_mu_{s}.png", dpi=150)
    plt.close(fig)


def fmt(x: float, e: float | None = None, nd: int = 2) -> str:
    return f"{x:+.{nd}f} ± {e:.{nd}f}" if e is not None else f"{x:.{nd}f}"


def tables(d: dict) -> str:
    lines = []
    for s, res in d["estimators"].items():
        lines += [f"\n### Estimators, {s} (all entries)\n",
                  "| Estimator | p68 PU200 [um] | d p68 [um] | p68 incl. [um] | sigma PU200 [um] "
                  "| d sigma [um] | Eff PU200 [%] | dEff [pt] | Fakes/evt PU200 | bias [um] |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
        for e, r in res.items():
            p, i = r[PU200], r["inclusive"]
            lines.append(
                f"| {EST_LABEL[e]} | {p['p68_um']:.2f} | {fmt(p.get('d_p68_um', 0), p.get('d_p68_um_err', 0))} "
                f"| {i['p68_um']:.2f} | {p['sigma_um']:.1f} | {fmt(p.get('d_sigma_um', 0), p.get('d_sigma_um_err', 0), 1)} "
                f"| {100 * p['eff']:.2f} | {fmt(p.get('d_eff_pt', 0), p.get('d_eff_pt_err', 0))} "
                f"| {p['fp_per_evt']:.2f} | {r['bias_um']:+.2f} |"
            )  # fmt: skip
    knobs = d["cuts"]["knobs"]
    for s in d["estimators"]:
        for setname in ("report", "tune") if s == "r16638" else ("all",):
            lines += [f"\n### Cut configurations, {s} {setname}, PU200\n",
                      "| Configuration | Knob | Eff [%] | dEff [pt] | Fakes/evt | dFakes | Splits/evt "
                      "| Truth merged/evt | p68 [um] | d p68 | sigma [um] | d sigma | HS 0.1 mm [%] |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
            for c, by in d["cuts"][s][setname].items():
                p = by[PU200]
                kn = knobs[c]["rounded"] if c in knobs else ""
                lines.append(
                    f"| {FAM_LABEL.get(c, c)} | {kn} | {100 * p['eff']:.2f} "
                    f"| {fmt(p.get('d_eff_pt', 0), p.get('d_eff_pt_err', 0))} | {p['fp_per_evt']:.2f} "
                    f"| {fmt(p.get('d_fp', 0), p.get('d_fp_err', 0))} | {p.get('split_per_evt', 0):.3f} "
                    f"| {p.get('truth_merged_per_evt', 0):.2f} | {p['p68_um']:.2f} "
                    f"| {fmt(p.get('d_p68_um', 0), p.get('d_p68_um_err', 0))} | {p['sigma_um']:.1f} "
                    f"| {fmt(p.get('d_sigma_um', 0), p.get('d_sigma_um_err', 0), 1)} | {100 * p['hs_eff']:.2f} |"
                )  # fmt: skip
    for s in d["estimators"]:
        setname = "report" if s == "r16638" else "all"
        r = d["cuts"][s][setname]
        lines += [f"\n### vs mu, {s} {setname}: Eff [%] / fakes per event\n",
                  "| Configuration | " + " | ".join(MU_ORDER + ["inclusive"]) + " |",
                  "|---|" + "---:|" * (len(MU_ORDER) + 1)]  # fmt: skip
        for c in ("frozen", "integral_only", "integral_only_w1", "height_only",
                  "amplitude_only", "AMVF"):  # fmt: skip
            cells = [f"{100 * r[c][m]['eff']:.2f} / {r[c][m]['fp_per_evt']:.2f}"
                     for m in MU_ORDER + ["inclusive"]]  # fmt: skip
            lines.append(f"| {FAM_LABEL[c]} | " + " | ".join(cells) + " |")
        lines += ["\n dEff [pt] vs current (paired):\n",
                  "| Configuration | " + " | ".join(MU_ORDER + ["inclusive"]) + " |",
                  "|---|" + "---:|" * (len(MU_ORDER) + 1)]  # fmt: skip
        for c in ("integral_only", "integral_only_w1", "height_only", "amplitude_only"):
            cells = [
                fmt(r[c][m]["d_eff_pt"], r[c][m]["d_eff_pt_err"])
                for m in MU_ORDER + ["inclusive"]
            ]
            lines.append(f"| {FAM_LABEL[c]} | " + " | ".join(cells) + " |")
        lines += ["\n dFakes/event vs current (paired):\n",
                  "| Configuration | " + " | ".join(MU_ORDER + ["inclusive"]) + " |",
                  "|---|" + "---:|" * (len(MU_ORDER) + 1)]  # fmt: skip
        for c in ("integral_only", "integral_only_w1", "height_only", "amplitude_only"):
            cells = [
                fmt(r[c][m]["d_fp"], r[c][m]["d_fp_err"])
                for m in MU_ORDER + ["inclusive"]
            ]
            lines.append(f"| {FAM_LABEL[c]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    out = Path(a.dir)
    d = json.loads((out / "consolidation.json").read_text())
    for s, res in d["estimators"].items():
        plot_estimators(res, s, out)
        plot_tradeoff(d, s, out)
        plot_vs_mu(d, s, out)
    (out / "tables.md").write_text(tables(d))
    print("wrote plots and", out / "tables.md")


if __name__ == "__main__":
    main()
