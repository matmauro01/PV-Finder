"""ATLAS-style figures for the HS / Fig. 7.2 extension study.

Reads ``hs_chain_summary.json`` (from ``hs_chain_summary.py``) and draws, for
the 25,000 flat-mu ITk validation entries and the 1,920-event PU200 subset:

- hs_reco_eff_vs_mu          (a) HS reconstruction efficiency, 0.1 mm
- hs_clean_fraction_vs_mu    (b) fraction of events with a Clean HS vertex
- hs_track_fraction_vs_mu    (c) HS tracks on the HS-matched vertex
- hs_vertex_purity_vs_mu     (c') HS fraction of the HS-matched vertex tracks
- hs_selection_eff_vs_mu     (d) reconstruction and selection efficiency
- clean_yield_stages_pu200_<window>  Fig. 7.2 with the z0 baseline bars
- clean_yield_stages_vs_mu   Fig. 7.2 stages versus pile-up

Usage:
    PYTHONPATH=src python -u src/gnn/diagnostics/plot_hs_chain.py \\
        --summary outputs/11_09_2026_output/hs_chain_study/hs_chain_summary.json \\
        -o outputs/11_09_2026_output/hs_chain_study/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.plot_style import atlas_label, save_figure, use_atlas_style
from pv_finder.utils.working_points import MATCH_WINDOW_MM, window_tag

W = window_tag(MATCH_WINDOW_MM)  # key suffix of the main matching window

# (label, colour, marker, filled) per series; Okabe-Ito colours, distinct markers
STYLE = {
    "gnn_v4_t095": ("PV-Finder + GNN ($t=0.95$)", "#0072B2", "o", True),
    "gnn_v4_t098": ("PV-Finder + GNN ($t=0.98$)", "#56B4E9", "D", True),
    "gnn_v4_t000": ("PV-Finder + GNN (no score cut)", "#CC79A7", "v", True),
    "z0_nearest": (r"PV-Finder + nearest $z_0$", "#009E73", "^", True),
    "z0_3sigma": (r"PV-Finder + nearest $z_0$ ($<3\sigma$)", "#E69F00", "<", False),
    "gnn_fullmu_t098": (
        r"PV-Finder + GNN, full-$\mu$ training ($t=0.98$)",
        "#000000",
        "o",
        False,
    ),
    "amvf": ("AMVF", "#D55E00", "s", True),
    "amvf_native": ("AMVF, stored primary vertex", "#D55E00", "s", True),
    "amvf_rerank": (r"AMVF, re-ranked by stored track lists", "#D55E00", "s", False),
    "pvf_peaks": ("PV-Finder", "#0072B2", "o", True),
    "truth": ("Truth-guided assignment", "0.35", "P", True),
    "finder": ("One-to-one finder match", "0.6", "X", True),
}
SAMPLE = r"$\sqrt{s}=14$ TeV, $t\bar{t}$, ITk"


def _label(ax: plt.Axes, line2: str) -> None:
    atlas_label(ax, desc=f"{SAMPLE}\n{line2}", desc_xy=(0.05, 0.875))


def _series_frac(bins: list[dict], key: str) -> tuple[np.ndarray, ...]:
    v = np.array([b["fractions"][key]["value"] for b in bins])
    lo = np.array([b["fractions"][key]["low"] for b in bins])
    hi = np.array([b["fractions"][key]["high"] for b in bins])
    return v, v - lo, hi - v


def _series_ratio(bins: list[dict], block: str, arm: str) -> tuple[np.ndarray, ...]:
    v = np.array([b[block][arm]["value"] for b in bins])
    e = np.array([b[block][arm]["error"] for b in bins])
    return v, e, e


def plot_vs_mu(
    bins: list[dict],
    series: list[tuple[str, tuple[np.ndarray, ...]]],
    ylabel: str,
    ylim: tuple[float, float],
    out: Path,
    name: str,
    note: str,
    ratio_ylim: tuple[float, float] = (0.9, 1.1),
) -> None:
    """Points with error bars in mu bins, plus a ratio panel to the last series."""
    edges = np.array([b["mu_low"] for b in bins] + [bins[-1]["mu_high"]])
    x = 0.5 * (edges[1:] + edges[:-1])
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(8, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
    )
    n = len(series)
    ref_v, ref_lo, ref_hi = series[-1][1]
    for k, (key, (v, elo, ehi)) in enumerate(series):
        label, color, marker, filled = STYLE[key]
        shift = (k - (n - 1) / 2) * 1.6  # small offsets so error bars do not overlap
        kw = dict(color=color, marker=marker, ms=7, ls="none", capsize=0, elinewidth=1.3,
                  mfc=color if filled else "white", mew=1.4)  # fmt: skip
        ax.errorbar(
            x + shift,
            v,
            yerr=[elo, ehi],
            label=label,
            **kw,
        )
        if k < n - 1:
            r = v / ref_v
            rerr = r * np.hypot(0.5 * (elo + ehi) / v, 0.5 * (ref_lo + ref_hi) / ref_v)
            rax.errorbar(x + shift, r, yerr=rerr, **kw)
    rax.axhline(1.0, color="0.4", lw=1, ls="--")
    # headroom above the data for the ATLAS label, sample text and legend
    ax.set_ylim(ylim[0], ylim[1] + 0.55 * (ylim[1] - ylim[0]))
    ax.set_ylabel(ylabel)
    rax.set_ylim(*ratio_ylim)
    rax.set_ylabel(f"Ratio to\n{STYLE[series[-1][0]][0]}", fontsize=13)
    rax.set_xlabel(r"Number of interactions $\mu$")
    rax.set_xlim(0, 200)
    ax.legend(loc="upper right", fontsize=11, frameon=False, borderaxespad=0.3)
    _label(ax, note)
    save_figure(fig, out, name)


def plot_stage_bars(pu: dict, window: str, out: Path) -> None:
    """Fig. 7.2 with z0-baseline and AMVF bars (PU200, reco >= 1 track)."""
    w_key = {W: f"{MATCH_WINDOW_MM:g}", "0p5": "0.5"}[window]
    cpt = pu["clean_per_truth"]
    rows = [
        ("Truth vertices ($n_{\\mathrm{trk}}\\geq2$)", 1.0, 0.0, "0.55"),
        (f"One-to-one finder match ({w_key} mm)",
         pu["one_to_one_per_truth"][f"peaks_1to1_{window}"]["value"],
         pu["one_to_one_per_truth"][f"peaks_1to1_{window}"]["error"], "#1f4e79"),
        (f"Truth-guided track assignment ({w_key} mm)", cpt[f"truth_{window}"]["value"],
         cpt[f"truth_{window}"]["error"], "#E69F00"),
        ("GNN track assignment ($t=0.98$)", cpt["gnn_v4_t098"]["value"],
         cpt["gnn_v4_t098"]["error"], "#009E73"),
        ("Nearest-$z_0$ assignment", cpt["z0_nearest"]["value"], cpt["z0_nearest"]["error"], "#56B4E9"),
        ("Nearest-$z_0$ assignment ($<3\\sigma_{z_0}$)", cpt["z0_3sigma"]["value"],
         cpt["z0_3sigma"]["error"], "#A6CEE3"),
        ("AMVF (stored track lists)", cpt["amvf"]["value"], cpt["amvf"]["error"], "#D55E00"),
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(9, 6.6))
    y = np.arange(len(rows))[::-1]
    for yi, (label, v, e, color) in zip(y, rows):
        ax.barh(yi, v, height=0.62, color=color)
        if e:
            ax.errorbar(v, yi, xerr=e, color="k", capsize=3, lw=1.2)
        ax.text(v + 0.012, yi, f"{100 * v:.1f}%", va="center", fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=12)
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.6, len(rows) + 0.2)
    ax.set_xlabel(r"Clean vertices / truth vertices ($n_{\mathrm{trk}}\geq2$)")
    ax.tick_params(axis="y", which="both", left=False, right=False)
    desc = (
        SAMPLE + rf", $\langle\mu\rangle={pu['mean_mu']:.1f}$, {pu['events']:,} events"
    )
    atlas_label(ax, desc=desc, desc_xy=(0.03, 0.955))
    save_figure(fig, out, f"clean_yield_stages_pu200_{window}")


def make_all(summary: dict, out: Path, min_tracks: int = 1) -> None:
    """Draw every figure for one reco-track-count convention."""
    use_atlas_style()
    blk = summary[f"reco_min_tracks_{min_tracks}"]
    bins = blk["mu_bins"]
    sfx = "" if min_tracks == 1 else f"_reco{min_tracks}"
    note = r"$0\leq\mu\leq200$, 25,000 events"
    plot_vs_mu(
        bins,
        [("pvf_peaks", _series_frac(bins, "a_reco_pvf_peaks")),
         ("amvf", _series_frac(bins, "a_reco_amvf"))],
        "HS vertex reco. efficiency", (0.97, 1.005), out,
        "hs_reco_eff_vs_mu" + sfx, note, (0.99, 1.01),
    )  # fmt: skip
    hs_arms = ("gnn_v4_t095", "gnn_fullmu_t098", "z0_nearest", "amvf")
    plot_vs_mu(
        bins, [(a, _series_frac(bins, f"b_clean_hs_{a}")) for a in hs_arms],
        "Events with a Clean HS vertex", (0.65, 1.0), out,
        "hs_clean_fraction_vs_mu" + sfx, note, (0.7, 1.1),
    )  # fmt: skip
    arms_c = ("gnn_v4_t000", "gnn_v4_t095", "gnn_fullmu_t098", "z0_nearest", "amvf")
    plot_vs_mu(
        bins, [(a, _series_ratio(bins, "c_hs_track_fraction", a)) for a in arms_c],
        "HS-track fraction on HS vertex", (0.35, 1.0), out,
        "hs_track_fraction_vs_mu" + sfx, note, (0.4, 1.3),
    )  # fmt: skip
    plot_vs_mu(
        bins, [(a, _series_ratio(bins, "c_hs_vertex_purity", a)) for a in arms_c],
        "HS purity of HS vertex", (0.7, 1.0), out,
        "hs_vertex_purity_vs_mu" + sfx, note, (0.8, 1.1),
    )  # fmt: skip
    sel = [(a, _series_frac(bins, f"d_select_{a}_itk_0p1")) for a in hs_arms[:3]]
    sel.append(("amvf_rerank", _series_frac(bins, "d_select_amvf_itk_0p1")))
    sel.append(("amvf_native", _series_frac(bins, "d_native_amvf_itk_0p1")))
    plot_vs_mu(
        bins, sel, "Vertex sel. and reco. efficiency", (0.6, 1.0), out,
        "hs_selection_eff_vs_mu" + sfx, note, (0.6, 1.1),
    )  # fmt: skip
    stages = [
        ("finder", _series_ratio(bins, "one_to_one_per_truth", f"peaks_1to1_{W}")),
        ("truth", _series_ratio(bins, "clean_per_truth", f"truth_{W}")),
        ("gnn_v4_t098", _series_ratio(bins, "clean_per_truth", "gnn_v4_t098")),
        ("gnn_fullmu_t098", _series_ratio(bins, "clean_per_truth", "gnn_fullmu_t098")),
        ("z0_nearest", _series_ratio(bins, "clean_per_truth", "z0_nearest")),
        ("amvf", _series_ratio(bins, "clean_per_truth", "amvf")),
    ]
    plot_vs_mu(
        bins, stages, "Clean vertices / truth vertices", (0.2, 1.0),
        out, "clean_yield_stages_vs_mu" + sfx, note, (0.5, 3.0),
    )  # fmt: skip
    if min_tracks == 1:  # Fig. 7.2 uses the note's populated-vertex convention
        for window in (W, "0p5"):
            plot_stage_bars(blk["pu200"], window, out)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("-o", "--out", required=True, type=Path)
    args = p.parse_args()
    summary = json.loads(args.summary.read_text())
    for mt in (1, 2):
        make_all(summary, args.out, mt)
    print("figures in", args.out)


if __name__ == "__main__":
    main()
