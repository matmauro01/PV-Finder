#!/usr/bin/env python3
"""Figures for the ITk-style hard-scatter resolution study.

Reads ``results.json`` and ``events.npz`` written by
``scripts/itk_hs_resolution.py``; ATLAS style via ``gnn.diagnostics.plot_style``.

    venv/bin/python -u scripts/itk_hs_resolution_plots.py \
        --indir outputs/11_09_2026_output/itk_hs_resolution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from gnn.diagnostics.plot_style import (  # noqa: E402
    ALGO_COLORS,
    atlas_label,
    save_figure,
    use_atlas_style,
)

PVF_C, AMVF_C = ALGO_COLORS["retrained"], ALGO_COLORS["amvf"]
SERIES = (
    ("pvf", "PV-Finder", PVF_C, "o", True),
    ("amvf_pos", "AMVF", AMVF_C, "s", True),
    ("amvf_trk", "AMVF, track-matched HS", AMVF_C, "s", False),
)
SAMPLE = r"$t\bar{t}$, $\sqrt{s}=14$ TeV, ITk"


def _arr(rows: list[dict], key: str) -> np.ndarray:
    return np.array([np.nan if r[key] is None else r[key] for r in rows], dtype=float)


def curve_plot(
    curves: dict, xlabel: str, desc: str, name: str, outdir: Path, xlim: tuple
) -> None:
    """Resolution [um] vs x with a PV-Finder / AMVF ratio panel."""
    fig, (ax, rx) = plt.subplots(
        2, 1, figsize=(8, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ref = curves["amvf_pos"]
    for key, label, color, marker, filled in SERIES:
        rows = curves[key]
        lo, hi = _arr(rows, "lo"), _arr(rows, "hi")
        x = 0.5 * (lo + hi)
        y, e = 1e3 * _arr(rows, "rms"), 1e3 * _arr(rows, "rms_err")
        off = 0.0 if filled else 0.12 * (hi - lo)
        ax.errorbar(
            x + off, y, yerr=e, xerr=0.5 * (hi - lo) if filled else None,
            fmt=marker, color=color, mfc=color if filled else "white",
            label=label, capsize=0, lw=1.5, ms=7,
        )  # fmt: skip
        if key == "pvf":
            ry, re_ = 1e3 * _arr(ref, "rms"), 1e3 * _arr(ref, "rms_err")
            ratio = y / ry
            rerr = ratio * np.hypot(e / y, re_ / ry)
            rx.errorbar(x, ratio, yerr=rerr, xerr=0.5 * (hi - lo), fmt="o",
                        color=PVF_C, lw=1.5, ms=6)  # fmt: skip
    rx.axhline(1.0, color="grey", ls="--", lw=1)
    rx.set_ylabel("PVF / AMVF", fontsize="small")
    rx.set_ylim(0.6, 1.6)
    rx.set_xlabel(xlabel)
    ax.set_ylabel(r"Long. position resolution [$\mu$m]", fontsize="small")
    fig.align_ylabels([ax, rx])
    ymax = np.nanmax([1e3 * _arr(curves[k], "rms") for k, *_ in SERIES])
    ax.set_ylim(0, 1.8 * ymax)
    ax.set_xlim(*xlim)
    atlas_label(ax, desc=desc, desc_xy=(0.05, 0.88))
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    save_figure(fig, outdir, name)


def residual_plot(ev: dict, res: dict, outdir: Path) -> None:
    """Signed HS residuals on the PU200 subset with the final 3-RMS ranges."""
    pu = (ev["mu"] >= 185) & (ev["mu"] <= 215)
    prim = res["pu200_primary"]
    r0 = prim["radius_mm"]
    bins = np.linspace(-1e3 * r0, 1e3 * r0, 51)
    fig, ax = plt.subplots(figsize=(8, 6))
    hmax = 1.0
    for key, label, color, _, _ in SERIES[:2]:
        r = 1e3 * ev[f"{key}@{r0:g}"][pu]
        r = r[np.isfinite(r)]
        h, _ = np.histogram(r, bins)
        hmax = max(hmax, float(h.max()))
        c = 0.5 * (bins[1:] + bins[:-1])
        it = prim[key]
        lab = (f"{label}: iter. RMS {1e3 * it['rms']:.1f} $\\pm$ "
               f"{1e3 * prim['bootstrap'][key]['boot_err']:.1f} $\\mu$m")  # fmt: skip
        ax.errorbar(c, h, yerr=np.sqrt(h), fmt="o" if key == "pvf" else "s",
                    color=color, ms=5, label=lab)  # fmt: skip
        for s in (-1, 1):
            x3 = 1e3 * (it["mean"] + s * 3 * it["rms"])
            ax.plot([x3, x3], [0.5, 2 * hmax], color=color, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 60 * hmax)  # head room for the labels
    ax.set_xlabel(r"$z_{\mathrm{reco}} - z_{\mathrm{HS}}$ [$\mu$m]")
    ax.set_ylabel(f"Events / {bins[1] - bins[0]:.0f} $\\mu$m")
    atlas_label(
        ax,
        desc=f"{SAMPLE}\n$185\\leq\\mu\\leq215$, {int(pu.sum())} events\n"
        f"closest vertex within {1e3 * r0:.0f} $\\mu$m of the HS",
        desc_xy=(0.05, 0.88),
    )
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    save_figure(fig, outdir, "hs_residuals_pu200")


def main(args: argparse.Namespace) -> None:
    indir = Path(args.indir)
    res = json.loads((indir / "results.json").read_text())
    ev = dict(np.load(indir / "events.npz"))
    use_atlas_style()
    cv = res["curves"]
    n_all = res["n_events_all"]
    curve_plot(
        cv["density_all"],
        "Local stored-truth PU density [vertices/mm]",
        f"{SAMPLE}, $0\\leq\\mu\\leq200$\n{n_all} events, HS within 0.1 mm",
        "hs_resolution_vs_density",
        indir,
        (0, 4),
    )
    curve_plot(
        cv["mu_all"],
        r"$\mu$ (ActualNumOfInt)",
        f"{SAMPLE}\n{n_all} events, HS within 0.1 mm",
        "hs_resolution_vs_mu",
        indir,
        (0, 200),
    )
    curve_plot(
        cv["ntrk_pu200"],
        r"HS truth $n_{\mathrm{trk}}$",
        f"{SAMPLE}, $185\\leq\\mu\\leq215$\nHS within 0.1 mm",
        "hs_resolution_vs_ntrk_pu200",
        indir,
        (0, 130),
    )
    residual_plot(ev, res, indir)
    print(f"figures written to {indir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", default="outputs/11_09_2026_output/itk_hs_resolution")
    main(ap.parse_args())
