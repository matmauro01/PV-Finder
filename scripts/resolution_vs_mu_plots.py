"""Plots for scripts/resolution_vs_mu.py: per-bin fit PDF and the summary figure."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from gnn.diagnostics.plot_style import (  # noqa: E402
    ALGO_COLORS,
    atlas_label,
    save_figure,
    use_atlas_style,
)
from pv_finder.diagnostics.amvf_fair_comparison.scan import sigmoid_fit  # noqa: E402

DESC = r"$\sqrt{s}$ = 14 TeV, $t\bar{t}$, ITk, $0 \leq \mu \leq 200$"
NAMES = {"pvf": "PV-Finder", "amvf": "AMVF"}
COLORS = {"pvf": ALGO_COLORS["pvf_gnn"], "amvf": ALGO_COLORS["amvf"]}


def plot_fits(
    hists: dict, k: np.ndarray, rows: list[dict[str, Any]], path: Path
) -> None:
    """One page per mu bin: pairwise Delta z counts and the sigmoid, both algorithms."""
    use_atlas_style()
    e = hists["edges"]
    c = 0.5 * (e[:-1] + e[1:])
    with PdfPages(path) as pdf:
        for b, r in enumerate(rows):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
            for ax, n in zip(axes, ("pvf", "amvf")):
                cnt = hists[n][k == b].sum(0)
                ax.errorbar(c, cnt, yerr=np.sqrt(np.maximum(cnt, 1)), fmt=".", ms=3,
                            color=COLORS[n], label=NAMES[n])  # fmt: skip
                p = r[n]["popt"]
                if np.all(np.isfinite(p)):
                    ax.plot(c, sigmoid_fit(c, *p), "k-", lw=1.2,
                            label=rf"$\sigma$ = {1e3 * r[n]['sigma_mm']:.0f} $\pm$ "
                                  rf"{1e3 * r[n]['boot_err_mm']:.0f} $\mu$m")  # fmt: skip
                ax.set_xlabel(r"$\Delta z$ [mm]")
                ax.set_ylabel("Vertex pairs / 0.04 mm")
                ax.set_title(f"{NAMES[n]}, {r['mu_lo']:.0f} $\\leq \\mu$ < {r['mu_hi']:.0f}, "
                             f"{r['n_events']} events, $\\chi^2$/ndf = "
                             f"{r[n]['chi2']:.0f}/{r[n]['ndf']}", fontsize=11)  # fmt: skip
                ax.legend(fontsize=10, loc="lower right")
            pdf.savefig(fig)
            plt.close(fig)


def plot_summary(rows: list[dict[str, Any]], out: Path, label: str) -> None:
    """sigma_vtx-vtx vs mu (top) and the 68% single-vertex residual vs mu (bottom).

    sigma_vtx-vtx is drawn with filled markers only where the dip dominates the
    pairwise distribution (merge length >= 0); elsewhere the correlated excess
    of close reconstructed pairs exceeds the dip and the marker is open.
    """
    use_atlas_style()
    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05},
    )  # fmt: skip
    lo = np.array([r["mu_lo"] for r in rows])
    hi = np.array([r["mu_hi"] for r in rows])
    x = np.array([r["mu_mean"] for r in rows])
    for n in ("pvf", "amvf"):
        mk = "o" if n == "pvf" else "s"
        s = np.array([1e3 * r[n]["sigma_mm"] for r in rows])
        se = np.array([1e3 * r[n]["boot_err_mm"] for r in rows])
        ok = np.array([r[n]["merge_len_mm"] >= 0 for r in rows])
        xe = np.array([x - lo, hi - x])
        a1.errorbar(x[ok], s[ok], xerr=xe[:, ok], yerr=se[ok], fmt=mk, ms=5,
                    color=COLORS[n], label=NAMES[n])  # fmt: skip
        a1.errorbar(x[~ok], s[~ok], xerr=xe[:, ~ok], yerr=se[~ok], fmt=mk, ms=5,
                    mfc="white", color=COLORS[n])  # fmt: skip
        p = np.array([r[n]["p68_um"] for r in rows])
        pe = np.array([r[n]["p68_err_um"] for r in rows])
        a2.errorbar(x, p, xerr=xe, yerr=pe, fmt=mk, ms=5, color=COLORS[n],
                    label=NAMES[n])  # fmt: skip
    a1.set_ylabel(r"$\sigma_{\mathrm{vtx-vtx}}$ [$\mu$m]")
    a1.text(0.97, 0.05, "Open markers: close-pair excess exceeds the dip",
            transform=a1.transAxes, ha="right", fontsize="x-small")  # fmt: skip
    a2.set_ylabel(r"68% $|z_{\mathrm{reco}} - z_{\mathrm{true}}|$ [$\mu$m]")
    a2.set_xlabel(r"Number of interactions $\mu$")
    for ax in (a1, a2):
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        ax.legend(loc="upper right")
    a2.set_xlim(0, 200)
    atlas_label(a1, desc=DESC)
    save_figure(fig, out, "resolution_vs_mu")
    print(f"summary plot: {out / 'resolution_vs_mu.png'} ({label})", flush=True)
