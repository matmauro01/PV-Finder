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


# sigma_vtx-vtx is quoted only for mu >= 100 (paper and note treatment,
# docs/evaluation/resolution_vs_mu_2026-11-11.md): below it the
# correlated excess of close reconstructed pairs exceeds the dip (merge length
# < 0 in the v6 study) and the fitted sigma measures that excess, not the
# separation of independent interactions.  Those bins are drawn open.
SIGMA_QUOTE_MU_MIN = 100.0


def _xerr(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Bin-mean mu and the asymmetric bin-edge x errors."""
    x = np.array([r["mu_mean"] for r in rows])
    lo = np.array([r["mu_lo"] for r in rows])
    hi = np.array([r["mu_hi"] for r in rows])
    return x, np.array([x - lo, hi - x])


def _draw_p68(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    """68% single-vertex residual vs mu, PV-Finder and AMVF."""
    x, xe = _xerr(rows)
    for n in ("pvf", "amvf"):
        p = np.array([r[n]["p68_um"] for r in rows])
        pe = np.array([r[n]["p68_err_um"] for r in rows])
        ax.errorbar(x, p, xerr=xe, yerr=pe, fmt="o" if n == "pvf" else "s", ms=5,
                    color=COLORS[n], label=NAMES[n])  # fmt: skip
    ax.set_ylabel(r"68% $|z_{\mathrm{reco}} - z_{\mathrm{true}}|$ [$\mu$m]")


def _draw_sigma(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    """sigma_vtx-vtx vs mu; open markers below SIGMA_QUOTE_MU_MIN."""
    x, xe = _xerr(rows)
    ok = np.array([r["mu_lo"] >= SIGMA_QUOTE_MU_MIN for r in rows])
    for n in ("pvf", "amvf"):
        mk = "o" if n == "pvf" else "s"
        s = np.array([1e3 * r[n]["sigma_mm"] for r in rows])
        se = np.array([1e3 * r[n]["boot_err_mm"] for r in rows])
        ax.errorbar(x[ok], s[ok], xerr=xe[:, ok], yerr=se[ok], fmt=mk, ms=5,
                    color=COLORS[n], label=NAMES[n])  # fmt: skip
        ax.errorbar(x[~ok], s[~ok], xerr=xe[:, ~ok], yerr=se[~ok], fmt=mk, ms=5,
                    mfc="white", color=COLORS[n])  # fmt: skip
        odd = [r["mu_lo"] for r, q in zip(rows, ok) if (r[n]["merge_len_mm"] >= 0) != q]
        if odd:
            print(f"NOTE {NAMES[n]}: merge-length sign disagrees with the quoting rule "
                  f"mu >= {SIGMA_QUOTE_MU_MIN:.0f} in bins starting at mu = {odd}",
                  flush=True)  # fmt: skip
    ax.set_ylabel(r"$\sigma_{\mathrm{vtx-vtx}}$ [$\mu$m]")
    ax.text(0.97, 0.05, rf"Open markers ($\mu$ < {SIGMA_QUOTE_MU_MIN:.0f}): "
            "close-pair excess exceeds the dip",
            transform=ax.transAxes, ha="right", fontsize="x-small")  # fmt: skip


def plot_summary(rows: list[dict[str, Any]], out: Path, label: str) -> None:
    """Paper plot (68% residual vs mu) and the two-panel p68 + sigma_vtx-vtx figure.

    ``resolution_vs_mu_p68``: the 68% single-vertex residual vs mu alone.
    ``resolution_vs_mu``: the same on top, sigma_vtx-vtx vs mu below.
    """
    use_atlas_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_p68(ax, rows)
    ax.set_xlabel(r"Number of interactions $\mu$")
    ax.set_xlim(0, 200)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(loc="upper right")
    atlas_label(ax, desc=DESC)
    save_figure(fig, out, "resolution_vs_mu_p68")

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.05},
    )  # fmt: skip
    _draw_p68(a1, rows)
    _draw_sigma(a2, rows)
    a2.set_xlabel(r"Number of interactions $\mu$")
    for a in (a1, a2):
        a.set_ylim(0, a.get_ylim()[1] * 1.35)
        a.legend(loc="upper right")
    a2.set_xlim(0, 200)
    atlas_label(a1, desc=DESC)
    save_figure(fig, out, "resolution_vs_mu")
    print(f"summary plots: {out / 'resolution_vs_mu_p68.png'}, "
          f"{out / 'resolution_vs_mu.png'} ({label})", flush=True)  # fmt: skip
