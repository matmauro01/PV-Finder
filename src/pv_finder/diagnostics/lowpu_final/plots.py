"""ATLAS-style figures of the low pile-up final recompute (paper Figs. 8, 9, 10).

Styling goes through ``paper_figures.style`` / ``gnn.diagnostics.plot_style``
(mplhep ATLAS, PDF + PNG). Simulation figures carry "Simulation Internal",
collision-data figures "Internal".
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pv_finder.diagnostics.data_mc_validation.plots import mu_profile
from pv_finder.diagnostics.paper_figures import style

ARM_STYLE = {
    "pvf": ("PV-Finder", style.PVF_COLOR, "o"),
    "pvf_gnn": ("PV-Finder + GNN", style.PVF_COLOR, "o"),
    "amvf": ("AMVF", style.AMVF_COLOR, "s"),
    "truth": ("Truth vertices, $\\geq 2$ tracks", "black", None),
}


def yields_vs_mu(
    mu: np.ndarray, counts: dict[str, np.ndarray], lines: Sequence[str],
    out: Path, name: str,
) -> list[Path]:  # fmt: skip
    """Mean number of vertices per event versus integer mu (Fig. 8b)."""
    mus, n, prof = mu_profile(mu, counts)
    ok = n >= 5
    fig, ax = style.new_figure()
    for k, (mean, sem) in prof.items():
        lab, col, mk = ARM_STYLE[k]
        if mk is None:
            ax.plot(mus[ok], mean[ok], "--", color=col, lw=1.5, label=lab)
        else:
            ax.errorbar(mus[ok], mean[ok], yerr=sem[ok], fmt=mk, color=col, ms=4,
                        elinewidth=1, label=lab)  # fmt: skip
    ax.set_xlabel("$\\mu$")
    ax.set_ylabel("Vertices / event")
    ax.set_ylim(0, max(prof["truth"][0][ok].max(), 1) * 1.45)
    style.label_block(ax, style.SIM_STATUS, lines)
    ax.legend(loc="lower right", frameon=False, fontsize="small")
    return style.save(fig, out, name)


def category_bars(
    fractions: dict[str, dict[str, tuple[float, float]]], lines: Sequence[str],
    out: Path, name: str,
) -> list[Path]:  # fmt: skip
    """Clean / Merged / Split / Fake fractions of reco vertices, two arms (Fig. 9b)."""
    cats = ("clean", "merged", "split", "fake")
    fig, ax = style.new_figure()
    x = np.arange(len(cats))
    width = 0.38
    for j, (arm, vals) in enumerate(fractions.items()):
        lab, col, _ = ARM_STYLE[arm]
        v = np.array([100 * vals[c][0] for c in cats])
        e = np.array([100 * vals[c][1] for c in cats])
        bars = ax.bar(x + (j - 0.5) * width, v, width, yerr=e, color=col, alpha=0.85,
                      label=lab, capsize=2)  # fmt: skip
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 1.5, f"{val:.1f}",
                    ha="center", fontsize=10)  # fmt: skip
    ax.set_xticks(x, ["Clean", "Merged", "Split", "Fake"])
    ax.set_ylabel("Fraction of vertices with tracks [%]")
    ax.set_ylim(0, 140)
    style.label_block(ax, style.SIM_STATUS, lines)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    return style.save(fig, out, name)


def eff_vs_ntrk(
    edges: np.ndarray, truth: np.ndarray, matched: dict[str, np.ndarray],
    lines: Sequence[str], out: Path, name: str,
) -> list[Path]:  # fmt: skip
    """Clean-vertex (matched) efficiency versus truth-vertex track count (Fig. 9a)."""
    fig, ax = style.new_figure()
    ok = truth > 0
    lo, hi = edges[:-1][ok], edges[1:][ok]
    hi = np.where(hi > 200, 150, hi)
    xc = np.sqrt(lo * hi)
    for j, (arm, m) in enumerate(matched.items()):
        lab, col, mk = ARM_STYLE[arm]
        eff = m[ok] / truth[ok]
        err = np.sqrt(eff * (1 - eff) / truth[ok])
        ax.errorbar(xc * (1 + 0.03 * (j - 0.5)), 100 * eff, yerr=100 * err,
                    xerr=None, fmt=mk, color=col, ms=5, label=lab)  # fmt: skip
    ax.set_xscale("log")
    ax.set_xlim(1.5, 250)
    ax.set_xlabel("Number of tracks of the truth vertex")
    ax.set_ylabel("Matched-vertex efficiency [%]")
    ax.set_ylim(0, 140)
    style.label_block(ax, style.SIM_STATUS, lines)
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    return style.save(fig, out, name)


def t_scan(
    t: np.ndarray, curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    chosen: float, lines: Sequence[str], out: Path, name: str,
) -> list[Path]:  # fmt: skip
    """Metrics versus the GNN score cut t, AMVF as horizontal lines."""
    fig, ax = style.new_figure(9.0, 6.5)
    colors = ("#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
    for (lab, (v, e, amvf)), col in zip(curves.items(), colors):
        ax.errorbar(t, 100 * v, yerr=100 * e, fmt="o-", color=col, ms=4, lw=1.2,
                    label=lab)  # fmt: skip
        ax.axhline(100 * amvf, color=col, ls=":", lw=1.2)
    ax.axvline(chosen, color="grey", ls="--", lw=1)
    ax.text(chosen, 41, f" t = {chosen:g}", color="grey", fontsize=11)
    ax.set_xlim(0.84, 1.0)
    ax.set_ylim(40, 125)
    ax.set_xlabel("GNN score cut $t$")
    ax.set_ylabel("[%]   (dotted: AMVF)")
    style.label_block(ax, style.SIM_STATUS, lines)
    ax.legend(loc="upper right", frameon=False, fontsize="x-small", ncol=2)
    return style.save(fig, out, name)


def paths(p: list[Path]) -> list[str]:
    return [str(x) for x in p]


def summary_paths(d: dict[str, Any]) -> dict[str, Any]:
    """Convert every list of Paths in ``d`` to strings (JSON)."""
    return {k: paths(v) if isinstance(v, list) else v for k, v in d.items()}
