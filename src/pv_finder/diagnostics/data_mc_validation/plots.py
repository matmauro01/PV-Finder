"""Data/MC figures for note Fig. 6.2: resolution and yield with MC overlaid.

ATLAS conventions: data as black filled markers with statistical error bars,
simulation as coloured lines / open markers, ratio panels Data/MC.  Styling,
labels and PDF+PNG output go through ``gnn.diagnostics.plot_style``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gnn.diagnostics.plot_style import (  # noqa: E402
    atlas_label,
    save_figure,
    use_atlas_style,
)
from pv_finder.diagnostics.data_mc_validation.samples import (  # noqa: E402
    FitResult,
    sigmoid,
)

MC_COLOR = "#0072B2"  # Okabe-Ito blue
FIT_COLOR = "#D55E00"
AMVF_MC_COLOR = "#D55E00"
PVF_MC_COLOR = "#0072B2"
MIN_EVENTS_RATIO = 5  # mu bins with fewer events (data or MC) get no ratio point


@dataclass
class Labels:
    """Annotation text for one data/MC pairing."""

    data: str  # e.g. "Data 2018, $\\sqrt{s}$ = 13 TeV, $\\langle\\mu\\rangle$ = 42.5"
    mc: str  # e.g. "$t\\bar{t}$ sim., $\\sqrt{s}$ = 13 TeV, ..."
    caveat: str | None = None  # extra line (e.g. MC does not match conditions)


def _desc(labels: Labels) -> str:
    lines = [labels.data, labels.mc]
    if labels.caveat:
        lines.append(labels.caveat)
    return "\n".join(lines)


def _ratio(num: np.ndarray, num_err: np.ndarray, den: np.ndarray,
           den_err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    with np.errstate(divide="ignore", invalid="ignore"):
        r = num / den
        e = r * np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2)
    return r, e


def plot_resolution_overlay(
    data_fit: FitResult,
    mc_fit: FitResult,
    labels: Labels,
    out_dir: Path,
    name: str,
) -> dict[str, object]:
    """Pairwise dz of PV-Finder vertices: data points + fit, MC shape overlaid.

    MC is scaled to the data pair count in the plotted range (shape comparison),
    its fit curve scaled by the same factor.  Returns the normalisation and
    the saved paths.
    """
    use_atlas_style()
    edges, x = data_fit.edges, data_fit.centres
    nd, nm = data_fit.counts.astype(float), mc_fit.counts.astype(float)
    if not np.array_equal(edges, mc_fit.edges):
        raise ValueError("data and MC histograms must share binning")
    norm = nd.sum() / nm.sum()
    mc_scaled, mc_err = nm * norm, np.sqrt(nm) * norm

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})  # fmt: skip
    ax.stairs(mc_scaled, edges, color=MC_COLOR, lw=1.6, fill=False,
              label=f"Simulation (norm.), $\\sigma$ = {mc_fit.sigma:.3f} $\\pm$ "
                    f"{mc_fit.sigma_err:.3f} mm")  # fmt: skip
    ax.stairs(mc_scaled + mc_err, edges, baseline=mc_scaled - mc_err, fill=True,
              color=MC_COLOR, alpha=0.25, lw=0)  # fmt: skip
    xf = np.linspace(edges[0], edges[-1], 2000)
    ax.plot(xf, sigmoid(xf, *mc_fit.popt) * norm, color=MC_COLOR, ls="--", lw=1.2,
            label="Simulation fit (norm.)")  # fmt: skip
    ax.plot(xf, sigmoid(xf, *data_fit.popt), color=FIT_COLOR, lw=1.8,
            label=f"Data fit, $\\sigma$ = {data_fit.sigma:.3f} $\\pm$ "
                  f"{data_fit.sigma_err:.3f} mm")  # fmt: skip
    ax.errorbar(x, nd, yerr=np.sqrt(nd), fmt="o", color="black", ms=4,
                elinewidth=1, capsize=0, label="Data", zorder=5)  # fmt: skip
    for s in (-data_fit.sigma, data_fit.sigma):
        ax.axvline(s, color=FIT_COLOR, ls=":", lw=1, alpha=0.8)
    ax.set_ylabel("PV-Finder vertex pairs / 0.2 mm")
    ax.set_ylim(0, max(nd.max(), mc_scaled.max()) * 2.1)
    atlas_label(ax, status="Internal", desc=_desc(labels), desc_xy=(0.04, 0.88))
    ax.legend(loc="upper right", fontsize="x-small", frameon=False,
              bbox_to_anchor=(1.0, 0.74))  # fmt: skip

    r, re = _ratio(nd, np.sqrt(nd), mc_scaled, mc_err)
    ok = (nd >= 10) & (nm >= 10)  # near-empty dip bins carry no ratio information
    rax.errorbar(x[ok], r[ok], yerr=re[ok], fmt="o", color="black", ms=3, elinewidth=1)
    rax.axhline(1.0, color=MC_COLOR, lw=1)
    rax.set_ylim(0.0, 2.0)
    rax.set_ylabel("Data / Sim.", fontsize="small")
    fig.align_ylabels([ax, rax])
    rax.set_xlabel(r"$\Delta z$ between PV-Finder vertices [mm]")
    paths = save_figure(fig, out_dir, name)
    plateau = np.abs(x) > 2.0
    chi2 = float(np.sum(((nd - mc_scaled) ** 2 / (nd + mc_err**2))[plateau & ok]))
    return {
        "mc_norm": float(norm),
        "n_pairs_data": int(nd.sum()),
        "n_pairs_mc": int(nm.sum()),
        "plateau_chi2_over_ndf": [chi2, int(np.sum(plateau & ok))],
        "paths": [str(p) for p in paths],
    }


def mu_profile(
    mu: np.ndarray, values: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Mean and standard error of each value per integer-mu bin (rounded mu).

    SEM uses ``np.std`` (ddof 0) / sqrt(n), as the current note figures.
    """
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, m in enumerate(mu):
        buckets[int(round(m))].append(i)
    mus = np.array(sorted(buckets))
    n = np.array([len(buckets[m]) for m in mus])
    out = {}
    for k, v in values.items():
        mean = np.array([np.mean(v[buckets[m]]) for m in mus])
        sem = np.array([np.std(v[buckets[m]]) / np.sqrt(len(buckets[m]))
                        if len(buckets[m]) > 1 else 0.0 for m in mus])  # fmt: skip
        out[k] = (mean, sem)
    return mus, n, out


def plot_yield_overlay(
    data_mu: np.ndarray,
    data_vals: dict[str, np.ndarray],
    mc_mu: np.ndarray,
    mc_vals: dict[str, np.ndarray],
    labels: Labels,
    out_dir: Path,
    name: str,
) -> dict[str, object]:
    """Mean PV-Finder / AMVF vertices per event vs mu, data and MC, with ratio.

    ``*_vals`` hold per-event counts under keys ``"pvf"`` and ``"amvf"``.
    Bottom panel: data mu distribution, and MC scaled to the data event count.
    """
    use_atlas_style()
    dm, dn, dp = mu_profile(data_mu, data_vals)
    mm, mn, mp = mu_profile(mc_mu, mc_vals)
    fig, (ax, rax, hax) = plt.subplots(3, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.1, 1.1], "hspace": 0.06})  # fmt: skip
    style = {
        "pvf": ("o", PVF_MC_COLOR, "PV-Finder"),
        "amvf": ("s", AMVF_MC_COLOR, "AMVF"),
    }
    for k, (mk, col, lab) in style.items():
        ax.plot(mm, mp[k][0], color=col, lw=1.4, zorder=1)
        ax.errorbar(mm, mp[k][0], yerr=mp[k][1], fmt=mk, mfc="white", mec=col,
                    color=col, ms=5, elinewidth=1, label=f"{lab}, simulation")  # fmt: skip
        ax.errorbar(dm, dp[k][0], yerr=dp[k][1], fmt=mk, color="black", ms=5,
                    elinewidth=1, label=f"{lab}, data", zorder=5)  # fmt: skip
    ax.set_ylabel("Vertices / event")
    lo, hi = dm.min() - 1.5, dm.max() + 1.5
    ax.set_xlim(lo, hi)
    in_rng = (mm >= lo) & (mm <= hi)
    top = max(dp["pvf"][0].max(), mp["pvf"][0][in_rng].max())
    ax.set_ylim(0, top * 1.45)
    atlas_label(ax, status="Internal", desc=_desc(labels), desc_xy=(0.04, 0.88))
    ax.legend(loc="lower right", fontsize="x-small", frameon=False, ncol=2)

    common = np.intersect1d(dm[dn >= MIN_EVENTS_RATIO], mm[mn >= MIN_EVENTS_RATIO])
    di, mi = np.searchsorted(dm, common), np.searchsorted(mm, common)
    ratios = {}
    for k, (mk, col, lab) in style.items():
        r, e = _ratio(dp[k][0][di], dp[k][1][di], mp[k][0][mi], mp[k][1][mi])
        off = -0.15 if k == "pvf" else 0.15
        rax.errorbar(common + off, r, yerr=e, fmt=mk, color=col, ms=4, elinewidth=1,
                     label=lab)  # fmt: skip
        ratios[k] = {"mu": common.tolist(), "ratio": r.tolist(), "err": e.tolist()}
    rax.axhline(1.0, color="grey", lw=1)
    rax.set_ylim(0.7, 1.3)
    rax.set_ylabel("Data / Sim.", fontsize="small")
    rax.legend(loc="upper left", fontsize="xx-small", frameon=False, ncol=2)

    hax.bar(
        dm, dn, width=1.0, color="#BBBBBB", edgecolor="#666666", lw=0.5, label="Data"
    )
    scale = dn.sum() / mn.sum()
    # Open bars, not stairs: integer-mu bins may have gaps.
    hax.bar(mm, mn * scale, width=1.0, fill=False, edgecolor=MC_COLOR, lw=1.2,
            label="Simulation (norm.)")  # fmt: skip
    hax.set_ylabel("Events", fontsize="small")
    hax.set_xlabel(r"$\mu$ (ActualNumOfInt)")
    hax.legend(loc="upper left", fontsize="xx-small", frameon=False)
    paths = save_figure(fig, out_dir, name)
    return {
        "ratio_points": ratios,
        "ratio_min_events": MIN_EVENTS_RATIO,
        "paths": [str(p) for p in paths],
    }
