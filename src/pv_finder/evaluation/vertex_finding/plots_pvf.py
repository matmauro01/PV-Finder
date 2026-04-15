"""plots_pvf.py — Plotting helpers for run_eval_pvf.py.

Functions, each saves one PNG to output_dir:
  plot_resolution        — pairwise Δz distribution + sigmoid fit
  plot_performance       — reco category fractions + efficiency vs pileup
  plot_stats             — avg count/event per category vs pileup (all events)
  plot_reco_vs_mu        — total reco PVs/event vs pileup, PV-Finder vs AMVF vs truth
  plot_category_counts   — per-event distribution of clean/merged/split/fake counts
"""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — prevents X11/display crashes over SSH + tmux
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Consistent styling
_COLORS = {
    "clean": "#1f77b4",
    "merged": "#2ca02c",
    "split": "#d62728",
    "fake": "#7f7f7f",
}
_MARKERS = {"clean": "o", "merged": "s", "split": "^", "fake": "v"}
_FONT = {"fontsize": 13}


def plot_resolution(
    dz_arr: np.ndarray,
    sigma_vtx_vtx: float,
    popt,
    sigmoid_fit,
    mode_label: str,
    output_dir: Path,
    title: str = "",
) -> None:
    """Pairwise Δz distribution (points + Poisson errors) with sigmoid fit."""
    bins_res = np.linspace(-6.0, 6.0, 61)
    bin_ctrs = 0.5 * (bins_res[:-1] + bins_res[1:])
    counts, _ = np.histogram(dz_arr, bins=bins_res)
    errors = np.sqrt(counts.astype(float))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        bin_ctrs,
        counts,
        yerr=errors,
        fmt="ko",
        ms=4,
        capsize=2,
        elinewidth=1,
        label="Reconstructed PV pairs",
    )
    if popt is not None:
        xf = np.linspace(-6, 6, 400)
        ax.plot(
            xf,
            sigmoid_fit(xf, *popt),
            "r-",
            lw=2,
            label=f"Sigmoid fit  σ = {sigma_vtx_vtx:.3f} mm",
        )
        ax.axvline(sigma_vtx_vtx, color="red", ls="--", alpha=0.6)
        ax.axvline(
            -sigma_vtx_vtx,
            color="red",
            ls="--",
            alpha=0.6,
            label=f"±σ = ±{sigma_vtx_vtx:.3f} mm",
        )
    ax.set_xlabel("Δz between reconstructed PV pairs (mm)", **_FONT)
    ax.set_ylabel("Counts", **_FONT)
    ax.set_title(title or f"PVF Resolution — {mode_label}", **_FONT)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "resolution_plot.png", dpi=150)
    plt.close()


def plot_performance(
    per_event: list,
    overall_eff: float,
    fp_rate: float,
    sigma_vtx_vtx: float,
    root_z_available: bool,
    mode_label: str,
    output_dir: Path,
    title: str = "",
) -> None:
    """Reco category fractions and efficiency vs pileup proxy."""
    pu: dict = defaultdict(lambda: defaultdict(list))
    for r in per_event:
        p = round(r["mu"]) if root_z_available else r["n_truth"]
        nd = r["n_pred"] or 1
        for k in ("clean", "merged", "split", "fake"):
            pu[p][k].append(r[k] / nd)
        pu[p]["eff"].append(r["eff"])

    pu_vals = sorted(pu.keys())
    _m = lambda lst: float(np.mean(lst)) if lst else 0.0  # noqa: E731
    _sem = lambda lst: float(np.std(lst) / np.sqrt(len(lst))) if len(lst) > 1 else 0.0  # noqa: E731

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax = axes[0]
    for key in ("clean", "merged", "split", "fake"):
        means = [_m(pu[p][key]) for p in pu_vals]
        sems = [_sem(pu[p][key]) for p in pu_vals]
        ax.errorbar(
            pu_vals,
            means,
            yerr=sems,
            fmt=f"-{_MARKERS[key]}",
            color=_COLORS[key],
            ms=4,
            capsize=2,
            label=key.capitalize(),
        )
    ax.set_ylabel("Fraction of reconstructed PVs", **_FONT)
    ax.set_title(title or f"PVF Performance — {mode_label}", **_FONT)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    ax = axes[1]
    eff_means = [_m(pu[p]["eff"]) for p in pu_vals]
    eff_sems = [_sem(pu[p]["eff"]) for p in pu_vals]
    ax.errorbar(
        pu_vals, eff_means, yerr=eff_sems, fmt="-o", color="#1f77b4", ms=4, capsize=2
    )
    ax.axhline(
        overall_eff,
        color="red",
        ls="--",
        alpha=0.7,
        label=f"Overall eff = {overall_eff:.3f}",
    )
    xlabel = "ActualNumOfInt (μ)" if root_z_available else "N truth PVs/evt"
    ax.set_xlabel(xlabel, **_FONT)
    ax.set_ylabel("Efficiency (matched / truth)", **_FONT)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)

    # Annotation box with key metrics
    txt = f"σ = {sigma_vtx_vtx:.3f} mm\nEff = {overall_eff:.3f}\nFP = {fp_rate:.2f}/evt"
    ax.text(
        0.98,
        0.05,
        txt,
        transform=ax.transAxes,
        fontsize=11,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "performance_plot.png", dpi=150)
    plt.close()


def plot_stats(
    per_event: list,
    root_z_available: bool,
    mode_label: str,
    output_dir: Path,
    title: str = "",
) -> None:
    """Avg count/event for clean/merged/split/fake vs pileup (all events).

    Adjacent-mu bins, mean ± SEM lines per category (mattia_finder style).
    X-axis: ActualNumOfInt (rounded) when ROOT available, else N truth PVs.
    """
    sep: dict = defaultdict(
        lambda: {k: [] for k in ("clean", "merged", "split", "fake")}
    )
    for r in per_event:
        p = round(r["mu"]) if root_z_available else r["n_truth"]
        for k in ("clean", "merged", "split", "fake"):
            sep[p][k].append(r[k])

    mu_vals = sorted(sep.keys())
    cats = ("clean", "merged", "split", "fake")
    mu_binned: list = []
    means: dict = {k: [] for k in cats}
    errs: dict = {k: [] for k in cats}
    for i in range(0, len(mu_vals) - 1, 2):
        mu_binned.append(0.5 * (mu_vals[i] + mu_vals[i + 1]))
        for k in cats:
            data = sep[mu_vals[i]][k] + sep[mu_vals[i + 1]][k]
            means[k].append(float(np.mean(data)) if data else 0.0)
            errs[k].append(
                float(np.std(data) / np.sqrt(len(data))) if len(data) > 1 else 0.0
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in cats:
        ax.errorbar(
            mu_binned,
            means[key],
            yerr=errs[key],
            fmt=f"-{_MARKERS[key]}",
            color=_COLORS[key],
            ms=4,
            capsize=3,
            label=key.capitalize(),
        )
    xlabel = "ActualNumOfInt (μ)" if root_z_available else "N truth PVs/evt"
    ax.set_xlabel(xlabel, **_FONT)
    ax.set_ylabel("Avg count / event", **_FONT)
    ax.set_title(title or f"PVF Reco Categories vs Pileup — {mode_label}", **_FONT)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(output_dir / "stats_histogram.png", dpi=150)
    plt.close()


def plot_reco_vs_mu(
    per_event: list,
    mode_label: str,
    output_dir: Path,
    title: str = "",
) -> None:
    """Total reconstructed PVs per event vs pileup (μ) — PV-Finder vs AMVF.

    Reuses the `n_pred` (PV-Finder, peak finder with integral_threshold=0.5)
    and `n_amvf` (AMVF nTracks≥2 from ROOT) fields already stored in the
    per-event result records. Also overlays `n_truth` as a reference line.
    All three curves are binned by rounded ActualNumOfInt and shown with SEM
    error bars.
    """
    buckets = defaultdict(lambda: {"pvf": [], "amvf": [], "truth": []})
    for r in per_event:
        if r.get("mu") is None:
            continue
        mu = int(round(r["mu"]))
        buckets[mu]["pvf"].append(r["n_pred"])
        buckets[mu]["truth"].append(r["n_truth"])
        if r.get("n_amvf") is not None:
            buckets[mu]["amvf"].append(r["n_amvf"])

    if not buckets:
        return

    mus = sorted(buckets.keys())

    def _stats(key):
        means, sems = [], []
        for m in mus:
            vals = buckets[m][key]
            if vals:
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)  # fmt: skip
            else:
                means.append(np.nan)
                sems.append(0.0)
        return np.array(means), np.array(sems)

    pvf_m, pvf_e = _stats("pvf")
    amvf_m, amvf_e = _stats("amvf")
    tr_m, _ = _stats("truth")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(mus, pvf_m, yerr=pvf_e, fmt="-o", ms=5, lw=1.8, capsize=3,
                color="#1f77b4", label="PV-Finder", zorder=3)  # fmt: skip
    has_amvf = np.any(~np.isnan(amvf_m))
    if has_amvf:
        ax.errorbar(mus, amvf_m, yerr=amvf_e, fmt="-s", ms=5, lw=1.8, capsize=3,
                    color="#d62728", label="AMVF (nTracks≥2)", zorder=2)  # fmt: skip
    ax.plot(mus, tr_m, "--", lw=1.5, color="#555555", alpha=0.8,
            label="Truth PVs (nTracks≥2)", zorder=1)  # fmt: skip

    n_evt = sum(len(buckets[m]["pvf"]) for m in mus)
    ax.set_xlabel("ActualNumOfInt (μ)", **_FONT)
    ax.set_ylabel("Avg reconstructed PVs / event", **_FONT)
    ax.set_title(
        title or f"Reconstructed vertices vs pileup — {mode_label}  ({n_evt} events)",
        **_FONT,
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="upper left", frameon=True)
    ax.tick_params(labelsize=11)

    if has_amvf:
        overall_pvf = float(np.nanmean([np.mean(buckets[m]["pvf"]) for m in mus]))
        overall_amvf = float(
            np.nanmean([np.mean(buckets[m]["amvf"]) for m in mus if buckets[m]["amvf"]])
        )
        overall_tr = float(np.nanmean([np.mean(buckets[m]["truth"]) for m in mus]))
        txt = (
            f"⟨PV-Finder⟩ = {overall_pvf:.2f}/evt\n"
            f"⟨AMVF⟩     = {overall_amvf:.2f}/evt\n"
            f"⟨Truth⟩    = {overall_tr:.2f}/evt"
        )
        ax.text(0.98, 0.05, txt, transform=ax.transAxes, fontsize=10,
                va="bottom", ha="right", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.85))  # fmt: skip

    plt.tight_layout()
    plt.savefig(output_dir / "reco_vs_mu.png", dpi=150)
    plt.close()


def plot_category_counts(
    per_event: list,
    mode_label: str,
    output_dir: Path,
    title: str = "",
    eval_label: str = "",
) -> None:
    """Per-event distribution of clean / merged / split / fake counts.

    Four overlaid step-filled histograms on integer-valued bins, one per
    category. Each legend entry shows the category mean, σ, and total.
    `eval_label` is written as a small annotation in the upper-right corner
    so the plot is self-identifying when compared across runs.
    """
    cats = ("clean", "merged", "split", "fake")
    data = {k: np.asarray([r[k] for r in per_event], dtype=float) for k in cats}
    if not per_event or all(v.size == 0 for v in data.values()):
        return

    max_cnt = int(max(v.max() for v in data.values() if v.size))
    bins = np.arange(-0.5, max_cnt + 1.5, 1.0)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for k in cats:
        vals = data[k]
        mean = float(vals.mean())
        std = float(vals.std())
        total = int(vals.sum())
        label = f"{k.capitalize():<6}  ⟨N⟩={mean:5.2f}  σ={std:4.2f}  Σ={total:>5d}"
        ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.28,
                color=_COLORS[k], edgecolor=_COLORS[k], linewidth=2.2,
                label=label)  # fmt: skip

    ax.set_xlabel("Count per event", **_FONT)
    ax.set_ylabel("Events", **_FONT)
    ax.set_title(title or f"Per-event category counts — {mode_label}", **_FONT)
    ax.set_xlim(-0.5, max_cnt + 0.5)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=11)
    leg = ax.legend(
        fontsize=10, loc="upper right", framealpha=0.9, prop={"family": "monospace"}
    )
    leg.set_title(f"{len(per_event)} events", prop={"size": 10, "weight": "bold"})

    if eval_label:
        ax.text(
            0.02, 0.97, eval_label, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f5f5f5", ec="#cccccc",
                      alpha=0.9),
        )  # fmt: skip

    plt.tight_layout()
    plt.savefig(output_dir / "category_counts_hist.png", dpi=150)
    plt.close()
