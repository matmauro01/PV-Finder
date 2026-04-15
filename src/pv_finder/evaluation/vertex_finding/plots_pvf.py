"""plots_pvf.py — Plotting helpers for run_eval_pvf.py.

Functions, each saves one PNG to output_dir:
  plot_resolution        — pairwise Δz distribution + sigmoid fit
  plot_performance       — reco category fractions + efficiency vs pileup
  plot_stats             — total reco PVs/event vs pileup, PV-Finder vs AMVF
  plot_reco_vs_mu        — total reco PVs/event vs pileup, PV-Finder vs AMVF vs truth
  plot_category_counts   — 5-bar summary (total + 4 categories) in high-pileup window
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
    """Total reconstructed PVs / event vs pileup — PV-Finder vs AMVF.

    Two curves: PV-Finder (`n_pred` = clean+merged+split+fake) and AMVF
    (nTracks ≥ 2). The AMVF source is `n_amvf` when present (MC eval has
    RecoVertex_nTracks from ROOT) and falls back to `n_truth` for real-data
    evaluations (`run_eval_pvf_run3.py`), where `n_truth` already holds the
    AMVF reference vertex list.
    """
    if not per_event:
        return

    amvf_key = "n_amvf" if per_event[0].get("n_amvf") is not None else "n_truth"

    buckets: dict = defaultdict(lambda: {"pvf": [], "amvf": []})
    for r in per_event:
        if root_z_available and r.get("mu") is not None:
            p = round(r["mu"])
        else:
            p = r["n_truth"]
        buckets[p]["pvf"].append(r["n_pred"])
        buckets[p]["amvf"].append(r[amvf_key])

    mus = sorted(buckets.keys())

    def _curve(key):
        m, e = [], []
        for mu in mus:
            vals = buckets[mu][key]
            m.append(float(np.mean(vals)) if vals else np.nan)
            e.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)  # fmt: skip
        return np.array(m), np.array(e)

    pvf_m, pvf_e = _curve("pvf")
    amvf_m, amvf_e = _curve("amvf")

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.errorbar(mus, pvf_m, yerr=pvf_e, fmt="-o", ms=5, lw=1.8, capsize=3,
                color="#1f77b4", label="PV-Finder (Σ clean+merged+split+fake)")  # fmt: skip
    ax.errorbar(mus, amvf_m, yerr=amvf_e, fmt="-s", ms=5, lw=1.8, capsize=3,
                color="#d62728", label="AMVF (nTracks ≥ 2)")  # fmt: skip

    xlabel = "ActualNumOfInt (μ)" if root_z_available else "N truth PVs/evt"
    ax.set_xlabel(xlabel, **_FONT)
    ax.set_ylabel("Total reconstructed PVs / event", **_FONT)
    ax.set_title(
        title
        or f"Total reconstructed PVs vs pileup — {mode_label}  ({len(per_event)} events)",  # noqa: E501
        **_FONT,
    )
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="upper left", frameon=True)
    ax.tick_params(labelsize=11)

    overall_pvf = float(
        np.nanmean([np.mean(buckets[m]["pvf"]) for m in mus if buckets[m]["pvf"]])
    )
    overall_amvf = float(
        np.nanmean([np.mean(buckets[m]["amvf"]) for m in mus if buckets[m]["amvf"]])
    )
    txt = f"⟨PV-Finder⟩ = {overall_pvf:.2f}/evt\n⟨AMVF⟩     = {overall_amvf:.2f}/evt"
    ax.text(0.98, 0.05, txt, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.85))  # fmt: skip

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
    mu_min: int = 55,
    mu_max: int = 65,
) -> None:
    """Mean per-event reco counts as a 5-bar chart in the high-pileup window.

    Bars (left → right): Total, Clean, Merged, Split, Fake. Total is the
    sum of the four categories (equivalent to `n_pred`).

    Events are filtered to `mu ∈ [mu_min, mu_max]` when pileup is available,
    otherwise all events are used. The pileup window is written into the
    default title. Error bars are SEM across events in the window.
    """
    if not per_event:
        return

    events_with_mu = [r for r in per_event if r.get("mu") is not None]
    if not events_with_mu:
        # Pileup unavailable (ROOT truth not loaded) — refuse to draw a
        # "high pileup" plot rather than silently include everything.
        print(
            f"  [plot_category_counts] skipped: no per-event mu "
            f"(need --root-truth); cannot filter to [{mu_min}, {mu_max}]"
        )
        return
    filt = [r for r in events_with_mu if mu_min <= round(r["mu"]) <= mu_max]
    if not filt:
        print(f"  [plot_category_counts] skipped: 0 events in μ ∈ [{mu_min}, {mu_max}]")
        return

    labels = ("Total", "Clean", "Merged", "Split", "Fake")
    totals = np.asarray(
        [
            r.get("n_pred", r["clean"] + r["merged"] + r["split"] + r["fake"])
            for r in filt
        ],  # noqa: E501
        dtype=float,
    )
    cleans = np.asarray([r["clean"] for r in filt], dtype=float)
    mergeds = np.asarray([r["merged"] for r in filt], dtype=float)
    splits = np.asarray([r["split"] for r in filt], dtype=float)
    fakes = np.asarray([r["fake"] for r in filt], dtype=float)
    stacks = (totals, cleans, mergeds, splits, fakes)

    means = np.array([float(v.mean()) for v in stacks])
    n = len(filt)
    sems = np.array([float(v.std() / np.sqrt(n)) if n > 1 else 0.0 for v in stacks])

    # Vivid, saturated palette (distinct from the muted _COLORS used elsewhere)
    vivid = {
        "total": "#2C3E50",  # slate navy
        "clean": "#3498DB",  # bright azure
        "merged": "#2ECC71",  # emerald
        "split": "#E74C3C",  # alizarin
        "fake": "#F39C12",  # vivid orange
    }
    colors = [vivid[k] for k in ("total", "clean", "merged", "split", "fake")]
    edges = ["#1a2530", "#1f6396", "#1b8449", "#962a1f", "#9c600a"]

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafbfc")
    x = np.arange(len(labels))
    bars = ax.bar(
        x, means, width=0.72, yerr=sems,
        color=colors, edgecolor=edges, linewidth=1.8, alpha=0.95,
        error_kw=dict(ecolor="#222", elinewidth=1.5, capsize=7, capthick=1.4),
    )  # fmt: skip

    ymax = float((means + sems).max()) if means.size else 1.0
    pad = 0.025 * ymax
    for bar, m, s in zip(bars, means, sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s + pad,
            f"{m:.2f}", ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#1a1a1a",
        )  # fmt: skip

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight="semibold")
    ax.set_ylabel("Mean reconstructed PVs / event", fontsize=14, fontweight="semibold")
    mu_desc = f"μ ∈ [{mu_min}, {mu_max}]"
    ax.set_title(
        title or f"Per-event reco counts, {mu_desc}  —  {mode_label}",
        fontsize=15, fontweight="bold", pad=14,
    )  # fmt: skip
    ax.set_ylim(bottom=0, top=ymax * 1.28)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.35, ls="--", lw=0.8, color="#666")
    ax.tick_params(labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666")
        ax.spines[spine].set_linewidth(1.2)

    info = f"{n} events\n{mu_desc}"
    if eval_label:
        info = f"{eval_label}\n{info}"
    ax.text(
        0.985, 0.97, info, transform=ax.transAxes, fontsize=9,
        va="top", ha="right", family="monospace", color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#2C3E50", lw=1.1,
                  alpha=0.92),
    )  # fmt: skip

    plt.tight_layout()
    plt.savefig(output_dir / "category_counts_hist.png", dpi=160,
                bbox_inches="tight", facecolor="white")  # fmt: skip
    plt.close()
