"""
Visualization functions for comparing KDE model predictions vs analytical KDE.

Provides full-event overlays, per-vertex zoom plots, agreement summaries,
residual distributions, and MC-vs-Run3 comparison figures.
"""

import os

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

from pv_finder.data.feature_loading import (
    SUBEVENT_STARTS,
    SUBEVENT_WIDTH,
    Z_MAX,
    Z_MIN,
)

# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)

# Colour palette
COL_ANALYTICAL = "#1f77b4"
COL_MODEL = "#d62728"
COL_TRUTH = "#2ca02c"


# ---------------------------------------------------------------------------
# Z-axis helpers
# ---------------------------------------------------------------------------


def _full_z_axis():
    """Return the 12000-bin z-axis covering [-240, 240] mm."""
    half_bin = (Z_MAX - Z_MIN) / 24000.0
    return np.linspace(Z_MIN, Z_MAX, 12000, endpoint=False) + half_bin


def _subevent_z_axis(sub_idx):
    """Return the 1000-bin z-axis for subevent *sub_idx*."""
    start = SUBEVENT_STARTS[sub_idx]
    half_bin = SUBEVENT_WIDTH / 2000.0
    return np.linspace(start, start + SUBEVENT_WIDTH, 1000, endpoint=False) + half_bin


def _flatten_kde(kde):
    """Flatten a (12, 1000) KDE array to (12000,)."""
    return np.asarray(kde).reshape(-1)


def _ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Full-event KDE overlay
# ---------------------------------------------------------------------------


def plot_event_overlay(
    analytical,
    model_pred,
    event_idx,
    vtx_positions,
    output_dir,
    dataset_label,
    truth_kde=None,
):
    """Plot full-event KDE overlay with residual panel."""
    _ensure_dir(output_dir)
    z = _full_z_axis()
    ana_flat = _flatten_kde(analytical)
    mod_flat = _flatten_kde(model_pred)

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(16, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # -- Top panel: KDE overlay --
    ax_top.plot(z, ana_flat, color=COL_ANALYTICAL, label="Analytical")
    ax_top.plot(z, mod_flat, color=COL_MODEL, linestyle="--", label="Model")
    if truth_kde is not None:
        truth_flat = _flatten_kde(truth_kde)
        ax_top.plot(z, truth_flat, color=COL_TRUTH, linestyle=":", label="Truth (H5)")

    for vz in vtx_positions:
        ax_top.axvline(vz, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax_top.set_ylabel("KDE value")
    ax_top.legend(loc="upper right")
    ax_top.set_title(f"{dataset_label.upper()} Event {event_idx}")

    # -- Bottom panel: residual --
    residual = mod_flat - ana_flat
    ax_bot.plot(z, residual, color=COL_MODEL, linewidth=1.0)
    ax_bot.axhline(0, color="black", linewidth=0.8)
    ax_bot.set_xlabel("z [mm]")
    ax_bot.set_ylabel("Residual")
    ax_bot.set_xlim(Z_MIN, Z_MAX)

    stem = f"{dataset_label}_event{event_idx:04d}_overlay"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Per-vertex zoom
# ---------------------------------------------------------------------------


def plot_per_vertex(
    analytical,
    model_pred,
    vtx_z,
    vtx_idx,
    event_idx,
    output_dir,
    dataset_label,
    window_mm=5.0,
    truth_kde=None,
):
    """Plot zoomed KDE overlay around a single vertex with residual panel."""
    _ensure_dir(output_dir)
    z = _full_z_axis()
    ana_flat = _flatten_kde(analytical)
    mod_flat = _flatten_kde(model_pred)

    lo = vtx_z - window_mm
    hi = vtx_z + window_mm
    mask = (z >= lo) & (z <= hi)
    z_win = z[mask]
    ana_win = ana_flat[mask]
    mod_win = mod_flat[mask]

    if len(z_win) == 0:
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # -- Top panel --
    ax_top.plot(z_win, ana_win, color=COL_ANALYTICAL, label="Analytical")
    ax_top.plot(z_win, mod_win, color=COL_MODEL, linestyle="--", label="Model")
    if truth_kde is not None:
        truth_flat = _flatten_kde(truth_kde)
        truth_win = truth_flat[mask]
        ax_top.plot(
            z_win, truth_win, color=COL_TRUTH, linestyle=":", label="Truth (H5)"
        )

    ax_top.axvline(vtx_z, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_top.set_ylabel("KDE value")
    ax_top.legend(loc="upper right")
    ax_top.set_title(
        f"{dataset_label.upper()} Event {event_idx}, "
        f"Vertex {vtx_idx} (z={vtx_z:.1f} mm)"
    )

    # Peak height ratio and local RMSE
    ana_peak = np.max(ana_win) if np.max(ana_win) > 0 else 1.0
    mod_peak = np.max(mod_win)
    peak_ratio = mod_peak / ana_peak
    local_rmse = np.sqrt(np.mean((mod_win - ana_win) ** 2))
    ax_top.text(
        0.02,
        0.95,
        f"Peak ratio: {peak_ratio:.3f}\nLocal RMSE: {local_rmse:.4f}",
        transform=ax_top.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # -- Bottom panel: residual --
    residual = mod_win - ana_win
    ax_bot.plot(z_win, residual, color=COL_MODEL, linewidth=1.0)
    ax_bot.axhline(0, color="black", linewidth=0.8)
    ax_bot.set_xlabel("z [mm]")
    ax_bot.set_ylabel("Residual")
    ax_bot.set_xlim(lo, hi)

    stem = f"{dataset_label}_event{event_idx:04d}_vtx{vtx_idx:02d}"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Agreement summary (2x2 figure)
# ---------------------------------------------------------------------------


def plot_agreement_summary(metrics_list, dataset_label, output_dir):
    """Plot 2x2 summary: Pearson r, RMSE, integral ratio, peak matching."""
    _ensure_dir(output_dir)

    # Collect pooled values
    all_pearson = []
    all_mse = []
    integral_ratios = []
    matched = []
    missed = []
    extra = []

    for m in metrics_list:
        all_pearson.extend(m["pearson_r_per_sub"])
        all_mse.extend(m["mse_per_sub"])
        integral_ratios.append(m["integral_ratio"])
        matched.append(m["n_peaks_matched"])
        missed.append(m["n_peaks_missed"])
        extra.append(m["n_peaks_extra"])

    all_pearson = np.array(all_pearson)
    all_rmse = np.sqrt(np.array(all_mse))
    integral_ratios = np.array(integral_ratios)
    matched = np.array(matched)
    missed = np.array(missed)
    extra = np.array(extra)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (0,0) Histogram of per-subevent Pearson r
    ax = axes[0, 0]
    finite_r = all_pearson[np.isfinite(all_pearson)]
    ax.hist(finite_r, bins=50, color=COL_ANALYTICAL, edgecolor="white", alpha=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Count")
    ax.set_title("Per-subevent Pearson r")

    # (0,1) Histogram of per-subevent RMSE
    ax = axes[0, 1]
    finite_rmse = all_rmse[np.isfinite(all_rmse)]
    ax.hist(finite_rmse, bins=50, color=COL_MODEL, edgecolor="white", alpha=0.8)
    ax.set_xlabel("RMSE")
    ax.set_ylabel("Count")
    ax.set_title("Per-subevent RMSE")

    # (1,0) Scatter of integral ratio per event
    ax = axes[1, 0]
    evt_idx = np.arange(len(integral_ratios))
    ax.scatter(evt_idx, integral_ratios, s=12, color=COL_ANALYTICAL, alpha=0.7)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Event index")
    ax.set_ylabel("Integral ratio (model / analytical)")
    ax.set_title("Integral Ratio per Event")

    # (1,1) Stacked bar chart: matched / missed / extra
    ax = axes[1, 1]
    evt_idx = np.arange(len(matched))
    ax.bar(evt_idx, matched, label="Matched", color=COL_TRUTH, alpha=0.8)
    ax.bar(evt_idx, missed, bottom=matched, label="Missed", color=COL_MODEL, alpha=0.8)
    ax.bar(
        evt_idx,
        extra,
        bottom=matched + missed,
        label="Extra",
        color="#ff7f0e",
        alpha=0.8,
    )
    ax.set_xlabel("Event index")
    ax.set_ylabel("Peak count")
    ax.set_title("Peak Matching per Event")
    ax.legend(loc="upper right")

    fig.suptitle(f"Agreement Summary \u2014 {dataset_label.upper()}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    stem = f"{dataset_label}_agreement_summary"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Residual distributions
# ---------------------------------------------------------------------------


def plot_residual_distributions(all_residuals, dataset_label, output_dir):
    """Plot pooled residual histogram and CDF of |residuals|."""
    _ensure_dir(output_dir)

    pooled = np.concatenate(all_residuals)

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: histogram --
    ax_hist.hist(pooled, bins=200, color=COL_ANALYTICAL, edgecolor="none", alpha=0.8)
    mean_val = np.mean(pooled)
    std_val = np.std(pooled)
    ax_hist.axvline(mean_val, color="black", linewidth=1.0, linestyle="--")
    ax_hist.text(
        0.97,
        0.95,
        f"mean = {mean_val:.2e}\nstd  = {std_val:.2e}",
        transform=ax_hist.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax_hist.set_xlabel("Residual (model \u2212 analytical)")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Residual Distribution")

    # -- Right: CDF of |residuals| --
    abs_res = np.abs(pooled)
    sorted_abs = np.sort(abs_res)
    cdf = np.arange(1, len(sorted_abs) + 1) / len(sorted_abs)
    ax_cdf.plot(sorted_abs, cdf, color=COL_MODEL, linewidth=1.2)

    p50 = np.percentile(abs_res, 50)
    p95 = np.percentile(abs_res, 95)
    ax_cdf.axvline(
        p50, color="gray", linestyle="--", linewidth=0.8, label=f"50th pctl = {p50:.2e}"
    )
    ax_cdf.axvline(
        p95,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"95th pctl = {p95:.2e}",
    )
    ax_cdf.set_xlabel("|Residual|")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("CDF of |Residuals|")
    ax_cdf.legend(loc="lower right")

    fig.suptitle(f"Residual Distributions \u2014 {dataset_label.upper()}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    stem = f"{dataset_label}_residual_distributions"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. MC vs Run 3 comparison
# ---------------------------------------------------------------------------


def plot_mc_vs_run3_comparison(mc_metrics, r3_metrics, output_dir):
    """Plot side-by-side MC vs Run 3 agreement histograms and box plots."""
    _ensure_dir(output_dir)

    mc_pearson = []
    r3_pearson = []
    mc_rmse_events = []
    r3_rmse_events = []
    mc_integral = []
    r3_integral = []

    for m in mc_metrics:
        mc_pearson.extend(m["pearson_r_per_sub"])
        mc_rmse_events.append(m["rmse_event"])
        mc_integral.append(m["integral_ratio"])

    for m in r3_metrics:
        r3_pearson.extend(m["pearson_r_per_sub"])
        r3_rmse_events.append(m["rmse_event"])
        r3_integral.append(m["integral_ratio"])

    mc_pearson = np.array(mc_pearson)
    r3_pearson = np.array(r3_pearson)
    mc_rmse_events = np.array(mc_rmse_events)
    r3_rmse_events = np.array(r3_rmse_events)
    mc_integral = np.array(mc_integral)
    r3_integral = np.array(r3_integral)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (0) Overlaid histograms of Pearson r
    ax = axes[0]
    bins_r = np.linspace(
        min(np.nanmin(mc_pearson), np.nanmin(r3_pearson)),
        1.0,
        50,
    )
    ax.hist(
        mc_pearson[np.isfinite(mc_pearson)],
        bins=bins_r,
        color=COL_ANALYTICAL,
        alpha=0.6,
        label="MC",
        edgecolor="white",
    )
    ax.hist(
        r3_pearson[np.isfinite(r3_pearson)],
        bins=bins_r,
        color=COL_MODEL,
        alpha=0.6,
        label="Run 3",
        edgecolor="white",
    )
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Count")
    ax.set_title("Per-subevent Pearson r")
    ax.legend()

    # (1) Overlaid histograms of event RMSE
    ax = axes[1]
    all_rmse = np.concatenate([mc_rmse_events, r3_rmse_events])
    bins_rmse = np.linspace(0, np.percentile(all_rmse, 99), 50)
    ax.hist(
        mc_rmse_events,
        bins=bins_rmse,
        color=COL_ANALYTICAL,
        alpha=0.6,
        label="MC",
        edgecolor="white",
    )
    ax.hist(
        r3_rmse_events,
        bins=bins_rmse,
        color=COL_MODEL,
        alpha=0.6,
        label="Run 3",
        edgecolor="white",
    )
    ax.set_xlabel("Event RMSE")
    ax.set_ylabel("Count")
    ax.set_title("Event-level RMSE")
    ax.legend()

    # (2) Box plots of integral ratio
    ax = axes[2]
    bp = ax.boxplot(
        [mc_integral, r3_integral],
        labels=["MC", "Run 3"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor(COL_ANALYTICAL)
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(COL_MODEL)
    bp["boxes"][1].set_alpha(0.6)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Integral ratio")
    ax.set_title("Integral Ratio")

    fig.suptitle(
        "MC vs Run 3 \u2014 Model Agreement Degradation",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    stem = "mc_vs_run3_comparison"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
