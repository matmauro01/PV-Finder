"""
Plotting functions for MC vs Run 3 feature distribution comparison -- Part 1.

Contains Figures 1--3:
    - plot_core_track_parameters()  (Figure 1)
    - plot_2d_correlations()        (Figure 2)
    - plot_tensor_distributions()   (Figure 3)

Extracted verbatim from compare_feature_distributions.py.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

from scipy.stats import ks_2samp

from pv_finder.data.feature_loading import (
    Z_MIN,
    Z_MAX,
    N_SUBEVENTS,
    CHANNEL_NAMES,
    CHANNEL_SHORT,
    N_FEATURES,
    safe_percentile,
)

# ---------------------------------------------------------------------------
# Matplotlib configuration (publication-quality)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
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
})

# Colour palette
COL_MC = "#4c72b0"
COL_R3 = "#dd8452"


# ============================================================================
# Utility helpers
# ============================================================================

def save_fig(fig, output_dir, name):
    """Save figure as both PNG and PDF."""
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ks_annotation(ax, mc_vals, r3_vals, x=0.03, y=0.95):
    """Run a two-sample KS test and annotate the result on *ax*.

    Returns (statistic, p-value) or (nan, nan) if inputs are too small.
    """
    if len(mc_vals) < 2 or len(r3_vals) < 2:
        return np.nan, np.nan
    stat, pval = ks_2samp(mc_vals, r3_vals)
    ax.text(
        x, y,
        f"KS = {stat:.3f}\np = {pval:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    return stat, pval


# ============================================================================
# Figure 1: Core Track Parameters (3x2)
# ============================================================================

def plot_core_track_parameters(mc_feats, r3_feats, output_dir):
    """Figure 1: 4x2 grid comparing d0, z0, d0_err, z0_err, d0_z0_cov,
    track count, and log-scale views of d0 and d0_z0_cov."""
    print("\n  [Figure 1] Core Track Parameters ...")

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    ks_results = {}

    # Feature definitions: (mc_array, r3_array, xlabel, name, log_y, log_x)
    features = [
        (mc_feats["d0"],       r3_feats["d0"],       "d0 [mm]",           "d0",       False, False),
        (mc_feats["z0"],       r3_feats["z0"],       "z0 [mm]",           "z0",       False, False),
        (mc_feats["d0_err"],   r3_feats["d0_err"],   "d0_err [mm]",       "d0_err",   True,  True),
        (mc_feats["z0_err"],   r3_feats["z0_err"],   "z0_err [mm]",       "z0_err",   True,  True),
        (mc_feats["d0_z0_cov"],r3_feats["d0_z0_cov"],"d0_z0_cov [mm^2]", "d0_z0_cov",True,  False),
    ]

    for idx, (mc_v, r3_v, xlabel, name, use_log_y, use_log_x) in enumerate(features):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        if len(mc_v) == 0 and len(r3_v) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        combined = np.concatenate([mc_v, r3_v])

        if use_log_x and np.all(combined > 0):
            # Log-spaced bins for positive-only features
            lo, hi = safe_percentile(combined, [0.5, 99.5])
            lo = max(lo, 1e-4)
            bins = np.logspace(np.log10(lo), np.log10(hi), 80)
            ax.set_xscale("log")
        else:
            lo, hi = safe_percentile(combined, [1, 99])
            bins = np.linspace(lo, hi, 80)

        ax.hist(mc_v, bins=bins, alpha=0.6, color=COL_MC, density=True,
                label=f"MC (n={len(mc_v):,})", edgecolor="none")
        ax.hist(r3_v, bins=bins, alpha=0.6, color=COL_R3, density=True,
                label=f"Run3 (n={len(r3_v):,})", edgecolor="none")

        if use_log_y:
            ax.set_yscale("log")

        # Add mean lines
        if len(mc_v) > 0:
            ax.axvline(np.mean(mc_v), color=COL_MC, linestyle="--",
                       linewidth=1.2, alpha=0.7)
        if len(r3_v) > 0:
            ax.axvline(np.mean(r3_v), color=COL_R3, linestyle="--",
                       linewidth=1.2, alpha=0.7)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(f"{name} distribution", fontsize=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")

        stat, pval = ks_annotation(ax, mc_v, r3_v)
        ks_results[name] = {"statistic": float(stat), "p_value": float(pval)}

    # Panel (2,1): track count per sub-event
    ax = axes[2, 1]
    mc_tc = mc_feats["track_counts"]
    r3_tc = r3_feats["track_counts"]
    mc_tc_nz = mc_tc[mc_tc > 0]
    r3_tc_nz = r3_tc[r3_tc > 0]
    max_tc = max(
        int(np.max(mc_tc_nz)) if len(mc_tc_nz) > 0 else 1,
        int(np.max(r3_tc_nz)) if len(r3_tc_nz) > 0 else 1,
    )
    bins = np.arange(0, min(max_tc + 2, 200), 1)
    ax.hist(mc_tc_nz, bins=bins, alpha=0.6, color=COL_MC, density=True,
            label=f"MC (med={np.median(mc_tc_nz):.0f})", edgecolor="none")
    ax.hist(r3_tc_nz, bins=bins, alpha=0.6, color=COL_R3, density=True,
            label=f"Run3 (med={np.median(r3_tc_nz):.0f})", edgecolor="none")
    ax.set_xlabel("Track count per sub-event", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Track multiplicity (non-empty sub-events)", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    stat, pval = ks_annotation(ax, mc_tc_nz.astype(float), r3_tc_nz.astype(float))
    ks_results["track_count"] = {"statistic": float(stat), "p_value": float(pval)}

    # Panel (3,0): d0 log-scale view (to see tails)
    ax = axes[3, 0]
    mc_v = mc_feats["d0"]
    r3_v = r3_feats["d0"]
    combined = np.concatenate([mc_v, r3_v])
    lo, hi = safe_percentile(combined, [0.1, 99.9])
    bins = np.linspace(lo, hi, 100)
    ax.hist(mc_v, bins=bins, alpha=0.6, color=COL_MC, density=True,
            label="MC", edgecolor="none")
    ax.hist(r3_v, bins=bins, alpha=0.6, color=COL_R3, density=True,
            label="Run3", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("d0 [mm]", fontsize=14)
    ax.set_ylabel("Density (log)", fontsize=14)
    ax.set_title("d0 distribution (log scale -- tail comparison)", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Panel (3,1): d0_z0_cov log-scale view (to see tails)
    ax = axes[3, 1]
    mc_v = mc_feats["d0_z0_cov"]
    r3_v = r3_feats["d0_z0_cov"]
    combined = np.concatenate([mc_v, r3_v])
    lo, hi = safe_percentile(combined, [0.1, 99.9])
    bins = np.linspace(lo, hi, 100)
    ax.hist(mc_v, bins=bins, alpha=0.6, color=COL_MC, density=True,
            label=f"MC (std={np.std(mc_v):.4f})", edgecolor="none")
    ax.hist(r3_v, bins=bins, alpha=0.6, color=COL_R3, density=True,
            label=f"Run3 (std={np.std(r3_v):.4f})", edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("d0_z0_cov [mm^2]", fontsize=14)
    ax.set_ylabel("Density (log)", fontsize=14)
    ax.set_title("d0_z0_cov (log scale -- MC has 3x heavier tails)", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Figure 1: Core Track Parameter Distributions (MC vs Run 3)",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_core_track_parameters")
    print("    Saved: fig1_core_track_parameters.png/.pdf")

    return ks_results


# ============================================================================
# Figure 2: 2D Correlations (2x3)
# ============================================================================

def plot_2d_correlations(mc_feats, r3_feats, output_dir):
    """Figure 2: 2x3 hexbin grid of 2D feature correlations."""
    print("\n  [Figure 2] 2D Correlations ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Each column is one pair of features; top = MC, bottom = Run3
    pairs = [
        ("z0", "d0", "z0 [mm]", "d0 [mm]", "d0 vs z0"),
        ("d0_err", "z0_err", "d0_err [mm]", "z0_err [mm]", "d0_err vs z0_err"),
        ("d0", "d0_err", "d0 [mm]", "d0_err [mm]", "d0 vs d0_err"),
    ]

    for col, (x_key, y_key, xlabel, ylabel, title_base) in enumerate(pairs):
        # Determine plot ranges from both datasets combined
        mc_x = mc_feats[x_key]
        mc_y = mc_feats[y_key]
        r3_x = r3_feats[x_key]
        r3_y = r3_feats[y_key]

        all_x = np.concatenate([mc_x, r3_x]) if len(mc_x) > 0 and len(r3_x) > 0 else mc_x if len(mc_x) > 0 else r3_x
        all_y = np.concatenate([mc_y, r3_y]) if len(mc_y) > 0 and len(r3_y) > 0 else mc_y if len(mc_y) > 0 else r3_y

        x_lo, x_hi = safe_percentile(all_x, [1, 99])
        y_lo, y_hi = safe_percentile(all_y, [1, 99])
        extent = [x_lo, x_hi, y_lo, y_hi]

        for row, (data_x, data_y, label, cmap) in enumerate([
            (mc_x, mc_y, "MC", "Blues"),
            (r3_x, r3_y, "Run 3", "Oranges"),
        ]):
            ax = axes[row, col]
            if len(data_x) > 0:
                hb = ax.hexbin(data_x, data_y, gridsize=60, cmap=cmap,
                               extent=extent, mincnt=1, norm=LogNorm())
                plt.colorbar(hb, ax=ax, label="count")
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(f"{label}: {title_base}", fontsize=15)

    fig.suptitle("Figure 2: 2D Feature Correlations (MC vs Run 3)",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig2_2d_correlations")
    print("    Saved: fig2_2d_correlations.png/.pdf")


# ============================================================================
# Figure 3: Model Input Tensor Distributions (4x2)
# ============================================================================

def plot_tensor_distributions(mc_channels, r3_channels, output_dir):
    """Figure 3: one panel per channel (0-6) plus OOD fraction bar chart."""
    print("\n  [Figure 3] Model Input Tensor Distributions ...")

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    ks_results = {}
    ood_fractions = {}
    mc_ranges = {}

    # Compute OOD fractions first (Run3 values outside MC [p1, p99])
    for ch in range(N_FEATURES):
        mc_v = mc_channels[ch]
        r3_v = r3_channels[ch]
        if len(mc_v) < 10:
            ood_fractions[ch] = 0.0
            mc_ranges[ch] = (0.0, 0.0)
            continue
        p1, p99 = np.percentile(mc_v, [1, 99])
        mc_ranges[ch] = (float(p1), float(p99))
        if len(r3_v) > 0:
            outside = np.sum((r3_v < p1) | (r3_v > p99))
            ood_fractions[ch] = float(outside) / len(r3_v)
        else:
            ood_fractions[ch] = 0.0

    # Plot each channel
    for ch in range(N_FEATURES):
        row, col = divmod(ch, 2)
        ax = axes[row, col]

        mc_v = mc_channels[ch]
        r3_v = r3_channels[ch]
        name = CHANNEL_NAMES[ch]

        if len(mc_v) == 0 and len(r3_v) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        combined = np.concatenate([mc_v, r3_v])
        lo, hi = np.percentile(combined, [0.5, 99.5])
        bins = np.linspace(lo, hi, 80)

        ax.hist(mc_v, bins=bins, alpha=0.6, color=COL_MC, density=True,
                label="MC", edgecolor="none")
        ax.hist(r3_v, bins=bins, alpha=0.6, color=COL_R3, density=True,
                label="Run3", edgecolor="none")

        # Shade MC [p1, p99] range
        p1, p99 = mc_ranges[ch]
        ax.axvspan(p1, p99, alpha=0.1, color=COL_MC, label="MC [p1,p99]")

        # Annotate OOD fraction
        ood = ood_fractions[ch]
        ax.text(0.97, 0.95,
                f"Run3 OOD: {ood:.1%}",
                transform=ax.transAxes,
                fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="lightyellow", alpha=0.85))

        ax.set_xlabel(f"Channel {ch}: {name}", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(f"Ch{ch}: {CHANNEL_SHORT[ch]}", fontsize=15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")

        stat, pval = ks_annotation(ax, mc_v, r3_v, x=0.03, y=0.80)
        ks_results[f"ch{ch}_{CHANNEL_SHORT[ch]}"] = {
            "statistic": float(stat), "p_value": float(pval),
        }

    # Last panel: OOD fraction bar chart
    ax = axes[3, 1]
    x_pos = np.arange(N_FEATURES)
    ood_vals = [ood_fractions[ch] for ch in range(N_FEATURES)]
    bars = ax.bar(x_pos, ood_vals, alpha=0.8, edgecolor="white")

    for bar, val in zip(bars, ood_vals):
        if val > 0.20:
            bar.set_facecolor("#c44e52")
        elif val > 0.10:
            bar.set_facecolor("#dd8452")
        else:
            bar.set_facecolor("#55a868")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CHANNEL_SHORT, fontsize=10)
    ax.set_ylabel("OOD fraction", fontsize=14)
    ax.set_title("Run 3 Out-of-Distribution fraction per channel", fontsize=15)
    ax.set_ylim(0, max(max(ood_vals) * 1.3, 0.05))
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    for bar, val in zip(bars, ood_vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "Figure 3: Model Input Tensor Distributions (MC vs Run 3)\n"
        "CORRECT mapping: Ch0=d0, Ch1=z0, Ch2=d0_err, Ch3=z0_err, "
        "Ch4=d0_z0_cov, Ch5=z_start, Ch6=z_end",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_tensor_distributions")
    print("    Saved: fig3_tensor_distributions.png/.pdf")

    return ks_results, ood_fractions, mc_ranges
