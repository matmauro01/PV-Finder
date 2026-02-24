"""
Plotting functions for MC vs Run 3 feature distribution comparison -- Part 1.

Contains Figures 1--2:
    - plot_feature_distributions()   (Figure 1: consolidated 3x2)
    - plot_2d_correlations()         (Figure 2)
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import ks_2samp

from pv_finder.data.feature_loading import (
    safe_percentile,
)

# ---------------------------------------------------------------------------
# Matplotlib configuration (publication-quality)
# ---------------------------------------------------------------------------
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
        x,
        y,
        f"KS = {stat:.3f}\np = {pval:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    return stat, pval


def _compute_ood(mc_vals, r3_vals):
    """Compute OOD fraction: Run3 values outside MC [p1, p99]."""
    if len(mc_vals) < 10 or len(r3_vals) == 0:
        return 0.0, (0.0, 0.0)
    p1, p99 = np.percentile(mc_vals, [1, 99])
    outside = np.sum((r3_vals < p1) | (r3_vals > p99))
    return float(outside) / len(r3_vals), (float(p1), float(p99))


# ============================================================================
# Figure 1: Consolidated Feature Distributions (3x2)
# ============================================================================


def plot_feature_distributions(mc_feats, r3_feats, output_dir):
    """Figure 1: 3x2 grid — d0, z0, d0_err, z0_err, d0_z0_cov, track count.

    Error features use linear x, log y. Each panel has KS + OOD annotation.

    Returns (ks_results, ood_fractions, mc_ranges).
    """
    print("\n  [Figure 1] Feature Distributions ...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    ks_results = {}
    ood_fractions = {}
    mc_ranges = {}

    # Feature definitions: (key, xlabel, name, use_log_y)
    features = [
        ("d0", "d0 [mm]", "d0", False),
        ("z0", "z0 [mm]", "z0", False),
        ("d0_err", "d0_err [mm]", "d0_err", True),
        ("z0_err", "z0_err [mm]", "z0_err", True),
        ("d0_z0_cov", "d0_z0_cov [mm^2]", "d0_z0_cov", True),
    ]

    for idx, (key, xlabel, name, use_log_y) in enumerate(features):
        row, col = divmod(idx, 2)
        ax = axes[row, col]

        mc_v = mc_feats[key]
        r3_v = r3_feats[key]

        if len(mc_v) == 0 and len(r3_v) == 0:
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center"
            )
            continue

        combined = np.concatenate([mc_v, r3_v])
        lo, hi = safe_percentile(combined, [1, 99])
        bins = np.linspace(lo, hi, 80)

        ax.hist(
            mc_v,
            bins=bins,
            alpha=0.6,
            color=COL_MC,
            density=True,
            label=f"MC (n={len(mc_v):,})",
            edgecolor="none",
        )
        ax.hist(
            r3_v,
            bins=bins,
            alpha=0.6,
            color=COL_R3,
            density=True,
            label=f"Run3 (n={len(r3_v):,})",
            edgecolor="none",
        )

        if use_log_y:
            ax.set_yscale("log")

        # Mean lines
        if len(mc_v) > 0:
            ax.axvline(
                np.mean(mc_v), color=COL_MC, linestyle="--", linewidth=1.2, alpha=0.7
            )
        if len(r3_v) > 0:
            ax.axvline(
                np.mean(r3_v), color=COL_R3, linestyle="--", linewidth=1.2, alpha=0.7
            )

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_title(f"{name} distribution", fontsize=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")

        # KS annotation
        stat, pval = ks_annotation(ax, mc_v, r3_v)
        ks_results[name] = {"statistic": float(stat), "p_value": float(pval)}

        # OOD annotation
        ood, (p1, p99) = _compute_ood(mc_v, r3_v)
        ood_fractions[name] = ood
        mc_ranges[name] = (p1, p99)
        ax.text(
            0.97,
            0.95,
            f"Run3 OOD: {ood:.1%}",
            transform=ax.transAxes,
            fontsize=9,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
        )

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
    ax.hist(
        mc_tc_nz,
        bins=bins,
        alpha=0.6,
        color=COL_MC,
        density=True,
        label=f"MC (med={np.median(mc_tc_nz):.0f})",
        edgecolor="none",
    )
    ax.hist(
        r3_tc_nz,
        bins=bins,
        alpha=0.6,
        color=COL_R3,
        density=True,
        label=f"Run3 (med={np.median(r3_tc_nz):.0f})",
        edgecolor="none",
    )
    ax.set_xlabel("Track count per sub-event", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Track multiplicity (non-empty sub-events)", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    stat, pval = ks_annotation(ax, mc_tc_nz.astype(float), r3_tc_nz.astype(float))
    ks_results["track_count"] = {"statistic": float(stat), "p_value": float(pval)}

    fig.suptitle("Figure 1: Feature Distributions (MC vs Run 3)", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig1_feature_distributions")
    print("    Saved: fig1_feature_distributions.png/.pdf")

    return ks_results, ood_fractions, mc_ranges


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

        all_x = (
            np.concatenate([mc_x, r3_x])
            if len(mc_x) > 0 and len(r3_x) > 0
            else mc_x
            if len(mc_x) > 0
            else r3_x
        )
        all_y = (
            np.concatenate([mc_y, r3_y])
            if len(mc_y) > 0 and len(r3_y) > 0
            else mc_y
            if len(mc_y) > 0
            else r3_y
        )

        x_lo, x_hi = safe_percentile(all_x, [1, 99])
        y_lo, y_hi = safe_percentile(all_y, [1, 99])
        extent = [x_lo, x_hi, y_lo, y_hi]

        for row, (data_x, data_y, label, cmap) in enumerate(
            [
                (mc_x, mc_y, "MC", "Blues"),
                (r3_x, r3_y, "Run 3", "Oranges"),
            ]
        ):
            ax = axes[row, col]
            if len(data_x) > 0:
                hb = ax.hexbin(
                    data_x,
                    data_y,
                    gridsize=60,
                    cmap=cmap,
                    extent=extent,
                    mincnt=1,
                    norm=LogNorm(),
                )
                plt.colorbar(hb, ax=ax, label="count")
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(f"{label}: {title_base}", fontsize=15)

    fig.suptitle("Figure 2: 2D Feature Correlations (MC vs Run 3)", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, output_dir, "fig2_2d_correlations")
    print("    Saved: fig2_2d_correlations.png/.pdf")
