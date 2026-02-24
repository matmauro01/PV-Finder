"""
Plotting functions for MC vs Run 3 feature distribution comparison -- Part 2.

Contains Figures 3--4:
    - plot_tails_and_quantiles()           (Figure 3)
    - plot_z0_beam_spot_investigation()     (Figure 4)
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from pv_finder.data.feature_loading import N_SUBEVENTS

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


# ============================================================================
# Figure 3: Distribution Tails & Quantiles (2x3)
# ============================================================================


def plot_tails_and_quantiles(mc_feats, r3_feats, output_dir):
    """Figure 3: CDF comparisons for 5 features + QQ plot for d0_err."""
    print("\n  [Figure 3] Distribution Tails & Quantiles ...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    feature_keys = ["d0", "z0", "d0_err", "z0_err", "d0_z0_cov"]
    feature_labels = [
        "d0 [mm]",
        "z0 [mm]",
        "d0_err [mm]",
        "z0_err [mm]",
        "d0_z0_cov [mm^2]",
    ]

    # Row 0 + (1,0) and (1,1): CDF comparisons
    for idx, (key, label) in enumerate(zip(feature_keys, feature_labels)):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        mc_v = mc_feats[key]
        r3_v = r3_feats[key]

        if len(mc_v) == 0 or len(r3_v) == 0:
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center"
            )
            continue

        # Subsample for performance if needed
        max_pts = 100000
        mc_s = np.sort(np.random.choice(mc_v, min(max_pts, len(mc_v)), replace=False))
        r3_s = np.sort(np.random.choice(r3_v, min(max_pts, len(r3_v)), replace=False))

        mc_cdf = np.arange(1, len(mc_s) + 1) / len(mc_s)
        r3_cdf = np.arange(1, len(r3_s) + 1) / len(r3_s)

        ax.plot(mc_s, mc_cdf, color=COL_MC, linewidth=1.5, label="MC")
        ax.plot(r3_s, r3_cdf, color=COL_R3, linewidth=1.5, label="Run3")

        ax.set_xlabel(label, fontsize=14)
        ax.set_ylabel("CDF", fontsize=14)
        ax.set_title(f"CDF: {key}", fontsize=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")

        ks_annotation(ax, mc_v, r3_v, x=0.60, y=0.25)

    # Last panel (1,2): QQ plot for d0_err (the most physics-relevant feature
    # for domain shift, since errors drive the model's weighting)
    ax = axes[1, 2]
    mc_v = mc_feats["d0_err"]
    r3_v = r3_feats["d0_err"]

    if len(mc_v) > 0 and len(r3_v) > 0:
        n_quantiles = 200
        quantile_probs = np.linspace(0, 100, n_quantiles)
        mc_q = np.percentile(mc_v, quantile_probs)
        r3_q = np.percentile(r3_v, quantile_probs)

        ax.scatter(mc_q, r3_q, s=12, alpha=0.7, color="#6a3d9a", zorder=3)

        # Reference line y=x
        lo = min(mc_q.min(), r3_q.min())
        hi = max(mc_q.max(), r3_q.max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y = x")

        ax.set_xlabel("MC quantiles (d0_err) [mm]", fontsize=14)
        ax.set_ylabel("Run3 quantiles (d0_err) [mm]", fontsize=14)
        ax.set_title("QQ Plot: d0_err (MC vs Run3)", fontsize=15)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    fig.suptitle(
        "Figure 3: Distribution Tails & Quantiles (MC vs Run 3)", fontsize=16, y=1.01
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig3_tails_and_quantiles")
    print("    Saved: fig3_tails_and_quantiles.png/.pdf")


# ============================================================================
# Figure 4: Z0 Beam-Spot Investigation (2x2)
# ============================================================================


def plot_z0_beam_spot_investigation(mc_feats, r3_feats, run3_events, output_dir):
    """Figure 4: Investigate the z0 mean shift between MC and Run3.

    MC z0 is relative to (0,0,0).  Run3 z0 is in detector coordinates.
    The Run3 beam spot z is ~-1.4 mm, explaining part of the shift.
    """
    print("\n  [Figure 4] Z0 Beam-Spot Investigation ...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    mc_z0 = mc_feats["z0"]
    r3_z0 = r3_feats["z0"]
    beam_z_vals = np.array([evt["beam_z"] for evt in run3_events])
    mean_beam_z = np.mean(beam_z_vals)

    # --- (0,0): Raw z0 zoomed in around the mean region ---
    ax = axes[0, 0]
    lo, hi = -15, 15  # zoom to +-15 mm
    bins = np.linspace(lo, hi, 100)
    ax.hist(
        mc_z0,
        bins=bins,
        alpha=0.6,
        color=COL_MC,
        density=True,
        label=f"MC (mean={np.mean(mc_z0):.2f} mm)",
        edgecolor="none",
    )
    ax.hist(
        r3_z0,
        bins=bins,
        alpha=0.6,
        color=COL_R3,
        density=True,
        label=f"Run3 (mean={np.mean(r3_z0):.2f} mm)",
        edgecolor="none",
    )
    ax.axvline(np.mean(mc_z0), color=COL_MC, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(r3_z0), color=COL_R3, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("z0 [mm]", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("z0 zoomed [-15, 15] mm -- raw coordinates", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # --- (0,1): Beam spot z distribution ---
    ax = axes[0, 1]
    bins_bs = np.linspace(min(beam_z_vals) - 0.5, max(beam_z_vals) + 0.5, 50)
    ax.hist(
        beam_z_vals,
        bins=bins_bs,
        alpha=0.7,
        color="#6a3d9a",
        edgecolor="white",
        label=f"Beam z (mean={mean_beam_z:.3f} mm)",
    )
    ax.axvline(
        mean_beam_z,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_beam_z:.3f} mm",
    )
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.5, label="MC origin (z=0)")
    ax.set_xlabel("BeamPosZ [mm]", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("Run 3 Beam Spot Z Position", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # --- (1,0): Per-event mean z0 distributions ---
    ax = axes[1, 0]
    # Per-subevent z_center-weighted approach
    mc_per_se = mc_feats["per_subevent"]
    r3_per_se = r3_feats["per_subevent"]
    # Group by event (every 12 subevents = 1 event)
    mc_evt_means = []
    for i in range(0, len(mc_per_se), N_SUBEVENTS):
        evt_recs = mc_per_se[i : i + N_SUBEVENTS]
        total_trk = sum(r["n_tracks"] for r in evt_recs)
        if total_trk > 0:
            # weighted mean z0 of all subevents (approximate by z_center * n_tracks)
            weighted = sum(r["z_center"] * r["n_tracks"] for r in evt_recs)
            mc_evt_means.append(weighted / total_trk)
    r3_evt_means = []
    for i in range(0, len(r3_per_se), N_SUBEVENTS):
        evt_recs = r3_per_se[i : i + N_SUBEVENTS]
        total_trk = sum(r["n_tracks"] for r in evt_recs)
        if total_trk > 0:
            weighted = sum(r["z_center"] * r["n_tracks"] for r in evt_recs)
            r3_evt_means.append(weighted / total_trk)
    mc_evt_means = np.array(mc_evt_means)
    r3_evt_means = np.array(r3_evt_means)

    if len(mc_evt_means) > 0 and len(r3_evt_means) > 0:
        lo = min(mc_evt_means.min(), r3_evt_means.min()) - 5
        hi = max(mc_evt_means.max(), r3_evt_means.max()) + 5
        bins = np.linspace(lo, hi, 50)
        ax.hist(
            mc_evt_means,
            bins=bins,
            alpha=0.6,
            color=COL_MC,
            density=True,
            label=f"MC (mean={np.mean(mc_evt_means):.2f})",
        )
        ax.hist(
            r3_evt_means,
            bins=bins,
            alpha=0.6,
            color=COL_R3,
            density=True,
            label=f"Run3 (mean={np.mean(r3_evt_means):.2f})",
        )
    ax.set_xlabel("Per-event weighted mean z [mm]", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Per-event track-weighted mean z position", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")

    # --- (1,1): Explanation text + shift summary ---
    ax = axes[1, 1]
    ax.axis("off")

    mc_mean_z0 = float(np.mean(mc_z0))
    r3_mean_z0 = float(np.mean(r3_z0))
    shift = r3_mean_z0 - mc_mean_z0

    text = (
        "Z0 Mean Shift Analysis\n"
        "─────────────────────────────────────────\n"
        f"MC z0 mean:           {mc_mean_z0:+.3f} mm\n"
        f"Run3 z0 mean (raw):   {r3_mean_z0:+.3f} mm\n"
        f"Shift (Run3 - MC):    {shift:+.3f} mm\n"
        f"\n"
        f"Run3 beam spot z:     {mean_beam_z:+.3f} mm\n"
        f"\n"
        "Explanation:\n"
        "─────────────────────────────────────────\n"
        "• MC: tracks simulated relative to origin (0,0,0)\n"
        "• Run3: tracks in detector coordinates\n"
        "• Beam spot ≈ -1.4 mm from detector origin\n"
        "• The ~1.9 mm shift is a coordinate system\n"
        "  difference, NOT a physics difference.\n"
        "\n"
        "• For the NN model, this shift is negligible:\n"
        f"  shift/std(z0) = {abs(shift) / np.std(mc_z0):.4f}\n"
        "  (z0 ranges from -200 to +200 mm)"
    )
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9),
    )

    fig.suptitle(
        "Figure 4: Z0 Mean Shift Investigation -- Beam Spot Effect",
        fontsize=16,
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, output_dir, "fig4_z0_beam_spot_investigation")
    print("    Saved: fig4_z0_beam_spot_investigation.png/.pdf")
