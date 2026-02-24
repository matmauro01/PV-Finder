#!/usr/bin/env python3
"""
Compare feature distributions between MC (training) and Run 3 (real ATLAS) data.

Produces 6 publication-quality figures and a JSON summary comparing raw track
parameters, 2D correlations, model-input tensors, distribution tails, and
per-subevent statistics for MC vs Run 3 data.

CRITICAL: The MC H5 feature channels are decoded as:
    Channel 0: d0        (RAW transverse impact parameter, mm)
    Channel 1: z0        (longitudinal impact parameter, mm)
    Channel 2: d0_err    (uncertainty on d0, mm) -- ALWAYS POSITIVE
    Channel 3: z0_err    (uncertainty on z0, mm) -- ALWAYS POSITIVE
    Channel 4: d0_z0_cov (covariance between d0 and z0, mm^2)
    Channel 5: z_start   (sub-event boundary, mm)
    Channel 6: z_end     (sub-event boundary, mm)

Previous scripts (investigate_domain_shift.py, compare_kde_theory_vs_model.py)
had CRITICAL bugs decoding channels 0/2/3 as d0/2, theta/3, (phi+pi)/3.
This script uses the CORRECT mapping confirmed by eval_run3_v2.py.

Usage:
    python compare_feature_distributions.py [--n-events 200]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Suppress harmless warnings during batch plot generation
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Imports from extracted modules
# ---------------------------------------------------------------------------
from pv_finder.data.feature_loading import (
    N_FEATURES,
    CHANNEL_SHORT,
    load_mc_data,
    load_run3_data,
    collect_features,
    collect_tensor_values,
    feature_stats,
)

from pv_finder.diagnostics.feature_plots_1 import (
    plot_core_track_parameters,
    plot_2d_correlations,
    plot_tensor_distributions,
)

from pv_finder.diagnostics.feature_plots_2 import (
    plot_tails_and_quantiles,
    plot_per_subevent_statistics,
    plot_z0_beam_spot_investigation,
)


# ============================================================================
# Statistical Summary
# ============================================================================

def compute_summary(mc_feats, r3_feats, ks_fig1, ks_fig3,
                    ood_fractions, mc_ranges, run3_events):
    """Compute and return the full statistical summary dict."""
    summary = {
        "script": "compare_feature_distributions.py",
        "correct_feature_mapping": {
            "channel_0": "d0 (RAW transverse impact parameter, mm)",
            "channel_1": "z0 (longitudinal impact parameter, mm)",
            "channel_2": "d0_err (uncertainty on d0, mm) -- ALWAYS POSITIVE",
            "channel_3": "z0_err (uncertainty on z0, mm) -- ALWAYS POSITIVE",
            "channel_4": "d0_z0_cov (covariance between d0 and z0, mm^2)",
            "channel_5": "z_start (sub-event boundary, mm)",
            "channel_6": "z_end (sub-event boundary, mm)",
        },
        "WARNING": (
            "Previous scripts decoded Ch0 as d0*2, Ch2 as theta/3, "
            "Ch3 as (phi+pi)/3. Those were WRONG. This script uses the "
            "correct mapping."
        ),
    }

    # Per-feature statistics
    feature_keys = ["d0", "z0", "d0_err", "z0_err", "d0_z0_cov"]
    per_feature = {}
    for key in feature_keys:
        per_feature[key] = {
            "mc": feature_stats(mc_feats[key]),
            "run3": feature_stats(r3_feats[key]),
        }
        # KS test
        mc_v = mc_feats[key]
        r3_v = r3_feats[key]
        if len(mc_v) >= 2 and len(r3_v) >= 2:
            stat, pval = ks_2samp(mc_v, r3_v)
            per_feature[key]["ks_statistic"] = float(stat)
            per_feature[key]["ks_p_value"] = float(pval)
        else:
            per_feature[key]["ks_statistic"] = None
            per_feature[key]["ks_p_value"] = None

    summary["per_feature"] = per_feature

    # Track count stats
    mc_tc = mc_feats["track_counts"]
    r3_tc = r3_feats["track_counts"]
    summary["track_counts"] = {
        "mc": feature_stats(mc_tc[mc_tc > 0].astype(float)),
        "run3": feature_stats(r3_tc[r3_tc > 0].astype(float)),
    }

    # KS test results from figures
    summary["ks_tests_fig1"] = ks_fig1
    summary["ks_tests_fig3"] = ks_fig3

    # OOD fractions
    summary["ood_fractions"] = {
        CHANNEL_SHORT[ch]: ood_fractions[ch] for ch in range(N_FEATURES)
    }
    summary["mc_ranges_p1_p99"] = {
        CHANNEL_SHORT[ch]: mc_ranges[ch] for ch in range(N_FEATURES)
    }

    # Beam spot positions from Run3
    beam_positions = [evt["beam_z"] for evt in run3_events]
    summary["run3_beam_spot"] = {
        "mean_z": float(np.mean(beam_positions)),
        "std_z": float(np.std(beam_positions)),
        "min_z": float(np.min(beam_positions)),
        "max_z": float(np.max(beam_positions)),
        "n_events": len(beam_positions),
    }

    return summary


def print_summary(summary):
    """Print a formatted summary table to stdout."""
    print("\n" + "=" * 76)
    print("  FEATURE DISTRIBUTION COMPARISON -- SUMMARY")
    print("=" * 76)

    print("\n  CORRECT Feature Mapping:")
    for ch_key, desc in summary["correct_feature_mapping"].items():
        print(f"    {ch_key}: {desc}")

    print(f"\n  WARNING: {summary['WARNING']}")

    print("\n  Per-Feature Statistics:")
    print(f"  {'Feature':<15s} {'Dataset':<8s} {'Mean':>10s} {'Std':>10s} "
          f"{'Median':>10s} {'Min':>10s} {'Max':>10s} {'N':>10s}")
    print(f"  {'-' * 73}")

    for key, data in summary["per_feature"].items():
        for ds_label in ["mc", "run3"]:
            s = data[ds_label]
            print(f"  {key:<15s} {ds_label:<8s} {s['mean']:>10.4f} "
                  f"{s['std']:>10.4f} {s['median']:>10.4f} "
                  f"{s['min']:>10.4f} {s['max']:>10.4f} {s['n']:>10,}")

    print("\n  KS Test Results (Figure 1):")
    print(f"  {'Feature':<20s} {'Statistic':>12s} {'p-value':>15s} {'Significant?':>14s}")
    print(f"  {'-' * 61}")
    for name, res in summary.get("ks_tests_fig1", {}).items():
        stat = res.get("statistic", float("nan"))
        pval = res.get("p_value", float("nan"))
        if np.isnan(pval):
            sig = "N/A"
        elif pval < 0.001:
            sig = "*** YES"
        elif pval < 0.01:
            sig = "**  YES"
        elif pval < 0.05:
            sig = "*   YES"
        else:
            sig = "    no"
        print(f"  {name:<20s} {stat:>12.4f} {pval:>15.4e} {sig:>14s}")

    print("\n  OOD Fractions (Run3 outside MC [p1, p99]):")
    print(f"  {'Channel':<20s} {'OOD fraction':>15s}")
    print(f"  {'-' * 35}")
    for ch_name, frac in summary.get("ood_fractions", {}).items():
        print(f"  {ch_name:<20s} {frac:>14.1%}")

    print("\n  Run 3 Beam Spot Position:")
    bs = summary.get("run3_beam_spot", {})
    print(f"    Mean z: {bs.get('mean_z', 0):.4f} mm")
    print(f"    Std z:  {bs.get('std_z', 0):.4f} mm")

    print("\n" + "=" * 76)


# ============================================================================
# JSON serialization helper
# ============================================================================

def _sanitize_for_json(obj):
    """Make numpy objects JSON-serializable."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    return str(obj)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare feature distributions between MC training data "
            "and Run 3 ATLAS data, using the CORRECT feature mapping."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run3-cache",
        default="data/run3/cache_file3_2000ev_seed42.npz",
        help="Path to Run 3 NPZ cache",
    )
    parser.add_argument(
        "--mc-h5",
        default="data/monte_carlo/training_data.h5",
        help="Path to MC HDF5 training data",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/domain_shift_investigation",
        help="Output directory for plots and JSON",
    )
    parser.add_argument(
        "--n-events", type=int, default=200,
        help="Number of events per dataset",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    base = Path(__file__).resolve().parent.parent.parent.parent
    run3_cache = base / args.run3_cache
    mc_h5 = base / args.mc_h5
    output_dir = str(base / args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # --- Banner ---
    print("=" * 76)
    print("  FEATURE DISTRIBUTION COMPARISON: MC vs Run 3")
    print("  (Using CORRECT feature mapping -- no theta/phi confusion)")
    print("=" * 76)
    print(f"  Run 3 cache  : {run3_cache}")
    print(f"  MC H5        : {mc_h5}")
    print(f"  Output dir   : {output_dir}")
    print(f"  N events     : {args.n_events}")
    print("=" * 76)

    # --- Validate input files ---
    for p, desc in [(run3_cache, "Run 3 cache"), (mc_h5, "MC H5")]:
        if not p.exists():
            print(f"ERROR: {desc} not found: {p}")
            sys.exit(1)

    # ===================================================================
    # Load data
    # ===================================================================
    print("\n--- Loading data ---")
    mc_events = load_mc_data(str(mc_h5), n_events=args.n_events)
    run3_events = load_run3_data(str(run3_cache), min_pileup=3,
                                  max_events=args.n_events)

    # ===================================================================
    # Collect features
    # ===================================================================
    print("\n--- Collecting features ---")
    mc_feats, r3_feats = collect_features(mc_events, run3_events)
    mc_channels, r3_channels = collect_tensor_values(mc_events, run3_events)

    # ===================================================================
    # Generate figures
    # ===================================================================
    print("\n--- Generating figures ---")

    ks_fig1 = plot_core_track_parameters(mc_feats, r3_feats, output_dir)
    plot_2d_correlations(mc_feats, r3_feats, output_dir)
    ks_fig3, ood_fractions, mc_ranges = plot_tensor_distributions(
        mc_channels, r3_channels, output_dir
    )
    plot_tails_and_quantiles(mc_feats, r3_feats, output_dir)
    plot_per_subevent_statistics(mc_feats, r3_feats, output_dir)
    plot_z0_beam_spot_investigation(mc_feats, r3_feats, run3_events, output_dir)

    # ===================================================================
    # Summary
    # ===================================================================
    print("\n--- Computing summary ---")
    summary = compute_summary(
        mc_feats, r3_feats, ks_fig1, ks_fig3,
        ood_fractions, mc_ranges, run3_events,
    )
    print_summary(summary)

    # Save JSON
    summary_json = _sanitize_for_json(summary)
    summary_path = os.path.join(output_dir, "feature_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"\nSaved summary JSON: {summary_path}")

    # ===================================================================
    # Final report
    # ===================================================================
    print("\n" + "=" * 76)
    print("  COMPLETE -- all outputs in: " + output_dir)
    print("=" * 76)
    print("  Figures:")
    for name in [
        "fig1_core_track_parameters",
        "fig2_2d_correlations",
        "fig3_tensor_distributions",
        "fig4_tails_and_quantiles",
        "fig5_per_subevent_statistics",
        "fig6_z0_beam_spot_investigation",
    ]:
        for ext in ("png", "pdf"):
            p = os.path.join(output_dir, f"{name}.{ext}")
            exists = "OK" if os.path.isfile(p) else "MISSING"
            print(f"    [{exists}] {p}")
    print(f"  Summary: {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
