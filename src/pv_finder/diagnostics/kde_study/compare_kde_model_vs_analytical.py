#!/usr/bin/env python3
"""
Compare T2KDE model predictions against analytically computed KDE on both
MC validation data and Run 3 ATLAS data.  Produces overlay plots, per-vertex
zooms, agreement summaries, residual distributions, and a JSON summary.

Usage:
    python -m pv_finder.diagnostics.kde_study.compare_kde_model_vs_analytical [options]
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import pearsonr

from pv_finder.data.feature_loading import (
    N_SUBEVENTS,
    Z_MAX,
    Z_MIN,
    load_mc_data,
    load_run3_data,
)
from pv_finder.diagnostics.kde_study.analytical_kde import compute_analytical_kdes_batch
from pv_finder.diagnostics.kde_study.kde_comparison_plots import (
    plot_agreement_summary,
    plot_event_overlay,
    plot_mc_vs_run3_comparison,
    plot_per_vertex,
    plot_residual_distributions,
)
from pv_finder.diagnostics.kde_study.kde_model_inference import (
    load_t2kde_model,
    run_model_on_events,
)

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# JSON helper (reused pattern from compare_feature_distributions.py)
# ---------------------------------------------------------------------------


def _sanitize_for_json(obj):
    """Make numpy objects JSON-serializable."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Peak finding
# ---------------------------------------------------------------------------


def _find_peaks_in_kde(kde_flat, threshold_frac=0.5, min_distance=50):
    """Return peak bin indices in a 1-D KDE array."""
    if len(kde_flat) == 0 or np.max(kde_flat) <= 0:
        return []
    peaks, _ = find_peaks(
        kde_flat, height=threshold_frac * np.max(kde_flat), distance=min_distance
    )
    return peaks.tolist()


def _bin_to_z(bin_idx, n_bins=12000):
    """Convert a full-range bin index to z (mm)."""
    return Z_MIN + (bin_idx + 0.5) / n_bins * (Z_MAX - Z_MIN)


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------


def compute_agreement_metrics(analytical, model_pred, peak_threshold=0.5, peak_tol=5):
    """Compute per-subevent and event-level agreement metrics."""
    ana = np.asarray(analytical, dtype=np.float64)
    mod = np.asarray(model_pred, dtype=np.float64)

    pearson_r_per_sub, mse_per_sub = [], []
    for si in range(N_SUBEVENTS):
        a, m = ana[si], mod[si]
        if np.std(a) < 1e-12 or np.std(m) < 1e-12:
            pearson_r_per_sub.append(np.nan)
        else:
            pearson_r_per_sub.append(float(pearsonr(a, m)[0]))
        mse_per_sub.append(float(np.mean((a - m) ** 2)))

    ana_flat, mod_flat = ana.reshape(-1), mod.reshape(-1)
    rmse_event = float(np.sqrt(np.mean((ana_flat - mod_flat) ** 2)))
    integral_ratio = float(np.sum(mod_flat) / (np.sum(ana_flat) + 1e-10))

    # Peak matching
    ana_peaks = _find_peaks_in_kde(ana_flat, threshold_frac=peak_threshold)
    mod_peaks = _find_peaks_in_kde(mod_flat, threshold_frac=peak_threshold)

    def _cluster(bins):
        if not bins:
            return []
        groups = [[bins[0]]]
        for b in bins[1:]:
            if b - groups[-1][-1] <= 1:
                groups[-1].append(b)
            else:
                groups.append([b])
        return groups

    ana_pos = [g[int(np.argmax(ana_flat[g]))] for g in _cluster(ana_peaks)]
    mod_pos = [g[int(np.argmax(mod_flat[g]))] for g in _cluster(mod_peaks)]

    matched_ana, matched_mod = set(), set()
    for mi, mp in enumerate(mod_pos):
        for ai, ap in enumerate(ana_pos):
            if ai not in matched_ana and abs(mp - ap) < peak_tol:
                matched_ana.add(ai)
                matched_mod.add(mi)
                break

    return {
        "pearson_r_per_sub": pearson_r_per_sub,
        "mse_per_sub": mse_per_sub,
        "rmse_event": rmse_event,
        "integral_ratio": integral_ratio,
        "n_peaks_analytical": len(ana_pos),
        "n_peaks_model": len(mod_pos),
        "n_peaks_matched": len(matched_ana),
        "n_peaks_missed": len(ana_pos) - len(matched_ana),
        "n_peaks_extra": len(mod_pos) - len(matched_mod),
    }


# ---------------------------------------------------------------------------
# Vertex finding
# ---------------------------------------------------------------------------


def find_vertices_mc(h5_path, event_indices, val_start_sub=428400):
    """Find MC truth vertex z-positions from the target histogram."""
    all_vtx = []
    with h5py.File(h5_path, "r") as f:
        ty = f["target_y_split"]
        for ei in event_indices:
            s = val_start_sub + ei * N_SUBEVENTS
            flat = ty[s : s + N_SUBEVENTS, 0, :].reshape(-1)
            all_vtx.append([_bin_to_z(b) for b in _find_peaks_in_kde(flat, 0.5, 50)])
    return all_vtx


def find_vertices_run3(analytical_kde):
    """Find vertex z-positions from peaks in the analytical KDE."""
    flat = np.asarray(analytical_kde).reshape(-1)
    return [_bin_to_z(b) for b in _find_peaks_in_kde(flat, 0.5, 50)]


# ---------------------------------------------------------------------------
# MC truth KDE loading
# ---------------------------------------------------------------------------


def load_mc_truth_kdes(h5_path, n_events, val_start_sub=428400):
    """Load ground-truth KDE_A_z from H5 validation split."""
    with h5py.File(h5_path, "r") as f:
        n_subs = n_events * N_SUBEVENTS
        raw = f["kde_split"][val_start_sub : val_start_sub + n_subs, 0, :]
    reshaped = raw.reshape(n_events, N_SUBEVENTS, -1)
    return [reshaped[i].astype(np.float64) for i in range(n_events)]


# ---------------------------------------------------------------------------
# Representative event selection
# ---------------------------------------------------------------------------


def select_representative_events(metrics_list, n=3):
    """Select best, median, and worst events by mean Pearson r."""
    mean_r = []
    for m in metrics_list:
        vals = [v for v in m["pearson_r_per_sub"] if np.isfinite(v)]
        mean_r.append(np.mean(vals) if vals else -1.0)
    order = np.argsort(mean_r)[::-1]
    picks = []
    if len(order) >= 1:
        picks.append(int(order[0]))
    if len(order) >= 2:
        picks.append(int(order[len(order) // 2]))
    if len(order) >= 3:
        picks.append(int(order[-1]))
    return picks[:n]


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def _dataset_summary(metrics_list):
    """Aggregate per-event metrics into dataset-level summary dict."""
    all_r, all_rmse, all_ir = [], [], []
    total_ana, total_matched = 0, 0
    for m in metrics_list:
        all_r.extend(v for v in m["pearson_r_per_sub"] if np.isfinite(v))
        all_rmse.append(m["rmse_event"])
        all_ir.append(m["integral_ratio"])
        total_ana += m["n_peaks_analytical"]
        total_matched += m["n_peaks_matched"]
    all_r = np.array(all_r) if all_r else np.array([0.0])
    return {
        "n_events": len(metrics_list),
        "mean_pearson_r": float(np.nanmean(all_r)),
        "std_pearson_r": float(np.nanstd(all_r)),
        "mean_rmse": float(np.mean(all_rmse)),
        "mean_integral_ratio": float(np.mean(all_ir)),
        "total_peaks_analytical": total_ana,
        "total_peaks_matched": total_matched,
        "peak_matching_rate": float(total_matched / max(total_ana, 1)),
        "per_event": metrics_list,
    }


def _print_summary_table(mc_metrics, r3_metrics):
    """Print a compact summary table to stdout."""

    def _show(metrics_list, label):
        all_r, all_rmse, all_ir = [], [], []
        t_ana, t_match = 0, 0
        for m in metrics_list:
            all_r.extend(v for v in m["pearson_r_per_sub"] if np.isfinite(v))
            all_rmse.append(m["rmse_event"])
            all_ir.append(m["integral_ratio"])
            t_ana += m["n_peaks_analytical"]
            t_match += m["n_peaks_matched"]
        r = np.array(all_r)
        print(f"\n  {label} ({len(metrics_list)} events)")
        print(
            f"    Pearson r  : mean={np.mean(r):.4f}  std={np.std(r):.4f}  "
            f"min={np.min(r):.4f}  max={np.max(r):.4f}"
        )
        print(
            f"    Event RMSE : mean={np.mean(all_rmse):.6f}  std={np.std(all_rmse):.6f}"
        )
        print(f"    Integral R : mean={np.mean(all_ir):.4f}  std={np.std(all_ir):.4f}")
        print(f"    Peak match : {t_match}/{t_ana} ({t_match / max(t_ana, 1):.1%})")

    print("\n" + "=" * 70)
    print("  KDE MODEL vs ANALYTICAL -- AGREEMENT SUMMARY")
    print("=" * 70)
    _show(mc_metrics, "MC (validation)")
    _show(r3_metrics, "Run 3")
    print("\n" + "=" * 70)


# ============================================================================
# Main
# ============================================================================


def main():
    ap = argparse.ArgumentParser(
        description="Compare T2KDE model predictions vs analytical KDE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mc-h5", default="data/monte_carlo/training_data.h5")
    ap.add_argument("--run3-cache", default="data/run3/cache_file3_2000ev_seed42.npz")
    ap.add_argument(
        "--model-path", default="model_weights/tracks2kde_KDE_A_z_epoch180.pyt"
    )
    ap.add_argument("--output-dir", default="outputs/kde_comparison")
    ap.add_argument("--n-events", type=int, default=200)
    ap.add_argument("--n-viz-events", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent.parent.parent
    mc_h5 = base / args.mc_h5
    run3_cache = base / args.run3_cache
    model_path = base / args.model_path
    output_dir = str(base / args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  KDE MODEL vs ANALYTICAL COMPARISON")
    print("=" * 70)
    print(f"  MC H5        : {mc_h5}")
    print(f"  Run 3 cache  : {run3_cache}")
    print(f"  Model        : {model_path}")
    print(f"  Output dir   : {output_dir}")
    print(f"  N events     : {args.n_events}")
    print(f"  Device       : {args.device}")
    print("=" * 70)

    for p, desc in [
        (mc_h5, "MC H5"),
        (run3_cache, "Run 3 cache"),
        (model_path, "Model weights"),
    ]:
        if not p.exists():
            print(f"ERROR: {desc} not found: {p}")
            sys.exit(1)

    # 1. Load data
    print("\n--- Loading data ---")
    mc_events = load_mc_data(str(mc_h5), n_events=args.n_events)
    run3_events = load_run3_data(
        str(run3_cache), min_pileup=3, max_events=args.n_events
    )
    n_mc, n_r3 = len(mc_events), len(run3_events)

    # 2. Load model
    print("\n--- Loading T2KDE model ---")
    model = load_t2kde_model(str(model_path), device=args.device)
    print("  Model loaded successfully.")

    # 3. Model inference
    print("\n--- Running model inference ---")
    mc_mod = run_model_on_events(
        model, mc_events, "mc", args.device, batch_size=args.batch_size
    )
    r3_mod = run_model_on_events(
        model, run3_events, "run3", args.device, batch_size=args.batch_size
    )

    # 4. Analytical KDEs
    print("\n--- Computing analytical KDEs ---")
    mc_ana = compute_analytical_kdes_batch(mc_events, "mc")
    r3_ana = compute_analytical_kdes_batch(run3_events, "run3")

    # 5. MC truth KDEs
    print("\n--- Loading MC truth KDEs from H5 ---")
    mc_truth = load_mc_truth_kdes(str(mc_h5), n_mc)

    # 6. Agreement metrics
    print("\n--- Computing agreement metrics ---")
    mc_met = [compute_agreement_metrics(mc_ana[i], mc_mod[i]) for i in range(n_mc)]
    r3_met = [compute_agreement_metrics(r3_ana[i], r3_mod[i]) for i in range(n_r3)]

    # 7. Representative events
    mc_repr = select_representative_events(mc_met, n=args.n_viz_events)
    r3_repr = select_representative_events(r3_met, n=args.n_viz_events)
    print(f"  MC representative events  : {mc_repr}")
    print(f"  Run3 representative events: {r3_repr}")

    # 8. Vertex positions
    print("\n--- Finding vertex positions ---")
    mc_vtx = find_vertices_mc(str(mc_h5), mc_repr)
    r3_vtx = [find_vertices_run3(r3_ana[i]) for i in r3_repr]

    # 9. Plots
    print("\n--- Generating plots ---")
    mc_ov = os.path.join(output_dir, "mc_event_overlays")
    r3_ov = os.path.join(output_dir, "run3_event_overlays")
    mc_pvd = os.path.join(output_dir, "mc_per_vertex")
    r3_pvd = os.path.join(output_dir, "run3_per_vertex")

    for idx, ei in enumerate(mc_repr):
        plot_event_overlay(
            mc_ana[ei], mc_mod[ei], ei, mc_vtx[idx], mc_ov, "mc", truth_kde=mc_truth[ei]
        )
    for idx, ei in enumerate(r3_repr):
        plot_event_overlay(r3_ana[ei], r3_mod[ei], ei, r3_vtx[idx], r3_ov, "run3")

    for idx, ei in enumerate(mc_repr):
        for vi, vz in enumerate(mc_vtx[idx]):
            plot_per_vertex(
                mc_ana[ei], mc_mod[ei], vz, vi, ei, mc_pvd, "mc", truth_kde=mc_truth[ei]
            )
    for idx, ei in enumerate(r3_repr):
        for vi, vz in enumerate(r3_vtx[idx]):
            plot_per_vertex(r3_ana[ei], r3_mod[ei], vz, vi, ei, r3_pvd, "run3")

    plot_agreement_summary(mc_met, "mc", output_dir)
    plot_agreement_summary(r3_met, "run3", output_dir)

    mc_res = [
        (np.asarray(mc_mod[i]) - np.asarray(mc_ana[i])).reshape(-1) for i in range(n_mc)
    ]
    r3_res = [
        (np.asarray(r3_mod[i]) - np.asarray(r3_ana[i])).reshape(-1) for i in range(n_r3)
    ]
    plot_residual_distributions(mc_res, "mc", output_dir)
    plot_residual_distributions(r3_res, "run3", output_dir)
    plot_mc_vs_run3_comparison(mc_met, r3_met, output_dir)

    # 10. JSON summary
    print("\n--- Saving JSON summary ---")
    summary = {
        "mc": _dataset_summary(mc_met),
        "run3": _dataset_summary(r3_met),
        "config": {
            "model_path": str(model_path),
            "n_events": args.n_events,
            "peak_threshold": 0.5,
            "peak_tolerance_bins": 5,
        },
    }
    summary_path = os.path.join(output_dir, "kde_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(_sanitize_for_json(summary), f, indent=2)
    print(f"  Saved: {summary_path}")

    # 11. Print summary
    _print_summary_table(mc_met, r3_met)
    print("\n" + "=" * 70)
    print("  COMPLETE -- all outputs in: " + output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
