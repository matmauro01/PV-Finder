#!/usr/bin/env python3
"""Regenerate plots from an existing eval_results.pkl without re-running inference.

Usage:
    python replot_from_pkl.py <output_dir> [--dataset-name NAME] [--mu-min N] [--mu-max N]
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402
from plots_pvf import (  # noqa: E402
    plot_category_counts,
    plot_performance,
    plot_reco_vs_mu,
    plot_resolution,
    plot_stats,
)


def sigmoid_fit(x, a, b, c, rcc):
    return a / (1.0 + np.exp(b * (rcc - np.abs(x)))) + c


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "output_dir", help="Eval output directory containing eval_results.pkl"
    )
    p.add_argument("--dataset-name", default="", dest="dataset_name")
    p.add_argument("--mu-min", type=int, default=55, dest="mu_min")
    p.add_argument("--mu-max", type=int, default=65, dest="mu_max")
    p.add_argument("--integral-threshold", type=float, default=0.5,
                   dest="integral_threshold")  # fmt: skip
    args = p.parse_args()

    outdir = Path(args.output_dir)
    pkl_path = outdir / "eval_results.pkl"
    if not pkl_path.exists():
        sys.exit(f"ERROR: {pkl_path} not found")

    with open(pkl_path, "rb") as f:
        r = pickle.load(f)

    per_event = r["per_event"]
    sigma = r["sigma_vtx_vtx_mm"]
    overall_eff = r["overall_efficiency"]
    fp_rate = r["fp_rate_per_evt"]
    dz_arr = r["pairwise_dz_mm"]
    popt = r.get("fit_params")
    ckpt_path = r.get("e2e_checkpoint") or r.get("k2h_checkpoint") or ""
    ckpt = Path(ckpt_path).stem if ckpt_path else "unknown"
    mode_label = f"E2E ({ckpt})" if r.get("mode") == "e2e" else f"pipeline ({ckpt})"
    has_mu = any(e.get("mu") is not None for e in per_event)
    ds = args.dataset_name or ("MC" if not has_mu else "Real Data")

    print(f"Replotting {outdir} ({len(per_event)} events, dataset='{ds}')")

    plot_resolution(dz_arr, sigma, np.array(popt) if popt else None, sigmoid_fit,
                    mode_label, outdir,
                    title=f"PVF Resolution — {ds}\n({mode_label})")  # fmt: skip
    t = f"PVF — {ds}\n({mode_label})"
    plot_performance(per_event, overall_eff, fp_rate, sigma, has_mu,
                     mode_label, outdir, title=t)  # fmt: skip
    plot_stats(per_event, has_mu, mode_label, outdir, title=t)
    if has_mu:
        plot_reco_vs_mu(per_event, mode_label, outdir, title=t)
    has_truth = any(r.get("amvf_clean") is not None for r in per_event)
    truth_avg = float(np.mean([r["n_truth"] for r in per_event])) if has_truth else None
    plot_category_counts(per_event, mode_label, outdir, title="",
        eval_label=f"ckpt: {ckpt}\nintegral_threshold = {args.integral_threshold}",
        mu_min=args.mu_min, mu_max=args.mu_max,
        all_events=args.mu_min >= 100,
        truth_pvs_per_evt=truth_avg)  # fmt: skip
    print(f"  Done: {outdir}")


if __name__ == "__main__":
    main()
