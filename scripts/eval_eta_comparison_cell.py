#!/usr/bin/env python3
"""Evaluate one model/track-production cell on a preselected PU200 sample.

Unlike ``run_eval_pvf_run3.py``, this script selects the requested pile-up
window before inference.  It is intended for the small model-by-production
comparison used in the technical-note appendix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

from pv_finder.diagnostics.bump_model_vs_data import (
    build_model,
    find_peaks,
    load_entries,
    predict_hist,
    select_entries,
)
from pv_finder.evaluation.vertex_finding.efficiency_res_optimized_atlas import (
    compare_res_reco,
)
from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import sigmoid_fit

Z_MIN_MM = -240.0
BIN_WIDTH_MM = 0.04
PAIRWISE_RANGE_MM = 6.0


def _pairwise_dz(positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return one signed difference for each unordered pair."""
    shuffled = rng.permutation(positions)
    left, right = np.triu_indices(len(shuffled), k=1)
    return shuffled[left] - shuffled[right]


def _fit_resolution(dz: np.ndarray, n_bins: int) -> tuple[float, float]:
    edges = np.linspace(-PAIRWISE_RANGE_MM, PAIRWISE_RANGE_MM, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(dz, bins=edges)
    baseline = float(np.median(counts))
    initial = [
        max(baseline - float(counts.min()), 1.0),
        10.0,
        max(baseline, 1.0),
        0.5,
    ]
    pars, covariance = curve_fit(
        sigmoid_fit,
        centres,
        counts.astype(float),
        p0=initial,
        maxfev=10000,
        bounds=([0, 0, 0, 0], [np.inf] * 4),
    )
    return float(abs(pars[3])), float(np.sqrt(np.diag(covariance))[3])


def _classify(
    truth_mm: np.ndarray, reco_mm: np.ndarray, window_mm: float
) -> dict[str, int]:
    if len(reco_mm) == 0:
        return {
            "reco": 0,
            "clean": 0,
            "merged": 0,
            "split": 0,
            "fake": 0,
            "truth_clean": 0,
            "truth_merged": 0,
            "truth_missed": len(truth_mm),
        }
    truth_bins = (truth_mm - Z_MIN_MM) / BIN_WIDTH_MM
    reco_bins = (reco_mm - Z_MIN_MM) / BIN_WIDTH_MM
    result, truth_categories, _ = compare_res_reco(
        truth_bins,
        reco_bins,
        np.full(len(reco_bins), window_mm / BIN_WIDTH_MM),
        debug=0,
    )
    return {
        "reco": len(reco_mm),
        "clean": result.reco_clean,
        "merged": result.reco_merged,
        "split": result.reco_split,
        "fake": result.reco_fake,
        "truth_clean": int(np.sum(truth_categories == "clean")),
        "truth_merged": int(np.sum(truth_categories == "merged")),
        "truth_missed": int(np.sum(truth_categories == "missed")),
    }


def _average(records: list[dict[str, int]], key: str) -> float:
    return float(np.mean([record[key] for record in records]))


def main(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
    entries = select_entries(
        args.root, args.mu_min, args.mu_max, args.n_events, args.entry_stop
    )
    events = load_entries(args.root, entries)["events"]
    model = build_model(args.model, device)

    predicted: list[np.ndarray] = []
    resolution_dz: list[np.ndarray] = []
    rng = np.random.default_rng(args.seed)
    for index, event in enumerate(events):
        histogram = predict_hist(model, event, device)
        peaks, _, _ = find_peaks(
            histogram,
            args.peak_threshold,
            args.integral_threshold,
            args.min_width,
            args.min_height,
            args.centroid_halfwidth,
        )
        resolution_peaks, _, _ = find_peaks(
            histogram,
            args.peak_threshold,
            args.resolution_integral_threshold,
            args.min_width,
            args.min_height,
            args.centroid_halfwidth,
        )
        predicted.append(peaks)
        resolution_dz.append(_pairwise_dz(resolution_peaks, rng))
        if index < 5 or (index + 1) % 100 == 0:
            print(
                f"event {index + 1:4d}/{len(events)}: "
                f"tracks={len(event['z0'])} truth={len(event['truth_z'])} "
                f"peaks={len(peaks)}",
                flush=True,
            )

    dz = np.concatenate(resolution_dz)
    sigma, sigma_error = _fit_resolution(dz, args.pairwise_bins)
    pvf_records = [
        _classify(event["truth_z"], peaks, sigma)
        for event, peaks in zip(events, predicted)
    ]
    amvf_records = [
        _classify(event["truth_z"], event["amvf_z"], sigma) for event in events
    ]

    n_truth = sum(len(event["truth_z"]) for event in events)
    n_pvf_recovered = sum(
        record["truth_clean"] + record["truth_merged"] for record in pvf_records
    )
    n_amvf_recovered = sum(
        record["truth_clean"] + record["truth_merged"] for record in amvf_records
    )

    def summarize(records: list[dict[str, int]], recovered: int) -> dict[str, float]:
        return {
            "reco_per_event": _average(records, "reco"),
            "clean_per_event": _average(records, "clean"),
            "merged_per_event": _average(records, "merged"),
            "split_per_event": _average(records, "split"),
            "fake_per_event": _average(records, "fake"),
            "efficiency": recovered / n_truth,
        }

    summary = {
        "root": args.root,
        "model": args.model,
        "event_entries": entries.tolist(),
        "n_events": len(events),
        "mu_window": [args.mu_min, args.mu_max],
        "tracks_per_event": float(np.mean([len(event["z0"]) for event in events])),
        "truth_vertices_per_event": n_truth / len(events),
        "sigma_vtx_vtx_mm": sigma,
        "sigma_vtx_vtx_error_mm": sigma_error,
        "pvfinder": summarize(pvf_records, n_pvf_recovered),
        "amvf": summarize(amvf_records, n_amvf_recovered),
        "settings": {
            "peak_threshold": args.peak_threshold,
            "integral_threshold": args.integral_threshold,
            "resolution_integral_threshold": args.resolution_integral_threshold,
            "min_width": args.min_width,
            "min_height": args.min_height,
            "centroid_halfwidth": args.centroid_halfwidth,
            "pairwise_bins": args.pairwise_bins,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n-events", type=int, default=1920)
    parser.add_argument("--entry-stop", type=int, default=25000)
    parser.add_argument("--mu-min", type=float, default=185.0)
    parser.add_argument("--mu-max", type=float, default=215.0)
    parser.add_argument("--peak-threshold", type=float, default=0.01)
    parser.add_argument("--integral-threshold", type=float, default=0.30)
    parser.add_argument("--resolution-integral-threshold", type=float, default=0.50)
    parser.add_argument("--min-width", type=int, default=3)
    parser.add_argument("--min-height", type=float, default=0.03)
    parser.add_argument("--centroid-halfwidth", type=int, default=3)
    parser.add_argument("--pairwise-bins", type=int, default=300)
    parser.add_argument("--seed", type=int, default=12345)
    main(parser.parse_args())
