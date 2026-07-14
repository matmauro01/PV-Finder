"""Measure chain-like augmentation parameters on the mu~60 PVF chain.

The mu60 counterpart of the PU200 measurement in chain_gap_decomposition:
greedy peak-truth matching between the PVF-e400 inference graphs and the
event-keyed h5 truth yields the finder decomposition and the empirical
distributions (dz residuals, height ratios, junk rate/heights, miss
probability vs nTracks) that drive gnn.data.graph_augmentation.

Per-peak sigmas are recovered from the stored edge attributes instead of
re-running the finder: long_sig = dz/sqrt(sigma_z0^2 + sigma_pv^2) and
|dz| are both stored per edge, and sigma_z0 is a track feature, so
sigma_pv = sqrt((dz/long_sig)^2 - sigma_z0^2) (median over the peak's
edges).

Outputs the same two JSONs (gap_decomposition.json +
augmentation_params.json) that AugmentationParams reads.

Usage:
    python -u -m gnn.diagnostics.mu60_aug_params \\
        --graphs outputs/07_12_2026_ttva_reproduction/regen/graphs_pvf_e400_regen.pt \\
        --h5 /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \\
        --indices configs/qibin_test_main_indices_v2.p \\
        -o outputs/07_14_2026_ttva_gap/mu60_aug_params/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from gnn.data.graph_construction import load_event_indices
from gnn.diagnostics.chain_gap_decomposition import (
    QUANTILES,
    WINDOWS_MM,
    greedy_match,
    truth_recipe_heights,
)
from pv_finder.data.resolution_presets import RESOLUTION_PRESETS


def peak_sigmas_from_edges(graph) -> np.ndarray:
    """Per-PV-node sigma recovered from (long_sig, |dz|) edge attributes."""
    edge_index = graph[("track", "to", "pv")].edge_index.numpy()
    attr = graph[("track", "to", "pv")].edge_attr.numpy()
    sig_z0 = graph["track"].x[:, 3].numpy().astype(np.float64)

    long_sig = attr[:, 0].astype(np.float64)
    abs_dz = attr[:, 2].astype(np.float64)
    ok = np.abs(long_sig) > 1e-6
    combined_sq = np.zeros_like(abs_dz)
    combined_sq[ok] = (abs_dz[ok] / np.abs(long_sig[ok])) ** 2
    sig_pv_sq = combined_sq - sig_z0[edge_index[0]] ** 2

    n_pv = graph["pv"].num_nodes
    sigmas = np.zeros(n_pv)
    for j in range(n_pv):
        vals = sig_pv_sq[ok & (edge_index[1] == j)]
        vals = vals[vals > 0]
        sigmas[j] = np.sqrt(np.median(vals)) if len(vals) else 0.0
    return sigmas


def main() -> None:  # noqa: PLR0915
    """CLI entry point."""
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_params = RESOLUTION_PRESETS[args.resolution_preset]

    graphs = torch.load(args.graphs, weights_only=False)
    indices = load_event_indices(args.indices)
    assert len(graphs) == len(indices)

    stats = {w: dict(matched=0, missed=0, junk=0, multi=0) for w in WINDOWS_MM}
    n_truth_total = n_peaks_total = 0
    missed_ntrk, matched_ntrk = [], []
    junk_heights, matched_dz, height_ratio = [], [], []
    sig_matched, sig_junk = [], []
    junk_per_event = []

    with h5py.File(args.h5, "r") as f:
        for graph, idx in tqdm(list(zip(graphs, indices)), desc="mu60 params"):
            key = f"Event{int(idx)}"
            tz_all = f["pv_loc_z"][key][:].astype(np.float64)
            tn_all = f["pv_ntracks"][key][:].astype(np.int64)
            keep = tn_all >= 2
            tz, tn = tz_all[keep], tn_all[keep]

            pv_x = graph["pv"].x.numpy()
            peak_z, peak_h = pv_x[:, 0].astype(np.float64), pv_x[:, 1]
            n_truth_total += len(tz)
            n_peaks_total += len(peak_z)

            for w in WINDOWS_MM:
                p_idx, t_idx = greedy_match(peak_z, tz, w)
                stats[w]["matched"] += len(p_idx)
                stats[w]["missed"] += len(tz) - len(t_idx)
                stats[w]["junk"] += len(peak_z) - len(p_idx)
                if len(peak_z) and len(tz):
                    dz = np.abs(peak_z[:, None] - tz[None, :])
                    stats[w]["multi"] += int(((dz < w).sum(axis=1) >= 2).sum())

            p_idx, t_idx = greedy_match(peak_z, tz, 0.5)
            sel_missed = np.setdiff1d(np.arange(len(tz)), t_idx)
            missed_ntrk.append(tn[sel_missed])
            matched_ntrk.append(tn[t_idx])
            sel_junk = np.setdiff1d(np.arange(len(peak_z)), p_idx)
            junk_heights.append(peak_h[sel_junk])
            junk_per_event.append(len(sel_junk))
            matched_dz.append(peak_z[p_idx] - tz[t_idx])
            recipe_h = truth_recipe_heights(tz[t_idx], tn[t_idx], res_params)
            height_ratio.append(peak_h[p_idx] / np.maximum(recipe_h, 1e-9))

            sigmas = peak_sigmas_from_edges(graph)
            sig_matched.append(sigmas[p_idx])
            sig_junk.append(sigmas[sel_junk])

    missed_ntrk = np.concatenate(missed_ntrk)
    matched_ntrk = np.concatenate(matched_ntrk)
    junk_heights = np.concatenate(junk_heights)
    matched_dz = np.concatenate(matched_dz)
    height_ratio = np.concatenate(height_ratio)
    sig_matched = np.concatenate(sig_matched)
    sig_junk = np.concatenate(sig_junk)
    sig_matched = sig_matched[sig_matched > 0]
    sig_junk = sig_junk[sig_junk > 0]

    edges = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 300])
    gap = {
        "n_events": len(graphs),
        "n_truth_pvs": int(n_truth_total),
        "n_peaks": int(n_peaks_total),
        "windows": {
            str(w): {k: int(v) for k, v in stats[w].items()} for w in WINDOWS_MM
        },
        "finder_cap_per_window": {
            str(w): stats[w]["matched"] / n_truth_total for w in WINDOWS_MM
        },
        "missed_ntrk_hist": np.histogram(missed_ntrk, bins=edges)[0].tolist(),
        "matched_ntrk_hist": np.histogram(matched_ntrk, bins=edges)[0].tolist(),
        "ntrk_bin_edges": edges.tolist(),
        "missed_ntrk_median": float(np.median(missed_ntrk)),
        "matched_ntrk_median": float(np.median(matched_ntrk)),
    }
    aug = {
        "jitter_dz_quantiles": np.quantile(matched_dz, QUANTILES).tolist(),
        "height_ratio_quantiles": np.quantile(height_ratio, QUANTILES).tolist(),
        "junk_height_quantiles": np.quantile(junk_heights, QUANTILES).tolist(),
        "junk_per_event_mean": float(np.mean(junk_per_event)),
        "junk_per_peak": float(np.sum(junk_per_event) / n_peaks_total),
        "matched_sigma_quantiles": np.quantile(sig_matched, QUANTILES).tolist(),
        "junk_sigma_quantiles": np.quantile(sig_junk, QUANTILES).tolist(),
        "quantile_grid": QUANTILES.tolist(),
    }
    with open(out_dir / "gap_decomposition.json", "w") as f:
        json.dump(gap, f, indent=2)
    with open(out_dir / "augmentation_params.json", "w") as f:
        json.dump(aug, f, indent=2)

    cap = gap["finder_cap_per_window"]
    print(f"Truth PVs: {n_truth_total}, peaks: {n_peaks_total}")
    for w in WINDOWS_MM:
        s = gap["windows"][str(w)]
        print(f"  window {w} mm: cap {cap[str(w)]:.4f}, missed {s['missed']}, "
              f"junk {s['junk']}, multi {s['multi']}")  # fmt: skip
    print(f"sigma matched median {np.median(sig_matched):.4f} mm, "
          f"junk {np.median(sig_junk):.4f} mm; "
          f"junk/peak {aug['junk_per_peak']:.4f}")  # fmt: skip
    print(f"Saved to {out_dir}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--graphs", required=True, type=str)
    p.add_argument("--h5", required=True, type=str)
    p.add_argument("--indices", required=True, type=str)
    p.add_argument("--resolution-preset", default="run3", type=str)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    main()
