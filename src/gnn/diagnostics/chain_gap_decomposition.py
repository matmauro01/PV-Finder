"""Decompose the PU200 full-chain clean/truth gap: finder vs associator.

Two passes over the chain evaluation slice:

Part A (all chain graphs, no GPU): greedy 1-1 position matching between
PVF peaks (PV nodes of the chain graphs) and truth PVs (nTrk >= 2, from
ROOT) at several |dz| windows. Yields the finder-side decomposition
(missed truth PVs vs nTracks, junk peaks, multi-truth "merge pressure")
and the matched-pair statistics (dz residuals, height ratio vs the truth
recipe) that parametrize the v3 training-graph augmentation.

Part B (--sigma-events, GPU): re-runs the PVF+peak-finding stages on a
subset to record per-peak sigmas (not stored in the graphs) and saves the
histograms for the peak-finder vectorization check (P3).

Usage:
    python -u -m gnn.diagnostics.chain_gap_decomposition \\
        --graphs data/run4/ttva_graphs/pu200_chain_v4b_k20_test.pt \\
        --root data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root \\
        --entry-start 28500 \\
        --pvf-weights model_weights/hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3_fullstate.pth \\
        --sigma-events 300 -d 0 -o outputs/07_14_2026_ttva_gap/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from tqdm import tqdm

from pv_finder.data.resolution_presets import RESOLUTION_PRESETS

QUANTILES = np.linspace(0.0, 1.0, 101)
WINDOWS_MM = (0.3, 0.5, 1.0)


def greedy_match(peak_z: np.ndarray, truth_z: np.ndarray, window: float):
    """Greedy 1-1 matching by |dz| < window; returns (peak_idx, truth_idx) pairs."""
    if len(peak_z) == 0 or len(truth_z) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    dz = np.abs(peak_z[:, None] - truth_z[None, :])
    pi, ti = np.nonzero(dz < window)
    order = np.argsort(dz[pi, ti], kind="stable")
    used_p = np.zeros(len(peak_z), bool)
    used_t = np.zeros(len(truth_z), bool)
    out_p, out_t = [], []
    for k in order:
        p, t = pi[k], ti[k]
        if not used_p[p] and not used_t[t]:
            used_p[p] = used_t[t] = True
            out_p.append(p)
            out_t.append(t)
    return np.array(out_p, np.int64), np.array(out_t, np.int64)


def truth_recipe_heights(
    truth_z: np.ndarray, ntracks: np.ndarray, res_params
) -> np.ndarray:
    """PV-node heights the truth-graph recipe would assign to these vertices."""
    from gnn.data.graph_construction import compute_truth_pv_heights
    from gnn.data.root_to_graphs import compute_pv_sigma_preset

    pv_res = compute_pv_sigma_preset(ntracks.astype(np.float64), *res_params)
    return compute_truth_pv_heights(truth_z, pv_res)


def part_a(graphs, root_events, res_params, out: dict) -> dict:
    """Position-based finder decomposition + augmentation statistics."""
    stats = {w: dict(matched=0, missed=0, junk=0, multi=0) for w in WINDOWS_MM}
    n_truth_total = 0
    n_peaks_total = 0
    missed_ntrk: list[np.ndarray] = []
    matched_ntrk: list[np.ndarray] = []
    junk_heights: list[np.ndarray] = []
    matched_dz: list[np.ndarray] = []
    height_ratio: list[np.ndarray] = []
    junk_per_event: list[int] = []

    for graph, (truth_z, truth_ntrk) in tqdm(
        list(zip(graphs, root_events)), desc="Part A"
    ):
        pv_x = graph["pv"].x.numpy()
        peak_z, peak_h = pv_x[:, 0].astype(np.float64), pv_x[:, 1]
        keep = truth_ntrk >= 2
        tz, tn = truth_z[keep], truth_ntrk[keep]
        n_truth_total += len(tz)
        n_peaks_total += len(peak_z)

        for w in WINDOWS_MM:
            p_idx, t_idx = greedy_match(peak_z, tz, w)
            stats[w]["matched"] += len(p_idx)
            stats[w]["missed"] += len(tz) - len(t_idx)
            stats[w]["junk"] += len(peak_z) - len(p_idx)
            # peaks with >= 2 truth PVs inside the window (merge pressure)
            if len(peak_z) and len(tz):
                dz = np.abs(peak_z[:, None] - tz[None, :])
                stats[w]["multi"] += int(((dz < w).sum(axis=1) >= 2).sum())

        # Augmentation statistics at the reference window (0.5 mm)
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

    missed_ntrk = np.concatenate(missed_ntrk)
    matched_ntrk = np.concatenate(matched_ntrk)
    junk_heights = np.concatenate(junk_heights)
    matched_dz = np.concatenate(matched_dz)
    height_ratio = np.concatenate(height_ratio)

    out["n_events"] = len(graphs)
    out["n_truth_pvs"] = int(n_truth_total)
    out["n_peaks"] = int(n_peaks_total)
    out["windows"] = {
        str(w): {k: int(v) for k, v in stats[w].items()} for w in WINDOWS_MM
    }
    out["finder_cap_per_window"] = {
        str(w): stats[w]["matched"] / n_truth_total for w in WINDOWS_MM
    }
    edges = np.array([2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 300])
    out["missed_ntrk_hist"] = np.histogram(missed_ntrk, bins=edges)[0].tolist()
    out["matched_ntrk_hist"] = np.histogram(matched_ntrk, bins=edges)[0].tolist()
    out["ntrk_bin_edges"] = edges.tolist()
    out["missed_ntrk_median"] = float(np.median(missed_ntrk))
    out["matched_ntrk_median"] = float(np.median(matched_ntrk))

    aug = {
        "jitter_dz_quantiles": np.quantile(matched_dz, QUANTILES).tolist(),
        "height_ratio_quantiles": np.quantile(height_ratio, QUANTILES).tolist(),
        "junk_height_quantiles": np.quantile(junk_heights, QUANTILES).tolist(),
        "junk_per_event_mean": float(np.mean(junk_per_event)),
        "junk_per_peak": float(np.sum(junk_per_event) / n_peaks_total),
        "quantile_grid": QUANTILES.tolist(),
    }
    return aug


def oracle_rows(graphs, root_events, window: float = 1.0) -> dict:
    """Chain ceiling: assign each track to the peak nearest its TRUE vertex.

    Tracks whose true vertex has no peak within *window* mm (and truthless
    tracks) stay unassigned. Classifying these oracle assignments bounds
    what ANY associator could achieve on the peaks PVF actually found:
    remaining Merged/Fake vertices are properties of the peak set itself.
    """
    from gnn.evaluation.classification import classify_assignments

    totals = np.zeros(6, dtype=np.int64)
    totals_drop = np.zeros(6, dtype=np.int64)
    for graph, (truth_z, truth_ntrk) in tqdm(
        list(zip(graphs, root_events)), desc="Oracle"
    ):
        pv_x = graph["pv"].x.numpy()
        peak_z = pv_x[:, 0].astype(np.float64)
        track_truth = graph["track"].truth_pv.numpy()
        pt = graph["track"].x[:, 7].numpy()

        n_peaks = len(peak_z)
        matched_tracks_per_pv: list[list[int]] = [[] for _ in range(n_peaks)]
        if n_peaks:
            has_truth = np.nonzero(track_truth >= 0)[0]
            tz_of_track = truth_z[track_truth[has_truth]]
            nearest = np.abs(tz_of_track[:, None] - peak_z[None, :]).argmin(axis=1)
            dz = np.abs(tz_of_track - peak_z[nearest])
            for trk, peak, d in zip(has_truth, nearest, dz):
                if d < window:
                    matched_tracks_per_pv[peak].append(int(trk))

        keep = truth_ntrk >= 2
        tti = []
        tpi = []
        for i in np.nonzero(track_truth >= 0)[0]:
            tti.append(float(i))
            tpi.append(float(track_truth[i]))
        matched_arrays = [
            np.array(sorted(m), dtype=np.int64) for m in matched_tracks_per_pv
        ]
        tti_a, tpi_a, n_t = np.array(tti), np.array(tpi), int(keep.sum())
        rows, _ = classify_assignments(matched_arrays, pt, tti_a, tpi_a, n_t)
        totals += np.array(rows, dtype=np.int64)
        # Same, but zero-track vertices dropped from the reco list first
        # (no vertexing chain outputs a trackless vertex; the classifier
        # otherwise books each one as Fake).
        rows_d, _ = classify_assignments(
            [m for m in matched_arrays if len(m)], pt, tti_a, tpi_a, n_t
        )
        totals_drop += np.array(rows_d, dtype=np.int64)

    def _summary(t: np.ndarray) -> dict:
        clean, merged, split, fake, n_reco, n_truth = (int(x) for x in t)
        return {
            "window_mm": window,
            "clean": clean, "merged": merged, "split": split, "fake": fake,
            "n_reco": n_reco, "n_truth": n_truth,
            "fake_rate": fake / n_reco,
            "clean_per_truth": clean / n_truth,
        }  # fmt: skip

    return {"all_peaks": _summary(totals), "drop_empty": _summary(totals_drop)}


def part_b(args, device: torch.device, out_dir: Path, aug: dict) -> None:
    """Re-run PVF+peaks on a subset: per-peak sigmas + saved histograms."""
    from gnn.data.pu200_chain_graphs import build_pvf_model
    from pv_finder.data.run3_io import Run3Event
    from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import (
        build_subevent_inputs,
        run_inference,
    )
    from pv_finder.utils.peak_finding import pv_locations_updated_res

    pvf = build_pvf_model(
        args.pvf_weights, device, args.unet_channels, args.latent_channels,
        [args.hidden_nodes] * 5,
    )  # fmt: skip
    tree = uproot.open(args.root)["PVFinderData"]
    stop = args.entry_start + args.sigma_events

    hists, sig_matched, sig_junk = [], [], []
    branches = ["RecoTrack_z0", "RecoTrack_d0", "RecoTrack_ErrD0", "RecoTrack_ErrZ0",
                "RecoTrack_ErrD0Z0", "TruthVertex_z", "TruthVertex_nTracks"]  # fmt: skip
    for chunk in tree.iterate(
        branches, step_size=100, entry_start=args.entry_start, entry_stop=stop
    ):
        for event in tqdm(chunk, desc="Part B", leave=False):
            z0 = ak.to_numpy(event["RecoTrack_z0"]).astype(np.float32)
            ev = Run3Event(
                z0=z0,
                d0=ak.to_numpy(event["RecoTrack_d0"]).astype(np.float32),
                d0_err=ak.to_numpy(event["RecoTrack_ErrD0"]).astype(np.float32),
                z0_err=ak.to_numpy(event["RecoTrack_ErrZ0"]).astype(np.float32),
                d0_z0_cov=ak.to_numpy(event["RecoTrack_ErrD0Z0"]).astype(np.float32),
                amvf_z=np.array([]), amvf_ntrks=np.array([]), beam_z=0.0,
                mu=None, event_idx=0, n_tracks=len(z0),
            )  # fmt: skip
            hist = run_inference(build_subevent_inputs(ev), device, e2e=pvf)
            hists.append(np.asarray(hist, dtype=np.float32))
            pz, _, _, ps = pv_locations_updated_res(
                hist, args.peak_threshold, args.integral_threshold,
                args.min_width, args.min_height,
            )  # fmt: skip
            tz = ak.to_numpy(event["TruthVertex_z"]).astype(np.float64)
            tn = ak.to_numpy(event["TruthVertex_nTracks"]).astype(np.int64)
            tz = tz[tn >= 2]
            p_idx, _ = greedy_match(pz.astype(np.float64), tz, 0.5)
            junk = np.setdiff1d(np.arange(len(pz)), p_idx)
            sig_matched.append(ps[p_idx])
            sig_junk.append(ps[junk])

    sig_matched = np.concatenate(sig_matched)
    sig_junk = np.concatenate(sig_junk)
    aug["matched_sigma_quantiles"] = np.quantile(sig_matched, QUANTILES).tolist()
    aug["junk_sigma_quantiles"] = np.quantile(sig_junk, QUANTILES).tolist()
    np.savez_compressed(out_dir / "histograms_300ev.npz", hists=np.stack(hists))
    print(f"sigma (matched): median {np.median(sig_matched):.4f} mm; "
          f"(junk): median {np.median(sig_junk):.4f} mm")  # fmt: skip


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    graphs = torch.load(args.graphs, weights_only=False)
    tree = uproot.open(args.root)["PVFinderData"]
    stop = args.entry_start + len(graphs)
    root_events = []
    for chunk in tree.iterate(
        ["TruthVertex_z", "TruthVertex_nTracks"],
        step_size=500, entry_start=args.entry_start, entry_stop=stop,
    ):  # fmt: skip
        for event in chunk:
            root_events.append((
                ak.to_numpy(event["TruthVertex_z"]).astype(np.float64),
                ak.to_numpy(event["TruthVertex_nTracks"]).astype(np.int64),
            ))  # fmt: skip
    assert len(root_events) == len(graphs)

    out: dict = {"config": vars(args)}
    res_params = RESOLUTION_PRESETS[args.resolution_preset]
    aug = part_a(graphs, root_events, res_params, out)
    out["oracle"] = oracle_rows(graphs, root_events)

    if args.sigma_events > 0:
        device = torch.device(
            f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
        )
        part_b(args, device, out_dir, aug)

    with open(out_dir / "gap_decomposition.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(out_dir / "augmentation_params.json", "w") as f:
        json.dump(aug, f, indent=2)

    cap = out["finder_cap_per_window"]
    print(f"\nTruth PVs: {out['n_truth_pvs']}, peaks: {out['n_peaks']}")
    for w in WINDOWS_MM:
        s = out["windows"][str(w)]
        print(f"  window {w} mm: matched {s['matched']} "
              f"(cap {cap[str(w)]:.4f}), missed {s['missed']}, "
              f"junk {s['junk']}, multi-truth peaks {s['multi']}")  # fmt: skip
    print(f"missed nTrk median {out['missed_ntrk_median']:.1f} vs "
          f"matched {out['matched_ntrk_median']:.1f}")  # fmt: skip
    for key, orc in out["oracle"].items():
        print(f"oracle [{key}]: clean/truth {orc['clean_per_truth']:.4f}, "
              f"fake_rate {orc['fake_rate']:.4f}, merged {orc['merged']}, "
              f"split {orc['split']}")  # fmt: skip
    print(f"Done in {time.time() - t0:.1f} s -> {out_dir}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--graphs", required=True, type=str)
    p.add_argument("--root", required=True, type=str)
    p.add_argument("--entry-start", default=28500, type=int)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    p.add_argument("--resolution-preset", default="hllhc", type=str)
    # Part B (sigma measurement) options
    p.add_argument("--sigma-events", default=0, type=int)
    p.add_argument("--pvf-weights", default=None, type=str)
    p.add_argument("-d", "--device-id", default=0, type=int)
    p.add_argument("--peak-threshold", default=1e-2, type=float)
    p.add_argument("--integral-threshold", default=0.40, type=float)
    p.add_argument("--min-width", default=3, type=int)
    p.add_argument("--min-height", default=0.03, type=float)
    p.add_argument("--unet-channels", default=280, type=int)
    p.add_argument("--latent-channels", default=4, type=int)
    p.add_argument("--hidden-nodes", default=128, type=int)
    return p.parse_args()


if __name__ == "__main__":
    main()
