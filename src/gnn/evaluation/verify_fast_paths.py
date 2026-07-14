"""Verify the optimized peak-finder and edge-selection implementations.

Two checks, each with timing:

1. Peak finder: pv_locations_updated_res_fast (numba) vs the Python
   original on real PU200 histograms (saved by chain_gap_decomposition),
   at the production operating point. Requires bit-identical outputs.

2. MaxScore selection: get_top1_associations_fast vs
   get_top_k_associations(k=1) on real GNN scores (mu60 PVF-e400 graphs
   and/or PU200 chain graphs). Differences are only acceptable at exact
   score ties (top-2 gap == 0), which the regression guard already
   classifies as knife-edge; every difference is checked to be a tie.

Usage:
    python -u -m gnn.evaluation.verify_fast_paths \\
        --hists outputs/07_14_2026_ttva_gap/histograms_300ev.npz \\
        --graphs data/run4/ttva_graphs/pu200_chain_v4k20_test.pt \\
        --weights model_weights/ttva_gnn_hllhc/ttva_gat_pu200_k20_epoch_175.pyt \\
        -d 0 -o outputs/07_14_2026_ttva_fastpaths/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gnn.evaluation.classification import (
    get_top1_associations_fast,
    get_top_k_associations,
)
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.utils.peak_finding import pv_locations_updated_res
from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast

PEAK_PARAMS = dict(
    threshold=1e-2, integral_threshold=0.40, min_width=3, min_height=0.03
)


def verify_peaks(hists_path: str) -> dict:
    """Bit-exactness + timing of the numba peak finder on real histograms."""
    hists = np.load(hists_path)["hists"]
    pv_locations_updated_res_fast(hists[0], **PEAK_PARAMS)  # JIT warmup

    t_legacy, t_fast = [], []
    n_diff = 0
    for h in tqdm(hists, desc="peaks"):
        t0 = time.perf_counter()
        legacy = pv_locations_updated_res(h, **PEAK_PARAMS)
        t1 = time.perf_counter()
        fast = pv_locations_updated_res_fast(h, **PEAK_PARAMS)
        t2 = time.perf_counter()
        t_legacy.append(t1 - t0)
        t_fast.append(t2 - t1)
        if not all(np.array_equal(a, b) for a, b in zip(legacy, fast)):
            n_diff += 1

    result = {
        "n_events": len(hists),
        "n_events_differing": int(n_diff),
        "bit_exact": n_diff == 0,
        "legacy_ms_median": 1e3 * float(np.median(t_legacy)),
        "fast_ms_median": 1e3 * float(np.median(t_fast)),
    }
    result["speedup"] = result["legacy_ms_median"] / result["fast_ms_median"]
    return result


def _tie_only(scores: np.ndarray, tracks: np.ndarray, diff_mask: np.ndarray) -> bool:
    """True iff every differing edge belongs to a track with a tied maximum."""
    for trk in np.unique(tracks[diff_mask]):
        s = scores[tracks == trk]
        if (s == s.max()).sum() < 2:
            return False
    return True


def verify_selection(
    graphs_path: str, weights: str, device: torch.device, thresholds: list[float]
) -> dict:
    """Agreement + timing of vectorized vs loop selection on real scores."""
    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)
    model.eval()

    graphs = torch.load(graphs_path, weights_only=False)
    cached = []
    with torch.no_grad():
        for g in tqdm(graphs, desc="forward"):
            logits = model(g.to(device))
            cached.append(
                (
                    torch.sigmoid(logits).cpu().numpy(),
                    g[("track", "to", "pv")].edge_index.cpu().numpy(),
                )
            )

    out: dict = {"graphs": graphs_path, "n_events": len(cached), "points": []}
    for t in thresholds:
        t_legacy = t_fast = 0.0
        edges_diff = events_diff = 0
        non_tie_events = []
        for i, (scores, edge_index) in enumerate(tqdm(cached, desc=f"t={t}")):
            t0 = time.perf_counter()
            legacy = get_top_k_associations(scores, edge_index, k=1, threshold=t)
            t1 = time.perf_counter()
            fast = get_top1_associations_fast(scores, edge_index, threshold=t)
            t2 = time.perf_counter()
            t_legacy += t1 - t0
            t_fast += t2 - t1
            diff = legacy != fast
            if diff.any():
                events_diff += 1
                edges_diff += int(diff.sum())
                if not _tie_only(scores, edge_index[0], diff):
                    non_tie_events.append(i)
        out["points"].append(
            {
                "t": t,
                "edges_differing": edges_diff,
                "events_differing": events_diff,
                "all_diffs_are_ties": not non_tie_events,
                "non_tie_events": non_tie_events,
                "legacy_ms_per_event": 1e3 * t_legacy / len(cached),
                "fast_ms_per_event": 1e3 * t_fast / len(cached),
                "speedup": t_legacy / max(t_fast, 1e-12),
            }
        )
    return out


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.device_id}"
        if args.device_id >= 0 and torch.cuda.is_available()
        else "cpu"
    )

    report: dict = {}
    if args.hists:
        report["peak_finder"] = verify_peaks(args.hists)
        p = report["peak_finder"]
        print(f"peaks: bit_exact={p['bit_exact']} "
              f"({p['n_events_differing']}/{p['n_events']} differ), "
              f"{p['legacy_ms_median']:.2f} -> {p['fast_ms_median']:.3f} ms "
              f"({p['speedup']:.0f}x)")  # fmt: skip

    if args.graphs:
        report["selection"] = verify_selection(
            args.graphs, args.weights, device, [0.5, 0.98, 0.995]
        )
        for pt in report["selection"]["points"]:
            print(f"selection t={pt['t']}: {pt['edges_differing']} edges / "
                  f"{pt['events_differing']} events differ "
                  f"(ties only: {pt['all_diffs_are_ties']}), "
                  f"{pt['legacy_ms_per_event']:.2f} -> "
                  f"{pt['fast_ms_per_event']:.3f} ms/evt "
                  f"({pt['speedup']:.0f}x)")  # fmt: skip

    with open(out_dir / "fast_paths_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {out_dir / 'fast_paths_report.json'}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hists", default=None, type=str, help="histograms .npz")
    p.add_argument("--graphs", default=None, type=str, help="graphs .pt")
    p.add_argument("--weights", default=None, type=str, help="GNN weights .pyt")
    p.add_argument("-d", "--device-id", default=0, type=int)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    main()
