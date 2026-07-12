"""Classify AMVF vertices as Clean/Merged/Split/Fake — GNN comparison baseline.

Runs the exact same classification as the GNN TTVA evaluation
(gnn.evaluation.classification.classify_assignments), but with track-vertex
assignments taken from AMVF's own output (reco_pv_assoc_tracks in the MC HDF5)
instead of GNN edge scores. This compares the full chains:
AMVF finding+association vs PVF finding + GNN association.

Usage:
    python -m gnn.evaluation.evaluate_amvf_ttva \\
        -f /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \\
        -i configs/qibin_test_main_indices_v2.p \\
        -o outputs/<date>/amvf/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from gnn.data.graph_construction import load_event_indices
from gnn.evaluation.classification import build_truth_adjacency, classify_assignments


def evaluate_amvf(
    h5_path: str | Path,
    event_keys: list[str],
    min_ntracks: int = 0,
) -> tuple[list[list[int]], list[list[dict[str, Any]]]]:
    """Classify AMVF vertices against truth for each event.

    Args:
        h5_path: Event-keyed ATLAS HDF5 with truth (pv_*) and AMVF (reco_pv_*)
            datasets.
        event_keys: HDF5 event keys to process.
        min_ntracks: Keep only AMVF vertices with at least this many
            associated tracks (0 = keep all).

    Returns:
        Tuple of (total_results, total_reco_pv_info) in the same format as
        gnn.evaluation.evaluate_ttva.evaluate_gnn.
    """
    total_results: list[list[int]] = []
    total_info: list[list[dict[str, Any]]] = []

    totals = np.zeros(6, dtype=np.int64)

    with h5py.File(str(h5_path), "r") as f:
        pt = f["recoTrk_pt"]
        pv_loc_z = f["pv_loc_z"]
        pv_assoctracks = f["pv_assoc_tracks"]
        pv_ntracks = f["pv_ntracks"]
        amvf_ntracks = f["reco_pv_ntracks"]
        amvf_assoctracks = f["reco_pv_assoc_tracks"]

        for event_key in tqdm(event_keys):
            pt_event = pt[event_key][:]

            truth_track_indices, truth_pv_indices = build_truth_adjacency(
                pv_loc_z[event_key][:],
                pv_ntracks[event_key][:],
                pv_assoctracks[event_key][:],
            )
            truth_pvs_count = int((pv_ntracks[event_key][:] >= 2).sum())

            # Unpack AMVF's flat per-vertex track lists
            amvf_nt = amvf_ntracks[event_key][:].astype(int)
            amvf_at = amvf_assoctracks[event_key][:].astype(int)
            matched_tracks_per_pv: list[np.ndarray] = []
            counter = 0
            for n in amvf_nt:
                tracks = np.unique(amvf_at[counter : counter + n])
                counter += n
                if n >= min_ntracks:
                    matched_tracks_per_pv.append(tracks)

            results, info = classify_assignments(
                matched_tracks_per_pv,
                pt_event,
                truth_track_indices,
                truth_pv_indices,
                truth_pvs_count,
            )
            totals += np.array(results, dtype=np.int64)
            total_results.append(results)
            total_info.append(info)

    clean, merged, split, fake, n_reco, n_truth = totals
    print(f"Total Reconstructed PVs: {n_reco}")
    print(f"Total Truth PVs: {n_truth}")
    if n_reco > 0:
        print(f"Total Clean: {clean}, Rate: {clean / n_reco:.4f}")
        print(f"Total Merged: {merged}, Rate: {merged / n_reco:.4f}")
        print(f"Total Split: {split}, Rate: {split / n_reco:.4f}")
        print(f"Total Fake: {fake}, Rate: {fake / n_reco:.4f}")

    return total_results, total_info


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Classify AMVF vertices as Clean/Merged/Split/Fake"
    )
    parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        type=str,
        help="Event-keyed ATLAS HDF5 with pv_* truth and reco_pv_* AMVF data",
    )
    parser.add_argument(
        "-i",
        "--indices",
        required=True,
        type=str,
        help="Event indices (.npy or pickle)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        type=str,
        help="Directory for result files (default: current directory)",
    )
    parser.add_argument(
        "--min-ntracks",
        default=0,
        type=int,
        help="Keep only AMVF vertices with >= N tracks (default: 0 = all)",
    )
    parser.add_argument(
        "-n",
        "--max-events",
        default=None,
        type=int,
        help="Evaluate only the first N events (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    indices = load_event_indices(args.indices)
    event_keys = [f"Event{idx}" for idx in indices]
    if args.max_events is not None:
        event_keys = event_keys[: args.max_events]

    total_results, total_info = evaluate_amvf(
        args.filepath, event_keys, args.min_ntracks
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "total_results_AMVF.npy"
    info_path = out_dir / "total_reco_pv_info_list_AMVF.npy"
    np.save(results_path, np.array(total_results, dtype=object), allow_pickle=True)
    np.save(info_path, np.array(total_info, dtype=object), allow_pickle=True)
    print(f"Saved results to {results_path} and {info_path}")


if __name__ == "__main__":
    main()
