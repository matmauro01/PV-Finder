"""Evaluate GNN Track-to-Vertex Association on reconstructed PVs.

Classifies each reconstructed primary vertex as Clean, Merged, Split, or
Fake by comparing GNN edge predictions against truth associations from the
HDF5 data file.

Migrated from atlas_pvfinder/tracks_to_vertex/Evaluation_GNN_TTVA.py.
Restored from commit 51523df (deleted by mistake in 0da13fd) and moved
into the gnn package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.data.graph_construction import load_event_indices
from gnn.evaluation.classification import categorize_event
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.utils.constants import GNN_SCORE_THRESHOLD


def evaluate_gnn(
    model: TTVAGATModel,
    reco_graphs: list[HeteroData],
    h5_path: str | Path,
    event_keys: list[str],
    eval_method: str,
    threshold: float,
    device: torch.device,
) -> tuple[list[list[int]], list[list[dict[str, Any]]]]:
    """Run categorize_event over all events and aggregate results.

    Args:
        model: Trained TTVAGATModel on *device*.
        reco_graphs: Pre-loaded list of HeteroData reco graphs.
        h5_path: Path to the ATLAS HDF5 file with truth information.
        event_keys: HDF5 event keys (e.g. ``["Event0", "Event42", ...]``).
        eval_method: ``"MaxScore"`` or ``"Threshold"``.
        threshold: Score threshold for edge selection.
        device: Torch device.

    Returns:
        Tuple of (total_results, total_reco_pv_info_list).
    """
    total_clean = total_merged = total_fake = total_split = 0
    total_reco_pvs = total_truth_pvs = 0

    total_results: list[list[int]] = []
    total_reco_pv_info: list[list[dict[str, Any]]] = []

    if len(event_keys) != len(reco_graphs):
        print(
            f"Warning: Mismatch test_keys length ({len(event_keys)}) "
            f"and reco_graphs length ({len(reco_graphs)})"
        )

    with h5py.File(str(h5_path), "r") as data_file:
        d_0 = data_file["recoTrk_d"]
        z_0 = data_file["recoTrk_z"]
        pt = data_file["recoTrk_pt"]
        pv_loc_z = data_file["pv_loc_z"]
        pv_assoctracks = data_file["pv_assoc_tracks"]
        pv_ntracks = data_file["pv_ntracks"]

        for i, event_key in enumerate(tqdm(event_keys)):
            d_0_event = d_0[event_key][:]
            z_0_event = z_0[event_key][:]
            pt_event = pt[event_key][:]

            pv_loc_z_event = pv_loc_z[event_key][:]
            pv_assoctracks_event = pv_assoctracks[event_key][:]
            pv_ntracks_event = pv_ntracks[event_key][:]

            tracks_event_stack = np.stack([d_0_event, z_0_event, pt_event]).astype(
                np.float32
            )
            reco_graph_event = reco_graphs[i]

            results, reco_pv_info_list = categorize_event(
                model,
                reco_graph_event,
                tracks_event_stack,
                pv_loc_z_event,
                pv_assoctracks_event,
                pv_ntracks_event,
                eval_method,
                threshold,
                device,
            )

            total_clean += results[0]
            total_merged += results[1]
            total_split += results[2]
            total_fake += results[3]
            total_reco_pvs += results[4]
            total_truth_pvs += results[5]

            total_results.append(results)
            total_reco_pv_info.append(reco_pv_info_list)

    # Print summary
    if total_reco_pvs > 0:
        clean_rate = total_clean / total_reco_pvs
        merged_rate = total_merged / total_reco_pvs
        fake_rate = total_fake / total_reco_pvs
        split_rate = total_split / total_reco_pvs
    else:
        clean_rate = merged_rate = fake_rate = split_rate = 0.0

    print(f"Total Reconstructed PVs: {total_reco_pvs}")
    print(f"Total Truth PVs: {total_truth_pvs}")
    print(f"Total Clean: {total_clean}, Rate: {clean_rate:.4f}")
    print(f"Total Merged: {total_merged}, Rate: {merged_rate:.4f}")
    print(f"Total Split: {total_split}, Rate: {split_rate:.4f}")
    print(f"Total Fake: {total_fake}, Rate: {fake_rate:.4f}")

    return total_results, total_reco_pv_info


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for GNN TTVA evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate GNN TTVA on reconstructed PVs"
    )
    parser.add_argument(
        "-r",
        "--reco-graph-path",
        required=True,
        type=str,
        help="Path to reco graphs (.pyt from graph_construction)",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        type=str,
        help="Path to ATLAS HDF5 file with truth information",
    )
    parser.add_argument(
        "-i",
        "--indices",
        required=True,
        type=str,
        help="Path to event indices (.npy or pickled list)",
    )
    parser.add_argument(
        "-w",
        "--model-weights",
        required=True,
        type=str,
        help="Path to saved model weights (.pt)",
    )
    parser.add_argument(
        "-e",
        "--eval-method",
        required=True,
        type=str,
        choices=["MaxScore", "Threshold"],
        help="Evaluation method for GNN output",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=GNN_SCORE_THRESHOLD,
        type=float,
        help="Score threshold value (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--device-id",
        required=True,
        type=int,
        help="CUDA device id (use -1 for CPU)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        type=str,
        help="Directory for result files (default: current directory)",
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
    """CLI entry point: load data, run evaluation, save results as .npy."""
    args = _parse_args()

    # Device setup (-1 = CPU)
    if args.device_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Model
    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    pretrained = torch.load(args.model_weights, map_location=device)
    model.load_state_dict(pretrained)
    model.to(device)

    # Data (weights_only=False: graphs are pickled HeteroData, not weights)
    reco_graphs: list[HeteroData] = torch.load(args.reco_graph_path, weights_only=False)

    # Load event indices (.npy or pickled list, e.g. qibin_test_main_indices_v2.p)
    indices_array = load_event_indices(args.indices)
    event_keys = [f"Event{int(idx)}" for idx in indices_array]

    if args.max_events is not None:
        event_keys = event_keys[: args.max_events]
        reco_graphs = reco_graphs[: args.max_events]

    print(f"filepath: {args.filepath}")

    total_results, total_reco_pv_info_list = evaluate_gnn(
        model=model,
        reco_graphs=reco_graphs,
        h5_path=args.filepath,
        event_keys=event_keys,
        eval_method=args.eval_method,
        threshold=args.threshold,
        device=device,
    )

    # Save results as .npy
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"total_results_{args.eval_method}.npy"
    info_path = out_dir / f"total_reco_pv_info_list_{args.eval_method}.npy"
    np.save(results_path, np.array(total_results, dtype=object), allow_pickle=True)
    np.save(
        info_path, np.array(total_reco_pv_info_list, dtype=object), allow_pickle=True
    )

    print(
        f"Done evaluating GNN model on reco PVs! "
        f"Results saved to {results_path} and {info_path}."
    )


if __name__ == "__main__":
    main()
