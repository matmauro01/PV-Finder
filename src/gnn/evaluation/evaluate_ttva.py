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
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.utils.constants import GNN_SCORE_THRESHOLD, PURITY_THRESHOLD


def get_top_k_associations(
    pred_scores: torch.Tensor | np.ndarray,
    edge_index_track_pv: np.ndarray,
    k: int = 1,
    threshold: float = GNN_SCORE_THRESHOLD,
) -> np.ndarray:
    """Select top-k PV associations per track from predicted edge scores.

    For each track, keeps at most *k* edges whose score is highest, but only
    if the maximum score for that track meets *threshold*.

    Args:
        pred_scores: Per-edge association scores (after sigmoid).
        edge_index_track_pv: (2, n_edges) array of [track_idx, pv_idx].
        k: Number of top associations to keep per track.
        threshold: Minimum max-score required to keep any association.

    Returns:
        Boolean mask of shape (n_edges,) -- True for selected edges.
    """
    track_indices = edge_index_track_pv[0]

    if isinstance(pred_scores, torch.Tensor):
        pred_scores = pred_scores.cpu().numpy()
    if isinstance(track_indices, torch.Tensor):
        track_indices = track_indices.cpu().numpy()

    # Sort indices to group by track
    sort_indices = np.argsort(track_indices)
    sorted_tracks = track_indices[sort_indices]

    # Find track boundaries using searchsorted
    unique_tracks = np.unique(track_indices)
    boundaries = np.searchsorted(sorted_tracks, unique_tracks, side="right")
    boundaries = np.concatenate([[0], boundaries])

    output = np.zeros_like(pred_scores, dtype=bool)

    for i in range(len(unique_tracks)):
        start = boundaries[i]
        end = boundaries[i + 1]

        track_edge_indices = sort_indices[start:end]
        track_scores = pred_scores[track_edge_indices]
        max_score = np.max(track_scores)

        # Only proceed if the highest score meets the threshold
        if max_score >= threshold:
            num_to_select = min(k, len(track_scores))
            if num_to_select > 0:
                top_local_indices = np.argpartition(track_scores, -num_to_select)[
                    -num_to_select:
                ]
                global_indices = track_edge_indices[top_local_indices]
                output[global_indices] = True

    return output


def categorize_event(
    model: TTVAGATModel,
    reco_graph_event: HeteroData,
    tracks_event_stack: np.ndarray,
    pv_loc_z_event: np.ndarray,
    pv_assoctracks_event: np.ndarray,
    pv_ntracks_event: np.ndarray,
    eval_method: str,
    threshold: float,
    device: torch.device,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Classify each reco PV in one event as Clean/Merged/Split/Fake.

    A reco PV is:
      - **Clean** if its dominant truth PV contributes >= PURITY_THRESHOLD of
        matched tracks.
      - **Merged** if the dominant contribution is below that threshold.
      - **Split** if another reco PV already claimed the same dominant truth
        PV with higher sum(pT^2).
      - **Fake** if no truth PV matches any of its tracks.

    Args:
        model: Trained TTVAGATModel (already on *device*).
        reco_graph_event: Single-event HeteroData graph.
        tracks_event_stack: (3, n_tracks) array -- row 0=d0, 1=z0, 2=pt.
        pv_loc_z_event: Truth PV z-positions.
        pv_assoctracks_event: Flat array of truth-associated track indices.
        pv_ntracks_event: Number of truth tracks per truth PV.
        eval_method: ``"MaxScore"`` or ``"Threshold"``.
        threshold: Score threshold for edge selection.
        device: Torch device.

    Returns:
        Tuple of (results_list, reco_pv_info_list) where results_list is
        [clean, merged, split, fake, n_reco_pvs, n_truth_pvs].
    """
    # Skip events with no PVs
    if reco_graph_event["pv"].num_nodes == 0:
        return [0, 0, 0, 0, 0, int((pv_ntracks_event >= 2).sum())], []

    graph_edge_index_track, graph_edge_index_pv = reco_graph_event[
        ("track", "to", "pv")
    ].edge_index

    track_batch = torch.zeros(reco_graph_event["track"].num_nodes, dtype=torch.long)
    pv_batch = torch.zeros(reco_graph_event["pv"].num_nodes, dtype=torch.long)
    reco_graph_event["pv"].batch = pv_batch
    reco_graph_event["track"].batch = track_batch

    model.eval()
    with torch.no_grad():
        output_edge = model(reco_graph_event.to(device))

    if eval_method == "MaxScore":
        output_probs = get_top_k_associations(
            torch.sigmoid(output_edge),
            reco_graph_event[("track", "to", "pv")].edge_index.cpu().numpy(),
            k=1,
            threshold=threshold,
        )
    elif eval_method == "Threshold":
        output_probs = torch.sigmoid(output_edge).cpu().numpy() > threshold
    else:
        msg = "Evaluation method must be MaxScore or Threshold!"
        raise ValueError(msg)

    # tracks_event_stack layout: row 0 = d0, row 1 = z0, row 2 = pt
    # The caller stacks only (d0, z0, pt) into a (3, N) array.
    # d0 and z0 are available at [0,:] and [1,:] for downstream use.
    pt_event = tracks_event_stack[2, :]

    truth_track_indices = np.zeros(len(pv_assoctracks_event))
    truth_pv_indices = np.zeros(len(pv_assoctracks_event))

    # Build truth adjacency: truth associations between reco tracks and truth PVs
    counter = 0
    for i in range(len(pv_loc_z_event)):
        n = int(pv_ntracks_event[i])
        truth_track_indices[counter : counter + n] = pv_assoctracks_event[
            counter : counter + n
        ]
        truth_pv_indices[counter : counter + n] = i
        counter += n

    num_reco_pvs_in_event = len(reco_graph_event["pv"].x[:, 0].cpu().numpy())
    truth_pvs_count = int((pv_ntracks_event >= 2).sum())

    reco_pv_info_list: list[dict[str, Any]] = []

    for i in range(num_reco_pvs_in_event):
        # Get all edge indices associated with this reconstructed vertex
        i_pv_indices = (
            torch.nonzero(graph_edge_index_pv == i, as_tuple=False).view(-1).cpu()
        )
        output_pv = output_probs[i_pv_indices]
        i_track_indices = graph_edge_index_track[i_pv_indices]

        # Grab tracks that are associated to reco pv
        matched_tracks = torch.unique(i_track_indices[output_pv], sorted=True).cpu()

        # Calculate sum pT^2 of all tracks associated with PV
        pv_assoc_pt_event = pt_event[matched_tracks]
        sum_pt_sq = (pv_assoc_pt_event**2).sum()

        # Calculate if reco PV is clean, merged, split, or fake
        w_total_reco = output_pv.sum()
        w_pvtruth_in_pvvreco: dict[str, int] = {}

        for j in matched_tracks:
            # Find truth PV each reco track is truth associated with
            truth_index = torch.nonzero(truth_track_indices == j, as_tuple=False).view(
                -1
            )
            truth_pv_num = truth_pv_indices[truth_index]

            if not truth_pv_num.size > 0:
                pv_dict_name = "Fake"
            else:
                pv_dict_name = f"Truth_PV_{int(truth_pv_num)}"

            if pv_dict_name in w_pvtruth_in_pvvreco:
                w_pvtruth_in_pvvreco[pv_dict_name] += 1
            else:
                w_pvtruth_in_pvvreco[pv_dict_name] = 0
                w_pvtruth_in_pvvreco[pv_dict_name] += 1

        current_reco_pv_info: dict[str, Any] = {
            "reco_pv_idx": i,
            "w_total_reco": w_total_reco,
            "contributions": w_pvtruth_in_pvvreco,
            "primary_truth_pv": None,
            "primary_truth_pv_weight": 0,
            "sum_pt2": sum_pt_sq,
            "classification": None,
        }

        if len(matched_tracks) == 0:
            max_key = "Fake"
        else:
            max_key = max(w_pvtruth_in_pvvreco, key=w_pvtruth_in_pvvreco.get)  # type: ignore[arg-type]
            max_value = w_pvtruth_in_pvvreco[max_key]
            current_reco_pv_info["primary_truth_pv"] = max_key
            current_reco_pv_info["primary_truth_pv_weight"] = max_value

        if max_key == "Fake":
            current_reco_pv_info["classification"] = "Fake"
        elif (max_value / w_total_reco) >= PURITY_THRESHOLD:
            current_reco_pv_info["classification"] = "Clean"
        else:
            current_reco_pv_info["classification"] = "Merged"

        reco_pv_info_list.append(current_reco_pv_info)

    # Detect Split vertices: multiple reco PVs claiming the same truth PV
    truth_reco_assoc: dict[str, list[tuple[int, Any, str | None]]] = {}
    for info in reco_pv_info_list:
        primary_truth_id = info["primary_truth_pv"]
        if primary_truth_id != "Fake" and primary_truth_id is not None:
            if primary_truth_id not in truth_reco_assoc:
                truth_reco_assoc[primary_truth_id] = []
            truth_reco_assoc[primary_truth_id].append(
                (info["reco_pv_idx"], info["sum_pt2"], info["classification"])
            )

    for _truth_pv_id, associated_recos in truth_reco_assoc.items():
        if len(associated_recos) > 1:
            associated_recos.sort(key=lambda x: x[1], reverse=True)
            for i in range(1, len(associated_recos)):
                reco_idx_to_split = associated_recos[i][0]
                for info_dict in reco_pv_info_list:
                    if (
                        info_dict["reco_pv_idx"] == reco_idx_to_split
                        and info_dict["classification"] != "Fake"
                    ):
                        info_dict["classification"] = "Split"

    # Count classifications
    event_clean = event_merged = event_fake = event_split = 0
    for info in reco_pv_info_list:
        classification = info["classification"]
        if classification == "Clean":
            event_clean += 1
        elif classification == "Merged":
            event_merged += 1
        elif classification == "Fake":
            event_fake += 1
        elif classification == "Split":
            event_split += 1

    results = [
        event_clean,
        event_merged,
        event_split,
        event_fake,
        num_reco_pvs_in_event,
        truth_pvs_count,
    ]
    return results, reco_pv_info_list


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
    return parser.parse_args()


def main() -> None:
    """CLI entry point: load data, run evaluation, save results as .npy."""
    args = _parse_args()

    # Device setup
    torch.cuda.set_device(args.device_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    results_path = f"total_results_{args.eval_method}.npy"
    info_path = f"total_reco_pv_info_list_{args.eval_method}.npy"
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
