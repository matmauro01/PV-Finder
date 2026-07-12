"""Per-event vertex classification for GNN TTVA evaluation.

Edge selection (top-k per track / global threshold) and Clean/Merged/
Split/Fake categorization of reconstructed PVs against truth associations.

Split out of evaluate_ttva.py to respect the 500-line file limit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

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
