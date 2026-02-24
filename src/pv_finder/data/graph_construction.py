"""Graph construction for Track-to-Vertex Association (TTVA).

Builds heterogeneous bipartite graphs (track <-> PV) for the GNN model.
Two entry points:

  - create_training_graph: from MC truth (H5 file with true PV locations)
  - create_inference_graph: from PV-Finder output (pre-computed peaks)

Migrated from:
  - atlas_pvfinder/tracks_to_vertex/h5_to_graph.py (training)
  - atlas_pvfinder/tracks_to_vertex/pvfinder_output_to_graph.py (inference)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numba
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from tqdm import tqdm

from pv_finder.utils.constants import (
    BIN_WIDTH_MM,
    BINS_PER_MM,
    PT_SCALE,
    PV_MIN_TRACKS,
    PV_RES_A,
    PV_RES_B,
    PV_RES_C,
    Z_MIN,
)

# ---------------------------------------------------------------------------
# Derived binning arrays (matching original ProbRange computation)
# ---------------------------------------------------------------------------
_bins = np.arange(-5, 6)  # +/- 5 neighboring bins
_edges = np.array([-BIN_WIDTH_MM / 2, BIN_WIDTH_MM / 2])
PROB_RANGE: np.ndarray = (
    BIN_WIDTH_MM * _bins[np.newaxis, :] + _edges[:, np.newaxis] + Z_MIN
)


# ---------------------------------------------------------------------------
# Numba-vectorized helpers (kept identical to source)
# ---------------------------------------------------------------------------
@numba.vectorize(nopython=True)
def bin_number(mean: float) -> int:
    """Get bin number given a z value."""
    return int(np.floor((mean - Z_MIN) * BINS_PER_MM))


@numba.vectorize(nopython=True)
def bin_center(zmin: float, zmax: float, nbins: int, ibin: int) -> float:
    """Get z value given a bin number."""
    return ((ibin + 0.5) / nbins) * (zmax - zmin) + zmin


@numba.vectorize(nopython=True)
def compute_pv_sigma(ntrks: float) -> float:
    """Compute PV resolution from track multiplicity.

    sigma_pv(N) = A * N^(-B) + C, fitted in ResolutionFit_ATLAS.ipynb.
    Returns BIN_WIDTH_MM for masked (< PV_MIN_TRACKS) vertices.
    """
    if ntrks < PV_MIN_TRACKS:
        return BIN_WIDTH_MM
    else:
        return PV_RES_A * np.power(ntrks, -1 * PV_RES_B) + PV_RES_C


@numba.vectorize(nopython=True)
def norm_cdf(mu: float, sigma: float, x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


# ---------------------------------------------------------------------------
# Shared edge attribute computation
# ---------------------------------------------------------------------------
def _compute_edge_attributes(
    z_0_event: np.ndarray,
    d_0_event: np.ndarray,
    sig_z_0_event: np.ndarray,
    sig_d_0_event: np.ndarray,
    pv_z: np.ndarray,
    pv_res: np.ndarray,
    track_indices_flat: np.ndarray,
    pv_indices_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edge attributes for track-PV pairs.

    Returns (longitudinal_significance, horizontal_significance, abs_diff_z).
    """
    z0_for_edges = z_0_event[track_indices_flat]
    d0_for_edges = d_0_event[track_indices_flat]
    sig_z0_for_edges = sig_z_0_event[track_indices_flat]
    sig_d0_for_edges = sig_d_0_event[track_indices_flat]

    pv_z_for_edges = pv_z[pv_indices_flat]
    pv_res_for_edges = pv_res[pv_indices_flat]

    diff_z = z0_for_edges - pv_z_for_edges
    sigma_z_combined = np.sqrt(sig_z0_for_edges**2 + pv_res_for_edges**2)

    longitudinal_significance = np.divide(
        diff_z,
        sigma_z_combined,
        out=np.zeros_like(diff_z),
        where=sigma_z_combined != 0,
    )
    horizontal_significance = np.divide(
        d0_for_edges,
        sig_d0_for_edges,
        out=np.zeros_like(d0_for_edges),
        where=sig_d0_for_edges != 0,
    )

    return longitudinal_significance, horizontal_significance, np.abs(diff_z)


# ---------------------------------------------------------------------------
# Training graph construction (from MC truth)
# ---------------------------------------------------------------------------
def create_training_graph(event_data: dict[str, np.ndarray]) -> HeteroData:
    """Build a heterogeneous graph from a single MC event with truth labels.

    Expects event_data dict with keys:
        z_0, d_0, sig_z_0, sig_d_0, tracks_event_stack,
        pv_loc_z, pv_ntracks, pv_assoctracks, pv_type

    Returns a HeteroData graph with ToUndirected applied.
    """
    z_0_event = event_data["z_0"]
    d_0_event = event_data["d_0"]
    sig_z_0_event = event_data["sig_z_0"]
    sig_d_0_event = event_data["sig_d_0"]
    tracks_event_stack = event_data["tracks_event_stack"]

    pv_loc_z_event = event_data["pv_loc_z"]
    pv_ntracks_event = event_data["pv_ntracks"]
    pv_assoctracks_event = event_data["pv_assoctracks"]
    pv_type_event = event_data["pv_type"]

    pv_res_all = compute_pv_sigma(pv_ntracks_event)

    # Get bin numbers for all PVs
    nbins_all = bin_number(pv_loc_z_event)

    # Calculate probabilities and peak vals for all z values
    z_centers_prob = (nbins_all / BINS_PER_MM) + Z_MIN
    z_probRanges_all = z_centers_prob[:, np.newaxis, np.newaxis] + PROB_RANGE

    probValues_all = norm_cdf(
        pv_loc_z_event[:, np.newaxis, np.newaxis],
        pv_res_all[:, np.newaxis, np.newaxis],
        z_probRanges_all,
    )

    populate_all = probValues_all[:, 1, :] - probValues_all[:, 0, :]

    scaling_factor = 0.15 / pv_res_all[:, np.newaxis]
    condition = scaling_factor > 1

    populate_all = np.where(condition, scaling_factor * populate_all, populate_all)

    pv_heights_all = np.max(populate_all, axis=1)

    # Construct PV nodes
    pv_event_features = np.stack((pv_loc_z_event, pv_heights_all)).T.astype(np.float32)

    # Create ground truth associations
    truth_pv_indices_all = np.repeat(
        np.arange(len(pv_ntracks_event)), pv_ntracks_event.astype(int)
    )
    truth_track_indices_all = pv_assoctracks_event

    # Create edge index and edge attributes
    num_tracks = len(z_0_event)
    num_good_pvs = len(pv_loc_z_event)

    track_indices, pv_indices = np.meshgrid(
        np.arange(num_tracks), np.arange(num_good_pvs), indexing="ij"
    )
    track_indices_flat = track_indices.flatten()
    pv_indices_flat = pv_indices.flatten()

    longitudinal_significance, horizontal_significance, abs_diff_z = (
        _compute_edge_attributes(
            z_0_event,
            d_0_event,
            sig_z_0_event,
            sig_d_0_event,
            pv_loc_z_event,
            pv_res_all,
            track_indices_flat,
            pv_indices_flat,
        )
    )

    truth_set = set(zip(truth_track_indices_all, truth_pv_indices_all))
    truth_edges = np.array(
        [
            1 if (t, p) in truth_set else 0
            for t, p in zip(track_indices_flat, pv_indices_flat)
        ],
        dtype=np.int64,
    )

    # Construct HeteroData graph
    data = HeteroData()
    data["track"].x = torch.from_numpy(tracks_event_stack).float()
    data["pv"].x = torch.from_numpy(pv_event_features).float()
    data["pv"].y_hs = torch.from_numpy(pv_type_event).float()

    edge_stacking = np.stack([track_indices_flat, pv_indices_flat])
    data[("track", "to", "pv")].edge_index = torch.from_numpy(edge_stacking).long()
    data[("track", "to", "pv")].y = torch.from_numpy(truth_edges).float()

    edge_attr_tensor = torch.from_numpy(
        np.stack(
            [longitudinal_significance, horizontal_significance, abs_diff_z],
            axis=1,
        )
    ).float()
    data[("track", "to", "pv")].edge_attr = edge_attr_tensor

    return T.ToUndirected()(data)


# ---------------------------------------------------------------------------
# Inference graph construction (from PV-Finder output)
# ---------------------------------------------------------------------------
def create_inference_graph(
    z_0_event: np.ndarray,
    d_0_event: np.ndarray,
    sig_z_0_event: np.ndarray,
    sig_d_0_event: np.ndarray,
    tracks_event_stack: np.ndarray,
    pred_z: np.ndarray,
    pred_heights: np.ndarray,
    pred_sigmas: np.ndarray,
) -> HeteroData:
    """Build a heterogeneous graph from pre-computed PV-Finder predictions.

    Unlike create_training_graph, this does NOT embed peak finding.
    The caller provides pre-computed pred_z, pred_heights, pred_sigmas.

    Returns a HeteroData graph with ToUndirected applied.
    """
    num_tracks = len(z_0_event)
    num_pvs = len(pred_z)

    track_indices, pv_indices = np.meshgrid(
        np.arange(num_tracks), np.arange(num_pvs), indexing="ij"
    )
    track_indices_flat = track_indices.flatten()
    pv_indices_flat = pv_indices.flatten()

    pv_event_features = np.stack((pred_z, pred_heights)).T

    longitudinal_significance, horizontal_significance, abs_diff_z = (
        _compute_edge_attributes(
            z_0_event,
            d_0_event,
            sig_z_0_event,
            sig_d_0_event,
            pred_z,
            pred_sigmas,
            track_indices_flat,
            pv_indices_flat,
        )
    )

    data = HeteroData()
    data["track"].x = torch.from_numpy(tracks_event_stack).float()
    data["pv"].x = torch.from_numpy(pv_event_features).float()

    edge_stacking = np.stack([track_indices_flat, pv_indices_flat])
    data[("track", "to", "pv")].edge_index = torch.from_numpy(edge_stacking).long()

    edge_attr_tensor = torch.from_numpy(
        np.stack(
            [longitudinal_significance, horizontal_significance, abs_diff_z],
            axis=1,
        )
    ).float()
    data[("track", "to", "pv")].edge_attr = edge_attr_tensor

    return T.ToUndirected()(data)


# ---------------------------------------------------------------------------
# Batch processing: build training graphs from H5
# ---------------------------------------------------------------------------
def build_training_graphs_from_h5(
    filepath: str,
    indices_path: str,
    nevents: int | None = None,
) -> list[HeteroData]:
    """Build training graphs for all events in an H5 file.

    Args:
        filepath: Path to ATLAS HDF5 file (from CreatingTargetHistogram.py).
        indices_path: Path to .npy file with event indices.
        nevents: Max number of events to process (None = all valid events).

    Returns:
        List of HeteroData graphs, one per valid event.
    """
    import h5py

    event_data_list: list[HeteroData] = []

    with h5py.File(filepath, "r") as dataFile:
        d_0 = dataFile["recoTrk_d"]
        z_0 = dataFile["recoTrk_z"]
        sig_d_0 = dataFile["recoTrk_d_err"]
        sig_z_0 = dataFile["recoTrk_z_err"]
        sig_d_0_z_0 = dataFile["recoTrk_d_z_err"]
        pt = dataFile["recoTrk_pt"]
        theta = dataFile["recoTrk_theta"]
        phi = dataFile["recoTrk_phi"]

        pv_loc_z = dataFile["pv_loc_z"]
        pv_assoctracks = dataFile["pv_assoc_tracks"]
        pv_ntracks = dataFile["pv_ntracks"]
        pv_type = dataFile["pv_type"]

        # Get available events from HDF5 file
        print("Scanning HDF5 file for available events...")
        available_events_in_hdf5 = set(d_0.keys())
        print(f"Found {len(available_events_in_hdf5)} events in HDF5 file.")

        # Load indices file (.npy)
        pubnote_indices = [int(x) for x in np.load(indices_path)]

        # Filter indices to only include events that exist in HDF5
        requested_event_keys = [f"Event{i}" for i in pubnote_indices]
        valid_event_keys = [
            ek for ek in requested_event_keys if ek in available_events_in_hdf5
        ]

        # Report on missing events
        missing_events = set(requested_event_keys) - set(valid_event_keys)
        if missing_events:
            print(
                f"Warning: {len(missing_events)} events from indices file "
                "are not in HDF5 file."
            )
            print(f"First few missing: {sorted(list(missing_events))[:10]}")

        # Auto-detect number of events if not specified
        if nevents is None or nevents <= 0:
            nevents = len(valid_event_keys)
            print(f"Processing {nevents} valid events.")
        else:
            valid_event_keys = valid_event_keys[:nevents]
            print(
                f"Processing first {len(valid_event_keys)} valid events "
                f"(requested: {nevents})."
            )

        for event_key in tqdm(valid_event_keys):
            # Node: Tracks
            d_0_event = d_0[event_key][:]
            z_0_event = z_0[event_key][:]
            sig_d_0_event = sig_d_0[event_key][:]
            sig_z_0_event = sig_z_0[event_key][:]
            sig_d_0_z_0_event = sig_d_0_z_0[event_key][:]
            pt_event = pt[event_key][:] / PT_SCALE
            theta_event = theta[event_key][:]
            phi_event = phi[event_key][:]

            # Node: PVs
            pv_loc_z_event = pv_loc_z[event_key][:]

            # Label/Edge: Track-Vertex Associativity
            pv_assoctracks_event = pv_assoctracks[event_key][:]
            pv_ntracks_event = pv_ntracks[event_key][:]
            pv_type_event = pv_type[event_key][:]

            tracks_event_stack = np.stack(
                [
                    d_0_event,
                    z_0_event,
                    sig_d_0_event,
                    sig_z_0_event,
                    sig_d_0_z_0_event,
                    theta_event,
                    phi_event,
                    pt_event,
                ]
            ).T.astype(np.float32)

            current_event_data = {
                "z_0": z_0_event,
                "d_0": d_0_event,
                "sig_z_0": sig_z_0_event,
                "sig_d_0": sig_d_0_event,
                "tracks_event_stack": tracks_event_stack,
                "pv_loc_z": pv_loc_z_event,
                "pv_ntracks": pv_ntracks_event,
                "pv_assoctracks": pv_assoctracks_event,
                "pv_type": pv_type_event,
            }

            graph = create_training_graph(current_event_data)

            if graph is not None:
                event_data_list.append(graph)

    print(f"Finished constructing {len(event_data_list)} graphs.")
    return event_data_list


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training graph construction."""
    parser = argparse.ArgumentParser(
        description=(
            "Construct training graphs from an ATLAS HDF5 file "
            "(from CreatingTargetHistogram.py)."
        )
    )
    parser.add_argument(
        "-f",
        "--filepath",
        help="Path to input ATLAS HDF5 file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--indices",
        help="Path to .npy file with event indices",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--nevents",
        help="Number of events to process (default: all valid)",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output filepath (.pt) for constructed graphs",
        type=str,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    event_data_list = build_training_graphs_from_h5(
        args.filepath, args.indices, args.nevents
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(event_data_list, output_path)
    print(f"Saved {len(event_data_list)} graphs to {output_path}")
