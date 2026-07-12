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

import math
from pathlib import Path

import numba
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

from pv_finder.utils.constants import (
    BIN_WIDTH_MM,
    BINS_PER_MM,
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
# Indices loading (.npy or pickled list)
# ---------------------------------------------------------------------------
def load_event_indices(indices_path: str | Path) -> list[int]:
    """Load event indices from a .npy file or a pickled list (.p/.pkl).

    The original atlas_pvfinder pipeline used pickled index lists
    (e.g. qibin_test_main_indices_v2.p); newer files use .npy.
    """
    path = Path(indices_path)
    if path.suffix == ".npy":
        return [int(x) for x in np.load(path)]
    import pickle

    with open(path, "rb") as f:
        return [int(x) for x in pickle.load(f)]


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
def create_training_graph(
    event_data: dict[str, np.ndarray],
    knn: int | None = None,
    pv_res_all: np.ndarray | None = None,
) -> HeteroData:
    """Build a heterogeneous graph from a single MC event with truth labels.

    Expects event_data dict with keys:
        z_0, d_0, sig_z_0, sig_d_0, tracks_event_stack,
        pv_loc_z, pv_ntracks, pv_assoctracks, pv_type

    Args:
        event_data: Per-event arrays (see above).
        knn: If set, connect each track only to its knn nearest PVs in |dz|
            instead of all PVs (required at high pileup: fully-connected
            mu=200 events have ~117k edges). None = fully connected
            (backward-compatible default; bit-exact with earlier builds).
        pv_res_all: Per-PV z resolution in mm. None = Run-3 fit constants
            via compute_pv_sigma (backward-compatible default). Pass a
            custom array to use e.g. the HL-LHC resolution preset.

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

    if pv_res_all is None:
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

    # Create edge index: fully connected, or k nearest PVs per track in |dz|
    num_tracks = len(z_0_event)
    num_good_pvs = len(pv_loc_z_event)

    if knn is not None and 0 < knn < num_good_pvs:
        dz_matrix = np.abs(z_0_event[:, np.newaxis] - pv_loc_z_event[np.newaxis, :])
        nearest = np.argpartition(dz_matrix, knn - 1, axis=1)[:, :knn]
        nearest = np.sort(nearest, axis=1)  # deterministic edge ordering
        track_indices_flat = np.repeat(np.arange(num_tracks), knn)
        pv_indices_flat = nearest.flatten()
    else:
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
