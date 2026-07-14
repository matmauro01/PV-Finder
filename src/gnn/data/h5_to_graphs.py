"""Build TTVA truth-training graphs from an event-keyed ATLAS HDF5 file.

CLI wrapper around create_training_graph for MC samples with truth
associations in HDF5 form (e.g. recoTracks_incamvfassoc.h5).
For ROOT ntuples (HL-LHC PU200) use gnn.data.root_to_graphs instead.

Split out of graph_construction.py to respect the 500-line file limit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.data.graph_augmentation import AugmentationParams, augment_event
from gnn.data.graph_construction import (
    compute_pv_sigma,
    compute_truth_pv_heights,
    create_training_graph,
    load_event_indices,
)
from pv_finder.utils.constants import PT_SCALE


# ---------------------------------------------------------------------------
# Batch processing: build training graphs from H5
# ---------------------------------------------------------------------------
def build_training_graphs_from_h5(
    filepath: str,
    indices_path: str,
    nevents: int | None = None,
    start_event: int = 0,
    augmenter: tuple[AugmentationParams, np.random.Generator, float] | None = None,
) -> list[HeteroData]:
    """Build training graphs for all events in an H5 file.

    Args:
        filepath: Path to ATLAS HDF5 file (from CreatingTargetHistogram.py).
        indices_path: Path to event indices (.npy or pickled list).
        nevents: Max number of events to process (None = all valid events).
        start_event: First position in the indices list (for sharding).
        augmenter: Optional (params, rng, aug_prob) triple for chain-like
            augmentation (see gnn.data.graph_augmentation).

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

        # Load indices file (.npy or pickled list)
        pubnote_indices = load_event_indices(indices_path)

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

        valid_event_keys = valid_event_keys[start_event:]
        # Auto-detect number of events if not specified
        if nevents is None or nevents <= 0:
            nevents = len(valid_event_keys)
            print(f"Processing {nevents} valid events from {start_event}.")
        else:
            valid_event_keys = valid_event_keys[:nevents]
            print(
                f"Processing {len(valid_event_keys)} valid events "
                f"from position {start_event}."
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

            pv_res_all = None
            pv_heights = None
            if augmenter is not None:
                params, rng, aug_prob = augmenter
                if rng.random() < aug_prob:
                    pv_res_all = compute_pv_sigma(pv_ntracks_event)
                    recipe_h = compute_truth_pv_heights(pv_loc_z_event, pv_res_all)
                    current_event_data, pv_res_all, pv_heights, _ = augment_event(
                        current_event_data, recipe_h, params, rng
                    )

            graph = create_training_graph(
                current_event_data,
                pv_res_all=pv_res_all,
                pv_heights_override=pv_heights,
            )

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
        help="Path to event indices (.npy or pickled list)",
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
    parser.add_argument(
        "--start-event",
        default=0,
        type=int,
        help="First position in the indices list (default: 0)",
    )
    parser.add_argument(
        "--augment-params",
        default=None,
        type=str,
        help="Directory with augmentation_params.json + gap_decomposition.json",
    )
    parser.add_argument("--aug-prob", default=0.7, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    augmenter = None
    if args.augment_params is not None:
        augmenter = (
            AugmentationParams(args.augment_params),
            np.random.default_rng(args.seed),
            args.aug_prob,
        )
    event_data_list = build_training_graphs_from_h5(
        args.filepath,
        args.indices,
        args.nevents,
        start_event=args.start_event,
        augmenter=augmenter,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(event_data_list, output_path)
    print(f"Saved {len(event_data_list)} graphs to {output_path}")


if __name__ == "__main__":
    main()
