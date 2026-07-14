"""Build TTVA truth-training graphs directly from a PVFinderData ROOT ntuple.

For samples without an event-keyed HDF5 (e.g. HL-LHC PU200), reads
RecoTrack_* features and TruthVertex_* associations with uproot and builds
HeteroData graphs via create_training_graph.

High-pileup defaults: kNN edge construction (each track connects to its
--knn nearest truth PVs in |dz|; fully-connected mu=200 events would have
~117k edges) and the 'hllhc' vertex-resolution preset for PV-node heights
and edge significances. Coverage measured on this sample (2026-07-12,
200 events): k=20 retains 99.50% of true track-PV edges.

Each graph additionally stores data['track'].truth_pv — the per-track true
PV index (-1 if none) — so downstream evaluation stays exact even for
tracks whose true edge was dropped by kNN selection.

Usage:
    python -m gnn.data.root_to_graphs \\
        --input data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root \\
        --output data/run4/ttva_graphs/pu200_truth_k20_30k.pt \\
        --max-events 30000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.data.graph_augmentation import AugmentationParams, augment_event
from gnn.data.graph_construction import (
    compute_truth_pv_heights,
    create_training_graph,
)
from pv_finder.data.resolution_presets import (
    DEFAULT_RESOLUTION_PRESET,
    RESOLUTION_PRESETS,
)
from pv_finder.utils.constants import BIN_WIDTH_MM, PT_SCALE, PV_MIN_TRACKS

TRACK_BRANCHES = [
    "RecoTrack_d0",
    "RecoTrack_z0",
    "RecoTrack_ErrD0",
    "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0",
    "RecoTrack_theta",
    "RecoTrack_phi",
    "RecoTrack_pT",
]
TRUTH_BRANCHES = [
    "TruthVertex_z",
    "TruthVertex_nTracks",
    "TruthVertex_assocTracks",
    "TruthVertex_type",
]


def compute_pv_sigma_preset(
    ntracks: np.ndarray, res_a: float, res_b: float, res_c: float
) -> np.ndarray:
    """sigma_z(n) = A * n^(-B) + C [mm]; BIN_WIDTH_MM for masked (n < 2) PVs."""
    ntracks = ntracks.astype(np.float64)
    powered = np.power(ntracks, -res_b, out=np.ones_like(ntracks), where=ntracks > 0)
    sigma = res_a * powered + res_c
    return np.where(ntracks < PV_MIN_TRACKS, BIN_WIDTH_MM, sigma)


def build_graph_from_event(
    event: dict,
    knn: int | None,
    res_params: tuple[float, float, float],
    augmenter: tuple[AugmentationParams, np.random.Generator, float] | None = None,
) -> HeteroData:
    """Build one truth-training graph from uproot event arrays.

    Args:
        augmenter: Optional (params, rng, aug_prob) triple. With probability
            aug_prob the event is made chain-like (see graph_augmentation);
            otherwise (and when None) the pristine truth graph is built.
    """
    d0 = ak.to_numpy(event["RecoTrack_d0"]).astype(np.float64)
    z0 = ak.to_numpy(event["RecoTrack_z0"]).astype(np.float64)
    err_d0 = ak.to_numpy(event["RecoTrack_ErrD0"]).astype(np.float64)
    err_z0 = ak.to_numpy(event["RecoTrack_ErrZ0"]).astype(np.float64)
    err_d0z0 = ak.to_numpy(event["RecoTrack_ErrD0Z0"]).astype(np.float64)
    theta = ak.to_numpy(event["RecoTrack_theta"]).astype(np.float64)
    phi = ak.to_numpy(event["RecoTrack_phi"]).astype(np.float64)
    pt = ak.to_numpy(event["RecoTrack_pT"]).astype(np.float64) / PT_SCALE

    pv_z = ak.to_numpy(event["TruthVertex_z"]).astype(np.float64)
    pv_ntracks = ak.to_numpy(event["TruthVertex_nTracks"]).astype(np.float64)
    pv_type = ak.to_numpy(event["TruthVertex_type"]).astype(np.float64)

    # Flatten jagged per-vertex track lists to the (flat, ntracks) convention
    assoc_jagged = event["TruthVertex_assocTracks"]
    pv_assoctracks = ak.to_numpy(ak.flatten(assoc_jagged)).astype(np.int64)

    tracks_event_stack = np.stack(
        [d0, z0, err_d0, err_z0, err_d0z0, theta, phi, pt]
    ).T.astype(np.float32)

    event_data = {
        "z_0": z0,
        "d_0": d0,
        "sig_z_0": err_z0,
        "sig_d_0": err_d0,
        "tracks_event_stack": tracks_event_stack,
        "pv_loc_z": pv_z,
        "pv_ntracks": pv_ntracks,
        "pv_assoctracks": pv_assoctracks,
        "pv_type": pv_type,
    }

    pv_res_all = compute_pv_sigma_preset(pv_ntracks, *res_params)
    pv_heights = None
    if augmenter is not None:
        params, rng, aug_prob = augmenter
        if rng.random() < aug_prob:
            recipe_heights = compute_truth_pv_heights(pv_z, pv_res_all)
            event_data, pv_res_all, pv_heights, _ = augment_event(
                event_data, recipe_heights, params, rng
            )

    graph = create_training_graph(
        event_data, knn=knn, pv_res_all=pv_res_all, pv_heights_override=pv_heights
    )

    # Per-track true PV index (-1 = no truth association). Exact even when
    # kNN selection drops the true edge from the graph. Uses the possibly
    # augmented arrays so indices match the graph's PV nodes (tracks of
    # dropped vertices correctly fall back to -1).
    final_ntracks = event_data["pv_ntracks"]
    final_assoc = event_data["pv_assoctracks"]
    track_truth_pv = np.full(len(z0), -1, dtype=np.int64)
    truth_pv_idx = np.repeat(np.arange(len(final_ntracks)), final_ntracks.astype(int))
    track_truth_pv[final_assoc] = truth_pv_idx
    graph["track"].truth_pv = torch.from_numpy(track_truth_pv)

    return graph


def build_graphs_from_root(
    input_path: str | Path,
    tree_name: str,
    knn: int | None,
    res_params: tuple[float, float, float],
    max_events: int | None = None,
    start_event: int = 0,
    chunk_size: int = 500,
    augmenter: tuple[AugmentationParams, np.random.Generator, float] | None = None,
) -> list[HeteroData]:
    """Build graphs for all (or max_events) events of a ROOT ntuple."""
    graphs: list[HeteroData] = []
    tree = uproot.open(input_path)[tree_name]
    n_total = tree.num_entries
    entry_stop = (
        n_total if max_events is None else min(n_total, start_event + max_events)
    )

    pbar = tqdm(total=entry_stop - start_event)
    for chunk in tree.iterate(
        TRACK_BRANCHES + TRUTH_BRANCHES,
        entry_start=start_event,
        entry_stop=entry_stop,
        step_size=chunk_size,
    ):
        for event in chunk:
            graphs.append(build_graph_from_event(event, knn, res_params, augmenter))
            pbar.update(1)
    pbar.close()

    print(f"Built {len(graphs)} graphs.")
    return graphs


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build TTVA truth graphs from a PVFinderData ROOT ntuple"
    )
    parser.add_argument("--input", required=True, type=str, help="ROOT file path")
    parser.add_argument(
        "--tree",
        default="PVFinderData",
        type=str,
        help="Tree name (default: %(default)s)",
    )
    parser.add_argument(
        "--output", required=True, type=str, help="Output .pt file for graph list"
    )
    parser.add_argument(
        "--knn",
        default=20,
        type=int,
        help="Connect each track to its K nearest PVs in |dz|; 0 = fully "
        "connected (default: %(default)s)",
    )
    parser.add_argument(
        "--resolution-preset",
        default=DEFAULT_RESOLUTION_PRESET,
        choices=sorted(RESOLUTION_PRESETS),
        help="sigma_z(n) fit constants for PV heights/significances "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--max-events", default=None, type=int, help="Events to process (default: all)"
    )
    parser.add_argument(
        "--start-event", default=0, type=int, help="First event index (default: 0)"
    )
    parser.add_argument(
        "--augment-params",
        default=None,
        type=str,
        help="Directory with augmentation_params.json + gap_decomposition.json; "
        "enables chain-like augmentation (default: off)",
    )
    parser.add_argument(
        "--aug-prob",
        default=0.7,
        type=float,
        help="Per-event probability of augmentation (default: %(default)s)",
    )
    parser.add_argument("--seed", default=42, type=int, help="Augmentation RNG seed")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    res_params = RESOLUTION_PRESETS[args.resolution_preset]
    knn = args.knn if args.knn > 0 else None

    augmenter = None
    if args.augment_params is not None:
        augmenter = (
            AugmentationParams(args.augment_params),
            np.random.default_rng(args.seed),
            args.aug_prob,
        )

    print(
        f"Building graphs: knn={knn}, resolution preset "
        f"'{args.resolution_preset}' (A, B, C) = {res_params}, "
        f"augment={'off' if augmenter is None else f'p={args.aug_prob}'}"
    )
    graphs = build_graphs_from_root(
        args.input,
        args.tree,
        knn,
        res_params,
        max_events=args.max_events,
        start_event=args.start_event,
        augmenter=augmenter,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, out)
    print(f"Saved {len(graphs)} graphs to {out}")


if __name__ == "__main__":
    main()
