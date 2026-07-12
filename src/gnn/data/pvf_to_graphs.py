"""Build TTVA inference graphs from PV-Finder histogram outputs.

Takes full-event 12000-bin PVF histograms, runs peak finding to get candidate
PV nodes (z, height, sigma), and builds fully-connected bipartite track-PV
graphs with create_inference_graph.

Replaces atlas_pvfinder/tracks_to_vertex/pvfinder_output_to_graph.py.
Default peak-finding parameters match the Nov 2025 baseline
(threshold=1e-2, integral_threshold=0.2, min_width=3).

Histogram input formats: .npy, pickled ndarray (.p/.pkl), or HDF5 (--h5-key).
Row i of the histogram array must correspond to entry i of the indices file.

Usage:
    python -m gnn.data.pvf_to_graphs \\
        -r /path/to/pvf_histograms.p \\
        -f /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \\
        -i configs/qibin_test_main_indices_v2.p \\
        -o outputs/<date>/inference_graphs.pt
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from gnn.data.graph_construction import create_inference_graph, load_event_indices
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding import pv_locations_updated_res


def load_histograms(path: str | Path, h5_key: str = "pred_target_y") -> np.ndarray:
    """Load an (n_events, 12000) histogram array from .npy, pickle, or HDF5."""
    path = Path(path)
    if path.suffix == ".npy":
        hists = np.load(path)
    elif path.suffix in (".h5", ".hdf5"):
        with h5py.File(path, "r") as f:
            hists = f[h5_key][:]
    else:  # pickled ndarray (.p/.pkl)
        with open(path, "rb") as f:
            hists = pickle.load(f)
    hists = np.asarray(hists)
    if hists.ndim != 2:
        msg = f"Expected 2-D histogram array, got shape {hists.shape}"
        raise ValueError(msg)
    return hists


def build_inference_graphs(
    histograms: np.ndarray,
    tracks_h5_path: str | Path,
    event_keys: list[str],
    threshold: float,
    integral_threshold: float,
    min_width: int,
) -> list[HeteroData]:
    """Peak-find each histogram and build one inference graph per event.

    Args:
        histograms: (n_events, n_bins) PVF output histograms, aligned with
            event_keys.
        tracks_h5_path: Event-keyed ATLAS HDF5 file with recoTrk_* datasets.
        event_keys: HDF5 event keys (e.g. ["Event47450", ...]).
        threshold: Peak-finding bin threshold.
        integral_threshold: Minimum region integral to record a PV.
        min_width: Minimum region width in bins.

    Returns:
        List of HeteroData graphs (one per event, same order as event_keys).
    """
    if len(histograms) != len(event_keys):
        msg = (
            f"histograms ({len(histograms)}) and event_keys "
            f"({len(event_keys)}) length mismatch"
        )
        raise ValueError(msg)

    graphs: list[HeteroData] = []

    with h5py.File(str(tracks_h5_path), "r") as f:
        d_0 = f["recoTrk_d"]
        z_0 = f["recoTrk_z"]
        sig_d_0 = f["recoTrk_d_err"]
        sig_z_0 = f["recoTrk_z_err"]
        sig_d_0_z_0 = f["recoTrk_d_z_err"]
        pt = f["recoTrk_pt"]
        theta = f["recoTrk_theta"]
        phi = f["recoTrk_phi"]

        for i, event_key in enumerate(tqdm(event_keys)):
            d_0_event = d_0[event_key][:]
            z_0_event = z_0[event_key][:]
            sig_d_0_event = sig_d_0[event_key][:]
            sig_z_0_event = sig_z_0[event_key][:]
            sig_d_0_z_0_event = sig_d_0_z_0[event_key][:]
            pt_event = pt[event_key][:] / PT_SCALE
            theta_event = theta[event_key][:]
            phi_event = phi[event_key][:]

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

            pred_z, pred_heights, _, pred_sigmas = pv_locations_updated_res(
                histograms[i], threshold, integral_threshold, min_width
            )

            graphs.append(
                create_inference_graph(
                    z_0_event=z_0_event,
                    d_0_event=d_0_event,
                    sig_z_0_event=sig_z_0_event,
                    sig_d_0_event=sig_d_0_event,
                    tracks_event_stack=tracks_event_stack,
                    pred_z=pred_z,
                    pred_heights=pred_heights,
                    pred_sigmas=pred_sigmas,
                )
            )

    print(f"Built {len(graphs)} inference graphs.")
    return graphs


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build TTVA inference graphs from PV-Finder histograms"
    )
    parser.add_argument(
        "-r",
        "--histograms",
        required=True,
        type=str,
        help="PVF output histograms (.npy, .p/.pkl pickle, or .h5)",
    )
    parser.add_argument(
        "--h5-key",
        default="pred_target_y",
        type=str,
        help="Dataset key when --histograms is HDF5 (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        required=True,
        type=str,
        help="Event-keyed ATLAS HDF5 file with recoTrk_* track features",
    )
    parser.add_argument(
        "-i",
        "--indices",
        required=True,
        type=str,
        help="Event indices aligned with histogram rows (.npy or pickle)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        help="Output .pt file for the graph list",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=1e-2,
        type=float,
        help="Peak-finding bin threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--integral-threshold",
        default=0.2,
        type=float,
        help="Minimum region integral (default: %(default)s, Nov 2025 baseline)",
    )
    parser.add_argument(
        "--min-width",
        default=3,
        type=int,
        help="Minimum region width in bins (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--max-events",
        default=None,
        type=int,
        help="Process only the first N events (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    histograms = load_histograms(args.histograms, args.h5_key)
    indices = load_event_indices(args.indices)
    event_keys = [f"Event{idx}" for idx in indices]

    if len(event_keys) != len(histograms):
        print(
            f"Warning: indices ({len(event_keys)}) vs histograms "
            f"({len(histograms)}) mismatch — truncating to the shorter."
        )
        n = min(len(event_keys), len(histograms))
        event_keys, histograms = event_keys[:n], histograms[:n]

    if args.max_events is not None:
        event_keys = event_keys[: args.max_events]
        histograms = histograms[: args.max_events]

    graphs = build_inference_graphs(
        histograms,
        args.filepath,
        event_keys,
        args.threshold,
        args.integral_threshold,
        args.min_width,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, out)
    print(f"Saved {len(graphs)} graphs to {out}")


if __name__ == "__main__":
    main()
