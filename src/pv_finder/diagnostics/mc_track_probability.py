"""GNN track-to-vertex association score distribution on MC data.

Loads events directly from the MC track_associations HDF5 file, builds
heterogeneous graphs using truth PV positions, runs the GNN, and plots the
distribution of per-edge association scores.

Unlike the Run3 version, no PVF inference step is needed: truth PV positions
(pv_loc_z) are used directly as PV nodes, matching the training setup.

Usage:
    python -m pv_finder.diagnostics.mc_track_probability \\
        --h5 data/monte_carlo/track_associations.h5 \\
        --gnn-weights model_weights/gnn_ttva_epoch100.pyt \\
        --output-dir outputs/mc_track_probability \\
        --nevents 500 \\
        --device-id 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from pv_finder.data.graph_construction import create_training_graph
from pv_finder.models.ttva_gnn import TTVAGATModel
from pv_finder.utils.constants import GNN_SCORE_THRESHOLD, PT_SCALE


def _run_gnn_event(
    gnn_model: TTVAGATModel,
    device: torch.device,
    event_data: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Build graph from MC truth and run GNN. Returns sigmoid scores or None."""
    graph = create_training_graph(event_data)

    if graph["track"].num_nodes == 0 or graph["pv"].num_nodes == 0:
        return None

    graph["track"].batch = torch.zeros(graph["track"].num_nodes, dtype=torch.long)
    graph["pv"].batch = torch.zeros(graph["pv"].num_nodes, dtype=torch.long)

    gnn_model.eval()
    with torch.no_grad():
        logits = gnn_model(graph.to(device))

    return torch.sigmoid(logits).cpu().numpy()


def _plot(
    all_scores: np.ndarray,
    max_scores: np.ndarray,
    output_dir: str,
    threshold: float,
    n_events: int,
) -> None:
    """Save two-panel score distribution figure."""
    bins = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1 — all edge scores
    ax = axes[0]
    ax.hist(all_scores, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(
        threshold,
        color="crimson",
        lw=1.5,
        ls="--",
        label=f"Threshold = {threshold:.2f}",
    )
    frac_above = float((all_scores >= threshold).mean())
    ax.set_xlabel("GNN association score (per edge)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"All edge scores  [{len(all_scores):,} edges, {n_events} events]",
        fontsize=11,
    )
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")
    ax.text(
        0.97,
        0.97,
        f"Fraction ≥ threshold: {frac_above:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    # Panel 2 — max score per track
    ax = axes[1]
    ax.hist(max_scores, bins=bins, color="darkorange", edgecolor="none", alpha=0.85)
    ax.axvline(
        threshold,
        color="crimson",
        lw=1.5,
        ls="--",
        label=f"Threshold = {threshold:.2f}",
    )
    frac_assoc = float((max_scores >= threshold).mean())
    ax.set_xlabel("Max GNN score per track", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Max score per track  [{len(max_scores):,} tracks]", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, ls="--")
    ax.text(
        0.97,
        0.97,
        f"Fraction ≥ threshold: {frac_assoc:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    fig.suptitle(
        "GNN track association scores — MC data (truth PVs)", fontsize=13, y=1.01
    )
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out / f"mc_track_probability.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN track association score distribution on MC data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5",
        required=True,
        help="Path to MC track_associations HDF5 file",
    )
    parser.add_argument(
        "--gnn-weights",
        required=True,
        help="Path to GNN state dict (.pyt)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mc_track_probability",
        help="Directory for output plots and JSON",
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=500,
        help="Number of MC events to process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for event sampling",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=GNN_SCORE_THRESHOLD,
        help="GNN score threshold (shown as vertical line on plots)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="CUDA device id; -1 for CPU",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()

    if args.device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load GNN model ─────────────────────────────────────────────────────────
    print(f"\nLoading GNN model: {args.gnn_weights}")
    gnn_model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    state_dict = torch.load(args.gnn_weights, map_location=device, weights_only=False)
    gnn_model.load_state_dict(state_dict)
    gnn_model.to(device).eval()

    # ── Sample event keys ──────────────────────────────────────────────────────
    print(f"\nOpening H5: {args.h5}")
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.h5, "r") as f:
        all_keys = list(f["recoTrk_d"].keys())
        n_process = min(args.nevents, len(all_keys))
        chosen = sorted(rng.choice(len(all_keys), size=n_process, replace=False))
        selected_keys = [all_keys[i] for i in chosen]
        print(f"  {len(all_keys)} events available; processing {n_process}")

        # ── Main loop ──────────────────────────────────────────────────────────
        all_scores_list: list[np.ndarray] = []
        max_scores_list: list[np.ndarray] = []
        n_skipped = 0

        for key in tqdm(selected_keys, desc="Events"):
            d0 = f["recoTrk_d"][key][:]
            z0 = f["recoTrk_z"][key][:]
            d0e = f["recoTrk_d_err"][key][:]
            z0e = f["recoTrk_z_err"][key][:]
            cov = f["recoTrk_d_z_err"][key][:]
            theta = f["recoTrk_theta"][key][:]
            phi = f["recoTrk_phi"][key][:]
            pt = f["recoTrk_pt"][key][:] / PT_SCALE

            pv_loc_z = f["pv_loc_z"][key][:]
            pv_ntracks = f["pv_ntracks"][key][:]
            pv_assoctracks = f["pv_assoc_tracks"][key][:]
            pv_type = f["pv_type"][key][:]

            if len(z0) == 0 or len(pv_loc_z) == 0:
                n_skipped += 1
                continue

            tracks_stack = np.stack(
                [d0, z0, d0e, z0e, cov, theta, phi, pt], axis=1
            ).astype(np.float32)

            event_data = {
                "z_0": z0,
                "d_0": d0,
                "sig_z_0": z0e,
                "sig_d_0": d0e,
                "tracks_event_stack": tracks_stack,
                "pv_loc_z": pv_loc_z,
                "pv_ntracks": pv_ntracks,
                "pv_assoctracks": pv_assoctracks,
                "pv_type": pv_type,
            }

            scores = _run_gnn_event(gnn_model, device, event_data)
            if scores is None:
                n_skipped += 1
                continue

            scores = np.atleast_1d(scores)
            all_scores_list.append(scores)

            n_tracks, n_pvs = len(z0), len(pv_loc_z)
            if len(scores) == n_tracks * n_pvs:
                max_scores_list.append(scores.reshape(n_tracks, n_pvs).max(axis=1))

    n_done = n_process - n_skipped
    print(f"\n  Events processed: {n_done}  |  skipped: {n_skipped}")

    all_scores = np.concatenate(all_scores_list)
    max_scores = np.concatenate(max_scores_list) if max_scores_list else np.array([])
    print(f"  Total edges: {len(all_scores):,}  |  Total tracks: {len(max_scores):,}")

    _plot(all_scores, max_scores, args.output_dir, args.threshold, n_done)

    summary = {
        "n_events_processed": n_done,
        "n_events_skipped": n_skipped,
        "n_edges_total": int(len(all_scores)),
        "n_tracks_total": int(len(max_scores)),
        "all_scores": {
            "mean": float(all_scores.mean()),
            "std": float(all_scores.std()),
            "fraction_above_threshold": float((all_scores >= args.threshold).mean()),
        },
        "max_scores_per_track": {
            "mean": float(max_scores.mean()) if len(max_scores) else None,
            "std": float(max_scores.std()) if len(max_scores) else None,
            "fraction_above_threshold": (
                float((max_scores >= args.threshold).mean())
                if len(max_scores)
                else None
            ),
        },
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "mc_track_probability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
