"""GNN track-to-vertex association score distribution on Run3 data.

Runs the full PV-Finder → peak-finding → GNN pipeline on Run3 events from
the pre-extracted NPZ cache and plots the distribution of per-edge GNN
association scores (sigmoid of edge logits).

Two panels are produced:
  1. All edge scores   — one value per (track, PV) pair in the bipartite graph
  2. Max score per track — for each track, the highest score across all PVs

NOTE: The PVF model weights were saved as a pickled module object from the
legacy 'model.autoencoder_models' package.  Pass --legacy-model-path pointing
to the mattia_finder/ directory so that module is importable at load time.

Usage:
    python -m gnn.diagnostics.run3_track_probability \\
        --cache data/run3/cache_file3_2000ev_seed42.npz \\
        --pvf-weights model_weights/tracks2kde_KDE_A_z_epoch180.pyt \\
        --gnn-weights model_weights/gnn_ttva_epoch100.pyt \\
        --legacy-model-path /path/to/atlas_pvfinder/mattia_finder \\
        --output-dir outputs/run3_track_probability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from gnn.data.graph_construction import create_inference_graph
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.utils.constants import GNN_SCORE_THRESHOLD, PT_SCALE
from pv_finder.utils.peak_finding import pv_locations_updated_res

# ── Constants (must match PVF model training) ─────────────────────────────────
_N_SUBEVENTS = 12
_SUBEVENT_WIDTH = 40.0  # mm
_SUBEVENT_STARTS = np.array([-240.0 + i * 40.0 for i in range(_N_SUBEVENTS)])
_N_FEATURES = 7
_CHUNK_SIZE = 100  # tracks per chunk for PVF model input
_MASK_VAL = -999999.0
_SUBEVENT_BINS = 1000  # bins per subevent; total histogram = 12 000 bins


# ── PVF helpers ───────────────────────────────────────────────────────────────


def _build_pvf_tensors(
    z0: np.ndarray,
    d0: np.ndarray,
    d0_err: np.ndarray,
    z0_err: np.ndarray,
    cov: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Build padded (_CHUNK_SIZE tracks) subevent tensors for the PVF model.

    If a subevent has more than _CHUNK_SIZE tracks, it is split into multiple
    chunks; their model outputs are accumulated (summed) by the caller.

    Returns a list of (sub_idx, tensor) pairs where each tensor has shape
    (_N_FEATURES, _CHUNK_SIZE).
    """
    tensors: list[tuple[int, np.ndarray]] = []
    for sub_idx in range(_N_SUBEVENTS):
        z_start = _SUBEVENT_STARTS[sub_idx]
        z_end = z_start + _SUBEVENT_WIDTH
        in_sub = (z0 >= z_start) & (z0 < z_end)
        n = int(in_sub.sum())

        if n == 0:
            blank = np.full((_N_FEATURES, _CHUNK_SIZE), _MASK_VAL, dtype=np.float32)
            tensors.append((sub_idx, blank))
            continue

        order = np.argsort(z0[in_sub])
        z0_s = z0[in_sub][order]
        d0_s = d0[in_sub][order]
        d0e_s = d0_err[in_sub][order]
        z0e_s = z0_err[in_sub][order]
        cov_s = cov[in_sub][order]

        n_chunks = max(1, (n + _CHUNK_SIZE - 1) // _CHUNK_SIZE)
        for c in range(n_chunks):
            s, e = c * _CHUNK_SIZE, min((c + 1) * _CHUNK_SIZE, n)
            nf = e - s
            t = np.full((_N_FEATURES, _CHUNK_SIZE), _MASK_VAL, dtype=np.float32)
            t[0, :nf] = d0_s[s:e]
            t[1, :nf] = z0_s[s:e]
            t[2, :nf] = d0e_s[s:e]
            t[3, :nf] = z0e_s[s:e]
            t[4, :nf] = cov_s[s:e]
            t[5, :nf] = z_start
            t[6, :nf] = z_end
            tensors.append((sub_idx, t))

    return tensors


def _run_pvf(
    pvf_model: torch.nn.Module,
    device: torch.device,
    z0: np.ndarray,
    d0: np.ndarray,
    d0_err: np.ndarray,
    z0_err: np.ndarray,
    cov: np.ndarray,
) -> np.ndarray:
    """Run the PVF model; return the full 12 000-bin histogram."""
    subevent_tensors = _build_pvf_tensors(z0, d0, d0_err, z0_err, cov)
    outputs: dict[int, np.ndarray] = {
        i: np.zeros(_SUBEVENT_BINS, dtype=np.float32) for i in range(_N_SUBEVENTS)
    }
    with torch.no_grad():
        for sub_idx, tensor in subevent_tensors:
            x = torch.from_numpy(tensor).float().unsqueeze(0).to(device)
            pred = pvf_model(x).squeeze().cpu().numpy()
            outputs[sub_idx] += pred
    return np.concatenate([outputs[i] for i in range(_N_SUBEVENTS)])


# ── GNN helper ────────────────────────────────────────────────────────────────


def _run_gnn_event(
    gnn_model: TTVAGATModel,
    device: torch.device,
    z0: np.ndarray,
    d0: np.ndarray,
    d0_err: np.ndarray,
    z0_err: np.ndarray,
    cov: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    pt: np.ndarray,
    pred_z: np.ndarray,
    pred_heights: np.ndarray,
    pred_sigmas: np.ndarray,
) -> np.ndarray | None:
    """Run GNN on one event; return 1-D array of sigmoid edge scores or None."""
    tracks_stack = np.stack(
        [d0, z0, d0_err, z0_err, cov, theta, phi, pt / PT_SCALE], axis=1
    ).astype(np.float32)  # (n_tracks, 8)

    graph = create_inference_graph(
        z0, d0, d0_err, z0_err, tracks_stack, pred_z, pred_heights, pred_sigmas
    )

    if graph["track"].num_nodes == 0 or graph["pv"].num_nodes == 0:
        return None

    # Batch attributes required for single-graph GNN inference
    graph["track"].batch = torch.zeros(graph["track"].num_nodes, dtype=torch.long)
    graph["pv"].batch = torch.zeros(graph["pv"].num_nodes, dtype=torch.long)

    gnn_model.eval()
    with torch.no_grad():
        logits = gnn_model(graph.to(device))

    return torch.sigmoid(logits).cpu().numpy()


# ── Plotting ──────────────────────────────────────────────────────────────────


def _plot(
    all_scores: np.ndarray,
    max_scores: np.ndarray,
    output_dir: str,
    threshold: float,
    n_events: int,
) -> None:
    """Save a two-panel figure of GNN score distributions."""
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

    fig.suptitle("GNN track association scores — Run3 data", fontsize=13, y=1.01)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out / f"run3_track_probability.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN track association score distribution on Run3 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        required=True,
        help="Path to Run3 NPZ cache",
    )
    parser.add_argument(
        "--pvf-weights",
        required=True,
        help="Path to PVF model weights (.pyt, pickled full model)",
    )
    parser.add_argument(
        "--gnn-weights",
        required=True,
        help="Path to GNN state dict (.pyt)",
    )
    parser.add_argument(
        "--legacy-model-path",
        required=True,
        help=(
            "Directory containing the legacy 'model' package "
            "(typically atlas_pvfinder/mattia_finder/) — needed to unpickle PVF weights"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/run3_track_probability",
        help="Directory for output plots and JSON",
    )
    parser.add_argument(
        "--nevents",
        type=int,
        default=500,
        help="Number of Run3 events to process",
    )
    parser.add_argument(
        "--min-pileup",
        type=int,
        default=3,
        help="Minimum NumRecoVtx to accept an event",
    )
    parser.add_argument(
        "--pvf-threshold",
        type=float,
        default=0.02,
        help="PVF peak-finding amplitude threshold",
    )
    parser.add_argument(
        "--pvf-integral-threshold",
        type=float,
        default=0.5,
        help="PVF peak-finding integral threshold",
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

    # ── Device ─────────────────────────────────────────────────────────────────
    if args.device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load PVF model (legacy pickle) ─────────────────────────────────────────
    print(f"\nLoading PVF model: {args.pvf_weights}")
    sys.path.insert(0, str(args.legacy_model_path))
    pvf_model = torch.load(args.pvf_weights, map_location=device, weights_only=False)
    pvf_model.to(device).eval()

    # ── Load GNN model (state dict) ────────────────────────────────────────────
    print(f"Loading GNN model:  {args.gnn_weights}")
    gnn_model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    state_dict = torch.load(args.gnn_weights, map_location=device, weights_only=False)
    gnn_model.load_state_dict(state_dict)
    gnn_model.to(device).eval()

    # ── Load cache ─────────────────────────────────────────────────────────────
    print(f"\nLoading Run3 cache: {args.cache}")
    raw = np.load(args.cache, allow_pickle=True)
    pileup = np.array([int(x) for x in raw["NumRecoVtx"]])
    valid_idx = np.where(pileup >= args.min_pileup)[0]
    n_process = min(args.nevents, len(valid_idx))
    selected = valid_idx[:n_process]
    print(f"  Selected {n_process} events (pileup ≥ {args.min_pileup})")

    # ── Main loop ──────────────────────────────────────────────────────────────
    all_scores_list: list[np.ndarray] = []
    max_scores_list: list[np.ndarray] = []
    n_skipped = 0

    for idx in tqdm(selected, desc="Events"):
        z0 = np.asarray(raw["RecoTrack_z0"][idx], dtype=np.float32)
        d0 = np.asarray(raw["RecoTrack_d0"][idx], dtype=np.float32)
        d0e = np.asarray(raw["RecoTrack_ErrD0"][idx], dtype=np.float32)
        z0e = np.asarray(raw["RecoTrack_ErrZ0"][idx], dtype=np.float32)
        cov = np.asarray(raw["RecoTrack_ErrD0Z0"][idx], dtype=np.float32)
        theta = np.asarray(raw["RecoTrack_theta"][idx], dtype=np.float32)
        phi = np.asarray(raw["RecoTrack_phi"][idx], dtype=np.float32)
        pt = np.asarray(raw["RecoTrack_pT"][idx], dtype=np.float32)

        if len(z0) == 0:
            n_skipped += 1
            continue

        full_hist = _run_pvf(pvf_model, device, z0, d0, d0e, z0e, cov)
        pred_z, pred_heights, _, pred_sigmas = pv_locations_updated_res(
            full_hist, args.pvf_threshold, args.pvf_integral_threshold
        )

        if len(pred_z) == 0:
            n_skipped += 1
            continue

        scores = _run_gnn_event(
            gnn_model,
            device,
            z0,
            d0,
            d0e,
            z0e,
            cov,
            theta,
            phi,
            pt,
            pred_z,
            pred_heights,
            pred_sigmas,
        )
        if scores is None:
            n_skipped += 1
            continue

        scores = np.atleast_1d(scores)
        all_scores_list.append(scores)

        # Max score per track: scores are ordered (track_idx × n_pvs) by meshgrid
        n_tracks, n_pvs = len(z0), len(pred_z)
        expected_edges = n_tracks * n_pvs
        if len(scores) == expected_edges:
            max_scores_list.append(scores.reshape(n_tracks, n_pvs).max(axis=1))

    n_done = n_process - n_skipped
    print(f"\n  Events processed: {n_done}  |  skipped: {n_skipped}")

    all_scores = np.concatenate(all_scores_list)
    max_scores = np.concatenate(max_scores_list) if max_scores_list else np.array([])
    print(f"  Total edges: {len(all_scores):,}  |  Total tracks: {len(max_scores):,}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    _plot(all_scores, max_scores, args.output_dir, args.threshold, n_done)

    # ── JSON summary ───────────────────────────────────────────────────────────
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
    summary_path = out / "run3_track_probability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
