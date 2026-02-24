"""Two-step PV-Finder vertex-finding evaluation.

Step 1 (Resolution): inference -> peak finding -> pairwise distances -> fit sigma_vtx_vtx.
Step 2 (Classification): use sigma_vtx_vtx to match predicted peaks to truth -> categorize
as Clean / Merged / Split / Fake.  Truth positions come from peak-finding on the truth
KDE histograms (same algorithm and parameters as prediction peak-finding).

Migrated from atlas_pvfinder/mattia_finder/evaluation/{test_model,evaluate_model}.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pv_finder.data.h5_dataset import H5Dataset_tracksHists
from pv_finder.evaluation.vertex_matching import (
    compare_res_reco,
    make_resolution_plot,
)
from pv_finder.utils.constants import (
    BIN_WIDTH_MM,
    BINS_PER_SUBEVENT,
    N_SUBEVENTS,
    Z_MIN,
)
from pv_finder.utils.efficiency import efficiency
from pv_finder.utils.peak_finding import pv_locations_updated_res

# LHCb-style efficiency parameters (fixed; separate from peak-finding params)
_EFF_PARAMS = {
    "difference": 5.0,
    "threshold": 1e-2,
    "integral_threshold": 0.2,
    "min_width": 3,
}

# Test split constants
_TEST_START_SUBEVENT = 581_400
_TEST_MAIN_EVENT_OFFSET = 48_450
_DEFAULT_N_EVENTS = 2550


# ---------------------------------------------------------------------------
# Step 0 -- Inference
# ---------------------------------------------------------------------------


def run_pvf_inference(
    model_weights_path: str | Path,
    h5_path: str | Path,
    subevent_indices: np.ndarray,
    device: torch.device,
    batch_size: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    """Run PVF model on test data and return aggregated 12000-bin histograms.

    Returns (predictions, truth) each shaped (N_events, 12000).
    """
    # Old checkpoints were saved from the 'model' namespace; alias to our module
    import sys
    import types

    import pv_finder.models.autoencoder_models as _ae

    if "model" not in sys.modules:
        _m = types.ModuleType("model")
        _m.autoencoder_models = _ae  # type: ignore[attr-defined]
        sys.modules["model"] = _m
        sys.modules["model.autoencoder_models"] = _ae

    model = torch.load(str(model_weights_path), map_location="cpu", weights_only=False)
    model = model.to(device)
    model.eval()

    dataset = H5Dataset_tracksHists(str(h5_path))
    subset = torch.utils.data.Subset(dataset, subevent_indices.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

    n_subevents = len(subevent_indices)
    all_preds: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []

    # Process in memory-friendly chunks
    chunk_preds: list[np.ndarray] = []
    chunk_truth: list[np.ndarray] = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(tqdm(loader, total=n_subevents)):
            inputs = inputs.float().to(device)
            output = model(inputs).cpu().numpy()
            chunk_preds.append(output)
            # target_y_split has shape (batch, 2, 1000); channel 0 is the target hist
            truth_arr = labels.cpu().numpy()
            if truth_arr.ndim >= 2 and truth_arr.shape[-2] == 2:
                truth_arr = truth_arr[..., 0, :]
            chunk_truth.append(truth_arr)

            # Flush chunk when batch_size reached or at end
            if len(chunk_preds) >= batch_size or idx == n_subevents - 1:
                all_preds.extend(chunk_preds)
                all_truth.extend(chunk_truth)
                chunk_preds = []
                chunk_truth = []

    # Stack and reshape: (N_subevents, 1000) -> (N_events, 12000)
    preds_flat = np.concatenate(all_preds, axis=0).squeeze()
    truth_flat = np.concatenate(all_truth, axis=0).squeeze()

    if preds_flat.ndim == 1:
        preds_flat = preds_flat.reshape(-1, BINS_PER_SUBEVENT)
    if truth_flat.ndim == 1:
        truth_flat = truth_flat.reshape(-1, BINS_PER_SUBEVENT)

    n_events = n_subevents // N_SUBEVENTS
    preds_agg = preds_flat.reshape(n_events, N_SUBEVENTS * BINS_PER_SUBEVENT)
    truth_agg = truth_flat.reshape(n_events, N_SUBEVENTS * BINS_PER_SUBEVENT)

    return preds_agg.astype(np.float32), truth_agg.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1 -- Resolution
# ---------------------------------------------------------------------------


def compute_resolution(
    predictions: np.ndarray,
    output_dir: str | Path,
    threshold: float,
    integral_threshold: float,
    min_width: int,
) -> float:
    """Compute sigma_vtx_vtx from pairwise distances between predicted PVs.

    Returns the fitted sigma in mm.
    """
    all_distances: list[float] = []
    total_pvs = 0

    for i in tqdm(range(len(predictions)), desc="Resolution"):
        z_mm, _, _, _ = pv_locations_updated_res(
            predictions[i],
            threshold=threshold,
            integral_threshold=integral_threshold,
            min_width=min_width,
        )
        total_pvs += len(z_mm)
        # Shuffle before pairwise distances so z[a]-z[b] is symmetric around 0
        z_shuffled = z_mm.copy()
        np.random.shuffle(z_shuffled)
        for a in range(len(z_shuffled) - 1):
            for b in range(a + 1, len(z_shuffled)):
                all_distances.append(float(z_shuffled[a] - z_shuffled[b]))

    print(f"      Total predicted PVs: {total_pvs}")
    print(f"      Pairwise distances: {len(all_distances)}")

    sigma_mm, sigma_err = make_resolution_plot(
        all_distances, output_dir, label="PV-Finder"
    )
    print(f"      sigma_vtx-vtx = {sigma_mm:.2f} +/- {sigma_err:.2f} mm")
    return sigma_mm


# ---------------------------------------------------------------------------
# Step 2 -- Classification
# ---------------------------------------------------------------------------


def evaluate_vertices(
    predictions: np.ndarray,
    truth_histograms: np.ndarray,
    sigma_vtx_vtx: float,
    threshold: float,
    integral_threshold: float,
    min_width: int,
) -> dict:
    """Classify predicted peaks as Clean/Merged/Split/Fake.

    Truth positions come from peak-finding on the truth histograms (KDE labels),
    matching how the model was trained.

    Returns a dict with per-event arrays and aggregate summary.
    """
    n_events = len(predictions)
    sigma_bins = sigma_vtx_vtx / BIN_WIDTH_MM

    per_event = np.zeros((n_events, 4), dtype=int)  # clean, merged, split, fake
    total_s, total_sp, total_mt, total_fp = 0, 0, 0, 0
    total_truth = 0

    for i in tqdm(range(n_events), desc="Classifying"):
        # Peak finding on prediction
        pred_z_mm, _, _, _ = pv_locations_updated_res(
            predictions[i],
            threshold=threshold,
            integral_threshold=integral_threshold,
            min_width=min_width,
        )

        # Truth: peak-find on truth histogram (same algorithm, same params)
        truth_z_mm, _, _, _ = pv_locations_updated_res(
            truth_histograms[i],
            threshold=threshold,
            integral_threshold=integral_threshold,
            min_width=min_width,
        )
        total_truth += len(truth_z_mm)

        # Convert mm -> bins
        pred_bins = (pred_z_mm - Z_MIN) / BIN_WIDTH_MM
        truth_bins = (truth_z_mm - Z_MIN) / BIN_WIDTH_MM
        reco_res = sigma_bins * np.ones(len(pred_bins))

        perf, _truth_cls, _density = compare_res_reco(truth_bins, pred_bins, reco_res)
        per_event[i] = [
            perf.reco_clean,
            perf.reco_merged,
            perf.reco_split,
            perf.reco_fake,
        ]

        # LHCb-style efficiency (operates on raw histograms)
        eff = efficiency(
            truth_histograms[i].astype(np.float32),
            predictions[i].astype(np.float32),
            **_EFF_PARAMS,
        )
        total_s += eff.S
        total_sp += eff.Sp
        total_mt += eff.MT
        total_fp += eff.FP

    totals = per_event.sum(axis=0)
    n_reco = int(totals.sum())

    summary = {
        "n_events": n_events,
        "clean": int(totals[0]),
        "merged": int(totals[1]),
        "split": int(totals[2]),
        "fake": int(totals[3]),
        "n_reco": n_reco,
        "n_truth": total_truth,
        "lhcb_S": total_s,
        "lhcb_Sp": total_sp,
        "lhcb_MT": total_mt,
        "lhcb_FP": total_fp,
    }
    return {"per_event": per_event, "summary": summary}


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------


def _make_category_bar(summary: dict, output_dir: Path) -> None:
    """Bar chart of vertex categories with counts and percentages."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = ["Clean", "Merged", "Split", "Fake"]
    keys = ["clean", "merged", "split", "fake"]
    counts = [summary[k] for k in keys]
    total = summary.get("n_reco", sum(counts)) or 1
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(cats, counts, color=colors, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, counts):
        pct = 100.0 * count / total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("PVF Vertex Classification", fontsize=16, pad=15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(output_dir / "pvf_category_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PV-Finder vertex-finding evaluation (resolution + classification)"
    )
    # Mode selectors
    p.add_argument("--pvf-weights", type=str, help="PVF model weights (full mode)")
    p.add_argument("--pvf-h5", type=str, help="Training H5 file (for inference)")
    p.add_argument("--histograms", type=str, help="Pre-computed histograms .npy")
    p.add_argument(
        "--track-h5", type=str, help="Track associations H5 (unused, kept for compat)"
    )
    p.add_argument(
        "--sigma-vtx-vtx", type=float, default=None, help="Pre-computed sigma (mm)"
    )
    p.add_argument("--n-events", type=int, default=_DEFAULT_N_EVENTS)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--device", type=int, default=0, help="-1 for CPU")
    # Peak finding
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--integral-threshold", type=float, default=0.5)
    p.add_argument("--min-width", type=int, default=3)
    return p


def main() -> None:
    """CLI entry point with verbose step-by-step output."""
    args = _build_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    n_events = args.n_events
    n_subevents = n_events * N_SUBEVENTS

    truth_histograms: np.ndarray | None = None

    # -- Determine mode --
    if args.histograms:
        # Classify-only mode
        print(f"[1/4] Loading pre-computed histograms: {args.histograms}")
        predictions = np.load(args.histograms)
        n_events = len(predictions)
        # Try loading truth histograms from same directory
        truth_path = Path(args.histograms).parent / "pvf_truth_histograms.npy"
        if truth_path.exists():
            truth_histograms = np.load(str(truth_path))
            print(f"      Truth histograms: {truth_path.name}")
        print(f"      Events: {n_events}")
        step_offset = 2  # skip resolution if sigma provided
    elif args.pvf_weights:
        # Full mode
        print("[1/4] Loading PVF model and running inference...")
        print(f"      Model: {Path(args.pvf_weights).name}")
        print(f"      Events: {n_events} ({n_subevents} subevents)")

        sub_indices = np.arange(
            _TEST_START_SUBEVENT, _TEST_START_SUBEVENT + n_subevents
        )
        predictions, truth_histograms = run_pvf_inference(
            args.pvf_weights, args.pvf_h5, sub_indices, device
        )
        print(f"      Aggregating -> {len(predictions)} events")
        np.save(out / "pvf_histograms.npy", predictions)
        np.save(out / "pvf_truth_histograms.npy", truth_histograms)
        print(f"      Saved: {out / 'pvf_histograms.npy'}")
        step_offset = 2
    else:
        print("Error: provide either --pvf-weights or --histograms")
        raise SystemExit(1)

    # -- Resolution --
    if args.sigma_vtx_vtx is not None:
        sigma = args.sigma_vtx_vtx
        print(f"\n[{step_offset}/4] Using provided sigma_vtx-vtx = {sigma:.2f} mm")
    else:
        print(f"\n[{step_offset}/4] Computing vertex-vertex resolution...")
        print(
            f"      Peak params: threshold={args.threshold}, "
            f"integral={args.integral_threshold}, "
            f"min_width={args.min_width}"
        )
        sigma = compute_resolution(
            predictions,
            out,
            threshold=args.threshold,
            integral_threshold=args.integral_threshold,
            min_width=args.min_width,
        )
        print(f"      Saved: {out / 'deltaz_resolution.png'}")

    # -- Classification --
    if truth_histograms is None:
        print("\nError: truth histograms required for classification.")
        print("       Re-run in full mode or ensure pvf_truth_histograms.npy exists.")
        raise SystemExit(1)

    print("\n[3/4] Classifying vertices...")
    print("      Truth source: peak-finding on truth histograms (KDE labels)")
    print(f"      Resolution: sigma = {sigma:.2f} mm ({sigma / BIN_WIDTH_MM:.1f} bins)")

    results = evaluate_vertices(
        predictions,
        truth_histograms,
        sigma_vtx_vtx=sigma,
        threshold=args.threshold,
        integral_threshold=args.integral_threshold,
        min_width=args.min_width,
    )

    # Save outputs
    np.save(out / "pvf_per_event.npy", results["per_event"])
    with open(out / "pvf_results.json", "w") as f:
        json.dump(results["summary"], f, indent=2)
    _make_category_bar(results["summary"], out)

    # -- Print summary --
    s = results["summary"]
    print("\n[4/4] Results")
    print(f"      Events evaluated:  {s['n_events']}")
    print(f"      Truth PVs:         {s['n_truth']}")
    print(f"      Reco PVs:          {s['n_reco']}")
    print("      ---")
    print(
        f"      Clean:   {s['clean']:>6d}  ({100 * s['clean'] / max(s['n_reco'], 1):.1f}%)"
    )
    print(
        f"      Merged:  {s['merged']:>6d}  ({100 * s['merged'] / max(s['n_reco'], 1):.1f}%)"
    )
    print(
        f"      Split:   {s['split']:>6d}  ({100 * s['split'] / max(s['n_reco'], 1):.1f}%)"
    )
    print(
        f"      Fake:    {s['fake']:>6d}  ({100 * s['fake'] / max(s['n_reco'], 1):.1f}%)"
    )
    print("      ---")
    real = s["lhcb_S"] + s["lhcb_MT"]
    if real > 0:
        print(f"      LHCb eff:   {s['lhcb_S'] / real:.4f}")
    print(f"      LHCb FP/ev: {s['lhcb_FP'] / max(s['n_events'], 1):.3f}")
    print("\n      Saved: pvf_results.json, pvf_per_event.npy, pvf_category_bar.png")


if __name__ == "__main__":
    main()
