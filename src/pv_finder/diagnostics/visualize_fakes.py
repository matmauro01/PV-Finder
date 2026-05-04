#!/usr/bin/env python3
"""Visualize fake peaks from eval artifacts, especially near-real sidelobes.

Loads eval_results.pkl (with histograms), identifies fake peaks, and produces
per-vertex zoom plots centered on each fake. Prioritizes fakes that are close
to a matched (real) peak — the "low Δz" sidelobe candidates.

Usage:
    python -u src/pv_finder/diagnostics/visualize_fakes.py \
        outputs/04_23_2026_output/hllhc_v2_ep100_artifacts/eval_results.pkl \
        --output-dir outputs/05_03_2026_output/fake_peaks \
        --max-fakes 50 --max-dz 2.0
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

Z_MIN, Z_MAX, N_BINS = -240.0, 240.0, 12000
BIN_WIDTH = (Z_MAX - Z_MIN) / N_BINS


def _z_axis() -> np.ndarray:
    return np.linspace(Z_MIN + BIN_WIDTH / 2, Z_MAX - BIN_WIDTH / 2, N_BINS)


def _rescale_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src_mx = float(np.max(np.abs(source)))
    tgt_mx = float(np.max(np.abs(target)))
    if src_mx > 1e-30 and tgt_mx > 1e-30:
        return source * (tgt_mx / src_mx)
    return source.copy()


def plot_fake_zoom(
    hist: np.ndarray,
    fake_z: float,
    fake_height: float,
    pred_peaks: list[tuple[float, float]],
    truth_vertices: list[float],
    event_idx: int,
    fake_idx: int,
    nearest_truth_dz: float,
    nearest_reco_dz: float,
    output_dir: str,
    window_mm: float = 4.0,
    match_window_mm: float = 0.3,
) -> None:
    """Two-panel zoom plot centered on a fake peak."""
    os.makedirs(output_dir, exist_ok=True)
    z = _z_axis()
    lo, hi = fake_z - window_mm, fake_z + window_mm
    mask = (z >= lo) & (z <= hi)
    z_win = z[mask]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )

    # Panel 1: histogram with peaks and truth
    ax1.plot(z_win, hist[mask], color="#d62728", lw=1.5, label="Predicted hist.")
    ax1.axvline(
        fake_z,
        color="#d62728",
        ls=":",
        lw=2,
        alpha=0.8,
        label=f"FAKE peak (h={fake_height:.4f})",
    )

    # Truth vertices in window
    truth_drawn = False
    for vz in truth_vertices:
        if lo <= vz <= hi:
            lbl = "Truth vertex" if not truth_drawn else ""
            ax1.axvline(vz, color="black", ls="--", lw=1.2, alpha=0.6, label=lbl)
            ax1.axvspan(
                vz - match_window_mm, vz + match_window_mm, alpha=0.1, color="#2ca02c"
            )
            truth_drawn = True

    # All predicted peaks in window
    matched_drawn = fake_drawn = False
    for pz, ph in pred_peaks:
        if not (lo <= pz <= hi):
            continue
        is_this_fake = abs(pz - fake_z) < 0.01
        d_truth = min(abs(pz - t) for t in truth_vertices) if truth_vertices else 999
        is_matched = d_truth <= match_window_mm

        if is_this_fake:
            ax1.plot(
                pz, ph, "o", ms=10, color="#d62728", mfc="#d62728", mew=2, zorder=10
            )
        elif is_matched:
            lbl = "Matched peak" if not matched_drawn else ""
            ax1.plot(
                pz,
                ph,
                "o",
                ms=7,
                color="#1f77b4",
                mfc="#1f77b4",
                mew=1.5,
                zorder=5,
                label=lbl,
            )
            matched_drawn = True
        else:
            lbl = "Other fake" if not fake_drawn else ""
            ax1.plot(
                pz,
                ph,
                "o",
                ms=7,
                color="#d62728",
                mfc="none",
                mew=1.5,
                zorder=5,
                label=lbl,
            )
            fake_drawn = True

    ax1.set_ylabel("Predicted histogram")
    ax1.set_title(
        f"FAKE peak — Event {event_idx} | z={fake_z:.2f} mm | "
        f"h={fake_height:.4f}\n"
        f"nearest truth: {nearest_truth_dz:.3f} mm | "
        f"nearest reco: {nearest_reco_dz:.3f} mm",
        fontsize=11,
    )
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: residual detail (just the histogram zoomed to see structure)
    ax2.fill_between(z_win, hist[mask], alpha=0.3, color="#d62728")
    ax2.plot(z_win, hist[mask], color="#d62728", lw=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    for vz in truth_vertices:
        if lo <= vz <= hi:
            ax2.axvline(vz, color="black", ls="--", lw=0.8, alpha=0.5)
    ax2.axvline(fake_z, color="#d62728", ls=":", lw=1.5, alpha=0.8)
    ax2.set_xlabel("z [mm]")
    ax2.set_ylabel("Histogram")
    ax2.set_xlim(lo, hi)

    stem = f"fake_{fake_idx:03d}_evt{event_idx}_z{fake_z:+.1f}mm"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize fake peaks from eval artifacts"
    )
    parser.add_argument("pkl", help="Path to eval_results.pkl (with histograms)")
    parser.add_argument("--output-dir", default="outputs/fake_peaks")
    parser.add_argument(
        "--max-fakes", type=int, default=50, help="Max fake plots to generate"
    )
    parser.add_argument(
        "--max-dz",
        type=float,
        default=2.0,
        help="Only show fakes within this distance of a matched peak (mm)",
    )
    parser.add_argument(
        "--window-mm", type=float, default=4.0, help="Zoom window half-width (mm)"
    )
    parser.add_argument(
        "--match-window", type=float, default=0.3, help="Truth matching window (mm)"
    )
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        r = pickle.load(f)

    if r.get("histograms") is None:
        print("ERROR: pkl has no histograms. Re-run eval with --save-histograms.")
        return

    pred_pos = r["pred_pvs_mm"]
    pred_hts = r["pred_heights"]
    truth_pos = r["truth_pvs_mm"]
    histograms = r["histograms"]
    mw = args.match_window

    # Collect all fakes with metadata
    fakes = []
    for ei in range(len(pred_pos)):
        pp = np.array(pred_pos[ei])
        ph = np.array(pred_hts[ei])
        tt = np.array(truth_pos[ei])
        if len(pp) == 0 or len(tt) == 0:
            continue
        for pi in range(len(pp)):
            d_truth = float(np.min(np.abs(pp[pi] - tt)))
            if d_truth <= mw:
                continue  # matched, skip
            # Distance to nearest OTHER predicted peak
            others = np.concatenate([pp[:pi], pp[pi + 1 :]])
            d_reco = float(np.min(np.abs(pp[pi] - others))) if len(others) > 0 else 999
            # Is nearest reco matched?
            if d_reco <= args.max_dz:
                fakes.append(
                    dict(
                        ei=ei,
                        pi=pi,
                        z=float(pp[pi]),
                        h=float(ph[pi]),
                        d_truth=d_truth,
                        d_reco=d_reco,
                    )
                )

    print(f"Found {len(fakes)} fakes within {args.max_dz} mm of another reco peak")

    # Sort by distance to nearest reco (closest first — most sidelobe-like)
    fakes.sort(key=lambda x: x["d_reco"])

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    n_plot = min(args.max_fakes, len(fakes))
    print(f"Generating {n_plot} fake peak plots...")
    for idx, fk in enumerate(fakes[:n_plot]):
        ei = fk["ei"]
        pp = pred_pos[ei]
        ph = pred_hts[ei]
        tt = truth_pos[ei]
        peaks = list(zip([float(p) for p in pp], [float(h) for h in ph]))
        truth_list = [float(t) for t in tt]

        plot_fake_zoom(
            hist=histograms[ei],
            fake_z=fk["z"],
            fake_height=fk["h"],
            pred_peaks=peaks,
            truth_vertices=truth_list,
            event_idx=ei,
            fake_idx=idx,
            nearest_truth_dz=fk["d_truth"],
            nearest_reco_dz=fk["d_reco"],
            output_dir=str(outdir),
            window_mm=args.window_mm,
            match_window_mm=mw,
        )
        if idx < 5 or idx % 10 == 0:
            print(
                f"  [{idx:3d}] evt={ei} z={fk['z']:+.2f}mm h={fk['h']:.4f} "
                f"d_truth={fk['d_truth']:.3f}mm d_reco={fk['d_reco']:.3f}mm"
            )

    # Summary stats
    dz_arr = np.array([f["d_reco"] for f in fakes[:n_plot]])
    ht_arr = np.array([f["h"] for f in fakes[:n_plot]])
    print(f"\nPlotted {n_plot} fakes to {outdir}")
    print(
        f"  d_reco: median={np.median(dz_arr):.3f} mm, range=[{dz_arr.min():.3f}, {dz_arr.max():.3f}]"
    )
    print(
        f"  height: median={np.median(ht_arr):.4f}, range=[{ht_arr.min():.4f}, {ht_arr.max():.4f}]"
    )


if __name__ == "__main__":
    main()
