#!/usr/bin/env python3
"""Post-hoc operating-point scan for v4b fake peaks.

Runs the v4b end-to-end model on N HL-LHC events, finds peaks, classifies
every reco peak against MC ``TruthVertex`` (clean / merged / split / fake),
and then answers two questions with data before we commit to any retraining:

  1. **Are fakes neighbours or isolated?**  For every fake peak we record its
     amplitude and its distance to the nearest *real* (clean/merged) reco peak.
     Fakes within ~0.85 mm of a taller real peak live in the
     ``suppress_neighbor_peaks`` regime (sidelobes); the rest are isolated low
     bumps that only a height floor can remove.

  2. **What does a minimum-height floor buy?**  We sweep a height floor, drop
     peaks below it, re-classify, and report fake-rate vs efficiency at each
     floor — the cheapest, fully-reversible post-hoc lever for fakes.

Reuses ``load_v4b`` / ``_to_dict`` from ``failure_mode_viz`` and the shared
peak-finding + greedy-matching code, so the numbers match the real eval.

Usage::

    PYTHONPATH=src python -u \
      src/pv_finder/diagnostics/peak_operating_point.py \
      --n-events 300 --device 0 \
      --output-dir outputs/06_09_2026_output/peak_operating_point
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pv_finder.data.run3_io import load_run3_from_root
from pv_finder.diagnostics.per_vertex_visualization.failure_mode_viz import (
    _DEFAULT_MODEL,
    _DEFAULT_ROOT,
    _to_dict,
    load_v4b,
)
from pv_finder.diagnostics.per_vertex_visualization.inference import run_e2e_on_events
from pv_finder.diagnostics.per_vertex_visualization.peak_matching import (
    classify_vertices,
    find_histogram_peaks,
)

# suppress_neighbor_peaks default separation — anything closer to a taller real
# peak is a sidelobe candidate.
_NEIGHBOR_MM = 0.85


def _classify_event(
    truth: list[float], peaks: list[tuple[float, float]], match_window_mm: float
) -> tuple[int, int, int, int, int]:
    """Return (n_truth, clean, merged, fake, split) for one event's peaks."""
    tlab, rlab = classify_vertices(truth, peaks, match_window_mm=match_window_mm)
    clean = sum(t == "clean" for t in tlab)
    merged = sum(t == "merged" for t in tlab)
    fake = sum(r == "fake" for r in rlab)
    split = sum(r == "split" for r in rlab)
    return len(truth), clean, merged, fake, split


def _floor_sweep(
    records: list[tuple[list[float], list[tuple[float, float]]]],
    floors: list[float],
    match_window_mm: float,
) -> list[dict]:
    """Apply each height floor to the cached peaks and re-classify."""
    n_evt = max(len(records), 1)
    rows: list[dict] = []
    for floor in floors:
        tot_truth = tot_clean = tot_merged = tot_fake = tot_split = 0
        for truth, peaks in records:
            kept = [p for p in peaks if p[1] >= floor]
            nt, c, m, f, s = _classify_event(truth, kept, match_window_mm)
            tot_truth += nt
            tot_clean += c
            tot_merged += m
            tot_fake += f
            tot_split += s
        eff = (tot_clean + tot_merged) / max(tot_truth, 1)
        rows.append(
            {
                "floor": round(floor, 4),
                "efficiency": round(eff, 4),
                "fake_per_evt": round(tot_fake / n_evt, 4),
                "split_per_evt": round(tot_split / n_evt, 4),
                "n_clean": tot_clean,
                "n_merged": tot_merged,
                "n_fake": tot_fake,
            }
        )
    return rows


def _analyse_fakes(
    records: list[tuple[list[float], list[tuple[float, float]]]],
    match_window_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (fake_heights, fake_dist_to_nearest_real, real_heights)."""
    fake_h: list[float] = []
    fake_d: list[float] = []
    real_h: list[float] = []
    for truth, peaks in records:
        tlab, rlab = classify_vertices(truth, peaks, match_window_mm=match_window_mm)
        real_z = np.array(
            [peaks[i][0] for i in range(len(peaks)) if rlab[i] in ("clean", "merged")]
        )
        real_h.extend(
            peaks[i][1] for i in range(len(peaks)) if rlab[i] in ("clean", "merged")
        )
        for i in range(len(peaks)):
            if rlab[i] != "fake":
                continue
            z, h = peaks[i]
            fake_h.append(h)
            fake_d.append(
                float(np.min(np.abs(real_z - z))) if real_z.size else float("inf")
            )
    return np.asarray(fake_h), np.asarray(fake_d), np.asarray(real_h)


def _plot(
    fake_h: np.ndarray,
    fake_d: np.ndarray,
    real_h: np.ndarray,
    sweep: list[dict],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # (1) amplitude: fakes vs real peaks
    bins = np.linspace(
        0, max(0.4, float(np.percentile(real_h, 99)) if real_h.size else 0.4), 60
    )
    ax[0].hist(
        real_h, bins=bins, alpha=0.6, label=f"real (clean+merged), n={real_h.size}"
    )
    ax[0].hist(fake_h, bins=bins, alpha=0.6, label=f"fake, n={fake_h.size}")
    ax[0].set_xlabel("peak amplitude")
    ax[0].set_ylabel("count")
    ax[0].set_title("Amplitude: real vs fake peaks")
    ax[0].legend()
    ax[0].set_yscale("log")

    # (2) fake distance to nearest real peak
    finite = fake_d[np.isfinite(fake_d)]
    ax[1].hist(np.clip(finite, 0, 3.0), bins=60)
    ax[1].axvline(
        _NEIGHBOR_MM, color="r", ls="--", label=f"{_NEIGHBOR_MM} mm (sidelobe)"
    )
    ax[1].set_xlabel("fake distance to nearest real peak (mm)")
    ax[1].set_ylabel("count")
    ax[1].set_title("Are fakes neighbours or isolated?")
    ax[1].legend()

    # (3) operating-point sweep
    floors = [r["floor"] for r in sweep]
    eff = [r["efficiency"] for r in sweep]
    fr = [r["fake_per_evt"] for r in sweep]
    ax2 = ax[2]
    ax2.plot(floors, fr, "o-", color="tab:red", label="fake / evt")
    ax2.set_xlabel("height floor")
    ax2.set_ylabel("fake / evt", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax3 = ax2.twinx()
    ax3.plot(floors, eff, "s-", color="tab:blue", label="efficiency")
    ax3.set_ylabel("efficiency", color="tab:blue")
    ax3.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("Height-floor operating point")

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--root", default=_DEFAULT_ROOT)
    ap.add_argument("--n-events", type=int, default=300)
    ap.add_argument("--integral-threshold", type=float, default=0.2)
    ap.add_argument("--match-window-mm", type=float, default=0.5)
    ap.add_argument("--e2e-unet-channels", type=int, default=280)
    ap.add_argument("--e2e-latent-channels", type=int, default=4)
    ap.add_argument("--entry-start", type=int, default=80000)
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument(
        "--output-dir", default="outputs/06_09_2026_output/peak_operating_point"
    )
    args = ap.parse_args()

    device = (
        torch.device(f"cuda:{args.device}")
        if args.device >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    events = load_run3_from_root(
        args.root, max_events=args.n_events, min_tracks=50,
        min_amvf_vtx=5, entry_start=args.entry_start,
    )  # fmt: skip
    if not events or events[0].truth_z is None:
        print("ERROR: no events or no MC TruthVertex in this ROOT file.")
        sys.exit(1)
    print(f"Loaded {len(events)} events (MC TruthVertex ground truth)")

    model = load_v4b(
        args.model, device, args.e2e_unet_channels, args.e2e_latent_channels
    )
    hists = run_e2e_on_events(
        model, [_to_dict(e) for e in events], "run3", device=str(device)
    )

    # Cache (truth, peaks) once; the floor sweep reuses these.
    records: list[tuple[list[float], list[tuple[float, float]]]] = []
    for evt, hist in zip(events, hists):
        truth = sorted(float(z) for z in evt.truth_z)
        peaks = find_histogram_peaks(
            hist.reshape(-1), integral_threshold=args.integral_threshold
        )
        records.append((truth, peaks))

    # --- Fake characterisation (neighbour vs isolated) ---
    fake_h, fake_d, real_h = _analyse_fakes(records, args.match_window_mm)
    n_evt = len(records)
    print(f"\n=== Fakes: {fake_h.size} total ({fake_h.size / n_evt:.2f}/evt) ===")
    if fake_h.size:
        for q in (50, 75, 90, 95):
            print(f"  amplitude p{q}: {np.percentile(fake_h, q):.4f}")
        finite = fake_d[np.isfinite(fake_d)]
        for w in (0.3, 0.5, _NEIGHBOR_MM, 1.5):
            frac = float(np.mean(finite <= w)) if finite.size else 0.0
            print(f"  within {w:>4} mm of a real peak: {frac * 100:5.1f}%")
        print(f"  isolated (> 1.5 mm from any real peak): "
              f"{float(np.mean(finite > 1.5)) * 100:5.1f}%")  # fmt: skip
    if real_h.size:
        print(f"  (real-peak amplitude p10={np.percentile(real_h, 10):.4f}, "
              f"median={np.median(real_h):.4f})")  # fmt: skip

    # --- Height-floor sweep ---
    floors = [round(0.005 * k, 3) for k in range(0, 31)]  # 0.000 .. 0.150
    sweep = _floor_sweep(records, floors, args.match_window_mm)
    print("\n=== Height-floor sweep ===")
    print(f"  {'floor':>7} {'eff':>7} {'fake/evt':>9} {'split/evt':>9}")
    for r in sweep:
        print(f"  {r['floor']:>7.3f} {r['efficiency']:>7.4f} "
              f"{r['fake_per_evt']:>9.4f} {r['split_per_evt']:>9.4f}")  # fmt: skip

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "height_floor_sweep.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    plot_path = os.path.join(args.output_dir, "operating_point.png")
    _plot(fake_h, fake_d, real_h, sweep, plot_path)
    print(f"\nWrote {csv_path}\nWrote {plot_path}")


if __name__ == "__main__":
    main()
