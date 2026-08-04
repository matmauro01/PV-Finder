#!/usr/bin/env python3
"""Where satellite peaks come from: split rule, upsampling lattice, prominence.

Companion to ``bump_model_vs_data.py``. That script answers *how much* of the
pairwise-Delta-z excess is model and how much is data; this one answers *by what
mechanism* a satellite is emitted, with everything regenerated from committed
code so the numbers in ``docs/research/pairwise_dz_bump.md`` have provenance.

Four measurements on the same held-out events:

1. **Conjoined-split origin.** The peak finder ends a region on any single-bin
   non-monotonicity once the region maximum has been passed
   (``peak_finding.py``), with zero noise tolerance. A peak is counted as
   split-emitted when its contiguous above-threshold run contains more than one
   detected peak.

2. **Upsampling lattice.** ``UNet_1000_v2`` pools twice and upsamples twice with
   ``F.interpolate(mode="nearest")``, so the decoder writes on a stride-4
   lattice. If satellites are lattice artefacts their argmax bins are not
   uniform mod 4, while genuine peaks -- whose position is set by the tracks --
   should be. Reported as a chi-square on 3 degrees of freedom per peak class.

3. **Topographic prominence.** Height minus the higher of the two flanking
   minima. A real vertex stands alone; a ripple promoted by the split rule does
   not. Also: the band excess left after dropping peaks below a prominence gate.

4. **Target reachability.** Fraction of truth vertices whose target Gaussian is
   narrower than one 0.04 mm bin under a given resolution preset, and the
   amplitude scale ``max(1, 0.15/sigma)`` those vertices receive. The network is
   asked for spikes no 0.04 mm histogram can represent.

Run from the repo root with the venv active, inside tmux::

    PYTHONPATH=src python -u src/pv_finder/diagnostics/satellite_mechanism.py \\
        --root data/run4_all_etas/.../..._r16443_PU200.root \\
        --ckpt model_weights/hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth \\
        --preset hllhc_alleta --n-events 800 --device 3 \\
        --output-dir outputs/08_04_2026_output/satellite_mechanism
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from pv_finder.data.resolution_presets import RESOLUTION_PRESETS  # noqa: E402
from pv_finder.diagnostics.bump_model_vs_data import (  # noqa: E402
    build_model,
    find_peaks,
    load_entries,
    predict_hist,
    select_entries,
)
from pv_finder.diagnostics.pairwise_dz_metrics import (  # noqa: E402
    observables,
    pair_hists,
)

BIN_WIDTH = 0.04  # mm
SAT_LO, SAT_HI = 0.25, 0.8  # mm, the shell a band satellite occupies
MATCH_WINDOW = 0.3  # mm, greedy peak-to-truth matching
LATTICE = 4  # decoder stride: two nearest-neighbour x2 upsamples
CENTROID_WIDTHS = (2, 3, 5)  # bins, half-width of the local-centroid variants


def prominences(hist: np.ndarray, peak_bins: np.ndarray) -> np.ndarray:
    """Topographic prominence of each peak: height minus the higher saddle.

    Walks outward from the peak until the histogram rises above the peak's own
    height or the array ends, tracking the minimum on each side. The prominence
    is the peak height minus the *larger* of the two minima, which is the
    standard definition and the one that separates a stand-alone vertex from a
    ripple riding on a neighbour's flank.
    """
    out = np.empty(len(peak_bins), dtype=np.float64)
    n = len(hist)
    for k, b in enumerate(peak_bins):
        h = hist[b]
        saddles = []
        for step in (-1, 1):
            lo = h
            i = b + step
            while 0 <= i < n and hist[i] <= h:
                lo = min(lo, hist[i])
                i += step
            saddles.append(lo)
        out[k] = h - max(saddles)
    return out


def split_emitted(
    hist: np.ndarray, peak_bins: np.ndarray, threshold: float
) -> np.ndarray:
    """True where the peak shares its above-threshold run with another peak."""
    on = hist >= threshold
    # run id of every bin: cumulative count of rising edges, -1 where off
    edges = np.diff(np.concatenate(([False], on)).astype(np.int8)) == 1
    run_id = np.cumsum(edges) - 1
    run_of_peak = run_id[peak_bins]
    counts = np.bincount(run_of_peak)
    return counts[run_of_peak] > 1


def greedy_match(z: np.ndarray, truth: np.ndarray, window: float) -> np.ndarray:
    """Greedy 1-to-1 nearest matching; True where the peak claimed a truth PV."""
    matched = np.zeros(len(z), dtype=bool)
    if len(z) == 0 or len(truth) == 0:
        return matched
    d = np.abs(z[:, None] - truth[None, :])
    used = np.zeros(len(truth), dtype=bool)
    order = np.argsort(d.min(axis=1))
    for i in order:
        row = np.where(used, np.inf, d[i])
        j = int(np.argmin(row))
        if row[j] <= window:
            matched[i] = True
            used[j] = True
    return matched


def chi2_mod(bins: np.ndarray, m: int = LATTICE) -> dict:
    """Uniformity of ``bins % m``: counts, modulation depth and chi-square."""
    if len(bins) == 0:
        return {"n": 0}
    c = np.bincount(bins % m, minlength=m).astype(float)
    e = len(bins) / m
    return {
        "n": int(len(bins)),
        "counts": c.astype(int).tolist(),
        "modulation": float((c.max() - c.min()) / c.mean()),
        "chi2": float(((c - e) ** 2 / e).sum()),
        "ndf": m - 1,
    }


def target_reachability(ntrk: np.ndarray, preset: str, ntrk_threshold: int = 2) -> dict:
    """Sub-bin fraction and amplitude scaling of the target Gaussians."""
    a, b, c = RESOLUTION_PRESETS[preset]
    # Only nTrk >= NTRK_THRESHOLD vertices go into the trained channel of the
    # target (root_to_h5._build_truth_histogram: cat 0); the rest land in the
    # untrained channel 1 at a fixed one-bin width.
    n = np.asarray(ntrk, dtype=float)
    n = n[n >= ntrk_threshold]
    sigma = np.maximum(a * n**-b + c, 0.002)
    scale = np.maximum(1.0, 0.15 / sigma)
    return {
        "preset": preset,
        "abc": [a, b, c],
        "n_vertices": int(len(sigma)),
        "frac_sigma_below_bin_width": float(np.mean(sigma < BIN_WIDTH)),
        "median_sigma_mm": float(np.median(sigma)),
        "min_sigma_mm": float(sigma.min()),
        "max_amplitude_scale": float(scale.max()),
        "median_amplitude_scale": float(np.median(scale)),
        "frac_scale_above_2": float(np.mean(scale > 2)),
    }


def band_excess(z_per_event: list[np.ndarray]) -> float:
    """Mean relative excess over the plateau in the 0.3-0.7 mm band."""
    hists, npeaks = pair_hists(z_per_event)
    return observables(hists.sum(0), float(npeaks.sum()))["band_excess_pct"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--preset", default="hllhc_alleta", choices=sorted(RESOLUTION_PRESETS)
    )
    p.add_argument("--n-events", type=int, default=800)
    p.add_argument("--entry-stop", type=int, default=12000)
    p.add_argument("--mu-min", type=float, default=185.0)
    p.add_argument("--mu-max", type=float, default=215.0)
    p.add_argument("--peak-threshold", type=float, default=1e-2)
    p.add_argument("--integral-threshold", type=float, default=0.2)
    p.add_argument("--min-width", type=int, default=3)
    p.add_argument("--device", type=int, default=3)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(f"cuda:{args.device}")
        if args.device >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    entries = select_entries(
        args.root, args.mu_min, args.mu_max, args.n_events, args.entry_stop
    )
    print(f"  {len(entries)} events, device {device}")
    events = load_entries(args.root, entries)["events"]
    model = build_model(args.ckpt, device)

    bins_all, prom_all, split_all, match_all, sat_all, hgt_all = [], [], [], [], [], []
    z_per_event: list[np.ndarray] = []
    z_centroid: dict[int, list[np.ndarray]] = {w: [] for w in CENTROID_WIDTHS}
    for i, ev in enumerate(events):
        hist = predict_hist(model, ev, device).astype(np.float64)
        pf = (args.peak_threshold, args.integral_threshold, args.min_width, 0.0)
        # Reference peak list: the historical full-region weighted mean, the
        # estimator behind every published PV-Finder number.
        z, h, pb = find_peaks(hist, *pf, 0)
        matched = greedy_match(z, ev["truth_z"], MATCH_WINDOW)
        # A band satellite is a surplus peak sitting SAT_LO-SAT_HI mm from a
        # truth-matched peak: the population that fills the pairwise-dz band.
        sat = np.zeros(len(z), dtype=bool)
        if matched.any():
            d = np.abs(z[:, None] - z[None, :][:, matched])
            near = d.min(axis=1)
            sat = (~matched) & (near >= SAT_LO) & (near <= SAT_HI)
        z_per_event.append(z)
        for w in CENTROID_WIDTHS:
            z_centroid[w].append(find_peaks(hist, *pf, w)[0])
        hgt_all.append(h)
        bins_all.append(pb)
        prom_all.append(prominences(hist, pb))
        split_all.append(split_emitted(hist, pb, args.peak_threshold))
        match_all.append(matched)
        sat_all.append(sat)
        if i % 100 == 0:
            print(f"    {i}/{len(events)}", flush=True)

    pb = np.concatenate(bins_all)
    hgt = np.concatenate(hgt_all)
    prom = np.concatenate(prom_all)
    split = np.concatenate(split_all)
    match = np.concatenate(match_all)
    sat = np.concatenate(sat_all)
    classes = {
        "all": np.ones(len(pb), dtype=bool),
        "truth_matched": match,
        "surplus": ~match,
        "band_satellite": sat,
    }

    res: dict = {
        "sample": {
            "n_events": len(events),
            "n_peaks": int(len(pb)),
            "peaks_per_event": float(len(pb) / len(events)),
            "surplus_per_event": float((~match).sum() / len(events)),
            "band_satellites_per_event": float(sat.sum() / len(events)),
        },
        "conjoined_split_fraction": {
            k: float(split[m].mean()) for k, m in classes.items() if m.sum()
        },
        "lattice_mod4": {k: chi2_mod(pb[m]) for k, m in classes.items()},
        "prominence": {
            k: {
                "median": float(np.median(prom[m])),
                "frac_below_0p005": float(np.mean(prom[m] < 0.005)),
                "frac_below_0p01": float(np.mean(prom[m] < 0.01)),
            }
            for k, m in classes.items()
            if m.sum()
        },
    }

    # min_height and a prominence gate are both pure post-hoc gates on a peak
    # that the finder has already recorded, so cutting the peak list here is
    # bit-exact equivalent to re-running the finder with that setting.
    off, per_peak = 0, []
    for zz in z_per_event:
        per_peak.append((prom[off : off + len(zz)], hgt[off : off + len(zz)]))
        off += len(zz)
    for name, col, gates in (
        ("band_excess_vs_prominence_gate", 0, [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]),
        ("band_excess_vs_height_floor", 1, [0.0, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]),
    ):
        res[name] = []
        for g in gates:
            kept = [zz[pk[col] >= g] for zz, pk in zip(z_per_event, per_peak)]
            res[name].append(
                {
                    "gate": g,
                    "peaks_per_event": float(np.mean([len(k) for k in kept])),
                    "band_excess_pct": band_excess(kept),
                }
            )

    res["band_excess_vs_position_estimator"] = [
        {
            "estimator": "historical full-region weighted mean",
            "band_excess_pct": band_excess(z_per_event),
        }
    ] + [
        {
            "estimator": f"local centroid +-{w} bins ({w * BIN_WIDTH:.2f} mm)",
            "band_excess_pct": band_excess(z_centroid[w]),
        }
        for w in CENTROID_WIDTHS
    ]

    res["target_reachability"] = target_reachability(
        np.concatenate([e["truth_ntrk"] for e in events]), args.preset
    )

    np.savez_compressed(
        outdir / "peaks.npz",
        z=np.concatenate(z_per_event),
        n_per_event=np.array([len(z) for z in z_per_event]),
        peak_bin=pb,
        height=hgt,
        prominence=prom,
        split_emitted=split,
        truth_matched=match,
        band_satellite=sat,
    )
    with open(outdir / "satellite_mechanism.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    print(f"\n  wrote {outdir}/satellite_mechanism.json")


if __name__ == "__main__":
    main()
