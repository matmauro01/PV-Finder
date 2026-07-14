"""HGTD timing coverage of reconstructed tracks in the PU200 samples.

Quantifies which tracks carry a valid RecoTrack_Time (sentinel: -1 /
TimeResolution <= 0) as a function of eta = -ln(tan(theta/2)). Documents
why timing is a forward-track study for TTVA rather than a global
feature: the HGTD only instruments the forward edge of the acceptance.

Usage:
    python -u -m gnn.diagnostics.hgtd_timing_coverage \\
        --root data/run4/PU200_withTiming/..._1.root \\
        --max-events 5000 -o outputs/<date>/timing_coverage/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

ETA_EDGES = np.linspace(-5.0, 5.0, 51)


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    tree = uproot.open(args.root)["PVFinderData"]
    arrays = tree.arrays(
        ["RecoTrack_theta", "RecoTrack_Time", "RecoTrack_TimeResolution"],
        entry_stop=args.max_events,
    )
    theta = ak.to_numpy(ak.flatten(arrays["RecoTrack_theta"]))
    time_val = ak.to_numpy(ak.flatten(arrays["RecoTrack_Time"]))
    time_res = ak.to_numpy(ak.flatten(arrays["RecoTrack_TimeResolution"]))

    eta = -np.log(np.tan(theta / 2.0))
    valid = (time_res > 0) & (time_val != -1)

    counts_all, _ = np.histogram(eta, bins=ETA_EDGES)
    counts_valid, _ = np.histogram(eta[valid], bins=ETA_EDGES)
    coverage = np.divide(
        counts_valid,
        counts_all,
        out=np.zeros_like(counts_valid, dtype=np.float64),
        where=counts_all > 0,
    )

    forward = np.abs(eta) > 2.3
    out = {
        "root": args.root,
        "n_events": int(args.max_events),
        "n_tracks": int(len(eta)),
        "coverage_overall": float(valid.mean()),
        "coverage_forward_eta_gt_2p3": float(valid[forward].mean()),
        "forward_track_fraction": float(forward.mean()),
        "median_time_resolution_valid_ps": float(np.median(time_res[valid]) * 1e3)
        if valid.any()
        else None,
        "eta_edges": ETA_EDGES.tolist(),
        "coverage_vs_eta": coverage.tolist(),
        "tracks_vs_eta": counts_all.tolist(),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "timing_coverage.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"tracks: {out['n_tracks']}, coverage overall "
          f"{out['coverage_overall']:.4f}, |eta|>2.3 "
          f"{out['coverage_forward_eta_gt_2p3']:.4f} "
          f"(forward fraction {out['forward_track_fraction']:.4f})")  # fmt: skip
    print(f"Saved {out_dir / 'timing_coverage.json'}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, type=str)
    p.add_argument("--max-events", default=5000, type=int)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    main()
