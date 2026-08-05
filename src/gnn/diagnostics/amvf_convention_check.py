"""Reconcile the two AMVF vertex-classification conventions.

The same AMVF vertex collection, on the same events, is reported with two
very different Clean rates:

- ``efficiency_res_optimized_atlas.compare_res_reco`` (the PV-finder
  evaluation) is **positional**: it greedily matches reco z to truth z inside
  a resolution window and never inspects tracks. A reco vertex is Fake when
  no truth vertex lies in its window.
- ``gnn.evaluation.classification.classify_assignments`` (the TTVA
  evaluation) is **track-purity based**: a reco vertex is Clean when at least
  ``PURITY_THRESHOLD`` of its assigned tracks come from one truth vertex, and
  Fake only when none of its tracks has any truth vertex at all.

Both partition the identical reco collection, so the totals agree while the
categories do not. This script measures the purity distribution that drives
the difference and scans the Clean rate against the purity cut, turning the
explanation into a measurement.

Usage:
    python -u -m gnn.diagnostics.amvf_convention_check \\
        --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \\
        --entry-indices outputs/<date>/gnn_ttva_v4/chain_test/entry_indices.npy \\
        -o outputs/<date>/gnn_ttva_v4/amvf_convention/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot
from tqdm import tqdm

from gnn.data.event_selection import iterate_entries
from gnn.data.pu200_chain_graphs import amvf_matched_lists, truth_arrays
from gnn.evaluation.classification import classify_assignments

BRANCHES = [
    "RecoTrack_pT",
    "RecoTrack_z0",
    "TruthVertex_nTracks",
    "TruthVertex_assocTracks",
    "RecoVertex_assocTracks",
]

PURITY_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def vertex_purities(
    matched: list[np.ndarray], track_truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex (dominant-truth fraction, n_assigned_tracks).

    The fraction is over ALL assigned tracks, matching how
    classify_assignments computes ``max_value / w_total_reco``.
    """
    fracs, sizes = [], []
    for tracks in matched:
        n = len(tracks)
        if n == 0:
            continue
        truths = track_truth[tracks]
        real = truths[truths >= 0]
        if len(real) == 0:
            fracs.append(0.0)
        else:
            _, counts = np.unique(real, return_counts=True)
            fracs.append(float(counts.max()) / float(n))
        sizes.append(n)
    return np.asarray(fracs), np.asarray(sizes, dtype=np.int64)


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tree = uproot.open(args.root)["PVFinderData"]
    # The mu window makes the selection non-contiguous, so the entry list
    # written by pu200_chain_graphs is the only correct way back to the events.
    entries = np.load(args.entry_indices)
    print(f"{len(entries)} events from {args.entry_indices}")

    totals = np.zeros(6, dtype=np.int64)
    clean_at_cut = dict.fromkeys(PURITY_GRID, 0)
    all_fracs: list[np.ndarray] = []
    all_sizes: list[np.ndarray] = []
    n_reco_nonempty = 0

    for event in tqdm(
        iterate_entries(tree, BRANCHES, entries, step_size=200),
        total=len(entries),
        desc="events",
    ):
        pt = np.asarray(event["RecoTrack_pT"], dtype=np.float64)
        n_tracks = len(np.asarray(event["RecoTrack_z0"]))
        tti, tpi, truth_count = truth_arrays(event, n_tracks)

        track_truth = np.full(n_tracks, -1, dtype=np.int64)
        track_truth[tti.astype(np.int64)] = tpi.astype(np.int64)

        matched = amvf_matched_lists(event, n_tracks)
        rows, _ = classify_assignments(matched, pt, tti, tpi, truth_count)
        totals += np.array(rows, dtype=np.int64)

        fracs, sizes = vertex_purities(matched, track_truth)
        all_fracs.append(fracs)
        all_sizes.append(sizes)
        n_reco_nonempty += len(fracs)
        for cut in PURITY_GRID:
            clean_at_cut[cut] += int((fracs >= cut).sum())

    fracs = np.concatenate(all_fracs)
    sizes = np.concatenate(all_sizes)
    clean, merged, split, fake, n_reco, n_truth = (int(x) for x in totals)
    n_ev = len(entries)

    report = {
        "root": args.root,
        "entry_indices": args.entry_indices,
        "n_events": n_ev,
        "track_purity_convention": {
            "clean": clean,
            "merged": merged,
            "split": split,
            "fake": fake,
            "n_reco": n_reco,
            "n_truth": n_truth,
            "reco_per_event": n_reco / n_ev,
            "truth_per_event": n_truth / n_ev,
            "clean_rate": clean / n_reco,
            "merged_rate": merged / n_reco,
            "split_rate": split / n_reco,
            "fake_rate": fake / n_reco,
            "clean_per_truth": clean / n_truth,
        },  # fmt: skip
        "purity": {
            "n_vertices_with_tracks": int(len(fracs)),
            "median": float(np.median(fracs)),
            "mean": float(fracs.mean()),
            "quantiles": {
                str(q): float(np.quantile(fracs, q))
                for q in (0.1, 0.25, 0.5, 0.75, 0.9)
            },
            "median_tracks_per_vertex": float(np.median(sizes)),
        },
        "clean_rate_vs_purity_cut": {
            str(cut): clean_at_cut[cut] / max(n_reco_nonempty, 1) for cut in PURITY_GRID
        },
    }
    with open(out_dir / "amvf_convention.json", "w") as f:
        json.dump(report, f, indent=2)

    tp = report["track_purity_convention"]
    print(f"\nAMVF, track-purity convention ({n_ev} events):")
    print(
        f"  reco/evt {tp['reco_per_event']:.2f}  truth/evt {tp['truth_per_event']:.2f}"
    )
    print(f"  clean {tp['clean_rate']:.4f}  merged {tp['merged_rate']:.4f}  "
          f"split {tp['split_rate']:.4f}  fake {tp['fake_rate']:.5f}")  # fmt: skip
    print(f"  clean/truth {tp['clean_per_truth']:.4f}")
    p = report["purity"]
    print(f"\nDominant-truth purity of AMVF vertices "
          f"(n={p['n_vertices_with_tracks']}, median {p['median_tracks_per_vertex']:.0f} trk):")  # fmt: skip
    print(f"  median {p['median']:.4f}  quantiles "
          f"{ {k: round(v, 3) for k, v in p['quantiles'].items()} }")  # fmt: skip
    print("\nClean rate vs purity cut (cut 0.7 is PURITY_THRESHOLD):")
    for cut in PURITY_GRID:
        print(f"  >= {cut:.1f} : {report['clean_rate_vs_purity_cut'][str(cut)]:.4f}")
    print(f"\nSaved {out_dir / 'amvf_convention.json'}")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", required=True, type=str)
    p.add_argument(
        "--entry-indices",
        required=True,
        type=str,
        help="entry_indices.npy written by pu200_chain_graphs",
    )
    p.add_argument("-o", "--output-dir", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    main()
