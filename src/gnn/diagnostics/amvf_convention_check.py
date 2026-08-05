"""Reconcile the two vertex-classification conventions, and measure what
actually drives the Fake rate.

The same AMVF vertex collection, on the same events, is reported with two very
different Clean rates:

- ``efficiency_res_optimized_atlas.compare_res_reco`` (the PV-finder
  evaluation) is **positional**: it greedily matches reco z to truth z inside a
  resolution window and never inspects tracks. A reco vertex is Fake when no
  truth vertex lies in its window.
- ``gnn.evaluation.classification.classify_assignments`` (the TTVA evaluation)
  is **track based**: see the exact rule below.

THE EXACT Fake RULE (read the code, not the old docstring). In
``classify_assignments`` every assigned track is bucketed by its truth vertex,
and tracks with **no** truth association all go into a single bucket keyed
``"Fake"``. The classification takes a **plurality over all buckets, including
that truthless one**. So a vertex is Fake when the truthless bucket is the
modal bucket -- NOT when the vertex has no truth-associated tracks at all.

Two consequences that a "no truth at all" reading gets wrong:

1. A vertex with 3 truthless tracks and 1 truth-associated track **is** Fake.
2. The truthless bucket is ONE bucket while truth tracks are split across MANY
   truth-PV buckets, so it only has to beat the largest *single* truth PV, not
   the truth tracks in total. A vertex that is 40% truthless, 30% truth-PV-A
   and 30% truth-PV-B is Fake despite being 60% truth-associated.

This module therefore measures the decisive comparison directly --
``n_truthless`` against ``max_over_truth_PVs(n_tracks)`` per vertex -- rather
than asserting a mechanism. It also measures the dominant-truth purity
distribution that drives the Clean/Merged split, and scans Clean rate against
the purity cut.

Works on AMVF vertices (from ROOT) and, with ``--graphs``/``--weights``, on the
chain peaks a trained associator produced, so the two can be compared.

Usage:
    python -u -m gnn.diagnostics.amvf_convention_check \\
        --root .../ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \\
        --entry-indices outputs/<date>/gnn_ttva_v4/chain_test/entry_indices.npy \\
        -o outputs/<date>/gnn_ttva_v4/amvf_convention/
    # optionally add the chain-peak arm:
    #   --graphs data/.../pu200_chain_v6_k20_test.pt \\
    #   --weights model_weights/.../epoch_107.pyt --threshold 0.98 -d 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
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


class Composition:
    """Accumulates the per-vertex track composition over many events."""

    def __init__(self) -> None:
        self.truthless_frac: list[float] = []
        self.dominant_frac: list[float] = []
        self.sizes: list[int] = []
        self.n_truthless_wins = 0  # truthless bucket strictly the largest
        self.n_truthless_ties = 0  # tie with the largest truth bucket
        self.n_vertices = 0

    def add(self, matched: list[np.ndarray], track_truth: np.ndarray) -> None:
        """Bucket one event's vertices exactly as classify_assignments does."""
        for tracks in matched:
            n = len(tracks)
            if n == 0:
                continue  # empty vertices are Fake by a separate branch
            self.n_vertices += 1
            truths = track_truth[tracks]
            n_truthless = int((truths < 0).sum())
            real = truths[truths >= 0]
            max_truth = (
                int(np.unique(real, return_counts=True)[1].max()) if len(real) else 0
            )

            self.truthless_frac.append(n_truthless / n)
            # The Clean/Merged test uses max_value / w_total_reco, where
            # max_value is the winning bucket's count over ALL assigned tracks.
            self.dominant_frac.append(max(max_truth, n_truthless) / n)
            self.sizes.append(n)
            if n_truthless > max_truth:
                self.n_truthless_wins += 1
            elif n_truthless == max_truth and n_truthless > 0:
                self.n_truthless_ties += 1

    def report(self) -> dict:
        """Summary statistics, all measured."""
        tf = np.asarray(self.truthless_frac)
        df = np.asarray(self.dominant_frac)
        sz = np.asarray(self.sizes)
        qs = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        return {
            "n_vertices_with_tracks": self.n_vertices,
            "median_tracks_per_vertex": float(np.median(sz)) if len(sz) else 0.0,
            "truthless_fraction": {
                "mean": float(tf.mean()) if len(tf) else 0.0,
                "median": float(np.median(tf)) if len(tf) else 0.0,
                "quantiles": {str(q): float(np.quantile(tf, q)) for q in qs}
                if len(tf)
                else {},
                "frac_vertices_all_truthless": float((tf == 1.0).mean())
                if len(tf)
                else 0.0,
                "frac_vertices_zero_truthless": float((tf == 0.0).mean())
                if len(tf)
                else 0.0,
            },
            "dominant_bucket_fraction": {
                "median": float(np.median(df)) if len(df) else 0.0,
                "quantiles": {str(q): float(np.quantile(df, q)) for q in qs}
                if len(df)
                else {},
            },
            "truthless_bucket_wins_plurality": self.n_truthless_wins,
            "truthless_bucket_ties_plurality": self.n_truthless_ties,
            "implied_fake_rate": self.n_truthless_wins / max(self.n_vertices, 1),
            "clean_rate_vs_purity_cut": {
                str(c): float((df >= c).mean()) if len(df) else 0.0 for c in PURITY_GRID
            },
        }


def amvf_pass(args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    """Classify AMVF vertices and measure their composition."""
    tree = uproot.open(args.root)["PVFinderData"]
    # The mu window makes the selection non-contiguous, so the entry list
    # written by pu200_chain_graphs is the only correct way back to the events.
    entries = np.load(args.entry_indices)
    print(f"{len(entries)} events from {args.entry_indices}")

    totals = np.zeros(6, dtype=np.int64)
    comp = Composition()
    for event in tqdm(
        iterate_entries(tree, BRANCHES, entries, step_size=200),
        total=len(entries),
        desc="AMVF events",
    ):
        pt = np.asarray(event["RecoTrack_pT"], dtype=np.float64)
        n_tracks = len(np.asarray(event["RecoTrack_z0"]))
        tti, tpi, truth_count = truth_arrays(event, n_tracks)
        track_truth = np.full(n_tracks, -1, dtype=np.int64)
        track_truth[tti.astype(np.int64)] = tpi.astype(np.int64)

        matched = amvf_matched_lists(event, n_tracks)
        rows, _ = classify_assignments(matched, pt, tti, tpi, truth_count)
        totals += np.array(rows, dtype=np.int64)
        comp.add(matched, track_truth)

    clean, merged, split, fake, n_reco, n_truth = (int(x) for x in totals)
    n_ev = len(entries)
    summary = {
        "clean": clean, "merged": merged, "split": split, "fake": fake,
        "n_reco": n_reco, "n_truth": n_truth,
        "reco_per_event": n_reco / n_ev, "truth_per_event": n_truth / n_ev,
        "clean_rate": clean / n_reco, "merged_rate": merged / n_reco,
        "split_rate": split / n_reco, "fake_rate": fake / n_reco,
        "clean_per_truth": clean / n_truth,
    }  # fmt: skip
    return {"summary": summary, "composition": comp.report()}, entries


def chain_pass(args: argparse.Namespace) -> dict:
    """Same measurement on the chain peaks a trained associator produced."""
    from gnn.evaluation.chain_scan import cache_event, matched_lists
    from gnn.models.ttva_gat import TTVAGATModel

    device = torch.device(
        f"cuda:{args.device_id}"
        if args.device_id >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    # Construct with explicit sizes and load the state dict, exactly as
    # gnn.evaluation.chain_scan does: load_state_dict populates the lazy
    # GATConv parameters, so no materializing forward pass is needed (and a
    # forward before .to(device) puts the encoder weights on the wrong device).
    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    graphs = torch.load(args.graphs, weights_only=False)

    totals = np.zeros(6, dtype=np.int64)
    comp = Composition()
    for graph in tqdm(graphs, desc=f"chain peaks t={args.threshold}"):
        cached = cache_event(model, graph, device)
        matched = matched_lists(cached, args.threshold)
        rows, _ = classify_assignments(
            matched, cached["pt"], cached["tti"], cached["tpi"],
            cached["truth_count"],
        )  # fmt: skip
        totals += np.array(rows, dtype=np.int64)
        comp.add(matched, cached["track_truth"])

    clean, merged, split, fake, n_reco, n_truth = (int(x) for x in totals)
    return {
        "weights": args.weights,
        "threshold": args.threshold,
        "summary": {
            "clean": clean,
            "merged": merged,
            "split": split,
            "fake": fake,
            "n_reco": n_reco,
            "n_truth": n_truth,
            "clean_rate": clean / max(n_reco, 1),
            "fake_rate_all_peaks": fake / max(n_reco, 1),
            "clean_per_truth": clean / max(n_truth, 1),
        },  # fmt: skip
        "composition": comp.report(),
    }


def _print_composition(label: str, c: dict) -> None:
    """Human-readable composition block."""
    tf = c["truthless_fraction"]
    print(f"\n--- {label}: track composition "
          f"(n={c['n_vertices_with_tracks']}, median {c['median_tracks_per_vertex']:.0f} trk) ---")  # fmt: skip
    print(f"  truthless-track fraction: median {tf['median']:.4f}  "
          f"mean {tf['mean']:.4f}")  # fmt: skip
    print(f"    quantiles { {k: round(v, 3) for k, v in tf['quantiles'].items()} }")
    print(
        f"    vertices with NO truthless tracks : {tf['frac_vertices_zero_truthless']:.4f}"
    )
    print(
        f"    vertices that are ALL truthless   : {tf['frac_vertices_all_truthless']:.4f}"
    )
    print(f"  truthless bucket wins the plurality : "
          f"{c['truthless_bucket_wins_plurality']} "
          f"({c['implied_fake_rate']:.5f})  [ties: {c['truthless_bucket_ties_plurality']}]")  # fmt: skip
    print("  Clean rate vs purity cut (0.7 is PURITY_THRESHOLD):")
    for cut, v in c["clean_rate_vs_purity_cut"].items():
        print(f"    >= {cut} : {v:.4f}")


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"root": args.root, "entry_indices": args.entry_indices}
    amvf, entries = amvf_pass(args)
    report["n_events"] = len(entries)
    report["amvf"] = amvf

    s = amvf["summary"]
    print(f"\nAMVF, track convention ({len(entries)} events):")
    print(f"  reco/evt {s['reco_per_event']:.2f}  truth/evt {s['truth_per_event']:.2f}")
    print(f"  clean {s['clean_rate']:.4f}  merged {s['merged_rate']:.4f}  "
          f"split {s['split_rate']:.4f}  fake {s['fake_rate']:.5f}")  # fmt: skip
    print(f"  clean/truth {s['clean_per_truth']:.4f}")
    _print_composition("AMVF vertices", amvf["composition"])

    if args.graphs and args.weights:
        chain = chain_pass(args)
        report["chain"] = chain
        cs = chain["summary"]
        print(f"\nChain peaks at t={args.threshold}: clean/truth "
              f"{cs['clean_per_truth']:.4f}, fake_rate (all_peaks) "
              f"{cs['fake_rate_all_peaks']:.5f}")  # fmt: skip
        _print_composition(f"chain peaks t={args.threshold}", chain["composition"])

    with open(out_dir / "amvf_convention.json", "w") as f:
        json.dump(report, f, indent=2)
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
    p.add_argument("--graphs", default=None, type=str, help="chain graphs .pt")
    p.add_argument("--weights", default=None, type=str, help="associator .pyt")
    p.add_argument("--threshold", default=0.98, type=float)
    p.add_argument("-d", "--device-id", default=-1, type=int)
    p.add_argument("-o", "--output-dir", required=True, type=str)
    return p.parse_args()


if __name__ == "__main__":
    main()
