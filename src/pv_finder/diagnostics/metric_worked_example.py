"""One synthetic event, walked through both vertex taxonomies.

The project classifies reconstructed vertices as clean / merged / split / fake
under two different definitions that share those four words:

* **positional** — ``compare_res_reco`` (the PV-Finder evaluation): a reco
  vertex is judged by whether it sits inside a matching window of a truth
  vertex.  It never looks at tracks.
* **track-purity** — ``classify_assignments`` (the TTVA / chain evaluation): a
  reco vertex is judged by whether >= ``PURITY_THRESHOLD`` of the tracks
  assigned to it come from one truth vertex.  It never looks at z.

This module builds one small event in which the disagreement is legible, runs
it through the *production* functions (not a reimplementation), and prints the
step-by-step trace.  ``tests/test_metric_worked_example.py`` pins every number,
so the worked example in ``docs/evaluation/metric_definitions.md`` cannot drift
away from the code.

Run::

    python -m pv_finder.diagnostics.metric_worked_example
"""

from __future__ import annotations

import contextlib
import io
import re
from typing import Any, NamedTuple

import numpy as np

from gnn.evaluation.classification import build_truth_adjacency, classify_assignments
from pv_finder.evaluation.vertex_finding.efficiency_res_optimized_atlas import (
    compare_res_reco,
)

Z_MIN: float = -240.0
BIN_WIDTH: float = 0.04
WINDOW_MM: float = 0.22  # ~ the fitted sigma_vtx-vtx of the v6 operating point
MIN_TRUTH_NTRK: int = 2  # the nTrk cut run3_io applies to truth and to AMVF


class Event(NamedTuple):
    """A synthetic event, described the way both taxonomies need it.

    ``truth_z`` / ``truth_tracks`` list **every** truth vertex, including ones
    below the nTrk cut — that cut is a property of the evaluation, not of the
    event, and applying it is one of the things this example demonstrates.
    """

    truth_z: np.ndarray
    truth_tracks: list[np.ndarray]
    reco_z: np.ndarray
    reco_tracks: list[np.ndarray]
    n_tracks: int
    truth_names: list[str]
    reco_names: list[str]


def build_event() -> Event:
    """The worked example: 6 truth vertices (one sub-threshold), 6 reco peaks.

    Every category is exercised, and two reco vertices are classified
    *differently* by the two taxonomies — in opposite directions, so the
    disagreement cannot be read as one convention simply being stricter.

    ==== ======= ====== ======================================================
    id   z (mm)  nTrk   role
    ==== ======= ====== ======================================================
    T0   -5.00   15     reconstructed by P_b
    T1   -4.70   10     no peak of its own; P_b absorbs 6 of its tracks
    T2   -0.08   15     close pair with T3, 0.20 mm apart
    T3   +0.12   12     close pair with T2
    T4   +8.10    9     isolated; P_e and P_f share its tracks
    T5   +3.00    1     a real interaction, dropped by the nTrk >= 2 cut
    ==== ======= ====== ======================================================

    ==== ======= ==========================================================
    id   z (mm)  tracks assigned by the associator
    ==== ======= ==========================================================
    P_a  -5.14   6 of T0                (split-off inside the window)
    P_b  -5.02   9 of T0 + 6 of T1      (right z, contaminated track list)
    P_c   0.00   15 of T2 + 8 of T3     (one peak over a close pair)
    P_d  +3.00   T5's track + 3 truthless  (junk peak sitting on T5)
    P_e  +8.10   5 of T4                (clean under both conventions)
    P_f  +8.42   4 of T4                (split-off *outside* the window)
    ==== ======= ==========================================================
    """
    truth_z = np.array([-5.00, -4.70, -0.08, 0.12, 8.10, 3.00], dtype=np.float32)
    truth_tracks = [
        np.arange(0, 15),  # T0, 15 tracks
        np.arange(15, 25),  # T1, 10 tracks
        np.arange(25, 40),  # T2, 15 tracks
        np.arange(40, 52),  # T3, 12 tracks
        np.arange(52, 61),  # T4, 9 tracks
        np.arange(61, 62),  # T5, 1 track  -> below the nTrk >= 2 cut
    ]
    # 62, 63, 64 belong to no truth vertex (secondaries, or interactions that
    # left no truth-matched track).  65 tracks in total.
    reco_z = np.array([-5.14, -5.02, 0.00, 3.00, 8.10, 8.42], dtype=np.float32)
    reco_tracks = [
        np.arange(9, 15),  # P_a: 6 of T0
        np.concatenate([np.arange(0, 9), np.arange(15, 21)]),  # P_b: 9 T0 + 6 T1
        np.concatenate([np.arange(25, 40), np.arange(40, 48)]),  # P_c: 15 T2 + 8 T3
        np.array([61, 62, 63, 64]),  # P_d: 1 track of T5 + 3 truthless
        np.arange(52, 57),  # P_e: 5 of T4
        np.arange(57, 61),  # P_f: 4 of T4
    ]
    return Event(
        truth_z=truth_z,
        truth_tracks=truth_tracks,
        reco_z=reco_z,
        reco_tracks=reco_tracks,
        n_tracks=65,
        truth_names=["T0", "T1", "T2", "T3", "T4", "T5"],
        reco_names=["P_a", "P_b", "P_c", "P_d", "P_e", "P_f"],
    )


# ---------------------------------------------------------------------------
# Positional taxonomy  (efficiency_res_optimized_atlas.compare_res_reco)
# ---------------------------------------------------------------------------


def positional_pass(
    event: Event,
    window_mm: float = WINDOW_MM,
    min_truth_ntrk: int = MIN_TRUTH_NTRK,
) -> dict[str, Any]:
    """Run the finder's positional classifier exactly as ``run_eval_pvf_run3`` does.

    Truth is filtered to ``nTrk >= min_truth_ntrk`` (``run3_io._filter_amvf``),
    positions are converted to bin units, and every reco vertex is given the
    same window, ``sigma_vtx-vtx / BIN_WIDTH`` bins.
    """
    ntrk = np.array([len(t) for t in event.truth_tracks])
    keep = np.nonzero(ntrk >= min_truth_ntrk)[0]
    order = keep[np.argsort(event.truth_z[keep], kind="stable")]
    truth_z = event.truth_z[order]
    truth_labels = [event.truth_names[i] for i in order]

    truth_bins = (truth_z - Z_MIN) / BIN_WIDTH
    reco_bins = (event.reco_z - Z_MIN) / BIN_WIDTH
    window_bins = window_mm / BIN_WIDTH

    perf, truth_cls, _ = compare_res_reco(
        truth_bins, reco_bins, window_bins * np.ones(len(reco_bins)), debug=0
    )
    n_truth = len(truth_bins)
    n_clean = int(np.sum(truth_cls == "clean"))
    n_merged = int(np.sum(truth_cls == "merged"))
    n_missed = int(np.sum(truth_cls == "missed"))
    return {
        "window_mm": window_mm,
        "min_truth_ntrk": min_truth_ntrk,
        "truth_labels": truth_labels,
        "truth_cls": list(truth_cls),
        "n_truth": n_truth,
        "n_reco": len(reco_bins),
        "clean": perf.reco_clean,
        "merged": perf.reco_merged,
        "split": perf.reco_split,
        "fake": perf.reco_fake,
        "truth_clean": n_clean,
        "truth_merged": n_merged,
        "truth_missed": n_missed,
        "efficiency": (n_clean + n_merged) / n_truth,
        "fake_per_event": float(perf.reco_fake),
        "clean_per_truth": perf.reco_clean / n_truth,
        "reco_cls": _reco_labels(truth_bins, reco_bins, window_bins),
    }


_DEBUG_LINE = re.compile(r"^\s*reco (\d+): (\w+),")


def _reco_labels(
    truth_bins: np.ndarray, reco_bins: np.ndarray, window_bins: float
) -> list[str]:
    """Per-reco positional label, taken from ``compare_res_reco``'s own trace.

    ``compare_res_reco`` returns only totals, but with ``debug=1`` it prints its
    per-vertex decision.  Capturing that is the only way to get the production
    labels without reimplementing the classifier — and the parsed labels are
    cross-checked against the returned totals, so a change to either the debug
    format or the logic fails loudly instead of printing a plausible lie.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        perf, _, _ = compare_res_reco(
            truth_bins, reco_bins, window_bins * np.ones(len(reco_bins)), debug=1
        )
    labels: list[str] = [""] * len(reco_bins)
    for line in buf.getvalue().splitlines():
        match = _DEBUG_LINE.match(line)
        if match:
            labels[int(match.group(1))] = match.group(2)
    counted = {k: labels.count(k) for k in ("clean", "merged", "split", "fake")}
    expected = {
        "clean": perf.reco_clean,
        "merged": perf.reco_merged,
        "split": perf.reco_split,
        "fake": perf.reco_fake,
    }
    if counted != expected:
        msg = (
            "compare_res_reco debug trace does not reproduce its own totals: "
            f"parsed {counted}, returned {expected}. The debug print format in "
            "efficiency_res_optimized_atlas.compare_res_reco has changed."
        )
        raise RuntimeError(msg)
    return labels


# ---------------------------------------------------------------------------
# Track-purity taxonomy  (gnn.evaluation.classification.classify_assignments)
# ---------------------------------------------------------------------------


def track_pass(
    event: Event, min_truth_ntrk: int = MIN_TRUTH_NTRK
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the TTVA classifier on the same event.

    Note the asymmetry the production code has: the truth *adjacency* covers
    every truth vertex, while ``truth_pvs_count`` — the denominator — counts
    only those with ``nTrk >= min_truth_ntrk``.
    """
    pv_ntracks = np.array([len(t) for t in event.truth_tracks])
    assoc = np.concatenate(event.truth_tracks).astype(np.float64)
    tti, tpi = build_truth_adjacency(event.truth_z, pv_ntracks, assoc)
    truth_count = int((pv_ntracks >= min_truth_ntrk).sum())

    pt = np.ones(event.n_tracks, dtype=np.float64)  # unit pT: sum(pT^2) = nTracks
    matched = [np.asarray(t, dtype=np.int64) for t in event.reco_tracks]
    rows, info = classify_assignments(matched, pt, tti, tpi, truth_count)

    clean, merged, split, fake, n_reco, n_truth = rows
    return {
        "clean": clean,
        "merged": merged,
        "split": split,
        "fake": fake,
        "n_reco": n_reco,
        "n_truth": n_truth,
        "clean_rate": clean / n_reco,
        "fake_rate": fake / n_reco,
        "clean_per_truth": clean / n_truth,
    }, info


# ---------------------------------------------------------------------------
# The order-dependence demonstration
# ---------------------------------------------------------------------------


def order_dependence(window_mm: float = WINDOW_MM) -> dict[str, Any]:
    """Show that the positional reco-side clean/merged split is not mirror-symmetric.

    Four truth vertices and two reco vertices, arranged so that both reco
    vertices have an unclaimed truth vertex in their window.  ``compare_res_reco``
    walks the reco list in index order and lets the first one it reaches absorb
    the shared neighbour, so reflecting the event in z moves a vertex between
    ``clean`` and ``merged``.  Truth-side efficiency is unaffected, because the
    set of truth vertices claimed by *some* assigned reco does not depend on the
    walk order.
    """
    truth = np.array([-0.15, 0.00, 0.15, 0.30])
    reco = np.array([0.00, 0.30])
    out: dict[str, Any] = {"window_mm": window_mm}
    for tag, sign in (("nominal", 1.0), ("mirrored", -1.0)):
        t_z = np.sort(sign * truth)
        r_z = np.sort(sign * reco)
        perf, truth_cls, _ = compare_res_reco(
            (t_z - Z_MIN) / BIN_WIDTH,
            (r_z - Z_MIN) / BIN_WIDTH,
            (window_mm / BIN_WIDTH) * np.ones(len(r_z)),
            debug=0,
        )
        out[tag] = {
            "clean": perf.reco_clean,
            "merged": perf.reco_merged,
            "split": perf.reco_split,
            "fake": perf.reco_fake,
            "efficiency": float(np.mean(truth_cls != "missed")),
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_positional(title: str, res: dict[str, Any]) -> None:
    """Print one positional pass."""
    print(f"\n{title}")
    print(f"  window {res['window_mm']:.2f} mm, truth filtered to "
          f"nTrk >= {res['min_truth_ntrk']}  ->  {res['n_truth']} truth vertices")  # fmt: skip
    print(f"  reco : clean {res['clean']}  merged {res['merged']}  "
          f"split {res['split']}  fake {res['fake']}   (n_reco {res['n_reco']})")  # fmt: skip
    print(f"  truth: clean {res['truth_clean']}  merged {res['truth_merged']}  "
          f"missed {res['truth_missed']}")  # fmt: skip
    print(f"  efficiency = ({res['truth_clean']}+{res['truth_merged']})/"
          f"{res['n_truth']} = {res['efficiency']:.4f}   "
          f"fake/evt = {res['fake_per_event']:.1f}   "
          f"clean/truth = {res['clean_per_truth']:.4f}")  # fmt: skip


def main() -> None:
    """Print the whole worked example."""
    event = build_event()
    print("=" * 72)
    print("  One event, two taxonomies")
    print("=" * 72)
    ntrk = [len(t) for t in event.truth_tracks]
    print("\nTruth vertices:")
    for name, z, n in zip(event.truth_names, event.truth_z, ntrk):
        flag = "" if n >= MIN_TRUTH_NTRK else "   <- dropped by the nTrk >= 2 cut"
        print(f"  {name}  z = {z:+7.2f} mm   nTrk = {n:2d}{flag}")
    print("\nReco vertices:")
    for name, z, trk in zip(event.reco_names, event.reco_z, event.reco_tracks):
        print(f"  {name}  z = {z:+7.2f} mm   {len(trk):2d} tracks assigned")

    base = positional_pass(event)
    print("\n" + "-" * 72)
    print("  POSITIONAL taxonomy — compare_res_reco")
    print("-" * 72)
    for name, label in zip(event.reco_names, base["reco_cls"]):
        print(f"  {name}: {label}")
    for name, label in zip(base["truth_labels"], base["truth_cls"]):
        print(f"  {name}: {label}")
    _print_positional("Totals", base)

    print("\n" + "-" * 72)
    print("  TRACK-PURITY taxonomy — classify_assignments")
    print("-" * 72)
    trk, info = track_pass(event)
    for name, i in zip(event.reco_names, info):
        purity = (
            i["primary_truth_pv_weight"] / i["w_total_reco"] if i["w_total_reco"] else 0
        )
        print(f"  {name}: {i['classification']:<7} dominant {str(i['primary_truth_pv']):<11}"
              f" {i['primary_truth_pv_weight']:2d}/{i['w_total_reco']:2d} = {purity:.3f}"
              f"   contributions {i['contributions']}")  # fmt: skip
    print(f"\n  reco : clean {trk['clean']}  merged {trk['merged']}  "
          f"split {trk['split']}  fake {trk['fake']}   (n_reco {trk['n_reco']})")  # fmt: skip
    print(f"  n_truth (nTrk >= 2) = {trk['n_truth']}   "
          f"clean/truth = {trk['clean_per_truth']:.4f}   "
          f"clean_rate = {trk['clean_rate']:.4f}   "
          f"fake_rate = {trk['fake_rate']:.4f}")  # fmt: skip

    print("\n" + "-" * 72)
    print("  SIDE BY SIDE")
    print("-" * 72)
    print(f"  {'vertex':<8}{'positional':<14}{'track-purity':<14}")
    for name, a, i in zip(event.reco_names, base["reco_cls"], info):
        mark = "   <-- disagree" if a.lower() != i["classification"].lower() else ""
        print(f"  {name:<8}{a:<14}{i['classification'].lower():<14}{mark}")
    print(f"\n  clean/truth: positional {base['clean_per_truth']:.4f}   "
          f"track-purity {trk['clean_per_truth']:.4f}")  # fmt: skip

    print("\n" + "-" * 72)
    print("  VARIANTS — one convention changed at a time")
    print("-" * 72)
    _print_positional("(W) same event, matching window 0.22 -> 0.35 mm",
                      positional_pass(event, window_mm=0.35))  # fmt: skip
    _print_positional("(N) same event, truth NOT filtered to nTrk >= 2",
                      positional_pass(event, min_truth_ntrk=1))  # fmt: skip

    print("\n(O) positional reco-side labels are not mirror-symmetric in z")
    od = order_dependence()
    for tag in ("nominal", "mirrored"):
        r = od[tag]
        print(f"  {tag:<10} clean {r['clean']}  merged {r['merged']}  "
              f"split {r['split']}  fake {r['fake']}   "
              f"efficiency {r['efficiency']:.4f}")  # fmt: skip
    print("  -> reco-side clean/merged moves, truth-side efficiency does not.")


if __name__ == "__main__":
    main()
