"""Knife-edge-tolerant regression guard for TTVA eval reproducibility.

The GNN forward on GPU is nondeterministic at the ~1e-5 level (GATConv
scatter atomics). An event's MaxScore selection is unstable when some
track's top-2 PV scores are closer than that, or its max score sits on the
selection threshold — such events may legitimately flip between runs.
The guard therefore passes if per-event result rows are bit-exact against
the saved baseline, or if every differing event is a knife-edge event.

Split out of threshold_scan.py to respect the 500-line file limit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

KNIFE_EDGE_TOL = 1e-4


def _is_knife_edge(cache_ev: dict[str, Any], threshold: float) -> bool:
    """True if the event has a near-degenerate argmax or threshold decision."""
    scores = cache_ev["scores"]
    tracks = cache_ev["edge_index"][0]
    order = np.lexsort((scores, tracks))
    sorted_tracks = tracks[order]
    sorted_scores = scores[order]
    boundaries = np.searchsorted(
        sorted_tracks, np.arange(sorted_tracks.max() + 2 if len(tracks) else 1)
    )
    for trk in range(len(boundaries) - 1):
        start, end = boundaries[trk], boundaries[trk + 1]
        if end == start:
            continue
        top1 = sorted_scores[end - 1]
        if abs(top1 - threshold) < KNIFE_EDGE_TOL:
            return True
        if end - start >= 2 and top1 - sorted_scores[end - 2] < KNIFE_EDGE_TOL:
            return True
    return False


def check_regression(
    rows: list[list[int]],
    reference_path: str | Path,
    cache: list[dict[str, Any]],
    threshold: float,
) -> bool:
    """Compare per-event rows to the saved baseline.

    Bit-exact match passes outright. Otherwise, every differing event must
    be a knife-edge event (see _is_knife_edge) for the guard to pass.
    """
    reference = np.load(reference_path, allow_pickle=True)
    ref_arr = np.array(reference.tolist(), dtype=np.int64)
    new_arr = np.array(rows, dtype=np.int64)
    if ref_arr.shape != new_arr.shape:
        print(f"Regression guard vs {reference_path}: FAIL (shape mismatch)")
        return False
    diff = np.nonzero((ref_arr != new_arr).any(axis=1))[0]
    if len(diff) == 0:
        print(f"Regression guard vs {reference_path}: PASS (bit-exact)")
        return True
    knife = [int(i) for i in diff if _is_knife_edge(cache[i], threshold)]
    ok = len(knife) == len(diff)
    status = "PASS (knife-edge only)" if ok else "FAIL"
    print(
        f"Regression guard vs {reference_path}: {status} — "
        f"{len(diff)} differing events {diff.tolist()[:10]}, "
        f"{len(knife)} of them knife-edge (GPU-nondeterminism tolerance)"
    )
    return ok
