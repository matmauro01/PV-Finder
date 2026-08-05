"""Vertex matching for the fair PV-Finder / AMVF comparison.

``match_vertices`` reproduces the classification of
``efficiency_res_optimized_atlas.compare_res_reco`` exactly — same greedy
closest-first assignment, same pass-2 merge claiming, same tie ordering — but
drops the O(n_truth^2) local-density calculation that the production matcher
computes and the caller here never uses, and returns the *truth-side*
efficiency, which the production eval computes for PV-Finder and silently
discards for AMVF.

``tests/test_amvf_fair_matching.py`` asserts the two agree count-for-count.  Do
not take a number from this module without that test passing: the whole point of
the study is that the comparison is symmetric, which is worth nothing if the
matcher itself has drifted from the reference.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Cost assigned to a pair outside the matching window in the optimal-assignment
# residual measurement.  Any value far larger than the window works.
_UNMATCHABLE: float = 1.0e6


class MatchCounts(NamedTuple):
    """Outcome of matching one event's reco vertices against truth.

    ``n_truth_matched`` is ``clean + merged`` on the *truth* side and is the
    numerator of the efficiency the production eval quotes; it is not in general
    equal to ``reco_clean + reco_merged``, because one merged reco absorbs two or
    more truth vertices and is credited with all of them.

    ``n_truth_primary`` is the stricter numerator: truth vertices that won a
    *dedicated* reco vertex in the closest-first pass, i.e. one-to-one with no
    merge credit.  The difference between the two is the merge convention, which
    is close to the ATLAS "merged" definition and is a convention choice rather
    than a bug — but it inflates the absolute efficiency, by more as the window
    widens, so both are carried here and the window scan quotes both.
    """

    n_truth: int
    n_reco: int
    n_truth_matched: int
    n_truth_primary: int
    reco_clean: int
    reco_merged: int
    reco_split: int
    reco_fake: int
    fake_idx: np.ndarray


def match_vertices(
    truth_z: np.ndarray,
    reco_z: np.ndarray,
    window_mm: float,
) -> MatchCounts:
    """Match reco vertices to truth vertices within a common window.

    Parameters
    ----------
    truth_z, reco_z:
        Vertex positions in mm.  Any consistent frame; the caller is
        responsible for using the same one for both.
    window_mm:
        Half-width of the matching window, in mm, applied identically to every
        reco vertex of every algorithm.  This is the quantity the production
        eval sets to PV-Finder's own fitted sigma.

    Returns
    -------
    MatchCounts
    """
    n_truth = int(len(truth_z))
    n_reco = int(len(reco_z))
    if n_truth == 0 or n_reco == 0:
        return MatchCounts(
            n_truth=n_truth,
            n_reco=n_reco,
            n_truth_matched=0,
            n_truth_primary=0,
            reco_clean=0,
            reco_merged=0,
            reco_split=0,
            reco_fake=n_reco,
            fake_idx=np.arange(n_reco, dtype=np.int64),
        )

    # (n_reco, n_truth) distances.  Deliberately computed in the caller's dtype,
    # elementwise, exactly as the reference does it row by row — casting to
    # float64 here would move pairs across the ``<=`` boundary relative to the
    # reference and the bit-for-bit test would fail for no physical reason.
    dist = np.abs(reco_z[:, None] - truth_z[None, :])
    # np.nonzero returns row-major order, i.e. reco-major with truth ascending —
    # the same order in which the reference builds its ``pairs`` list.  That
    # matters because the sort below is stable, so distance ties are broken
    # identically.
    ri_all, tj_all = np.nonzero(dist <= window_mm)
    neighbors: list[list[int]] = [[] for _ in range(n_reco)]
    for ri, tj in zip(ri_all.tolist(), tj_all.tolist()):
        neighbors[ri].append(tj)

    order = np.argsort(dist[ri_all, tj_all], kind="stable")

    # Pass 1: greedy closest-first 1-to-1 assignment.
    reco_to_truth = np.full(n_reco, -1, dtype=np.int64)
    truth_to_reco = np.full(n_truth, -1, dtype=np.int64)
    for ri, tj in zip(ri_all[order].tolist(), tj_all[order].tolist()):
        if reco_to_truth[ri] == -1 and truth_to_reco[tj] == -1:
            reco_to_truth[ri] = tj
            truth_to_reco[tj] = ri
    # Strict one-to-one numerator, before any merge credit is handed out below.
    n_truth_primary = int(np.count_nonzero(truth_to_reco >= 0))

    # Pass 2: classify reco.  Order matters — a reco claims its unmatched
    # neighbours, which removes them from the pool for later reco.
    reco_clean = reco_merged = reco_split = reco_fake = 0
    fake_idx: list[int] = []
    for ri in range(n_reco):
        if reco_to_truth[ri] == -1:
            # Unassigned: its truth neighbours were all claimed by a closer reco
            # ("split"), or it had none at all ("fake").
            if neighbors[ri]:
                reco_split += 1
            else:
                reco_fake += 1
                fake_idx.append(ri)
            continue
        unmatched = [tj for tj in neighbors[ri] if truth_to_reco[tj] == -1]
        if unmatched:
            reco_merged += 1
            for tj in unmatched:
                truth_to_reco[tj] = ri
        else:
            reco_clean += 1

    n_truth_matched = int(np.count_nonzero(truth_to_reco >= 0))
    return MatchCounts(
        n_truth=n_truth,
        n_reco=n_reco,
        n_truth_matched=n_truth_matched,
        n_truth_primary=n_truth_primary,
        reco_clean=reco_clean,
        reco_merged=reco_merged,
        reco_split=reco_split,
        reco_fake=reco_fake,
        fake_idx=np.asarray(fake_idx, dtype=np.int64),
    )


def matched_residuals(
    truth_z: np.ndarray,
    reco_z: np.ndarray,
    window_mm: float,
) -> np.ndarray:
    """|Δz| for reco/truth pairs assigned one-to-one, closest first.

    Used to measure the *core position resolution* — how well an algorithm
    locates a vertex it has found — which is a different quantity from
    sigma_vtx-vtx (how far apart two vertices must be before the algorithm
    reports them separately).  Conflating the two is what put a window derived
    from sigma_vtx-vtx into the eval.

    The window should be generous (1 mm) so the percentile is set by the core
    and not by the cut.
    """
    if len(truth_z) == 0 or len(reco_z) == 0:
        return np.zeros(0, dtype=np.float64)
    dist = np.abs(reco_z[:, None] - truth_z[None, :])
    ri_all, tj_all = np.nonzero(dist <= window_mm)
    if len(ri_all) == 0:
        return np.zeros(0, dtype=np.float64)
    d = dist[ri_all, tj_all]
    order = np.argsort(d, kind="stable")
    used_r = np.zeros(len(reco_z), dtype=bool)
    used_t = np.zeros(len(truth_z), dtype=bool)
    out: list[float] = []
    for k in order.tolist():
        ri, tj = int(ri_all[k]), int(tj_all[k])
        if not used_r[ri] and not used_t[tj]:
            used_r[ri] = used_t[tj] = True
            out.append(float(d[k]))
    return np.asarray(out, dtype=np.float64)


def matched_residuals_optimal(
    truth_z: np.ndarray,
    reco_z: np.ndarray,
    window_mm: float,
) -> np.ndarray:
    """|Δz| for reco/truth pairs under a globally optimal assignment.

    Greedy closest-first is the right convention for *scoring*, because it is
    what the production eval does — but it is the wrong one for estimating a
    resolution.  It takes the tightest pair in the event first and works
    outwards, so it creams off the core and leaves the tail unmatched, biasing
    the width low: on 400 held-out events the p68 comes out 0.0510 mm greedy
    against 0.0563 mm optimal for PV-Finder, a 10 % difference.

    The window used for the assignment is deliberately generous so the
    percentile is set by the core and not by the cut.
    """
    if len(truth_z) == 0 or len(reco_z) == 0:
        return np.zeros(0, dtype=np.float64)
    from scipy.optimize import linear_sum_assignment

    dist = np.abs(
        reco_z[:, None].astype(np.float64) - truth_z[None, :].astype(np.float64)
    )
    # Pairs outside the window are made prohibitively expensive rather than
    # forbidden, so the assignment always exists; they are dropped afterwards.
    cost = np.where(dist <= window_mm, dist, _UNMATCHABLE)
    ri, tj = linear_sum_assignment(cost)
    d = dist[ri, tj]
    return d[d <= window_mm]


def excused_by_low_ntrk(
    fake_z: np.ndarray,
    low_ntrk_truth_z: np.ndarray,
    window_mm: float,
) -> int:
    """Count fakes that sit on a real but non-reconstructible truth interaction.

    Greedy closest-first and **exclusive**: one nTrk == 1 truth interaction can
    excuse at most one reconstructed vertex.  Without exclusivity a single low
    track-count interaction would excuse a whole cluster of nearby surplus
    peaks, which would overstate the correction.

    Parameters
    ----------
    fake_z:
        Positions of the reco vertices classified ``fake`` against the
        nTracks >= 2 truth set.
    low_ntrk_truth_z:
        Positions of the truth interactions excluded by the nTracks >= 2 cut
        (here always exactly nTracks == 1; the sample contains no nTracks == 0).
    window_mm:
        The same common window used for the primary matching.

    Returns
    -------
    int
        Number of fakes matched 1-to-1 to a low-nTrk truth interaction.
    """
    if len(fake_z) == 0 or len(low_ntrk_truth_z) == 0:
        return 0
    dist = np.abs(fake_z[:, None] - low_ntrk_truth_z[None, :])
    fi, tj = np.nonzero(dist <= window_mm)
    if len(fi) == 0:
        return 0
    order = np.argsort(dist[fi, tj], kind="stable")
    fi, tj = fi[order], tj[order]
    used_f = np.zeros(len(fake_z), dtype=bool)
    used_t = np.zeros(len(low_ntrk_truth_z), dtype=bool)
    n = 0
    for f, t in zip(fi.tolist(), tj.tolist()):
        if not used_f[f] and not used_t[t]:
            used_f[f] = used_t[t] = True
            n += 1
    return n
