"""Independent truth-to-reco vertex matching for algorithm comparison.

Written from scratch for the AMVF fairness audit (2026-08-05) deliberately
*not* reusing ``evaluation.vertex_finding.efficiency_res_optimized_atlas``:
the point of the audit was to check that code, so re-deriving the matcher is
the whole method.  Two independent implementations that agree are evidence;
one implementation used twice is not.

Differences from the evaluation matcher, all deliberate:

* **Optimal, not greedy.**  ``match_optimal`` solves the assignment problem
  exactly (Hungarian, via ``scipy.optimize.linear_sum_assignment``) so the
  result cannot depend on the order vertices happen to arrive in.  The
  evaluation matcher assigns greedily closest-pair-first, which is
  order-dependent under exact distance ties.  ``match_greedy`` reproduces the
  greedy rule so the two can be compared directly.

* **One fake definition.**  A reco vertex that wins no truth is a fake, full
  stop.  The evaluation code splits those into ``split`` (had some truth in
  its window, lost it to a closer reco) and ``fake`` (had none), and the
  headline "FP/evt" counts only the latter — so a duplicate reco sitting on
  top of an already-matched truth is free.  ``unmatched_reco`` here counts
  both; ``n_split``/``n_fake`` are reported separately so the difference
  between the two conventions is visible rather than baked in.

* **Symmetric by construction.**  The same function, the same window, and the
  same truth list are used for every algorithm.  Nothing about the matcher
  knows which algorithm produced the reco list.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class MatchResult(NamedTuple):
    """Outcome of matching one event's reco list against one truth list.

    ``pairs`` are ``(truth_index, reco_index)`` of accepted matches, all with
    ``|dz| <= window``.  Every index appears at most once, in either column.
    """

    pairs: np.ndarray  # (n_matched, 2) int
    n_truth: int
    n_reco: int
    n_matched: int
    n_split: int  # unmatched reco that had >=1 truth inside its window
    n_fake: int  # unmatched reco with no truth inside its window
    residuals: np.ndarray  # signed reco - truth for accepted matches, mm

    @property
    def n_unmatched_reco(self) -> int:
        """Every reco that won no truth. ``n_split + n_fake``."""
        return self.n_reco - self.n_matched

    @property
    def n_missed(self) -> int:
        """Truth vertices that won no reco."""
        return self.n_truth - self.n_matched


def _empty(
    n_truth: int, n_reco: int, truth_z: np.ndarray, reco_z: np.ndarray, window: float
) -> MatchResult:
    """Result for a degenerate event (no truth or no reco)."""
    n_fake = n_reco
    n_split = 0
    if n_truth and n_reco:  # unreachable, kept for symmetry of the contract
        n_fake, n_split = 0, n_reco
    return MatchResult(
        pairs=np.zeros((0, 2), dtype=int),
        n_truth=n_truth,
        n_reco=n_reco,
        n_matched=0,
        n_split=n_split,
        n_fake=n_fake,
        residuals=np.zeros(0, dtype=float),
    )


def _classify_unmatched(ok: np.ndarray, matched_ri: np.ndarray) -> tuple[int, int]:
    """Split unmatched reco into (split, fake) using the evaluation's rule.

    ``ok`` is the (n_truth, n_reco) "within window" mask.  ``split`` = unmatched
    reco with at least one truth inside its window, ``fake`` = unmatched reco
    with none.  Reported only so the two fake-rate conventions can be compared;
    the audit's own fake rate is their sum.
    """
    has_truth = ok.any(axis=0)
    unmatched = np.ones(ok.shape[1], dtype=bool)
    unmatched[matched_ri] = False
    return int((unmatched & has_truth).sum()), int((unmatched & ~has_truth).sum())


def match_optimal(
    truth_z: np.ndarray, reco_z: np.ndarray, window: float
) -> MatchResult:
    """Maximum-cardinality, minimum-total-distance 1-to-1 match within ``window``.

    Solves the rectangular assignment problem with forbidden (out-of-window)
    pairs given a cost so large that the optimiser always prefers to make one
    more legal match than to use a single forbidden one.  Assignments landing
    on a forbidden pair are then dropped.  The result is the largest possible
    set of within-window pairs and, among those, the one with the smallest
    total |dz| — independent of input ordering.
    """
    truth_z = np.asarray(truth_z, dtype=np.float64)
    reco_z = np.asarray(reco_z, dtype=np.float64)
    n_truth, n_reco = len(truth_z), len(reco_z)
    if n_truth == 0 or n_reco == 0:
        return _empty(n_truth, n_reco, truth_z, reco_z, window)

    dist = np.abs(truth_z[:, None] - reco_z[None, :])
    ok = dist <= window
    # Strictly greater than the largest total cost any all-legal solution could
    # accumulate, so trading a legal match for a forbidden one is never optimal.
    big = window * (n_truth + n_reco) + 1.0
    cost = np.where(ok, dist, big)

    ti, ri = linear_sum_assignment(cost)
    keep = ok[ti, ri]
    ti, ri = ti[keep], ri[keep]

    n_split, n_fake = _classify_unmatched(ok, ri)
    return MatchResult(
        pairs=np.stack([ti, ri], axis=1) if len(ti) else np.zeros((0, 2), dtype=int),
        n_truth=n_truth,
        n_reco=n_reco,
        n_matched=int(len(ti)),
        n_split=n_split,
        n_fake=n_fake,
        residuals=reco_z[ri] - truth_z[ti],
    )


def match_greedy(truth_z: np.ndarray, reco_z: np.ndarray, window: float) -> MatchResult:
    """Closest-pair-first greedy 1-to-1 match, the evaluation code's rule.

    Provided to measure how much the greedy approximation costs relative to
    :func:`match_optimal`, not because it is preferred.
    """
    truth_z = np.asarray(truth_z, dtype=np.float64)
    reco_z = np.asarray(reco_z, dtype=np.float64)
    n_truth, n_reco = len(truth_z), len(reco_z)
    if n_truth == 0 or n_reco == 0:
        return _empty(n_truth, n_reco, truth_z, reco_z, window)

    dist = np.abs(truth_z[:, None] - reco_z[None, :])
    ok = dist <= window
    ti_all, ri_all = np.where(ok)
    order = np.argsort(dist[ti_all, ri_all], kind="stable")
    used_t: set[int] = set()
    used_r: set[int] = set()
    pairs = []
    for k in order:
        tj, rj = int(ti_all[k]), int(ri_all[k])
        if tj in used_t or rj in used_r:
            continue
        used_t.add(tj)
        used_r.add(rj)
        pairs.append((tj, rj))

    arr = np.asarray(pairs, dtype=int) if pairs else np.zeros((0, 2), dtype=int)
    n_split, n_fake = _classify_unmatched(ok, arr[:, 1])
    res = (reco_z[arr[:, 1]] - truth_z[arr[:, 0]]) if len(arr) else np.zeros(0)
    return MatchResult(
        pairs=arr,
        n_truth=n_truth,
        n_reco=n_reco,
        n_matched=len(pairs),
        n_split=n_split,
        n_fake=n_fake,
        residuals=res,
    )


class Summary(NamedTuple):
    """Event-averaged performance of one algorithm at one window."""

    n_events: int
    efficiency: float
    fake_per_evt: float  # every unmatched reco
    fake_per_evt_excl_split: float  # the evaluation's convention
    split_per_evt: float
    truth_per_evt: float
    reco_per_evt: float
    matched_total: int
    truth_total: int


def summarise(results: list[MatchResult]) -> Summary:
    """Aggregate per-event match results the way the audit quotes them.

    Efficiency is pooled (total matched / total truth), which weights events by
    their vertex count.  Fake rate is per event.
    """
    n = len(results)
    if n == 0:
        return Summary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    matched = sum(r.n_matched for r in results)
    truth = sum(r.n_truth for r in results)
    reco = sum(r.n_reco for r in results)
    unmatched = sum(r.n_unmatched_reco for r in results)
    fake_only = sum(r.n_fake for r in results)
    split = sum(r.n_split for r in results)
    return Summary(
        n_events=n,
        efficiency=matched / truth if truth else 0.0,
        fake_per_evt=unmatched / n,
        fake_per_evt_excl_split=fake_only / n,
        split_per_evt=split / n,
        truth_per_evt=truth / n,
        reco_per_evt=reco / n,
        matched_total=matched,
        truth_total=truth,
    )


def bootstrap_paired(
    per_event: dict[str, list[MatchResult]],
    n_boot: int = 400,
    seed: int = 12345,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Paired bootstrap over events for several algorithms at once.

    Every algorithm is resampled with the *same* event draw, so differences
    between them are paired and their uncertainty reflects the shared event
    sample rather than treating the two as independent.  Returns
    ``{algo: {metric: (value, std)}}``; the ``__diff__`` entry carries the
    per-draw difference between the first two algorithms.
    """
    rng = np.random.default_rng(seed)
    names = list(per_event)
    n = len(per_event[names[0]])
    draws: dict[str, dict[str, list[float]]] = {
        k: {"efficiency": [], "fake_per_evt": []} for k in names
    }
    diff: dict[str, list[float]] = {"efficiency": [], "fake_per_evt": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals = {}
        for k in names:
            rs = [per_event[k][i] for i in idx]
            s = summarise(rs)
            vals[k] = s
            draws[k]["efficiency"].append(s.efficiency)
            draws[k]["fake_per_evt"].append(s.fake_per_evt)
        if len(names) >= 2:
            a, b = vals[names[0]], vals[names[1]]
            diff["efficiency"].append(a.efficiency - b.efficiency)
            diff["fake_per_evt"].append(a.fake_per_evt - b.fake_per_evt)

    out: dict[str, dict[str, tuple[float, float]]] = {}
    for k in names:
        s = summarise(per_event[k])
        out[k] = {
            "efficiency": (s.efficiency, float(np.std(draws[k]["efficiency"], ddof=1))),
            "fake_per_evt": (
                s.fake_per_evt,
                float(np.std(draws[k]["fake_per_evt"], ddof=1)),
            ),
        }
    if len(names) >= 2:
        sa, sb = summarise(per_event[names[0]]), summarise(per_event[names[1]])
        out["__diff__"] = {
            "efficiency": (
                sa.efficiency - sb.efficiency,
                float(np.std(diff["efficiency"], ddof=1)),
            ),
            "fake_per_evt": (
                sa.fake_per_evt - sb.fake_per_evt,
                float(np.std(diff["fake_per_evt"], ddof=1)),
            ),
        }
    return out
