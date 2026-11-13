"""Weighted Clean / Merged / Split / Fake classification of reconstructed vertices.

The ATLAS AMVF public note (ATL-PHYS-PUB-2019-015) classifies a vertex by the
truth composition of its *fit weights*: each truth interaction contributes the
sum of the fit weights of its tracks, normalised to the vertex's total weight.
:func:`classify_vertices_weighted` implements that rule with exactly the
conventions of ``gnn.evaluation.hs_metrics.classify_vertices`` (which it
reduces to, label for label, when every weight is 1):

- the *dominant* contribution of a vertex is the truth label (unlabelled
  tracks, label -1, form one group) with the largest summed weight; ties go to
  the label whose first track appears earliest in the vertex's track list;
- Clean: dominant is a truth vertex with >= 70% of the vertex's total weight;
  Merged: < 70%; Fake: dominant is the unlabelled group, or the vertex has no
  tracks, or all its weights are zero (no contribution at all);
- Split: among vertices with the same dominant truth vertex, all but the one
  with the largest sum(pT^2) (first in list order on ties). ``weighted_sumpt2``
  selects sum(w pT^2) (default) or the unweighted sum(pT^2).

A track may appear in several vertices (AMVF shares tracks between vertices
with different weights); each vertex is classified independently, so this
needs no special treatment.
"""

from __future__ import annotations

import numpy as np

from gnn.evaluation.hs_metrics import (
    CLEAN,
    FAKE,
    MERGED,
    SPLIT,
    VertexClassification,
    _first_per_group,
)
from pv_finder.utils.constants import PURITY_THRESHOLD

__all__ = ["CLEAN", "MERGED", "SPLIT", "FAKE", "classify_vertices_weighted"]


def classify_vertices_weighted(
    lists: list[np.ndarray],
    weights: list[np.ndarray],
    pt: np.ndarray,
    track_truth: np.ndarray,
    *,
    weighted_sumpt2: bool = True,
) -> VertexClassification:
    """Clean / Merged / Split / Fake per vertex from weighted track contributions.

    Args:
        lists: Track indices per vertex.
        weights: Non-negative weight of each listed track, same shapes as ``lists``.
        pt: Track transverse momenta (any unit; only ratios of sum pT^2 matter).
        track_truth: Truth index per track, -1 for tracks without truth.
        weighted_sumpt2: Rank vertices for Split by sum(w pT^2) if True,
            sum(pT^2) otherwise.

    Raises:
        ValueError: if a weight list has the wrong length or a negative or
            non-finite weight.

    ``sum_pt2`` of the returned classification is the ranking quantity used.
    """
    pt = np.asarray(pt, dtype=np.float64)
    track_truth = np.asarray(track_truth, dtype=np.int64)
    n = len(lists)
    if len(weights) != n:
        raise ValueError("one weight list per vertex is required")
    labels = np.full(n, FAKE, dtype=np.int64)
    dominant = np.full(n, -2, dtype=np.int64)
    ntrk = np.array([len(x) for x in lists], dtype=np.int64)
    if any(len(w) != k for w, k in zip(weights, ntrk)):
        raise ValueError("weight list length differs from its track list")
    if n == 0 or ntrk.sum() == 0:
        return VertexClassification(labels, dominant, np.zeros(n), ntrk)
    trk = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
    w = np.concatenate([np.asarray(x, dtype=np.float64) for x in weights])
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("weights must be finite and non-negative")
    vid = np.repeat(np.arange(n), ntrk)
    rank_w = w if weighted_sumpt2 else np.ones_like(w)
    sum_pt2 = np.bincount(vid, weights=rank_w * pt[trk] ** 2, minlength=n)
    total = np.bincount(vid, weights=w, minlength=n)
    key = np.where(track_truth[trk] < 0, -1, track_truth[trk]) + 1
    base = int(key.max()) + 1
    pair, first, inverse = np.unique(
        vid * base + key, return_index=True, return_inverse=True
    )
    wsum = np.bincount(inverse, weights=w, minlength=len(pair))
    pv, pkey = pair // base, pair % base - 1
    order = np.lexsort((first, -wsum, pv))
    winners = order[_first_per_group(pv, order)]
    # Vertices whose weights are all zero have no contribution: Fake, no dominant.
    winners = winners[total[pv[winners]] > 0]
    v_win = pv[winners]
    dominant[v_win] = pkey[winners]
    frac = wsum[winners] / total[v_win]
    truthful = pkey[winners] >= 0
    labels[v_win[truthful]] = np.where(
        frac[truthful] >= PURITY_THRESHOLD, CLEAN, MERGED
    )
    cand = np.flatnonzero(dominant >= 0)
    if len(cand):
        order = np.lexsort((cand, -sum_pt2[cand], dominant[cand]))
        keep = cand[order[_first_per_group(dominant[cand], order)]]
        labels[np.setdiff1d(cand, keep)] = SPLIT
    return VertexClassification(labels, dominant, sum_pt2, ntrk)
