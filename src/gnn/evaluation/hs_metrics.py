"""Hard-scatter (HS) and track-composition metrics shared by all TTVA arms.

Model-free and truth-only: every function takes a list of track indices per
reconstructed vertex (one list per vertex) and per-track truth labels, so
PV-Finder+GNN, the z0 baselines, the truth-guided reference and AMVF are all
judged by the same code.

Conventions (identical to ``gnn.evaluation.classification.classify_assignments``,
whose tie-breaking is reproduced exactly; checked in tests):

- ``track_truth[i]`` is the index of the stored truth vertex that owns track
  ``i`` (``TruthVertex_assocTracks``), or ``-1`` when the track has none.
- Track contributions are unweighted (AMVF fit weights are not stored).
- A vertex's *dominant* contribution is the largest group of its tracks
  sharing one truth label; tracks with label -1 form one extra group. Ties go
  to the group whose first track appears earliest in the vertex's track list.
- Clean: dominant is a truth vertex with >= 70% of the tracks. Merged: dominant
  is a truth vertex with < 70%. Fake: dominant is the unlabelled group (or the
  vertex has no tracks). Split: among vertices with the same dominant truth
  vertex, all but the largest sum(pT^2) one (first in list order on ties).

HS definitions (event level; ``hs`` is the truth index with
``TruthVertex_type == 1``):

- HS-matched vertex: the reconstructed vertex with the largest number of
  tracks whose truth label is ``hs`` (ITk paper Sec. 5.5 / Athena
  ``bestHardScatterMatch``); ties go to the vertex first in the sum(pT^2)-
  sorted container (strict ``>`` in Athena), see ``gnn.evaluation.pv_selection``.
  None if no vertex holds an HS track.
- Selected vertex: the largest sum(pT^2) vertex (first in list order on ties).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gnn.evaluation.pv_selection import best_hs_match, sorted_container_order
from pv_finder.utils.constants import PURITY_THRESHOLD
from pv_finder.utils.working_points import HS_MATCH_WINDOW_MM

CLEAN, MERGED, SPLIT, FAKE = 0, 1, 2, 3
CATEGORY_NAMES = ("Clean", "Merged", "Split", "Fake")


def track_truth_labels(
    assoc_flat: np.ndarray, truth_ntracks: np.ndarray, n_tracks: int
) -> np.ndarray:
    """Invert ``TruthVertex_assocTracks`` into one truth index per track.

    Args:
        assoc_flat: Concatenated track indices of all truth vertices.
        truth_ntracks: Number of tracks per truth vertex (``TruthVertex_nTracks``).
        n_tracks: Number of reconstructed tracks in the event.

    Raises:
        ValueError: if lengths disagree, an index is out of range, or a track
            is claimed by two truth vertices.
    """
    assoc_flat = np.asarray(assoc_flat, dtype=np.int64)
    counts = np.asarray(truth_ntracks, dtype=np.int64)
    if counts.sum() != len(assoc_flat):
        raise ValueError("TruthVertex_nTracks does not match the association length")
    if len(assoc_flat) and (assoc_flat.min() < 0 or assoc_flat.max() >= n_tracks):
        raise ValueError("Truth association index out of range")
    if len(np.unique(assoc_flat)) != len(assoc_flat):
        raise ValueError("A track is associated with two truth vertices")
    labels = np.full(n_tracks, -1, dtype=np.int64)
    labels[assoc_flat] = np.repeat(np.arange(len(counts)), counts)
    return labels


def lists_from_targets(targets: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Group tracks by target vertex; ``targets[i] = -1`` leaves track i unassigned.

    Each returned list is sorted ascending, as ``np.unique`` would give.
    """
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) and targets.max() >= n_vertices:
        raise ValueError("Target vertex index out of range")
    tracks = np.flatnonzero(targets >= 0)
    order = np.argsort(targets[tracks], kind="stable")
    tracks = tracks[order]
    bounds = np.searchsorted(targets[tracks], np.arange(n_vertices + 1))
    return [tracks[bounds[v] : bounds[v + 1]] for v in range(n_vertices)]


def assign_from_scores(
    targets: np.ndarray, scores: np.ndarray, threshold: float
) -> np.ndarray:
    """Top-1 GNN assignment: keep the best edge only if its score >= threshold."""
    targets = np.asarray(targets, dtype=np.int64)
    return np.where(np.asarray(scores) >= threshold, targets, -1)


def assign_nearest_z0(
    z0: np.ndarray,
    vertex_z: np.ndarray,
    z0_err: np.ndarray | None = None,
    n_sigma: float | None = None,
) -> np.ndarray:
    """Baseline: each track goes to the vertex nearest in |z0 - z_vertex|.

    With ``n_sigma`` (and ``z0_err``), the association is kept only if
    ``|z0 - z_vertex| < n_sigma * sigma(z0)``; otherwise the track is left
    unassigned. Ties in distance go to the lower vertex index.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    vertex_z = np.asarray(vertex_z, dtype=np.float64)
    if len(vertex_z) == 0:
        return np.full(len(z0), -1, dtype=np.int64)
    dist = np.abs(z0[:, None] - vertex_z[None, :])
    nearest = dist.argmin(axis=1).astype(np.int64)
    if n_sigma is None:
        return nearest
    if z0_err is None:
        raise ValueError("n_sigma requires z0_err")
    best = dist[np.arange(len(z0)), nearest]
    keep = best < n_sigma * np.asarray(z0_err, dtype=np.float64)
    return np.where(keep, nearest, -1)


def assign_truth_guided(
    track_truth: np.ndarray,
    truth_z: np.ndarray,
    vertex_z: np.ndarray,
    window: float,
) -> np.ndarray:
    """Reference (MC only): each labelled track goes to the vertex nearest its
    own truth vertex, if that vertex is within ``window`` mm. Unlabelled
    tracks and tracks without a nearby vertex stay unassigned."""
    track_truth = np.asarray(track_truth, dtype=np.int64)
    out = np.full(len(track_truth), -1, dtype=np.int64)
    vertex_z = np.asarray(vertex_z, dtype=np.float64)
    if len(vertex_z) == 0:
        return out
    labelled = np.flatnonzero(track_truth >= 0)
    tz = np.asarray(truth_z, dtype=np.float64)[track_truth[labelled]]
    dist = np.abs(tz[:, None] - vertex_z[None, :])
    nearest = dist.argmin(axis=1)
    ok = dist[np.arange(len(labelled)), nearest] < window
    out[labelled[ok]] = nearest[ok]
    return out


def _first_per_group(groups: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Positions (into ``order``) of the first row of each group, groups sorted."""
    g = groups[order]
    return np.flatnonzero(np.r_[True, g[1:] != g[:-1]])


@dataclass
class VertexClassification:
    """Per-vertex classification of one event's vertex list."""

    labels: np.ndarray  # CLEAN / MERGED / SPLIT / FAKE per vertex
    dominant: np.ndarray  # dominant truth index; -1 unlabelled; -2 empty
    sum_pt2: np.ndarray
    n_tracks: np.ndarray

    def counts(self) -> np.ndarray:
        """Number of Clean, Merged, Split, Fake vertices."""
        return np.bincount(self.labels, minlength=4)[:4].astype(np.int64)


def classify_vertices(
    lists: list[np.ndarray], pt: np.ndarray, track_truth: np.ndarray
) -> VertexClassification:
    """Clean / Merged / Split / Fake for every vertex (see module docstring).

    Vectorized; the dominant group of a vertex is the (vertex, label) pair
    with the largest count, ties broken by the earliest position of that
    label in the vertex's track list.
    """
    pt = np.asarray(pt, dtype=np.float64)
    track_truth = np.asarray(track_truth, dtype=np.int64)
    n = len(lists)
    labels = np.full(n, FAKE, dtype=np.int64)
    dominant = np.full(n, -2, dtype=np.int64)
    ntrk = np.array([len(x) for x in lists], dtype=np.int64)
    if n == 0 or ntrk.sum() == 0:
        return VertexClassification(labels, dominant, np.zeros(n), ntrk)
    trk = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
    vid = np.repeat(np.arange(n), ntrk)
    sum_pt2 = np.bincount(vid, weights=pt[trk] ** 2, minlength=n)
    key = np.where(track_truth[trk] < 0, -1, track_truth[trk]) + 1
    base = int(key.max()) + 1
    pair, first, count = np.unique(
        vid * base + key, return_index=True, return_counts=True
    )
    pv, pkey = pair // base, pair % base - 1
    order = np.lexsort((first, -count, pv))
    winners = order[_first_per_group(pv, order)]
    v_win = pv[winners]
    dominant[v_win] = pkey[winners]
    frac = count[winners] / ntrk[v_win]
    truthful = pkey[winners] >= 0
    labels[v_win[truthful]] = np.where(
        frac[truthful] >= PURITY_THRESHOLD, CLEAN, MERGED
    )
    # Split: keep the largest sum(pT^2) per dominant truth vertex (list order on ties)
    cand = np.flatnonzero(dominant >= 0)
    if len(cand):
        order = np.lexsort((cand, -sum_pt2[cand], dominant[cand]))
        keep = cand[order[_first_per_group(dominant[cand], order)]]
        split = np.setdiff1d(cand, keep)
        labels[split] = SPLIT
    return VertexClassification(labels, dominant, sum_pt2, ntrk)


# Fields of the per-event HS record, in storage order.
HS_FIELDS = (
    "n_hs_tracks",  # truth-labelled HS tracks in the event (denominator of c)
    "has_match",  # some vertex holds >= 1 HS track
    "match_hs_tracks",  # HS tracks on the HS-matched vertex (numerator of c)
    "match_tracks",  # all tracks on the HS-matched vertex (for purity)
    "match_label",  # category of the HS-matched vertex, -1 if none
    "clean_hs",  # a Clean vertex has the HS as dominant contribution
    "hs_dominated",  # a non-Split vertex has the HS as dominant (Clean|Merged)
    "selected_is_match",  # (d) ITk-style: selected vertex == HS-matched vertex
    "selected_dominant_hs",  # (d) legacy: selected vertex's dominant truth is HS
    "selected_within",  # |z_selected - z_HS| < window
    "match_within",  # |z_HS-matched - z_HS| < window
    "any_within",  # (a) any vertex in the list within window of z_HS
)


def hs_track_counts(
    lists: list[np.ndarray],
    classification: VertexClassification,
    track_truth: np.ndarray,
    hs_index: int,
) -> np.ndarray:
    """Number of HS-labelled tracks on each vertex."""
    if len(lists) == 0 or classification.n_tracks.sum() == 0:
        return np.zeros(len(lists), dtype=np.int64)
    trk = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
    vid = np.repeat(np.arange(len(lists)), classification.n_tracks)
    return np.bincount(vid[track_truth[trk] == hs_index], minlength=len(lists))


def hs_matched_index(
    nhs: np.ndarray,
    classification: VertexClassification,
    order: np.ndarray | None = None,
) -> int:
    """Vertex with the most HS tracks; -1 if none (Athena ``bestHardScatterMatch``).

    Ties go to the vertex first in the sum(pT^2)-sorted container. ``order``
    is that container position. By default it comes from this classification's
    sum(pT^2), descending, stable on ties. Pass the stored order for an Athena
    container (AMVF), whose sorting used Athena's own track links.
    """
    if order is None:
        order = sorted_container_order(classification.sum_pt2)
    return best_hs_match(nhs, order)


def selected_index(classification: VertexClassification) -> int:
    """Largest sum(pT^2) populated vertex (first on ties); -1 if none."""
    populated = classification.n_tracks > 0
    if not populated.any():
        return -1
    return int(np.argmax(np.where(populated, classification.sum_pt2, -1.0)))


def hs_event_record(
    lists: list[np.ndarray],
    classification: VertexClassification,
    track_truth: np.ndarray,
    hs_index: int,
    vertex_z: np.ndarray,
    hs_z: float,
    window: float = HS_MATCH_WINDOW_MM,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """All HS quantities for one event and one vertex list (see HS_FIELDS).

    ``order``: container position used for HS-match ties (see hs_matched_index).
    """
    vertex_z = np.asarray(vertex_z, dtype=np.float64)
    n_hs = int((track_truth == hs_index).sum())
    nhs = hs_track_counts(lists, classification, track_truth, hs_index)
    match = hs_matched_index(nhs, classification, order)
    has_match = match >= 0
    selected = selected_index(classification)
    dom_hs = classification.dominant == hs_index
    labels = classification.labels
    near = np.abs(vertex_z - hs_z) < window if len(vertex_z) else np.zeros(0, bool)
    return np.array(
        [
            n_hs,
            int(has_match),
            nhs[match] if match >= 0 else 0,
            classification.n_tracks[match] if match >= 0 else 0,
            labels[match] if match >= 0 else -1,
            int(np.any(dom_hs & (labels == CLEAN))),
            int(np.any(dom_hs & (labels != SPLIT))),
            int(match >= 0 and selected == match),
            int(selected >= 0 and classification.dominant[selected] == hs_index),
            int(selected >= 0 and bool(near[selected])),
            int(match >= 0 and bool(near[match])),
            int(near.any()),
        ],
        dtype=np.int64,
    )


def greedy_one_to_one(reco_z: np.ndarray, truth_z: np.ndarray, window: float) -> int:
    """Number of truth vertices with a dedicated reco vertex, |dz| < window.

    Closest pairs first; each reco and truth vertex used at most once. Same
    algorithm as ``chain_gap_decomposition.greedy_match``.
    """
    reco_z = np.asarray(reco_z, dtype=np.float64)
    truth_z = np.asarray(truth_z, dtype=np.float64)
    if len(reco_z) == 0 or len(truth_z) == 0:
        return 0
    dz = np.abs(reco_z[:, None] - truth_z[None, :])
    ri, ti = np.nonzero(dz < window)
    order = np.argsort(dz[ri, ti], kind="stable")
    used_r = np.zeros(len(reco_z), bool)
    used_t = np.zeros(len(truth_z), bool)
    n = 0
    for k in order:
        r, t = ri[k], ti[k]
        if not used_r[r] and not used_t[t]:
            used_r[r] = used_t[t] = True
            n += 1
    return n
