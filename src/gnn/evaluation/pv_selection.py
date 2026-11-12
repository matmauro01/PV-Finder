"""Primary-vertex (hard-scatter candidate) selection, as in ATLAS release 24.0.

One rule, applied identically to AMVF and PV-Finder+GNN vertices.

Athena 24.0, default ITk/Run 3 primary vertexing (``VertexSetup.ActsGaussAMVF``,
``Tracking/TrkConfig/python/VertexFindingFlags.py:45-59``), then
``InDetPriVxFinder`` with ``doVertexSorting=True``
(``InnerDetector/InDetConfig/python/InDetPriVxFinderConfig.py:10-22``) and the
default ``SumPt2Sorting`` (``VertexFindingFlags.py:64``):

- ``VertexCollectionSortingTool::sortVertexContainer``
  (``Tracking/TrkVertexFitter/TrkVertexTools/src/VertexCollectionSortingTool.cxx:85-114``)
  scores every non-dummy vertex, sorts by descending score (``Vertex_pair``
  ``operator<`` is ``first > other.first``, line 19) and labels the first
  vertex ``PriVtx``, the rest ``PileUp``.
- The score is ``SumPtVertexWeightCalculator::estimateSignalCompatibility``
  with ``DoSumPt2Selection=True``
  (``Tracking/TrkVertexFitter/TrkVertexWeightCalculators/src/SumPtVertexWeightCalculator.cxx:36-64``):
  the sum over **all** ``trackParticleLinks`` of the vertex of
  ``(pT / GeV)^2``, with ``pT = |1/(q/p)| sin(theta)`` from each track's own
  perigee. There are no track weights, no pT cap and no quality cut.
- Which tracks are linked is set by the finder
  (``Tracking/Acts/ActsVertexReconstruction/src/AdaptiveMultiPriVtxFinderTool.cxx:326-348``).
  With the default ``useFastCompatibility=true``, every track whose
  ``vertexCompatibility`` with the fitted vertex is at most ``maxVertexChi2``
  (18.42) is linked. A track may be linked to several vertices.

So the rule is: score(v) = sum of pT^2 over the tracks assigned to v and
compatible with it, and the selected vertex is the argmax. Ties go to the
earlier vertex in container order, which for Athena's own containers is the
stored order.

The hard-scatter match follows ``InDetVertexTruthMatchUtils::bestHardScatterMatch``
(``InnerDetector/InDetValidation/InDetTruthVertexValidation/Root/InDetVertexTruthMatchUtils.cxx:20-38``):
the vertex with the most HS tracks, with a strict ``>``, so ties go to the
first vertex in the (sorted) container.
"""

from __future__ import annotations

import numpy as np

MAX_VERTEX_CHI2 = 18.42  # AdaptiveMultiPriVtxFinderTool.h:136 (maxVertexChi2), 24.0


def track_vertex_chi2(
    d0: np.ndarray,
    z0: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    var_d0: np.ndarray,
    var_z0: np.ndarray,
    cov_d0z0: np.ndarray,
    vertex: tuple[float, float, float],
    var_vz: float = 0.0,
) -> np.ndarray:
    """Chi2 of each track's perigee (d0, z0) relative to a vertex [mm].

    Straight-line prediction for a track from vertex (x, y, z), in the sign
    convention measured on the ITk ntuple (see
    ``pv_finder.evaluation.vertex_finding.hs_position_estimators``):
    ``d0 = -x sin(phi) + y cos(phi)``, ``z0 = z - (x cos(phi) + y sin(phi)) cot(theta)``.
    ``var_vz`` adds the vertex z variance to the z0 variance.
    """
    vx, vy, vz = vertex
    phi = np.asarray(phi, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    r_d = np.asarray(d0, dtype=np.float64) - (-vx * np.sin(phi) + vy * np.cos(phi))
    r_z = np.asarray(z0, dtype=np.float64) - (
        vz - (vx * np.cos(phi) + vy * np.sin(phi)) / np.tan(theta)
    )
    v_dd = np.asarray(var_d0, dtype=np.float64)
    v_zz = np.asarray(var_z0, dtype=np.float64) + var_vz
    c = np.asarray(cov_d0z0, dtype=np.float64)
    det = np.clip(v_dd * v_zz - c**2, 1e-30, None)
    return (r_d**2 * v_zz - 2.0 * r_d * r_z * c + r_z**2 * v_dd) / det


def sumpt2_score(pt_gev: np.ndarray, compatible: np.ndarray | None = None) -> float:
    """Sum of pT^2 [GeV^2] over a vertex's tracks (optionally only compatible ones)."""
    pt = np.asarray(pt_gev, dtype=np.float64)
    if compatible is not None:
        pt = pt[np.asarray(compatible, dtype=bool)]
    return float(np.sum(pt**2))


def first_argmax(values: np.ndarray, order: np.ndarray | None = None) -> int:
    """Index of the largest value; ties go to the earliest entry in ``order``.

    ``order`` is the container position of each entry (default: its index).
    Returns -1 for an empty input.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return -1
    pos = np.arange(v.size) if order is None else np.asarray(order)
    tied = np.flatnonzero(v == v.max())
    return int(tied[np.argmin(pos[tied])])


def select_primary(scores: np.ndarray, order: np.ndarray | None = None) -> int:
    """Selected (PriVtx) vertex: the largest score, first in container order on ties."""
    return first_argmax(scores, order)


def best_hs_match(n_hs_tracks: np.ndarray, order: np.ndarray | None = None) -> int:
    """``bestHardScatterMatch``: most HS tracks (strict >), first in container order.

    ``order`` must be the position in the sum(pT^2)-sorted container. Returns
    -1 if no vertex holds an HS track.
    """
    n = np.asarray(n_hs_tracks)
    if n.size == 0 or n.max() <= 0:
        return -1
    return first_argmax(n, order)


def sorted_container_order(scores: np.ndarray) -> np.ndarray:
    """Container position after Athena's descending-score sort (stable on ties)."""
    s = np.asarray(scores, dtype=np.float64)
    rank = np.empty(s.size, dtype=np.int64)
    rank[np.argsort(-s, kind="stable")] = np.arange(s.size)
    return rank
