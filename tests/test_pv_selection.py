"""Tests for the Athena primary-vertex selection rule (gnn.evaluation.pv_selection)."""

from __future__ import annotations

import numpy as np

from gnn.evaluation import hs_metrics as hm
from gnn.evaluation.pv_selection import (
    MAX_VERTEX_CHI2,
    best_hs_match,
    first_argmax,
    select_primary,
    sorted_container_order,
    sumpt2_score,
    track_vertex_chi2,
)


def test_select_primary_largest_score_first_on_ties() -> None:
    assert select_primary(np.array([1.0, 5.0, 3.0])) == 1
    assert select_primary(np.array([5.0, 1.0, 5.0])) == 0
    # container order decides ties, not the index
    assert select_primary(np.array([5.0, 1.0, 5.0]), order=np.array([2, 0, 1])) == 2
    assert select_primary(np.array([])) == -1


def test_best_hs_match_strict_greater_like_athena() -> None:
    nhs = np.array([27, 27, 12])
    # Athena: first in the sorted container wins the tie
    assert best_hs_match(nhs, order=np.array([0, 1, 2])) == 0
    assert best_hs_match(nhs, order=np.array([1, 0, 2])) == 1
    assert best_hs_match(np.array([0, 0])) == -1


def test_sorted_container_order_descending_stable() -> None:
    order = sorted_container_order(np.array([3.0, 9.0, 3.0, 1.0]))
    np.testing.assert_array_equal(order, [1, 0, 2, 3])
    assert first_argmax(np.array([2, 2, 2]), order=np.array([2, 1, 0])) == 2


def test_sumpt2_score_with_compatibility() -> None:
    pt = np.array([1.0, 2.0, 100.0])
    assert sumpt2_score(pt) == 10005.0
    assert sumpt2_score(pt, np.array([True, True, False])) == 5.0


def test_track_vertex_chi2_zero_for_track_through_vertex() -> None:
    rng = np.random.default_rng(1)
    phi = rng.uniform(-np.pi, np.pi, 50)
    theta = rng.uniform(0.1, 3.0, 50)
    vx, vy, vz = 0.012, -0.008, 3.4
    d0 = -vx * np.sin(phi) + vy * np.cos(phi)
    z0 = vz - (vx * np.cos(phi) + vy * np.sin(phi)) / np.tan(theta)
    var = np.full(50, 1e-4)
    chi2 = track_vertex_chi2(d0, z0, phi, theta, var, var, np.zeros(50), (vx, vy, vz))
    np.testing.assert_allclose(chi2, 0.0, atol=1e-12)
    # a 1 mm z offset with a 0.01 mm error is far beyond the AMVF cut
    far = track_vertex_chi2(
        d0, z0 + 1.0, phi, theta, var, var, np.zeros(50), (vx, vy, vz)
    )
    assert np.all(far > MAX_VERTEX_CHI2)


def test_track_vertex_chi2_uses_correlation() -> None:
    v = track_vertex_chi2(
        np.array([1.0]), np.array([1.0]), np.zeros(1), np.full(1, np.pi / 2),
        np.ones(1), np.ones(1), np.array([0.5]), (0.0, 0.0, 0.0),
    )  # fmt: skip
    # vertex at the origin predicts (d0, z0) = (0, 0), so the residual is (1, 1);
    # chi2 = r^T V^-1 r with V = [[1, .5], [.5, 1]] gives 2 / 1.5
    np.testing.assert_allclose(v, [2.0 / 1.5])


def test_hs_matched_index_stored_order_for_athena_container() -> None:
    # Event-9560-like tie: two vertices with the same HS-track count; the
    # recomputed sum(pT^2) favours vertex 1, the stored container puts 0 first.
    lists = [np.array([0, 1]), np.array([2, 3])]
    pt = np.array([1.0, 1.0, 5.0, 5.0])
    truth = np.zeros(4, dtype=np.int64)
    cls = hm.classify_vertices(lists, pt, truth)
    nhs = hm.hs_track_counts(lists, cls, truth, 0)
    assert hm.hs_matched_index(nhs, cls) == 1  # default: our sum(pT^2) order
    assert hm.hs_matched_index(nhs, cls, np.arange(2)) == 0  # stored order
