"""Tests for gnn.evaluation.weighted_classification.classify_vertices_weighted."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from gnn.evaluation import hs_metrics as hm  # noqa: E402
from gnn.evaluation.weighted_classification import (  # noqa: E402
    classify_vertices_weighted,
)

C, M, S, F = hm.CLEAN, hm.MERGED, hm.SPLIT, hm.FAKE


def _unit(lists: list[np.ndarray]) -> list[np.ndarray]:
    return [np.ones(len(x)) for x in lists]


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("shared", [False, True])
def test_unit_weights_reproduce_unweighted_rule(seed: int, shared: bool) -> None:
    """Unit weights give hs_metrics.classify_vertices label for label."""
    rng = np.random.default_rng(seed)
    n_tracks, n_truth, n_vtx = 50, 5, 8
    truth = rng.integers(-1, n_truth, n_tracks)
    pt = rng.integers(1, 4, n_tracks).astype(float)  # sum(pT^2) ties likely
    if shared:  # unsorted lists, a track may sit in several vertices
        lists = [
            rng.choice(n_tracks, rng.integers(0, 12), replace=False)
            for _ in range(n_vtx)
        ]
    else:
        lists = hm.lists_from_targets(rng.integers(-1, n_vtx, n_tracks), n_vtx)
    ref = hm.classify_vertices(lists, pt, truth)
    got = classify_vertices_weighted(lists, _unit(lists), pt, truth)
    np.testing.assert_array_equal(got.labels, ref.labels)
    np.testing.assert_array_equal(got.dominant, ref.dominant)
    np.testing.assert_array_equal(got.sum_pt2, ref.sum_pt2)


def test_weights_change_clean_and_merged() -> None:
    truth = np.array([0, 0, 1, 1, 1])
    pt = np.ones(5)
    lists = [np.array([0, 1, 2, 3, 4])]
    # unweighted: label 1 has 3/5 = 60% -> Merged
    assert hm.classify_vertices(lists, pt, truth).labels.tolist() == [M]
    # weighted: label 1 has 2.7 / (0.2 + 2.7) = 93% -> Clean, dominant 1
    w = [np.array([0.1, 0.1, 0.9, 0.9, 0.9])]
    got = classify_vertices_weighted(lists, w, pt, truth)
    assert got.labels.tolist() == [C]
    assert got.dominant.tolist() == [1]
    # weighted: label 0 now dominates with 1.8 / 2.1 = 86% -> Clean, dominant 0
    w = [np.array([0.9, 0.9, 0.1, 0.1, 0.1])]
    got = classify_vertices_weighted(lists, w, pt, truth)
    assert got.labels.tolist() == [C]
    assert got.dominant.tolist() == [0]


def test_fake_by_weight_and_zero_weight_vertex() -> None:
    truth = np.array([-1, -1, 0])
    pt = np.ones(3)
    lists = [np.array([0, 1, 2]), np.array([2])]
    w = [np.array([0.6, 0.6, 1.0]), np.array([0.0])]
    got = classify_vertices_weighted(lists, w, pt, truth)
    assert got.labels.tolist() == [F, F]  # unlabelled 1.2 > 1.0; zero total weight
    assert got.dominant.tolist() == [-1, -2]


def test_split_ranking_weighted_or_not() -> None:
    truth = np.array([0, 0, 0, 0])
    pt = np.array([10.0, 1.0, 1.0, 1.0])
    lists = [np.array([0]), np.array([1, 2, 3])]
    w = [np.array([0.01]), np.array([1.0, 1.0, 1.0])]
    # sum(w pT^2): 1.0 vs 3.0 -> vertex 0 is Split
    got = classify_vertices_weighted(lists, w, pt, truth)
    assert got.labels.tolist() == [S, C]
    # sum(pT^2): 100 vs 3 -> vertex 1 is Split
    got = classify_vertices_weighted(lists, w, pt, truth, weighted_sumpt2=False)
    assert got.labels.tolist() == [C, S]


def test_empty_and_invalid_inputs() -> None:
    got = classify_vertices_weighted([], [], np.ones(0), np.zeros(0, int))
    assert len(got.labels) == 0
    got = classify_vertices_weighted([np.array([], int)], [np.array([])], np.ones(1), np.zeros(1, int))  # fmt: skip
    assert got.labels.tolist() == [F]
    with pytest.raises(ValueError):
        classify_vertices_weighted([np.array([0])], [np.array([-0.1])], np.ones(1), np.zeros(1, int))  # fmt: skip
    with pytest.raises(ValueError):
        classify_vertices_weighted([np.array([0])], [np.array([1.0, 1.0])], np.ones(1), np.zeros(1, int))  # fmt: skip
