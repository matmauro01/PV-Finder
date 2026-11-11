"""Tests for the shared HS / track-composition metrics (gnn.evaluation.hs_metrics)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from gnn.evaluation import hs_metrics as hm  # noqa: E402
from gnn.evaluation.classification import classify_assignments  # noqa: E402

NAMES = {"Clean": hm.CLEAN, "Merged": hm.MERGED, "Split": hm.SPLIT, "Fake": hm.FAKE}


def _reference(lists, pt, truth):
    tti = np.flatnonzero(truth >= 0).astype(float)
    tpi = truth[truth >= 0].astype(float)
    rows, info = classify_assignments(lists, pt, tti, tpi, 0)
    return np.array(rows[:4]), np.array([NAMES[r["classification"]] for r in info])


@pytest.mark.parametrize("seed", range(40))
def test_classification_matches_reference_on_random_events(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_tracks, n_truth, n_vtx = 60, 6, 9
    truth = rng.integers(-1, n_truth, n_tracks)  # includes unlabelled tracks
    # Coarse pT values make sum(pT^2) ties (Split tie-break) likely
    pt = rng.integers(1, 4, n_tracks).astype(float)
    targets = rng.integers(-1, n_vtx, n_tracks)
    lists = hm.lists_from_targets(targets, n_vtx)  # some vertices may be empty
    ref_counts, ref_labels = _reference(lists, pt, truth)
    got = hm.classify_vertices(lists, pt, truth)
    np.testing.assert_array_equal(got.labels, ref_labels)
    np.testing.assert_array_equal(got.counts(), ref_counts)


def test_dominant_tie_goes_to_first_listed_group() -> None:
    truth = np.array([-1, 0, 0, -1])
    lists = [np.array([0, 1, 2, 3])]  # unlabelled first, 2 vs 2
    got = hm.classify_vertices(lists, np.ones(4), truth)
    assert got.dominant[0] == -1 and got.labels[0] == hm.FAKE
    got = hm.classify_vertices([np.array([1, 0, 2, 3])], np.ones(4), truth)
    assert got.dominant[0] == 0 and got.labels[0] == hm.MERGED


def test_purity_boundary_is_inclusive() -> None:
    truth = np.array([0] * 7 + [1] * 3)
    got = hm.classify_vertices([np.arange(10)], np.ones(10), truth)
    assert got.labels[0] == hm.CLEAN  # exactly 70%


def test_truth_labels_and_duplicate_rejection() -> None:
    labels = hm.track_truth_labels(np.array([2, 0, 3]), np.array([2, 0, 1]), 5)
    np.testing.assert_array_equal(labels, [0, -1, 0, 2, -1])
    with pytest.raises(ValueError):
        hm.track_truth_labels(np.array([1, 1]), np.array([1, 1]), 3)
    with pytest.raises(ValueError):
        hm.track_truth_labels(np.array([5]), np.array([1]), 3)


def test_nearest_z0_baseline_and_sigma_cut() -> None:
    z0 = np.array([0.0, 0.9, 5.0])
    err = np.array([0.1, 0.5, 0.1])
    vz = np.array([0.05, 1.0])
    np.testing.assert_array_equal(hm.assign_nearest_z0(z0, vz), [0, 1, 1])
    np.testing.assert_array_equal(hm.assign_nearest_z0(z0, vz, err, 3.0), [0, 1, -1])
    assert np.all(hm.assign_nearest_z0(z0, np.array([])) == -1)


def test_truth_guided_window_and_unlabelled() -> None:
    truth = np.array([0, 1, -1])
    out = hm.assign_truth_guided(
        truth, np.array([0.0, 3.0]), np.array([0.2, 10.0]), 0.5
    )
    np.testing.assert_array_equal(out, [0, -1, -1])


def test_hs_record_definitions() -> None:
    # Tracks: 0-4 HS (truth 0), 5-8 pile-up (truth 1), 9 unlabelled
    truth = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, -1])
    pt = np.array([1, 1, 1, 1, 1, 10, 1, 1, 1, 1], dtype=float)
    lists = [np.array([0, 1, 2]), np.array([3, 4, 5, 6]), np.array([7, 8, 9])]
    cls = hm.classify_vertices(lists, pt, truth)
    rec = dict(
        zip(
            hm.HS_FIELDS,
            hm.hs_event_record(lists, cls, truth, 0, np.array([0.0, 0.05, 2.0]), 0.0),
        )
    )
    # v0: 3 HS tracks, pure. v1: 2 HS + 2 PU (tie -> HS, first listed), Merged,
    # and the largest sum(pT^2), so v1 keeps the HS and v0 becomes Split.
    np.testing.assert_array_equal(cls.labels, [hm.SPLIT, hm.MERGED, hm.MERGED])
    assert rec["n_hs_tracks"] == 5
    assert rec["match_hs_tracks"] == 3 and rec["match_tracks"] == 3  # v0
    assert rec["match_label"] == hm.SPLIT
    assert rec["clean_hs"] == 0 and rec["hs_dominated"] == 1
    # The two selection rules disagree here: legacy accepts v1 (dominant HS),
    # the ITk rule rejects it (the most-HS-track vertex is v0).
    assert rec["selected_dominant_hs"] == 1
    assert rec["selected_is_match"] == 0
    assert rec["selected_within"] == 1 and rec["match_within"] == 1
    assert rec["any_within"] == 1


def test_hs_record_match_tie_uses_sum_pt2_and_empty_event() -> None:
    truth = np.array([0, 0, 1])
    pt = np.array([1.0, 3.0, 1.0])
    lists = [np.array([0]), np.array([1, 2])]
    cls = hm.classify_vertices(lists, pt, truth)
    rec = dict(
        zip(hm.HS_FIELDS, hm.hs_event_record(lists, cls, truth, 0, np.zeros(2), 9.0))
    )
    assert rec["match_tracks"] == 2  # tie 1 vs 1 HS track -> larger sum(pT^2)
    assert rec["selected_is_match"] == 1 and rec["any_within"] == 0
    empty = hm.classify_vertices([], pt, truth)
    rec = dict(
        zip(hm.HS_FIELDS, hm.hs_event_record([], empty, truth, 0, np.zeros(0), 0.0))
    )
    assert (
        rec["has_match"] == 0
        and rec["match_label"] == -1
        and rec["selected_is_match"] == 0
    )


def test_greedy_one_to_one() -> None:
    # One reco between two truths: only one truth can be matched
    assert hm.greedy_one_to_one(np.array([0.0]), np.array([-0.1, 0.1]), 0.5) == 1
    assert (
        hm.greedy_one_to_one(np.array([0.0, 0.2]), np.array([0.01, 0.19]), 0.1694) == 2
    )
    assert (
        hm.greedy_one_to_one(np.array([0.0]), np.array([0.1694]), 0.1694) == 0
    )  # strict


def test_clopper_pearson_and_bootstrap() -> None:
    from gnn.diagnostics.hs_chain_summary import bootstrap_ratio, clopper_pearson

    full = clopper_pearson(10, 10)
    assert full["value"] == 1.0 and full["high"] == 1.0 and full["low"] < 1.0
    none = clopper_pearson(0, 10)
    assert none["low"] == 0.0 and none["high"] > 0.0
    assert clopper_pearson(0, 0)["value"] is None
    rng = np.random.default_rng(0)
    num = np.array([1.0, 2.0, 3.0])
    res = bootstrap_ratio(num, np.array([2.0, 4.0, 6.0]), rng, n_boot=200)
    assert res["value"] == pytest.approx(0.5) and res["error"] == pytest.approx(0.0)
    two = bootstrap_ratio(np.c_[num, num], np.c_[num * 2, num * 4], rng, n_boot=200)
    assert two["diff_vs_first"][0] == pytest.approx(0.25)
