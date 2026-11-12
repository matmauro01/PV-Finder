"""Tests for the freeze-comparison helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pv_finder.evaluation.vertex_finding.freeze_compare import (
    check_bookkeeping,
    clopper_pearson,
    hs_found,
    paired_ratio_diff,
    truth_found_any,
)


def test_truth_found_any_counts_within_inclusive_window() -> None:
    truth = np.array([0.0, 1.0, 2.0, 5.0])
    pred = np.array([0.17, 2.1, 9.0])
    # 0.0 (|dz| = 0.17, inclusive), 2.0 (0.1) found; 1.0 and 5.0 not.
    assert truth_found_any(truth, pred, 0.17) == 2
    assert truth_found_any(truth, pred, 0.16) == 1


def test_truth_found_any_empty_and_unsorted() -> None:
    assert truth_found_any(np.array([1.0]), np.array([]), 0.17) == 0
    assert truth_found_any(np.array([]), np.array([1.0]), 0.17) == 0
    assert (
        truth_found_any(np.array([3.0, -1.0]), np.array([3.05, -1.1, 50.0]), 0.17) == 2
    )


def test_truth_found_any_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        t = rng.uniform(-10, 10, rng.integers(0, 30))
        p = rng.uniform(-10, 10, rng.integers(0, 30))
        brute = sum(bool(len(p)) and np.min(np.abs(p - x)) <= 0.17 for x in t)
        assert truth_found_any(t, p, 0.17) == brute


def test_hs_found_is_strict() -> None:
    assert hs_found(np.array([1.05]), 1.0, 0.1)
    assert not hs_found(np.array([1.2]), 1.0, 0.1)
    assert not hs_found(np.array([]), 1.0, 0.1)


def test_check_bookkeeping() -> None:
    rec = {"clean": 3, "merged": 1, "split": 1, "fake": 2, "tc": 3, "tm": 2, "tmiss": 4}
    check_bookkeeping(rec, n_pred=7, n_truth=9)
    with pytest.raises(AssertionError):
        check_bookkeeping(rec, n_pred=6, n_truth=9)
    with pytest.raises(AssertionError):
        check_bookkeeping(rec, n_pred=7, n_truth=8)


def test_paired_ratio_diff_identical_arms_has_zero_error() -> None:
    rng = np.random.default_rng(1)
    num = rng.integers(0, 10, 100).astype(float)
    den = num + rng.integers(1, 5, 100)
    boots = rng.integers(0, 100, size=(200, 100))
    d, e = paired_ratio_diff(num, den, num, den, boots)
    assert d == 0.0 and e == 0.0


def test_clopper_pearson_contains_point_estimate() -> None:
    lo, hi = clopper_pearson(1918, 1920)
    assert lo < 1918 / 1920 < hi
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(10, 10)[1] == 1.0
