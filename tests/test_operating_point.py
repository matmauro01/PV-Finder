"""Tests for pv_finder.evaluation.vertex_finding.operating_point."""

from __future__ import annotations

import numpy as np
import pytest

from pv_finder.evaluation.vertex_finding.operating_point import (
    bisect_to_target,
    filter_by_height,
)


def test_filter_by_height_keeps_peaks_at_or_above_cut() -> None:
    pvs = np.array([1.0, 2.0, 3.0, 4.0])
    hts = np.array([0.02, 0.03, 0.5, 0.04])
    np.testing.assert_array_equal(filter_by_height(pvs, hts, 0.03), [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(filter_by_height(pvs, hts, 0.0), pvs)
    assert filter_by_height(pvs, hts, 1.0).size == 0


def test_filter_by_height_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        filter_by_height(np.zeros(3), np.zeros(2), 0.1)


def test_bisect_finds_threshold_of_monotone_curve() -> None:
    # fp(t) = 30 - 50 t, decreasing; fp = 18.73 at t = 0.2254
    thr, fp = bisect_to_target(lambda t: 30 - 50 * t, 18.73, 0.05, 0.30, tol=1e-3)
    assert abs(fp - 18.73) <= 1e-3
    assert abs(thr - 0.2254) < 1e-4


def test_bisect_handles_step_curve() -> None:
    # Discrete fake counts: closest achievable value is returned.
    def fp_of(t: float) -> float:
        return float(np.floor(20 - 40 * t))

    thr, fp = bisect_to_target(fp_of, 15.5, 0.0, 0.5, tol=0.01)
    assert fp in (15.0, 16.0)
    assert 0.0 <= thr <= 0.5


def test_bisect_raises_when_not_bracketed() -> None:
    with pytest.raises(ValueError):
        bisect_to_target(lambda t: 10 - t, 50.0, 0.0, 1.0)
