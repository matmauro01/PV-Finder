"""Tests for pv_finder.evaluation.vertex_finding.hs_position_estimators."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from pv_finder.evaluation.vertex_finding.hs_position_estimators import (
    BIN_WIDTH,
    parabolic_peak,
    robust_weighted_z0_mean,
    weighted_z0_mean,
)
from pv_finder.utils.constants import Z_MIN


def _binned_gaussian(mu: float, sigma: float, n: int = 12000) -> np.ndarray:
    centres = Z_MIN + np.arange(n) * BIN_WIDTH
    lo = (centres - BIN_WIDTH / 2 - mu) / sigma
    hi = (centres + BIN_WIDTH / 2 - mu) / sigma
    return ndtr(hi) - ndtr(lo)


def test_log_parabola_recovers_wide_gaussian() -> None:
    rng = np.random.default_rng(1)
    for mu in rng.uniform(-1.0, 1.0, 20):
        h = _binned_gaussian(mu, 0.08)
        z = parabolic_peak(h, int(np.argmax(h)), log=True)
        assert abs(z - mu) < 2e-4  # bin integration leaves a tiny bias


def test_parabola_symmetric_peak_is_bin_centre() -> None:
    h = np.zeros(100)
    h[49:52] = [1.0, 2.0, 1.0]
    assert parabolic_peak(h, 50) == Z_MIN + 50 * BIN_WIDTH
    assert parabolic_peak(h, 50, log=True) == Z_MIN + 50 * BIN_WIDTH


def test_parabola_falls_back_on_bad_input() -> None:
    h = np.zeros(10)
    h[5] = 1.0  # neighbours zero: log not defined
    assert parabolic_peak(h, 5, log=True) == Z_MIN + 5 * BIN_WIDTH
    assert parabolic_peak(h, 0) == Z_MIN  # edge
    flat = np.ones(10)
    assert parabolic_peak(flat, 5) == Z_MIN + 5 * BIN_WIDTH  # not concave


def test_weighted_mean() -> None:
    assert weighted_z0_mean(np.array([0.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    z = weighted_z0_mean(np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    assert np.isclose(z, 0.2)
    assert np.isnan(weighted_z0_mean(np.array([]), np.array([])))
    assert np.isnan(weighted_z0_mean(np.array([1.0]), np.array([0.0])))


def test_robust_mean_drops_outlier() -> None:
    z0 = np.array([0.0, 0.01, -0.01, 0.005, -0.005, 0.0, 0.01, -0.01, 0.0, 0.0, 0.5])
    err = np.full(len(z0), 0.02)
    assert weighted_z0_mean(z0, err) > 0.04
    assert abs(robust_weighted_z0_mean(z0, err)) < 1e-9
    # all compatible: identical to the plain mean
    z1, e1 = z0[:-1], err[:-1]
    assert robust_weighted_z0_mean(z1, e1) == weighted_z0_mean(z1, e1)


def test_seeded_robust_mean_ignores_dominant_outliers() -> None:
    z0 = np.array([0.0, 0.002, -0.002, 1.0, 1.0, 1.0, 1.0])
    err = np.full(len(z0), 0.01)
    assert robust_weighted_z0_mean(z0, err) > 0.5  # unseeded: pulled away
    assert abs(robust_weighted_z0_mean(z0, err, seed=0.001)) < 1e-12
    assert robust_weighted_z0_mean(z0, err, seed=5.0) == 5.0  # nothing compatible
