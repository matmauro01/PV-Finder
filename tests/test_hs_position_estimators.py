"""Tests for pv_finder.evaluation.vertex_finding.hs_position_estimators."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from pv_finder.evaluation.vertex_finding.hs_position_estimators import (
    BIN_WIDTH,
    adaptive_z_fit,
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


def test_adaptive_fit_recovers_true_z_and_rejects_outlier() -> None:
    rng = np.random.default_rng(1)
    err = rng.uniform(0.02, 0.08, 400)
    z0 = 1.5 + rng.normal(0.0, err)
    assert abs(adaptive_z_fit(z0, err, seed=1.45) - 1.5) < 0.004
    # A far, precise outlier pulls the plain mean but not the adaptive fit.
    z_out = np.append(z0[:30], 2.5)
    e_out = np.append(err[:30], 0.005)
    plain = float(np.sum(z_out / e_out**2) / np.sum(1 / e_out**2))
    fit = adaptive_z_fit(z_out, e_out, seed=1.5)
    assert abs(plain - 1.5) > 0.3
    assert abs(fit - 1.5) < 0.02


def test_adaptive_fit_prior_and_empty() -> None:
    z0 = np.array([0.0, 0.0, 0.05])
    err = np.array([0.03, 0.03, 0.03])
    # Zero prior removes the third track entirely.
    assert adaptive_z_fit(z0, err, seed=0.0, prior=np.array([1, 1, 0])) == 0.0
    assert adaptive_z_fit(np.array([]), np.array([]), seed=0.7) == 0.7


def test_adaptive_beamline_fit_recovers_displaced_vertex() -> None:
    from pv_finder.evaluation.vertex_finding.hs_position_estimators import (
        adaptive_beamline_fit,
    )

    rng = np.random.default_rng(3)
    n = 300
    xv, yv, zv = 0.010, -0.008, 2.0
    phi = rng.uniform(-np.pi, np.pi, n)
    theta = rng.uniform(0.04, np.pi - 0.04, n)  # up to |eta| ~ 3.9
    sd, sz = rng.uniform(0.005, 0.02, n), rng.uniform(0.01, 0.05, n)
    d0 = -xv * np.sin(phi) + yv * np.cos(phi) + rng.normal(0, sd)
    z0 = zv - (xv * np.cos(phi) + yv * np.sin(phi)) / np.tan(theta) + rng.normal(0, sz)
    x, y, z = adaptive_beamline_fit(
        d0, z0, phi, theta, sd**2, sz**2, np.zeros(n), seed_z=1.95
    )
    assert abs(z - zv) < 0.003 and abs(x - xv) < 0.003 and abs(y - yv) < 0.003
    # Ignoring the transverse offset (beam-axis z0 mean) is much worse here.
    z_axis = adaptive_z_fit(z0, sz, seed=1.95)
    assert abs(z_axis - zv) > abs(z - zv)
