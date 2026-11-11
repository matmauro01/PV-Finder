"""Tests for the ITk-style hard-scatter resolution helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pv_finder.evaluation.vertex_finding.itk_resolution import (  # noqa: E402
    binned_iterative_rms,
    hs_residual,
    iterative_rms,
    local_pu_density,
)


def test_gaussian_core_recovered_with_tails() -> None:
    """A 3-RMS iteration recovers a Gaussian core under a flat tail.

    A 3 sigma truncation of a pure Gaussian keeps 99.73 % and returns
    0.9866 sigma (the iteration settles slightly inside 3 sigma, ~0.985);
    with a 2 % flat tail the result must stay close to that.
    """
    rng = np.random.default_rng(1)
    core = rng.normal(0.0, 0.02, 200_000)
    tail = rng.uniform(-1.0, 1.0, 4_000)
    res = iterative_rms(np.concatenate([core, tail]), initial_halfrange=1.0)
    assert res.converged
    assert abs(res.rms / 0.02 - 0.9866) < 0.02
    assert abs(res.rms_err - res.rms / np.sqrt(2 * res.n_in)) < 1e-15
    plain = float(np.concatenate([core, tail]).std())
    assert plain > 4 * res.rms  # the 2 % tail dominates a plain RMS


def test_initial_range_and_nan_handling() -> None:
    """NaN marks unmatched events and is ignored; the start range is applied."""
    r = np.array([np.nan, 0.01, -0.01, 0.02, -0.02, 5.0])
    res = iterative_rms(r, initial_halfrange=0.1)
    assert res.n_total == 5
    assert res.n_in == 4
    assert abs(res.mean) < 1e-12
    empty = iterative_rms(np.array([np.nan, 3.0]), initial_halfrange=0.1)
    assert np.isnan(empty.rms) and empty.n_total == 1


def test_hs_residual_sign_and_radius() -> None:
    """Closest reco within the radius, signed reco minus truth."""
    reco = np.array([-3.0, 1.05, 0.98])
    assert np.isclose(hs_residual(reco, 1.0, 0.1), -0.02)
    assert np.isnan(hs_residual(reco, 2.0, 0.1))
    assert np.isnan(hs_residual(np.zeros(0), 0.0, 0.1))
    assert np.isclose(hs_residual(np.array([0.1]), 0.0, 0.1), 0.1)  # edge inclusive


def test_local_density_excludes_hs() -> None:
    """Other truth vertices with |dz| < 2 mm, per 4 mm; the HS itself excluded."""
    z = np.array([0.0, 1.0, -1.99, 2.0, 10.0])
    assert local_pu_density(z, 0) == 2 / 4.0


def test_binned_edges() -> None:
    """Left-inclusive bins, the last bin closed on the right."""
    x = np.array([0.0, 0.5, 1.0, 1.0])
    r = np.array([0.01, -0.01, 0.02, -0.02])
    rows = binned_iterative_rms(x, r, np.array([0.0, 0.5, 1.0]), 1.0)
    assert [row["n_total"] for row in rows] == [1, 3]
