"""Physics boundary cases for the event-level HS measurement."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "hs_reconstruction",
    Path(__file__).parents[1] / "src/pv_finder/diagnostics/hs_reconstruction.py",
)
hs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hs)


def test_empty_reconstruction_is_a_failed_event():
    assert np.isinf(hs.closest_distance([], 0.0))
    events = {"pvf_dz_mm": np.array([np.inf, 0.0]), "amvf_dz_mm": np.array([0.0, 0.0])}
    result = hs.compare(events, np.array([True, True]))
    assert result["pvf"]["total"] == 2
    assert result["pvf"]["passed"] == 1


def test_hs_match_does_not_require_exclusive_truth_assignment():
    # A pile-up truth at 0.01 would win a greedy match to reco=0.01.
    # The HS at zero still satisfies the published positional condition.
    assert hs.closest_distance([0.01], 0.0) < 0.1
    assert not hs.closest_distance([0.1], 0.0) < 0.1


def test_density_excludes_only_hs_and_uses_four_mm():
    assert hs.local_density(np.array([0.0, -1.99, 1.99, -2.0, 2.0, 0.0]), 0) == 0.75


def test_exact_intervals_keep_uncertainty_at_perfect_efficiency():
    result = hs.interval(10, 10)
    assert 0 < result["low"] < result["efficiency"] == result["high"] == 1
    assert hs.interval(0, 10)["high"] > 0
    assert hs.interval(0, 0)["efficiency"] is None
    with pytest.raises(ValueError):
        hs.interval(11, 10)


def test_binning_keeps_zero_and_final_edge_without_double_counting():
    events = {
        "x": np.array([0, 0.5, 1.0]),
        "pvf_dz_mm": np.zeros(3),
        "amvf_dz_mm": np.array([0.2, 0, 0]),
    }
    rows = hs.binned(events, "x", np.array([0, 0.5, 1]), np.ones(3, bool))
    assert [r["pvf"]["total"] for r in rows] == [1, 2]
    assert [r["paired"]["pvf_only_pass"] for r in rows] == [1, 0]
