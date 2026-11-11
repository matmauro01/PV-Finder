"""Target readout: the grid-parametrised builder equals the converter at 0.04 mm."""

from __future__ import annotations

import numpy as np

from pv_finder.data import root_to_h5
from pv_finder.diagnostics.run2_vs_run4.target_readout import (
    NOTE_CONFIG,
    build_truth_histogram,
    find_peaks,
)

ABC = (0.14318, 0.362, -0.01797)


def test_builder_matches_converter() -> None:
    rng = np.random.default_rng(1)
    z = rng.uniform(-239.0, 239.0, 300).astype(np.float32)
    n = rng.integers(0, 120, 300).astype(np.float32)
    root_to_h5.set_resolution(*ABC)
    ref = root_to_h5._build_truth_histogram(z, n)
    mine = build_truth_histogram(z, n, ABC, 0.04)
    np.testing.assert_allclose(mine, ref, rtol=0, atol=1e-9)  # bin-centre rounding only


def test_readout_recovers_isolated_vertex() -> None:
    for bin_mm in (0.04, 0.02):
        for z0 in (1.2345, -17.0071, 33.3333):
            h = build_truth_histogram(np.array([z0]), np.array([60]), ABC, bin_mm)
            p = find_peaks(h[0], NOTE_CONFIG, bin_mm)
            assert len(p) == 1
            assert abs(p[0] - z0) < 2e-3  # well inside one 0.04 mm bin


def test_low_ntrk_goes_to_channel_one() -> None:
    h = build_truth_histogram(np.array([0.5, 5.0]), np.array([1, 10]), ABC)
    near_low = slice(int((0.5 + 240.0) / 0.04) - 6, int((0.5 + 240.0) / 0.04) + 7)
    assert h[1][near_low].sum() > 0
    assert h[0][near_low].sum() == 0
    p = find_peaks(h[0], NOTE_CONFIG)
    assert len(p) == 1 and abs(p[0] - 5.0) < 2e-3
