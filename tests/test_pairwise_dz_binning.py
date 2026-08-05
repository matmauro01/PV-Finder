"""Tests for the pairwise-Δz binning helper.

The property under test is *commensurability*: reconstructed PV positions are
combed at the model's 0.04 mm bin pitch, so histogramming their pairwise |Δz|
into bins whose width is not an integer multiple of 0.04 mm produces a beat in
the plateau of the resolution plot.

The comb is reproduced here from first principles — a synthetic quantised
position set — rather than taken from measured data, so the tests state the
mechanism instead of pinning a number.  Every threshold is expressed relative to
the *statistical* noise floor of the estimator, computed in the test.
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_finder.utils.constants import BIN_WIDTH_MM
from pv_finder.utils.pairwise_dz import (
    DEFAULT_PAIRWISE_BINS,
    PAIRWISE_RANGE_MM,
    is_commensurate,
    pairwise_bins,
)

# The binning the eval used until 2026-08-05: 0.05 mm bins, incommensurate.
LEGACY_BINS = 240
BASE_LO, BASE_HI = 1.2, PAIRWISE_RANGE_MM
BEAT_PERIOD_MM = 0.20  # lcm(0.04, 0.05)


def _quantised_dz(n_pairs: int = 2_000_000, jitter_frac: float = 0.18) -> np.ndarray:
    """|Δz| for position pairs that pile up on the 0.04 mm grid.

    ``jitter_frac`` is the RMS scatter about a grid point in units of the bin
    width; 0.18 gives roughly the 40 % |Δz| comb modulation measured on v6.
    """
    rng = np.random.default_rng(20260805)
    a = np.round(rng.uniform(-240, 240, n_pairs) / BIN_WIDTH_MM) * BIN_WIDTH_MM
    b = a + np.round(rng.uniform(0.2, 6.0, n_pairs) / BIN_WIDTH_MM) * BIN_WIDTH_MM
    jit = rng.normal(0.0, jitter_frac * BIN_WIDTH_MM, (2, n_pairs))
    return np.abs((b + jit[1]) - (a + jit[0]))


def _comb(dz: np.ndarray, n_bins: int) -> tuple[float, float]:
    """(amplitude, Poisson noise floor) of a 0.20 mm plateau modulation.

    Both are fractions of the plateau level, so they are directly comparable
    across binnings.
    """
    edges = np.linspace(0.0, PAIRWISE_RANGE_MM, n_bins // 2 + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    cnt, _ = np.histogram(dz, bins=edges)
    m = (ctr >= BASE_LO) & (ctr <= BASE_HI)
    y = cnt[m].astype(float)
    slope, icpt = np.polyfit(ctr[m], y, 1)
    resid = y - (slope * ctr[m] + icpt)
    ph = 2 * np.pi * ctr[m] / BEAT_PERIOD_MM
    amp = np.hypot(2 * np.mean(resid * np.cos(ph)), 2 * np.mean(resid * np.sin(ph)))
    level = float(np.mean(y))
    # A pure-Poisson plateau projects onto the two quadratures with RMS
    # sqrt(2/n) * sqrt(level) each, so the modulus averages ~2*sqrt(level/n).
    floor = 2.0 * np.sqrt(level / len(y))
    return float(amp / level), float(floor / level)


# --------------------------------------------------------------- the helper


def test_default_binning_matches_the_model_grid():
    """0.04 mm plot bins across ±6 mm."""
    assert DEFAULT_PAIRWISE_BINS == 300
    width = 2 * PAIRWISE_RANGE_MM / DEFAULT_PAIRWISE_BINS
    assert width == pytest.approx(BIN_WIDTH_MM)


@pytest.mark.parametrize("multiple, expected", [(1, 300), (2, 150), (3, 100), (4, 75)])
def test_pairwise_bins_multiples(multiple, expected):
    assert pairwise_bins(multiple=multiple) == expected
    assert is_commensurate(expected)


def test_pairwise_bins_rejects_bad_multiple():
    with pytest.raises(ValueError):
        pairwise_bins(multiple=0)


@pytest.mark.parametrize("n_bins", [300, 150, 100, 75, 60, 50, 30])
def test_commensurate_binnings(n_bins):
    assert is_commensurate(n_bins)


@pytest.mark.parametrize("n_bins", [240, 480, 600, 1200, 120, 200, 0, -1])
def test_incommensurate_binnings(n_bins):
    """240 (0.05 mm) is the historical default and is *not* commensurate.

    Sub-multiples fail too: 600 bins is 0.02 mm, so every second bin catches a
    comb tooth and every other one catches none.  Only integer multiples of the
    model grid spread the teeth evenly.
    """
    assert not is_commensurate(n_bins)


# ------------------------------------------------------------- the mechanism


def test_synthetic_positions_really_are_combed():
    """Sanity check on the fixture before anything is concluded from it."""
    dz = _quantised_dz()
    c, _ = np.histogram((dz / BIN_WIDTH_MM) % 1.0, bins=10, range=(0, 1))
    assert (c.max() - c.min()) / c.mean() > 0.20


def test_quantised_positions_beat_against_incommensurate_bins():
    """The legacy 0.05 mm binning turns a 0.04 mm comb into a 0.20 mm sawtooth."""
    amp, floor = _comb(_quantised_dz(), LEGACY_BINS)
    assert amp > 10 * floor
    assert amp > 0.02


def test_commensurate_bins_remove_the_beat():
    """Same data, commensurate bins: the modulation drops to the noise floor."""
    dz = _quantised_dz()
    legacy, _ = _comb(dz, LEGACY_BINS)
    fixed, floor = _comb(dz, DEFAULT_PAIRWISE_BINS)
    assert fixed < 0.05 * legacy
    assert fixed < 3 * floor


def test_beat_is_absent_for_unquantised_positions():
    """The beat is a property of the *positions*, not of the binning alone.

    Continuous positions show no comb at either binning, which is what makes
    this a quantisation artefact rather than a histogramming bug.
    """
    rng = np.random.default_rng(7)
    dz = rng.uniform(0.2, 6.0, 2_000_000)
    for n_bins in (LEGACY_BINS, DEFAULT_PAIRWISE_BINS):
        amp, floor = _comb(dz, n_bins)
        assert amp < 3 * floor


def test_rebinning_conserves_the_distribution():
    """Rebinning must not move the distribution, only how it is sampled."""
    dz = _quantised_dz()
    counts = []
    for n_bins in (LEGACY_BINS, DEFAULT_PAIRWISE_BINS):
        edges = np.linspace(0.0, PAIRWISE_RANGE_MM, n_bins // 2 + 1)
        cnt, _ = np.histogram(dz, bins=edges)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        counts.append(cnt[(ctr >= BASE_LO) & (ctr <= BASE_HI)].sum())
    # The two windows differ only by half a bin at each edge.
    assert counts[0] == pytest.approx(counts[1], rel=5e-3)
