"""Tests for the histogram peak finder and its local-centroid position estimator.

Three copies of the algorithm exist and must stay numerically identical:

* ``pv_finder.utils.peak_finding``                       — canonical
* ``pv_finder.utils.peak_finding_fast``                  — numba
* ``pv_finder.evaluation.vertex_finding.
   efficiency_res_optimized_atlas``                      — legacy (6-tuple)

Expected values here are computed from first principles inside the tests, never
copied from a docstring.  Note the bin -> mm convention: the finder maps bin
index ``b`` to ``Z_MIN + b * BIN_WIDTH`` (the bin's *left edge*), so a peak
built symmetrically about bin ``b0`` has its true position at exactly that
value, with no half-bin offset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "src"
        / "pv_finder"
        / "evaluation"
        / "vertex_finding"
    ),
)

from efficiency_res_optimized_atlas import (  # noqa: E402
    pv_locations_updated_res as legacy_finder,
)

from pv_finder.utils.peak_finding import (  # noqa: E402
    LEGACY_CENTROID_HALFWIDTH,
    RECOMMENDED_CENTROID_HALFWIDTH,
    pv_locations_updated_res,
)

N_BINS = 12000
Z_MIN, Z_MAX = -240.0, 240.0
BIN_WIDTH = (Z_MAX - Z_MIN) / N_BINS  # 0.04 mm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def z_of(bin_pos: float) -> float:
    """Finder's bin -> mm mapping."""
    return Z_MIN + bin_pos * BIN_WIDTH


def gaussian(centre_bin: float, sigma_bins: float, amp: float) -> np.ndarray:
    """A 12000-bin histogram holding one Gaussian sampled at bin centres."""
    x = np.arange(N_BINS, dtype=np.float64)
    return (amp * np.exp(-0.5 * ((x - centre_bin) / sigma_bins) ** 2)).astype(
        np.float32
    )


def only_peak(hist: np.ndarray, **kw) -> tuple[float, float, int, float]:
    """Run the finder and assert exactly one peak came out; return it."""
    kw.setdefault("integral_threshold", 0.2)
    z, h, b, s = pv_locations_updated_res(hist, **kw)
    assert len(z) == 1, f"expected 1 peak, got {len(z)}"
    return float(z[0]), float(h[0]), int(b[0]), float(s[0])


@pytest.fixture(scope="module")
def realistic_histograms() -> list[np.ndarray]:
    """Histograms with the structure the finder actually meets in production:
    many peaks, some overlapping into conjoined regions, plus low-level noise."""
    rng = np.random.default_rng(20260804)
    out = []
    for _ in range(6):
        h = np.zeros(N_BINS, dtype=np.float64)
        centres = np.sort(rng.uniform(1000, 11000, 220))
        for c in centres:
            h += rng.uniform(0.05, 6.0) * np.exp(
                -0.5 * ((np.arange(N_BINS) - c) / rng.uniform(1.5, 4.0)) ** 2
            )
        h += rng.uniform(0.0, 0.02, N_BINS)
        out.append(h.astype(np.float32))
    return out


# ---------------------------------------------------------------------------
# the three implementations agree
# ---------------------------------------------------------------------------

PARAM_SETS = [
    dict(threshold=0.01, integral_threshold=0.2, min_width=3, min_height=0.0),
    dict(threshold=0.01, integral_threshold=0.5, min_width=3, min_height=0.03),
    dict(threshold=0.02, integral_threshold=0.8, min_width=2, min_height=0.05),
]


@pytest.mark.parametrize("params", PARAM_SETS)
@pytest.mark.parametrize("halfwidth", [0, 2, 3, 5])
def test_legacy_copy_matches_canonical(realistic_histograms, params, halfwidth):
    """efficiency_res_optimized_atlas must stay numerically identical."""
    for hist in realistic_histograms:
        z, h, b, s = pv_locations_updated_res(
            hist, centroid_halfwidth=halfwidth, **params
        )
        lz, lh, lb, _cl, _cr, ls = legacy_finder(
            hist,
            params["threshold"],
            params["integral_threshold"],
            params["min_width"],
            params["min_height"],
            halfwidth,
        )
        assert len(z) == len(lz)
        np.testing.assert_array_equal(b, lb)
        np.testing.assert_allclose(z, lz, rtol=0, atol=1e-6)
        np.testing.assert_allclose(h, lh, rtol=0, atol=1e-7)
        np.testing.assert_allclose(s, ls, rtol=0, atol=1e-6)


@pytest.mark.parametrize("params", PARAM_SETS)
@pytest.mark.parametrize("halfwidth", [0, 3])
def test_numba_copy_matches_canonical(realistic_histograms, params, halfwidth):
    """peak_finding_fast must stay bit-identical to the canonical scan."""
    numba = pytest.importorskip("numba")  # noqa: F841
    from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast

    for hist in realistic_histograms:
        a = pv_locations_updated_res(hist, centroid_halfwidth=halfwidth, **params)
        b = pv_locations_updated_res_fast(hist, centroid_halfwidth=halfwidth, **params)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)


def test_defaults_agree_across_copies(realistic_histograms):
    """Calling every copy with *no* estimator argument must give the same answer.

    This is the guard against the three defaults drifting apart.
    """
    hist = realistic_histograms[0]
    z_can, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3)
    z_leg, _, _, _, _, _ = legacy_finder(hist, 0.01, 0.2, 3)
    np.testing.assert_allclose(z_can, z_leg, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# the estimator changes only the position
# ---------------------------------------------------------------------------


def test_peak_set_independent_of_estimator(realistic_histograms):
    """Region detection must not depend on centroid_halfwidth."""
    for hist in realistic_histograms:
        ref = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=0)
        for hw in (1, 2, 3, 5, 12):
            got = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=hw)
            assert len(got[0]) == len(ref[0])
            np.testing.assert_array_equal(got[1], ref[1])  # heights
            np.testing.assert_array_equal(got[2], ref[2])  # peak bins


def test_sigmas_independent_of_estimator(realistic_histograms):
    """pv_sigmas is documented as the whole-region width; it must not move."""
    for hist in realistic_histograms:
        ref = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=0)[3]
        for hw in (2, 3, 7):
            got = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=hw)[3]
            np.testing.assert_array_equal(got, ref)


def test_halfwidth_zero_is_the_full_region_weighted_mean():
    """centroid_halfwidth=0 must reproduce sum(v*i)/sum(v) over the whole region."""
    hist = gaussian(centre_bin=4000.3, sigma_bins=3.0, amp=5.0)
    z, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=0)
    on = hist >= 0.01
    idx = np.arange(N_BINS)[on]
    val = hist[on].astype(np.float64)
    expected = z_of(float((idx * val).sum() / val.sum()))
    assert len(z) == 1
    # positions are stored as float32; at |z| ~ 80 mm that is ~6e-6 mm of
    # representation error, i.e. 1.5e-4 bins.
    assert abs(float(z[0]) - expected) < 1e-5


def test_local_centroid_uses_only_the_window():
    """The emitted position must equal the centroid of exactly max +- k bins."""
    hist = gaussian(centre_bin=4000.3, sigma_bins=3.0, amp=5.0)
    k = 3
    z, _, b, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=k)
    lo, hi = int(b[0]) - k, int(b[0]) + k
    idx = np.arange(lo, hi + 1)
    val = hist[lo : hi + 1].astype(np.float64)
    expected = z_of(float((idx * val).sum() / val.sum()))
    assert abs(float(z[0]) - expected) < 1e-5


# ---------------------------------------------------------------------------
# the physics the change was made for
# ---------------------------------------------------------------------------


def test_isolated_symmetric_peak_both_estimators_agree():
    """On a clean isolated Gaussian the two estimators must agree, and both
    must sit on the true centre."""
    centre = 6000.0
    hist = gaussian(centre_bin=centre, sigma_bins=3.0, amp=4.0)
    z_full, _, _, _ = only_peak(hist, centroid_halfwidth=0)
    z_loc, _, _, _ = only_peak(hist, centroid_halfwidth=3)
    truth = z_of(centre)
    assert abs(z_full - truth) < 0.002  # < 0.05 bin
    assert abs(z_loc - truth) < 0.002
    assert abs(z_full - z_loc) < 0.002


def test_conjoined_pair_local_centroid_is_more_accurate():
    """Two overlapping Gaussians that never dip below threshold.

    The conjoined split cuts the region at the valley, so each side's
    *full-region* centroid is biased away from its own peak.  The local
    centroid must recover both true centres more accurately.
    """
    c1, c2 = 5000.0, 5010.0  # 10 bins = 0.40 mm apart
    hist = gaussian(c1, 3.0, 4.0) + gaussian(c2, 3.0, 4.0)
    assert hist[int((c1 + c2) / 2)] > 0.01, "valley must stay above threshold"

    z_full, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=0)
    z_loc, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=3)
    assert len(z_full) == 2 and len(z_loc) == 2, "conjoined split should give 2 peaks"

    truth = np.array([z_of(c1), z_of(c2)])
    err_full = np.abs(np.asarray(z_full, dtype=np.float64) - truth)
    err_loc = np.abs(np.asarray(z_loc, dtype=np.float64) - truth)
    assert err_loc.max() < err_full.max()
    assert err_loc.sum() < err_full.sum()
    # and the local centroid should be good to well under half a bin
    assert err_loc.max() < 0.5 * BIN_WIDTH


@pytest.mark.parametrize("sep_bins", [8, 10, 12, 16, 20])
def test_conjoined_pair_separation_is_better_recovered(sep_bins):
    """The reconstructed separation of a conjoined pair must be closer to truth
    with the local centroid, across a range of separations."""
    c1 = 5000.0
    c2 = c1 + sep_bins
    hist = gaussian(c1, 3.0, 4.0) + gaussian(c2, 3.0, 4.0)
    z_full, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=0)
    z_loc, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=3)
    if len(z_full) != 2 or len(z_loc) != 2:
        pytest.skip(f"separation {sep_bins} bins does not split into two regions")
    true_sep = sep_bins * BIN_WIDTH
    err_full = abs((float(z_full[1]) - float(z_full[0])) - true_sep)
    err_loc = abs((float(z_loc[1]) - float(z_loc[0])) - true_sep)
    assert err_loc < err_full


def test_asymmetric_peak_full_region_mean_is_dragged_into_the_tail():
    """A peak with a long right-hand tail.

    The full-region weighted mean is pulled *towards the tail*; the local
    centroid stays near the mode.  Assert the direction and the ordering — this
    is what the code does, not a docstring claim.
    """
    x = np.arange(N_BINS, dtype=np.float64)
    mode = 7000.0
    core = 4.0 * np.exp(-0.5 * ((x - mode) / 2.5) ** 2)
    tail = 0.35 * np.exp(-np.clip(x - mode, 0, None) / 25.0) * (x >= mode)
    hist = (core + tail).astype(np.float32)

    z_full, _, b_full, _ = only_peak(hist, centroid_halfwidth=0)
    z_loc, _, b_loc, _ = only_peak(hist, centroid_halfwidth=3)
    assert b_full == b_loc  # same region, same maximum bin
    z_mode = z_of(b_full)

    assert z_full > z_mode, "full-region mean should be dragged right by the tail"
    assert abs(z_loc - z_mode) < abs(z_full - z_mode)


def test_window_is_clipped_at_a_conjoined_split():
    """The window must stop at the split boundary, not run into the neighbour.

    This is the guard on the *clipped* design decision.  We choose a half-width
    (7) wide enough that an unclipped window would reach across the valley into
    the right-hand peak, then assert the emitted position equals the clipped
    centroid and differs from the unclipped one.
    """
    k = 7
    c1, c2 = 5000.0, 5010.0
    hist = gaussian(c1, 3.0, 4.0) + gaussian(c2, 3.0, 4.0)

    z, _, bins, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=k)
    assert len(z) == 2, "expected a conjoined split into two peaks"
    m = int(bins[0])

    # The split fires on the first rising bin after the valley, and that bin is
    # accumulated into the left region — so the left region ends at valley + 1.
    valley = int(np.argmin(hist[int(c1) : int(c2) + 1])) + int(c1)
    region_end = valley + 1
    assert region_end < m + k, "half-width must overhang the region for this test"

    def centroid(lo: int, hi: int) -> float:
        idx = np.arange(lo, hi + 1)
        val = hist[lo : hi + 1].astype(np.float64)
        return z_of(float((idx * val).sum() / val.sum()))

    clipped = centroid(m - k, region_end)
    unclipped = centroid(m - k, m + k)
    assert abs(clipped - unclipped) > 1e-3, "test is only meaningful if they differ"
    assert abs(float(z[0]) - clipped) < 1e-5
    assert abs(float(z[0]) - unclipped) > 1e-3


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


def test_empty_and_flat_histograms():
    for hist in (np.zeros(N_BINS, np.float32), np.full(N_BINS, 0.005, np.float32)):
        z, h, b, s = pv_locations_updated_res(hist, 0.01, 0.2, 3)
        assert len(z) == len(h) == len(b) == len(s) == 0


@pytest.mark.parametrize("centre", [1.0, 3.0, N_BINS - 4.0, N_BINS - 2.0])
def test_peaks_at_the_array_edges_do_not_crash(centre):
    """A window clipped by the array boundary must still produce a finite
    position inside the histogram range."""
    hist = gaussian(centre, 2.0, 5.0)
    z, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=5)
    assert len(z) >= 1
    assert np.all(np.isfinite(z))
    assert np.all(z >= Z_MIN) and np.all(z <= Z_MAX)


def test_library_default_is_backwards_compatible(realistic_histograms):
    """The library default must stay the historical full-region weighted mean.

    Downstream consumers (notably the GNN graph builders, whose TTVA checkpoints
    were trained on these positions and sigmas) call the finder with no estimator
    argument and must keep getting exactly what they always got.  Opting in is
    the caller's job.
    """
    assert LEGACY_CENTROID_HALFWIDTH == 0
    assert RECOMMENDED_CENTROID_HALFWIDTH > 0
    for hist in realistic_histograms[:2]:
        implicit = pv_locations_updated_res(hist, 0.01, 0.2, 3)
        explicit = pv_locations_updated_res(hist, 0.01, 0.2, 3, centroid_halfwidth=0)
        for a, b in zip(implicit, explicit):
            np.testing.assert_array_equal(a, b)


def test_recommended_halfwidth_actually_changes_positions():
    """Guard against the recommended setting silently becoming a no-op."""
    hist = gaussian(4000.3, 3.0, 5.0) + gaussian(4010.3, 3.0, 5.0)
    z_legacy = pv_locations_updated_res(hist, 0.01, 0.2, 3)[0]
    z_reco = pv_locations_updated_res(
        hist, 0.01, 0.2, 3, centroid_halfwidth=RECOMMENDED_CENTROID_HALFWIDTH
    )[0]
    assert len(z_legacy) == len(z_reco) == 2
    assert np.max(np.abs(np.asarray(z_legacy) - np.asarray(z_reco))) > 1e-3
