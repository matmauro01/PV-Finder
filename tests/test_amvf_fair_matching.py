"""The fair-comparison matcher must agree with the production one exactly.

``amvf_fair_comparison.matching.match_vertices`` exists only to add the
truth-side efficiency and to drop an unused O(n^2) local-density loop.  If it
ever disagrees with ``compare_res_reco`` about clean/merged/split/fake, every
number in the fair comparison is measuring something other than what the
production eval measures, and the comparison stops being a comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(
    0, str(Path(__file__).parents[1] / "src/pv_finder/evaluation/vertex_finding")
)

from efficiency_res_optimized_atlas import (  # noqa: E402
    compare_res_reco,
    truth_found_incl_split,
)

from pv_finder.diagnostics.amvf_fair_comparison.matching import (  # noqa: E402
    excused_by_low_ntrk,
    match_vertices,
)


def _random_event(
    rng: np.random.Generator, window: float
) -> tuple[np.ndarray, np.ndarray]:
    """A PU200-like event: ~110 truth, ~100 reco, beam-spot z profile."""
    n_truth = rng.integers(80, 130)
    truth = rng.normal(0.0, 45.0, size=n_truth).astype(np.float32)
    # Reco = a smeared subset of truth plus surplus peaks, so every category
    # (clean, merged, split, fake) is populated at a realistic window.
    keep = truth[rng.random(n_truth) < 0.85]
    reco = keep + rng.normal(0.0, 0.5 * window, size=len(keep)).astype(np.float32)
    extra = rng.normal(0.0, 45.0, size=rng.integers(5, 25)).astype(np.float32)
    dupe_mask = rng.random(len(keep)) < 0.05
    dupes = keep[dupe_mask] + rng.normal(0.0, window, size=int(dupe_mask.sum())).astype(
        np.float32
    )
    reco = np.concatenate([reco, extra, dupes]).astype(np.float32)
    rng.shuffle(reco)
    return truth, reco


@pytest.mark.parametrize("window", [0.10, 0.22, 0.30, 0.50, 1.00])
def test_matches_production_matcher(window: float) -> None:
    """Counts agree event-for-event with compare_res_reco at every window."""
    rng = np.random.default_rng(20260805)
    n_events = 120
    seen = {"clean": 0, "merged": 0, "split": 0, "fake": 0}
    for _ in range(n_events):
        truth, reco = _random_event(rng, window)
        ref, truth_cls, _ = compare_res_reco(
            truth, reco, np.full(len(reco), window, dtype=np.float32), 0
        )
        got = match_vertices(truth, reco, window)

        assert got.reco_clean == ref.reco_clean
        assert got.reco_merged == ref.reco_merged
        assert got.reco_split == ref.reco_split
        assert got.reco_fake == ref.reco_fake
        # Truth-side efficiency numerator, which the reference exposes only as
        # a per-truth label array.
        ref_matched = int(np.sum(truth_cls != "missed"))
        assert got.n_truth_matched == ref_matched
        assert len(got.fake_idx) == ref.reco_fake

        seen["clean"] += ref.reco_clean
        seen["merged"] += ref.reco_merged
        seen["split"] += ref.reco_split
        seen["fake"] += ref.reco_fake

    # A test that never produced a merge or a split would pass trivially.
    for key, count in seen.items():
        assert count > 0, f"category '{key}' never occurred — test is not exercising it"


def test_empty_inputs() -> None:
    """Degenerate events behave like the production eval's own special cases."""
    z = np.array([1.0, 2.0], dtype=np.float32)
    empty = np.zeros(0, dtype=np.float32)

    got = match_vertices(empty, z, 0.5)
    assert (got.n_truth, got.reco_fake, got.n_truth_matched) == (0, 2, 0)

    got = match_vertices(z, empty, 0.5)
    assert (got.n_reco, got.reco_fake, got.n_truth_matched) == (0, 0, 0)


def test_excused_is_exclusive() -> None:
    """One low-nTrk interaction excuses at most one fake, closest first."""
    fakes = np.array([0.00, 0.10, 0.20], dtype=np.float32)
    low = np.array([0.12], dtype=np.float32)
    # All three are within 0.5 mm of the single interaction; only one is excused.
    assert excused_by_low_ntrk(fakes, low, 0.5) == 1
    # Two interactions excuse two.
    assert excused_by_low_ntrk(fakes, np.array([0.12, 0.01], np.float32), 0.5) == 2
    # Nothing in range.
    assert excused_by_low_ntrk(fakes, np.array([9.0], np.float32), 0.5) == 0
    assert excused_by_low_ntrk(fakes, np.zeros(0, np.float32), 0.5) == 0


def test_window_monotonicity() -> None:
    """Widening the window can only ever reduce the fake count."""
    rng = np.random.default_rng(7)
    truth, reco = _random_event(rng, 0.3)
    fakes = [match_vertices(truth, reco, w).reco_fake for w in (0.1, 0.3, 0.5, 1.0)]
    assert fakes == sorted(fakes, reverse=True)


# --- Split-inclusive efficiency (reviewer request, 2026-11-09) -------------
#
# "Split" is a reco-side label: an unassigned reco with >= 1 truth in its
# window.  The counting rule under test: a truth vertex inside the window of a
# split reco also counts as found, at most once.  The claim the note relies on
# is that this adds nothing to either numerator; these tests measure it rather
# than assume it, including at distance ties, the window edge and with
# per-reco windows of different sizes.


def test_split_credit_hand_example() -> None:
    """One truth, two reco on top of it: one clean, one split; no extra credit."""
    truth = np.array([0.0, 5.0], dtype=np.float64)
    reco = np.array([0.02, -0.05, 9.0], dtype=np.float64)
    perf, tcls, _, rcls = compare_res_reco(
        truth, reco, np.full(3, 0.1), 0, return_reco_cls=True
    )
    assert list(rcls) == ["clean", "split", "fake"]
    assert list(tcls) == ["clean", "missed"]
    c = truth_found_incl_split(truth, reco, np.full(3, 0.1), tcls, rcls)
    assert c["split_credited"] == 1  # truth 0 sits in the split window...
    assert c["found_incl_split"] == c["found"] == 1  # ...but was already found
    assert c["primary_incl_split"] == c["primary"] == 1
    assert c["found_split_only"] == c["primary_split_only"] == 0
    got = match_vertices(truth, reco, 0.1)
    assert got.n_truth_matched_incl_split == got.n_truth_matched == 1
    assert got.n_truth_primary_incl_split == got.n_truth_primary == 1


def test_return_reco_cls_is_opt_in() -> None:
    """Existing callers keep the 3-tuple; the flag adds the reco labels only."""
    truth = np.array([0.0], dtype=np.float32)
    reco = np.array([0.01], dtype=np.float32)
    assert len(compare_res_reco(truth, reco, np.full(1, 0.1), 0)) == 3
    out = compare_res_reco(truth, reco, np.full(1, 0.1), 0, return_reco_cls=True)
    assert len(out) == 4 and list(out[3]) == ["clean"]


@pytest.mark.parametrize("window", [0.10, 0.1694, 0.30, 0.50, 1.00])
def test_split_credit_adds_nothing(window: float) -> None:
    """Split credit never adds a truth vertex, on realistic and on gridded events."""
    rng = np.random.default_rng(20261109)
    n_split_seen = n_credited_seen = 0
    for k in range(150):
        truth, reco = _random_event(rng, window)
        if k % 3 == 0:
            # Snap to the 0.04 mm output grid to force exact distance ties.
            truth = (np.round(truth / 0.04) * 0.04).astype(np.float32)
            reco = (np.round(reco / 0.04) * 0.04).astype(np.float32)
        res = np.full(len(reco), window, dtype=np.float32)
        perf, tcls, _, rcls = compare_res_reco(
            truth, reco, res, 0, return_reco_cls=True
        )
        c = truth_found_incl_split(truth, reco, res, tcls, rcls)
        assert c["found_split_only"] == 0
        assert c["primary_split_only"] == 0
        assert c["found_incl_split"] == int(np.sum(tcls != "missed"))
        assert c["primary_incl_split"] == int(np.sum(tcls == "clean"))
        got = match_vertices(truth, reco, window)
        assert got.n_truth_matched_incl_split == c["found_incl_split"]
        assert got.n_truth_primary_incl_split == c["primary_incl_split"]
        n_split_seen += perf.reco_split
        n_credited_seen += c["split_credited"]
    # Guard against a vacuous pass: splits must occur and must cover truth.
    assert n_split_seen > 0 and n_credited_seen > 0


def test_split_credit_with_per_reco_windows() -> None:
    """The identity also holds when every reco has its own window size."""
    rng = np.random.default_rng(3)
    for _ in range(100):
        truth, reco = _random_event(rng, 0.3)
        res = rng.uniform(0.05, 0.6, size=len(reco)).astype(np.float32)
        _, tcls, _, rcls = compare_res_reco(truth, reco, res, 0, return_reco_cls=True)
        c = truth_found_incl_split(truth, reco, res, tcls, rcls)
        assert c["found_split_only"] == 0 and c["primary_split_only"] == 0


def test_found_equals_any_reco_within_window() -> None:
    """Merged-credit found == 'some reco within w', the replay rule used in the note."""
    rng = np.random.default_rng(11)
    for _ in range(100):
        truth, reco = _random_event(rng, 0.1694)
        got = match_vertices(truth, reco, 0.1694)
        any_within = (np.abs(truth[:, None] - reco[None, :]) <= 0.1694).any(axis=1)
        assert got.n_truth_matched == int(any_within.sum())
