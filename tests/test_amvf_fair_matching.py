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

from efficiency_res_optimized_atlas import compare_res_reco  # noqa: E402

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
