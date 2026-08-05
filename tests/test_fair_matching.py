"""Tests for the independent audit matcher in ``diagnostics.fair_matching``."""

from __future__ import annotations

import numpy as np
import pytest

from pv_finder.diagnostics.fair_matching import (
    match_greedy,
    match_optimal,
    summarise,
)


def test_perfect_one_to_one():
    """Well-separated pairs inside the window all match, nothing left over."""
    t = np.array([0.0, 10.0, 20.0])
    r = np.array([0.1, 10.1, 20.1])
    m = match_optimal(t, r, 0.5)
    assert m.n_matched == 3
    assert m.n_unmatched_reco == 0
    assert m.n_missed == 0
    assert np.allclose(np.sort(m.residuals), 0.1)


def test_window_is_exclusive_beyond_edge():
    """A pair exactly at the window matches; just beyond it does not."""
    assert match_optimal(np.array([0.0]), np.array([0.5]), 0.5).n_matched == 1
    assert match_optimal(np.array([0.0]), np.array([0.5001]), 0.5).n_matched == 0


def test_optimal_beats_greedy():
    """The case greedy gets wrong: it takes the globally closest pair
    (t1,r0)=0.1 first, which consumes the only reco t0 could have used.
    The optimal solution pairs t0-r0 (0.9) and t1-r1 (1.0) for two matches."""
    t = np.array([0.0, 1.0])
    r = np.array([0.9, 2.0])
    assert match_optimal(t, r, 1.1).n_matched == 2
    assert match_greedy(t, r, 1.1).n_matched == 1


def test_one_to_one_enforced():
    """Two reco on one truth: one matches, the other is a leftover."""
    m = match_optimal(np.array([0.0]), np.array([-0.1, 0.1]), 0.5)
    assert m.n_matched == 1
    assert m.n_unmatched_reco == 1
    # the leftover sits inside the truth's window, so it is a "split"
    assert (m.n_split, m.n_fake) == (1, 0)


def test_split_vs_fake_accounting():
    """A duplicate near truth is 'split'; an isolated reco is 'fake'.
    Both are unmatched reco, which is what the audit counts."""
    m = match_optimal(np.array([0.0]), np.array([0.05, 0.1, 50.0]), 0.5)
    assert m.n_matched == 1
    assert m.n_split == 1
    assert m.n_fake == 1
    assert m.n_unmatched_reco == 2


def test_empty_inputs():
    """Degenerate events do not raise and account correctly."""
    m = match_optimal(np.array([]), np.array([1.0, 2.0]), 0.5)
    assert (m.n_matched, m.n_fake, m.n_unmatched_reco) == (0, 2, 2)
    m = match_optimal(np.array([1.0, 2.0]), np.array([]), 0.5)
    assert (m.n_matched, m.n_missed) == (0, 2)


def test_order_independence():
    """Optimal matching does not depend on input order; that is the whole
    reason it is used instead of the evaluation's greedy rule."""
    rng = np.random.default_rng(0)
    t = rng.uniform(-50, 50, 40)
    r = t + rng.normal(0, 0.1, 40)
    r = np.concatenate([r, rng.uniform(-50, 50, 10)])
    base = match_optimal(t, r, 0.3)
    for _ in range(5):
        pt, pr = rng.permutation(len(t)), rng.permutation(len(r))
        m = match_optimal(t[pt], r[pr], 0.3)
        assert m.n_matched == base.n_matched
        assert m.n_unmatched_reco == base.n_unmatched_reco


def test_optimal_never_worse_than_greedy():
    """Random events: optimal matches at least as many truths as greedy."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        t = np.sort(rng.uniform(-20, 20, rng.integers(1, 30)))
        r = np.sort(rng.uniform(-20, 20, rng.integers(1, 30)))
        w = 0.4
        assert match_optimal(t, r, w).n_matched >= match_greedy(t, r, w).n_matched


def test_summarise_pools_correctly():
    """Efficiency pools over vertices, fake rate averages over events."""
    a = match_optimal(np.array([0.0, 5.0]), np.array([0.1]), 0.5)  # 1/2, 0 fake
    b = match_optimal(np.array([0.0]), np.array([0.1, 9.0]), 0.5)  # 1/1, 1 fake
    s = summarise([a, b])
    assert s.efficiency == pytest.approx(2 / 3)
    assert s.fake_per_evt == pytest.approx(0.5)
    assert s.truth_per_evt == pytest.approx(1.5)
