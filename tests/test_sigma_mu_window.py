"""The σ_vtx-vtx fit must run on the same events the summary is quoted on.

Until 2026-08-05 `run_eval_pvf_run3.py` accumulated pairwise Δz over *every*
event it read and applied the μ window only to the summary.  On the flat-μ
held-out files that meant σ was fitted at ⟨μ⟩ ≈ 100 and printed next to
efficiency measured at μ ∈ [185, 215].  Because σ is fed back as the matching
window, the error propagated into the headline efficiency.

These tests pin the shared selection predicate, and demonstrate on synthetic
data — using the production sigmoid — that the two populations really do give
different σ, so the fix is not cosmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import sigmoid_fit
from pv_finder.utils.pairwise_dz import (
    DEFAULT_PAIRWISE_BINS,
    PAIRWISE_RANGE_MM,
    event_mu,
    in_summary_window,
)

MU_MIN, MU_MAX = 185, 215


# --------------------------------------------------------------- the predicate


def test_event_mu_falls_back_to_the_truth_count():
    assert event_mu(192.5, 100) == 192.5
    assert event_mu(None, 100) == 100.0


@pytest.mark.parametrize("mu", [185, 190, 200, 215])
def test_events_inside_the_window_are_selected(mu):
    assert in_summary_window(mu, 100, MU_MIN, MU_MAX)


@pytest.mark.parametrize("mu", [0, 60, 100, 184, 216, 300])
def test_events_outside_the_window_are_rejected(mu):
    assert not in_summary_window(mu, 100, MU_MIN, MU_MAX)


def test_events_with_no_truth_vertex_are_excluded():
    """They cannot enter an efficiency, so they must not enter the σ fit either."""
    assert not in_summary_window(200.0, 0, MU_MIN, MU_MAX)
    assert not in_summary_window(None, 0, MU_MIN, MU_MAX)


def test_rounding_convention_is_bankers_rounding():
    """Pinned, not relied on: `round(184.5)` is 184 and `round(215.5)` is 216."""
    assert not in_summary_window(184.5, 100, MU_MIN, MU_MAX)
    assert in_summary_window(185.5, 100, MU_MIN, MU_MAX)
    assert in_summary_window(215.4, 100, MU_MIN, MU_MAX)
    assert not in_summary_window(215.5, 100, MU_MIN, MU_MAX)


def test_missing_mu_uses_the_truth_count_as_the_selector():
    """No ActualNumOfInt: μ falls back to the truth count, so the window acts on it."""
    assert in_summary_window(None, 200, MU_MIN, MU_MAX)
    assert not in_summary_window(None, 60, MU_MIN, MU_MAX)


# ------------------------------------------------------- why it actually matters


def _fit_sigma(dz: np.ndarray) -> float:
    """The production fit, verbatim in structure."""
    bins = np.linspace(-PAIRWISE_RANGE_MM, PAIRWISE_RANGE_MM, DEFAULT_PAIRWISE_BINS + 1)
    ctrs = 0.5 * (bins[:-1] + bins[1:])
    cnts, _ = np.histogram(dz, bins=bins)
    base = float(np.median(cnts))
    p0 = [max(base - float(cnts.min()), 1.0), 10.0, max(base, 1.0), 0.5]
    popt, _ = curve_fit(
        sigmoid_fit, ctrs, cnts.astype(float), p0=p0, maxfev=20000,
        bounds=([0, 0, 0, 0], [np.inf] * 4),
    )  # fmt: skip
    return float(abs(popt[3]))


def _events(n_events: int, mu: float, dip_mm: float, seed: int):
    """Synthetic per-event pairwise Δz: a flat plateau with a box-shaped dip.

    Pair count grows like μ², which is what makes the mixed-μ average a
    *weighted* one and the two σ different.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_events):
        n_pairs = max(int(rng.poisson(mu) ** 2 // 40), 4)
        dz = rng.uniform(-PAIRWISE_RANGE_MM, PAIRWISE_RANGE_MM, n_pairs)
        out.append(dz[np.abs(dz) > dip_mm])
    return out


@pytest.fixture(scope="module")
def mixed_population():
    """A flat-μ file: a low-μ bulk with a wider dip plus a high-μ tail.

    The counts are chosen so the low-μ bulk contributes a comparable number of
    *pairs* to the high-μ tail, which is the situation on the real held-out
    files: 1920 of 25 000 events sit in the window, but each carries ~4x the
    pairs of a mean-μ event.
    """
    low = _events(4000, mu=60.0, dip_mm=0.34, seed=1)
    high = _events(200, mu=200.0, dip_mm=0.20, seed=2)
    per_event = low + high
    mus = [60.0] * len(low) + [200.0] * len(high)
    return per_event, mus


def test_selection_changes_the_fitted_sigma(mixed_population):
    """The whole point: the two populations do not give the same answer."""
    per_event, mus = mixed_population
    keep = [in_summary_window(m, 100, MU_MIN, MU_MAX) for m in mus]
    assert 0 < sum(keep) < len(per_event)

    sigma_all = _fit_sigma(np.concatenate(per_event))
    sigma_win = _fit_sigma(np.concatenate([d for d, k in zip(per_event, keep) if k]))
    assert abs(sigma_all - sigma_win) > 0.03, (
        f"populations too similar to mean anything: "
        f"all={sigma_all:.4f} window={sigma_win:.4f}"
    )


def test_window_fit_recovers_the_window_population(mixed_population):
    """Selecting reproduces a fit on that population alone, exactly."""
    per_event, mus = mixed_population
    keep = [in_summary_window(m, 100, MU_MIN, MU_MAX) for m in mus]
    selected = np.concatenate([d for d, k in zip(per_event, keep) if k])
    alone = np.concatenate(per_event[-sum(keep) :])
    assert selected.size == alone.size
    assert _fit_sigma(selected) == pytest.approx(_fit_sigma(alone))


def test_concatenation_preserves_the_flat_all_events_array(mixed_population):
    """Grouping Δz per event must not change the all-events array it replaces."""
    per_event, _ = mixed_population
    flat = [float(v) for d in per_event for v in d]
    assert np.array_equal(np.concatenate(per_event), np.array(flat, dtype=np.float64))


def test_no_window_selects_everything_and_changes_nothing(mixed_population):
    """With a window that spans the sample, the fix is a no-op by construction."""
    per_event, mus = mixed_population
    keep = [in_summary_window(m, 100, 0, 1000) for m in mus]
    assert all(keep)
    assert _fit_sigma(
        np.concatenate([d for d, k in zip(per_event, keep) if k])
    ) == pytest.approx(_fit_sigma(np.concatenate(per_event)))
