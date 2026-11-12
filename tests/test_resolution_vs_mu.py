"""Tests for pv_finder.diagnostics.resolution_vs_mu."""

from __future__ import annotations

import numpy as np

from pv_finder.diagnostics.amvf_fair_comparison.scan import sigmoid_fit
from pv_finder.diagnostics.resolution_vs_mu import (
    bin_summary,
    fit_chi2,
    mu_bin_index,
    select_entries,
)


def test_mu_bin_index_closes_last_bin() -> None:
    edges = np.array([0.0, 10.0, 20.0])
    mu = np.array([0, 9.9, 10, 19.99, 20, 20.5, -1])
    assert mu_bin_index(mu, edges).tolist() == [0, 0, 1, 1, 1, -1, -1]


def test_select_entries_caps_and_fills_from_second_file() -> None:
    edges = np.array([0.0, 10.0, 20.0])
    a = np.array([1, 2, 15, 3, 16])
    b = np.array([4, 17, 5, 18])
    out = select_entries({"a": a, "b": b}, edges, np.array([4, 2]))
    assert out["a"].tolist() == [0, 1, 2, 3, 4]  # 3 low (all), 2 high (cap)
    assert out["b"].tolist() == [0]  # one more low, high already full


def test_select_entries_is_deterministic() -> None:
    edges = np.array([0.0, 100.0])
    mu = np.random.default_rng(0).uniform(0, 100, 50)
    x = select_entries({"f": mu}, edges, np.array([10]))
    y = select_entries({"f": mu}, edges, np.array([10]))
    assert x["f"].tolist() == y["f"].tolist() == list(range(10))


def test_fit_chi2_is_zero_on_the_model() -> None:
    edges = np.linspace(-6, 6, 301)
    c = 0.5 * (edges[:-1] + edges[1:])
    popt = np.array([100.0, 8.0, 20.0, 0.3])
    chi2, ndf = fit_chi2(edges, sigmoid_fit(c, *popt), popt)
    assert chi2 < 1e-9 and ndf == 296


def test_bin_summary_recovers_a_hard_dip() -> None:
    """Vertices never closer than 0.3 mm: sigma near 0.3 mm in every bin."""
    rng = np.random.default_rng(1)
    events, truth, mu = [], [], []
    for i in range(400):
        z = np.sort(rng.uniform(-150, 150, 60))
        z = z[np.concatenate([[True], np.diff(z) > 0.3])]
        events.append(z)
        truth.append(z + rng.normal(0, 0.02, len(z)))
        mu.append(50.0 if i < 200 else 150.0)
    rows, _ = bin_summary(events, events, truth, np.array(mu),
                          np.array([0.0, 100.0, 200.0]), n_boot=20)  # fmt: skip
    for r in rows:
        assert r["n_events"] == 200
        assert abs(r["pvf"]["sigma_mm"] - 0.3) < 0.05
        assert r["pvf"]["sigma_mm"] == r["amvf"]["sigma_mm"]
        assert 10.0 < r["pvf"]["p68_um"] < 30.0
