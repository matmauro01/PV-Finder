"""Tests for pv_finder.diagnostics.resolution_vs_mu."""

from __future__ import annotations

import numpy as np

from pv_finder.diagnostics.amvf_fair_comparison.scan import sigmoid_fit
from pv_finder.diagnostics.resolution_vs_mu import (
    bin_summary,
    fit_chi2,
    markdown_table,
    mu_bin_index,
    paired_bin_diff,
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


def test_paired_bin_diff_is_zero_for_identical_arms() -> None:
    rng = np.random.default_rng(2)
    H = rng.poisson(20, size=(50, 300))
    res = [np.abs(rng.normal(0, 0.05, 10)) for _ in range(50)]
    edges = np.linspace(-6, 6, 301)
    d = paired_bin_diff(H, H, edges, res, res, np.arange(50), n_boot=5)
    assert d["d_sigma_um"] == 0.0 and d["d_p68_um"] == 0.0
    assert d["d_sigma_err_um"] == 0.0 and d["d_p68_err_um"] == 0.0


def test_paired_bin_diff_sees_a_residual_shift() -> None:
    rng = np.random.default_rng(3)
    H = rng.poisson(20, size=(40, 300))
    res = [np.abs(rng.normal(0, 0.05, 20)) for _ in range(40)]
    wider = [2.0 * r for r in res]  # every residual doubled: p68 doubles exactly
    d = paired_bin_diff(H, H, np.linspace(-6, 6, 301), wider, res, np.arange(40),
                        n_boot=5)  # fmt: skip
    p68 = 1e3 * np.percentile(np.concatenate(res), 68.0)
    assert np.isclose(d["d_p68_um"], p68)


def test_markdown_table_has_one_row_per_bin() -> None:
    arm = dict(sigma_mm=0.2, boot_err_mm=0.002, rel_err=0.01, chi2=592.0, ndf=296,
               merge_len_mm=0.1, p68_um=55.0, p68_err_um=0.2)  # fmt: skip
    rows = [dict(mu_lo=0.0, mu_hi=10.0, n_events=5, pvf=arm, amvf=arm)] * 3
    t = markdown_table(rows).splitlines()
    assert len(t) == 5
    assert "| 0-10 | 5 | 200.0 ± 2.0 (1.0%)" in t[2] and "| 2.0 |" in t[2]


def test_script_defaults_are_the_frozen_hllhc_point() -> None:
    """scripts/resolution_vs_mu.py runs at get_regime('hllhc') unless overridden."""
    import argparse
    import importlib
    import sys
    from pathlib import Path

    from pv_finder.utils.working_points import get_regime

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    rvm = importlib.import_module("resolution_vs_mu")
    keys = list(rvm.MODEL_ARGS) + list(rvm.PEAK_ARGS)
    a = argparse.Namespace(**dict.fromkeys(keys))
    rvm.resolve_working_point(a)
    r = get_regime("hllhc")
    assert a.overrides == {}
    assert Path(a.model).name == Path(r.density.checkpoint).name
    assert (a.unet_channels, a.latent, a.hidden) == (80, 1, [128] * 5)
    assert a.peaks == dict(threshold=0.01, integral_threshold=0.184, min_width=3,
                           min_height=0.0184, centroid_halfwidth=3)  # fmt: skip
    b = argparse.Namespace(**dict.fromkeys(keys))
    b.integral_threshold, b.unet_channels, b.min_width = 0.30, 280, 3
    rvm.resolve_working_point(b)
    assert b.overrides == {"integral_threshold": 0.30, "unet_channels": 280}
    assert b.peaks["integral_threshold"] == 0.30


def test_load_chunk_splits_events(tmp_path) -> None:  # noqa: ANN001
    """One npz chunk -> per-event peaks, truth, AMVF, mu and ids, in order."""
    import importlib
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    rvm = importlib.import_module("resolution_vs_mu")
    H = np.zeros((2, 12000), dtype=np.float32)
    H[0, 6000:6005] = 0.2  # one clear peak near z = 0 in event 0, none in event 1
    f = tmp_path / "r16443_000000_005000.npz"
    np.savez(f, histograms=H, event_idx=np.array([7, 9]), mu=np.array([10.0, 20.0]),
             truth_z=np.array([0.1, 1.0, 2.0]), truth_off=np.array([0, 1, 3]),
             amvf_z=np.array([0.2]), amvf_off=np.array([0, 1, 1]))  # fmt: skip
    peaks = dict(threshold=0.01, integral_threshold=0.184, min_width=3,
                 min_height=0.0184, centroid_halfwidth=3)  # fmt: skip
    rec = rvm.load_chunk(f, peaks)
    assert rec["id"] == ["r16443:7", "r16443:9"] and rec["mu"] == [10.0, 20.0]
    assert len(rec["pvf"][0]) == 1 and len(rec["pvf"][1]) == 0
    assert abs(rec["pvf"][0][0] - 0.1) < 0.05
    assert rec["truth"][1].tolist() == [1.0, 2.0] and rec["amvf"][1].tolist() == []
