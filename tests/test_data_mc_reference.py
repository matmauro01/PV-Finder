"""Checks for the data-style matching and paired bootstrap used in note Sec. 6.2."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from pv_finder.diagnostics.data_mc_validation.reference import (  # noqa: E402
    bootstrap,
    match_events,
    paired_delta,
)
from pv_finder.diagnostics.data_mc_validation.samples import fit_sigma  # noqa: E402


def test_toy_event_counts() -> None:
    # ref at 0, 10, 10.3; reco at 0.05 (clean), 10.15 (merges 10 and 10.3),
    # 0.2 (split: its only neighbour is taken), 50 (no reference).
    refs = [np.array([0.0, 10.0, 10.3]), np.zeros(0)]
    recos = [np.array([0.05, 10.15, 0.2, 50.0]), np.array([1.0])]
    s = match_events(refs, recos, window_mm=0.2).summary()
    assert s["n_events"] == 1  # empty-reference event dropped, as on data
    assert s["n_ref"] == 3 and s["ref_matched"] == 3
    assert (s["clean"], s["merged"], s["split"], s["no_ref"]) == (1, 1, 1, 1)
    assert s["recovery"] == 1.0 and s["no_ref_per_evt"] == 1.0


def test_paired_bootstrap_identical_inputs_give_zero_delta() -> None:
    rng = np.random.default_rng(0)
    # 3 mm spacing: no merges possible at a 0.1 mm window.
    refs = [np.arange(20) * 3.0 + rng.uniform(-1, 1) for _ in range(200)]
    recos = [r[:15] + 0.01 for r in refs]  # 15 of 20 found, well inside 0.1 mm
    c = match_events(refs, recos, 0.1)
    d = paired_delta(c, c, n_boot=200)
    for m in d.values():
        assert m["delta"] == 0.0 and m["delta_err"] == 0.0
    (b,) = bootstrap([c], n_boot=200)
    assert np.all(np.abs(b["recovery"] - 0.75) < 1e-12)  # 15 of 20 every event


def test_sigma_fit_recovers_dip_width() -> None:
    rng = np.random.default_rng(1)
    flat = rng.uniform(-6, 6, 400_000)
    keep = rng.random(len(flat)) < 1.0 / (1.0 + np.exp(25.0 * (0.4 - np.abs(flat))))
    f = fit_sigma(flat[keep], "adaptive")
    assert abs(f.sigma - 0.4) < 0.03
