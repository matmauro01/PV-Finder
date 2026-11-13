"""Tests for the Eq. (4) association and track bookkeeping of scripts/hllhc_final_chain_extras.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hllhc_final_chain_extras import (  # noqa: E402
    COUNT_FIELDS,
    WINDOW,
    arm_counts,
    best_significance,
    no_vertex_near,
)

from gnn.evaluation import hs_metrics as hm  # noqa: E402


def _event(seed: int, n_trk: int, n_pk: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    z0 = rng.uniform(-50, 50, n_trk).astype(np.float32)
    ez0 = rng.uniform(0.01, 0.5, n_trk).astype(np.float32)
    pz = np.sort(rng.uniform(-50, 50, n_pk)).astype(np.float32)
    ps = rng.uniform(0.02, 0.3, n_pk).astype(np.float32)
    return z0, ez0, pz, ps


@pytest.mark.parametrize(("seed", "n_pk"), [(s, n) for s in range(10) for n in (1, 5, 20, 21, 60)])  # fmt: skip
def test_best_significance_bruteforce(seed: int, n_pk: int) -> None:
    z0, ez0, pz, ps = _event(seed, 40, n_pk)
    knn = 20
    best, s, cand, n_zero = best_significance(z0, ez0, pz, ps, knn)
    assert n_zero == 0
    for i in range(len(z0)):
        dz = np.abs(z0[i] - pz)
        near = np.argsort(dz, kind="stable")[: min(knn, n_pk)]
        assert set(cand[i]) == set(near)
        si = np.abs((z0[i] - pz[near]) / np.sqrt(ez0[i] ** 2 + ps[near] ** 2))
        assert best[i] == near[np.argmin(si)]
        assert s[i] == pytest.approx(si.min(), rel=1e-6)


def test_best_significance_matches_graph_edge_feature() -> None:
    """Candidates and s are those of create_inference_graph (the GNN's graph)."""
    tg = pytest.importorskip("gnn.data.graph_construction")
    z0, ez0, pz, ps = _event(7, 50, 45)
    d0 = np.zeros_like(z0)
    stack = np.zeros((len(z0), 8), np.float32)
    g = tg.create_inference_graph(z0, d0, ez0, ez0, stack, pz, ps, ps, knn=20)
    ei = g[("track", "to", "pv")].edge_index.numpy()
    ea = g[("track", "to", "pv")].edge_attr.numpy()
    best, s, cand, _ = best_significance(z0, ez0, pz, ps, 20)
    for i in range(len(z0)):
        sel = ei[0] == i
        assert np.array_equal(np.sort(ei[1][sel]), cand[i])
        sig = np.abs(ea[sel, 0])
        assert ei[1][sel][np.argmin(sig)] == best[i]
        assert s[i] == pytest.approx(sig.min(), rel=1e-6)


def test_best_significance_no_peaks_and_threshold() -> None:
    z0, ez0, _, _ = _event(0, 5, 1)
    best, s, cand, _ = best_significance(z0, ez0, np.zeros(0, np.float32), np.zeros(0, np.float32), 20)  # fmt: skip
    assert np.all(best == -1) and np.all(np.isinf(s)) and cand.shape == (5, 0)
    # one peak, s exactly computable: |0.3 - 0| / sqrt(0.3^2 + 0.4^2) = 0.6
    b, s1, _, _ = best_significance(
        np.array([0.3], np.float32), np.array([0.3], np.float32),
        np.array([0.0], np.float32), np.array([0.4], np.float32), 20,
    )  # fmt: skip
    assert b[0] == 0 and s1[0] == pytest.approx(0.6, rel=1e-6)


def test_no_vertex_near() -> None:
    truth = np.array([0, 0, 1, -1, 2])
    tz = np.array([0.0, 5.0, 10.0])
    vz = np.array([0.05, 5.0 + 1.5 * WINDOW, 9.0])  # truth 1 just outside
    got = no_vertex_near(truth, tz, vz, WINDOW)
    assert got.tolist() == [False, False, True, False, True]
    assert no_vertex_near(truth, tz, np.zeros(0), WINDOW).tolist() == [True, True, True, False, True]  # fmt: skip


def test_arm_counts_bookkeeping() -> None:
    # tracks: 0-2 truth 0, 3-4 truth 1, 5 truth 2 (nTrk 1), 6-7 unlabelled
    truth = np.array([0, 0, 0, 1, 1, 2, -1, -1])
    tz, tn = np.array([0.0, 1.0, 3.0]), np.array([3, 2, 1])
    pt = np.ones(len(truth))
    peaks = np.array([0.0, 3.0, 8.0])  # nothing near truth 1
    tgt = np.array([0, 0, -1, 0, -1, 1, 2, -1])
    lists = [x for x in hm.lists_from_targets(tgt, len(peaks)) if len(x)]
    lost = no_vertex_near(truth, tz, peaks, WINDOW)
    c, cls = arm_counts(lists, peaks, pt, truth, tz, tn, lost)
    f = dict(zip(COUNT_FIELDS, c.tolist()))
    # vertex 0: tracks 0,1,3 -> truth 0 has 2/3 < 70% -> Merged; vertex 1: track 5
    # (truth 2, 1/1) -> Clean with an nTrk < 2 dominant; vertex 2: track 6 -> Fake
    assert (f["clean"], f["merged"], f["split"], f["fake"], f["n_reco"]) == (
        1,
        1,
        0,
        1,
        3,
    )
    assert (f["n_correct"], f["n_assigned_truth"], f["n_assigned"]) == (3, 4, 5)
    assert f["wrong"] == 1 and f["n_assigned_unique"] == 5
    assert f["n_labelled_unassigned"] == 2  # tracks 2, 4
    assert f["n_unlabelled_unassigned"] == 1  # track 7
    assert f["n_lost_unassigned"] == 1  # track 4: truth 1 has no peak nearby
    assert f["n_lost_unassigned_out"] == 1  # same on the output positions
    assert f["clean_dom_ge2"] == 0
    assert cls.labels.tolist() == [hm.MERGED, hm.CLEAN, hm.FAKE]
