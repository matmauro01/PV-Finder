"""Tests for the GNN v2 freeze chain code (chain_inference, chain_t_metrics, chain_t_summary)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from gnn.evaluation.chain_inference import best_edges, parse_entries  # noqa: E402
from gnn.evaluation.chain_t_metrics import COUNT_FIELDS, list_counts  # noqa: E402
from gnn.evaluation.chain_t_summary import (  # noqa: E402
    ChainRecord,
    check_bookkeeping,
    ratio_table,
)
from gnn.evaluation.classification import classify_assignments  # noqa: E402


@pytest.mark.parametrize("seed", range(20))
def test_best_edges_matches_bruteforce(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_trk, k = 30, 4
    src = np.repeat(np.arange(n_trk), k)
    dst = rng.integers(0, 10, n_trk * k)
    # coarse scores create ties; ties must go to the first edge
    scores = rng.integers(0, 4, n_trk * k).astype(np.float32) / 4
    tg, sc = best_edges(scores, np.stack([src, dst]), n_trk)
    for t in range(n_trk):
        s = scores[t * k : (t + 1) * k]
        j = int(np.argmax(s))  # numpy argmax returns the first maximum
        assert tg[t] == dst[t * k + j] and sc[t] == s[j]


def test_best_edges_handles_tracks_without_edges() -> None:
    tg, sc = best_edges(np.zeros(0, np.float32), np.zeros((2, 0), np.int64), 3)
    assert tg.tolist() == [-1, -1, -1] and sc.tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        best_edges(np.ones(2, np.float32), np.array([[1, 0], [0, 0]]), 2)


def test_parse_entries(tmp_path: Path) -> None:
    assert parse_entries("3:6").tolist() == [3, 4, 5]
    p = tmp_path / "e.npy"
    np.save(p, np.array([9, 2, 5]))
    assert parse_entries(str(p)).tolist() == [2, 5, 9]


@pytest.mark.parametrize("seed", range(15))
def test_list_counts_against_reference_classifier(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_trk, n_truth, n_vtx = 50, 5, 7
    truth = rng.integers(-1, n_truth, n_trk)
    pt = rng.uniform(1, 20, n_trk)
    target = rng.integers(-1, n_vtx, n_trk)
    lists = [np.flatnonzero(target == v) for v in range(n_vtx)]
    lists = [x for x in lists if len(x)]
    vz = np.sort(rng.uniform(-5, 5, len(lists)))
    truth_z = rng.uniform(-5, 5, n_truth)
    hs = 0
    row = dict(zip(COUNT_FIELDS, list_counts(lists, vz, pt, truth, truth_z, hs, truth_z[hs], None)))  # fmt: skip
    tti = np.flatnonzero(truth >= 0).astype(float)
    tpi = truth[truth >= 0].astype(float)
    ref, info = classify_assignments(lists, pt, tti, tpi, n_truth)
    assert [row[k] for k in ("clean", "merged", "split", "fake", "n_reco")] == list(
        ref[:5]
    )
    assert sum(row[k] for k in ("p_clean", "p_merged", "p_split", "p_fake")) == len(
        lists
    )
    correct = 0
    for x, r in zip(lists, info):
        p = r["primary_truth_pv"]
        if p not in (None, "Fake"):
            correct += int((truth[x] == int(p.rsplit("_", 1)[1])).sum())
    assert row["n_correct"] == correct
    assert row["n_assigned"] == sum(len(x) for x in lists)
    assert row["n_assigned_truth"] == sum(int((truth[x] >= 0).sum()) for x in lists)


def _record(counts: np.ndarray, events: np.ndarray) -> ChainRecord:
    meta = {
        "t_grid": [0.5],
        "count_fields": list(COUNT_FIELDS),
        "event_fields": [
            "mu",
            "n_truth_ge2",
            "n_labelled",
            "n_tracks",
            "n_hs_tracks",
            "has_hs",
        ],
        "hs_computed": False,
    }
    n = len(events)
    return ChainRecord(counts, np.zeros((n, 3, 4)), np.zeros((n, 2)), np.zeros((n, 1)), events, meta)  # fmt: skip


def test_bookkeeping_and_ratio_of_sums() -> None:
    f = {k: i for i, k in enumerate(COUNT_FIELDS)}
    counts = np.zeros((2, 2, 2, len(COUNT_FIELDS)), np.int64)
    # event 0: 3 clean of 4 reco; event 1: 1 clean of 1 reco
    for ev, (c, fk) in enumerate([(3, 1), (1, 0)]):
        for a in range(2):
            for m in range(2):
                counts[ev, a, m, f["clean"]] = c
                counts[ev, a, m, f["fake"]] = fk
                counts[ev, a, m, f["p_clean"]] = c + fk
                counts[ev, a, m, f["n_reco"]] = c + fk
    events = np.array([[10, 4, 0, 0, 0, 1], [100, 2, 0, 0, 0, 1]], float)
    rec = _record(counts, events)
    check_bookkeeping(rec)
    w = np.ones((5, 2))
    r = ratio_table(rec, np.ones(2, bool), w)
    assert r["clean_per_truth"]["value"][0] == pytest.approx(
        4 / 6
    )  # not mean of 3/4, 1/2
    assert r["clean_per_reco"]["value"][0] == pytest.approx(4 / 5)
    bad = counts.copy()
    bad[0, 0, 0, f["fake"]] = 0
    with pytest.raises(AssertionError):
        check_bookkeeping(_record(bad, events))
