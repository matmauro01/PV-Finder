"""Pin the worked example in docs/evaluation/metric_definitions.md.

The document quotes these numbers as the output of the production classifiers.
If either classifier changes, this fails and the document has to be updated
rather than quietly going stale.
"""

from __future__ import annotations

import numpy as np
import pytest

from pv_finder.diagnostics.metric_worked_example import (
    build_event,
    order_dependence,
    positional_pass,
    track_pass,
)


@pytest.fixture(scope="module")
def event():
    """The synthetic event both taxonomies are run on."""
    return build_event()


def test_event_is_self_consistent(event):
    """Every truth track is owned once; every assigned track exists."""
    owned = np.concatenate(event.truth_tracks)
    assert len(owned) == len(np.unique(owned)), "a track belongs to two truth PVs"
    assert owned.max() < event.n_tracks
    for tracks in event.reco_tracks:
        assert len(tracks) == len(np.unique(tracks)), "a track assigned twice"
        assert np.all(tracks < event.n_tracks)
    assert len(event.truth_z) == len(event.truth_tracks)
    assert len(event.reco_z) == len(event.reco_tracks)


def test_positional_categories(event):
    """compare_res_reco at the production 0.22 mm window, nTrk >= 2 truth."""
    res = positional_pass(event)
    assert res["n_truth"] == 5, "T5 (nTrk = 1) must be filtered out of truth"
    assert res["n_reco"] == 6
    assert (res["clean"], res["merged"], res["split"], res["fake"]) == (2, 1, 1, 2)
    assert res["reco_cls"] == ["split", "clean", "merged", "fake", "clean", "fake"]
    assert (res["truth_clean"], res["truth_merged"], res["truth_missed"]) == (3, 1, 1)
    assert res["efficiency"] == pytest.approx(0.80)
    assert res["fake_per_event"] == pytest.approx(2.0)
    assert res["clean_per_truth"] == pytest.approx(0.40)


def test_track_purity_categories(event):
    """classify_assignments on the identical objects."""
    res, info = track_pass(event)
    assert (res["clean"], res["merged"], res["split"], res["fake"]) == (1, 2, 2, 1)
    assert res["n_reco"] == 6
    assert res["n_truth"] == 5, "denominator is nTrk >= 2 truth PVs"
    assert res["clean_per_truth"] == pytest.approx(0.20)
    assert [i["classification"] for i in info] == [
        "Split",
        "Merged",
        "Merged",
        "Fake",
        "Clean",
        "Split",
    ]
    # P_b sits at the right z but its track list is 60% pure, below the 0.70 cut.
    assert info[1]["primary_truth_pv"] == "Truth_PV_0"
    assert info[1]["primary_truth_pv_weight"] / info[1][
        "w_total_reco"
    ] == pytest.approx(0.60)
    # P_d is Fake because the *plurality* of its tracks are truthless, even
    # though one of them belongs to a real (nTrk = 1) truth vertex.
    assert info[3]["contributions"] == {"Truth_PV_5": 1, "Fake": 3}
    assert info[3]["classification"] == "Fake"


def test_the_two_taxonomies_disagree_by_two(event):
    """The headline of the document: same objects, clean/truth 0.40 vs 0.20."""
    pos = positional_pass(event)
    trk, _ = track_pass(event)
    assert pos["n_reco"] == trk["n_reco"]
    assert pos["n_truth"] == trk["n_truth"]
    assert pos["clean_per_truth"] == pytest.approx(2 * trk["clean_per_truth"])


def test_window_widening_moves_efficiency(event):
    """(W) The window is an input, not a property of the reco list."""
    wide = positional_pass(event, window_mm=0.35)
    assert wide["efficiency"] == pytest.approx(1.00)
    assert (wide["clean"], wide["merged"], wide["split"], wide["fake"]) == (1, 2, 2, 1)


def test_truth_filter_moves_the_fake_rate(event):
    """(N) Dropping the nTrk >= 2 cut turns a 'fake' into a clean match."""
    unfiltered = positional_pass(event, min_truth_ntrk=1)
    assert unfiltered["n_truth"] == 6
    assert unfiltered["fake_per_event"] == pytest.approx(1.0)
    assert unfiltered["efficiency"] == pytest.approx(5 / 6)


def test_reco_labels_are_not_mirror_symmetric():
    """(O) Reco-side clean/merged depends on the z walk order; efficiency does not."""
    out = order_dependence()
    assert (out["nominal"]["clean"], out["nominal"]["merged"]) == (1, 1)
    assert (out["mirrored"]["clean"], out["mirrored"]["merged"]) == (0, 2)
    assert out["nominal"]["efficiency"] == out["mirrored"]["efficiency"] == 1.0
