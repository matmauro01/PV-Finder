"""Reconcile AMVF HS selection efficiency (note 80.7%) with ITk Fig. 27 (~92-93%).

One definitional change at a time, on the same events, AMVF stored track
lists only (no PV-Finder, no GNN). Event-level flags are computed once per
event; the reconciliation table is a sequence of cumulative changes.

Definitions (ITk paper arXiv:2412.15090, Sec. 4.2 and 5.5; Athena
InDetPerfPlot_VertexTruthMatching / InDetVertexTruthMatchUtils):
  selected vertex: largest sum(pT^2) of its tracks (in Athena, the stored
      primary vertex, RecoVertex_type == 1 in our ntuple).
  HS-matched vertex: the vertex with the largest number of tracks matched to
      HS particles (bestHardScatterMatch).
  selection efficiency: selected == HS-matched; "reconstructed and selected"
      additionally needs |z_selected - z_HS| < 0.1 mm.

Samples:
  itk:   r16443 flat-mu ITk ttbar, entries 0-24999 (ROOT only).
  lowmu: 51k ttbar mu 0-80 ID sample (ROOT + event-keyed HDF5 with the
         AMVF track lists), aligned by entry and checked event by event.

Usage (repo root, tmux):
    PYTHONPATH=src python -u src/gnn/diagnostics/hs_selection_reconciliation.py \\
        itk --out outputs/11_09_2026_output/hs_selection_reconciliation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot

from gnn.evaluation import hs_metrics as hm

REPO = Path(__file__).resolve().parents[3]
ITK_ROOT = REPO / (
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"
)
LOWMU_ROOT = REPO / "data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root"
LOWMU_H5 = Path("/share/lazy/qibinlei/recoTracks_incamvfassoc.h5")
PT_CAPS_GEV = (10.0, 15.0, 20.0, 50.0)
FLAGS = (
    "rerank_plurality",  # note: max sum(pT^2) of stored links, dominant truth == HS
    "rerank_mosths",  # re-ranked, ITk criterion (most HS tracks)
    "rerank_mosths_0p1",  # ... and |dz| < 0.1 mm
    "native_plurality",  # stored primary, dominant truth == HS
    "native_mosths",  # stored primary, ITk criterion
    "native_mosths_0p1",  # ... and |dz| < 0.1 mm  (ITk Fig. 27 analogue)
    "hs_reco_0p1",  # any AMVF vertex (>= 2 tracks) within 0.1 mm of the HS
    "native_is_rerank",  # stored primary == re-ranked leader
    *(f"cap{int(c)}_is_native" for c in PT_CAPS_GEV),
    "eta25_is_native",  # re-rank using only |eta| < 2.5 tracks
    "leader_has_100gev",  # re-ranked leader holds a track with pT > 100 GeV
    "cap15_mosths_0p1",  # re-ranked with pT capped at 15 GeV, ITk criterion, 0.1 mm
)


def labelled_plurality(tracks: np.ndarray, truth: np.ndarray) -> int:
    """Most common truth label among labelled tracks (lowest index on ties).

    This is the rule of ``gnn.evaluation.hs_id_pu200.dominant_truth`` behind
    the note's 80.7%: unlabelled tracks are ignored, unlike the vertex
    classification, where they form their own group.
    """
    labels = truth[tracks]
    labels = labels[labels >= 0]
    if len(labels) == 0:
        return -1
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def event_flags(
    lists: list[np.ndarray],
    pt_gev: np.ndarray,
    eta: np.ndarray,
    truth: np.ndarray,
    hs: int,
    hs_z: float,
    vertex_z: np.ndarray,
    native: int,
) -> np.ndarray:
    """All reconciliation flags for one event (AMVF vertices with >= 2 tracks)."""
    keep = np.array([len(x) >= 2 for x in lists], dtype=bool)
    if not keep[native]:
        raise ValueError("stored primary has fewer than two tracks")
    idx = np.flatnonzero(keep)
    lists = [lists[i] for i in idx]
    vertex_z = vertex_z[idx]
    native = int(np.flatnonzero(idx == native)[0])
    cls = hm.classify_vertices(lists, pt_gev, truth)
    nhs = hm.hs_track_counts(lists, cls, truth, hs)
    # ITk criterion with ties accepted (Athena keeps the first of tied
    # vertices in its sum(pT^2)-sorted container; any tied vertex may be it).
    best = nhs.max() if len(nhs) else 0

    def is_match(v: int) -> bool:
        return best > 0 and nhs[v] == best

    lead = hm.selected_index(cls)

    def lead_with(weights: np.ndarray) -> int:
        return int(np.argmax([(weights[x] ** 2).sum() for x in lists]))

    near = np.abs(vertex_z - hs_z) < 0.1
    cap15 = lead_with(np.minimum(pt_gev, 15.0))
    flags = [
        labelled_plurality(lists[lead], truth) == hs,
        is_match(lead),
        is_match(lead) and near[lead],
        labelled_plurality(lists[native], truth) == hs,
        is_match(native),
        is_match(native) and near[native],
        near.any(),
        native == lead,
        *(lead_with(np.minimum(pt_gev, c)) == native for c in PT_CAPS_GEV),
        lead_with(np.where(np.abs(eta) < 2.5, pt_gev, 0.0)) == native,
        pt_gev[lists[lead]].max() > 100,
        is_match(cap15) and near[cap15],
    ]
    return np.array(flags, dtype=np.int64)


def run_itk(n_events: int) -> tuple[np.ndarray, np.ndarray]:
    """Flags and mu for entries [0, n_events) of the ITk flat-mu sample."""
    tree = uproot.open(ITK_ROOT)["PVFinderData"]
    branches = [
        "ActualNumOfInt", "RecoTrack_pT", "RecoTrack_eta", "TruthVertex_z",
        "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type",
        "RecoVertex_z", "RecoVertex_type", "RecoVertex_assocTracks",
    ]  # fmt: skip
    out, mus = [], []
    for start in range(0, n_events, 2500):
        data = tree.arrays(
            branches, entry_start=start, entry_stop=min(n_events, start + 2500)
        )
        for ev in data:
            pt = ak.to_numpy(ev["RecoTrack_pT"]).astype(np.float64) / 1000
            tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
            truth = hm.track_truth_labels(
                ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, len(pt)
            )
            hs = int(np.flatnonzero(ak.to_numpy(ev["TruthVertex_type"]) == 1)[0])
            native = np.flatnonzero(ak.to_numpy(ev["RecoVertex_type"]) == 1)
            if len(native) != 1:
                raise ValueError("expected exactly one stored primary")
            lists = [
                np.unique(ak.to_numpy(x).astype(np.int64))
                for x in ev["RecoVertex_assocTracks"]
            ]
            out.append(
                event_flags(
                    lists, pt, ak.to_numpy(ev["RecoTrack_eta"]), truth, hs,
                    float(ev["TruthVertex_z"][hs]),
                    ak.to_numpy(ev["RecoVertex_z"]).astype(np.float64), int(native[0]),
                )
            )  # fmt: skip
            mus.append(float(ev["ActualNumOfInt"]))
        print(f"itk {start + 2500}/{n_events}", flush=True)
    return np.array(out), np.array(mus)


def run_lowmu(n_events: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Flags for the mu 0-80 ID ttbar sample.

    Tracks, truth vertices (with ``pv_type``) and AMVF vertices (with
    ``reco_pv_type`` and track lists) come from the event-keyed HDF5. The
    ROOT entry supplies mu; track pT and AMVF z are checked against it to
    confirm the EventN <-> entry N alignment. The HDF5 truth-vertex order
    differs from the ROOT order, so ROOT truth branches are not used.
    """
    tree = uproot.open(LOWMU_ROOT)["PVFinderData"]
    branches = ["ActualNumOfInt", "RecoTrack_pT", "RecoVertex_z", "RecoVertex_nTracks"]
    out, mus = [], []
    checks = {"skipped_no_hs": 0, "skipped_no_native": 0, "events": 0,
              "vertices_ntracks_differs_from_root": 0, "vertices": 0}  # fmt: skip
    with h5py.File(LOWMU_H5, "r") as f:
        for start in range(0, n_events, 2500):
            stop = min(n_events, start + 2500)
            data = tree.arrays(branches, entry_start=start, entry_stop=stop)
            for k, ev in enumerate(data):
                key = f"Event{start + k}"
                pt = f["recoTrk_pt"][key][:].astype(np.float64)
                np.testing.assert_allclose(
                    pt, ak.to_numpy(ev["RecoTrack_pT"]), rtol=1e-5, err_msg=key
                )
                rz = f["reco_pv_loc_z"][key][:].astype(np.float64)
                np.testing.assert_allclose(
                    rz, ak.to_numpy(ev["RecoVertex_z"]), atol=1e-4, err_msg=key
                )
                rnt = f["reco_pv_ntracks"][key][:].astype(np.int64)
                root_nt = ak.to_numpy(ev["RecoVertex_nTracks"]).astype(np.int64)
                checks["vertices"] += len(rnt)
                checks["vertices_ntracks_differs_from_root"] += int(
                    (rnt != root_nt).sum()
                )
                truth = hm.track_truth_labels(
                    f["pv_assoc_tracks"][key][:], f["pv_ntracks"][key][:], len(pt)
                )
                tz = f["pv_loc_z"][key][:].astype(np.float64)
                hs_c = np.flatnonzero(f["pv_type"][key][:] == 1)
                native = np.flatnonzero(f["reco_pv_type"][key][:] == 1)
                if len(hs_c) != 1:
                    checks["skipped_no_hs"] += 1
                    continue
                if len(native) != 1 or rnt[native[0]] < 2:
                    checks["skipped_no_native"] += 1
                    continue
                flat = f["reco_pv_assoc_tracks"][key][:].astype(np.int64)
                if flat.size != rnt.sum():
                    raise ValueError(f"{key}: AMVF association length mismatch")
                lists = [np.unique(x) for x in np.split(flat, np.cumsum(rnt)[:-1])]
                eta = -np.log(np.tan(f["recoTrk_theta"][key][:].astype(np.float64) / 2))
                out.append(
                    event_flags(
                        lists, pt / 1000, eta, truth, int(hs_c[0]),
                        float(tz[hs_c[0]]), rz, int(native[0]),
                    )
                )  # fmt: skip
                mus.append(float(ev["ActualNumOfInt"]))
                checks["events"] += 1
            print(f"lowmu {stop}/{n_events}", flush=True)
    return np.array(out), np.array(mus), checks


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sample", choices=("itk", "lowmu"))
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-events", type=int, default=None)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.sample == "itk":
        flags, mu = run_itk(args.n_events or 25000)
        checks: dict = {}
    else:
        flags, mu, checks = run_lowmu(args.n_events or 51000)
    np.savez_compressed(args.out / f"flags_{args.sample}.npz", flags=flags, mu=mu)
    meta = {"flags": FLAGS, "checks": checks, "pt_caps_gev": PT_CAPS_GEV,
            "root": str(ITK_ROOT if args.sample == "itk" else LOWMU_ROOT)}  # fmt: skip
    (args.out / f"flags_{args.sample}.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print("done", flush=True)


if __name__ == "__main__":
    main()
