"""QA for the extended-|eta| HL-LHC PU200 re-production (``data/run4_all_etas/``).

The previous PU200 ntuples were written with reconstructed tracks truncated at
``|eta| < 2.5``; the July-2026 re-production keeps the full ITk acceptance
(``|eta| < 4``). Before rebuilding the training HDF5s we want to confirm that
(a) nothing else changed, and (b) the extra forward tracks do not break any
assumption baked into ``root_to_h5.py`` -- above all the padded tracks-tensor
width ``--max-tracks-per-sub`` (default 1024).

Run from the repo root with the venv active::

    python -u src/pv_finder/diagnostics/all_etas_data_qa.py \
        --events 20000 \
        --output-dir outputs/07_31_2026_output/all_etas_qa

Outputs five PNGs plus ``summary.json``:

``track_kinematics.png``   per-track distributions, normalised per event
``event_level.png``        multiplicities, pileup, beam spot, truth vertices
``eta_profiles.png``       track density and resolutions vs ``|eta|``
``subevent_occupancy.png`` tracks per 40 mm sub-event vs the padding width
``timing.png``             HGTD timing acceptance and values
"""

from __future__ import annotations

import argparse
import json
import os
from typing import NamedTuple

import numpy as np
import uproot

from pv_finder.diagnostics.all_etas_qa_plots import (
    N_SUBEVENTS,
    PT_TO_GEV,
    SUBEVENT_WIDTH,
    TIME_SENTINEL,
    Z_MIN,
    PlotSpec,
    SampleData,
    _event,
    _grid,
    _safe,
    _significance,
    _timed,
    _track,
    _vertex,
    plot_eta_profiles,
    plot_subevent_occupancy,
)

TREE = "PVFinderData"
DEFAULT_MAX_TRACKS = 1024  # root_to_h5.py --max-tracks-per-sub default

NEW_DEFAULT = (
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root"
)
OLD_DEFAULT = (
    "data/run4/PU200_withTiming/ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root"
)

TRACK_BRANCHES = [
    "RecoTrack_d0",
    "RecoTrack_z0",
    "RecoTrack_pT",
    "RecoTrack_eta",
    "RecoTrack_phi",
    "RecoTrack_theta",
    "RecoTrack_ErrD0",
    "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0",
    "RecoTrack_Time",
    "RecoTrack_TimeResolution",
]
EVENT_BRANCHES = [
    "NumRecoTrks",
    "NumRecoVtx",
    "NumTruthVtx",
    "ActualNumOfInt",
    "BeamPosZ",
    "BeamPosSigmaZ",
]
VERTEX_BRANCHES = ["TruthVertex_z", "TruthVertex_nTracks", "RecoVertex_z"]


class Sample(NamedTuple):
    """One overlaid curve: a label, a ROOT path and a colour."""

    label: str
    path: str
    color: str


def load_sample(sample: Sample, n_events: int) -> SampleData:
    """Read the first ``n_events`` events of one file into flat arrays."""
    tree = uproot.open(sample.path)[TREE]
    avail = set(tree.keys())
    stop = min(n_events, tree.num_entries)
    want = sorted(
        {b for b in TRACK_BRANCHES + EVENT_BRANCHES + VERTEX_BRANCHES if b in avail}
    )
    arrs = tree.arrays(want, entry_stop=stop, library="np")

    def flat(name: str) -> np.ndarray:
        if name not in arrs:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(
            [np.asarray(x, dtype=np.float32) for x in arrs[name]]
            or [np.empty(0, dtype=np.float32)]
        )

    track = {b: flat(b) for b in TRACK_BRANCHES}
    event = {
        b: np.asarray(arrs[b], dtype=np.float32) for b in EVENT_BRANCHES if b in arrs
    }
    vertex = {b: flat(b) for b in VERTEX_BRANCHES}
    sub_counts = _subevent_counts(arrs["RecoTrack_z0"])

    print(
        f"  loaded {sample.label:28s}: {stop:7d} events, "
        f"{len(track['RecoTrack_z0']):>11d} tracks"
    )
    return SampleData(
        sample.label, sample.color, track, event, vertex, sub_counts, stop
    )


def _subevent_counts(z0_jagged: np.ndarray) -> np.ndarray:
    """Tracks per 40 mm sub-event, using the same binning as ``root_to_h5.py``.

    Returns one entry per (event, sub-event) pair. Tracks outside
    ``[Z_MIN, Z_MIN + 12*40)`` are dropped, exactly as the converter does.
    """
    edges = Z_MIN + SUBEVENT_WIDTH * np.arange(N_SUBEVENTS + 1)
    counts = np.empty((len(z0_jagged), N_SUBEVENTS), dtype=np.int32)
    for i, z0 in enumerate(z0_jagged):
        counts[i], _ = np.histogram(np.asarray(z0, dtype=np.float64), bins=edges)
    return counts.reshape(-1)


# --- figures ---------------------------------------------------------------

TRACK_PLOTS = [
    PlotSpec("d0", "d0 [mm]", _track("RecoTrack_d0"), per_event=True),
    PlotSpec("z0", "z0 [mm]", _track("RecoTrack_z0"), per_event=True),
    PlotSpec(
        "pT", "pT [GeV]", _track("RecoTrack_pT", PT_TO_GEV), log_x=True, log_y=True
    ),
    PlotSpec("eta", "eta", _track("RecoTrack_eta"), per_event=True),
    PlotSpec("phi", "phi [rad]", _track("RecoTrack_phi"), per_event=True),
    PlotSpec("theta", "theta [rad]", _track("RecoTrack_theta"), per_event=True),
    PlotSpec("ErrD0", "ErrD0 [mm]", _track("RecoTrack_ErrD0"), log_y=True),
    PlotSpec("ErrZ0", "ErrZ0 [mm]", _track("RecoTrack_ErrZ0"), log_y=True),
    PlotSpec("ErrD0Z0 (covariance)", "ErrD0Z0 [mm^2]", _track("RecoTrack_ErrD0Z0")),
    PlotSpec("d0 significance", "d0 / ErrD0", _significance),
]

EVENT_PLOTS = [
    PlotSpec("Reco tracks / event", "NumRecoTrks", _event("NumRecoTrks")),
    PlotSpec("Reco vertices / event", "NumRecoVtx", _event("NumRecoVtx")),
    PlotSpec("Truth vertices / event", "NumTruthVtx", _event("NumTruthVtx")),
    PlotSpec("Pileup", "ActualNumOfInt", _event("ActualNumOfInt")),
    PlotSpec("Beam spot z", "BeamPosZ [mm]", _event("BeamPosZ")),
    PlotSpec("Beam spot sigma_z", "BeamPosSigmaZ [mm]", _event("BeamPosSigmaZ")),
    PlotSpec(
        "Truth vertex z", "TruthVertex_z [mm]", _vertex("TruthVertex_z"), per_event=True
    ),
    PlotSpec(
        "Truth vertex nTracks",
        "TruthVertex_nTracks",
        _vertex("TruthVertex_nTracks"),
        log_y=True,
        per_event=True,
    ),
    PlotSpec(
        "Reco (AMVF) vertex z",
        "RecoVertex_z [mm]",
        _vertex("RecoVertex_z"),
        per_event=True,
    ),
]

TIMING_PLOTS = [
    PlotSpec("Track time (timed tracks)", "Time [ns]", _timed("RecoTrack_Time")),
    PlotSpec(
        "Time resolution (timed tracks)",
        "TimeResolution [ps]",
        _timed("RecoTrack_TimeResolution", 1.0e3),
    ),
]


def summarise(samples: list[SampleData], max_tracks: int) -> dict:
    """Numbers worth checking by eye, and the converter-critical maxima."""
    out: dict[str, dict] = {}
    for s in samples:
        eta = _safe(s.track.get("RecoTrack_eta"))
        t = s.track.get("RecoTrack_Time")
        timed = float(np.mean(t > TIME_SENTINEL)) if t is not None and len(t) else 0.0
        c = s.sub_counts
        rec = {
            "n_events": s.n_events,
            "n_tracks": int(len(eta)),
            "tracks_per_event": float(len(eta) / max(s.n_events, 1)),
            "abs_eta_max": float(np.abs(eta).max()) if len(eta) else 0.0,
            "abs_eta_p99": float(np.percentile(np.abs(eta), 99)) if len(eta) else 0.0,
            "frac_tracks_abs_eta_gt_2p5": (
                float(np.mean(np.abs(eta) > 2.5)) if len(eta) else 0.0
            ),
            "timed_track_fraction": timed,
            "max_tracks_per_subevent": int(c.max()) if len(c) else 0,
            "p99_9_tracks_per_subevent": (
                float(np.percentile(c, 99.9)) if len(c) else 0.0
            ),
            "subevents_over_max_tracks": int((c > max_tracks).sum()) if len(c) else 0,
            "frac_subevents_over_max_tracks": (
                float(np.mean(c > max_tracks)) if len(c) else 0.0
            ),
        }
        for br in ("NumRecoTrks", "NumRecoVtx", "NumTruthVtx", "ActualNumOfInt"):
            v = s.event.get(br)
            if v is not None and len(v):
                rec[f"mean_{br}"] = float(np.mean(v))
                rec[f"max_{br}"] = float(np.max(v))
        ntrk = _safe(s.vertex.get("TruthVertex_nTracks"))
        if len(ntrk):
            rec["truth_vtx_ntracks_mean"] = float(np.mean(ntrk))
            rec["truth_vtx_frac_ntrk_ge2"] = float(np.mean(ntrk >= 2))
        out[s.label] = rec
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-file", default=NEW_DEFAULT, help="extended-|eta| ROOT file")
    p.add_argument(
        "--old-file",
        default=OLD_DEFAULT,
        help="matching |eta|<2.5 ROOT file (reference); pass '' to skip",
    )
    p.add_argument("--events", type=int, default=20000, help="events read per file")
    p.add_argument(
        "--max-tracks",
        type=int,
        default=DEFAULT_MAX_TRACKS,
        help="padded tracks-tensor width to test against",
    )
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    specs = [Sample("new: all eta (|eta|<4)", args.new_file, "#d62728")]
    if args.old_file:
        specs.append(Sample("old: |eta|<2.5", args.old_file, "#1f77b4"))

    print("Loading samples...")
    samples = [load_sample(s, args.events) for s in specs]

    print("Plotting...")
    _grid(
        TRACK_PLOTS,
        samples,
        "Track parameters",
        os.path.join(args.output_dir, "track_kinematics.png"),
        ncols=4,
    )
    _grid(
        EVENT_PLOTS,
        samples,
        "Event-level quantities",
        os.path.join(args.output_dir, "event_level.png"),
        ncols=3,
    )
    _grid(
        TIMING_PLOTS,
        samples,
        "HGTD timing",
        os.path.join(args.output_dir, "timing.png"),
        ncols=2,
    )
    plot_eta_profiles(samples, os.path.join(args.output_dir, "eta_profiles.png"))
    plot_subevent_occupancy(
        samples,
        os.path.join(args.output_dir, "subevent_occupancy.png"),
        args.max_tracks,
    )

    summary = summarise(samples, args.max_tracks)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote {args.output_dir}/summary.json\n")
    for label, rec in summary.items():
        print(f"--- {label}")
        for k, v in rec.items():
            print(f"    {k:36s} {v}")


if __name__ == "__main__":
    main()
