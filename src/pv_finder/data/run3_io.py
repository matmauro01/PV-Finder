"""Run 3 data I/O for PV-Finder evaluation.

Provides two loaders — one for ROOT files (via uproot) and one for
pre-extracted NPZ caches — that return a common ``Run3Event`` named tuple.
Both loaders filter events by minimum track count and AMVF vertex count,
and filter AMVF vertices to ``nTracks >= 2`` before storing.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Run3Event(NamedTuple):
    """Single Run 3 event with tracks, AMVF reco vertices, and optional MC truth.

    All arrays are float32. Lengths of track arrays (z0, d0, ...) are
    equal within an event.  ``amvf_z`` and ``amvf_ntrks`` contain only
    vertices with nTracks >= 2.  ``truth_z`` and ``truth_ntrks`` are
    populated from ``TruthVertex_z``/``TruthVertex_nTracks`` branches
    when available (HL-LHC MC); ``None`` for real data (Run 3 NPZ).
    """

    z0: np.ndarray
    d0: np.ndarray
    d0_err: np.ndarray
    z0_err: np.ndarray
    d0_z0_cov: np.ndarray
    amvf_z: np.ndarray
    amvf_ntrks: np.ndarray
    beam_z: float
    mu: float | None
    event_idx: int
    n_tracks: int
    truth_z: np.ndarray | None = None
    truth_ntrks: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _filter_amvf(
    vtx_z: np.ndarray,
    vtx_ntrks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only AMVF vertices with nTracks >= 2."""
    mask = vtx_ntrks >= 2
    return (
        np.asarray(vtx_z[mask], dtype=np.float32),
        np.asarray(vtx_ntrks[mask], dtype=np.float32),
    )


def _parse_beam_z(raw: object) -> float:
    """Extract a scalar beam-z from potentially weird numpy shapes.

    BeamPosZ may arrive as a 0-d array, a 1-d array with one element,
    or a plain scalar.  This always returns a Python float.
    """
    arr = np.atleast_1d(np.asarray(raw, dtype=np.float64))
    return float(arr[0]) if len(arr) > 0 else 0.0


# ---------------------------------------------------------------------------
# ROOT loader
# ---------------------------------------------------------------------------


def load_run3_from_root(
    root_path: str,
    *,
    max_events: int = 0,
    min_tracks: int = 1,
    min_amvf_vtx: int = 1,
    entry_start: int = 0,
    entry_stop: int | None = None,
) -> list[Run3Event]:
    """Load Run 3 events from a ROOT file via uproot.

    Parameters
    ----------
    root_path : str
        Path to the ROOT file containing the ``PVFinderData`` tree.
    max_events : int, optional
        Maximum number of events to return.  0 means no limit.
    min_tracks : int, optional
        Skip events with fewer reconstructed tracks than this.
    min_amvf_vtx : int, optional
        Skip events with fewer AMVF vertices (after nTracks >= 2 filter).
    entry_start : int, optional
        First TTree entry to read (for memory control).
    entry_stop : int or None, optional
        One-past-last TTree entry to read.  ``None`` means end of tree.

    Returns
    -------
    list[Run3Event]
        Filtered Run 3 events.
    """
    import uproot  # lazy import — only needed for ROOT files

    tree = uproot.open(root_path)["PVFinderData"]
    n_entries = tree.num_entries
    print(f"[run3_io] ROOT: {root_path}  ({n_entries} entries)")

    branches = [
        "RecoTrack_z0",
        "RecoTrack_d0",
        "RecoTrack_ErrD0",
        "RecoTrack_ErrZ0",
        "RecoTrack_ErrD0Z0",
        "RecoVertex_z",
        "RecoVertex_nTracks",
        "BeamPosZ",
        "ActualNumOfInt",
    ]
    # Check for MC truth branches (present in HL-LHC MC, absent in real data)
    available = set(tree.keys())
    has_truth = "TruthVertex_z" in available and "TruthVertex_nTracks" in available
    if has_truth:
        branches += ["TruthVertex_z", "TruthVertex_nTracks"]
        print("[run3_io]   Found TruthVertex branches — loading MC truth vertices")
    else:
        print("[run3_io]   No TruthVertex branches — truth_z will be None")

    events: list[Run3Event] = []
    n_skipped = 0
    n_read = 0
    done = False

    for chunk in tree.iterate(
        branches,
        library="np",
        step_size=500,
        entry_start=entry_start,
        entry_stop=entry_stop,
    ):
        for i in range(len(chunk["RecoTrack_z0"])):
            z0 = np.asarray(chunk["RecoTrack_z0"][i], dtype=np.float32)
            n_read += 1
            if len(z0) < min_tracks:
                n_skipped += 1
                continue

            amvf_z, amvf_ntrks = _filter_amvf(
                np.asarray(chunk["RecoVertex_z"][i], dtype=np.float32),
                np.asarray(chunk["RecoVertex_nTracks"][i], dtype=np.float32),
            )
            if len(amvf_z) < min_amvf_vtx:
                n_skipped += 1
                continue

            # MC truth vertices (nTracks >= 2 filter, same as AMVF)
            truth_z = truth_ntrks = None
            if has_truth:
                truth_z, truth_ntrks = _filter_amvf(
                    np.asarray(chunk["TruthVertex_z"][i], dtype=np.float32),
                    np.asarray(chunk["TruthVertex_nTracks"][i], dtype=np.float32),
                )

            events.append(
                Run3Event(
                    z0=z0,
                    d0=np.asarray(chunk["RecoTrack_d0"][i], dtype=np.float32),
                    d0_err=np.asarray(chunk["RecoTrack_ErrD0"][i], dtype=np.float32),
                    z0_err=np.asarray(chunk["RecoTrack_ErrZ0"][i], dtype=np.float32),
                    d0_z0_cov=np.asarray(
                        chunk["RecoTrack_ErrD0Z0"][i], dtype=np.float32
                    ),
                    amvf_z=amvf_z,
                    amvf_ntrks=amvf_ntrks,
                    beam_z=_parse_beam_z(chunk["BeamPosZ"][i]),
                    mu=float(chunk["ActualNumOfInt"][i]),
                    event_idx=entry_start + n_read - 1,
                    n_tracks=len(z0),
                    truth_z=truth_z,
                    truth_ntrks=truth_ntrks,
                )
            )

            if max_events > 0 and len(events) >= max_events:
                done = True
                break
        if done:
            break

    print(
        f"[run3_io]   Loaded {len(events)} events, "
        f"skipped {n_skipped} "
        f"(min_tracks={min_tracks}, min_amvf_vtx={min_amvf_vtx})"
    )
    return events


# ---------------------------------------------------------------------------
# NPZ loader
# ---------------------------------------------------------------------------


def load_run3_from_npz(
    npz_path: str,
    *,
    max_events: int = 0,
    min_tracks: int = 1,
    min_amvf_vtx: int = 1,
) -> list[Run3Event]:
    """Load Run 3 events from a pre-extracted NPZ cache.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` file (e.g.
        ``data/run3/cache_file3_2000ev_seed42.npz``).
    max_events : int, optional
        Maximum number of events to return.  0 means no limit.
    min_tracks : int, optional
        Skip events with fewer reconstructed tracks than this.
    min_amvf_vtx : int, optional
        Skip events with fewer AMVF vertices (after nTracks >= 2 filter).

    Returns
    -------
    list[Run3Event]
        Filtered Run 3 events.

    Notes
    -----
    NPZ files use ``allow_pickle=True`` because arrays are variable-length
    (object dtype).  ``ActualNumOfInt`` is not stored in the NPZ format, so
    ``mu`` is always ``None``.  ``BeamPosZ`` may be a 0-d array, 1-d array,
    or scalar; all cases are handled.
    """
    print(f"[run3_io] NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    z0_all = data["RecoTrack_z0"]
    d0_all = data["RecoTrack_d0"]
    err_d0_all = data["RecoTrack_ErrD0"]
    err_z0_all = data["RecoTrack_ErrZ0"]
    err_d0z0_all = data["RecoTrack_ErrD0Z0"]
    vtx_z_all = data["RecoVertex_z"]
    vtx_ntrks_all = data["RecoVertex_nTracks"]
    beam_z_all = data["BeamPosZ"]

    n_total = len(z0_all)
    print(f"[run3_io]   {n_total} entries in file")

    events: list[Run3Event] = []
    n_skipped = 0

    for i in range(n_total):
        z0 = np.asarray(z0_all[i], dtype=np.float32)
        if len(z0) < min_tracks:
            n_skipped += 1
            continue

        amvf_z, amvf_ntrks = _filter_amvf(
            np.asarray(vtx_z_all[i], dtype=np.float32),
            np.asarray(vtx_ntrks_all[i], dtype=np.float32),
        )
        if len(amvf_z) < min_amvf_vtx:
            n_skipped += 1
            continue

        events.append(
            Run3Event(
                z0=z0,
                d0=np.asarray(d0_all[i], dtype=np.float32),
                d0_err=np.asarray(err_d0_all[i], dtype=np.float32),
                z0_err=np.asarray(err_z0_all[i], dtype=np.float32),
                d0_z0_cov=np.asarray(err_d0z0_all[i], dtype=np.float32),
                amvf_z=amvf_z,
                amvf_ntrks=amvf_ntrks,
                beam_z=_parse_beam_z(beam_z_all[i]),
                mu=None,
                event_idx=i,
                n_tracks=len(z0),
            )
        )

        if max_events > 0 and len(events) >= max_events:
            break

    print(
        f"[run3_io]   Loaded {len(events)} events, "
        f"skipped {n_skipped} "
        f"(min_tracks={min_tracks}, min_amvf_vtx={min_amvf_vtx})"
    )
    return events
