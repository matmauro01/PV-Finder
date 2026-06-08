"""Peak finding in histograms and truth vertex loading.

Uses the same peak-finding algorithm as evaluation (pv_locations_updated_res)
to ensure consistent PV detection across the project. Supports matching
predicted peaks to ground-truth vertex positions for both MC (generator-level)
and Run 3 (AMVF reconstructed) data.
"""

from __future__ import annotations

import numpy as np

from pv_finder.data.feature_loading import N_SUBEVENTS, Z_MAX, Z_MIN
from pv_finder.utils.peak_finding import pv_locations_updated_res

# 12 sub-events x 1000 bins per sub-event
N_BINS_FULL = N_SUBEVENTS * 1000


def _bin_to_z_mm(bin_idx: int | float) -> float:
    """Convert a 0-based bin index to z in mm (bin centre)."""
    return Z_MIN + (bin_idx + 0.5) / N_BINS_FULL * (Z_MAX - Z_MIN)


def find_histogram_peaks(
    hist_flat: np.ndarray,
    threshold: float = 0.01,
    integral_threshold: float = 0.5,
    min_width: int = 3,
) -> list[tuple[float, float]]:
    """Find peaks in a 12000-bin histogram using the standard PV-Finder algorithm.

    Delegates to ``pv_locations_updated_res`` (shared with evaluation) which
    scans contiguous above-threshold regions and applies integral and width
    cuts.  Each region yields exactly one peak at the weighted-mean position.

    Returns list of (z_mm, height) sorted by z_mm.
    Returns empty list when the histogram is all zeros.
    """
    if float(np.max(np.abs(hist_flat))) == 0.0:
        return []

    z_pos, heights, *_ = pv_locations_updated_res(
        hist_flat,
        threshold=threshold,
        integral_threshold=integral_threshold,
        min_width=min_width,
    )

    peaks = [(float(z), float(h)) for z, h in zip(z_pos, heights)]
    peaks.sort(key=lambda p: p[0])
    return peaks


def peaks_in_vertex_window(
    pred_peaks: list[tuple[float, float]],
    truth_z: float,
    window_mm: float = 0.5,
) -> list[tuple[float, float]]:
    """Return predicted peaks (z_mm, height) within |z - truth_z| <= window_mm."""
    return [p for p in pred_peaks if abs(p[0] - truth_z) <= window_mm]


def load_mc_truth_vertices(
    h5_path: str,
    n_events: int,
    val_start_event: int = 35700,  # = 428400 // 12
) -> list[list[float]]:
    """Load generator-level truth vertex z-positions from H5 ``pv`` dataset.

    H5 pv dataset shape: (51000, 92), dtype float64.
    Indexed by event: pv[val_start_event : val_start_event + n_events].
    Padding sentinel: -999999.0 -- filter out any value <= -500 (safe margin).
    Returns list of n_events lists, each containing valid z-positions sorted by z.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        pv_block = f["pv"][val_start_event : val_start_event + n_events]

    result: list[list[float]] = []
    for row in pv_block:
        valid = row[row > -500.0]
        result.append(sorted(float(v) for v in valid))
    return result


def classify_vertices(
    truth_vertices: list[float],
    pred_peaks: list[tuple[float, float]],
    match_window_mm: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Classify truth and reco vertices following the eval nomenclature.

    Replicates the logic of ``compare_res_reco`` from
    ``efficiency_res_optimized_atlas.py``, using a fixed matching window in mm
    (consistent with the visual peak marking already used in the plots).

    Uses the SAME greedy closest-first 1-to-1 assignment as
    ``efficiency_res_optimized_atlas.compare_res_reco`` so that two cleanly
    separated but nearby truth vertices (each with its own reco) are counted as
    two "clean" matches, NOT as "merged". A reco is only "merged" when it is the
    best match for one truth AND absorbs an *extra* truth in its window that no
    closer reco claimed.

    Returns
    -------
    truth_labels
        One label per truth vertex: ``"clean"``, ``"merged"``, or ``"missed"``.
    reco_labels
        One label per predicted peak: ``"clean"``, ``"merged"``, ``"split"``,
        or ``"fake"``.
    """
    n_truth = len(truth_vertices)
    n_reco = len(pred_peaks)
    truth_arr = np.asarray(truth_vertices, dtype=float)

    # Candidate (reco, truth, distance) pairs within the matching window
    pairs: list[tuple[int, int, float]] = []
    reco_neighbors: list[list[int]] = [[] for _ in range(n_reco)]
    for i, (pz, _) in enumerate(pred_peaks):
        if n_truth == 0:
            continue
        dists = np.abs(truth_arr - pz)
        for j in np.where(dists <= match_window_mm)[0]:
            pairs.append((i, int(j), float(dists[j])))
            reco_neighbors[i].append(int(j))

    # Greedy closest-first 1-to-1 assignment
    pairs.sort(key=lambda x: x[2])
    reco_assigned: dict[int, int] = {}
    truth_assigned: dict[int, int] = {}
    for i, j, _ in pairs:
        if i not in reco_assigned and j not in truth_assigned:
            reco_assigned[i] = j
            truth_assigned[j] = i

    # Primaries won a dedicated reco in pass 1 -> clean even if their reco later
    # absorbs a neighbour; only absorbed (pass-2) truths are the merge casualties.
    primary_truth = set(truth_assigned)

    reco_labels = ["fake"] * n_reco
    for i in range(n_reco):
        nb = reco_neighbors[i]
        if i not in reco_assigned:
            # truth in window but claimed by a closer reco -> split; else fake
            reco_labels[i] = "split" if nb else "fake"
        else:
            unmatched = [j for j in nb if j not in truth_assigned]
            if unmatched:
                reco_labels[i] = "merged"
                for j in unmatched:
                    truth_assigned[j] = i  # absorbed by this reco
            else:
                reco_labels[i] = "clean"

    truth_labels = ["missed"] * n_truth
    for j in range(n_truth):
        if j in primary_truth:
            truth_labels[j] = "clean"  # has a dedicated reco
        elif j in truth_assigned:
            truth_labels[j] = "merged"  # absorbed by a closer truth's reco

    return truth_labels, reco_labels


def load_run3_amvf_vertices(
    cache_path: str,
    event_indices: list[int],
    min_ntracks: int = 2,
) -> list[list[float]]:
    """Load beam-corrected AMVF vertex z-positions from a Run 3 NPZ cache.

    NPZ keys used:
        RecoVertex_z[i]       -- per-event array of raw AMVF vertex z (mm)
        RecoVertex_nTracks[i] -- per-event array of track counts per vertex
        BeamPosZ[i]           -- beam z position (may be 0-d array or scalar)

    For each event_idx:
        1. beam_z = float(np.atleast_1d(BeamPosZ[event_idx])[0])
        2. z_corr = RecoVertex_z[event_idx] - beam_z
        3. keep vertices where RecoVertex_nTracks[event_idx] >= min_ntracks
        4. sort by z value

    Note: beam correction shifts vertices to the beam frame while track z0
    values (from load_run3_data) remain in the detector frame.  The offset
    is typically O(1 mm) or less and is within the default matching window.

    Returns list of lists (same length as event_indices).
    """
    data = np.load(cache_path, allow_pickle=True)
    reco_z = data["RecoVertex_z"]
    reco_n = data["RecoVertex_nTracks"]
    beam_pos = data["BeamPosZ"]

    result: list[list[float]] = []
    for idx in event_indices:
        beam_z = float(np.atleast_1d(beam_pos[idx])[0])
        z_corr = np.asarray(reco_z[idx], dtype=np.float64) - beam_z
        n_trk = np.asarray(reco_n[idx], dtype=np.int64)
        keep = z_corr[n_trk >= min_ntracks]
        result.append(sorted(float(z) for z in keep))
    return result
