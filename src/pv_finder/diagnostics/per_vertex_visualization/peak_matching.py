"""Peak finding in histograms and truth vertex loading.

Supports matching predicted histogram peaks to ground-truth vertex positions
for both MC (generator-level) and Run 3 (AMVF reconstructed) data.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from pv_finder.data.feature_loading import Z_MAX, Z_MIN

# N_BINS_FULL is not exported from feature_loading; defined here per spec.
N_BINS_FULL = 12000  # 12 sub-events × 1000 bins


def _bin_to_z_mm(bin_idx: int | float) -> float:
    """Convert a 0-based bin index to z in mm."""
    return Z_MIN + (bin_idx + 0.5) / N_BINS_FULL * (Z_MAX - Z_MIN)


def find_histogram_peaks(
    hist_flat: np.ndarray,
    threshold_frac: float = 0.05,
    min_distance_bins: int = 5,
) -> list[tuple[float, float]]:
    """Find peaks in a 12000-bin histogram.

    Uses scipy.signal.find_peaks with:
        height    = threshold_frac * max(hist_flat)
        distance  = min_distance_bins
    Returns list of (z_mm, height) sorted by z_mm.
    Returns empty list when max(hist_flat) == 0.
    """
    global_max = float(np.max(hist_flat))
    if global_max == 0.0:
        return []

    peak_indices, _ = find_peaks(
        hist_flat,
        height=threshold_frac * global_max,
        distance=min_distance_bins,
    )

    peaks = [(_bin_to_z_mm(idx), float(hist_flat[idx])) for idx in peak_indices]
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
    """Load generator-level truth vertex z-positions from H5 'pv' dataset.

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

    Returns list of lists (same length as event_indices).
    """
    import numpy as np  # noqa: PLC0415

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
