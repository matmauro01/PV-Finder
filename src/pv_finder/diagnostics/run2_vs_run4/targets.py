"""Measure the training-target peak widths of the two PV-Finder models.

The Run 2 reproduction model and the HL-LHC v6 model were trained on target
histograms whose Gaussian widths come from *different* resolution models.  The
HL-LHC h5 records its (A, B, C) in its attributes; the Run 2 h5 records nothing,
so its widths are measured here directly from the stored targets: for isolated
truth vertices (no other stored truth vertex within ``iso_mm``), the RMS of the
nTrk >= 2 target channel about the vertex position over +-``half_mm``.

The truth nTracks of each vertex is attached by exact position match to the
ROOT ntuple the h5 was converted from.
"""

from __future__ import annotations

import h5py
import numpy as np

Z_MIN_MM: float = -240.0
BIN_MM: float = 0.04
N_BINS: int = 12000


def _peak_rms(target: np.ndarray, z: float, half_mm: float) -> tuple[float, float]:
    """Central RMS (mm, about the target's own centroid) and integral within +-half_mm.

    The RMS is taken about the target centroid, not ``z``, so a half-bin
    convention offset in the h5 cannot inflate it.
    """
    c = (z - Z_MIN_MM) / BIN_MM
    lo = max(int(np.floor(c - half_mm / BIN_MM)), 0)
    hi = min(int(np.ceil(c + half_mm / BIN_MM)) + 1, N_BINS)
    x = (np.arange(lo, hi) + 0.5) * BIN_MM + Z_MIN_MM
    w = np.asarray(target[lo:hi], dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return float("nan"), 0.0
    m = float(np.sum(w * x) / s)
    return float(np.sqrt(np.sum(w * (x - m) ** 2) / s)), float(s)


def measure_target_widths(
    h5_path: str,
    events: list[int],
    truth_z: list[np.ndarray],
    truth_ntrk: list[np.ndarray],
    *,
    split: bool,
    iso_mm: float = 1.5,
    half_mm: float = 0.6,
) -> dict[str, np.ndarray]:
    """Per isolated truth vertex (nTrk >= 2): nTracks, target RMS, target integral.

    ``truth_z`` / ``truth_ntrk`` are the ROOT truth lists of the same events, in
    the same order as ``events`` (h5 event indices).  ``split`` selects the
    per-subevent layout (``target_y_split``) instead of the full ``target_y``.
    The bin-centre convention of the target is +0.5 bin (bin i covers
    [z_min + i*bw, z_min + (i+1)*bw)).
    """
    nt, rms, integ = [], [], []
    with h5py.File(h5_path, "r") as f:
        for e, tz, tn in zip(events, truth_z, truth_ntrk):
            if split:
                t = f["target_y_split"][12 * e : 12 * e + 12, 0, :].reshape(-1)
            else:
                t = f["target_y"][e, 0, :]
            tz = np.asarray(tz, dtype=np.float64)
            for k in np.flatnonzero(np.asarray(tn) >= 2):
                d = np.abs(tz - tz[k])
                d[k] = np.inf
                if d.min() < iso_mm or abs(tz[k]) > 235.0:
                    continue
                r, s = _peak_rms(t, float(tz[k]), half_mm)
                nt.append(int(tn[k]))
                rms.append(r)
                integ.append(s)
    return {
        "ntrk": np.asarray(nt),
        "rms_mm": np.asarray(rms),
        "integral": np.asarray(integ),
    }


def model_rms(
    ntrk: np.ndarray, a: float, b: float, c: float, half_mm: float = 0.6
) -> np.ndarray:
    """RMS a target built with sigma = A n^-B + C would have, binned like the h5.

    Binned at 0.04 mm with the vertex at a random sub-bin offset, averaged, so it
    is comparable with :func:`measure_target_widths` (binning adds ~bw^2/12 in
    quadrature for wide peaks, less for peaks narrower than a bin).
    """
    from scipy.special import erf

    rng = np.random.default_rng(0)
    out = []
    for n in np.asarray(ntrk):
        s = max(a * float(n) ** (-b) + c, 0.002)
        vals = []
        for off in rng.uniform(0, BIN_MM, 16):
            edges = np.arange(-half_mm - BIN_MM, half_mm + 2 * BIN_MM, BIN_MM) - off
            cdf = 0.5 * (1 + erf(edges / (np.sqrt(2) * s)))
            w = np.diff(cdf)
            x = 0.5 * (edges[:-1] + edges[1:])
            keep = np.abs(x) <= half_mm + BIN_MM / 2
            w, x = w[keep], x[keep]
            m = np.sum(w * x) / w.sum()
            vals.append(np.sqrt(np.sum(w * (x - m) ** 2) / w.sum()))
        out.append(float(np.mean(vals)))
    return np.asarray(out)


def truth_vertex_precision(root: str, n_entries: int) -> dict[str, np.ndarray]:
    """Ideal z precision of each truth vertex from its own reconstructed tracks.

    ``sigma_ideal = (sum_i 1 / ErrZ0_i^2)^(-1/2)`` over the tracks the ntuple
    associates with the truth vertex (``TruthVertex_assocTracks``).  This is the
    best z uncertainty any algorithm could assign to that vertex from these
    tracks, defined identically in both ntuples, so it isolates what the
    detector (and the ntuple track selection) hands to the vertexers.  It ignores
    track-to-track correlations and the beam constraint.
    """
    import uproot

    tree = uproot.open(root)["PVFinderData"]
    a = tree.arrays(
        ["TruthVertex_assocTracks", "TruthVertex_type", "TruthVertex_nTracks",
         "RecoTrack_ErrZ0", "ActualNumOfInt"],
        entry_stop=n_entries, library="np",
    )  # fmt: skip
    sig, typ, ntrk, mu = [], [], [], []
    for e in range(len(a["ActualNumOfInt"])):
        err = np.asarray(a["RecoTrack_ErrZ0"][e], dtype=np.float64)
        for v, trks in enumerate(a["TruthVertex_assocTracks"][e]):
            idx = np.asarray(trks, dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < len(err))]
            if len(idx) == 0:
                continue
            sig.append(float(np.sum(err[idx] ** -2.0) ** -0.5))
            typ.append(int(a["TruthVertex_type"][e][v]))
            ntrk.append(int(a["TruthVertex_nTracks"][e][v]))
            mu.append(float(a["ActualNumOfInt"][e]))
    return {"sigma_ideal_mm": np.asarray(sig), "type": np.asarray(typ),
            "ntrk": np.asarray(ntrk), "mu": np.asarray(mu)}  # fmt: skip
