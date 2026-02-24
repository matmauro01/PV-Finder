"""
Analytical per-track 2D Gaussian KDE computation.

Reproduces the KDE target that the T2KDE network was trained to predict,
following the algorithm from the ACTS_KDE_generation notebook.

Algorithm (per z-bin):
    1. Filter active tracks whose z0 +/- 3*sigma_z overlaps this z-bin.
    2. Coarse scan: sum 2D Gaussian PDFs (center=(|d0|,z0), covariance from
       track errors) on a 60x60 grid in [-0.6,0.6], r = sqrt(x^2+y^2).
    3. Fine scan: 7x7 grid (+/-3 steps of 0.01) around the coarse best.
    4. KDE[z-bin] = max PDF sum across all coarse + fine scan points.

Pure numpy -- no numba dependency.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from pv_finder.data.feature_loading import (
    MASK_VAL,
    N_SUBEVENTS,
    SUBEVENT_STARTS,
    SUBEVENT_WIDTH,
)

# ---------------------------------------------------------------------------
# Scan-grid constants (match ACTS_KDE_generation notebook)
# ---------------------------------------------------------------------------
_COARSE_N = 60
_COARSE_LO, _COARSE_HI = -0.6, 0.6
_FINE_HALF = 3
_FINE_STEP = 0.01
_SIGMA_CUT = 3.0
_DET_FLOOR = 1e-12


def _build_coarse_grid() -> tuple[NDArray, NDArray, NDArray]:
    """Build 60x60 coarse (x, y, r) scan arrays, each length 3600.

    Uses bin-center formula matching ACTS_KDE_generation notebook:
    center_i = xymin + (i + 0.5) * bin_width, giving points from -0.59 to 0.59.
    """
    bin_width = (_COARSE_HI - _COARSE_LO) / _COARSE_N
    lin = _COARSE_LO + (np.arange(_COARSE_N) + 0.5) * bin_width
    gx, gy = np.meshgrid(lin, lin, indexing="ij")
    x, y = gx.ravel(), gy.ravel()
    return x, y, np.sqrt(x**2 + y**2)


def _build_fine_offsets() -> tuple[NDArray, NDArray]:
    """Build 7x7 fine-scan offsets excluding (0,0) -> 48 points."""
    steps = np.arange(-_FINE_HALF, _FINE_HALF + 1) * _FINE_STEP
    dx, dy = np.meshgrid(steps, steps, indexing="ij")
    dx, dy = dx.ravel(), dy.ravel()
    keep = ~((dx == 0.0) & (dy == 0.0))
    return dx[keep], dy[keep]


_COARSE_X, _COARSE_Y, _COARSE_R = _build_coarse_grid()
_FINE_DX, _FINE_DY = _build_fine_offsets()


# ---------------------------------------------------------------------------
# Core vectorized 2D Gaussian PDF
# ---------------------------------------------------------------------------
def compute_gaussian_pdf_vectorized(
    d0: NDArray,
    z0: NDArray,
    var_d0: NDArray,
    var_z0: NDArray,
    cov_d0z0: NDArray,
    scan_r: NDArray | float,
    scan_z: NDArray | float,
) -> NDArray:
    """Evaluate 2D Gaussian PDF for every (scan-point, track) pair.

    d0, z0, var_d0, var_z0, cov_d0z0: (n_tracks,).
    scan_r, scan_z: scalar or (n_scan,).
    Returns (n_scan, n_tracks).  Singular covariances (det<=1e-12) -> 0.
    """
    scan_r = np.atleast_1d(np.asarray(scan_r, dtype=np.float64))
    scan_z = np.atleast_1d(np.asarray(scan_z, dtype=np.float64))
    center_r = np.abs(d0)
    det = var_d0 * var_z0 - cov_d0z0**2
    valid = det > _DET_FLOOR
    n_scan, n_tracks = len(scan_r), len(d0)
    if not np.any(valid):
        return np.zeros((n_scan, n_tracks), dtype=np.float64)

    inv_det = np.zeros_like(det)
    inv_det[valid] = 1.0 / det[valid]
    # Inv of [[var_d0,cov],[cov,var_z0]] = (1/det)*[[var_z0,-cov],[-cov,var_d0]]
    inv_00 = var_z0 * inv_det
    inv_01 = -cov_d0z0 * inv_det
    inv_11 = var_d0 * inv_det
    norm = np.zeros_like(det)
    norm[valid] = 1.0 / (2.0 * np.pi * np.sqrt(det[valid]))

    dr = scan_r[:, None] - center_r[None, :]
    dz = scan_z[:, None] - z0[None, :]
    quad = inv_00 * dr**2 + 2.0 * inv_01 * dr * dz + inv_11 * dz**2
    pdf = norm[None, :] * np.exp(-0.5 * quad)
    pdf[:, ~valid] = 0.0
    return pdf


# ---------------------------------------------------------------------------
# Per-subevent KDE
# ---------------------------------------------------------------------------
def compute_analytical_kde_subevent(
    d0: NDArray,
    z0: NDArray,
    d0_err: NDArray,
    z0_err: NDArray,
    d0_z0_cov: NDArray,
    z_start: float,
    z_end: float,
    n_bins: int = 1000,
) -> NDArray:
    """Compute analytical 2D-Gaussian KDE for one subevent -> (n_bins,).

    d0_err/z0_err are std devs (squared internally); d0_z0_cov is covariance.
    """
    kde = np.zeros(n_bins, dtype=np.float64)
    if len(d0) == 0:
        return kde

    d0 = np.asarray(d0, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    var_d0 = np.asarray(d0_err, dtype=np.float64) ** 2
    var_z0 = np.asarray(z0_err, dtype=np.float64) ** 2
    cov = np.asarray(d0_z0_cov, dtype=np.float64)
    det = var_d0 * var_z0 - cov**2

    good = det > _DET_FLOOR
    if not np.any(good):
        return kde
    d0, z0 = d0[good], z0[good]
    var_d0, var_z0, cov, det = var_d0[good], var_z0[good], cov[good], det[good]

    center_r, center_z = np.abs(d0), z0
    inv_00, inv_01, inv_11 = var_z0 / det, -cov / det, var_d0 / det
    norm = 1.0 / (2.0 * np.pi * np.sqrt(det))

    # 3-sigma z window per track; sort for sliding-window iteration
    sigma_z = np.sqrt(var_z0)
    trk_z_lo = z0 - _SIGMA_CUT * sigma_z
    trk_z_hi = z0 + _SIGMA_CUT * sigma_z
    order = np.argsort(trk_z_lo)
    center_r, center_z = center_r[order], center_z[order]
    inv_00, inv_01, inv_11, norm = (
        inv_00[order],
        inv_01[order],
        inv_11[order],
        norm[order],
    )
    trk_z_lo, trk_z_hi = trk_z_lo[order], trk_z_hi[order]

    n_good = len(center_r)
    bin_width = (z_end - z_start) / n_bins
    coarse_r = _COARSE_R

    start_ptr = 0
    for bi in range(n_bins):
        z_bin_lo = z_start + bi * bin_width
        z_bin_hi = z_bin_lo + bin_width
        z_val = z_bin_lo + 0.5 * bin_width

        while start_ptr < n_good and trk_z_hi[start_ptr] < z_bin_lo:
            start_ptr += 1
        end_ptr = start_ptr
        while end_ptr < n_good and trk_z_lo[end_ptr] <= z_bin_hi:
            end_ptr += 1
        if end_ptr <= start_ptr:
            continue

        s = slice(start_ptr, end_ptr)
        a_cr, a_cz = center_r[s], center_z[s]
        a_i00, a_i01, a_i11, a_n = inv_00[s], inv_01[s], inv_11[s], norm[s]

        # Coarse scan (3600 x n_active)
        dr = coarse_r[:, None] - a_cr[None, :]
        dz_val = z_val - a_cz
        quad = a_i00 * dr**2 + 2.0 * a_i01 * dr * dz_val + a_i11 * dz_val**2
        sums = (a_n * np.exp(-0.5 * quad)).sum(axis=1)
        best_idx = int(np.argmax(sums))
        best_val = float(sums[best_idx])

        # Fine scan (48 points around coarse best)
        fine_r = np.sqrt(
            (_COARSE_X[best_idx] + _FINE_DX) ** 2
            + (_COARSE_Y[best_idx] + _FINE_DY) ** 2,
        )
        dr_f = fine_r[:, None] - a_cr[None, :]
        quad_f = a_i00 * dr_f**2 + 2.0 * a_i01 * dr_f * dz_val + a_i11 * dz_val**2
        fine_best = float(np.max((a_n * np.exp(-0.5 * quad_f)).sum(axis=1)))
        kde[bi] = max(best_val, fine_best)

    return kde


# ---------------------------------------------------------------------------
# Event-level wrappers
# ---------------------------------------------------------------------------
def compute_analytical_kde_event_mc(tracks_tensor: NDArray) -> NDArray:
    """Compute analytical KDE for one MC event -> (12, 1000).

    tracks_tensor: (12, 7, N_max).  Channels:
    0=d0, 1=z0, 2=d0_err, 3=z0_err, 4=d0_z0_cov, 5=z_start, 6=z_end.
    """
    result = np.zeros((N_SUBEVENTS, 1000), dtype=np.float64)
    for si in range(N_SUBEVENTS):
        sub = tracks_tensor[si]  # (7, N_max)
        valid = sub[1, :] > (MASK_VAL + 1)
        if not np.any(valid):
            continue
        result[si] = compute_analytical_kde_subevent(
            d0=sub[0, valid],
            z0=sub[1, valid],
            d0_err=sub[2, valid],
            z0_err=sub[3, valid],
            d0_z0_cov=sub[4, valid],
            z_start=float(sub[5, valid][0]),
            z_end=float(sub[6, valid][0]),
        )
    return result


def compute_analytical_kde_event_run3(event_dict: dict) -> NDArray:
    """Compute analytical KDE for one Run 3 event -> (12, 1000).

    event_dict keys: z0, d0, d0_err, z0_err, d0_z0_cov (all 1D arrays).
    """
    result = np.zeros((N_SUBEVENTS, 1000), dtype=np.float64)
    z0_all = event_dict["z0"]
    d0_all = event_dict["d0"]
    d0_err_all = event_dict["d0_err"]
    z0_err_all = event_dict["z0_err"]
    cov_all = event_dict["d0_z0_cov"]

    for si in range(N_SUBEVENTS):
        z_start = SUBEVENT_STARTS[si]
        z_end = z_start + SUBEVENT_WIDTH
        mask = (z0_all >= z_start) & (z0_all < z_end)
        if not np.any(mask):
            continue
        result[si] = compute_analytical_kde_subevent(
            d0=d0_all[mask],
            z0=z0_all[mask],
            d0_err=d0_err_all[mask],
            z0_err=z0_err_all[mask],
            d0_z0_cov=cov_all[mask],
            z_start=z_start,
            z_end=z_end,
        )
    return result


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------
def compute_analytical_kdes_batch(
    events: list[dict],
    dataset_type: str,
    n_events: int | None = None,
) -> list[NDArray]:
    """Compute analytical KDE for a batch of events.

    events: list of MC event dicts (key 'tracks') or Run 3 event dicts.
    dataset_type: "mc" or "run3".
    Returns list of (12, 1000) arrays.  Shows tqdm progress bar.
    """
    if dataset_type not in ("mc", "run3"):
        raise ValueError(f"dataset_type must be 'mc' or 'run3', got {dataset_type!r}")

    subset = events[:n_events] if n_events is not None else events
    results: list[NDArray] = []
    for evt in tqdm(subset, desc=f"Analytical KDE ({dataset_type})"):
        if dataset_type == "mc":
            kde = compute_analytical_kde_event_mc(evt["tracks"])
        else:
            kde = compute_analytical_kde_event_run3(evt)
        results.append(kde)
    return results
