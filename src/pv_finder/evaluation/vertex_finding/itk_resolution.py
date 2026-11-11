"""Hard-scatter longitudinal vertex resolution, following the ITk paper.

Reference: ATLAS, "Expected tracking performance of the ATLAS Inner Tracker at
the HL-LHC", arXiv:2412.15090 (IDTR-2023-01), Section 5.5 and Figure 28:
"Longitudinal position resolution of the reconstructed primary vertex,
evaluated in ttbar events with <mu> = 200", plotted against the local pile-up
density in vertices/mm.  The paper text does not spell out the estimator.  The
Athena monitoring code that produces this kind of plot
(``InDetPerfPlot_VertexTruthMatching``, histogram
``resolution_vs_PUdensity_hsVxTruthLong``) does the following:

* residual ``z_truth(HS) - z_reco(best HS match)`` per event, filled only when
  the HS has a reconstructed match;
* local pile-up density = number of other truth interactions within 2 mm of the
  HS, divided by 4 mm;
* per density bin, the width is the **iterative RMS** of the residuals
  (``ResolutionHelper::iterRMS_convergence``, the default method): the range is
  reset to mean +- 3 RMS and the RMS recomputed until the range stops changing
  (at most 100 iterations).  The quoted error is ROOT's ``GetRMSError``,
  RMS / sqrt(2 N).

Our ntuples have no vertex-fit truth weights, so "best HS match" cannot be the
track-weight match Athena uses.  :func:`hs_residual` implements the positional
analogue used throughout the note: the reconstructed vertex closest in z to the
true HS, within a candidate radius.  We report reco - truth (the sign does not
affect the width).

Unbinned: the iteration acts on the residual sample itself rather than on a
pre-binned histogram, so there is no dependence on a histogram binning that the
paper does not state.  The starting range (the Athena histogram axis) is an
explicit argument because it is the one free choice of the method.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Athena ResolutionHelper::setIterativeConvergence defaults.
ITER_RMS_N_RMS: float = 3.0
ITER_RMS_MAX_ITER: int = 100


class IterRMS(NamedTuple):
    """Result of an iterative-RMS width estimate."""

    rms: float
    rms_err: float  # RMS / sqrt(2 N_in), as ROOT's TH1::GetRMSError
    mean: float
    mean_err: float  # RMS / sqrt(N_in)
    n_total: int
    n_in: int  # residuals inside the final mean +- n_rms * RMS range
    frac_out: float
    n_iter: int
    converged: bool


_NAN_RESULT = IterRMS(np.nan, np.nan, np.nan, np.nan, 0, 0, np.nan, 0, False)


def iterative_rms(
    residuals: np.ndarray,
    initial_halfrange: float,
    *,
    n_rms: float = ITER_RMS_N_RMS,
    max_iter: int = ITER_RMS_MAX_ITER,
) -> IterRMS:
    """Iterative RMS about the mean, truncated at ``n_rms`` RMS.

    Start from ``|r| <= initial_halfrange``; then repeatedly keep
    ``|r - mean| <= n_rms * rms`` (mean and RMS of the currently kept sample)
    until the kept set no longer changes.  Non-finite residuals are ignored
    (they mark events with no match).
    """
    r = np.asarray(residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    n_total = int(len(r))
    keep = np.abs(r) <= initial_halfrange
    if keep.sum() < 2:
        return _NAN_RESULT._replace(n_total=n_total)
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        sel = r[keep]
        mean, rms = float(sel.mean()), float(sel.std())
        new_keep = np.abs(r - mean) <= n_rms * rms
        if new_keep.sum() < 2:
            break
        if np.array_equal(new_keep, keep):
            converged = True
            break
        keep = new_keep
    sel = r[keep]
    n_in = int(len(sel))
    mean, rms = float(sel.mean()), float(sel.std())
    return IterRMS(
        rms=rms,
        rms_err=rms / np.sqrt(2.0 * n_in),
        mean=mean,
        mean_err=rms / np.sqrt(n_in),
        n_total=n_total,
        n_in=n_in,
        frac_out=1.0 - n_in / n_total if n_total else np.nan,
        n_iter=n_iter,
        converged=converged,
    )


def hs_residual(reco_z: np.ndarray, hs_z: float, max_dz: float) -> float:
    """Signed ``z_reco - z_HS`` of the reco vertex closest to the HS.

    Returns NaN when there is no reconstructed vertex within ``max_dz`` (the HS
    is then "not reconstructed" and does not enter the resolution).
    """
    reco_z = np.asarray(reco_z, dtype=np.float64)
    if len(reco_z) == 0:
        return float("nan")
    d = reco_z - float(hs_z)
    k = int(np.argmin(np.abs(d)))
    return float(d[k]) if abs(d[k]) <= max_dz else float("nan")


def local_pu_density(
    truth_z: np.ndarray, hs_index: int, halfwidth: float = 2.0
) -> float:
    """Other truth vertices with ``|z - z_HS| < halfwidth``, per mm (``2*halfwidth``).

    Matches Athena ``getLocalPUDensity`` (radial window 2 mm).  The caller must
    state which truth population ``truth_z`` contains: our ntuples store only
    interactions with at least one selected track.
    """
    truth_z = np.asarray(truth_z, dtype=np.float64)
    near = np.abs(truth_z - truth_z[hs_index]) < halfwidth
    near[hs_index] = False
    return float(near.sum() / (2.0 * halfwidth))


def binned_iterative_rms(
    x: np.ndarray,
    residuals: np.ndarray,
    edges: np.ndarray,
    initial_halfrange: float,
    **kwargs: float,
) -> list[dict[str, float]]:
    """:func:`iterative_rms` in bins of ``x``; last bin closed on the right."""
    x = np.asarray(x, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    rows = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        upper = x <= hi if j == len(edges) - 2 else x < hi
        sel = (x >= lo) & upper
        res = iterative_rms(residuals[sel], initial_halfrange, **kwargs)
        rows.append({"lo": float(lo), "hi": float(hi), **res._asdict()})
    return rows
