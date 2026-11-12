"""Helpers for the HL-LHC density-model freeze comparison (Nov 2026, step 2a).

Pure functions, independent of the production matcher, used to cross-check
the saved evaluations of the freeze candidates (v6, s5, s8):

- :func:`truth_found_any` is the paper's Eq. (1) numerator computed from
  positions only: a truth vertex counts as found if at least one
  reconstructed vertex lies within the window. With closest-first matching
  this equals the code's merged-credit numerator (clean + merged truth); the
  comparison script asserts it per event.
- :func:`hs_found` is the hard-scatter reconstruction criterion (ITk paper
  Sec. 5.5): some reconstructed vertex within the HS window of the true HS z.
- :func:`check_bookkeeping` asserts the category identities of one event.
- :func:`paired_ratio_diff` gives a ratio-of-sums difference between two
  arms with the same event resampling applied to both.
- :func:`clopper_pearson` is the exact binomial interval.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import beta


def truth_found_any(truth_z: np.ndarray, pred_z: np.ndarray, window_mm: float) -> int:
    """Number of truth vertices with at least one prediction within the window.

    The window is inclusive (|dz| <= window), as in the production matcher.
    The inputs keep their dtype: to reproduce the matcher exactly at the
    window edge, pass the same arrays it receives (float32 bin units,
    ``(z - Z_MIN) / BIN_WIDTH``) and the window in bins; the distance is then
    computed with the same arithmetic as ``compare_res_reco``.
    """
    t = np.asarray(truth_z)
    p = np.asarray(pred_z)
    if len(t) == 0 or len(p) == 0:
        return 0
    p = np.sort(p)
    i = np.searchsorted(p, t)
    left = np.abs(t - p[np.clip(i - 1, 0, len(p) - 1)])
    right = np.abs(p[np.clip(i, 0, len(p) - 1)] - t)
    return int(np.sum(np.minimum(left, right) <= window_mm))


def hs_found(pred_z: np.ndarray, hs_z: float, window_mm: float) -> bool:
    """True if some prediction lies strictly within the window of the true HS z."""
    p = np.asarray(pred_z, dtype=np.float64)
    return bool(len(p) and np.min(np.abs(p - hs_z)) < window_mm)


def check_bookkeeping(rec: Mapping[str, float], n_pred: int, n_truth: int) -> None:
    """Assert the per-event category identities; raise AssertionError if not.

    Reco side: clean + merged + split + fake = number of predictions.
    Truth side: truth clean + truth merged + missed = number of truth vertices.
    """
    reco = rec["clean"] + rec["merged"] + rec["split"] + rec["fake"]
    if reco != n_pred:
        raise AssertionError(f"event {rec.get('event_idx')}: reco {reco} != {n_pred}")
    truth = rec["tc"] + rec["tm"] + rec["tmiss"]
    if truth != n_truth:
        raise AssertionError(
            f"event {rec.get('event_idx')}: truth {truth} != {n_truth}"
        )


def paired_ratio_diff(
    num_a: np.ndarray,
    den_a: np.ndarray,
    num_b: np.ndarray,
    den_b: np.ndarray,
    boots: np.ndarray,
) -> tuple[float, float]:
    """(sum num_a / sum den_a) - (sum num_b / sum den_b) and its bootstrap std.

    ``boots`` is an (n_boot, n_events) array of resampled event indices,
    shared by both arms so the error is that of the paired difference.
    """
    na, da, nb, db = (
        np.asarray(x, dtype=np.float64) for x in (num_a, den_a, num_b, den_b)
    )
    diff = na.sum() / da.sum() - nb.sum() / db.sum()
    bs = na[boots].sum(1) / da[boots].sum(1) - nb[boots].sum(1) / db[boots].sum(1)
    return float(diff), float(np.std(bs, ddof=1))


def clopper_pearson(k: int, n: int, cl: float = 0.6827) -> tuple[float, float]:
    """Exact (Clopper-Pearson) central interval for k successes in n trials."""
    a = (1.0 - cl) / 2.0
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi
