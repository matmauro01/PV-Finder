"""Candidate-level peak finding for the HL-LHC consolidation study (13 Nov 2026).

The production scan (``pv_locations_updated_res``) segments the density into
regions at a fixed amplitude, then keeps a region if it passes three cuts
(width, integral, height) and places it with the local centroid. The
segmentation does not depend on those cuts, so this module records every
region once, with all its properties and the position under each candidate
estimator. Any cut configuration and estimator is then a mask plus a column.

Bit-identity with the production scan is checked by ``verify_against_production``
before any number is taken from it (same statements, same float64
accumulators, same float32 output cast).

Efficiency and fakes are computed without the matcher, through two identities
of the closest-first matcher (``compare_res_reco``), both checked per event in
``verify_against_production``:

* found truth (clean + merged) = truth with any reco within the window
  (Eq. (1), ``freeze_compare.truth_found_any``);
* fake reco = reco with no truth within the window.
"""

from __future__ import annotations

import numba
import numpy as np

Z_MIN, BIN_WIDTH, N_BINS = -240.0, 0.04, 12000

# Estimator name -> column of the candidate position array.
ESTIMATORS: tuple[str, ...] = (
    "max_bin",
    "centroid_1",
    "centroid_2",
    "centroid_3",
    "centroid_5",
    "full_region",
    "parabola_3pt",
    "gauss_3pt",
)
_CENTROID_HW = {"centroid_1": 1, "centroid_2": 2, "centroid_3": 3, "centroid_5": 5}
N_EST = len(ESTIMATORS)
# Scalar properties per candidate.
PROPS: tuple[str, ...] = ("start", "end", "cmax", "width", "integral", "height")


@numba.njit(cache=True)
def _local_centroid(targets, lo, hi, fallback):
    tot = 0.0
    wsum = 0.0
    for j in range(lo, hi + 1):
        v = targets[j]
        if v > 0.0:
            tot += v
            wsum += j * v
    return wsum / tot if tot > 0.0 else float(fallback)


@numba.njit(cache=True)
def _three_point(y0, y1, y2):
    """Vertex offset of the parabola through three equally spaced points."""
    den = y0 - 2.0 * y1 + y2
    if den >= 0.0:
        return 0.0
    d = 0.5 * (y0 - y2) / den
    if d > 1.0:
        d = 1.0
    if d < -1.0:
        d = -1.0
    return d


@numba.njit(cache=True)
def _regions(targets, threshold, props, pos):
    """All regions of one histogram (no width/integral/height cut).

    Mirrors ``peak_finding_fast._scan`` statement for statement; only the
    record condition is dropped. Returns the number of regions.
    """
    state = 0
    integral = 0.0
    sum_wl = 0.0
    currentmax = 0
    region_start = 0
    peak_passed = False
    n = 0
    nb = len(targets)
    for i in range(nb):
        if state == 0:
            currentmax = i
        if targets[i] >= threshold:
            if state == 0:
                region_start = i
            state += 1
            integral += targets[i]
            sum_wl += i * targets[i]
            if targets[i] > targets[currentmax]:
                currentmax = i
            if i > 0 and targets[i - 1] > targets[i]:
                peak_passed = True
        conjoined_split = i > 0 and (targets[i - 1] < targets[i]) and peak_passed
        if (targets[i] < threshold or i == nb - 1 or conjoined_split) and state > 0:
            region_end = i if targets[i] >= threshold else i - 1
            m = currentmax
            props[n, 0] = region_start
            props[n, 1] = region_end
            props[n, 2] = m
            props[n, 3] = state
            props[n, 4] = integral
            props[n, 5] = targets[m]
            # max bin
            pos[n, 0] = float(m)
            # local centroids, clipped to the region (production rule)
            for c, hw in enumerate((1, 2, 3, 5)):
                lo = max(region_start, m - hw)
                hi = min(region_end, m + hw)
                pos[n, 1 + c] = _local_centroid(targets, lo, hi, m)
            # full-region centroid (historical estimator)
            pos[n, 5] = sum_wl / integral
            # 3-point fits on the raw neighbours of the maximum
            if 0 < m < nb - 1:
                y0, y1, y2 = targets[m - 1], targets[m], targets[m + 1]
                pos[n, 6] = m + _three_point(y0, y1, y2)
                if y0 > 0.0 and y1 > 0.0 and y2 > 0.0:
                    pos[n, 7] = m + _three_point(np.log(y0), np.log(y1), np.log(y2))
                else:
                    pos[n, 7] = pos[n, 6]
            else:
                pos[n, 6] = float(m)
                pos[n, 7] = float(m)
            n += 1
            state = 0
            integral = 0.0
            sum_wl = 0.0
            peak_passed = False
    return n


def extract(hists: list[np.ndarray], threshold: float = 0.01) -> dict:
    """Candidates of every event: flat arrays plus per-event offsets.

    Positions are returned in mm with the production float32 cast
    (``pos_bin * 0.04 + (-240)`` in float64, stored as float32).
    """
    props_all, pos_all, offs = [], [], [0]
    buf_p = np.empty((N_BINS, len(PROPS)), np.float64)
    buf_z = np.empty((N_BINS, N_EST), np.float64)
    for h in hists:
        h = np.ascontiguousarray(h)
        n = _regions(h, threshold, buf_p, buf_z)
        props_all.append(buf_p[:n].copy())
        pos_all.append((buf_z[:n] * BIN_WIDTH + Z_MIN).astype(np.float32))
        offs.append(offs[-1] + n)
    props = np.concatenate(props_all)
    return {
        "offsets": np.asarray(offs, dtype=np.int64),
        "width": props[:, 3].astype(np.int32),
        "integral": props[:, 4],
        "height": props[:, 5].astype(np.float32),
        "cmax": props[:, 2].astype(np.int32),
        "pos": np.concatenate(pos_all),
        "threshold": threshold,
    }


def select(
    cand: dict, *, min_width: int, integral: float, min_height: float
) -> np.ndarray:
    """Boolean mask of the candidates that pass the production record rule."""
    return (
        (cand["width"] >= min_width)
        & (cand["integral"] >= integral)
        & (cand["height"] >= min_height)
    )


def peaks_by_event(cand: dict, mask: np.ndarray, estimator: str) -> list[np.ndarray]:
    """Per-event float32 peak positions (mm) for a mask and an estimator."""
    col = cand["pos"][:, ESTIMATORS.index(estimator)]
    o = cand["offsets"]
    return [col[o[k] : o[k + 1]][mask[o[k] : o[k + 1]]] for k in range(len(o) - 1)]


def _n_within(a: np.ndarray, b: np.ndarray, w: float) -> int:
    """Number of entries of a with some entry of b within w (inclusive).

    Same float32 arithmetic and ``<=`` as ``compare_res_reco`` on bin units.
    """
    if len(a) == 0 or len(b) == 0:
        return 0
    b = np.sort(b)
    i = np.searchsorted(b, a)
    left = np.abs(a - b[np.clip(i - 1, 0, len(b) - 1)])
    right = np.abs(a - b[np.clip(i, 0, len(b) - 1)])
    return int(np.sum((left <= w) | (right <= w)))


def to_bins(z: np.ndarray) -> np.ndarray:
    """The matcher's bin-unit input (float32 in, float32 out)."""
    return (np.asarray(z) - Z_MIN) / BIN_WIDTH


def found_fake(
    truth_b: list[np.ndarray], preds_mm: list[np.ndarray], window_mm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event (found truth, fake reco, n reco) without the matcher."""
    w = window_mm / BIN_WIDTH
    n = len(truth_b)
    found, fake, npred = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        pb = to_bins(preds_mm[k])
        found[k] = _n_within(truth_b[k], pb, w)
        fake[k] = len(pb) - _n_within(pb, truth_b[k], w)
        npred[k] = len(pb)
    return found, fake, npred


def matcher_counts(
    truth_b: list[np.ndarray], preds_mm: list[np.ndarray], window_mm: float
) -> dict[str, np.ndarray]:
    """Per-event production-matcher categories (slow; for final configurations)."""
    from efficiency_res_optimized_atlas import compare_res_reco

    w = window_mm / BIN_WIDTH
    keys = ("found", "clean", "merged", "split", "fake", "t_merged", "npred")
    out = {k: np.zeros(len(truth_b)) for k in keys}
    for k, (tb, p) in enumerate(zip(truth_b, preds_mm)):
        pb = to_bins(p)
        out["npred"][k] = len(pb)
        if len(pb) == 0:
            continue
        res, tc, *_ = compare_res_reco(
            tb, pb, w * np.ones(len(pb)), debug=0, return_reco_cls=True
        )
        out["found"][k] = np.sum(tc == "clean") + np.sum(tc == "merged")
        out["t_merged"][k] = np.sum(tc == "merged")
        out["clean"][k], out["merged"][k] = res.reco_clean, res.reco_merged
        out["split"][k], out["fake"][k] = res.reco_split, res.reco_fake
    return out


def verify_against_production(
    hists: list[np.ndarray],
    cand: dict,
    peaks_cfg: dict,
    saved_pvs: list[np.ndarray],
    truth_b: list[np.ndarray],
    window_mm: float,
    n_check: int = 400,
) -> dict:
    """Bit-identity with the production scan and the two matcher identities.

    * every event: the frozen-config ``centroid_3`` peaks equal the saved
      ``pred_pvs_mm`` exactly;
    * ``n_check`` events: equal to a fresh ``pv_locations_updated_res`` call for
      every centroid half-width the production code supports (1, 2, 3, 5 and
      0 = full region), at the frozen cuts;
    * every event: found / fake from the identities equal the matcher's.
    """
    from efficiency_res_optimized_atlas import pv_locations_updated_res

    mask = select(
        cand, **{k: peaks_cfg[k] for k in ("min_width", "integral", "min_height")}
    )
    p3 = peaks_by_event(cand, mask, "centroid_3")
    for k, (a, b) in enumerate(zip(p3, saved_pvs)):
        if not np.array_equal(a, np.asarray(b, dtype=np.float32)):
            raise AssertionError(f"event {k}: candidate peaks differ from saved peaks")
    est_hw = {"full_region": 0, **_CENTROID_HW}
    for est, hw in est_hw.items():
        pe = peaks_by_event(cand, mask, est)
        for k in range(min(n_check, len(hists))):
            ref, *_ = pv_locations_updated_res(
                hists[k], cand["threshold"], peaks_cfg["integral"],
                peaks_cfg["min_width"], peaks_cfg["min_height"], hw,
            )  # fmt: skip
            if not np.array_equal(pe[k], ref):
                raise AssertionError(f"{est}, event {k}: differs from production scan")
    found, fake, _ = found_fake(truth_b, p3, window_mm)
    m = matcher_counts(truth_b, p3, window_mm)
    if not (np.array_equal(found, m["found"]) and np.array_equal(fake, m["fake"])):
        raise AssertionError("found/fake identities differ from the matcher")
    return {
        "events_bit_identical_to_saved": len(p3),
        "events_checked_vs_production_scan_per_estimator": min(n_check, len(hists)),
        "matcher_identities_events": len(p3),
        "matcher": m,
    }
