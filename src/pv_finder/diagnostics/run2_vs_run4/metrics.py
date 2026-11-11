"""Shared estimators for the Run 2 vs HL-LHC comparison.

sigma_vtx-vtx
    The headline uses ``amvf_fair_comparison.scan.fit_sigma_vtx_vtx`` directly.
    Variations (bins, range, event resampling, weights) need the per-event
    histograms, so :func:`event_hists` re-implements exactly its loop (same
    seeded shuffle, same signed i < j differences, same |dz| <= range cut) and
    :func:`fit_counts` exactly its ``curve_fit`` call.  :func:`verify_against_scan`
    asserts the two give the identical sigma before anything is quoted.
Position resolution
    ``matched_residuals_optimal`` per event, concatenated, 68th percentile:
    identical to ``scan.core_position_resolution`` (also verified), but keeps
    the per-event split so whole events can be resampled.
HS resolution
    ``itk_resolution.hs_residual`` / ``local_pu_density`` / ``iterative_rms``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from pv_finder.diagnostics.amvf_fair_comparison.matching import (
    matched_residuals_optimal,
)
from pv_finder.diagnostics.amvf_fair_comparison.scan import (
    core_position_resolution,
    fit_sigma_vtx_vtx,
    sigmoid_fit,
)
from pv_finder.evaluation.vertex_finding.itk_resolution import (
    hs_residual,
    iterative_rms,
    local_pu_density,
)
from pv_finder.utils.pairwise_dz import DEFAULT_PAIRWISE_BINS, PAIRWISE_RANGE_MM

SEED: int = 20260805  # scan.fit_sigma_vtx_vtx default


def event_hists(
    z_by_event: list[np.ndarray],
    *,
    n_bins: int = DEFAULT_PAIRWISE_BINS,
    range_mm: float = PAIRWISE_RANGE_MM,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-event signed pairwise-dz histograms, exactly as fit_sigma_vtx_vtx builds them.

    Returns ``(H, edges)`` with ``H`` of shape ``(n_events, n_bins)``.  Events
    with fewer than two vertices get an all-zero row (fit_sigma_vtx_vtx skips
    them without consuming random numbers, and so does this).
    """
    rng = np.random.default_rng(seed)
    edges = np.linspace(-range_mm, range_mm, n_bins + 1)
    H = np.zeros((len(z_by_event), n_bins), dtype=np.int32)
    for k, z in enumerate(z_by_event):
        if len(z) < 2:
            continue
        zs = np.asarray(z, dtype=np.float64).copy()
        rng.shuffle(zs)
        dz = (zs[:, None] - zs[None, :])[np.triu_indices(len(zs), k=1)]
        H[k] = np.histogram(dz[np.abs(dz) <= range_mm], bins=edges)[0]
    return H, edges


def fit_counts(edges: np.ndarray, cnts: np.ndarray) -> tuple[float, float, np.ndarray]:
    """The sigmoid fit of fit_sigma_vtx_vtx on given counts: (sigma, err, popt)."""
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    cnts = np.asarray(cnts, dtype=float)
    base = float(np.median(cnts))
    p0 = [max(base - float(cnts.min()), 1.0), 10.0, max(base, 1.0), 0.5]
    try:
        popt, pcov = curve_fit(
            sigmoid_fit, ctrs, cnts, p0=p0, maxfev=20000,
            bounds=([0, 0, 0, 0], [np.inf] * 4),
        )  # fmt: skip
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), np.full(4, np.nan)
    return float(abs(popt[3])), float(np.sqrt(np.diag(pcov))[3]), popt


def verify_against_scan(z_by_event: list[np.ndarray]) -> dict[str, float]:
    """Assert the histogram path reproduces fit_sigma_vtx_vtx bit for bit."""
    s_ref, e_ref, n_ref = fit_sigma_vtx_vtx(z_by_event)
    H, edges = event_hists(z_by_event)
    s, e, _ = fit_counts(edges, H.sum(0))
    if not (s == s_ref and e == e_ref and int(H.sum()) == n_ref):
        raise AssertionError(f"histogram path {s},{e} != scan {s_ref},{e_ref}")
    return {"sigma_mm": s_ref, "err_mm": e_ref, "n_pairs": n_ref}


def merge_length(edges: np.ndarray, cnts: np.ndarray, plateau=(3.0, 6.0)) -> float:
    """Model-free dip size: integral of (1 - density/plateau) over 0 < |dz| < 3 mm.

    For an ideal step that removes every pair closer than d and none further,
    this equals d; it does not assume a sigmoid shape, so it is insensitive to
    how sharp the dip walls are.  Plateau = mean count per bin at 3 < |dz| < 6.
    """
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    bw = float(edges[1] - edges[0])
    a = np.abs(ctrs)
    plat = float(np.mean(cnts[(a >= plateau[0]) & (a < plateau[1])]))
    if plat <= 0:
        return float("nan")
    core = a < plateau[0]
    # Both signs are histogrammed, so halve the summed deficit.
    return float(np.sum(1.0 - cnts[core] / plat) * bw / 2.0)


def sigma_summary(
    H: np.ndarray,
    edges: np.ndarray,
    idx: np.ndarray | None = None,
    *,
    n_boot: int = 200,
    seed: int = SEED,
) -> dict[str, float]:
    """Sigma, fit error, bootstrap-over-events error, merge length, pairs."""
    idx = np.arange(len(H)) if idx is None else np.asarray(idx)
    c = H[idx].sum(0)
    s, e, popt = fit_counts(edges, c)
    rng = np.random.default_rng(seed)
    bs, bl = [], []
    for _ in range(n_boot):
        cb = H[rng.choice(idx, size=len(idx))].sum(0)
        bs.append(fit_counts(edges, cb)[0])
        bl.append(merge_length(edges, cb))
    return {
        "sigma_mm": s, "fit_err_mm": e,
        "boot_err_mm": float(np.nanstd(bs, ddof=1)) if n_boot > 1 else float("nan"),
        "merge_len_mm": merge_length(edges, c),
        "merge_len_err_mm": float(np.nanstd(bl, ddof=1)) if n_boot > 1 else float("nan"),
        "slope_per_mm": float(popt[1]), "plateau": float(popt[0] + popt[2]),
        "floor_frac": float(popt[2] / (popt[0] + popt[2])) if popt[0] + popt[2] else np.nan,
        "n_pairs": int(c.sum()), "n_events": int(len(idx)),
    }  # fmt: skip


def paired_ratio(
    Ha: np.ndarray,
    Hb: np.ndarray,
    edges: np.ndarray,
    idx: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = SEED,
) -> dict[str, float]:
    """sigma_a / sigma_b with the same event resampling applied to both arms."""
    sa = fit_counts(edges, Ha[idx].sum(0))[0]
    sb = fit_counts(edges, Hb[idx].sum(0))[0]
    rng = np.random.default_rng(seed)
    rr = []
    for _ in range(n_boot):
        j = rng.choice(idx, size=len(idx))
        rr.append(
            fit_counts(edges, Ha[j].sum(0))[0] / fit_counts(edges, Hb[j].sum(0))[0]
        )
    return {"ratio": sa / sb, "err": float(np.nanstd(rr, ddof=1))}


def per_event_residuals(
    truth_ge2: list[np.ndarray], reco: list[np.ndarray], window_mm: float = 1.0
) -> list[np.ndarray]:
    """|dz| of optimal one-to-one pairs within the window, kept per event."""
    out = []
    for t, r in zip(truth_ge2, reco):
        if len(t) and len(r):
            out.append(matched_residuals_optimal(t, r, window_mm))
        else:
            out.append(np.zeros(0))
    return out


def p68_summary(
    res: list[np.ndarray], idx: np.ndarray, *, n_boot: int = 200, seed: int = SEED
) -> dict[str, float]:
    """Core p68 of |dz| over the selected events, with event-bootstrap error."""
    cat = [res[i] for i in idx]
    p = float(np.percentile(np.concatenate(cat), 68.0))
    rng = np.random.default_rng(seed)
    bs = [
        np.percentile(np.concatenate([res[i] for i in rng.choice(idx, len(idx))]), 68.0)
        for _ in range(n_boot)
    ]
    return {"p68_um": 1e3 * p, "err_um": 1e3 * float(np.std(bs, ddof=1)),
            "n_matched": int(sum(len(c) for c in cat))}  # fmt: skip


def verify_p68(truth_ge2: list[np.ndarray], reco: list[np.ndarray]) -> float:
    """The per-event path must equal scan.core_position_resolution exactly."""
    ref = core_position_resolution(truth_ge2, reco, window_mm=1.0)
    mine = float(
        np.percentile(np.concatenate(per_event_residuals(truth_ge2, reco)), 68.0)
    )
    if mine != ref:
        raise AssertionError(f"p68 {mine} != core_position_resolution {ref}")
    return ref


def hs_table(
    truth_z: list[np.ndarray],
    truth_type: list[np.ndarray],
    reco: dict[str, list[np.ndarray]],
    max_dz: float = 1.0,
) -> dict[str, np.ndarray]:
    """Per-event HS residual for each algorithm, and the local PU density.

    HS = TruthVertex_type == 1 (exactly one per event, checked).  The density
    counts every stored truth interaction (all have >= 1 selected track).
    """
    n = len(truth_z)
    out: dict[str, np.ndarray] = {"density": np.zeros(n), "hs_z": np.zeros(n)}
    for a in reco:
        out[a] = np.full(n, np.nan)
    for k in range(n):
        hs = np.flatnonzero(truth_type[k] == 1)
        if len(hs) != 1:
            raise ValueError(f"event {k}: {len(hs)} HS vertices")
        h = int(hs[0])
        out["hs_z"][k] = truth_z[k][h]
        out["density"][k] = local_pu_density(truth_z[k], h, 2.0)
        for a, r in reco.items():
            out[a][k] = hs_residual(r[k], float(truth_z[k][h]), max_dz)
    return out


def hs_summary(
    res: np.ndarray, idx: np.ndarray, halfrange: float, *, n_boot: int = 200,
    seed: int = SEED,
) -> dict[str, Any]:  # fmt: skip
    """Iterative-RMS HS resolution on the selected events, bootstrap error too."""
    r = iterative_rms(res[idx], halfrange)
    rng = np.random.default_rng(seed)
    bs = [
        iterative_rms(res[rng.choice(idx, len(idx))], halfrange).rms
        for _ in range(n_boot)
    ]
    d = r._asdict()
    d["boot_err"] = float(np.nanstd(bs, ddof=1))
    d["match_eff"] = float(np.mean(np.isfinite(res[idx])))
    return d


def matched_indices(
    target: np.ndarray, pool: np.ndarray, edges: np.ndarray, seed: int = SEED
) -> np.ndarray:
    """Draw pool events (without replacement) whose binned variable matches target's.

    For every target event one pool event from the same bin; if a bin runs out,
    it is filled with replacement and the count of such reuses is reported by
    the caller via duplicates in the returned index list.
    """
    rng = np.random.default_rng(seed)
    tb = np.digitize(target, edges)
    pb = np.digitize(pool, edges)
    out: list[int] = []
    for b in np.unique(tb):
        need = int((tb == b).sum())
        cand = np.flatnonzero(pb == b)
        if len(cand) == 0:
            continue
        replace = len(cand) < need
        out.extend(rng.choice(cand, size=need, replace=replace).tolist())
    return np.sort(np.asarray(out, dtype=np.int64))
