"""Vertex matching and categorization for PV-Finder evaluation.

Ports the ATLAS vertex matching logic from the original atlas_pvfinder repo.
All matching positions are in **bins** (12000 bins over [-240, 240] mm).
Peak finding is NOT included here -- see ``pv_finder.utils.peak_finding``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import curve_fit

from pv_finder.utils.constants import TOTAL_NUM_BINS, Z_MAX, Z_MIN

_FWHM_TO_SIGMA = 2.335  # 2 * sqrt(2 * ln 2)


# -- Resolution fitting (from run3_infer_compare_amvf.py) ------------------


def fit_func_resolution(
    x: np.ndarray | float, a: float, b: float, c: float, rcc: float
) -> np.ndarray | float:
    """Sigmoid: ``a / (1 + exp(b * (rcc - |x|))) + c``."""
    return a / (1 + np.exp(b * (rcc - np.abs(x)))) + c


def fit_sigma_vtx_vtx(
    all_pv_distances_mm: np.ndarray | list[float],
) -> tuple[float, float]:
    """Fit pairwise vertex distances to extract resolution sigma.

    Histograms distances in [-6, 6] mm (61 bins), fits the sigmoid
    ``fit_func_resolution``.  Falls back to std of |d| < 2 mm on failure.

    Returns ``(sigma_mm, sigma_err_mm)``.
    """
    distances = np.asarray(all_pv_distances_mm, dtype=float)
    counts, edges = np.histogram(distances, bins=61, range=(-6.0, 6.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    cf = counts.astype(float)

    # Skip first bin for the fit (matches original code)
    try:
        p0 = [float(np.max(cf)), 10.0, float(np.min(cf)), 0.5]
        popt, pcov = curve_fit(
            fit_func_resolution, centers[1:], cf[1:], p0=p0, maxfev=10_000
        )
        return float(abs(popt[3])), float(np.sqrt(np.diag(pcov))[3])
    except RuntimeError:
        close = distances[np.abs(distances) < 2.0]
        return (float(np.std(close)) if len(close) > 0 else 0.0), 0.0


def make_resolution_plot(
    all_pv_distances_mm: np.ndarray | list[float],
    output_dir: str | os.PathLike[str],
    label: str = "PV-Finder",
) -> tuple[float, float]:
    """Histogram + sigmoid fit + save PNG/PDF.  Returns ``(sigma_mm, sigma_err_mm)``."""
    import matplotlib  # lazy import

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distances = np.asarray(all_pv_distances_mm, dtype=float)
    if len(distances) < 10:
        return 0.0, 0.0

    fig, ax = plt.subplots(figsize=(12, 8))
    counts, bins, _ = ax.hist(
        distances,
        bins=61,
        range=(-6.0, 6.0),
        color="steelblue",
        alpha=0.6,
        edgecolor="darkblue",
        linewidth=0.8,
        label=label,
    )
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.errorbar(
        centers,
        counts,
        yerr=np.sqrt(counts + 1),
        fmt="none",
        ecolor="darkblue",
        elinewidth=1.5,
        capsize=2,
        alpha=0.6,
    )

    sigma_fit, sigma_err = 0.0, 0.0
    try:
        p0 = [float(np.max(counts)), 10.0, float(np.min(counts)), 0.5]
        popt, pcov = curve_fit(
            fit_func_resolution,
            centers[1:],
            counts[1:],
            p0=p0,
            maxfev=10_000,
        )
        sigma_fit = float(abs(popt[3]))
        sigma_err = float(np.sqrt(np.diag(pcov))[3])
        x_fit = np.linspace(-6.0, 6.0, 1000)
        ax.plot(
            x_fit,
            fit_func_resolution(x_fit, *popt),
            "r-",
            linewidth=2.5,
            label=f"Fit: sigma = {sigma_fit:.2f} mm",
            zorder=10,
        )
    except RuntimeError:
        close = distances[np.abs(distances) < 2.0]
        if len(close) > 0:
            sigma_fit = float(np.std(close))

    ax.set_xlabel(r"$\Delta z_{\mathrm{vtx-vtx}}$ [mm]", fontsize=18)
    ax.set_ylabel("Counts", fontsize=18)
    ax.set_title(
        "Distance Between Pairs of Nearby Reconstructed Vertices", fontsize=16, pad=15
    )
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=-50)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(out / f"deltaz_resolution.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return sigma_fit, sigma_err


# -- FWHM-based resolution (from efficiency_res_optimized_atlas.py) --------


def get_reco_resolution(
    peak_bins: np.ndarray,
    conjoined_l: np.ndarray,
    conjoined_r: np.ndarray,
    histogram: np.ndarray,
    *,
    nsig_res: float = 5.0,
    steps_extrapolation: int = 10,
    ratio_max: float = 0.5,
) -> np.ndarray:
    """Per-peak resolution from the FWHM of the predicted KDE.

    Measures the full-width at *ratio_max* of peak maximum, then converts to
    an n-sigma window: ``nsig_res * FWHM / 2.335``.

    Modes: ``steps_extrapolation == 0`` (bin-level, no interpolation) or
    ``> 0`` (sub-bin linear interpolation).  Conjoined peaks use one-sided
    FWHM doubled to avoid contamination from the neighbour.
    """
    n_peaks = len(peak_bins)
    reco_reso = np.empty(n_peaks, dtype=float)
    steps = steps_extrapolation

    if steps == 0:
        for i in range(n_peaks):
            peak_idx = int(peak_bins[i])
            fwhm_level = ratio_max * histogram[peak_idx]
            ibin_min, ibin_max = -1, -1

            for ibin in range(peak_idx, peak_idx - 20, -1):
                if histogram[ibin] < fwhm_level:
                    ibin_min = ibin
                    break
            for ibin in range(peak_idx, peak_idx + 20):
                if histogram[ibin] < fwhm_level:
                    ibin_max = ibin
                    break

            # Conjoined logic: use the clean side doubled
            if not conjoined_r[i] and conjoined_l[i]:
                fwhm_w = 2 * (peak_idx - ibin_min)
            elif conjoined_r[i] and not conjoined_l[i]:
                fwhm_w = 2 * (ibin_max - peak_idx)
            else:
                fwhm_w = ibin_max - ibin_min

            reco_reso[i] = nsig_res * fwhm_w / _FWHM_TO_SIGMA
    else:
        for i in range(n_peaks):
            peak_idx = int(peak_bins[i])
            fwhm_level = ratio_max * histogram[peak_idx]
            ibin_min_ext: float = 0.0
            ibin_max_ext: float = float(TOTAL_NUM_BINS)
            found_min, found_max = False, False

            # Search leftward with sub-bin interpolation
            for ibin in range(peak_idx, max(peak_idx - 20, 1), -1):
                if not found_min:
                    val_curr = histogram[ibin]
                    delta = (histogram[ibin - 1] - val_curr) / steps
                    interp_val = val_curr
                    for sub in range(steps):
                        interp_val -= delta * sub
                        if interp_val < fwhm_level:
                            ibin_min_ext = (ibin * steps - sub) / steps
                            found_min = True
                            break

            # Search rightward with sub-bin interpolation
            upper = min(peak_idx + 20, TOTAL_NUM_BINS - 1)
            for ibin in range(peak_idx, upper):
                if not found_max:
                    val_curr = histogram[ibin]
                    delta = (val_curr - histogram[ibin + 1]) / steps
                    interp_val = val_curr
                    for sub in range(steps):
                        interp_val -= delta * sub
                        if interp_val < fwhm_level:
                            ibin_max_ext = (ibin * steps + sub) / steps
                            found_max = True
                            break

            # Conjoined logic: use the clean side doubled
            if not conjoined_r[i] and conjoined_l[i]:
                fwhm_w = 2 * (peak_idx - ibin_min_ext)
            elif conjoined_r[i] and not conjoined_l[i]:
                fwhm_w = 2 * (ibin_max_ext - peak_idx)
            else:
                fwhm_w = ibin_max_ext - ibin_min_ext

            reco_reso[i] = nsig_res * fwhm_w / _FWHM_TO_SIGMA

    return reco_reso


# -- NaN filtering (from efficiency_res_optimized_atlas.py) ----------------


def filter_nans_res(items_bins: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Boolean mask: True where the truth KDE at the bin is not NaN."""
    valid = np.zeros(items_bins.shape, dtype=bool)
    for i, item in enumerate(items_bins):
        idx = int(round(item))
        if not np.isnan(mask[idx]):
            valid[i] = True
    return valid


# -- Vertex categorization (from efficiency_res_optimized_atlas.py) --------


class PerformanceInfo(NamedTuple):
    """Counts of vertex categories from ``compare_res_reco``."""

    reco_merged: int
    reco_split: int
    reco_clean: int
    reco_fake: int


def compare_res_reco(
    truth_bins: np.ndarray,
    pred_bins: np.ndarray,
    reco_res_bins: np.ndarray,
) -> tuple[PerformanceInfo, np.ndarray, np.ndarray]:
    """Categorize reco vertices as Clean / Merged / Split / Fake.

    Also classifies each truth vertex (clean/merged/missed) and computes
    local pileup density.  All positions and resolutions in **bin** units.

    Returns ``(PerformanceInfo, truth_classification, local_density)``.
    """
    n_truth, n_pred = len(truth_bins), len(pred_bins)

    truth_cls: list[list[str]] = [[] for _ in range(n_truth)]
    reco_cls: list[list[str]] = [[] for _ in range(n_pred)]
    truth_assign: list[list[int]] = [[] for _ in range(n_truth)]
    reco_assign: list[list[int]] = [[] for _ in range(n_pred)]

    # First pass: match predicted -> truth
    for i in range(n_pred):
        wt = np.argwhere(np.abs(truth_bins - pred_bins[i]) <= reco_res_bins[i])
        if len(wt) == 0:
            reco_cls[i] = reco_cls[i] + ["fake"]
            reco_assign[i] = reco_assign[i] + [-1]
        if len(wt) == 1:
            w = wt[0][0]
            reco_cls[i] = reco_cls[i] + ["clean"]
            truth_assign[w] = truth_assign[w] + [i]
            truth_cls[w] = truth_cls[w] + ["clean"]
            reco_assign[i] = reco_assign[i] + [w]
        if len(wt) > 1:
            reco_cls[i] = reco_cls[i] + ["merged"]
            for j in wt:
                reco_assign[i] = reco_assign[i] + [j[0]]
                truth_assign[j[0]] = truth_assign[j[0]] + [i]
                truth_cls[j[0]] = truth_cls[j[0]] + ["merged"]

    # Mark missed truth vertices
    for i in range(n_truth):
        if len(truth_cls[i]) == 0:
            truth_assign[i] = truth_assign[i] + [-1]
            truth_cls[i] = truth_cls[i] + ["missed"]

    # Handle split vertices (truth matched by multiple clean reco)
    for i in range(n_truth):
        if len(truth_cls[i]) > 1:
            wc = np.argwhere(np.array(truth_cls[i]) == "clean")
            wm = np.argwhere(np.array(truth_cls[i]) == "merged")
            if len(wc) > 1:
                rk = np.array(truth_assign[i])[np.array(wc).reshape(len(wc))]
                diff = np.abs(truth_bins[i] - pred_bins[rk]) / reco_res_bins[rk]
                best = rk[np.argmin(diff)]
                for k in rk:
                    if k != best:
                        reco_cls[k] = ["split"]
                        reco_assign[k] = [i]
            if len(wm) > 1:
                truth_cls[i] = ["merged"]
            if len(truth_cls[i]) > 1:
                truth_cls[i] = ["merged"]

    truth_classification = np.array(
        [c[0] if c else "missed" for c in truth_cls], dtype=object
    )
    reco_classification = np.array(
        [c[0] if c else "fake" for c in reco_cls], dtype=object
    )

    # Local pileup density (truth PVs within 1 mm)
    bins_1mm = TOTAL_NUM_BINS / (Z_MAX - Z_MIN)
    local_density = np.zeros(n_truth, dtype=float)
    for i in range(n_truth):
        local_density[i] = np.sum(np.abs(truth_bins[i] - truth_bins) <= bins_1mm) / 2

    ra = np.atleast_1d(reco_classification)
    perf = PerformanceInfo(
        reco_merged=int(np.sum(ra == "merged")),
        reco_split=int(np.sum(ra == "split")),
        reco_clean=int(np.sum(ra == "clean")),
        reco_fake=int(np.sum(ra == "fake")),
    )
    return perf, truth_classification, local_density


# -- S/MT/FP matching (from efficiency_res_optimized_atlas.py) -------------


def compare_res_reco2(
    truth_bins: np.ndarray,
    pred_bins: np.ndarray,
    reco_res_bins: np.ndarray,
) -> tuple[int, int, int]:
    """Count Succeed / Missed-Truth / False-Positive via greedy matching.

    Iterates over the larger set and matches against the smaller, removing
    matched entries to avoid double-counting.  All positions in **bin** units.

    Returns ``(succeed, missed, false_pos)``.
    """
    t = np.array(truth_bins, dtype=float)
    p = np.array(pred_bins, dtype=float)
    r = np.array(reco_res_bins, dtype=float)
    succeed, missed, false_pos = 0, 0, 0
    n_pred, n_truth = len(p), len(t)

    if n_pred >= n_truth:
        for i in range(n_pred):
            lo, hi = p[i] - r[i], p[i] + r[i]
            matched = False
            for j in range(len(t)):
                if lo <= t[j] <= hi:
                    matched = True
                    succeed += 1
                    t = np.delete(t, j)
                    break
            if not matched:
                false_pos += 1
        missed = n_truth - succeed
    else:
        for i in range(n_truth):
            matched = False
            for j in range(len(p)):
                lo, hi = p[j] - r[j], p[j] + r[j]
                if lo <= t[i] <= hi:
                    matched = True
                    succeed += 1
                    p = np.delete(p, j)
                    r = np.delete(r, j)
                    break
            if not matched:
                missed += 1
        false_pos = n_pred - succeed

    return succeed, missed, false_pos
