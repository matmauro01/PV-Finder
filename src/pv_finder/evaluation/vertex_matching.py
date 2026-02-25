"""Vertex matching and categorization for PV-Finder evaluation.

Faithful port of the ATLAS vertex matching logic from
atlas_pvfinder/mattia_finder/model/efficiency_res_optimized_atlas.py.

All matching positions are in **bins** (12000 bins over [-240, 240] mm).
Includes its own peak-finding function (_pv_locations_updated_res) that
returns a 6-tuple including conjoined flags.  The public
pv_finder.utils.peak_finding module returns only 4 values and is used
by diagnostics; the private function here is used by eval scripts.
"""

from __future__ import annotations

import os
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from pv_finder.utils.constants import TOTAL_NUM_BINS, Z_MAX, Z_MIN

_FWHM_TO_SIGMA = 2.335  # 2 * sqrt(2 * ln 2)
_BIN_WIDTH = (Z_MAX - Z_MIN) / TOTAL_NUM_BINS  # 0.04 mm


# -- Peak finding (from efficiency_res_optimized_atlas.py lines 59-173) -----


def _pv_locations_updated_res(
    targets: np.ndarray,
    threshold: float,
    integral_threshold: float,
    min_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the z positions from the input KDE using the parsed criteria.

    Inputs:
      * targets:
          Numpy array of KDE values (predicted or true)

      * threshold:
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold:
          The total integral required to trigger a hit - such as 0.2

      * min_width:
          The minimum width (in bins) of a feature - such as 2

    Returns:
      * (items, peakvals, peakpos, conjoinedleft, conjoinedright, pv_sigmas)

    """
    # Counter of "active bins" i.e. with values above input threshold value
    state = 0
    # Sum of active bin values
    integral = 0.0
    # Weighted Sum of active bin values weighted by the bin location
    sum_weights_locs = 0.0
    sum_weights_locs_sq = 0.0
    # keeps track of peak location (assuming first entry is close to zero)
    currentmax = 0

    bin_width_val = _BIN_WIDTH
    z_min_val = Z_MIN

    # Make an empty array and manually track the size (faster than python array)
    items = np.empty(500, np.float32)
    peakpos = np.empty(500, np.int32)
    peakvals = np.empty(500, np.float32)
    conjoinedleft = np.empty(500, bool)
    conjoinedright = np.empty(500, bool)
    pv_sigmas = np.empty(500, np.float32)
    # Number of recorded PVs
    nitems = 0

    # Account for special case where two close PV merge KDE so that
    # targets[i] never goes below the threshold before the two PVs are scanned through
    peak_passed = False

    # Loop over the bins in the KDE histogram
    for i in range(len(targets)):
        if state == 0:
            currentmax = i
        # If bin value above 'threshold', then trigger
        if targets[i] >= threshold:
            state += 1
            integral += targets[i]
            sum_weights_locs += i * targets[i]  # weight times location
            sum_weights_locs_sq += (i * i) * targets[i]

            if targets[i] > targets[currentmax]:
                currentmax = i

            if targets[i - 1] > targets[i]:
                peak_passed = True  # keeps track of whether we passed peak in predicted distribution

        if (
            targets[i] < threshold
            or i == len(targets) - 1
            or (targets[i - 1] < targets[i] and peak_passed)
        ) and state > 0:
            # Record a PV only if
            if state >= min_width and integral >= integral_threshold:
                if nitems >= len(items):
                    # Dynamically resize arrays
                    items = np.resize(items, len(items) + 1)
                    peakpos = np.resize(peakpos, len(peakpos) + 1)
                    peakvals = np.resize(peakvals, len(peakvals) + 1)
                    conjoinedleft = np.resize(conjoinedleft, len(conjoinedleft) + 1)
                    conjoinedright = np.resize(conjoinedright, len(conjoinedright) + 1)
                    pv_sigmas = np.resize(pv_sigmas, len(pv_sigmas) + 1)

                # Adding '+0.5' to account for the bin width
                weighted_mean_bin = sum_weights_locs / integral
                weighted_variance_bin = (sum_weights_locs_sq / integral) - (
                    weighted_mean_bin * weighted_mean_bin
                )
                items[nitems] = weighted_mean_bin * bin_width_val + z_min_val
                peakvals[nitems] = targets[currentmax]  # store peak value
                peakpos[nitems] = currentmax

                if weighted_variance_bin < 0:
                    weighted_variance_bin = 0.0

                pv_sigmas[nitems] = np.sqrt(weighted_variance_bin) * bin_width_val

                if targets[i - 1] < targets[i] and peak_passed:
                    conjoinedleft[nitems] = True
                else:
                    conjoinedleft[nitems] = False

                if nitems > 0 and conjoinedleft[nitems - 1]:
                    conjoinedright[nitems] = True
                else:
                    conjoinedright[nitems] = False

                nitems += 1

            # reset state
            state = 0
            integral = 0.0
            sum_weights_locs = 0.0
            sum_weights_locs_sq = 0.0
            peak_passed = False

    # Special case for final item (very rare or never occuring)
    # handled by above if len
    return (
        items[:nitems],
        peakvals[:nitems],
        peakpos[:nitems],
        conjoinedleft[:nitems],
        conjoinedright[:nitems],
        pv_sigmas[:nitems],
    )


# -- NaN filtering (from efficiency_res_optimized_atlas.py lines 179-205) ---


def filter_nans_res(items: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask bins in the predicted KDE array if the corresponding bin in the true KDE array is 'nan'.

    Inputs:
      * items:
          Numpy array of predicted PV z positions
      * mask:
          Numpy array of KDE values (true PVs)

    Returns:
      * Boolean-valued Numpy array corresponding to which items are valid

    """
    # Create array corresponding to items (will say which items are valid, aka not masked)
    validinds = np.zeros(items.shape, dtype=bool)
    # Loop over the predicted PV z positions
    for i, item in enumerate(items):
        index = int(round(item))
        not_valid = np.isnan(mask[index])
        if not not_valid:
            validinds[i] = True

    return validinds


# -- FWHM-based resolution (from efficiency_res_optimized_atlas.py lines 210-352)


def get_reco_resolution(
    pred_PVs_loc: np.ndarray,
    pred_PVs_peakloc: np.ndarray,
    pred_PVs_cleft: np.ndarray,
    pred_PVs_cright: np.ndarray,
    predict: np.ndarray,
    nsig_res: float,
    steps_extrapolation: int,
    ratio_max: float,
    debug: bool,
) -> np.ndarray:
    """Compute the resolution as a function of predicted KDE histogram.

    Inputs:
      * pred_PVs_loc:
          Numpy array of computed z positions of the predicted PVs (using KDEs)

      * predict:
          Numpy array of predictions

      * nsig_res:
          Empirical value representing the number of sigma wrt to the std resolution
          as a function of FWHM

      * threshold:
          The threshold for considering an "on" value - such as 1e-2

      * integral_threshold:
          The total integral required to trigger a hit - such as 0.2

      * min_width:
          The minimum width (in bins) of a feature - such as 2

      * debug:
          flag to print output for debugging purposes


    Ouputs:
        Numpy array of filtered and sorted (in z values) expected resolution on the reco PVs z position.
    """

    reco_reso = np.empty_like(pred_PVs_loc)

    steps = steps_extrapolation

    i_predict_pv = 0

    if steps == 0:
        # This is for the case where we do not extrapolate values in between bins
        for i, predict_pv in enumerate(pred_PVs_loc):
            predict_pv_KDE_max = predict[pred_PVs_peakloc[i]]

            FWHM = ratio_max * predict_pv_KDE_max

            ibin_min = -1
            ibin_max = -1

            for ibin in range(pred_PVs_peakloc[i], pred_PVs_peakloc[i] - 20, -1):
                predict_pv_KDE_val = predict[ibin]
                if predict_pv_KDE_val < FWHM:
                    ibin_min = ibin
                    break

            for ibin in range(pred_PVs_peakloc[i], pred_PVs_peakloc[i] + 20):
                predict_pv_KDE_val = predict[ibin]
                if predict_pv_KDE_val < FWHM:
                    ibin_max = ibin
                    break

            if (
                not pred_PVs_cright[i] and pred_PVs_cleft[i]
            ):  # if a conjoined PV exists to the right
                FWHM_w = 2 * (pred_PVs_peakloc[i] - ibin_min)
            if (
                pred_PVs_cright[i] and not pred_PVs_cleft[i]
            ):  # if a conjoined PV exists to the left
                FWHM_w = 2 * (ibin_max - pred_PVs_peakloc[i])
            else:
                FWHM_w = ibin_max - ibin_min

            standard_dev = FWHM_w / 2.335
            reco_reso[i_predict_pv] = nsig_res * standard_dev
            i_predict_pv += 1

    else:
        for i, predict_pv in enumerate(pred_PVs_loc):
            predict_pv_KDE_max = predict[pred_PVs_peakloc[i]]

            FWHM = ratio_max * predict_pv_KDE_max

            ibin_min_extrapol = -1
            ibin_max_extrapol = -1
            found_min = False
            found_max = False
            for ibin in range(
                pred_PVs_peakloc[i], max(pred_PVs_peakloc[i] - 20, 1), -1
            ):
                if not found_min:
                    predict_pv_KDE_val_ibin = predict[ibin]
                    predict_pv_KDE_val_prev = predict[ibin - 1]

                    # Apply a dummy linear extrapolation between the two neigbour bins values
                    delta_steps = (
                        predict_pv_KDE_val_prev - predict_pv_KDE_val_ibin
                    ) / steps
                    for sub_bin in range(int(steps)):
                        predict_pv_KDE_val_ibin -= delta_steps * sub_bin

                        if predict_pv_KDE_val_ibin < FWHM:
                            ibin_min_extrapol = (ibin * steps - sub_bin) / steps
                            found_min = True

            for ibin in range(
                pred_PVs_peakloc[i], min(pred_PVs_peakloc[i] + 20, 11999)
            ):
                if not found_max:
                    predict_pv_KDE_val_ibin = predict[ibin]
                    predict_pv_KDE_val_next = predict[ibin + 1]

                    # Apply a dummy linear extrapolation between the two neigbour bins values
                    delta_steps = (
                        predict_pv_KDE_val_ibin - predict_pv_KDE_val_next
                    ) / steps
                    for sub_bin in range(int(steps)):
                        predict_pv_KDE_val_ibin -= delta_steps * sub_bin

                        if predict_pv_KDE_val_ibin < FWHM:
                            ibin_max_extrapol = (ibin * steps + sub_bin) / steps
                            found_max = True

            if not found_max:
                ibin_max_extrapol = 12000
            if not found_min:
                ibin_min_extrapol = 0

            if (
                not pred_PVs_cright[i] and pred_PVs_cleft[i]
            ):  # if a conjoined PV exists to the right
                FWHM_w = 2 * (pred_PVs_peakloc[i] - ibin_min_extrapol)
            if (
                pred_PVs_cright[i] and not pred_PVs_cleft[i]
            ):  # if a conjoined PV exists to the left
                FWHM_w = 2 * (ibin_max_extrapol - pred_PVs_peakloc[i])
            else:
                FWHM_w = ibin_max_extrapol - ibin_min_extrapol

            standard_dev = FWHM_w / 2.335
            reco_reso[i_predict_pv] = nsig_res * standard_dev
            i_predict_pv += 1

    return reco_reso


# -- Vertex categorization (from efficiency_res_optimized_atlas.py lines 510-514)

PerformanceInfo = namedtuple(
    "PerformanceInfo", ("reco_merged", "reco_split", "reco_clean", "reco_fake")
)


# -- compare_res_reco (from efficiency_res_optimized_atlas.py lines 516-633) -
# BUG FIX: [[]]*len(...) -> [[] for _ in range(n)] to avoid shallow copies


def compare_res_reco(
    target_PVs_loc: np.ndarray,
    pred_PVs_loc: np.ndarray,
    reco_res: np.ndarray,
    debug: bool,
) -> tuple[PerformanceInfo, np.ndarray, np.ndarray]:
    """Categorize reco vertices as Clean / Merged / Split / Fake.

    Inputs argument:
      * target_PVs_loc:
          Numpy array of z positions (in terms of bin #) of the TruthVertex objects.
          ensure that the list is filtered to require ntrks>=4

      * pred_PVs_loc:
          Numpy array of computed z positions (in terms of bin #) of the predicted PVs (using KDEs)

      * reco_res:
          Numpy array with the "reco" resolution as a function of width of predicted KDE signal

      * debug:
          flag to print output for debugging purposes


    Returns:
        PerformanceInfo named tuple
    """

    # BUG FIX: original used [[]]*len(...) which creates shallow copies
    truth_classification = [[] for _ in range(len(target_PVs_loc))]
    reco_classification = [[] for _ in range(len(pred_PVs_loc))]
    truth_assignment = [[] for _ in range(len(target_PVs_loc))]
    reco_assignment = [[] for _ in range(len(pred_PVs_loc))]

    num_merged = []

    # iterate through predicted PV locations
    for i, pred_loc in enumerate(pred_PVs_loc):
        # get all truth vertices within (pred_loc-res,pred_loc+res)
        where_truth = np.argwhere(np.abs(target_PVs_loc - pred_loc) <= reco_res[i])

        # takes care of all fake predictions
        if len(where_truth) == 0:
            reco_classification[i] = reco_classification[i] + ["fake"]
            reco_assignment[i] = reco_assignment[i] + [-1]

        # takes care of sparse truth assigments (no other surrounding vertices)
        if len(where_truth) == 1:
            where = where_truth[0][0]

            reco_classification[i] = reco_classification[i] + ["clean"]
            truth_assignment[where] = truth_assignment[where] + [i]
            truth_classification[where] = truth_classification[where] + ["clean"]
            reco_assignment[i] = reco_assignment[i] + [where]

        target_PVs_loc[where_truth]

        # takes care of dense cases (merged)
        if len(where_truth) > 1:
            num_merged.append(len(where_truth))

            reco_classification[i] = reco_classification[i] + ["merged"]
            for j in where_truth:
                reco_assignment[i] = reco_assignment[i] + [j[0]]
                truth_assignment[j[0]] = truth_assignment[j[0]] + [i]
                truth_classification[j[0]] = truth_classification[j[0]] + ["merged"]

    # take care of remaining missing truth PVs
    for i in range(len(truth_classification)):
        if len(truth_classification[i]) == 0:
            truth_assignment[i] = truth_assignment[i] + [-1]
            truth_classification[i] = truth_classification[i] + ["missed"]

    # handling multiple classifications (split vertices)
    for i in range(len(truth_classification)):
        if len(truth_classification[i]) > 1:
            where_clean = np.argwhere(np.array(truth_classification[i]) == "clean")
            where_merged = np.argwhere(np.array(truth_classification[i]) == "merged")

            # "clean" in list cases
            if len(where_clean) > 1:
                # decide which clean assignment is best
                reco_keys = np.array(truth_assignment[i])[
                    np.array(where_clean).reshape((len(where_clean),))
                ]
                diff = (
                    np.abs(target_PVs_loc[i] - pred_PVs_loc[reco_keys])
                    / reco_res[reco_keys]
                )
                best_key = reco_keys[np.argmin(diff)]

                # assign split vertices
                for k in reco_keys:
                    if not k == best_key:
                        reco_classification[k] = ["split"]
                        reco_assignment[k] = [i]

            if len(where_merged) > 1:
                truth_classification[i] = ["merged"]

            if len(truth_classification[i]) > 1:
                truth_classification[i] = ["merged"]

    # add up each category
    truth_classification = np.array(truth_classification).reshape(
        len(truth_classification),
    )
    reco_classification = np.array(reco_classification).reshape(
        len(reco_classification),
    )
    # calculates local pileup density
    bins_1mm = 12000 / (240 - (-240))
    localdensity = np.zeros(len(target_PVs_loc))
    for i in range(len(target_PVs_loc)):
        localdensity[i] = (
            sum(np.abs(target_PVs_loc[i] - target_PVs_loc) <= bins_1mm) / 2
        )

    # calculate number of merged, split, fake, clean
    # Handle both scalar and array cases
    reco_class_array = np.atleast_1d(reco_classification)
    reco_merged = np.sum(reco_class_array == "merged")
    reco_split = np.sum(reco_class_array == "split")
    reco_clean = np.sum(reco_class_array == "clean")
    reco_fake = np.sum(reco_class_array == "fake")

    return (
        PerformanceInfo(reco_merged, reco_split, reco_clean, reco_fake),
        truth_classification,
        localdensity,
    )


# -- compare_res_reco2 (from efficiency_res_optimized_atlas.py lines 636-782)


def compare_res_reco2(
    target_PVs_loc: np.ndarray,
    pred_PVs_loc: np.ndarray,
    reco_res: np.ndarray,
    debug: bool,
) -> tuple[int, int, int]:
    """Compute the efficiency counters:
    - succeed    = number of successfully predicted PVs
    - missed     = number of missed true PVs
    - false_pos  = number of predicted PVs not matching any true PVs

    Inputs argument:
      * target_PVs_loc:
          Numpy array of computed z positions of the true PVs (using KDEs)

      * pred_PVs_loc:
          Numpy array of computed z positions of the predicted PVs (using KDEs)

      * reco_res:
          Numpy array with the "reco" resolution as a function of width of predicted KDE signal

      * debug:
          flag to print output for debugging purposes


    Returns:
        succeed, missed, false_pos
    """

    # Counters that will be iterated and returned by this method
    succeed = 0
    missed = 0
    false_pos = 0

    # Get the number of predicted PVs
    len_pred_PVs_loc = len(pred_PVs_loc)
    # Get the number of true PVs
    len_target_PVs_loc = len(target_PVs_loc)

    # Decide whether we have predicted equally or more PVs than trully present
    # this is important, because the logic for counting the MT an FP depend on this
    if len_pred_PVs_loc >= len_target_PVs_loc:
        if debug:
            print("In len(pred_PVs_loc) >= len(target_PVs_loc)")

        # Since we have N(pred_PVs) >= N(true_PVs),
        # we loop over the pred_PVs, and check each one of them to decide
        # whether they should be labelled as S, FP.
        # The number of MT is computed as: N(true_PVs) - S
        # Here the number of iteration is fixed to the original number of predicted PVs
        for i in range(len_pred_PVs_loc):
            if debug:
                print("pred_PVs_loc = ", pred_PVs_loc[i])
            # flag to check if the predicted PV is being matched to a true PV
            matched = 0

            # Get the window of interest: [min_val, max_val]
            # The window is obtained from the value of z of the true PV 'j'
            # +/- the resolution as a function of the number of tracks for the true PV 'j'
            min_val = pred_PVs_loc[i] - reco_res[i]
            max_val = pred_PVs_loc[i] + reco_res[i]
            if debug:
                print("resolution = ", (max_val - min_val) / 2.0)
                print("min_val = ", min_val)
                print("max_val = ", max_val)

            # Now looping over the true PVs.
            for j in range(len(target_PVs_loc)):
                # If condition is met, then the predicted PV is labelled as 'matched',
                # and the number of success is incremented by 1
                if min_val <= target_PVs_loc[j] and target_PVs_loc[j] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ", succeed)
                    # the true PV is removed from the original array to avoid associating
                    # one predicted PV to multiple true PVs
                    # (this could happen for PVs with close z values)
                    target_PVs_loc = np.delete(np.array(target_PVs_loc), [j])
                    # Since a predicted PV and a true PV where matched, go to the next predicted PV 'i'
                    break
            # In case, no true PV could be associated with the predicted PV 'i'
            # then it is assigned as a FP answer
            if not matched:
                false_pos += 1
                if debug:
                    print("false_pos = ", false_pos)
        # the number of missed true PVs is simply the difference between the original
        # number of true PVs and the number of successfully matched true PVs
        missed = len_target_PVs_loc - succeed
        if debug:
            print("missed = ", missed)

    else:
        if debug:
            print("In len(pred_PVs_loc) < len(target_PVs_loc)")
        # Since we have N(pred_PVs) < N(true_PVs),
        # we loop over the true_PVs, and check each one of them to decide
        # whether they should be labelled as S, MT.
        # The number of FP is computed as: N(pred_PVs) - S
        # Here the number of iteration is fixed to the original number of true PVs
        for i in range(len_target_PVs_loc):
            if debug:
                print("target_PVs_loc = ", target_PVs_loc[i])
            # flag to check if the true PV is being matched to a predicted PV
            matched = 0
            # Now looping over the predicted PVs.
            for j in range(len(pred_PVs_loc)):
                # Get the window of interest: [min_val, max_val]
                # The window is obtained from the value of z of the true PV 'i'
                # +/- the resolution as a function of the number of tracks for the true PV 'i'
                min_val = pred_PVs_loc[j] - reco_res[j]
                max_val = pred_PVs_loc[j] + reco_res[j]
                if debug:
                    print("pred_PVs_loc = ", pred_PVs_loc[j])
                    print("resolution = ", (max_val - min_val) / 2.0)
                    print("min_val = ", min_val)
                    print("max_val = ", max_val)
                # If condition is met, then the true PV is labelled as 'matched',
                # and the number of success is incremented by 1
                if min_val <= target_PVs_loc[i] and target_PVs_loc[i] <= max_val:
                    matched = 1
                    succeed += 1
                    if debug:
                        print("succeed = ", succeed)
                    # the predicted PV is removed from the original array to avoid associating
                    # one true PV to multiple predicted PVs
                    # (this could happen for PVs with close z values)
                    pred_PVs_loc = np.delete(pred_PVs_loc, [j])
                    # Since a predicted PV and a true PV where matched, go to the next true PV 'i'
                    reco_res = np.delete(reco_res, [j])
                    break
            # In case, no predicted PV could be associated with the true PV 'i'
            # then it is assigned as a MT answer
            if not matched:
                missed += 1
                if debug:
                    print("missed = ", missed)

        # the number of false positive predicted PVs is simply the difference between the original
        # number of predicted PVs and the number of successfully matched predicted PVs
        false_pos = len_pred_PVs_loc - succeed
        if debug:
            print("false_pos = ", false_pos)

    return succeed, missed, false_pos


# -- Resolution fitting (kept from current vertex_matching.py) ---------------


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
