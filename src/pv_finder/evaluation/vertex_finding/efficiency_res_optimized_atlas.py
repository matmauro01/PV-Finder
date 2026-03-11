"""Peak finding, NMS, and vertex matching for PV-Finder evaluation."""

from collections import namedtuple

import numpy as np

PerformanceInfo = namedtuple(
    "PerformanceInfo", ("reco_merged", "reco_split", "reco_clean", "reco_fake")
)


def pv_locations_updated_res(targets, threshold, integral_threshold, min_width):
    """
    Compute the z positions from the input KDE using the parsed criteria.

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
      * array of float32 values corresponding to the PV z positions

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

    bin_width_val = 0.04
    z_min_val = -240

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
        #         print("i = ", i, ", value = ", targets[i], ", state = ", state, ", currentmax = ", currentmax)
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
            # if (targets[i] < threshold or i == len(targets) - 1) and state > 0:

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


#####################################################################################


def suppress_neighbor_peaks(
    positions: np.ndarray,
    heights: np.ndarray,
    min_sep: float = 0.85,
    max_ratio: float = 0.3,
) -> np.ndarray:
    """Remove satellite peaks that are close to a taller neighbor.

    For each pair of peaks separated by less than *min_sep* mm, suppress the
    shorter one **only if** its height is less than *max_ratio* times the
    taller peak's height.  This preserves genuine close vertex pairs (which
    tend to have similar heights) while killing UNet sidelobe fakes (which
    are much shorter than their parent peak).

    Returns a boolean mask of peaks to **keep**.
    """
    n = len(positions)
    keep = np.ones(n, dtype=bool)
    order = np.argsort(-heights)  # tallest first
    for idx in order:
        if not keep[idx]:
            continue
        for jdx in order:
            if jdx == idx or not keep[jdx]:
                continue
            if abs(positions[idx] - positions[jdx]) < min_sep:
                if heights[jdx] / heights[idx] < max_ratio:
                    keep[jdx] = False
    return keep


#####################################################################################


def compare_res_reco(target_PVs_loc, pred_PVs_loc, reco_res, debug):
    """

    Inputs argument:
      * target_PVs_loc:
          Numpy array of z positions (in terms of bin #) of the TruthVertex objects. ensure that the list is filtered to require ntrks>=4

      * pred_PVs_loc:
          Numpy array of computed z positions (in terms of bin #) of the predicted PVs (using KDEs)

      * reco_res:
          Numpy array with the "reco" resolution as a function of width of predicted KDE signal

      * debug:
          flag to print output for debugging purposes


    Returns:
        PerformanceInfo named tuple
    """

    truth_classification = [[]] * len(target_PVs_loc)
    reco_classification = [[]] * len(pred_PVs_loc)
    truth_assignment = [[]] * len(target_PVs_loc)
    reco_assignment = [[]] * len(pred_PVs_loc)

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
    reco_merged = sum(reco_classification == "merged")
    reco_split = sum(reco_classification == "split")
    reco_clean = sum(reco_classification == "clean")
    reco_fake = sum(reco_classification == "fake")

    return (
        PerformanceInfo(reco_merged, reco_split, reco_clean, reco_fake),
        truth_classification,
        localdensity,
    )
