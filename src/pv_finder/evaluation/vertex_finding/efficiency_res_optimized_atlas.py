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
    """Classify reco and truth vertices into clean/merged/split/fake/missed.

    Uses greedy closest-first matching: all (reco, truth) pairs within the
    matching window are sorted by distance, then greedily assigned 1-to-1.
    After assignment, leftover reco with unmatched truth in their window are
    "merged"; leftover reco with multiple truth assigned are resolved via
    split logic.

    Parameters
    ----------
    target_PVs_loc : ndarray
        Truth vertex z positions (bin units). Should be filtered to nTracks>=2.
    pred_PVs_loc : ndarray
        Predicted (reco) vertex z positions (bin units).
    reco_res : ndarray
        Matching window per reco vertex (bin units), same length as pred_PVs_loc.
    debug : int
        Print debug output if > 0.

    Returns
    -------
    (PerformanceInfo, truth_classification, localdensity)
    """
    n_truth = len(target_PVs_loc)
    n_reco = len(pred_PVs_loc)

    # --- Pass 1: build candidate pairs and greedy closest-first assignment ---
    # Collect all (reco_i, truth_j, distance) pairs within matching window
    pairs = []
    # Also track which truth vertices fall within each reco's window
    reco_truth_neighbors = [[] for _ in range(n_reco)]
    for ri in range(n_reco):
        dists = np.abs(target_PVs_loc - pred_PVs_loc[ri])
        for tj in np.where(dists <= reco_res[ri])[0]:
            pairs.append((ri, int(tj), float(dists[tj])))
            reco_truth_neighbors[ri].append(int(tj))

    # Sort by distance (closest first) and greedily assign 1-to-1
    pairs.sort(key=lambda x: x[2])
    reco_assigned = {}  # ri -> tj
    truth_assigned = {}  # tj -> ri
    for ri, tj, _ in pairs:
        if ri not in reco_assigned and tj not in truth_assigned:
            reco_assigned[ri] = tj
            truth_assigned[tj] = ri

    # Primaries = truths that won a dedicated 1-to-1 reco in Pass 1. These are
    # cleanly reconstructed even if their reco later absorbs a neighbour; only
    # the *absorbed* (Pass-2) truths are the merge casualties.
    primary_truth = set(truth_assigned)

    # --- Pass 2: classify ---
    reco_cls = np.empty(n_reco, dtype=object)
    truth_cls = np.empty(n_truth, dtype=object)

    # Classify assigned reco: merged if unmatched truth in window, else clean
    for ri in range(n_reco):
        neighbors = reco_truth_neighbors[ri]
        if ri not in reco_assigned:
            # Unassigned reco: "split" if it had truth in its window (those
            # truth were claimed by closer reco), "fake" otherwise
            if neighbors:
                reco_cls[ri] = "split"
            else:
                reco_cls[ri] = "fake"
        else:
            # Check if any truth in this reco's window is unmatched (missed)
            unmatched_neighbors = [tj for tj in neighbors if tj not in truth_assigned]
            if unmatched_neighbors:
                reco_cls[ri] = "merged"
                for tj in unmatched_neighbors:
                    truth_assigned[tj] = ri  # claim them
            else:
                reco_cls[ri] = "clean"

    # Truth classification: a Pass-1 primary has its own dedicated reco -> clean
    # (even if that reco absorbed a neighbour); a truth absorbed in Pass 2 shares
    # a reco with a closer truth and had none of its own -> merged; else missed.
    # NOTE: this only re-labels assigned truths between clean/merged; the set of
    # non-missed truths (and hence efficiency = (clean+merged)/n_truth) is unchanged.
    for tj in range(n_truth):
        if tj in primary_truth:
            truth_cls[tj] = "clean"
        elif tj in truth_assigned:
            truth_cls[tj] = "merged"
        else:
            truth_cls[tj] = "missed"

    if debug:
        for ri in range(n_reco):
            tj = reco_assigned.get(ri, -1)
            print(
                f"  reco {ri}: {reco_cls[ri]}, assigned to truth {tj}, "
                f"neighbors={reco_truth_neighbors[ri]}"
            )

    # Local pileup density (unchanged from original)
    bins_1mm = 12000 / (240 - (-240))
    localdensity = np.zeros(n_truth)
    for i in range(n_truth):
        localdensity[i] = (
            sum(np.abs(target_PVs_loc[i] - target_PVs_loc) <= bins_1mm) / 2
        )

    reco_merged = int(np.sum(reco_cls == "merged"))
    reco_split = int(np.sum(reco_cls == "split"))
    reco_clean = int(np.sum(reco_cls == "clean"))
    reco_fake = int(np.sum(reco_cls == "fake"))

    return (
        PerformanceInfo(reco_merged, reco_split, reco_clean, reco_fake),
        truth_cls,
        localdensity,
    )
