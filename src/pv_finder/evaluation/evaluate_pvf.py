# gets info from pv-finder output needed to generate performance plots
import argparse
import pickle

import numpy as np
import uproot
from scipy.signal import find_peaks

from pv_finder.evaluation.vertex_matching import (
    _pv_locations_updated_res as pv_locations_updated_res,
)
from pv_finder.evaluation.vertex_matching import (
    compare_res_reco,
    filter_nans_res,
)
from pv_finder.utils.efficiency import efficiency

# parameters for LHCb-style efficiency
PARAM_EFF = {
    "difference": 5.0,  # maximum # bins between truth and predicted
    "threshold": 1e-2,  # start counting towards reconstructed PV when over this value
    "integral_threshold": 0.2,  # integral of reconstructed PV distribution must be over this value to be valid
    "min_width": 3,  # number of bins required for reconstructed PV to be valid
}


def main(
    dir_name,
    which_model,
    path_hdf5,
    path_root,
    nevents,
    index_path,
    sigma_vtx_vtx,
    use_label_truth=False,
    label_peak_thresh=0.1,
):
    # get indices (use same shuffle as when running TestModel.py)
    indices = pickle.load(open(index_path, "rb"))
    nevents = len(indices)
    start = 0
    end = nevents

    # get truth pv information
    if not use_label_truth:
        print("Loading truth from ROOT file...")
        from collections import namedtuple

        import awkward

        VertexInfo = namedtuple("VertexInfo", ("x", "y", "z", "n"))
        root_file = uproot.open(path_root)
        tree = root_file["PVFinderData"]
        all_x = tree["TruthVertex_x"].array()
        all_y = tree["TruthVertex_y"].array()
        all_z = tree["TruthVertex_z"].array()
        all_n = tree["TruthVertex_nTracks"].array()
        x_list = awkward.Array([all_x[i] for i in indices])
        y_list = awkward.Array([all_y[i] for i in indices])
        z_list = awkward.Array([all_z[i] for i in indices])
        n_list = awkward.Array([all_n[i] for i in indices])
        truth = VertexInfo(x_list, y_list, z_list, n_list)
        print("truth collected from ROOT file")
    else:
        print("Will use label-based truth (peaks from KDE labels)")
        truth = None  # Will extract truth from labels per-event

    # load inputs, labels, outputs obtained from running TestModel.py
    inputs = pickle.load(open(f"{dir_name}/inputs_pvf_fullcov_{which_model}.p", "rb"))
    labels = pickle.load(open(f"{dir_name}/labels_pvf_fullcov_{which_model}.p", "rb"))
    outputs = pickle.load(open(f"{dir_name}/outputs_pvf_fullcov_{which_model}.p", "rb"))

    # Handle multi-channel labels (select first channel only)
    labels = np.array(labels)
    if labels.ndim == 3 and labels.shape[1] > 1:
        print(f"Labels have {labels.shape[1]} channels, selecting first channel only")
        labels = labels[:, 0, :]  # Select first channel
    elif labels.ndim == 3:
        labels = labels.squeeze(1)  # Remove channel dimension if only 1 channel

    # Ensure consistent dtypes
    labels = labels.astype(np.float32)
    outputs = np.array(outputs, dtype=np.float32)

    # binning info
    totalNumBins = 12000
    zMax = 240
    zMin = -240
    binwidth = (zMax - zMin) / totalNumBins
    np.linspace(zMin, zMax, totalNumBins, endpoint=False) + binwidth / 2
    np.linspace(zMin, zMax, totalNumBins)
    totalNumBins / (zMax - zMin)

    ####### define efficiency parameters #######
    threshold = 1e-2  # above this value, start calculating predicted PV
    integral_threshold = (
        0.2  # require this integral or above when finding predicted PV (Qi Bin's value)
    )
    min_width = 3  # require predicted PV to be at least this number of bins wide

    nevts = end - start

    # initialize arrays to keep number of reconstructed vertices in
    reco_merged = np.zeros(nevts)
    reco_split = np.zeros(nevts)
    reco_clean = np.zeros(nevts)
    reco_fake = np.zeros(nevts)
    event_efficiency = np.zeros(nevts)

    # list to contain distances between reconstructed PVs
    predicted_pv_distances = []

    # for calculating efficiency as a function of ntrks
    truth_correct = []
    truth_ntrks = []

    # for lhcb-style efficiency
    S_total = 0
    Sp_total = 0
    MT_total = 0
    FP_total = 0

    total_reco_z = []
    for iEvt in range(nevts):
        print("iEvt = ", iEvt)

        # get current neural network info
        outputs_current = outputs[iEvt]
        inputs[iEvt]
        labels_current = labels[iEvt]

        # get current truth info
        if use_label_truth:
            # Extract truth from label peaks
            peaks, _ = find_peaks(labels_current, height=label_peak_thresh)
            truth_z_bins_current = peaks.astype(float)  # Already in bins
            truth_n_current = np.ones(len(peaks)) * 10  # Dummy nTracks (all valid)
            truth_z_current = (
                truth_z_bins_current * binwidth + zMin
            )  # Convert to mm for compatibility
        else:
            sortind = np.argsort(truth.z[iEvt])
            truth_n_current = truth.n[iEvt][sortind]
            truth.x[iEvt][sortind]
            truth.y[iEvt][sortind]
            truth_z_current = truth.z[iEvt][sortind]

        # get locations of predited PVs
        (
            predict_loc,
            predict_peakval,
            predict_peakpos,
            predict_cleft,
            predict_cright,
            predict_sigma,
        ) = pv_locations_updated_res(
            outputs_current, threshold, integral_threshold, min_width
        )

        # predict_loc is in mm from pv_locations_updated_res
        # Store for distance calculation
        total_reco_z.extend(predict_loc)

        # get distances between nearby vertices (using mm)
        predict_loc_mm = predict_loc.copy()
        np.random.shuffle(predict_loc_mm)
        for i in range(len(predict_loc_mm) - 1):
            for j in range(i + 1, len(predict_loc_mm)):
                predicted_pv_distances.append(predict_loc_mm[i] - predict_loc_mm[j])

        # Convert to bins for filtering and comparison
        predict_loc_bins = (predict_loc - zMin) / binwidth

        # whether each predicted PV is valid (masked == False)
        # filter_nans_res expects locations in bins
        validinds = filter_nans_res(predict_loc_bins, labels_current)
        validinds = list(
            range(len(predict_loc))
        )  # uncomment if you do not want to filter

        # update predicted PV list (keep in bins for compare_res_reco)
        filtered_predict_loc_bins = predict_loc_bins[validinds]
        predict_peakpos[validinds]
        predict_cleft[validinds]
        predict_cright[validinds]

        # constant resolution (convert mm to bins for comparison)
        sigma_vtx_vtx_bins = sigma_vtx_vtx / binwidth
        reco_res = sigma_vtx_vtx_bins * np.ones(len(filtered_predict_loc_bins))

        # consider truth vertices with more than 1 associated track
        if use_label_truth:
            truth_z_bins_valid = truth_z_bins_current  # Already in bins, all valid
        else:
            truth_z_bins_valid = (
                truth_z_current[truth_n_current >= 2] - zMin
            ) / binwidth

        # count reconstructed vertices and return related info
        current_result, truth_classification, localdensity = compare_res_reco(
            truth_z_bins_valid, filtered_predict_loc_bins, reco_res, 0
        )

        # calculate lhcb-style efficiency
        eff = efficiency(labels_current, outputs_current, **PARAM_EFF)
        S_total += eff[0]
        Sp_total += eff[1]
        MT_total += eff[2]
        FP_total += eff[3]

        # truth efficiency info
        truth_correct_event = []
        if use_label_truth:
            truth_n_current_filtered = (
                truth_n_current  # All valid for label-based truth
            )
        else:
            truth_n_current_filtered = truth_n_current[truth_n_current >= 2]
        for i in range(len(truth_classification)):
            truth_ntrks.append(truth_n_current_filtered[i])
            if (
                "clean" in truth_classification[i]
                or "merged" in truth_classification[i]
            ):
                truth_correct.append(1)
                truth_correct_event.append(1)
            else:
                truth_correct.append(0)
                truth_correct_event.append(0)

        print(current_result)

        reco_merged[iEvt] = current_result.reco_merged
        reco_split[iEvt] = current_result.reco_split
        reco_clean[iEvt] = current_result.reco_clean
        reco_fake[iEvt] = current_result.reco_fake
        event_efficiency[iEvt] = sum(truth_correct_event) / len(truth_correct_event)

    # load pileup information from ROOT tree. be careful to ensure that the indices correspond to the data loaded above
    PVFinderData = uproot.open(path_root)["PVFinderData"]
    ActualNumOfInt = list(PVFinderData["ActualNumOfInt"].array()[indices])

    # dictionaries with key corresponding to pileup and values = lists of numbers of clean/split/merged/fake from different events
    separated_clean = {}
    separated_merged = {}
    separated_split = {}
    separated_fake = {}
    separated_all = {}
    separated_eff = {}

    for i in range(len(reco_clean)):
        if np.rint(ActualNumOfInt[i]) not in separated_clean.keys():
            separated_clean[np.rint(ActualNumOfInt[i])] = []
        separated_clean[np.rint(ActualNumOfInt[i])].append(reco_clean[i])

        if np.rint(ActualNumOfInt[i]) not in separated_merged.keys():
            separated_merged[np.rint(ActualNumOfInt[i])] = []
        separated_merged[np.rint(ActualNumOfInt[i])].append(reco_merged[i])

        if np.rint(ActualNumOfInt[i]) not in separated_split.keys():
            separated_split[np.rint(ActualNumOfInt[i])] = []
        separated_split[np.rint(ActualNumOfInt[i])].append(reco_split[i])

        if np.rint(ActualNumOfInt[i]) not in separated_fake.keys():
            separated_fake[np.rint(ActualNumOfInt[i])] = []
        separated_fake[np.rint(ActualNumOfInt[i])].append(reco_fake[i])

        if np.rint(ActualNumOfInt[i]) not in separated_all.keys():
            separated_all[np.rint(ActualNumOfInt[i])] = []
        separated_all[np.rint(ActualNumOfInt[i])].append(
            reco_fake[i] + reco_clean[i] + reco_merged[i] + reco_split[i]
        )

        if np.rint(ActualNumOfInt[i]) not in separated_eff.keys():
            separated_eff[np.rint(ActualNumOfInt[i])] = []
        separated_eff[np.rint(ActualNumOfInt[i])].append(event_efficiency[i])

    # save these dictionaries
    pickle.dump(
        separated_all, open(f"{dir_name}/separated_all_pvf_{which_model}.p", "wb")
    )
    pickle.dump(
        separated_clean, open(f"{dir_name}/separated_clean_pvf_{which_model}.p", "wb")
    )
    pickle.dump(
        separated_merged, open(f"{dir_name}/separated_merged_pvf_{which_model}.p", "wb")
    )
    pickle.dump(
        separated_split, open(f"{dir_name}/separated_split_pvf_{which_model}.p", "wb")
    )
    pickle.dump(
        separated_fake, open(f"{dir_name}/separated_fake_pvf_{which_model}.p", "wb")
    )
    pickle.dump(
        separated_eff, open(f"{dir_name}/separated_eff_pvf_{which_model}.p", "wb")
    )

    print("Efficiency = ", sum(truth_correct) / len(truth_correct))
    print("FPR = ", np.average(reco_fake))
    print("Total Length = ", len(truth_correct))

    print("LHCb-style Efficiency = ", S_total / (S_total + MT_total))
    print("LHCb-style FalsePos = ", FP_total / len(indices))

    # save predicted truth efficiency and ntrk info
    pickle.dump(
        truth_correct, open(f"{dir_name}/truth_correct_pvf_{which_model}.p", "wb")
    )
    pickle.dump(truth_ntrks, open(f"{dir_name}/truth_ntrks_pvf_{which_model}.p", "wb"))
    pickle.dump(total_reco_z, open(f"{dir_name}/total_reco_z_{which_model}.p", "wb"))

    pickle.dump(
        predicted_pv_distances,
        open(f"{dir_name}/predicted_pv_distances_pvf_{which_model}.p", "wb"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This program finds the number of reconstructed vertices in each category for a PV-Finder model and saves the outputs"
    )
    parser.add_argument(
        "-n",
        "--nevents",
        help="Number of events in input file (95% of these are used for training)",
        default=51000,
        type=int,
    )
    parser.add_argument(
        "-i",
        "--indices",
        help="Pickled numpy array containing the indices to use for consistent train/test split (use same file as for training)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--dirname",
        help="directory to load pv-finder results from",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--modelname",
        help="name of model architecture (i.e. unet or unetplusplus)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f", "--path_hdf5", help="path to input hdf5 file", type=str, required=True
    )
    parser.add_argument(
        "-r",
        "--path_root",
        help="path to input root file (the same file used to generate the hdf5 file)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s", "--sigma", help="value of sigma_vtx_vtx to use", type=float, required=True
    )
    parser.add_argument(
        "--use_label_truth",
        help="Use peaks from KDE labels as truth instead of ROOT file",
        action="store_true",
    )
    parser.add_argument(
        "--label_peak_thresh",
        help="Height threshold for finding peaks in labels (default: 0.1)",
        type=float,
        default=0.1,
    )

    args = parser.parse_args()

    main(
        args.dirname,
        args.modelname,
        args.path_hdf5,
        args.path_root,
        args.nevents,
        args.indices,
        args.sigma,
        args.use_label_truth,
        args.label_peak_thresh,
    )
