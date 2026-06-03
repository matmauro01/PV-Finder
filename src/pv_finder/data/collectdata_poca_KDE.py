#########################################################################################
# This file contains the methods needed to load the training features from the hdf5     #
# file (output of CreatingTargetHistogram.py)                                           #
# Usage: python CreatingTargetHistogram.py -i inputfile.root -o outputfile.h5           #
#########################################################################################

# adapted from https://gitlab.cern.ch/LHCb-Reco-Dev/pv-finder/-/blob/kernel_histograms_from_poca_ellipsoids/model/collectdata_poca_KDE.py
import warnings
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from pv_finder.data.h5_dataset import (
    H5Dataset_kdeHists,
    H5Dataset_pocaHists,
    H5Dataset_pocaKDE,
    H5Dataset_tracksKDE,
    make_tracksHists_dataset,
)
from pv_finder.utils.utilities import Timer

# This can throw a warning about float - let's hide it for now.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import h5py

import awkward

VertexInfo = namedtuple("VertexInfo", ("x", "y", "z", "n"))


def collect_data_poca_ATLAS(
    filepath,
    data_pipeline,
    batch_size=32,
    device=None,
    masking=False,
    num_workers=16,
    prefetch_factor=2,
    train_split=[0.7, 0.15, 0.05],
    **kargs,
):
    """
    This function collects data.
    HARD CODED: only allows for one file for now, check prior function at bottom of document for how it was done prior
    Example: collect_data_poca('a.h5', 'b.h5')
    batch_size: The number of events per batch
    dtype: Select a different dtype (like float16)
    slice: Allow just a slice of data to be loaded
    device: The device to load onto (CPU by default)
    masking: Turn on or off (default) the masking of hits.
    **kargs: Any other keyword arguments will be passed on to torch's DataLoader
    """
    print("Loading data...")

    if data_pipeline == "tracks-to-KDE":
        print("Preparing dataset for tracks to KDE")
        dataset = H5Dataset_tracksKDE(filepath)
    elif data_pipeline == "KDE-to-hist":
        print("Preparing dataset for KDE to Hists")
        dataset = H5Dataset_kdeHists(filepath)
    elif data_pipeline == "tracks-to-hist":
        # filepath may be a single path or a list of paths (multi-file pool).
        # The factory handles both; multi-file builds a ConcatDataset and pads
        # tracks to the global max_tracks_per_subevent so batches stack.
        if isinstance(filepath, str):
            print("Preparing dataset for tracks to Hists")
        else:
            print(
                f"Preparing tracks-to-Hists multi-file dataset over "
                f"{len(filepath)} files"
            )
        dataset = make_tracksHists_dataset(filepath)
    elif data_pipeline == "poca-to-KDE":
        print("Preparing dataset for poca variables to KDE")
        dataset = H5Dataset_pocaKDE(filepath)
    elif data_pipeline == "poca-to-hist":
        print("Preparing dataset for poca variables to Hists")
        dataset = H5Dataset_pocaHists(filepath)
    else:
        raise TypeError(
            f"Expected data pipeline, but got {data_pipeline}. Try again with one of the following options: tracks-to-KDE, KDE-to-hist, tracks-to-hist, poca-to-KDE, or poca-to-hist."
        )

    # Globally shuffle the dataset index before partitioning so train/val/test
    # each see a representative mix of files (multi-file ConcatDataset case).
    # Seed is fixed so the split is reproducible across runs.
    n = len(dataset)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    train_size = int(n * train_split[0])
    val_size = int(n * train_split[1])
    print(
        f"Train Size: {train_size}  Val Size: {val_size}  "
        f"Test Size: {n - train_size - val_size}"
    )
    train_indices = perm[:train_size]
    val_indices = perm[train_size : train_size + val_size]
    test_indices = perm[train_size + val_size :]
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    np.save("train_indices.npy", train_indices)
    np.save("val_indices.npy", val_indices)
    np.save("test_indices.npy", test_indices)

    # persistent_workers keeps the dataloader worker pool alive across epochs
    # (avoids ~5-15s of HDF5 open/close per epoch when num_workers is large).
    # pin_memory enables fast async CPU->GPU copies. prefetch_factor=4 buys
    # extra batches in flight to mask LZF decompression latency.
    loader_kw = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    train_loader = DataLoader(train_dataset, **loader_kw)
    val_loader = DataLoader(val_dataset, **loader_kw)
    test_loader = DataLoader(test_dataset, **loader_kw)
    print("Created Data Loader")

    return train_loader, val_loader, test_loader


def load_data_from_file(XY_file, indices, dtype, load_xy, load_XandXsq, load_A_and_B):
    with h5py.File(XY_file, "r") as XY:
        # Load KDE data arrays
        X_A = np.array(
            [XY["poca_KDE_A_zdata"][f"Event{i}"] for i in indices], dtype=dtype
        )[:, np.newaxis, :]
        X_B = np.array(
            [XY["poca_KDE_B_zdata"][f"Event{i}"] for i in indices], dtype=dtype
        )[:, np.newaxis, :]

        # Target values
        Y = np.array([XY["Target_Y"][f"Event{i}"][0] for i in indices], dtype=dtype)
        Y_other = np.array(
            [XY["Target_Y"][f"Event{i}"][1] for i in indices], dtype=dtype
        )

        # Compute squared KDE-A if needed
        Xsq = X_A**2 if load_XandXsq else None

        # Optional x and y KDE max coordinates
        x, y = None, None
        if load_xy:
            x = np.array(
                [XY["poca_KDE_A_xmax"][f"Event{i}"] for i in indices], dtype=dtype
            )[:, np.newaxis, :]
            y = np.array(
                [XY["poca_KDE_A_ymax"][f"Event{i}"] for i in indices], dtype=dtype
            )[:, np.newaxis, :]

    return X_A, X_B, Xsq, Y, Y_other, x, y


def collect_truth_ATLAS(h5_file, indices=np.arange(0, 100, 1)):
    """
    This function collects the truth information from files as
    awkward arrays (JaggedArrays). Give it the same files as collect_data.

    indices: which events to load
    """

    # iterate through input files
    msg = f"Loaded {h5_file} in {{time:.4}} s"
    with Timer(msg), h5py.File(h5_file, mode="r") as XY:
        # load truth PV location and number of tracks
        x_list = awkward.Array([list(XY["pv_loc_x"][f"Event{i}"]) for i in indices])
        y_list = awkward.Array([list(XY["pv_loc_y"][f"Event{i}"]) for i in indices])
        z_list = awkward.Array([list(XY["pv_loc_z"][f"Event{i}"]) for i in indices])
        n_list = awkward.Array([list(XY["pv_ntracks"][f"Event{i}"]) for i in indices])

    return VertexInfo(x_list, y_list, z_list, n_list)
