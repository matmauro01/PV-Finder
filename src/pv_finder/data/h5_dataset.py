from __future__ import annotations

from typing import Sequence

import h5py
import numpy as np
from torch.utils.data import ConcatDataset, Dataset

# Mask sentinel for padded track slots — must match `MASK_VAL` in root_to_h5.py.
_TRACK_MASK_VAL = -999999.0


# H5 class to expedite DataLoading process for tracks to KDE
class H5Dataset_tracksKDE(Dataset):
    def __init__(self, filename):
        self.filename = filename

        with h5py.File(filename, "r") as dataset:
            # Get dataset length before file closes
            self.dataset_len = len(dataset["tracks"])
        # Note: we don't store references to dataset arrays here
        # because the file is closed. Access them in __getitem__ instead.

    def __len__(self):
        return self.dataset_len

    # Returns desired batch size of datafile
    def __getitem__(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            tracks = dataset["tracks"][idx]
            kde_split = dataset["kde_split"][idx]
        return tracks, kde_split

    # Getter functions for each of the variables
    def getTracks(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            tracks = dataset["tracks"][idx]
        return tracks

    def getKDE(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            kde = dataset["kde"][idx]
        return kde

    def getKDE_split(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            kde_split = dataset["kde_split"][idx]
        return kde_split

    def getTargetY(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            target_y = dataset["target_y"][idx]
        return target_y

    def getPV(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            pv = dataset["pv"][idx]
        return pv


# H5 class to expedite DataLoading process for tracks to hists
class H5Dataset_tracksHists(Dataset):
    """Per-file tracks->hist dataset.

    Tolerates files produced with ``--skip-target-y`` (no full-event
    ``target_y`` dataset). Accepts an optional ``target_max_tracks``: when
    set and larger than the file's own ``max_tracks_per_subevent``, the
    returned ``tracks`` tensor is right-padded with the mask sentinel to
    that width so it stacks cleanly with items from other files in a
    ``ConcatDataset`` batch.
    """

    def __init__(self, filename, target_max_tracks: int | None = None):
        self.filename = filename
        with h5py.File(filename, "r") as dataset:
            self.dataset_len = len(dataset["tracks"])
            self._local_max_tracks = int(dataset["tracks"].shape[-1])
            self._has_target_y = "target_y" in dataset
        if target_max_tracks is not None and target_max_tracks < self._local_max_tracks:
            raise ValueError(
                f"target_max_tracks={target_max_tracks} is smaller than this "
                f"file's stored max_tracks_per_subevent={self._local_max_tracks} "
                f"({filename}); refusing to truncate tracks."
            )
        self._target_max_tracks = target_max_tracks
        # Lazy-open the h5 handle on first __getitem__ inside each DataLoader
        # worker. h5py.File objects don't pickle, so we cannot open in __init__
        # when num_workers>0. With persistent_workers=True the handle stays
        # alive for the run; without persistent_workers it reopens per epoch.
        self._h5: h5py.File | None = None

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.filename, "r")
        return self._h5

    def __getstate__(self) -> dict:
        # Drop the (possibly opened) h5 handle before pickling to a worker.
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __len__(self):
        return self.dataset_len

    def _pad_tracks(self, tracks: np.ndarray) -> np.ndarray:
        if (
            self._target_max_tracks is None
            or tracks.shape[-1] == self._target_max_tracks
        ):
            return tracks
        out = np.full(
            tracks.shape[:-1] + (self._target_max_tracks,),
            _TRACK_MASK_VAL,
            dtype=tracks.dtype,
        )
        out[..., : tracks.shape[-1]] = tracks
        return out

    def __getitem__(self, idx):
        ds = self._open()
        tracks = ds["tracks"][idx]
        target_y_split = ds["target_y_split"][idx]
        return self._pad_tracks(tracks), target_y_split

    def getTracks(self, idx):
        return self._pad_tracks(self._open()["tracks"][idx])

    def getTargetY(self, idx):
        if not self._has_target_y:
            raise KeyError(
                f"{self.filename} was produced with --skip-target-y; "
                "full-event target_y is not in this file. Use "
                "getTargetY_split() instead."
            )
        return self._open()["target_y"][idx]

    def getTargetY_split(self, idx):
        return self._open()["target_y_split"][idx]

    def getPV(self, idx):
        return self._open()["pv"][idx]


def make_tracksHists_dataset(
    paths: str | Sequence[str],
) -> Dataset:
    """Build a tracks->hist dataset from one or many HDF5 files.

    Single path: returns a plain ``H5Dataset_tracksHists``.
    List of paths: opens each file's ``max_tracks_per_subevent`` attribute,
    determines the global max, builds a per-file dataset that right-pads
    tracks to that width, and returns a ``ConcatDataset`` so the trainer
    can iterate them as one stream. ``target_y_split`` has identical shape
    across files (always (2, 1000)) so it needs no padding.
    """
    if isinstance(paths, str):
        return H5Dataset_tracksHists(paths)

    paths = list(paths)
    if not paths:
        raise ValueError("make_tracksHists_dataset: empty path list")
    if len(paths) == 1:
        return H5Dataset_tracksHists(paths[0])

    per_file_max: list[int] = []
    ty_split_shape: tuple[int, ...] | None = None
    for p in paths:
        with h5py.File(p, "r") as f:
            local = f.attrs.get("max_tracks_per_subevent")
            if local is None:
                local = int(f["tracks"].shape[-1])
            per_file_max.append(int(local))
            this_shape = tuple(int(s) for s in f["target_y_split"].shape[1:])
            if ty_split_shape is None:
                ty_split_shape = this_shape
            elif this_shape != ty_split_shape:
                raise ValueError(
                    f"target_y_split shape mismatch across files: "
                    f"{ty_split_shape} vs {this_shape} in {p}"
                )
    global_max = max(per_file_max)
    return ConcatDataset(
        [H5Dataset_tracksHists(p, target_max_tracks=global_max) for p in paths]
    )


# H5 class to expedite DataLoading process for kde to hists
class H5Dataset_kdeHists(Dataset):
    def __init__(self, filename):
        self.filename = filename

        with h5py.File(filename, "r") as dataset:
            # self.tracks = dataset['tracks']
            # self.kde = dataset['kde']
            self.kde_split = dataset["kde_split"]
            self.target_y = dataset["target_y_split"]
            self.pv = dataset["pv"]
            self.dataset_len = len(self.kde_split)

    def __len__(self):
        return self.dataset_len

    # Returns desired batch size of datafile
    def __getitem__(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            kde = dataset["kde_split"][
                idx, 0:1, :
            ]  # Load ONLY KDE-A-z (1 channel) for 1-channel models
            target_y = dataset["target_y_split"][idx]
        return kde, target_y

    def getKDE_split(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            kde_split = dataset["kde_split"][idx]
        return kde_split

    def getTargetY(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            target_y = dataset["target_y"][idx]
        return target_y

    def getTargetY_split(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            target_y = dataset["target_y_split"][idx]
        return target_y

    def getPV(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            pv = dataset["pv"][idx]
        return pv


# H5 class to expedite DataLoading process for poca parameters to KDE
class H5Dataset_pocaKDE(Dataset):
    def __init__(self, filename):
        self.filename = filename

        with h5py.File(filename, "r") as dataset:
            self.poca_split = dataset["poca_split"]
            self.kde_split = dataset["kde_split"]
            #             self.kde = dataset['kde']
            #             self.target_y = dataset['target_y']
            #             self.pv = dataset['pv']
            self.dataset_len = len(self.poca_split)

    def __len__(self):
        return self.dataset_len

    # Returns desired batch size of datafile
    def __getitem__(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            poca_split = dataset["poca_split"][idx]
            kde_split = dataset["kde_split"][idx]
        return poca_split, kde_split

    # Getter functions for each of the variables
    def getPOCA(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            poca_split = dataset["poca_split"][idx]
        return poca_split

    def getKDE_split(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            kde_split = dataset["kde_split"][idx]
        return kde_split


# H5 class to expedite DataLoading process for poca parameters to KDE
class H5Dataset_pocaHists(Dataset):
    def __init__(self, filename):
        self.filename = filename

        with h5py.File(filename, "r") as dataset:
            self.poca_split = dataset["poca_split"]
            self.target_y_split = dataset["target_y_split"]
            #             self.kde_split = dataset['kde_split']
            #             self.kde = dataset['kde']
            #             self.target_y = dataset['target_y']
            #             self.pv = dataset['pv']
            self.dataset_len = len(self.poca_split)

    def __len__(self):
        return self.dataset_len

    # Returns desired batch size of datafile
    def __getitem__(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            poca_split = dataset["poca_split"][idx]
            target_y_split = dataset["target_y_split"][idx]
        return poca_split, target_y_split

    # Getter functions for each of the variables
    def getPOCA(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            poca_split = dataset["poca_split"][idx]
        return poca_split

    def getTargetY_split(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            target_y_split = dataset["target_y_split"][idx]
        return target_y_split
