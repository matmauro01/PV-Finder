import h5py
from torch.utils.data import Dataset


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
    def __init__(self, filename):
        self.filename = filename

        with h5py.File(filename, "r") as dataset:
            self.tracks = dataset["tracks"]
            self.target_y = dataset["target_y"]
            self.target_y_split = dataset["target_y_split"]
            self.pv = dataset["pv"]
            self.dataset_len = len(self.tracks)

    def __len__(self):
        return self.dataset_len

    # Returns desired batch size of datafile
    def __getitem__(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            tracks = dataset["tracks"][idx]
            target_y_split = dataset["target_y_split"][idx]
        return tracks, target_y_split

    # Getter functions for each of the variables
    def getTracks(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            tracks = dataset["tracks"][idx]
        return tracks

    def getTargetY(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            target_y = dataset["target_y"][idx]
        return target_y

    def getTargetY_split(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            target_y_split = dataset["target_y_split"][idx]
        return target_y_split

    def getPV(self, idx):
        with h5py.File(self.filename, "r") as dataset:
            pv = dataset["pv"][idx]
        return pv


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
