"""
Data loading and feature extraction for MC / Run 3 ATLAS comparisons.

Extracted from compare_feature_distributions.py -- all code verbatim.

CRITICAL: The MC H5 feature channels are decoded as:
    Channel 0: d0        (RAW transverse impact parameter, mm)
    Channel 1: z0        (longitudinal impact parameter, mm)
    Channel 2: d0_err    (uncertainty on d0, mm) -- ALWAYS POSITIVE
    Channel 3: z0_err    (uncertainty on z0, mm) -- ALWAYS POSITIVE
    Channel 4: d0_z0_cov (covariance between d0 and z0, mm^2)
    Channel 5: z_start   (sub-event boundary, mm)
    Channel 6: z_end     (sub-event boundary, mm)
"""

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (must match model training -- see eval_run3_v2.py)
# ---------------------------------------------------------------------------
Z_MIN = -240.0
Z_MAX = 240.0
N_SUBEVENTS = 12
SUBEVENT_WIDTH = 40.0  # mm
SUBEVENT_STARTS = [Z_MIN + i * SUBEVENT_WIDTH for i in range(N_SUBEVENTS)]
N_FEATURES = 7
MASK_VAL = -999999.0
N_TRACKS_PER_SUBEVENT = 100  # padding size used during training

# CORRECT feature channel names -- the whole point of this script
CHANNEL_NAMES = [
    "d0 [mm]",           # channel 0
    "z0 [mm]",           # channel 1
    "d0_err [mm]",       # channel 2  (NOT theta!)
    "z0_err [mm]",       # channel 3  (NOT phi!)
    "d0_z0_cov [mm^2]",  # channel 4
    "z_start [mm]",      # channel 5
    "z_end [mm]",        # channel 6
]
CHANNEL_SHORT = ["d0", "z0", "d0_err", "z0_err", "cov", "z_s", "z_e"]


# ============================================================================
# Utility helpers
# ============================================================================

def safe_percentile(arr, pcts):
    """np.percentile that handles empty arrays gracefully."""
    if len(arr) == 0:
        return [0.0] * len(pcts)
    return list(np.percentile(arr, pcts))


def feature_stats(arr):
    """Compute summary statistics for a 1-D array."""
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "n": int(len(arr)),
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_mc_data(h5_path, n_events=200):
    """Load MC validation events from the HDF5 training file.

    Uses sequential sub-event indices from the validation split
    (70%--95% of the 612000 sub-events, i.e. indices 428400 to 581400).
    Each event consists of 12 contiguous sub-events.

    Returns
    -------
    list of dict
        One dict per event with key 'tracks' -> shape (12, 7, 695).
    """
    import h5py

    print(f"Loading MC data: {h5_path}")
    total_subevents = 612000
    val_start = int(total_subevents * 0.7)    # 428400
    val_end = int(total_subevents * 0.95)      # 581400

    n_subevents_needed = n_events * N_SUBEVENTS
    if val_start + n_subevents_needed > val_end:
        n_subevents_needed = val_end - val_start
        n_events = n_subevents_needed // N_SUBEVENTS
    sub_end = val_start + n_events * N_SUBEVENTS

    with h5py.File(h5_path, "r") as f:
        tracks_all = f["tracks"][val_start:sub_end]  # (N, 7, 695)

    tracks_all = tracks_all.reshape(n_events, N_SUBEVENTS, 7, 695)

    events = []
    for i in range(n_events):
        events.append({
            "tracks": tracks_all[i],  # (12, 7, 695)
            "event_idx": i,
        })

    print(f"  Loaded {n_events} MC validation events "
          f"({n_events * N_SUBEVENTS} sub-events)")
    return events


def load_run3_data(cache_path, min_pileup=3, max_events=200):
    """Load Run 3 events from the pre-extracted NPZ cache.

    Returns a list of dicts with per-event track arrays.
    """
    print(f"Loading Run 3 cache: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)

    z0_all = data["RecoTrack_z0"]
    d0_all = data["RecoTrack_d0"]
    err_d0_all = data["RecoTrack_ErrD0"]
    err_z0_all = data["RecoTrack_ErrZ0"]
    err_d0z0_all = data["RecoTrack_ErrD0Z0"]
    beam_z_all = data["BeamPosZ"]
    num_reco_vtx = data["NumRecoVtx"]

    n_total = len(z0_all)
    events = []

    for i in range(n_total):
        nrv = int(num_reco_vtx[i])
        if nrv < min_pileup:
            continue

        z0 = np.asarray(z0_all[i], dtype=np.float32)
        d0 = np.asarray(d0_all[i], dtype=np.float32)
        d0_err = np.asarray(err_d0_all[i], dtype=np.float32)
        z0_err = np.asarray(err_z0_all[i], dtype=np.float32)
        d0_z0_cov = np.asarray(err_d0z0_all[i], dtype=np.float32)

        beam_z_raw = beam_z_all[i]
        # BeamPosZ may be a 0-d numpy array, 1-d array, or scalar
        beam_z_arr = np.atleast_1d(np.asarray(beam_z_raw, dtype=np.float64))
        beam_z = float(beam_z_arr[0]) if len(beam_z_arr) > 0 else 0.0

        events.append({
            "z0": z0,
            "d0": d0,
            "d0_err": d0_err,
            "z0_err": z0_err,
            "d0_z0_cov": d0_z0_cov,
            "beam_z": beam_z,
            "pileup": nrv,
            "event_idx": i,
        })
        if len(events) >= max_events:
            break

    print(f"  Selected {len(events)} Run 3 events (min pileup={min_pileup})")
    return events


# ============================================================================
# MC Track Decoding -- CORRECT VERSION
# ============================================================================

def decode_mc_tracks(tracks_tensor):
    """Decode a single MC sub-event tensor (7, N_max) to physical params.

    CORRECT mapping (confirmed by eval_run3_v2.py):
        Channel 0: d0        -- RAW, no scaling
        Channel 1: z0
        Channel 2: d0_err    -- ALWAYS POSITIVE
        Channel 3: z0_err    -- ALWAYS POSITIVE
        Channel 4: d0_z0_cov
        Channel 5: z_start
        Channel 6: z_end

    Returns dict with 1-D arrays for non-padded tracks, or None.
    """
    z0_raw = tracks_tensor[1, :]
    valid = z0_raw > (MASK_VAL + 1)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return None

    return {
        "d0": tracks_tensor[0, valid],          # RAW d0 (NO *2!)
        "z0": tracks_tensor[1, valid],           # z0
        "d0_err": tracks_tensor[2, valid],       # d0_err (NOT theta!)
        "z0_err": tracks_tensor[3, valid],       # z0_err (NOT phi!)
        "d0_z0_cov": tracks_tensor[4, valid],    # covariance
        "n_tracks": n_valid,
    }


# ============================================================================
# Run 3 Tensor Building -- matches training format exactly
# ============================================================================

def build_run3_subevent_tensor(z0, d0, d0_err, z0_err, d0_z0_cov, sub_idx):
    """Build a single sub-event tensor from Run 3 track arrays.

    Returns (tensor, n_tracks) where tensor is shape (7, N_tracks) or
    (7, 1) if empty, padded with MASK_VAL.
    """
    z_start = SUBEVENT_STARTS[sub_idx]
    z_end = z_start + SUBEVENT_WIDTH

    mask = (z0 >= z_start) & (z0 < z_end)
    n_trk = int(np.sum(mask))

    if n_trk == 0:
        tensor = np.full((N_FEATURES, 1), MASK_VAL, dtype=np.float32)
        return tensor, 0

    # Sort by z0 (matches training convention)
    z0_sub = z0[mask]
    sort_idx = np.argsort(z0_sub)

    tensor = np.full((N_FEATURES, n_trk), MASK_VAL, dtype=np.float32)
    tensor[0, :] = d0[mask][sort_idx]           # d0 RAW
    tensor[1, :] = z0_sub[sort_idx]             # z0
    tensor[2, :] = d0_err[mask][sort_idx]       # d0_err
    tensor[3, :] = z0_err[mask][sort_idx]       # z0_err
    tensor[4, :] = d0_z0_cov[mask][sort_idx]    # d0_z0_cov
    tensor[5, :] = z_start                      # z_start (constant)
    tensor[6, :] = z_end                        # z_end   (constant)

    return tensor, n_trk


# ============================================================================
# Data Collection
# ============================================================================

def collect_features(mc_events, run3_events):
    """Extract physical track features from both datasets.

    Returns two dicts (mc_data, r3_data) each containing:
        - d0, z0, d0_err, z0_err, d0_z0_cov: 1-D arrays of all tracks
        - track_counts: 1-D array of per-subevent track counts
        - per_subevent: list of dicts (one per subevent) with sub-level data
    """
    print("  Collecting MC track features ...")
    mc_feats = {k: [] for k in ["d0", "z0", "d0_err", "z0_err", "d0_z0_cov"]}
    mc_track_counts = []
    mc_per_subevent = []

    for evt in tqdm(mc_events, desc="    MC", leave=False):
        tracks = evt["tracks"]  # (12, 7, 695)
        for si in range(N_SUBEVENTS):
            decoded = decode_mc_tracks(tracks[si])
            z_start = SUBEVENT_STARTS[si]
            z_center = z_start + SUBEVENT_WIDTH / 2.0

            if decoded is None:
                mc_track_counts.append(0)
                mc_per_subevent.append({
                    "z_center": z_center, "n_tracks": 0,
                    "mean_d0": 0.0, "mean_d0_err": 0.0, "std_d0": 0.0,
                })
                continue

            mc_track_counts.append(decoded["n_tracks"])
            for key in mc_feats:
                mc_feats[key].append(decoded[key])

            mc_per_subevent.append({
                "z_center": z_center,
                "n_tracks": decoded["n_tracks"],
                "mean_d0": float(np.mean(decoded["d0"])),
                "mean_d0_err": float(np.mean(decoded["d0_err"])),
                "std_d0": float(np.std(decoded["d0"])),
            })

    # Concatenate all tracks
    for key in mc_feats:
        mc_feats[key] = np.concatenate(mc_feats[key]) if mc_feats[key] else np.array([])
    mc_feats["track_counts"] = np.array(mc_track_counts)
    mc_feats["per_subevent"] = mc_per_subevent

    print("  Collecting Run 3 track features ...")
    r3_feats = {k: [] for k in ["d0", "z0", "d0_err", "z0_err", "d0_z0_cov"]}
    r3_track_counts = []
    r3_per_subevent = []

    for evt in tqdm(run3_events, desc="    Run3", leave=False):
        z0 = evt["z0"]
        d0 = evt["d0"]
        d0_err = evt["d0_err"]
        z0_err = evt["z0_err"]
        d0_z0_cov = evt["d0_z0_cov"]

        for si in range(N_SUBEVENTS):
            z_start = SUBEVENT_STARTS[si]
            z_end = z_start + SUBEVENT_WIDTH
            z_center = z_start + SUBEVENT_WIDTH / 2.0
            mask = (z0 >= z_start) & (z0 < z_end)
            n_trk = int(np.sum(mask))
            r3_track_counts.append(n_trk)

            if n_trk == 0:
                r3_per_subevent.append({
                    "z_center": z_center, "n_tracks": 0,
                    "mean_d0": 0.0, "mean_d0_err": 0.0, "std_d0": 0.0,
                })
                continue

            r3_feats["d0"].append(d0[mask])
            r3_feats["z0"].append(z0[mask])
            r3_feats["d0_err"].append(d0_err[mask])
            r3_feats["z0_err"].append(z0_err[mask])
            r3_feats["d0_z0_cov"].append(d0_z0_cov[mask])

            r3_per_subevent.append({
                "z_center": z_center,
                "n_tracks": n_trk,
                "mean_d0": float(np.mean(d0[mask])),
                "mean_d0_err": float(np.mean(d0_err[mask])),
                "std_d0": float(np.std(d0[mask])),
            })

    for key in ["d0", "z0", "d0_err", "z0_err", "d0_z0_cov"]:
        r3_feats[key] = np.concatenate(r3_feats[key]) if r3_feats[key] else np.array([])
    r3_feats["track_counts"] = np.array(r3_track_counts)
    r3_feats["per_subevent"] = r3_per_subevent

    print(f"    MC: {len(mc_feats['d0']):,} tracks, "
          f"Run3: {len(r3_feats['d0']):,} tracks")

    return mc_feats, r3_feats


def collect_tensor_values(mc_events, run3_events):
    """Collect raw tensor values per channel for both datasets.

    For MC: directly from the H5 tensor.
    For Run 3: build tensors in the CORRECT training format.

    Returns two dicts mapping channel_idx -> 1-D array.
    """
    print("  Collecting tensor values ...")
    mc_channels = {i: [] for i in range(N_FEATURES)}
    r3_channels = {i: [] for i in range(N_FEATURES)}

    for evt in tqdm(mc_events, desc="    MC tensors", leave=False):
        tracks = evt["tracks"]  # (12, 7, 695)
        for si in range(N_SUBEVENTS):
            trk = tracks[si]
            valid = trk[1, :] > (MASK_VAL + 1)
            if np.sum(valid) == 0:
                continue
            for ch in range(N_FEATURES):
                mc_channels[ch].append(trk[ch, valid])

    for evt in tqdm(run3_events, desc="    Run3 tensors", leave=False):
        z0 = evt["z0"]
        d0 = evt["d0"]
        d0_err = evt["d0_err"]
        z0_err = evt["z0_err"]
        d0_z0_cov = evt["d0_z0_cov"]

        for si in range(N_SUBEVENTS):
            tensor, n_trk = build_run3_subevent_tensor(
                z0, d0, d0_err, z0_err, d0_z0_cov, si
            )
            if n_trk == 0:
                continue
            valid = tensor[1, :] > (MASK_VAL + 1)
            for ch in range(N_FEATURES):
                r3_channels[ch].append(tensor[ch, valid])

    for ch in range(N_FEATURES):
        mc_channels[ch] = (
            np.concatenate(mc_channels[ch]) if mc_channels[ch] else np.array([])
        )
        r3_channels[ch] = (
            np.concatenate(r3_channels[ch]) if r3_channels[ch] else np.array([])
        )

    print(f"    MC tensor values: {len(mc_channels[0]):,}, "
          f"Run3 tensor values: {len(r3_channels[0]):,}")

    return mc_channels, r3_channels
