"""
ROOT-to-HDF5 converter for ATLAS HLLHC PU200 data.

Converts ATLAS HLLHC ROOT ntuples (tree: PVFinderData) into the flat HDF5
format consumed by the PV-Finder training pipeline (collectdata_poca_KDE.py,
h5_dataset.py).

Usage:
    python -u src/pv_finder/data/root_to_h5.py \
        --input data/run4/.../PU200.root \
        --output data/run4/hllhc_pu200.h5 \
        --max-events 0
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np
import uproot
from scipy.special import ndtr  # vectorised standard-normal CDF in C
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants — must match feature_loading.py and CreatingTargetHistogram.py
# ---------------------------------------------------------------------------
Z_MIN = -240.0  # mm
Z_MAX = 240.0  # mm
TOTAL_BINS = 12000
BINS_PER_MM = TOTAL_BINS / (Z_MAX - Z_MIN)  # 25
BIN_WIDTH = 1.0 / BINS_PER_MM  # 0.04 mm
N_SUBEVENTS = 12
SUBEVENT_WIDTH = 40.0  # mm
SUBEVENT_BINS = 1000  # bins per subevent
SUBEVENT_STARTS = [Z_MIN + i * SUBEVENT_WIDTH for i in range(N_SUBEVENTS)]
N_FEATURES = 7  # d0, z0, d0_err, z0_err, d0_z0_cov, z_start, z_end
MASK_VAL = -999999.0
N_CATS = 2  # channel 0: nTracks>=2, channel 1: nTracks<2
NTRK_THRESHOLD = 2

# Resolution fit parameters live in resolution_presets.py — see that module
# to add a new preset. CLI exposes --resolution-preset (and --a/b/c-res for
# one-off overrides). The chosen (A, B, C) are also written to h5.attrs.
from pv_finder.data.resolution_presets import (  # noqa: E402
    DEFAULT_RESOLUTION_PRESET,
    RESOLUTION_PRESET_SOURCES,
    RESOLUTION_PRESETS,
)

A_RES, B_RES, C_RES = RESOLUTION_PRESETS[DEFAULT_RESOLUTION_PRESET]


def set_resolution(a_res: float, b_res: float, c_res: float) -> None:
    """Override the module-level (A, B, C) used by `_compute_sigma`."""
    global A_RES, B_RES, C_RES
    A_RES, B_RES, C_RES = a_res, b_res, c_res


# Pre-compute the +-5 bin edge offsets for Gaussian CDF target generation.
# Shape: (2, 11) — row 0 = left edges, row 1 = right edges of 11 bins.
_BIN_OFFSETS = np.arange(-5, 6)  # [-5, -4, ..., 5]
_EDGES = np.array([-BIN_WIDTH / 2, BIN_WIDTH / 2])
# ProbRange[edge_idx, bin_idx] gives the z-offset from bin center
_PROB_RANGE_TEMPLATE = BIN_WIDTH * _BIN_OFFSETS[np.newaxis, :] + _EDGES[:, np.newaxis]


# --- Helpers ---


def _compute_sigma(ntrks: int) -> float:
    """Vertex resolution from nTracks (matches CreatingTargetHistogram.py)."""
    if ntrks < NTRK_THRESHOLD:
        return BIN_WIDTH
    return A_RES * (ntrks ** (-B_RES)) + C_RES


def _build_truth_histogram(pv_z: np.ndarray, pv_ntrks: np.ndarray) -> np.ndarray:
    """Build a (2, 12000) truth histogram for one event.

    Vectorised port of CreatingTargetHistogram.py: one ``scipy.special.ndtr``
    call + one ``np.add.at`` scatter across all PVs. Numerically identical
    to the original per-PV loop within 1e-12.
    """
    n_pv = len(pv_z)
    hist = np.zeros((N_CATS, TOTAL_BINS), dtype=np.float64)
    if n_pv == 0:
        return hist

    pv_z_arr = np.asarray(pv_z, dtype=np.float64)
    pv_ntrks_arr = np.asarray(pv_ntrks, dtype=np.int64)

    ntrks_safe = np.maximum(pv_ntrks_arr, 1).astype(np.float64)
    sigma = np.where(
        pv_ntrks_arr < NTRK_THRESHOLD,
        BIN_WIDTH,
        A_RES * np.power(ntrks_safe, -B_RES) + C_RES,
    )
    nbin = np.floor((pv_z_arr - Z_MIN) * BINS_PER_MM).astype(np.int64)
    bin_center_z = nbin / BINS_PER_MM + Z_MIN

    z_prob = bin_center_z[None, None, :] + _PROB_RANGE_TEMPLATE[:, :, None]
    z_norm = (z_prob - pv_z_arr[None, None, :]) / sigma[None, None, :]
    populate = ndtr(z_norm[1]) - ndtr(z_norm[0])  # (11, n_pv)

    scale = np.maximum(1.0, 0.15 / sigma)
    populate = populate * scale[None, :]

    bin_indices = _BIN_OFFSETS[:, None] + nbin[None, :]
    cat = np.where(pv_ntrks_arr >= NTRK_THRESHOLD, 0, 1)
    cat_b = np.broadcast_to(cat[None, :], populate.shape)
    valid = (bin_indices >= 0) & (bin_indices < TOTAL_BINS)

    flat_idx = (cat_b[valid] * TOTAL_BINS + bin_indices[valid]).ravel()
    flat_vals = populate[valid].ravel()
    np.add.at(hist.reshape(-1), flat_idx, flat_vals)
    return hist


def _build_subevent_tracks(
    z0: np.ndarray,
    d0: np.ndarray,
    d0_err: np.ndarray,
    z0_err: np.ndarray,
    d0_z0_cov: np.ndarray,
    sub_idx: int,
    max_tracks: int,
) -> tuple[np.ndarray, int]:
    """Build a (7, max_tracks) padded track tensor for one subevent.

    Returns (tensor, n_tracks_in_subevent).
    """
    z_start = SUBEVENT_STARTS[sub_idx]
    z_end = z_start + SUBEVENT_WIDTH

    mask = (z0 >= z_start) & (z0 < z_end)
    n_trk = int(np.sum(mask))

    tensor = np.full((N_FEATURES, max_tracks), MASK_VAL, dtype=np.float32)
    if n_trk == 0:
        return tensor, 0

    # Sort by z0 within the subevent (training convention)
    idx = np.argsort(z0[mask])
    n_fill = min(n_trk, max_tracks)
    idx = idx[:n_fill]

    tensor[0, :n_fill] = d0[mask][idx]
    tensor[1, :n_fill] = z0[mask][idx]
    tensor[2, :n_fill] = d0_err[mask][idx]
    tensor[3, :n_fill] = z0_err[mask][idx]
    tensor[4, :n_fill] = d0_z0_cov[mask][idx]
    tensor[5, :n_fill] = z_start
    tensor[6, :n_fill] = z_end

    return tensor, n_trk


# --- First pass: determine MAX_TRACKS per subevent and MAX_PV ---


def scan_dimensions(
    tree: uproot.TTree,
    max_events: int,
    chunk_size: int = 5000,
) -> tuple[int, int, int]:
    """Scan ROOT tree to find max tracks per subevent and max PVs per event."""
    n_total = min(tree.num_entries, max_events) if max_events > 0 else tree.num_entries
    max_trk_sub = 0
    max_pv = 0

    branches = ["RecoTrack_z0", "TruthVertex_z"]
    desc = "Pass 1: scanning dimensions"

    for batch in tqdm(
        tree.iterate(branches, step_size=chunk_size, entry_stop=n_total, library="np"),
        total=math.ceil(n_total / chunk_size),
        desc=desc,
    ):
        z0_jagged = batch["RecoTrack_z0"]
        pv_z_jagged = batch["TruthVertex_z"]

        for i in range(len(z0_jagged)):
            z0 = np.asarray(z0_jagged[i], dtype=np.float32)
            for si in range(N_SUBEVENTS):
                z_lo = SUBEVENT_STARTS[si]
                z_hi = z_lo + SUBEVENT_WIDTH
                n = int(np.sum((z0 >= z_lo) & (z0 < z_hi)))
                if n > max_trk_sub:
                    max_trk_sub = n

            n_pv = len(pv_z_jagged[i])
            if n_pv > max_pv:
                max_pv = n_pv

    return n_total, max_trk_sub, max_pv


# --- Second pass: build arrays and write HDF5 ---

# Branches we actually need from the ROOT file
_BRANCHES = [
    "RecoTrack_z0",
    "RecoTrack_d0",
    "RecoTrack_ErrD0",
    "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0",
    "TruthVertex_z",
    "TruthVertex_nTracks",
]


_COMPRESSION_KWARGS: dict[str, dict] = {
    "none": {},
    "lzf": {"compression": "lzf"},
    "gzip": {"compression": "gzip", "compression_opts": 4, "shuffle": True},
}


def convert(
    input_path: str,
    output_path: str,
    max_events: int = 0,
    max_tracks_override: int = 0,
    chunk_size: int = 1000,
    compression: str = "lzf",
    skip_target_y: bool = True,
) -> None:
    """Run the full ROOT -> HDF5 conversion.

    ``compression``: HDF5 chunk filter ('lzf' default, 'gzip', or 'none').
    ``skip_target_y``: omit the full-event target_y (HL-LHC trainer reads
    only target_y_split — saves the biggest single chunk of disk).
    """
    if compression not in _COMPRESSION_KWARGS:
        raise ValueError(
            f"compression={compression!r} not in {sorted(_COMPRESSION_KWARGS)}"
        )
    comp_kw = _COMPRESSION_KWARGS[compression]

    tree = uproot.open(f"{input_path}:PVFinderData")

    # --- Pass 1: dimensions ---
    n_events, max_trk_sub_data, max_pv_data = scan_dimensions(
        tree, max_events, chunk_size=chunk_size
    )
    max_tracks = max_tracks_override if max_tracks_override > 0 else max_trk_sub_data
    max_pv = max_pv_data
    n_subevents = n_events * N_SUBEVENTS

    print("\n--- Dimensions ---")
    print(f"  Events:              {n_events}")
    print(f"  Subevents:           {n_subevents}")
    print(f"  Max tracks/subevent: {max_trk_sub_data} (using {max_tracks})")
    print(f"  Max PVs/event:       {max_pv}\n")

    # --- Create output HDF5 with pre-allocated datasets ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        ds_tracks = h5.create_dataset(
            "tracks",
            shape=(n_subevents, N_FEATURES, max_tracks),
            dtype=np.float32,
            chunks=(min(120, n_subevents), N_FEATURES, max_tracks),
            fillvalue=MASK_VAL,
            **comp_kw,
        )
        ds_ty_split = h5.create_dataset(
            "target_y_split",
            shape=(n_subevents, N_CATS, SUBEVENT_BINS),
            dtype=np.float16,
            chunks=(min(120, n_subevents), N_CATS, SUBEVENT_BINS),
            **comp_kw,
        )
        ds_ty = None
        if not skip_target_y:
            ds_ty = h5.create_dataset(
                "target_y",
                shape=(n_events, N_CATS, TOTAL_BINS),
                dtype=np.float16,
                chunks=(min(10, n_events), N_CATS, TOTAL_BINS),
                **comp_kw,
            )
        ds_pv = h5.create_dataset(
            "pv",
            shape=(n_events, max_pv),
            dtype=np.float32,
            chunks=(min(1000, n_events), max_pv),
            fillvalue=MASK_VAL,
            **comp_kw,
        )

        # Store metadata
        h5.attrs["z_min"] = Z_MIN
        h5.attrs["z_max"] = Z_MAX
        h5.attrs["n_subevents_per_event"] = N_SUBEVENTS
        h5.attrs["subevent_width_mm"] = SUBEVENT_WIDTH
        h5.attrs["bin_width_mm"] = BIN_WIDTH
        h5.attrs["max_tracks_per_subevent"] = max_tracks
        h5.attrs["max_pv"] = max_pv
        h5.attrs["n_events"] = n_events
        h5.attrs["source_file"] = str(input_path)
        h5.attrs["resolution_a_mm"] = float(A_RES)
        h5.attrs["resolution_b"] = float(B_RES)
        h5.attrs["resolution_c_mm"] = float(C_RES)
        h5.attrs["resolution_formula"] = "sigma_z(n) = A * n^(-B) + C  [mm]"
        h5.attrs["compression"] = compression
        h5.attrs["has_target_y"] = bool(not skip_target_y)

        # --- Pass 2: fill data ---
        evt_offset = 0
        sub_offset = 0
        trk_counts: list[int] = []

        desc = "Pass 2: converting"
        for batch in tqdm(
            tree.iterate(
                _BRANCHES, step_size=chunk_size, entry_stop=n_events, library="np"
            ),
            total=math.ceil(n_events / chunk_size),
            desc=desc,
        ):
            bsz = len(batch["RecoTrack_z0"])

            # Pre-allocate chunk buffers
            tracks_buf = np.full(
                (bsz * N_SUBEVENTS, N_FEATURES, max_tracks),
                MASK_VAL,
                dtype=np.float32,
            )
            ty_split_buf = np.zeros(
                (bsz * N_SUBEVENTS, N_CATS, SUBEVENT_BINS), dtype=np.float64
            )
            ty_buf = (
                np.zeros((bsz, N_CATS, TOTAL_BINS), dtype=np.float64)
                if not skip_target_y
                else None
            )
            pv_buf = np.full((bsz, max_pv), MASK_VAL, dtype=np.float32)

            for i in range(bsz):
                z0 = np.asarray(batch["RecoTrack_z0"][i], dtype=np.float32)
                d0 = np.asarray(batch["RecoTrack_d0"][i], dtype=np.float32)
                d0_err = np.asarray(batch["RecoTrack_ErrD0"][i], dtype=np.float32)
                z0_err = np.asarray(batch["RecoTrack_ErrZ0"][i], dtype=np.float32)
                d0_z0_cov = np.asarray(batch["RecoTrack_ErrD0Z0"][i], dtype=np.float32)
                pv_z = np.asarray(batch["TruthVertex_z"][i], dtype=np.float32)
                pv_ntrks = np.asarray(batch["TruthVertex_nTracks"][i], dtype=np.float32)

                # -- Tracks: build subevents --
                for si in range(N_SUBEVENTS):
                    tensor, n_trk = _build_subevent_tracks(
                        z0, d0, d0_err, z0_err, d0_z0_cov, si, max_tracks
                    )
                    tracks_buf[i * N_SUBEVENTS + si] = tensor
                    trk_counts.append(n_trk)

                # -- Truth histogram (full event); kept in scratch even when
                #    we don't persist `target_y`, because we still split it. --
                ty_full = _build_truth_histogram(pv_z, pv_ntrks)
                if ty_buf is not None:
                    ty_buf[i] = ty_full

                # -- Split truth histogram into subevents --
                for si in range(N_SUBEVENTS):
                    bin_lo = si * SUBEVENT_BINS
                    bin_hi = bin_lo + SUBEVENT_BINS
                    ty_split_buf[i * N_SUBEVENTS + si] = ty_full[:, bin_lo:bin_hi]

                # -- PV positions --
                n_pv = min(len(pv_z), max_pv)
                pv_buf[i, :n_pv] = pv_z[:n_pv]

            # Write chunk to HDF5
            ds_tracks[sub_offset : sub_offset + bsz * N_SUBEVENTS] = tracks_buf
            ds_ty_split[sub_offset : sub_offset + bsz * N_SUBEVENTS] = (
                ty_split_buf.astype(np.float16)
            )
            if ds_ty is not None and ty_buf is not None:
                ds_ty[evt_offset : evt_offset + bsz] = ty_buf.astype(np.float16)
            ds_pv[evt_offset : evt_offset + bsz] = pv_buf

            evt_offset += bsz
            sub_offset += bsz * N_SUBEVENTS

    # --- Summary statistics ---
    trk_arr = np.array(trk_counts)
    print("\n--- Summary ---")
    print(f"  Output:   {output_path}")
    print(f"  Events:   {n_events}")
    print(f"  Subevents:{n_subevents}")
    print(
        f"  Tracks/sub  mean={trk_arr.mean():.1f}  "
        f"median={np.median(trk_arr):.0f}  "
        f"max={trk_arr.max()}  "
        f"p99={np.percentile(trk_arr, 99):.0f}"
    )
    print(
        f"  Empty subevents: {(trk_arr == 0).sum()} "
        f"({100 * (trk_arr == 0).mean():.1f}%)"
    )
    target_y_part = (
        f"target_y({n_events},{N_CATS},{TOTAL_BINS}) "
        if not skip_target_y
        else "target_y[skipped] "
    )
    print(
        f"  Datasets: tracks({n_subevents},{N_FEATURES},{max_tracks}) "
        f"target_y_split({n_subevents},{N_CATS},{SUBEVENT_BINS}) "
        f"{target_y_part}"
        f"pv({n_events},{max_pv})"
    )
    print(f"  Compression: {compression}")
    on_disk_bytes = Path(output_path).stat().st_size
    print(f"  On-disk size: {on_disk_bytes / 1e9:.2f} GB")


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ATLAS HLLHC ROOT ntuple to flat HDF5 for PV-Finder."
    )
    parser.add_argument("--input", required=True, help="Input ROOT file path.")
    parser.add_argument("--output", required=True, help="Output HDF5 file path.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Max events to convert (0 = all).",
    )
    parser.add_argument(
        "--max-tracks-per-sub",
        type=int,
        default=0,
        help="Override max tracks per subevent (0 = auto-detect).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Events per processing chunk (default: 1000).",
    )
    parser.add_argument(
        "--resolution-preset",
        choices=sorted(RESOLUTION_PRESETS),
        default=DEFAULT_RESOLUTION_PRESET,
        help="Named (A, B, C) preset; see resolution_presets.py.",
    )
    parser.add_argument("--a-res", type=float, default=None,
                        help="Override A (mm).")  # fmt: skip
    parser.add_argument("--b-res", type=float, default=None,
                        help="Override B.")  # fmt: skip
    parser.add_argument("--c-res", type=float, default=None,
                        help="Override C (mm).")  # fmt: skip
    parser.add_argument(
        "--compression",
        choices=sorted(_COMPRESSION_KWARGS),
        default="lzf",
        help="HDF5 chunk filter. 'lzf' (default, fast lossless), 'gzip', 'none'.",
    )
    parser.add_argument(
        "--keep-target-y",
        action="store_true",
        help="Also write the full-event target_y; off by default (trainer "
        "uses target_y_split only).",
    )
    args = parser.parse_args()

    a, b, c = RESOLUTION_PRESETS[args.resolution_preset]
    if args.a_res is not None:
        a = args.a_res
    if args.b_res is not None:
        b = args.b_res
    if args.c_res is not None:
        c = args.c_res
    set_resolution(a, b, c)
    print(
        f"[root_to_h5] resolution: preset={args.resolution_preset!r} -> "
        f"sigma_z(n) = {a:.6f} * n^(-{b:.6f}) + {c:.6f}  [mm]"
    )
    print(f"[root_to_h5]   source: {RESOLUTION_PRESET_SOURCES[args.resolution_preset]}")

    convert(
        args.input,
        args.output,
        max_events=args.max_events,
        max_tracks_override=args.max_tracks_per_sub,
        chunk_size=args.chunk_size,
        compression=args.compression,
        skip_target_y=not args.keep_target_y,
    )


if __name__ == "__main__":
    main()
