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

# --- Constants (must match feature_loading.py and CreatingTargetHistogram.py) ---
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
    """Override the module-level (A, B, C) resolution parameters."""
    global A_RES, B_RES, C_RES
    A_RES, B_RES, C_RES = a_res, b_res, c_res


_BIN_OFFSETS = np.arange(-5, 6)
_EDGES = np.array([-BIN_WIDTH / 2, BIN_WIDTH / 2])
_PROB_RANGE_TEMPLATE = BIN_WIDTH * _BIN_OFFSETS[np.newaxis, :] + _EDGES[:, np.newaxis]


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
        # 2 um floor: no-op for existing presets, keeps C<0 presets safe
        np.maximum(A_RES * np.power(ntrks_safe, -B_RES) + C_RES, 0.002),
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


_SUBEVENT_EDGES = np.array(
    [Z_MIN + k * SUBEVENT_WIDTH for k in range(N_SUBEVENTS + 1)], dtype=np.float64
)


def _build_event_subevent_tracks(
    z0: np.ndarray,
    d0: np.ndarray,
    d0_err: np.ndarray,
    z0_err: np.ndarray,
    d0_z0_cov: np.ndarray,
    max_tracks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N_SUBEVENTS, N_FEATURES, max_tracks) tracks for one event via
    one global stable argsort + searchsorted (replaces 12 boolean masks +
    12 argsorts). Truncation: take lowest-z0 max_tracks per subevent.
    """
    tensor = np.full((N_SUBEVENTS, N_FEATURES, max_tracks), MASK_VAL, dtype=np.float32)
    n_per_sub = np.zeros(N_SUBEVENTS, dtype=np.int64)
    if z0.size == 0:
        return tensor, n_per_sub

    in_range = (z0 >= Z_MIN) & (z0 < _SUBEVENT_EDGES[-1])
    if not np.any(in_range):
        return tensor, n_per_sub

    z0r = z0[in_range]
    d0r = d0[in_range]
    d0_err_r = d0_err[in_range]
    z0_err_r = z0_err[in_range]
    d0z0r = d0_z0_cov[in_range]

    order = np.argsort(z0r, kind="stable")  # stable so ties are reproducible
    z0_s = z0r[order]
    bnd = np.searchsorted(z0_s, _SUBEVENT_EDGES, side="left")

    # Apply order once per branch, then slice 12 windows from each.
    d0_s = d0r[order]
    d0_err_s = d0_err_r[order]
    z0_err_s = z0_err_r[order]
    d0z0_s = d0z0r[order]

    for si in range(N_SUBEVENTS):
        lo = int(bnd[si])
        hi = int(bnd[si + 1])
        n_trk = hi - lo
        n_per_sub[si] = n_trk
        if n_trk == 0:
            continue
        n_fill = min(n_trk, max_tracks)
        s = slice(lo, lo + n_fill)
        tensor[si, 0, :n_fill] = d0_s[s]
        tensor[si, 1, :n_fill] = z0_s[s]
        tensor[si, 2, :n_fill] = d0_err_s[s]
        tensor[si, 3, :n_fill] = z0_err_s[s]
        tensor[si, 4, :n_fill] = d0z0_s[s]
        z_start = SUBEVENT_STARTS[si]
        tensor[si, 5, :n_fill] = z_start
        tensor[si, 6, :n_fill] = z_start + SUBEVENT_WIDTH

    return tensor, n_per_sub


# --- First pass: find MAX_TRACKS per subevent and MAX_PV (skipped by default) ---


def scan_dimensions(
    tree: uproot.TTree, max_events: int, chunk_size: int = 5000
) -> tuple[int, int, int]:
    """Scan tree to find max tracks per subevent and max PVs per event."""
    n_total = min(tree.num_entries, max_events) if max_events > 0 else tree.num_entries
    max_trk_sub, max_pv = 0, 0
    branches = ["RecoTrack_z0", "TruthVertex_z"]
    for batch in tqdm(
        tree.iterate(branches, step_size=chunk_size, entry_stop=n_total, library="np"),
        total=math.ceil(n_total / chunk_size),
        desc="Pass 1: scanning dimensions",
    ):
        for i in range(len(batch["RecoTrack_z0"])):
            z0 = np.asarray(batch["RecoTrack_z0"][i], dtype=np.float32)
            for si in range(N_SUBEVENTS):
                z_lo = SUBEVENT_STARTS[si]
                n = int(np.sum((z0 >= z_lo) & (z0 < z_lo + SUBEVENT_WIDTH)))
                if n > max_trk_sub:
                    max_trk_sub = n
            n_pv = len(batch["TruthVertex_z"][i])
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
    max_pv_override: int = 0,
    chunk_size: int = 1000,
    compression: str = "lzf",
    skip_target_y: bool = True,
) -> None:
    """Run the full ROOT -> HDF5 conversion.

    Pass 1 (full-tree scan for max_tracks / max_pv) is skipped when both
    ``max_tracks_override`` and ``max_pv_override`` are positive.
    """
    if compression not in _COMPRESSION_KWARGS:
        raise ValueError(
            f"compression={compression!r} not in {sorted(_COMPRESSION_KWARGS)}"
        )
    comp_kw = _COMPRESSION_KWARGS[compression]

    tree = uproot.open(f"{input_path}:PVFinderData")
    n_events = (
        min(int(tree.num_entries), max_events)
        if max_events > 0
        else int(tree.num_entries)
    )

    if max_tracks_override > 0 and max_pv_override > 0:
        max_tracks, max_pv = max_tracks_override, max_pv_override
        print(
            f"[root_to_h5] Skipping Pass 1 (max_tracks={max_tracks}, max_pv={max_pv} provided)"
        )
    else:
        _, sc_mt, sc_mp = scan_dimensions(tree, max_events, chunk_size=chunk_size)
        max_tracks = max_tracks_override if max_tracks_override > 0 else sc_mt
        max_pv = max_pv_override if max_pv_override > 0 else sc_mp
    n_subevents = n_events * N_SUBEVENTS

    print(
        f"\n--- Dimensions --- events={n_events} subevents={n_subevents} "
        f"max_tracks={max_tracks} max_pv={max_pv}\n"
    )
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

                # -- Tracks: all 12 subevents in one vectorised pass --
                evt_tensor, evt_counts = _build_event_subevent_tracks(
                    z0, d0, d0_err, z0_err, d0_z0_cov, max_tracks
                )
                tracks_buf[i * N_SUBEVENTS : (i + 1) * N_SUBEVENTS] = evt_tensor
                trk_counts.extend(int(c) for c in evt_counts)

                # -- Truth histogram (full event); kept in scratch even when
                #    we don't persist `target_y`, because we still split it. --
                ty_full = _build_truth_histogram(pv_z, pv_ntrks)
                if ty_buf is not None:
                    ty_buf[i] = ty_full

                # Split truth histogram into N_SUBEVENTS contiguous slices
                ty_split_buf[i * N_SUBEVENTS : (i + 1) * N_SUBEVENTS] = ty_full.reshape(
                    N_CATS, N_SUBEVENTS, SUBEVENT_BINS
                ).transpose(1, 0, 2)
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

    trk_arr = np.array(trk_counts)
    truncated = int((trk_arr > max_tracks).sum())
    print(f"\n--- Summary --- Output: {output_path}")
    print(
        f"  Tracks/sub  mean={trk_arr.mean():.1f} median={np.median(trk_arr):.0f} "
        f"max={trk_arr.max()} p99={np.percentile(trk_arr, 99):.0f} "
        f"empty={(trk_arr == 0).sum()}/{trk_arr.size}"
    )
    if truncated:
        print(
            f"  WARNING: {truncated} subevents had > max_tracks={max_tracks} "
            "tracks; extras dropped. Bump --max-tracks-per-sub."
        )
    ty_part = (
        f"target_y({n_events},{N_CATS},{TOTAL_BINS}) "
        if not skip_target_y
        else "target_y[skipped] "
    )
    print(
        f"  Datasets: tracks({n_subevents},{N_FEATURES},{max_tracks}) "
        f"target_y_split({n_subevents},{N_CATS},{SUBEVENT_BINS}) "
        f"{ty_part}pv({n_events},{max_pv})  compression={compression}"
    )
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
        default=1024,
        help="Padded tracks-tensor width per subevent (0 = scan via Pass 1).",
    )
    parser.add_argument(
        "--max-pv",
        type=int,
        default=300,
        help="Padded pv-array width per event (0 = scan via Pass 1).",
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
        max_pv_override=args.max_pv,
        chunk_size=args.chunk_size,
        compression=args.compression,
        skip_target_y=not args.keep_target_y,
    )


if __name__ == "__main__":
    main()
