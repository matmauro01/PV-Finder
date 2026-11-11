"""Read the training targets out through the production peak finder.

The target histogram is what a perfect network would output.  Running it
through the same peak finding and the same metrics as the network output
separates two limits: what the target definition and the 0.04 mm grid allow
(target readout) and what the network achieves (PV-Finder readout).

``build_truth_histogram`` is the converter's target builder
(``pv_finder.data.root_to_h5._build_truth_histogram``) with the bin width as a
parameter, so a finer grid can be tested; at 0.04 mm it is identical to the
converter (``tests/test_target_readout.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from pv_finder.evaluation.vertex_finding.efficiency_res_optimized_atlas import (
    pv_locations_updated_res,
)

Z_MIN_MM: float = -240.0
Z_MAX_MM: float = 240.0
NTRK_THRESHOLD: int = 2
MIN_SIGMA_MM: float = 0.002
SCALE_WIDTH_MM: float = 0.15
N_OFFSETS: int = 5  # the converter fills +-5 bins around each vertex


def build_truth_histogram(
    pv_z: np.ndarray,
    pv_ntrk: np.ndarray,
    abc: tuple[float, float, float],
    bin_mm: float = 0.04,
) -> np.ndarray:
    """(2, n_bins) target for one event, the converter's rule on any grid.

    Channel 0 holds vertices with nTracks >= 2 (sigma = A n^-B + C, floor
    2 um), channel 1 the others (sigma = one bin).  Bin ``i`` is centred on
    ``Z_MIN + i * bin_mm``, the convention the peak finder reads back.  Each
    peak is scaled by ``max(1, 0.15 mm / sigma)`` as in the converter.  The
    converter fills +-5 bins; on a finer grid the same physical half-range
    (0.2 mm) is used so that wide peaks are not truncated.
    """
    a, b, c = abc
    n_bins = int(round((Z_MAX_MM - Z_MIN_MM) / bin_mm))
    hist = np.zeros((2, n_bins), dtype=np.float64)
    z = np.asarray(pv_z, dtype=np.float64)
    if len(z) == 0:
        return hist
    nt = np.asarray(pv_ntrk, dtype=np.int64)
    sigma = np.where(
        nt < NTRK_THRESHOLD,
        bin_mm,
        np.maximum(a * np.power(np.maximum(nt, 1).astype(np.float64), -b) + c,
                   MIN_SIGMA_MM),
    )  # fmt: skip
    half = int(round(N_OFFSETS * 0.04 / bin_mm))
    offs = np.arange(-half, half + 1)
    nbin = np.floor((z - Z_MIN_MM) / bin_mm).astype(np.int64)
    centre = nbin * bin_mm + Z_MIN_MM
    lo = centre[None, :] + offs[:, None] * bin_mm - bin_mm / 2
    hi = lo + bin_mm
    pop = ndtr((hi - z[None, :]) / sigma[None, :]) - ndtr(
        (lo - z[None, :]) / sigma[None, :]
    )
    pop *= np.maximum(1.0, SCALE_WIDTH_MM / sigma)[None, :]
    idx = offs[:, None] + nbin[None, :]
    cat = np.broadcast_to(np.where(nt >= NTRK_THRESHOLD, 0, 1)[None, :], pop.shape)
    ok = (idx >= 0) & (idx < n_bins)
    np.add.at(hist.reshape(-1), (cat[ok] * n_bins + idx[ok]).ravel(), pop[ok].ravel())
    return hist


@dataclass(frozen=True)
class PeakConfig:
    """Peak-finder settings (bins and per-bin heights at 0.04 mm)."""

    threshold: float
    integral: float
    min_width: int
    min_height: float
    centroid_halfwidth: int


RUN2_CONFIG = PeakConfig(0.01, 0.5, 3, 0.0, 0)  # Run 2 reproduction eval
NOTE_CONFIG = PeakConfig(0.01, 0.30, 3, 0.03, 3)  # HL-LHC production counting point
# Target-only variant: HS targets are 1-2-bin spikes (sigma ~ 3-10 um) and fail
# the 3-bin minimum width, so the production setting drops them.
NOTE_W1_CONFIG = PeakConfig(0.01, 0.30, 1, 0.03, 3)


def find_peaks(hist0: np.ndarray, cfg: PeakConfig, bin_mm: float = 0.04) -> np.ndarray:
    """Peak positions (mm) from a channel-0 histogram with the production finder.

    On a grid finer than 0.04 mm the settings keep their physical meaning:
    widths and the centroid window in bins scale by ``0.04 / bin_mm`` and the
    per-bin threshold and height floor by ``bin_mm / 0.04`` (the integral is a
    sum of bin contents and does not change).
    """
    k = 0.04 / bin_mm
    pos, *_ = pv_locations_updated_res(
        np.asarray(hist0, dtype=np.float64),
        cfg.threshold / k,
        cfg.integral,
        int(round(cfg.min_width * k)),
        cfg.min_height / k,
        int(round(cfg.centroid_halfwidth * k)),
    )
    # The finder converts bins to mm with a fixed 0.04 mm bin.
    bins = (np.asarray(pos, dtype=np.float64) - Z_MIN_MM) / 0.04
    return bins * bin_mm + Z_MIN_MM
