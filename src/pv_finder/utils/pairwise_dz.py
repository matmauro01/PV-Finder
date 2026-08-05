"""Binning for the pairwise-Δz (σ_vtx-vtx) histogram.

The peak finder reports positions as a value-weighted centroid over a few bins
around a region maximum.  For a peak that is sharp compared with the 0.04 mm
output grid that centroid sits very close to the maximum's own bin, so
reconstructed positions — and therefore |Δz| between any two of them — pile up
on multiples of the model bin width.  Measured on v6 held-out data: 40 % peak-
to-trough modulation of |Δz| mod 0.04 mm.

Histogramming that comb into bins of an *incommensurate* width beats.  The
historical 240 bins across ±6 mm gives 0.05 mm bins, and
``lcm(0.04, 0.05) = 0.20 mm``, so the plateau of the resolution plot acquires a
4-bin sawtooth that has nothing to do with vertices: 3.9 % peak-to-peak, four
times the Poisson error, and it is the visible "ripple" in that figure.

Making the plot bin width an integer multiple of the model bin width removes it
by construction — each plot bin then receives the same number of comb teeth.
This is a *presentation* fix: it does not move a single peak.  It does tighten
the σ_vtx-vtx fit, because the sigmoid is fitted to these counts.

See ``docs/research/resolution_plot_ripple.md``.
"""

from __future__ import annotations

from pv_finder.utils.constants import BIN_WIDTH_MM

# Half-width of the pairwise-Δz histogram, in mm.  The eval plots [-6, +6].
PAIRWISE_RANGE_MM: float = 6.0

# Relative tolerance for "is an integer multiple", generous enough to accept
# bin counts that divide the range exactly in floating point.
_TOL: float = 1e-6


def pairwise_bins(
    range_mm: float = PAIRWISE_RANGE_MM,
    bin_width_mm: float = BIN_WIDTH_MM,
    multiple: int = 1,
) -> int:
    """Bins across ``[-range_mm, +range_mm]`` commensurate with the model grid.

    ``multiple`` widens the plot bin to ``multiple * bin_width_mm``; the result
    is still commensurate, just coarser.

    >>> pairwise_bins()          # 12 mm / 0.04 mm
    300
    >>> pairwise_bins(multiple=2)
    150
    """
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")
    return int(round(2.0 * range_mm / (bin_width_mm * multiple)))


def is_commensurate(
    n_bins: int,
    range_mm: float = PAIRWISE_RANGE_MM,
    bin_width_mm: float = BIN_WIDTH_MM,
) -> bool:
    """True if the plot bin width is an integer multiple of the model bin width.

    An incommensurate binning beats against the position quantisation and puts a
    spurious comb in the plateau of the resolution plot.
    """
    if n_bins <= 0:
        return False
    ratio = (2.0 * range_mm / n_bins) / bin_width_mm
    return abs(ratio - round(ratio)) <= _TOL * max(1.0, ratio) and round(ratio) >= 1


# The eval default.  0.04 mm plot bins across +-6 mm.
DEFAULT_PAIRWISE_BINS: int = pairwise_bins()
