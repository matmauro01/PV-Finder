"""Chain-like augmentation of TTVA truth-training graphs.

Truth graphs place PV nodes at exact truth z with recipe heights and
preset sigmas; deployment (chain) graphs place them at PVF peaks — jittered
z, peak-width sigmas, noisy heights, ~9% junk peaks, and ~19% of truth PVs
missing entirely. Models trained purely on truth graphs transfer
imperfectly to peaks (the v2 "fake floor"). This module makes a truth
event *statistically* chain-like while keeping exact truth labels:

- drop real PV nodes with the measured miss probability vs nTracks
  (their tracks' edges all become label 0 — the model learns to abstain);
- jitter surviving PV z by the measured peak-truth dz residuals;
- replace preset sigmas with measured matched-peak sigmas;
- scale recipe heights by the measured peak/recipe height ratio;
- append junk PV nodes (Poisson, measured rate) at random track z0
  positions with measured junk heights/sigmas and no true edges.

All distributions are empirical quantile grids measured by
gnn.diagnostics.chain_gap_decomposition (augmentation_params.json +
gap_decomposition.json in the same directory).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EmpiricalSampler:
    """Inverse-CDF sampling from a 101-point quantile grid."""

    def __init__(self, quantiles: list[float], grid: list[float]) -> None:
        self._q = np.asarray(quantiles, dtype=np.float64)
        self._grid = np.asarray(grid, dtype=np.float64)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw n samples."""
        return np.interp(rng.random(n), self._grid, self._q)


class AugmentationParams:
    """Measured chain statistics driving the augmentation."""

    def __init__(self, params_dir: str | Path) -> None:
        params_dir = Path(params_dir)
        with open(params_dir / "augmentation_params.json") as f:
            aug = json.load(f)
        with open(params_dir / "gap_decomposition.json") as f:
            gap = json.load(f)

        grid = aug["quantile_grid"]
        self.jitter = EmpiricalSampler(aug["jitter_dz_quantiles"], grid)
        self.height_ratio = EmpiricalSampler(aug["height_ratio_quantiles"], grid)
        self.junk_height = EmpiricalSampler(aug["junk_height_quantiles"], grid)
        self.matched_sigma = EmpiricalSampler(aug["matched_sigma_quantiles"], grid)
        self.junk_sigma = EmpiricalSampler(aug["junk_sigma_quantiles"], grid)
        self.junk_per_peak = float(aug["junk_per_peak"])

        # Per-nTracks-bin miss probability: missed / (missed + matched)
        missed = np.asarray(gap["missed_ntrk_hist"], dtype=np.float64)
        matched = np.asarray(gap["matched_ntrk_hist"], dtype=np.float64)
        total = missed + matched
        self.miss_prob = np.divide(
            missed, total, out=np.zeros_like(missed), where=total > 0
        )
        self.ntrk_bin_edges = np.asarray(gap["ntrk_bin_edges"], dtype=np.float64)

    def miss_probability(self, ntracks: np.ndarray) -> np.ndarray:
        """Finder-miss probability for vertices of these track multiplicities."""
        idx = np.clip(
            np.searchsorted(self.ntrk_bin_edges, ntracks, side="right") - 1,
            0,
            len(self.miss_prob) - 1,
        )
        prob = self.miss_prob[idx]
        # Vertices below the classifier's nTrk >= 2 cut carry no clean/truth
        # weight; leave them in place (they are already near-invisible peaks).
        return np.where(ntracks < 2, 0.0, prob)


def augment_event(
    event_data: dict[str, np.ndarray],
    recipe_heights: np.ndarray,
    params: AugmentationParams,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    """Make one truth event chain-like. Returns (event_data, res, heights, n_junk).

    The returned event_data has PV arrays modified in a copy (tracks are
    untouched); pv_res/heights are the arrays to pass to
    create_training_graph as pv_res_all / pv_heights_override.
    """
    pv_z = event_data["pv_loc_z"]
    pv_ntracks = event_data["pv_ntracks"]
    pv_type = event_data["pv_type"]
    pv_assoc = event_data["pv_assoctracks"]
    n_pv = len(pv_z)

    # --- 1. Drop finder-missed vertices ---------------------------------
    keep = rng.random(n_pv) >= params.miss_probability(pv_ntracks)
    counts = pv_ntracks.astype(int)
    keep_edge_mask = np.repeat(keep, counts)

    pv_z = pv_z[keep]
    pv_ntracks = pv_ntracks[keep]
    pv_type = pv_type[keep]
    pv_assoc = pv_assoc[keep_edge_mask]
    recipe_heights = recipe_heights[keep]
    n_kept = len(pv_z)

    # --- 2. Jitter positions, resample sigmas, rescale heights ----------
    pv_z = pv_z + params.jitter.sample(rng, n_kept)
    pv_res = params.matched_sigma.sample(rng, n_kept)
    heights = recipe_heights * params.height_ratio.sample(rng, n_kept)

    # --- 3. Junk PV nodes at random track positions ---------------------
    n_junk = int(rng.poisson(params.junk_per_peak * max(n_kept, 1)))
    z0 = event_data["z_0"]
    if n_junk > 0 and len(z0) > 0:
        junk_z = z0[rng.integers(0, len(z0), n_junk)] + rng.normal(0, 1.0, n_junk)
        pv_z = np.concatenate([pv_z, junk_z])
        pv_ntracks = np.concatenate([pv_ntracks, np.zeros(n_junk)])
        pv_type = np.concatenate([pv_type, np.full(n_junk, -1.0)])
        pv_res = np.concatenate([pv_res, params.junk_sigma.sample(rng, n_junk)])
        heights = np.concatenate([heights, params.junk_height.sample(rng, n_junk)])
    else:
        n_junk = 0

    out = dict(event_data)
    out["pv_loc_z"] = pv_z
    out["pv_ntracks"] = pv_ntracks
    out["pv_type"] = pv_type
    out["pv_assoctracks"] = pv_assoc
    return out, pv_res, heights.astype(np.float64), n_junk
