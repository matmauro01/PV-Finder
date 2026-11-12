"""Centralized physics constants for GNN Track-to-Vertex Association.

These constants are specific to the GNN TTVA pipeline. Vertex-finding
constants (Z_MIN, Z_MAX, TOTAL_NUM_BINS, etc.) live in feature_loading.py.
"""

from __future__ import annotations

# --- Track feature normalization ---
PT_SCALE: float = 50000.0  # pT normalization used during GNN training (MeV)

# --- PV resolution fit parameters (from ResolutionFit_ATLAS.ipynb) ---
# sigma_pv(N) = A * N^(-B) + C, where N = number of tracks
PV_RES_A: float = 0.23817443
PV_RES_B: float = 0.49491396
PV_RES_C: float = -0.000787436
PV_MIN_TRACKS: int = 2  # minimum tracks for a valid PV

# --- GNN evaluation thresholds ---
GNN_SCORE_THRESHOLD: float = 0.5  # edge score above which a track is "associated"
PURITY_THRESHOLD: float = 0.7  # fraction for Clean vs Merged classification

# --- Positional truth-matching windows ---
# Defined once in pv_finder.utils.working_points (single source of truth,
# 11 Nov 2026): 0.17 mm for every vertex study and both regimes (3 x the 68%
# single-vertex residual at PU200, PV-Finder 55.05 um / AMVF 56.45 um,
# rounded from 0.1694), 0.1 mm for hard-scatter studies (ITk paper,
# arXiv:2412.15090, Sec. 5.5). Re-exported here for existing imports.
from pv_finder.utils.working_points import (  # noqa: E402
    HS_MATCH_WINDOW_MM as HS_MATCH_WINDOW_MM,
)
from pv_finder.utils.working_points import (  # noqa: E402
    MATCH_WINDOW_MM as MATCH_WINDOW_MM,
)

# --- Binning (ATLAS-specific, duplicated from feature_loading for GNN use) ---
Z_MIN: float = -240.0  # mm
Z_MAX: float = 240.0  # mm
TOTAL_NUM_BINS: int = 12000
BINS_PER_MM: int = 25  # = 12000 / 480
BIN_WIDTH_MM: float = 0.04  # = 1 / 25

# --- Subevent structure ---
N_SUBEVENTS: int = 12
BINS_PER_SUBEVENT: int = 1000
