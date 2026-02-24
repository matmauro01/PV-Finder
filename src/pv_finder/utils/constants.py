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

# --- Binning (ATLAS-specific, duplicated from feature_loading for GNN use) ---
Z_MIN: float = -240.0  # mm
Z_MAX: float = 240.0  # mm
TOTAL_NUM_BINS: int = 12000
BINS_PER_MM: int = 25  # = 12000 / 480
BIN_WIDTH_MM: float = 0.04  # = 1 / 25

# --- Subevent structure ---
N_SUBEVENTS: int = 12
BINS_PER_SUBEVENT: int = 1000
