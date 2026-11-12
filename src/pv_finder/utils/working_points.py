"""Final working points: the single source of truth (decided 11 November 2026).

Every evaluation, diagnostic and plotting script takes its matching window,
peak-finding cuts, density model and GNN working point from here, through
``get_regime(name)``. A script may still override a value with an explicit
flag, but it must record the override (see ``Regime.to_dict`` and
``record_overrides``).

Two regimes only (docs/additional_studies/meeting_2026-11-11_plan.md):

* ``hllhc``: HL-LHC ITk simulation, 0 <= mu <= 200.
* ``lowpu``: Run 2 simulation, Run 2 and Run 3 collision data.

Both use one matching window, 0.17 mm (3 x the 68% single-vertex position
residual at PU200, PV-Finder 55.05 um / AMVF 56.45 um, rounded), and the
local-centroid position estimator. Hard-scatter studies alone use 0.1 mm,
following the ITk performance paper (arXiv:2412.15090, Sec. 5.5).

sigma_vtx-vtx is fitted on the same peaks that are counted: there is no
separate resolution threshold.

Items still to be frozen are marked ``frozen=False``. ``Regime.require_frozen``
raises for them, so a final result cannot be produced from a placeholder by
accident.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --- Matching windows (mm) ------------------------------------------------
MATCH_WINDOW_MM: float = 0.17
HS_MATCH_WINDOW_MM: float = 0.1
# Radius of the optimal one-to-one assignment used for single-vertex position
# residuals (the 68% width that sets MATCH_WINDOW_MM). Not a matching window.
RESIDUAL_ASSIGNMENT_RADIUS_MM: float = 1.0

# --- sigma_vtx-vtx fit ----------------------------------------------------
# 300 bins over +-6 mm = 0.04 mm, commensurate with the model grid
# (pv_finder.utils.pairwise_dz.DEFAULT_PAIRWISE_BINS).
PAIRWISE_BINS: int = 300


@dataclass(frozen=True)
class PeakFinding:
    """Peak-finding cuts applied to the predicted vertex density."""

    amplitude: float  # bins above this are "on" (``threshold`` in the code)
    min_width: int  # minimum region width in bins
    integral: float  # minimum region integral
    min_height: float  # minimum region maximum; 0 disables
    centroid_halfwidth: int  # local centroid +- bins about the maximum


@dataclass(frozen=True)
class DensityModel:
    """A PV-Finder density model together with the cuts it is run at."""

    name: str
    checkpoint: str
    model_type: str  # run_eval_pvf_run3 --e2e-type
    unet_channels: int
    latent_channels: int
    hidden: tuple[int, ...]
    peaks: PeakFinding
    frozen: bool
    note: str = ""


@dataclass(frozen=True)
class GNNWorkingPoint:
    """Track-to-vertex association network and its score cut."""

    checkpoint: str | None
    t: float | None
    frozen: bool
    note: str = ""


@dataclass(frozen=True)
class Regime:
    """One complete, self-consistent configuration."""

    name: str
    description: str
    match_window_mm: float
    hs_match_window_mm: float
    pairwise_bins: int
    density: DensityModel
    gnn: GNNWorkingPoint
    pending: tuple[str, ...] = field(default_factory=tuple)

    @property
    def peaks(self) -> PeakFinding:
        """Peak-finding cuts of the regime's density model."""
        return self.density.peaks

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for result metadata (JSON/pickle safe)."""
        return asdict(self)

    def require_frozen(self) -> None:
        """Raise if any part of the regime is still a placeholder."""
        if self.pending:
            raise RuntimeError(
                f"working point '{self.name}' is not frozen: " + ", ".join(self.pending)
            )


HLLHC_PEAKS = PeakFinding(
    amplitude=0.01, min_width=3, integral=0.30, min_height=0.03, centroid_halfwidth=3
)
LOWPU_PEAKS = PeakFinding(
    amplitude=0.01, min_width=3, integral=0.50, min_height=0.0, centroid_halfwidth=3
)

# HL-LHC density-model candidates (D3). FREEZE PENDING: v6 is the default
# until Mattia freezes one; switching is a one-line change of HLLHC.density.
V6 = DensityModel(
    name="v6",
    checkpoint="model_weights/hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth",
    model_type="v2",
    unet_channels=280,
    latent_channels=4,
    hidden=(128,) * 5,
    peaks=HLLHC_PEAKS,
    frozen=False,
    note="production v6; FREEZE PENDING (D3)",
)
S8 = DensityModel(
    name="s8",
    checkpoint="model_weights/hllhc_alleta_v6_s8_1p1ep_phase2_step_450000_fullstate.pth",
    model_type="v2",
    unet_channels=80,
    latent_channels=1,
    hidden=(128,) * 5,
    # Cuts that give v6's fake rate: matched-FP scan at 0.17 mm, 12 Nov 2026,
    # 0.182/0.0182 on r16443 and 0.186/0.0186 on r16638; the two-file mean
    # 0.184 gives fakes/event -0.06 / +0.10 vs v6 (0.18 gave +0.09 / +0.24).
    # docs/evaluation/model_freeze_candidates_2026-11-12.md
    peaks=PeakFinding(
        amplitude=0.01,
        min_width=3,
        integral=0.184,
        min_height=0.0184,
        centroid_halfwidth=3,
    ),
    frozen=False,
    note="CPU candidate, 9x faster; cuts matched to v6 fake rate; FREEZE PENDING",
)
# Low pile-up: PVF-e400 (two-phase, no KDE), decided 11 Nov 2026.
E400 = DensityModel(
    name="e400",
    # In-repo copy (sha1-identical) of atlas_pvfinder/model_weights/
    # e2e_attempt4_mlp_hist/, copied 2026-11-11 so the repo is self-contained.
    checkpoint=(
        "model_weights/lowpu_e400/"
        "e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_400_fullstate.pth"
    ),
    model_type="v1",
    unet_channels=64,
    latent_channels=1,
    hidden=(100,) * 5,
    peaks=LOWPU_PEAKS,
    frozen=True,
    note="config_mlp_hist_e2e.yml: 50 MLP + 400 end-to-end epochs, 359,616 params",
)

HLLHC = Regime(
    name="hllhc",
    description="HL-LHC ITk simulation, 0 <= mu <= 200",
    match_window_mm=MATCH_WINDOW_MM,
    hs_match_window_mm=HS_MATCH_WINDOW_MM,
    pairwise_bins=PAIRWISE_BINS,
    density=V6,
    gnn=GNNWorkingPoint(
        # Lowest validation-monitor loss of the 540 shard epochs (0.255807;
        # plateaued, last pass -0.07%); beats v1 e213 on identical events.
        # docs/evaluation/gnn_v2_freeze_2026-11-12.md
        checkpoint=(
            "model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/"
            "ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt"
        ),
        t=None,
        frozen=False,
        note="full-mu v2 e527 (chosen 12 Nov 2026); t PENDING (Mattia decides t)",
    ),
    pending=("density model (D3)", "GNN t"),
)
LOWPU = Regime(
    name="lowpu",
    description="Run 2 simulation, Run 2 and Run 3 collision data",
    match_window_mm=MATCH_WINDOW_MM,
    hs_match_window_mm=HS_MATCH_WINDOW_MM,
    pairwise_bins=PAIRWISE_BINS,
    density=E400,
    gnn=GNNWorkingPoint(
        checkpoint=None,
        t=0.99,
        frozen=False,
        note="t = 0.99 proposed; GNN to be re-run on e400 peaks",
    ),
    pending=("GNN checkpoint", "GNN t"),
)

REGIMES: dict[str, Regime] = {r.name: r for r in (HLLHC, LOWPU)}


def window_tag(window_mm: float) -> str:
    """Key-safe label of a window, e.g. 0.17 -> '0p17' (used in result keys)."""
    return f"{window_mm:g}".replace(".", "p")


def get_regime(name: str) -> Regime:
    """Return the named regime (``hllhc`` or ``lowpu``)."""
    try:
        return REGIMES[name]
    except KeyError:
        raise ValueError(f"unknown regime {name!r}; choose from {sorted(REGIMES)}")


def record_overrides(regime: Regime, used: dict[str, Any]) -> dict[str, Any]:
    """Return the entries of ``used`` that differ from the regime defaults.

    ``used`` maps keys of ``regime_defaults(regime)`` to the values a script
    actually ran with.
    """
    ref = regime_defaults(regime)
    return {k: v for k, v in used.items() if k in ref and ref[k] != v}


def regime_defaults(regime: Regime) -> dict[str, Any]:
    """Flat dict of the CLI-level defaults a regime implies."""
    p = regime.peaks
    return {
        "match_window_mm": regime.match_window_mm,
        "peak_threshold": p.amplitude,
        "integral_threshold": p.integral,
        "integral_threshold_res": p.integral,
        "min_height": p.min_height,
        "centroid_halfwidth": p.centroid_halfwidth,
        "pairwise_bins": regime.pairwise_bins,
    }
