"""
Per-vertex visualization of e2e histogram predictions vs analytical KDE.

Provides two plotting functions:
  - plot_vertex_zoom: zoomed 3-panel view around a single truth vertex
  - plot_event_overview: full z-range 2-panel overview for an entire event

All histogram curves are normalized to their own peak ([0, 1] scale) before
plotting so that the predicted histogram (e2e, typically O(1e-4)) and the
analytical KDE (typically O(1–100)) are directly comparable in shape.

Truth vertex sources:
  MC   -- generator-level z-positions from H5 ``pv`` dataset, labelled
          "Gen. truth vertex" (vertical line) + "Truth target" (dotted curve =
          the training-target histogram from ``target_y_split[:, 0, :]``).
  Run3 -- AMVF reconstructed vertices (RecoVertex_z − BeamPosZ, nTracks≥2),
          labelled "AMVF vertex".  No truth histogram curve is available.
"""

from __future__ import annotations

import os
import warnings

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pv_finder.data.feature_loading import Z_MAX, Z_MIN
from pv_finder.diagnostics.domain_shift_investigation.kde_study.kde_comparison_plots import (
    _flatten_kde,
    _full_z_axis,
)

# Suppress matplotlib font-fallback warning raised at caller's stacklevel
# (matplotlib uses stacklevel > 1, so the module filter "matplotlib" doesn't apply)
warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from current font")

# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

mpl.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 1.5,
    }
)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COL_E2E = "#d62728"  # red  -- e2e model output
COL_ANALYTICAL = "#1f77b4"  # blue -- analytical KDE
COL_TRUTH_HIST = "#2ca02c"  # green -- MC truth target histogram
COL_VERTEX = "black"  # truth vertex line
COL_WINDOW = "#2ca02c"  # +/-match_window shaded band


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _z_to_label(z_mm: float) -> str:
    """Format z_mm for filenames: 'z-42.4mm' (negative) or 'z11.9mm' (positive)."""
    return f"z{z_mm:+.1f}mm".replace("+", "")


def _safe_norm(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize array to [0, 1] by its maximum absolute value.

    Returns (normalized_array, scale_factor).  If the max is effectively
    zero the array is returned unchanged and scale_factor = 1.0.
    """
    mx = float(np.max(np.abs(arr)))
    if mx > 1e-30:
        return arr / mx, mx
    return arr.copy(), 1.0


def _vtx_label(dataset_label: str) -> str:
    """Legend label for truth vertex vertical lines."""
    return "Gen. truth vertex" if dataset_label.lower() == "mc" else "AMVF vertex"


# ---------------------------------------------------------------------------
# 1. Per-vertex zoom plot
# ---------------------------------------------------------------------------


def plot_vertex_zoom(
    hist_e2e: np.ndarray,
    hist_analytical: np.ndarray,
    truth_z: float,
    pred_peaks: list[tuple[float, float]],
    event_idx: int,
    vtx_idx: int,
    dataset_label: str,
    output_dir: str,
    window_mm: float = 8.0,
    tracks_z0: np.ndarray | None = None,
    tracks_d0: np.ndarray | None = None,
    tracks_d0_err: np.ndarray | None = None,
    hist_truth: np.ndarray | None = None,
    match_window_mm: float = 0.5,
) -> None:
    """Three-panel per-vertex visualization.

    Panel 1 -- Histogram overlay zoomed to [truth_z +/- window_mm].
      All curves normalized to own peak for shape comparison.
      - Red solid  : Predicted hist. (e2e model output)
      - Blue dashed: Analytical KDE
      - Green dotted (MC only): Truth target histogram (training target from H5)
      - Black dashed vertical line: truth vertex position (Gen. truth / AMVF)
      - Green shaded band: +/-match_window_mm matching window
      - Filled red dot: predicted peak within matching window
      - Open red circle: predicted peak outside matching window

    Panel 2 -- Normalized residual (pred - ana) in the same z window.

    Panel 3 -- Track impact-parameter significance vs z0 (same z range as panels 1/2).
      y-axis: |d0| / sigma_d0.  A track from the vertex at truth_z should have
      small d0 (close approach to the beam line) relative to its uncertainty,
      giving |d0|/sigma_d0 ~ 0.  Pileup tracks have systematically larger
      significance.  Coloured by log10(sigma_d0) to reveal the measurement
      quality of each track.  X-axis is aligned with panels 1 and 2.
    """
    _ensure_dir(output_dir)

    has_tracks = (
        tracks_z0 is not None and tracks_d0 is not None and tracks_d0_err is not None
    )

    if has_tracks:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 11),
            sharex=False,
            gridspec_kw={"height_ratios": [4, 1, 2], "hspace": 0.35},
        )
        ax_hist, ax_res, ax_trk = axes
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=False,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.35},
        )
        ax_hist, ax_res = axes

    z = _full_z_axis()
    e2e_norm, e2e_mx = _safe_norm(_flatten_kde(hist_e2e))
    ana_norm, _ = _safe_norm(_flatten_kde(hist_analytical))

    lo = truth_z - window_mm
    hi = truth_z + window_mm
    mask = (z >= lo) & (z <= hi)
    z_win = z[mask]

    # ------------------------------------------------------------------
    # Panel 1: histogram overlay
    # ------------------------------------------------------------------
    ax_hist.plot(z_win, e2e_norm[mask], color=COL_E2E, label="Predicted hist.")
    ax_hist.plot(
        z_win,
        ana_norm[mask],
        color=COL_ANALYTICAL,
        linestyle="--",
        label="Analytical KDE",
    )
    if hist_truth is not None:
        truth_norm, _ = _safe_norm(_flatten_kde(hist_truth))
        ax_hist.plot(
            z_win,
            truth_norm[mask],
            color=COL_TRUTH_HIST,
            linestyle=":",
            label="Truth target",
        )

    ax_hist.axvspan(
        truth_z - match_window_mm,
        truth_z + match_window_mm,
        alpha=0.15,
        color=COL_WINDOW,
        label=f"+/-{match_window_mm} mm",
    )
    # Truth vertex line — added to legend
    ax_hist.axvline(
        truth_z,
        color=COL_VERTEX,
        linestyle="--",
        linewidth=1.2,
        label=_vtx_label(dataset_label),
    )

    # Predicted peak markers (heights rescaled to normalized axis)
    for pz, ph in pred_peaks:
        if lo <= pz <= hi:
            ph_norm = ph / e2e_mx
            mfc = COL_E2E if abs(pz - truth_z) <= match_window_mm else "none"
            ax_hist.plot(
                pz,
                ph_norm,
                marker="o",
                markersize=7,
                color=COL_E2E,
                markerfacecolor=mfc,
                markeredgewidth=1.5,
                linestyle="none",
                zorder=5,
            )

    ax_hist.set_xlim(lo, hi)
    ax_hist.set_ylabel("Norm. amplitude")
    ax_hist.set_title(
        f"{dataset_label.upper()} Ev {event_idx} | Vtx {vtx_idx} | "
        f"truth z = {truth_z:.2f} mm"
    )
    ax_hist.legend(loc="upper right", fontsize=9)

    # ------------------------------------------------------------------
    # Panel 2: normalized residual strip
    # ------------------------------------------------------------------
    residual = e2e_norm - ana_norm
    ax_res.plot(z_win, residual[mask], color=COL_E2E, linewidth=1.0)
    ax_res.axhline(0, color="black", linewidth=0.8)
    ax_res.axvspan(
        truth_z - match_window_mm,
        truth_z + match_window_mm,
        alpha=0.1,
        color=COL_WINDOW,
    )
    ax_res.axvline(truth_z, color=COL_VERTEX, linestyle="--", linewidth=1.0)
    ax_res.set_xlim(lo, hi)
    ax_res.set_xlabel("z [mm]")
    ax_res.set_ylabel("pred - ana\n(norm.)")

    # ------------------------------------------------------------------
    # Panel 3: track scatter — same z range as panels 1/2 for axis alignment
    # ------------------------------------------------------------------
    if has_tracks:
        sel = (tracks_z0 >= lo) & (tracks_z0 <= hi)
        tz = tracks_z0[sel]
        td = tracks_d0[sel]
        terr = tracks_d0_err[sel]

        sig_d0 = np.where(terr > 0, np.abs(td) / terr, np.nan)
        log_err = np.log10(np.clip(terr, 1e-4, None))

        sc = ax_trk.scatter(
            tz, sig_d0, c=log_err, cmap="viridis", s=10, alpha=0.6, linewidths=0
        )
        if len(tz) > 0:
            cb = fig.colorbar(sc, ax=ax_trk)
            cb.set_label("log10(sigma_d0)")

        ax_trk.axvspan(
            truth_z - match_window_mm,
            truth_z + match_window_mm,
            alpha=0.1,
            color=COL_WINDOW,
        )
        ax_trk.axvline(truth_z, color=COL_VERTEX, linestyle="--", linewidth=1.0)
        ax_trk.set_xlim(lo, hi)  # aligned with panels 1 and 2
        ax_trk.set_xlabel("z0 [mm]")
        ax_trk.set_ylabel("|d0| / sigma_d0")
        ax_trk.set_ylim(bottom=0)

    stem = f"event{event_idx:04d}_vtx{vtx_idx:02d}_{_z_to_label(truth_z)}"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Full event overview
# ---------------------------------------------------------------------------


def plot_event_overview(
    hist_e2e: np.ndarray,
    hist_analytical: np.ndarray,
    truth_vertices: list[float],
    pred_peaks: list[tuple[float, float]],
    event_idx: int,
    dataset_label: str,
    output_dir: str,
    hist_truth: np.ndarray | None = None,
    match_window_mm: float = 0.5,
) -> None:
    """Full z-range 2-panel overview: normalized histogram overlay + residual strip.

    Truth vertex source depends on dataset_label (see module docstring).
    Only the first truth vertex line is labelled in the legend to avoid
    duplicate entries when many vertices are present.
    """
    _ensure_dir(output_dir)

    fig, (ax_main, ax_res) = plt.subplots(
        2,
        1,
        figsize=(18, 5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    z = _full_z_axis()
    e2e_norm, e2e_mx = _safe_norm(_flatten_kde(hist_e2e))
    ana_norm, _ = _safe_norm(_flatten_kde(hist_analytical))

    # ------------------------------------------------------------------
    # Panel 1: full-range histogram overlay (normalized)
    # ------------------------------------------------------------------
    ax_main.plot(z, e2e_norm, color=COL_E2E, linewidth=1.2, label="Predicted hist.")
    ax_main.plot(
        z,
        ana_norm,
        color=COL_ANALYTICAL,
        linestyle="--",
        linewidth=1.2,
        label="Analytical KDE",
    )
    if hist_truth is not None:
        truth_norm, _ = _safe_norm(_flatten_kde(hist_truth))
        ax_main.plot(
            z,
            truth_norm,
            color=COL_TRUTH_HIST,
            linestyle=":",
            linewidth=1.0,
            label="Truth target",
        )

    # Truth vertex lines: label only the first to avoid duplicate legend entries
    vtx_name = _vtx_label(dataset_label)
    for i, vz in enumerate(truth_vertices):
        lbl = vtx_name if i == 0 else ""
        ax_main.axvline(
            vz, color=COL_WINDOW, linestyle="--", linewidth=0.9, alpha=0.7, label=lbl
        )
        ax_main.axvspan(
            vz - match_window_mm, vz + match_window_mm, alpha=0.12, color=COL_WINDOW
        )

    # Predicted peak markers on normalized scale
    for pz, ph in pred_peaks:
        in_band = any(abs(pz - vz) <= match_window_mm for vz in truth_vertices)
        mfc = COL_E2E if in_band else "none"
        ax_main.plot(
            pz,
            ph / e2e_mx,
            marker="o",
            markersize=5,
            color=COL_E2E,
            markerfacecolor=mfc,
            markeredgewidth=1.2,
            linestyle="none",
            zorder=5,
        )

    ax_main.set_xlim(Z_MIN, Z_MAX)
    ax_main.set_ylabel("Norm. amplitude")
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.set_title(
        f"{dataset_label.upper()} Event {event_idx} -- full event overview"
    )

    # ------------------------------------------------------------------
    # Panel 2: normalized residual strip
    # ------------------------------------------------------------------
    ax_res.plot(z, e2e_norm - ana_norm, color=COL_E2E, linewidth=0.8)
    ax_res.axhline(0, color="black", linewidth=0.8)
    for vz in truth_vertices:
        ax_res.axvline(vz, color=COL_WINDOW, alpha=0.5, linewidth=0.7)

    ax_res.set_xlim(Z_MIN, Z_MAX)
    ax_res.set_xlabel("z [mm]")
    ax_res.set_ylabel("pred - ana\n(norm.)")

    stem = f"event{event_idx:04d}_overview"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
