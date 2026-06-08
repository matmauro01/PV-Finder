"""
Per-vertex visualization of e2e histogram predictions vs analytical KDE.

Two plotting functions:
  - plot_vertex_zoom: zoomed 3-panel view around a single truth vertex
  - plot_event_overview: full z-range 2-panel overview for an entire event

Y-axis: the e2e predicted histogram is plotted in raw (un-normalized) values so
that peak heights are directly readable.  The analytical KDE and truth target
(which live on different scales) are rescaled so their global peak matches the
e2e global peak, enabling visual shape comparison on the same axis.

Truth vertex sources:
  MC   -- generator-level z-positions from H5 ``pv`` dataset, labelled
          "Gen. truth vertex" (vertical line) + "Truth target" (dotted curve =
          the training-target histogram from ``target_y_split[:, 0, :]``).
  Run3 -- AMVF reconstructed vertices (RecoVertex_z - BeamPosZ, nTracks>=2),
          labelled "AMVF vertex".  No truth histogram curve is available.
          Note: AMVF vertices are beam-corrected while track z0 values are in
          the detector frame; the offset is typically O(1 mm).
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

# Suppress matplotlib font-fallback warnings
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
        # Clean axis style: no top/right spines, no tick marks
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "axes.unicode_minus": False,  # ASCII '-'; U+2212 is a box in the Agg font
    }
)

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
COL_E2E = "#d62728"  # red  -- e2e model output
COL_ANALYTICAL = "#1f77b4"  # blue -- analytical KDE
COL_T2KDE = "#ff7f0e"  # orange -- T2KDE model-computed KDE
COL_TRUTH_HIST = "#2ca02c"  # green -- MC truth target histogram
COL_VERTEX = "black"  # focused truth vertex line
COL_OTHER_VTX = "gray"  # other truth vertices visible in window
COL_WINDOW = "#2ca02c"  # +/-match_window shaded band


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _z_to_label(z_mm: float) -> str:
    """Format z_mm for filenames: 'z-42.4mm' (negative) or 'z11.9mm' (positive)."""
    return f"z{z_mm:+.1f}mm".replace("+", "")


def _rescale_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rescale *source* so its global peak matches *target*'s global peak.

    Used to overlay the analytical KDE (O(1--100)) on the same y-axis as
    the e2e predicted histogram (O(0.01--1)) for shape comparison.
    Returns a copy; never modifies the input arrays.
    """
    src_mx = float(np.max(np.abs(source)))
    tgt_mx = float(np.max(np.abs(target)))
    if src_mx > 1e-30 and tgt_mx > 1e-30:
        return source * (tgt_mx / src_mx)
    return source.copy()


def _vtx_label(dataset_label: str) -> str:
    """Legend label for truth vertex vertical lines."""
    return "Gen. truth vertex" if dataset_label.lower() == "mc" else "AMVF vertex"


def _draw_vertex_lines(
    ax: plt.Axes,
    vtx_list: list[float],
    focused_z: float,
    lo: float,
    hi: float,
    dataset_label: str,
    add_legend: bool = True,
    truth_name: str | None = None,
) -> None:
    """Draw vertical lines for all truth vertices visible in [lo, hi].

    The focused vertex is drawn in black; other vertices in grey.
    Legend entries are added only once per category. ``truth_name`` overrides
    the focused-vertex legend label (else derived from ``dataset_label``).
    """
    focused_drawn = False
    others_drawn = False
    name = truth_name or _vtx_label(dataset_label)
    for vz in vtx_list:
        if not (lo <= vz <= hi):
            continue
        if abs(vz - focused_z) < 1e-6:
            lbl = name if add_legend and not focused_drawn else ""
            ax.axvline(vz, color=COL_VERTEX, linestyle="--", linewidth=1.4, label=lbl)
            focused_drawn = True
        else:
            lbl = "Other vertices" if add_legend and not others_drawn else ""
            ax.axvline(
                vz,
                color=COL_OTHER_VTX,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=lbl,
            )
            others_drawn = True


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
    hist_t2kde: np.ndarray | None = None,
    match_window_mm: float = 0.5,
    all_truth_vertices: list[float] | None = None,
    vertex_label: str | None = None,
    truth_name: str | None = None,
) -> None:
    """Three-panel per-vertex visualization.

    Panel 1 -- Histogram overlay zoomed to [truth_z +/- window_mm]: predicted
    hist (red), analytical KDE (blue dashed), optional T2KDE (orange) and truth
    target (green); focused vertex (black) + others (grey); match-window band;
    filled/open red dots = peaks in/out of the window.
    Panel 2 -- Normalized residual (pred - ana). Panel 3 -- track |d0|/sigma_d0
    vs z0, coloured by log10(sigma_d0), x-aligned via sharex.

    Parameters
    ----------
    hist_t2kde:
        T2KDE model-computed KDE, shape ``(12, 1000)`` or ``(12000,)``.
        If provided, plotted as an orange dash-dot line (rescaled).
    all_truth_vertices:
        Full list of truth vertex z-positions for this event.  All vertices
        falling within the zoom window are drawn as vertical lines.
        If None, only the focused vertex (truth_z) is drawn.
    """
    _ensure_dir(output_dir)

    has_tracks = (
        tracks_z0 is not None and tracks_d0 is not None and tracks_d0_err is not None
    )

    n_panels = 3 if has_tracks else 2
    ratios = [4, 1, 2] if has_tracks else [4, 1]
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(12, 11 if has_tracks else 7),
        sharex=True,
        gridspec_kw={"height_ratios": ratios, "hspace": 0.12},
    )
    ax_hist = axes[0]
    ax_res = axes[1]
    ax_trk = axes[2] if has_tracks else None

    z = _full_z_axis()
    e2e_raw = _flatten_kde(hist_e2e)
    ana_raw = _flatten_kde(hist_analytical)
    # Rescale KDE to e2e scale for visual shape comparison
    ana_scaled = _rescale_to(ana_raw, e2e_raw)

    lo = truth_z - window_mm
    hi = truth_z + window_mm
    mask = (z >= lo) & (z <= hi)
    z_win = z[mask]

    vtx_list = all_truth_vertices if all_truth_vertices is not None else [truth_z]

    # ------------------------------------------------------------------
    # Panel 1: histogram overlay (raw e2e values)
    # ------------------------------------------------------------------
    ax_hist.plot(z_win, e2e_raw[mask], color=COL_E2E, label="Predicted hist.")
    ax_hist.plot(
        z_win,
        ana_scaled[mask],
        color=COL_ANALYTICAL,
        linestyle="--",
        label="Analytical KDE (rescaled)",
    )
    if hist_t2kde is not None:
        t2kde_raw = _flatten_kde(hist_t2kde)
        t2kde_scaled = _rescale_to(t2kde_raw, e2e_raw)
        ax_hist.plot(
            z_win,
            t2kde_scaled[mask],
            color=COL_T2KDE,
            linestyle="-.",
            label="T2KDE model (rescaled)",
        )
    if hist_truth is not None:
        truth_raw = _flatten_kde(hist_truth)
        truth_scaled = _rescale_to(truth_raw, e2e_raw)
        ax_hist.plot(
            z_win,
            truth_scaled[mask],
            color=COL_TRUTH_HIST,
            linestyle=":",
            label="Truth target (rescaled)",
        )

    ax_hist.axvspan(
        truth_z - match_window_mm,
        truth_z + match_window_mm,
        alpha=0.15,
        color=COL_WINDOW,
        label=f"+/-{match_window_mm} mm",
    )

    _draw_vertex_lines(
        ax_hist, vtx_list, truth_z, lo, hi, dataset_label, truth_name=truth_name
    )

    # Predicted peak markers (raw heights)
    for pz, ph in pred_peaks:
        if lo <= pz <= hi:
            mfc = COL_E2E if abs(pz - truth_z) <= match_window_mm else "none"
            ax_hist.plot(
                pz,
                ph,
                marker="o",
                markersize=7,
                color=COL_E2E,
                markerfacecolor=mfc,
                markeredgewidth=1.5,
                linestyle="none",
                zorder=5,
            )

    ax_hist.set_ylabel("Predicted histogram")
    label_str = f" [{vertex_label.upper()}]" if vertex_label else ""
    ax_hist.set_title(
        f"{dataset_label.upper()} Ev {event_idx} | Vtx {vtx_idx}{label_str} | "
        f"truth z = {truth_z:.2f} mm"
    )
    ax_hist.legend(loc="upper right", fontsize=9)

    # ------------------------------------------------------------------
    # Panel 2: residual strip (raw e2e - rescaled analytical)
    # ------------------------------------------------------------------
    residual = e2e_raw - ana_scaled
    ax_res.plot(z_win, residual[mask], color=COL_E2E, linewidth=1.0)
    ax_res.axhline(0, color="black", linewidth=0.8)
    ax_res.axvspan(
        truth_z - match_window_mm,
        truth_z + match_window_mm,
        alpha=0.1,
        color=COL_WINDOW,
    )
    _draw_vertex_lines(
        ax_res, vtx_list, truth_z, lo, hi, dataset_label, add_legend=False
    )
    ax_res.set_ylabel("pred - ana\n(rescaled)")
    # xlabel only on the bottom-most panel (set below or on ax_trk)
    if not has_tracks:
        ax_res.set_xlabel("z [mm]")

    # ------------------------------------------------------------------
    # Panel 3: track scatter -- aligned via sharex
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
            cb = fig.colorbar(
                sc,
                ax=list(axes),
                location="right",
                shrink=0.3,
                pad=0.02,
                anchor=(0.0, 0.0),
            )
            cb.set_label("log10(sigma_d0)")

        ax_trk.axvspan(
            truth_z - match_window_mm,
            truth_z + match_window_mm,
            alpha=0.1,
            color=COL_WINDOW,
        )
        _draw_vertex_lines(
            ax_trk, vtx_list, truth_z, lo, hi, dataset_label, add_legend=False
        )
        ax_trk.set_xlabel("z [mm]")
        ax_trk.set_ylabel("|d0| / sigma_d0")
        ax_trk.set_ylim(bottom=0)

    # Set shared x-limits (sharex propagates to all panels)
    axes[0].set_xlim(lo, hi)

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
    hist_t2kde: np.ndarray | None = None,
    match_window_mm: float = 0.5,
) -> None:
    """Full z-range 2-panel overview: normalized histogram overlay + residual.

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
    e2e_raw = _flatten_kde(hist_e2e)
    ana_raw = _flatten_kde(hist_analytical)
    ana_scaled = _rescale_to(ana_raw, e2e_raw)

    # ------------------------------------------------------------------
    # Panel 1: full-range histogram overlay (raw e2e values)
    # ------------------------------------------------------------------
    ax_main.plot(z, e2e_raw, color=COL_E2E, linewidth=1.2, label="Predicted hist.")
    ax_main.plot(
        z,
        ana_scaled,
        color=COL_ANALYTICAL,
        linestyle="--",
        linewidth=1.2,
        label="Analytical KDE (rescaled)",
    )
    if hist_t2kde is not None:
        t2kde_raw = _flatten_kde(hist_t2kde)
        t2kde_scaled = _rescale_to(t2kde_raw, e2e_raw)
        ax_main.plot(
            z,
            t2kde_scaled,
            color=COL_T2KDE,
            linestyle="-.",
            linewidth=1.2,
            label="T2KDE model (rescaled)",
        )
    if hist_truth is not None:
        truth_raw = _flatten_kde(hist_truth)
        truth_scaled = _rescale_to(truth_raw, e2e_raw)
        ax_main.plot(
            z,
            truth_scaled,
            color=COL_TRUTH_HIST,
            linestyle=":",
            linewidth=1.0,
            label="Truth target (rescaled)",
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

    # Predicted peak markers (raw heights)
    for pz, ph in pred_peaks:
        in_band = any(abs(pz - vz) <= match_window_mm for vz in truth_vertices)
        mfc = COL_E2E if in_band else "none"
        ax_main.plot(
            pz,
            ph,
            marker="o",
            markersize=5,
            color=COL_E2E,
            markerfacecolor=mfc,
            markeredgewidth=1.2,
            linestyle="none",
            zorder=5,
        )

    ax_main.set_xlim(Z_MIN, Z_MAX)
    ax_main.set_ylabel("Predicted histogram")
    ax_main.legend(loc="upper right", fontsize=9)
    ax_main.set_title(
        f"{dataset_label.upper()} Event {event_idx} -- full event overview"
    )

    # ------------------------------------------------------------------
    # Panel 2: residual strip (raw e2e - rescaled analytical)
    # ------------------------------------------------------------------
    ax_res.plot(z, e2e_raw - ana_scaled, color=COL_E2E, linewidth=0.8)
    ax_res.axhline(0, color="black", linewidth=0.8)
    for vz in truth_vertices:
        ax_res.axvline(vz, color=COL_WINDOW, alpha=0.5, linewidth=0.7)

    ax_res.set_xlim(Z_MIN, Z_MAX)
    ax_res.set_xlabel("z [mm]")
    ax_res.set_ylabel("pred - ana\n(rescaled)")

    stem = f"event{event_idx:04d}_overview"
    fig.savefig(os.path.join(output_dir, stem + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)
