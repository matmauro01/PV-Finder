"""HL-LHC pre-GNN vertex-finding figures (paper items H1/H2), ATLAS style.

Pure plotting: every function takes numbers already computed by
``scripts/h1_vertex_finding.py`` (from saved evaluations, the fair-comparison
cache and its window-scan JSON) and draws PV-Finder against AMVF with the
shared paper style (:mod:`pv_finder.diagnostics.paper_figures.style`).

- :func:`plot_yield_vs_mu`: reconstructed vertices per event vs mu (Fig. 6).
- :func:`plot_ratio_vs_mu`: a metric (efficiency, fake fraction, fakes or
  splits per event) vs mu, with the paired difference.
- :func:`plot_residuals`: signed and absolute single-vertex residuals (Fig. 7b).
- :func:`plot_window_justification`: |z_reco - z_true| up to 1 mm with the
  68% widths, 3 x 68% and the frozen 0.17 mm window marked.
- :func:`plot_window_scan`: efficiency vs common window and the difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.plot_style import use_atlas_style
from pv_finder.diagnostics.paper_figures import style

ARM = {
    "pvf": {"label": "PV-Finder", "color": style.PVF_COLOR, "marker": "o", "ls": "-"},
    "amvf": {"label": "AMVF", "color": style.AMVF_COLOR, "marker": "s", "ls": "--"},
}


def _two_panel(ratio: tuple[float, float] = (3.0, 1.3)) -> tuple[plt.Figure, Any]:
    """Main panel plus a difference panel sharing the x axis."""
    use_atlas_style()
    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(8.0, 7.6), sharex=True,
        gridspec_kw={"height_ratios": ratio, "hspace": 0.06},
    )  # fmt: skip
    return fig, (ax, axd)


def plot_yield_vs_mu(
    binned: dict[str, dict[str, list[float]]],
    label_lines: Sequence[str],
    out_dir: Path,
    name: str,
) -> list[Path]:
    """Mean reconstructed vertices per event vs mu; truth (nTrk >= 2) dashed.

    ``binned[key]`` holds lists ``x`` (mean mu), ``y`` (mean) and ``e`` (SEM)
    for ``key`` in ``pvf``, ``amvf``, ``truth``.
    """
    fig, ax = style.new_figure(8.0, 6.0)
    for arm in ("pvf", "amvf"):
        b, st = binned[arm], ARM[arm]
        ax.errorbar(b["x"], b["y"], yerr=b["e"], fmt=st["marker"], color=st["color"],
                    markersize=5, elinewidth=1.2, label=st["label"])  # fmt: skip
    t = binned["truth"]
    ax.plot(t["x"], t["y"], "--", color="0.35", linewidth=1.5,
            label="Truth vertices, $\\geq 2$ selected tracks")  # fmt: skip
    ax.set_xlim(0, 210)
    ax.set_ylim(0, 1.55 * max(max(t["y"]), max(binned["pvf"]["y"])))
    ax.set_xlabel("Actual number of interactions per crossing, $\\mu$")
    ax.set_ylabel("Vertices / event")
    style.label_block(ax, style.SIM_STATUS, label_lines)
    top = 0.95 - 0.055 * (len(label_lines) + 1) - 0.03
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, top), frameon=False,
              fontsize="small")  # fmt: skip
    return style.save(fig, out_dir, name)


def plot_ratio_vs_mu(
    series: dict[str, Any],
    ylabel: str,
    dlabel: str,
    label_lines: Sequence[str],
    out_dir: Path,
    name: str,
    bounded: bool = False,
) -> list[Path]:
    """One metric per mu bin, PV-Finder vs AMVF, and PVF - AMVF (paired errors).

    ``series`` is a ``ratio_vs_mu`` result of ``scripts/h1_vertex_finding.py``:
    ``x``, ``xerr`` and, per arm, ``val`` / ``err``, plus ``diff`` / ``diff_err``.
    ``bounded`` keeps an efficiency axis from running far past 100%.
    """
    fig, (ax, axd) = _two_panel()
    x, xe = np.asarray(series["x"]), np.asarray(series["xerr"])
    for arm in ("pvf", "amvf"):
        st = ARM[arm]
        ax.errorbar(x, series[arm]["val"], xerr=xe, yerr=series[arm]["err"],
                    fmt=st["marker"], color=st["color"], markersize=5,
                    elinewidth=1.2, label=st["label"])  # fmt: skip
    vals = np.concatenate([series["pvf"]["val"], series["amvf"]["val"]])
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if bounded:
        ax.set_ylim(lo - 0.35 * (100 - lo), 100 + 0.55 * (100 - lo))
        ax.legend(loc="lower left", frameon=False, fontsize="small")
    else:
        ax.set_ylim(0, 1.7 * hi)
        top = 0.95 - 0.055 * (len(label_lines) + 1) - 0.03
        ax.legend(loc="upper left", bbox_to_anchor=(0.03, top), frameon=False,
                  fontsize="small")  # fmt: skip
    ax.set_ylabel(ylabel)
    style.label_block(ax, style.SIM_STATUS, label_lines)
    axd.errorbar(x, series["diff"], xerr=xe, yerr=series["diff_err"], fmt="o",
                 color="black", markersize=4, elinewidth=1.2)  # fmt: skip
    axd.axhline(0.0, color="0.5", linewidth=1.0)
    d, de = np.asarray(series["diff"]), np.asarray(series["diff_err"])
    span = max(float(np.max(np.abs(d) + de)), 1e-3)
    axd.set_ylim(min(0.0, float(np.min(d - de))) - 0.3 * span,
                 max(0.0, float(np.max(d + de))) + 0.3 * span)  # fmt: skip
    axd.set_ylabel(dlabel, fontsize="small")
    axd.set_xlabel("Actual number of interactions per crossing, $\\mu$")
    axd.set_xlim(0, 200)
    return style.save(fig, out_dir, name)


def _hist_pct(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Histogram in percent of all entries (overflow included in the norm)."""
    cnt, _ = np.histogram(values, bins=edges)
    return 100.0 * cnt / max(len(values), 1)


def plot_residuals(
    signed_um: dict[str, np.ndarray],
    p68_um: dict[str, tuple[float, float]],
    label_lines: Sequence[str],
    out_dir: Path,
    name: str,
    bin_um: float = 2.0,
) -> list[Path]:
    """(a) signed and (b) absolute z_reco - z_truth, one-to-one within 1 mm.

    Dashed lines in (b) mark each algorithm's 68% value (``p68_um`` holds
    value and bootstrap error).
    """
    use_atlas_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    e_s = np.arange(-200.0, 200.0 + 1e-9, bin_um)
    e_a = np.arange(0.0, 200.0 + 1e-9, bin_um)
    for arm in ("pvf", "amvf"):
        st, s = ARM[arm], np.asarray(signed_um[arm])
        for ax, edges, vals in ((axes[0], e_s, s), (axes[1], e_a, np.abs(s))):
            ax.stairs(_hist_pct(vals, edges), edges, color=st["color"],
                      linestyle=st["ls"], linewidth=1.8, label=st["label"])  # fmt: skip
        v, e = p68_um[arm]
        axes[1].axvline(v, color=st["color"], linestyle=":", linewidth=1.6,
                        label=f"{st['label']} 68%: {v:.2f} $\\pm$ {e:.2f} $\\mu$m")  # fmt: skip
    axes[0].set_xlabel("$z_{\\mathrm{reco}} - z_{\\mathrm{truth}}$ [$\\mu$m]")
    axes[1].set_xlabel("$|z_{\\mathrm{reco}} - z_{\\mathrm{truth}}|$ [$\\mu$m]")
    for ax, tag in zip(axes, ("(a) Signed", "(b) Absolute")):
        ax.set_ylabel(f"Matched vertices / {bin_um:g} $\\mu$m [%]")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.7)  # room for the label block
        style.label_block(ax, style.SIM_STATUS, [*label_lines, tag])
        top = 0.95 - 0.055 * (len(label_lines) + 2) - 0.03
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, top), frameon=False,
                  fontsize="x-small")  # fmt: skip
    axes[0].set_xlim(-200, 200)
    axes[1].set_xlim(0, 200)
    fig.tight_layout()
    return style.save(fig, out_dir, name)


# Line styles of the reference lines in the window-justification plot, chosen
# so that two nearly equal 68% values (55.6 / 56.5 um) stay distinguishable.
_P68_LS = {"pvf": ":", "amvf": "--"}
_P68X3_LS = {"pvf": "-.", "amvf": (0, (6, 2, 1, 2, 1, 2))}


def _marks(ax: plt.Axes, p68_um: dict, window_mm: float, tops: dict) -> None:
    """68%, 3 x 68% and window lines, each ending at its own height ``tops``."""
    for arm in ("pvf", "amvf"):
        st, v = ARM[arm], p68_um[arm][0]
        ax.vlines(v, 0, tops[arm], color=st["color"], linestyle=_P68_LS[arm],
                  linewidth=1.8, label=f"{st['label']} 68%: {v:.2f} $\\mu$m")  # fmt: skip
        ax.vlines(3 * v, 0, tops[arm], color=st["color"], linestyle=_P68X3_LS[arm],
                  linewidth=1.5,
                  label=f"{st['label']} 3 $\\times$ 68%: {3 * v / 1e3:.4f} mm")  # fmt: skip
    ax.vlines(1e3 * window_mm, 0, tops["window"], color="black", linewidth=1.8,
              label=f"Matching window {window_mm:g} mm")  # fmt: skip


def plot_window_justification(
    abs_um: dict[str, np.ndarray],
    p68_um: dict[str, tuple[float, float]],
    window_mm: float,
    label_lines: Sequence[str],
    out_dir: Path,
    name: str,
) -> list[Path]:
    """|z_reco - z_true| with the 68% widths, 3 x 68% and the matching window.

    (a) linear, 0-250 um in 5 um bins, where the two 68% values are resolved;
    (b) log, the full 0-1 mm assignment range in 10 um bins. Reference lines
    stop below the label block so neither hides the other.
    """
    use_atlas_style()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2))
    for ax, (hi, bw, log, tag) in zip(
        axes, ((250.0, 5.0, False, "(a) Core"), (1000.0, 10.0, True, "(b) Full range"))
    ):
        edges = np.arange(0.0, hi + 1e-9, bw)
        ymax = 0.0
        for arm in ("pvf", "amvf"):
            st = ARM[arm]
            h = _hist_pct(np.asarray(abs_um[arm]), edges)
            ymax = max(ymax, float(h.max()))
            ax.stairs(h, edges, color=st["color"], linestyle=st["ls"], linewidth=1.8,
                      label=st["label"])  # fmt: skip
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-2, ymax * 200)
            tops = {"pvf": ymax * 1.2, "amvf": ymax * 0.8, "window": ymax * 1.6}
        else:
            ax.set_ylim(0, ymax * 2.0)
            tops = {"pvf": ymax * 1.15, "amvf": ymax * 0.95, "window": ymax * 1.3}
        _marks(ax, p68_um, window_mm, tops)
        ax.set_xlim(0, hi)
        ax.set_xlabel("$|z_{\\mathrm{reco}} - z_{\\mathrm{truth}}|$ [$\\mu$m]")
        ax.set_ylabel(f"Matched vertices / {bw:g} $\\mu$m [%]")
        style.label_block(ax, style.SIM_STATUS, [*label_lines, tag])
    # One legend, in the empty upper right of the log panel (above the tail,
    # right of every reference line).
    h, lab = axes[0].get_legend_handles_labels()
    axes[1].legend(h, lab, loc="upper right", bbox_to_anchor=(1.0, 0.70),
                   frameon=False, fontsize="x-small")  # fmt: skip
    fig.tight_layout()
    return style.save(fig, out_dir, name)


def plot_window_scan(
    rows: list[dict[str, Any]],
    window_mm: float,
    label_lines: Sequence[str],
    out_dir: Path,
    name: str,
) -> list[Path]:
    """Efficiency (merged credit, solid; one-to-one, open) vs window, and PVF - AMVF.

    ``rows`` are the scan rows of ``run_fair_comparison.py`` (``results_*.json``);
    errors are the paired event bootstrap stored there.
    """
    fig, (ax, axd) = _two_panel()
    w = np.array([r["window_mm"] for r in rows])
    for key, filled, suffix in (
        ("eff", True, ""),
        ("eff_strict", False, ", one-to-one"),
    ):
        for arm in ("pvf", "amvf"):
            st = ARM[arm]
            y = 100 * np.array([r[arm][key] for r in rows])
            e = 100 * np.array([r[arm][key + "_err"] for r in rows])
            ax.fill_between(w, y - e, y + e, color=st["color"], alpha=0.2, lw=0)
            ax.plot(w, y, color=st["color"], linestyle=st["ls"], marker=st["marker"],
                    markersize=3.5, markerfacecolor=st["color"] if filled else "none",
                    label=st["label"] + suffix)  # fmt: skip
        d = 100 * np.array([r["diff"][key] for r in rows])
        de = 100 * np.array([r["diff"][key + "_err"] for r in rows])
        axd.fill_between(w, d - de, d + de, color="0.6", alpha=0.35, lw=0)
        axd.plot(w, d, color="black", linestyle="-" if filled else "--",
                 marker="o", markersize=3, markerfacecolor="black" if filled else "none",
                 label="Eq. (1)" if filled else "one-to-one")  # fmt: skip
    acc = 100 * np.array([0.5 * (r["pvf"]["eff_accidental"] + r["amvf"]["eff_accidental"])
                          for r in rows])  # fmt: skip
    ax.plot(w, acc, ":", color="0.4", linewidth=1.5,
            label="Chance matches (3 mm shift, mean)")  # fmt: skip
    for a in (ax, axd):
        a.axvline(window_mm, color="black", linewidth=1.2, alpha=0.7)
    ax.text(window_mm + 0.01, 0.04, f"{window_mm:g} mm", transform=ax.get_xaxis_transform(),
            fontsize="small")  # fmt: skip
    ax.set_ylim(0, 135)
    ax.set_ylabel("Vertex-finding efficiency [%]")
    style.label_block(ax, style.SIM_STATUS, label_lines)
    # Below the efficiency curves, right of the rising chance-match line.
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), frameon=False,
              fontsize="x-small", ncol=2)  # fmt: skip
    axd.axhline(0.0, color="0.5", linewidth=1.0)
    axd.set_ylim(-0.5, 4.5)
    axd.set_ylabel("PVF $-$ AMVF [pt]", fontsize="small")
    axd.legend(loc="upper right", frameon=False, fontsize="x-small", ncol=2)
    axd.set_xlabel("Matching window [mm]")
    axd.set_xlim(float(w.min()), float(w.max()))
    return style.save(fig, out_dir, name)
