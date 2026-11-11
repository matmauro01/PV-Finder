"""Reconstructed-vertex categories (Figures 3 and 5).

Categories are positional, from the production matcher
(``amvf_fair_comparison.matching.match_vertices``, count-for-count identical to
``efficiency_res_optimized_atlas.compare_res_reco``): greedy closest-first
one-to-one assignment within a fixed window, then a reco vertex is *clean* if
it holds one truth vertex, *merged* if it also claims unassigned truth
vertices in its window, *split* if its window holds only truth vertices
already taken, and *fake* if its window holds none.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pv_finder.diagnostics.amvf_fair_comparison.matching import match_vertices
from pv_finder.diagnostics.paper_figures import style
from pv_finder.diagnostics.paper_figures.sources import Sample

COUNT_KEYS = ("n_truth", "all", "clean", "merged", "split", "fake")


def per_event_counts(
    truth: Sequence[np.ndarray], reco: Sequence[np.ndarray], window_mm: float
) -> dict[str, np.ndarray]:
    """Category counts per event for one algorithm at one window."""
    out = {k: np.zeros(len(reco), dtype=np.int64) for k in COUNT_KEYS}
    out["truth_found"] = np.zeros(len(reco), dtype=np.int64)
    for i, (t, r) in enumerate(zip(truth, reco)):
        m = match_vertices(np.asarray(t), np.asarray(r), window_mm)
        out["n_truth"][i] = m.n_truth
        out["all"][i] = m.n_reco
        out["clean"][i] = m.reco_clean
        out["merged"][i] = m.reco_merged
        out["split"][i] = m.reco_split
        out["fake"][i] = m.reco_fake
        out["truth_found"][i] = m.n_truth_matched
    return out


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    """Mean per event and its standard error (ddof = 1)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def summarize(
    counts: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, list[float]]:
    """``{key: [mean, sem]}`` over the selected events, plus pooled efficiency."""
    out = {k: list(mean_sem(counts[k][mask])) for k in COUNT_KEYS}
    out["efficiency"] = [float(counts["truth_found"][mask].sum()
                               / counts["n_truth"][mask].sum()), float("nan")]  # fmt: skip
    out["n_events"] = [float(mask.sum()), 0.0]
    return out


def plot_category_bars(
    pvf: dict[str, list[float]],
    amvf: dict[str, list[float]],
    label_lines: Sequence[str],
    status: str,
    out_dir: Path,
    name: str,
    reference_label: str = "Truth interactions",
) -> list[Path]:
    """Left panels: mean number of vertices per event by type, PVF vs AMVF."""
    fig, ax = style.new_figure(8.0, 6.0)
    keys = ("all", "clean", "merged", "split", "fake")
    x = np.arange(len(keys))
    width = 0.38
    for off, summary, label, color, hatch, fill in (
        (-width / 2, pvf, "PV-Finder", style.PVF_COLOR, None, True),
        (+width / 2, amvf, "AMVF", style.AMVF_COLOR, "////", False),
    ):
        means = [summary[k][0] for k in keys]
        errs = [summary[k][1] for k in keys]
        ax.bar(x + off, means, width, yerr=errs, label=label,
               color=color if fill else "none", edgecolor=color, hatch=hatch,
               linewidth=1.5, alpha=0.85 if fill else 1.0,
               error_kw={"ecolor": "black", "elinewidth": 1.2, "capsize": 3})  # fmt: skip
    # Reference: mean number of truth vertices passing the selection.
    t_mean = pvf["n_truth"][0]
    ax.hlines(t_mean, -0.5, 0.5, colors="0.35", linestyles="--", linewidth=1.5,
              label=f"{reference_label} ({t_mean:.1f})")  # fmt: skip
    ax.set_xticks(x)
    ax.set_xticklabels([style.CATEGORY_STYLE[k]["label"].replace(" reconstructed", "")
                        for k in keys])  # fmt: skip
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel("Vertices / event")
    ax.set_ylim(0, t_mean * 1.6)
    style.label_block(ax, status, label_lines)
    # Below the sample text, above the low Merged/Split/Fake bars.
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.74), frameon=False,
              fontsize="small")  # fmt: skip
    return style.save(fig, out_dir, name)


def _binned(mu: np.ndarray, values: np.ndarray, edges: np.ndarray, min_events: int):
    """Mean mu, mean value and SEM in each mu bin with enough events."""
    xs, ys, es = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mu >= lo) & (mu < hi)
        if sel.sum() < min_events:
            continue
        m, e = mean_sem(values[sel])
        xs.append(float(mu[sel].mean()))
        ys.append(m)
        es.append(e)
    return np.array(xs), np.array(ys), np.array(es)


def plot_categories_vs_mu(
    sample: Sample,
    counts: dict[str, np.ndarray],
    edges: np.ndarray,
    label_lines: Sequence[str],
    status: str,
    out_dir: Path,
    name: str,
    min_events: int = 10,
    show_interactions: bool = True,
) -> list[Path]:
    """Right panels: PVF vertices per event vs mu, split by type (cf. IDTR-2021-01 Fig. 25a)."""
    fig, ax = style.new_figure(8.0, 6.0)
    mu = sample.mu
    for key in ("all", *style.CATEGORIES):
        st = style.CATEGORY_STYLE[key]
        x, y, e = _binned(mu, counts[key].astype(float), edges, min_events)
        ax.errorbar(x, y, yerr=e, fmt=st["marker"], color=st["color"],
                    markersize=5, label=st["label"], elinewidth=1.2)  # fmt: skip
    x, y, _ = _binned(mu, counts["n_truth"].astype(float), edges, min_events)
    ax.plot(x, y, "--", color="0.35", linewidth=1.5,
            label="Interactions with $\\geq 2$ selected tracks")  # fmt: skip
    if show_interactions:
        grid = np.array([edges[0], edges[-1]], dtype=float)
        ax.plot(grid, grid, ":", color="0.55", linewidth=1.5, label="$y = \\mu$")
    ax.set_xlim(edges[0], edges[-1])
    ymax = max(edges[-1] if show_interactions else 0.0, float(np.max(y)))
    ax.set_ylim(0, ymax * 1.6)
    ax.set_xlabel("Actual number of interactions per crossing, $\\mu$")
    ax.set_ylabel("Vertices / event")
    style.label_block(ax, status, label_lines)
    top = 0.95 - 0.055 * (len(label_lines) + 1) - 0.03
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, top), ncol=2, frameon=False,
              fontsize="x-small", columnspacing=1.0)  # fmt: skip
    return style.save(fig, out_dir, name)
