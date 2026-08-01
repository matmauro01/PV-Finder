"""Category-count bar charts for the vertex-finding evaluation.

Split out of ``plots_pvf.py`` to keep both modules under the 500-line limit.
``plots_pvf`` re-exports :func:`plot_category_counts`, so existing
``from plots_pvf import plot_category_counts`` call sites keep working.
"""

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless — prevents X11/display crashes over SSH + tmux
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _short_label(mode_label: str, max_len: int = 55) -> str:
    """Truncate long mode_label for plot titles."""
    if len(mode_label) <= max_len:
        return mode_label
    return mode_label[: max_len - 3] + "..."


def plot_category_counts(
    per_event: list,
    mode_label: str,
    output_dir: Path,
    title: str = "",
    eval_label: str = "",
    mu_min: int = 55,
    mu_max: int = 65,
    all_events: bool = False,
    truth_pvs_per_evt: Optional[float] = None,
    outfile: str = "category_counts_hist.png",
) -> None:
    """Mean per-event reco counts as a 5-bar chart in the high-pileup window.

    Bars (left → right): Total, Clean, Merged, Split, Fake. Total is the
    sum of the four categories (equivalent to `n_pred`).

    Events are filtered to `mu ∈ [mu_min, mu_max]` when pileup is available,
    otherwise all events are used. The pileup window is written into the
    default title. Error bars are SEM across events in the window.
    """
    if not per_event:
        return

    if all_events:
        filt = list(per_event)
    else:
        events_with_mu = [r for r in per_event if r.get("mu") is not None]
        if not events_with_mu:
            # Pileup unavailable — refuse to draw a filtered plot.
            print(
                f"  [plot_category_counts] skipped: no per-event mu "
                f"(need --root-truth); cannot filter to [{mu_min}, {mu_max}]"
            )
            return
        filt = [r for r in events_with_mu if mu_min <= round(r["mu"]) <= mu_max]
        if not filt:
            print(
                f"  [plot_category_counts] skipped: 0 events in μ ∈ [{mu_min}, {mu_max}]"
            )
            return

    labels = ("Total", "Clean", "Merged", "Split", "Fake")
    n = len(filt)

    def _stats(keys):
        m, s = [], []
        for k in keys:
            v = np.asarray([r.get(k, 0) or 0 for r in filt], dtype=float)
            m.append(float(v.mean()))
            s.append(float(v.std() / np.sqrt(n)) if n > 1 else 0.0)
        return np.array(m), np.array(s)

    pvf_means, pvf_sems = _stats(("n_pred", "clean", "merged", "split", "fake"))
    has_amvf = any(r.get("amvf_clean") is not None for r in filt)
    if has_amvf:
        amvf_means, amvf_sems = _stats(
            ("n_amvf", "amvf_clean", "amvf_merged", "amvf_split", "amvf_fake")
        )

    vivid = {
        "total": "#2C3E50", "clean": "#3498DB", "merged": "#2ECC71",
        "split": "#E74C3C", "fake": "#F39C12",
    }  # fmt: skip
    colors = [vivid[k] for k in ("total", "clean", "merged", "split", "fake")]
    edges = ["#1a2530", "#1f6396", "#1b8449", "#962a1f", "#9c600a"]

    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafbfc")
    x = np.arange(len(labels))
    if has_amvf:
        w = 0.36
        bars = ax.bar(x - w / 2, pvf_means, width=w, yerr=pvf_sems,
            color=colors, edgecolor=edges, linewidth=1.5, alpha=0.95,
            label="PV-Finder",
            error_kw=dict(ecolor="#222", elinewidth=1.2, capsize=5))  # fmt: skip
        bars_a = ax.bar(x + w / 2, amvf_means, width=w, yerr=amvf_sems,
            color=colors, edgecolor=edges, linewidth=1.5, alpha=0.55, hatch="///",
            label="AMVF",
            error_kw=dict(ecolor="#222", elinewidth=1.2, capsize=5))  # fmt: skip
        ax.legend(fontsize=11, loc="upper left")
        for b, m in zip(bars, pvf_means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#1a1a1a")  # fmt: skip
        for b, m in zip(bars_a, amvf_means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:.1f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#555")  # fmt: skip
        ymax = float(np.maximum(pvf_means + pvf_sems, amvf_means + amvf_sems).max())
    else:
        bars = ax.bar(x, pvf_means, width=0.72, yerr=pvf_sems,
            color=colors, edgecolor=edges, linewidth=1.8, alpha=0.95,
            error_kw=dict(ecolor="#222", elinewidth=1.5, capsize=7))  # fmt: skip
        pad = 0.025 * float((pvf_means + pvf_sems).max())
        for b, m, s in zip(bars, pvf_means, pvf_sems):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + s + pad,
                f"{m:.2f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#1a1a1a")  # fmt: skip
        ymax = float((pvf_means + pvf_sems).max())

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight="semibold")
    ax.set_ylabel("Mean reconstructed PVs / event", fontsize=14, fontweight="semibold")
    mu_desc = "all events" if all_events else f"μ ∈ [{mu_min}, {mu_max}]"
    ax.set_title(
        title or f"Per-event reco counts, {mu_desc}  —  {_short_label(mode_label)}",
        fontsize=11, fontweight="bold", pad=10,
    )  # fmt: skip
    ax.set_ylim(bottom=0, top=ymax * 1.28)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.35, ls="--", lw=0.8, color="#666")
    ax.tick_params(labelsize=12)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666")
        ax.spines[spine].set_linewidth(1.2)

    # Derive the truth rate from the events actually drawn. Taking it as a
    # parameter let a mu-windowed value be printed next to all-event bars.
    if truth_pvs_per_evt is None:
        nt = [r["n_truth"] for r in filt if r.get("n_truth") is not None]
        truth_pvs_per_evt = float(np.mean(nt)) if nt else None
    info = f"{n} events\n{mu_desc}"
    if truth_pvs_per_evt is not None:
        info += f"\ntruth PVs/evt = {truth_pvs_per_evt:.1f}"
    if eval_label:
        info = f"{eval_label}\n{info}"
    ax.text(
        0.985, 0.97, info, transform=ax.transAxes, fontsize=9,
        va="top", ha="right", family="monospace", color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#2C3E50", lw=1.1,
                  alpha=0.92),
    )  # fmt: skip

    plt.tight_layout()
    plt.savefig(output_dir / outfile, dpi=160,
                bbox_inches="tight", facecolor="white")  # fmt: skip
    plt.close()


def plot_category_counts_both(
    per_event: list,
    mode_label: str,
    output_dir: Path,
    mu_min: int,
    mu_max: int,
    **kwargs,
) -> None:
    """Draw the pile-up-selected chart, and the all-events one when they differ.

    On a fixed-mu=200 sample every event is in the window and one chart says
    everything. On a flat-mu sample (the held-out r16443/r16638 tags) the two
    answer different questions -- "how does it do at PU200" versus "how does it
    do averaged over all pile-ups" -- and quoting the second next to a summary
    table computed on the first is how a factor-1.75 discrepancy went unnoticed.
    """
    n_all = len(per_event)
    n_win = sum(
        1
        for r in per_event
        if r.get("mu") is not None and mu_min <= round(r["mu"]) <= mu_max
    )
    plot_category_counts(per_event, mode_label, output_dir, mu_min=mu_min,
                         mu_max=mu_max, all_events=False, **kwargs)  # fmt: skip
    if n_win != n_all:
        plot_category_counts(per_event, mode_label, output_dir, mu_min=mu_min,
                             mu_max=mu_max, all_events=True,
                             outfile="category_counts_hist_allmu.png",
                             **kwargs)  # fmt: skip
