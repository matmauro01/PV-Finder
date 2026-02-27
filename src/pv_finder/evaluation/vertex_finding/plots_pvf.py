"""plots_pvf.py — Plotting helpers for run_eval_pvf.py.

Three functions, each saves one PNG to output_dir:
  plot_resolution   — pairwise Δz histogram + sigmoid fit
  plot_performance  — reco category fractions + efficiency vs pileup
  plot_stats        — avg count/event per category vs pileup (all events)
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_resolution(
    dz_arr: np.ndarray,
    sigma_vtx_vtx: float,
    popt,
    sigmoid_fit,
    mode_label: str,
    output_dir: Path,
) -> None:
    """Pairwise Δz histogram with fitted sigmoid curve."""
    bins_res = np.linspace(-6.0, 6.0, 61)
    bin_ctrs = 0.5 * (bins_res[:-1] + bins_res[1:])
    counts, _ = np.histogram(dz_arr, bins=bins_res)
    bw = bins_res[1] - bins_res[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        bin_ctrs,
        counts,
        width=bw * 0.9,
        color="steelblue",
        alpha=0.8,
        label="Reconstructed PV pairs",
    )
    if popt is not None:
        xf = np.linspace(-6, 6, 400)
        ax.plot(
            xf,
            sigmoid_fit(xf, *popt),
            "r-",
            lw=2,
            label=f"Sigmoid fit  σ = {sigma_vtx_vtx:.3f} mm",
        )
        ax.axvline(
            sigma_vtx_vtx,
            color="red",
            ls="--",
            alpha=0.7,
            label=f"±σ = ±{sigma_vtx_vtx:.3f} mm",
        )
        ax.axvline(-sigma_vtx_vtx, color="red", ls="--", alpha=0.7)
    ax.set_xlabel("Δz between reconstructed PV pairs (mm)")
    ax.set_ylabel("Counts")
    ax.set_title(f"PVF Resolution — MC Test Set\n({mode_label})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "resolution_plot.png", dpi=150)
    plt.close()


def plot_performance(
    per_event: list,
    overall_eff: float,
    root_z_available: bool,
    mode_label: str,
    output_dir: Path,
) -> None:
    """Reco category fractions and efficiency vs pileup proxy."""
    pu: dict = defaultdict(lambda: defaultdict(list))
    for r in per_event:
        p = round(r["mu"]) if root_z_available else r["n_truth"]
        nd = r["n_pred"] or 1
        for k in ("clean", "merged", "split", "fake"):
            pu[p][k].append(r[k] / nd)
        pu[p]["eff"].append(r["eff"])

    pu_vals = sorted(pu.keys())
    _m = lambda lst: float(np.mean(lst)) if lst else 0.0  # noqa: E731

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax = axes[0]
    for key, color, marker, label in [
        ("clean", "b", "o", "Clean"),
        ("merged", "g", "s", "Merged"),
        ("split", "r", "^", "Split"),
        ("fake", "k", "v", "Fake"),
    ]:
        ax.plot(
            pu_vals,
            [_m(pu[p][key]) for p in pu_vals],
            f"{color}-{marker}",
            ms=4,
            label=label,
        )
    ax.set_ylabel("Fraction of reconstructed PVs")
    ax.set_title(f"PVF Performance — Reco PV Categories\n({mode_label})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pu_vals, [_m(pu[p]["eff"]) for p in pu_vals], "b-o", ms=4)
    ax.axhline(
        overall_eff,
        color="red",
        ls="--",
        alpha=0.7,
        label=f"Overall = {overall_eff:.3f}",
    )
    xlabel = "ActualNumOfInt (μ)" if root_z_available else "N truth PVs/evt"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Efficiency (matched / truth)")
    ax.set_title("PVF Efficiency")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_plot.png", dpi=150)
    plt.close()


def plot_stats(
    per_event: list,
    root_z_available: bool,
    mode_label: str,
    output_dir: Path,
) -> None:
    """Avg count/event for clean/merged/split/fake vs pileup (all events).

    Adjacent-mu bins, mean ± SEM lines per category (mattia_finder style).
    X-axis: ActualNumOfInt (rounded) when ROOT available, else N truth PVs.
    """
    sep: dict = defaultdict(
        lambda: {k: [] for k in ("clean", "merged", "split", "fake")}
    )
    for r in per_event:
        p = round(r["mu"]) if root_z_available else r["n_truth"]
        for k in ("clean", "merged", "split", "fake"):
            sep[p][k].append(r[k])

    mu_vals = sorted(sep.keys())
    cats = ("clean", "merged", "split", "fake")
    mu_binned: list = []
    means: dict = {k: [] for k in cats}
    errs: dict = {k: [] for k in cats}
    for i in range(0, len(mu_vals) - 1, 2):
        mu_binned.append(0.5 * (mu_vals[i] + mu_vals[i + 1]))
        for k in cats:
            data = sep[mu_vals[i]][k] + sep[mu_vals[i + 1]][k]
            means[k].append(float(np.mean(data)) if data else 0.0)
            errs[k].append(
                float(np.std(data) / np.sqrt(len(data))) if len(data) > 1 else 0.0
            )

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color, marker in [
        ("clean", "b", "o"),
        ("merged", "g", "s"),
        ("split", "r", "^"),
        ("fake", "k", "v"),
    ]:
        ax.errorbar(
            mu_binned,
            means[key],
            yerr=errs[key],
            fmt=f"{color}-{marker}",
            ms=4,
            capsize=3,
            label=key.capitalize(),
        )
    xlabel = "ActualNumOfInt (μ, rounded)" if root_z_available else "N truth PVs/evt"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Avg count / event")
    ax.set_title(f"PVF Reco PV Categories vs Pileup\n({mode_label})")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "stats_histogram.png", dpi=150)
    plt.close()
