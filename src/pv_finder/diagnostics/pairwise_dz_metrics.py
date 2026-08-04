"""Pairwise-Delta-z observables shared by the bump studies.

The distribution of ``|z_i - z_j|`` over reconstructed vertex pairs is flat at
large separation (a plateau set by combinatorics) and suppressed inside the
algorithm's resolution. A satellite population -- surplus peaks emitted at a
fixed distance from a real vertex -- shows up as an excess sitting just outside
the dip. Two numbers summarise it:

``band_excess_pct``
    mean relative excess over the plateau in |dz| = 0.3-0.7 mm. Not multiplicity
    invariant: satellite pairs grow like n while total pairs grow like n^2, so a
    configuration that simply finds more peaks reports a smaller excess for the
    same per-peak pathology. Quote it, but do not compare it across samples with
    different vertex densities without also quoting the next one.

``satellites_per_peak``
    plateau-subtracted surplus companions in the 0.25-2.0 mm shell divided by
    the number of reconstructed peaks. Multiplicity invariant and truth free.

The binning is the positive half of the 240-bin +-6 mm grid the sigma_vtx-vtx
fit uses; the symmetrised and folded histograms have identical bin contents on
the positive side, so these numbers are directly comparable with
``pairwise_dz_comparison.py``.

Errors are bootstrap over events, never sqrt(N): pairs inside one event are
correlated. ``paired_contrast`` resamples the event list once and applies it to
both cells, which is what you want for a difference between two measurements on
the same events.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HALF_RANGE_MM = 6.0
N_HALF_BINS = 120  # 0.05 mm/bin, the positive half of the eval's 240-bin grid
BAND_LO, BAND_HI = 0.3, 0.7  # mm, the headline band
SHELL_LO, SHELL_HI = 0.25, 2.0  # mm, the satellite shell
PLATEAU_MIN = 3.0  # mm, where the distribution is flat


def pair_hists(z_per_event: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-event |dz| histograms on the positive half-grid, and peak counts."""
    edges = np.linspace(0.0, HALF_RANGE_MM, N_HALF_BINS + 1)
    hists = np.zeros((len(z_per_event), N_HALF_BINS), dtype=np.float64)
    npeaks = np.zeros(len(z_per_event), dtype=np.float64)
    for i, z in enumerate(z_per_event):
        z = np.asarray(z, dtype=np.float64)
        z = z[np.isfinite(z)]
        npeaks[i] = len(z)
        if len(z) < 2:
            continue
        d = np.abs(z[:, None] - z[None, :])[np.triu_indices(len(z), 1)]
        hists[i], _ = np.histogram(d, bins=edges)
    return hists, npeaks


def _centres() -> np.ndarray:
    edges = np.linspace(0.0, HALF_RANGE_MM, N_HALF_BINS + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def observables(counts: np.ndarray, n_peaks: float) -> dict:
    """Band excess, peak excess and satellites-per-peak from a summed profile."""
    c = _centres()
    base = float(np.median(counts[c > PLATEAU_MIN]))
    rel = (counts - base) / base
    band = (c >= BAND_LO) & (c <= BAND_HI)
    shell = (c >= SHELL_LO) & (c <= SHELL_HI)
    search = (c > 0.25) & (c < 2.5)
    i = int(np.argmax(np.where(search, rel, -np.inf)))
    dip = c < 0.1
    return {
        "plateau": base,
        "band_excess_pct": float(100 * rel[band].mean()),
        "peak_excess_pct": float(100 * rel[i]),
        "peak_excess_at_mm": float(c[i]),
        "dip_depth_pct": float(100 * rel[dip].mean()),
        "shell_excess_pairs": float((counts[shell] - base).sum()),
        "satellites_per_peak": float((counts[shell] - base).sum() / max(n_peaks, 1.0)),
    }


def bootstrap(
    hists: np.ndarray, npeaks: np.ndarray, n_boot: int, seed: int = 12345
) -> dict:
    """Bootstrap over events; returns mean and std for each observable."""
    rng = np.random.default_rng(seed)
    n = len(hists)
    keys = ("band_excess_pct", "peak_excess_pct", "satellites_per_peak", "plateau")
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
        o = observables(w @ hists, float(w @ npeaks))
        for k in keys:
            acc[k].append(o[k])
    return {f"{k}_err": float(np.std(acc[k], ddof=1)) for k in keys}


def paired_contrast(
    cache_a: tuple[np.ndarray, np.ndarray],
    cache_b: tuple[np.ndarray, np.ndarray],
    n_boot: int,
    seed: int = 4242,
) -> dict:
    """Difference b - a with a **paired** bootstrap.

    Every cell is measured on the same events in the same order, so the
    difference between two cells is far better determined than the quadrature
    sum of their individual errors suggests. Each replica resamples the event
    list once and applies that same resampling to both cells.
    """
    (ha, na), (hb, nb) = cache_a, cache_b
    if len(ha) != len(hb):
        raise ValueError("paired contrast needs the same events in both cells")
    rng = np.random.default_rng(seed)
    n = len(ha)
    keys = ("band_excess_pct", "satellites_per_peak", "peaks_per_event")
    acc: dict[str, list[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float64)
        oa = observables(w @ ha, float(w @ na))
        ob = observables(w @ hb, float(w @ nb))
        oa["peaks_per_event"] = float(w @ na) / n
        ob["peaks_per_event"] = float(w @ nb) / n
        for k in keys:
            acc[k].append(ob[k] - oa[k])
    fa = observables(ha.sum(0), float(na.sum()))
    fb = observables(hb.sum(0), float(nb.sum()))
    fa["peaks_per_event"] = float(na.mean())
    fb["peaks_per_event"] = float(nb.mean())
    return {
        k: {"delta": float(fb[k] - fa[k]), "err": float(np.std(acc[k], ddof=1))}
        for k in keys
    }


def measure(
    z_per_event: list[np.ndarray], n_boot: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Full observable set with bootstrap errors, plus the per-event cache."""
    hists, npeaks = pair_hists(z_per_event)
    out = observables(hists.sum(0), float(npeaks.sum()))
    out["peaks_per_event"] = float(npeaks.mean())
    out["n_events"] = int(len(z_per_event))
    if n_boot > 0:
        out.update(bootstrap(hists, npeaks, n_boot))
    return out, hists, npeaks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def plot_cells(profiles: dict[str, np.ndarray], out_path: Path, title: str) -> None:
    """Relative-deviation curves for every cell on one axis."""
    c = _centres()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")
    for ax, (lo, hi, ylo, yhi) in zip(axes, [(0, 3.0, -105, 30), (0.2, 1.6, -40, 30)]):
        for k, (name, counts) in enumerate(profiles.items()):
            base = float(np.median(counts[c > PLATEAU_MIN]))
            rel = 100 * (counts - base) / base
            m = (c >= lo) & (c <= hi)
            ax.step(c[m], rel[m], where="mid", lw=1.5, color=cmap(k % 10), label=name)
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvspan(BAND_LO, BAND_HI, color="0.85", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel(r"$|\Delta z|$ between reconstructed vertices [mm]")
        ax.set_ylabel(r"deviation from large-$|\Delta z|$ plateau [%]")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_table(summary: dict) -> None:
    """One line per cell."""
    hdr = (
        f"{'cell':<14}{'peaks/evt':>10}{'plateau':>10}"
        f"{'band 0.3-0.7 [%]':>20}{'peak exc [%]':>16}{'at [mm]':>9}"
        f"{'sat/peak':>16}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, s in summary.items():
        print(
            f"{name:<14}{s['peaks_per_event']:>10.2f}{s['plateau']:>10.0f}"
            f"{s['band_excess_pct']:>13.2f} +- {s.get('band_excess_pct_err', 0):<4.2f}"
            f"{s['peak_excess_pct']:>10.2f} +- {s.get('peak_excess_pct_err', 0):<3.1f}"
            f"{s['peak_excess_at_mm']:>9.2f}"
            f"{s['satellites_per_peak']:>11.4f} +- "
            f"{s.get('satellites_per_peak_err', 0):<4.4f}"
        )
