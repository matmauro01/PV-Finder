"""Plotting primitives for the extended-|eta| PU200 QA.

Split out of ``all_etas_data_qa.py`` to keep both files under the 500-line
limit. ``SampleData`` lives here because every getter and figure consumes it.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TIME_SENTINEL = -0.999  # real timing satisfies Time > TIME_SENTINEL
PT_TO_GEV = 1.0e-3  # ROOT pT is in MeV

# Sub-event geometry -- must mirror src/pv_finder/data/root_to_h5.py.
Z_MIN = -240.0
N_SUBEVENTS = 12
SUBEVENT_WIDTH = 40.0


class SampleData(NamedTuple):
    """Flattened arrays for one sample."""

    label: str
    color: str
    track: dict[str, np.ndarray]  # flat per-track
    event: dict[str, np.ndarray]  # per-event
    vertex: dict[str, np.ndarray]  # flat per-vertex
    sub_counts: np.ndarray  # tracks per 40 mm sub-event, flattened
    n_events: int


def _safe(a: np.ndarray | None) -> np.ndarray:
    if a is None or len(a) == 0:
        return np.empty(0, dtype=np.float32)
    return a[np.isfinite(a)]


class PlotSpec(NamedTuple):
    """One subplot: how to get values out of a SampleData and how to bin them."""

    title: str
    xlabel: str
    getter: Callable[[SampleData], np.ndarray]
    log_x: bool = False
    log_y: bool = False
    pct: tuple[float, float] = (0.2, 99.8)
    per_event: bool = False  # y axis = entries per event instead of unit area


def _track(name: str, scale: float = 1.0) -> Callable[[SampleData], np.ndarray]:
    return lambda s: _safe(s.track.get(name)) * scale


def _event(name: str) -> Callable[[SampleData], np.ndarray]:
    return lambda s: _safe(s.event.get(name))


def _vertex(name: str) -> Callable[[SampleData], np.ndarray]:
    return lambda s: _safe(s.vertex.get(name))


def _significance(s: SampleData) -> np.ndarray:
    d0, err = s.track.get("RecoTrack_d0"), s.track.get("RecoTrack_ErrD0")
    if d0 is None or err is None or len(d0) == 0:
        return np.empty(0, dtype=np.float32)
    ok = np.isfinite(d0) & np.isfinite(err) & (err > 0)
    return d0[ok] / err[ok]


def _timed(name: str, scale: float = 1.0) -> Callable[[SampleData], np.ndarray]:
    def g(s: SampleData) -> np.ndarray:
        t, v = s.track.get("RecoTrack_Time"), s.track.get(name)
        if t is None or v is None or len(t) == 0:
            return np.empty(0, dtype=np.float32)
        return _safe(v[t > TIME_SENTINEL]) * scale

    return g


def _bins(spec: PlotSpec, values: list[np.ndarray], nbins: int = 60) -> np.ndarray:
    nonempty = [v for v in values if len(v)]
    if not nonempty:
        return np.linspace(0, 1, nbins + 1)
    pooled = np.concatenate(nonempty)
    lo, hi = np.percentile(pooled, spec.pct)
    if spec.log_x:
        pos = pooled[pooled > 0]
        lo = max(lo, pos.min() if len(pos) else 1e-3)
        if hi <= lo:
            hi = lo * 10
        return np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, nbins + 1)


def _panel(ax: plt.Axes, spec: PlotSpec, samples: list[SampleData]) -> None:
    values = [spec.getter(s) for s in samples]
    edges = _bins(spec, values)
    ylabel = "entries / event" if spec.per_event else "normalised"
    for s, v in zip(samples, values):
        if len(v) == 0:
            continue
        counts, _ = np.histogram(v, bins=edges)
        if spec.per_event:
            y = counts / max(s.n_events, 1)
        else:
            width = np.diff(edges)
            y = counts / max(counts.sum(), 1) / width
        ax.step(edges[:-1], y, where="post", color=s.color, lw=1.5, label=s.label)
    ax.set_title(spec.title, fontsize=10)
    ax.set_xlabel(spec.xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    if spec.log_x:
        ax.set_xscale("log")
    if spec.log_y:
        ax.set_yscale("log")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)


def _grid(
    specs: list[PlotSpec], samples: list[SampleData], title: str, path: str, ncols: int
) -> None:
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, spec in zip(axes, specs):
        _panel(ax, spec, samples)
    for ax in axes[len(specs) :]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def _profile(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``y`` in bins of ``x``; NaN where a bin is empty."""
    idx = np.digitize(x, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(out)):
        sel = idx == b
        if sel.sum() > 20:
            out[b] = float(np.mean(y[sel]))
    return 0.5 * (edges[:-1] + edges[1:]), out


def plot_eta_profiles(samples: list[SampleData], path: str) -> None:
    """Track density and per-track resolutions as a function of |eta|."""
    edges = np.linspace(0, 4.2, 43)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for s in samples:
        eta = s.track.get("RecoTrack_eta")
        if eta is None or len(eta) == 0:
            continue
        aeta = np.abs(eta)
        counts, _ = np.histogram(aeta, bins=edges)
        axes[0, 0].step(
            centers,
            counts / s.n_events,
            where="mid",
            color=s.color,
            lw=1.5,
            label=s.label,
        )
        for ax, br, scale, lbl in [
            (axes[0, 1], "RecoTrack_ErrZ0", 1.0, "<ErrZ0> [mm]"),
            (axes[0, 2], "RecoTrack_ErrD0", 1.0, "<ErrD0> [mm]"),
            (axes[1, 0], "RecoTrack_pT", PT_TO_GEV, "<pT> [GeV]"),
            (axes[1, 1], "RecoTrack_z0", 1.0, "<|z0|> [mm]"),
        ]:
            v = s.track.get(br)
            if v is None or len(v) != len(aeta):
                continue
            vals = np.abs(v) * scale if br == "RecoTrack_z0" else v * scale
            ok = np.isfinite(vals)
            xc, prof = _profile(aeta[ok], vals[ok], edges)
            ax.plot(xc, prof, color=s.color, lw=1.5, label=s.label)
            ax.set_ylabel(lbl, fontsize=9)

        t = s.track.get("RecoTrack_Time")
        if t is not None and len(t) == len(aeta):
            num, _ = np.histogram(aeta[t > TIME_SENTINEL], bins=edges)
            den, _ = np.histogram(aeta, bins=edges)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            axes[1, 2].plot(centers, 100 * frac, color=s.color, lw=1.5, label=s.label)

    axes[0, 0].set_ylabel("tracks / event / bin", fontsize=9)
    axes[0, 0].set_title("Track density vs |eta|", fontsize=10)
    axes[0, 1].set_title("z0 uncertainty vs |eta|", fontsize=10)
    axes[0, 1].set_yscale("log")
    axes[0, 2].set_title("d0 uncertainty vs |eta|", fontsize=10)
    axes[0, 2].set_yscale("log")
    axes[1, 0].set_title("pT vs |eta|", fontsize=10)
    axes[1, 1].set_title("|z0| vs |eta|", fontsize=10)
    axes[1, 2].set_title("HGTD timing acceptance vs |eta|", fontsize=10)
    axes[1, 2].set_ylabel("tracks with a time [%]", fontsize=9)
    for ax in axes.ravel():
        ax.set_xlabel("|eta|", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Per-track quantities vs |eta|", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_subevent_occupancy(
    samples: list[SampleData], path: str, max_tracks: int
) -> None:
    """Tracks per 40 mm sub-event -- the quantity ``--max-tracks-per-sub`` must cover."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    hi = max(int(s.sub_counts.max()) for s in samples if len(s.sub_counts))
    edges = np.linspace(0, max(hi * 1.05, max_tracks * 1.05), 80)

    for s in samples:
        c = s.sub_counts
        if len(c) == 0:
            continue
        axes[0].hist(
            c,
            bins=edges,
            histtype="step",
            color=s.color,
            lw=1.5,
            density=True,
            label=f"{s.label} (max {c.max()})",
        )
        per_event_max = c.reshape(-1, N_SUBEVENTS).max(axis=1)
        axes[1].hist(
            per_event_max,
            bins=edges,
            histtype="step",
            color=s.color,
            lw=1.5,
            density=True,
            label=f"{s.label} (max {per_event_max.max()})",
        )
        # Occupancy vs sub-event index (z position of the 40 mm window).
        mean_per_sub = c.reshape(-1, N_SUBEVENTS).mean(axis=0)
        z_centers = Z_MIN + SUBEVENT_WIDTH * (np.arange(N_SUBEVENTS) + 0.5)
        axes[2].plot(
            z_centers, mean_per_sub, "o-", color=s.color, lw=1.5, label=s.label
        )

    for ax, title, xlabel in [
        (axes[0], "Tracks per 40 mm sub-event", "tracks / sub-event"),
        (axes[1], "Busiest sub-event per event", "max tracks / sub-event"),
    ]:
        ax.axvline(
            max_tracks,
            color="crimson",
            ls="--",
            lw=1.6,
            label=f"--max-tracks-per-sub = {max_tracks}",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("normalised", fontsize=9)
        ax.set_yscale("log")
    axes[2].set_title("Mean occupancy vs sub-event z", fontsize=10)
    axes[2].set_xlabel("sub-event center z [mm]", fontsize=9)
    axes[2].set_ylabel("mean tracks / sub-event", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Sub-event occupancy -- drives the padded tracks-tensor width", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")
