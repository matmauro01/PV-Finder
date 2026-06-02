"""Basic tracking-parameter QA for the HL-LHC PU200 *with-timing* ntuples.

Produces overlaid distribution plots so we can confirm the new
``data/run4/PU200_withTiming/`` samples are sane before training on them.

Each overlaid curve is one sample:
  * the four DSID-601229 reco tags (r16438 / r16443 / r16633 / r16638),
  * the pooled DSID-601237 sample (ttbar all-hadronic, 6 part files),
  * the old no-timing PU200 file as a known-good reference.

The new ``RecoTrack_Time`` / ``RecoTrack_TimeResolution`` branches use ``-1`` as
a "no timing" sentinel (only ~4% of tracks, all at the HGTD forward edge), so
those values are masked out before histogramming.

Run from the repo root with the venv active::

    python -u src/pv_finder/diagnostics/timing_data_qa.py \
        --events-per-file 5000 --output-dir outputs/06_02_2026_output/timing_data_qa

Outputs three PNGs (track kinematics, timing, event level) plus a JSON summary
of per-sample means / stds.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

TREE = "PVFinderData"
TIME_SENTINEL = -0.999  # real timing satisfies Time > TIME_SENTINEL
PT_TO_GEV = 1.0e-3  # ROOT pT is in MeV

DATA_ROOT = "data/run4"
TIMING_DIR = f"{DATA_ROOT}/PU200_withTiming"
OLD_REF = f"{DATA_ROOT}/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root"

# Track-level branches read from every file.
TRACK_BRANCHES = [
    "RecoTrack_d0",
    "RecoTrack_z0",
    "RecoTrack_pT",
    "RecoTrack_eta",
    "RecoTrack_phi",
    "RecoTrack_theta",
    "RecoTrack_ErrD0",
    "RecoTrack_ErrZ0",
]
TIMING_BRANCHES = ["RecoTrack_Time", "RecoTrack_TimeResolution"]
EVENT_BRANCHES = [
    "NumRecoTrks",
    "NumRecoVtx",
    "NumTruthVtx",
    "ActualNumOfInt",
    "BeamPosZ",
]
VERTEX_BRANCHES = ["TruthVertex_z", "TruthVertex_nTracks"]


class Sample(NamedTuple):
    """A named group of ROOT files plotted as one overlaid curve."""

    label: str
    files: list[str]
    color: str


class SampleData(NamedTuple):
    """Flattened arrays for one sample."""

    track: dict[str, np.ndarray]  # flat per-track arrays
    event: dict[str, np.ndarray]  # per-event arrays
    vertex: dict[str, np.ndarray]  # flat per-truth-vertex arrays
    n_events: int


def build_samples() -> list[Sample]:
    """Define the overlaid samples (one curve each)."""
    samples: list[Sample] = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c8c8c"]
    for rtag, color in zip(["r16438", "r16443", "r16633", "r16638"], colors):
        f = f"{TIMING_DIR}/ATLAS_PVFinderData_601229_e8481_s4494_{rtag}_PU200.root"
        samples.append(Sample(f"601229 {rtag}", [f], color))
    allhad = sorted(
        glob.glob(f"{TIMING_DIR}/**/ATLAS_PVFinderData_601237_*.root", recursive=True)
    )
    samples.append(Sample("601237 (all-had)", allhad, colors[4]))
    if os.path.exists(OLD_REF):
        samples.append(Sample("old PU200 (no timing)", [OLD_REF], colors[5]))
    return samples


def _flatten(jagged: np.ndarray) -> np.ndarray:
    """Flatten an uproot object-array of per-event arrays into one float array."""
    if len(jagged) == 0:
        return np.empty(0, dtype=np.float32)
    return np.concatenate([np.asarray(x, dtype=np.float32) for x in jagged])


def load_sample(
    sample: Sample, events_per_file: int, mu_min: float = -1.0
) -> SampleData:
    """Read up to ``events_per_file`` events from each file in the sample.

    If ``mu_min >= 0`` only events with ``ActualNumOfInt > mu_min`` are kept
    (used to drop the PU0 half of the mixed r16443 / r16638 reco tags).
    """
    track: dict[str, list[np.ndarray]] = {
        b: [] for b in TRACK_BRANCHES + TIMING_BRANCHES
    }
    event: dict[str, list[np.ndarray]] = {b: [] for b in EVENT_BRANCHES}
    vertex: dict[str, list[np.ndarray]] = {b: [] for b in VERTEX_BRANCHES}
    n_events = 0

    for path in sample.files:
        tree = uproot.open(path)[TREE]
        avail = set(tree.keys())
        stop = min(events_per_file, tree.num_entries)
        want = sorted(
            {b for b in track if b in avail}
            | {b for b in event if b in avail}
            | {b for b in vertex if b in avail}
            | {"ActualNumOfInt"}
        )
        arrs = tree.arrays(want, entry_stop=stop, library="np")
        keep = np.ones(stop, dtype=bool)
        if mu_min >= 0 and "ActualNumOfInt" in arrs:
            keep = np.asarray(arrs["ActualNumOfInt"], dtype=float) > mu_min
        n_events += int(keep.sum())
        for b in track:
            if b in arrs:
                track[b].append(_flatten(arrs[b][keep]))
        for b in event:
            if b in arrs:
                event[b].append(np.asarray(arrs[b], dtype=np.float32)[keep])
        for b in vertex:
            if b in arrs:
                vertex[b].append(_flatten(arrs[b][keep]))

    def _cat(d: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
        return {k: np.concatenate(v) for k, v in d.items() if v}

    print(
        f"  loaded {sample.label:24s}: {n_events:6d} events, "
        f"{sum(len(a) for a in track['RecoTrack_d0']):>10d} tracks"
    )
    return SampleData(_cat(track), _cat(event), _cat(vertex), n_events)


class PlotSpec(NamedTuple):
    """One subplot: how to extract values from a SampleData and how to bin them."""

    title: str
    xlabel: str
    getter: Callable[[SampleData], np.ndarray]
    log_x: bool = False
    log_y: bool = False
    pct: tuple = (0.5, 99.5)  # percentile range for auto x-limits


def _safe(a: np.ndarray | None) -> np.ndarray:
    if a is None or len(a) == 0:
        return np.empty(0, dtype=np.float32)
    return a[np.isfinite(a)]


# ----- value getters -------------------------------------------------------


def _track(name: str, scale: float = 1.0) -> Callable[[SampleData], np.ndarray]:
    return lambda s: _safe(s.track.get(name)) * scale


def _significance(s: SampleData) -> np.ndarray:
    d0 = s.track.get("RecoTrack_d0")
    err = s.track.get("RecoTrack_ErrD0")
    if d0 is None or err is None:
        return np.empty(0, dtype=np.float32)
    ok = np.isfinite(d0) & np.isfinite(err) & (err > 0)
    return d0[ok] / err[ok]


def _real_time(name: str, scale: float = 1.0) -> Callable[[SampleData], np.ndarray]:
    def g(s: SampleData) -> np.ndarray:
        t = s.track.get("RecoTrack_Time")
        v = s.track.get(name)
        if t is None or v is None:
            return np.empty(0, dtype=np.float32)
        mask = t > TIME_SENTINEL
        return _safe(v[mask]) * scale

    return g


def _event(name: str) -> Callable[[SampleData], np.ndarray]:
    return lambda s: _safe(s.event.get(name))


def _vertex(name: str) -> Callable[[SampleData], np.ndarray]:
    return lambda s: _safe(s.vertex.get(name))


TRACK_PLOTS = [
    PlotSpec("d0", "d0 [mm]", _track("RecoTrack_d0")),
    PlotSpec("z0", "z0 [mm]", _track("RecoTrack_z0")),
    PlotSpec(
        "pT", "pT [GeV]", _track("RecoTrack_pT", PT_TO_GEV), log_x=True, log_y=True
    ),
    PlotSpec("eta", "eta", _track("RecoTrack_eta")),
    PlotSpec("phi", "phi [rad]", _track("RecoTrack_phi")),
    PlotSpec("theta", "theta [rad]", _track("RecoTrack_theta")),
    PlotSpec("ErrD0", "ErrD0 [mm]", _track("RecoTrack_ErrD0")),
    PlotSpec("ErrZ0", "ErrZ0 [mm]", _track("RecoTrack_ErrZ0")),
    PlotSpec("d0 significance", "d0 / ErrD0", _significance),
]

TIMING_PLOTS = [
    PlotSpec("Track time (timed tracks)", "Time [ns]", _real_time("RecoTrack_Time")),
    PlotSpec(
        "Time resolution (timed tracks)",
        "TimeResolution [ps]",
        _real_time("RecoTrack_TimeResolution", 1.0e3),
    ),
]

EVENT_PLOTS = [
    PlotSpec("Reco tracks / event", "NumRecoTrks", _event("NumRecoTrks")),
    PlotSpec("Reco vertices / event", "NumRecoVtx", _event("NumRecoVtx")),
    PlotSpec("Truth vertices / event", "NumTruthVtx", _event("NumTruthVtx")),
    PlotSpec("Pileup", "ActualNumOfInt", _event("ActualNumOfInt")),
    PlotSpec("Beam spot z", "BeamPosZ [mm]", _event("BeamPosZ")),
    PlotSpec("Truth vertex z", "TruthVertex_z [mm]", _vertex("TruthVertex_z")),
    PlotSpec(
        "Truth vertex nTracks",
        "TruthVertex_nTracks",
        _vertex("TruthVertex_nTracks"),
        log_y=True,
    ),
]


def _auto_bins(spec: PlotSpec, values: list[np.ndarray]) -> np.ndarray:
    """Compute common bin edges across all samples for one PlotSpec."""
    nonempty = [v for v in values if len(v)]
    if not nonempty:
        return np.linspace(0, 1, 41)
    pooled = np.concatenate(nonempty)
    lo, hi = np.percentile(pooled, spec.pct)
    if spec.log_x:
        lo = max(lo, np.min(pooled[pooled > 0]) if np.any(pooled > 0) else 1e-3)
        return np.logspace(np.log10(lo), np.log10(hi), 51)
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, 51)


def plot_grid(
    specs: list[PlotSpec],
    data: dict[str, SampleData],
    samples: list[Sample],
    ncols: int,
    title: str,
    out_path: str,
) -> None:
    """Draw one overlaid (density) histogram per spec, one curve per sample."""
    nrows = (len(specs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    color = {s.label: s.color for s in samples}

    for ax, spec in zip(axes, specs):
        vals_by_label = {label: spec.getter(s) for label, s in data.items()}
        bins = _auto_bins(spec, list(vals_by_label.values()))
        for label, vals in vals_by_label.items():
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=bins,
                density=True,
                histtype="step",
                lw=1.4,
                color=color[label],
                label=label,
            )
        ax.set_title(spec.title, fontsize=10)
        ax.set_xlabel(spec.xlabel, fontsize=9)
        if spec.log_x:
            ax.set_xscale("log")
        if spec.log_y:
            ax.set_yscale("log")
        ax.tick_params(labelsize=8)

    for ax in axes[len(specs) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_timing_fraction(
    data: dict[str, SampleData], samples: list[Sample], out_path: str
) -> None:
    """Fraction of tracks carrying real timing vs |eta| (HGTD acceptance check)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    edges = np.linspace(0, 2.6, 27)
    centers = 0.5 * (edges[:-1] + edges[1:])
    color = {s.label: s.color for s in samples}
    for label, s in data.items():
        eta = s.track.get("RecoTrack_eta")
        t = s.track.get("RecoTrack_Time")
        if eta is None or t is None:
            continue
        aeta = np.abs(eta)
        timed = t > TIME_SENTINEL
        num, _ = np.histogram(aeta[timed], bins=edges)
        den, _ = np.histogram(aeta, bins=edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(den > 0, num / den, np.nan)
        ax.plot(centers, frac, lw=1.6, color=color[label], label=label)
    ax.set_xlabel("|eta|", fontsize=11)
    ax.set_ylabel("fraction of tracks with timing", fontsize=11)
    ax.set_title("Timing acceptance vs |eta| (HGTD forward region)", fontsize=12)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def summarize(data: dict[str, SampleData], samples: list[Sample]) -> dict:
    """Per-sample mean/std for every plotted quantity (for the meeting notes)."""
    all_specs = TRACK_PLOTS + TIMING_PLOTS + EVENT_PLOTS
    out: dict = {}
    for label, s in data.items():
        rec: dict = {
            "n_events": s.n_events,
            "n_tracks": int(len(s.track.get("RecoTrack_d0", []))),
        }
        timed = 0
        if "RecoTrack_Time" in s.track:
            timed = int(np.sum(s.track["RecoTrack_Time"] > TIME_SENTINEL))
        rec["n_timed_tracks"] = timed
        rec["timed_fraction"] = (timed / rec["n_tracks"]) if rec["n_tracks"] else 0.0
        mu = s.event.get("ActualNumOfInt")
        if mu is not None and len(mu):
            rec["mean_mu"] = float(np.mean(mu))
            rec["frac_mu_lt100"] = float(np.mean(mu < 100))
        for spec in all_specs:
            v = spec.getter(s)
            if len(v):
                rec[spec.title] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        out[label] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events-per-file", type=int, default=5000)
    ap.add_argument("--output-dir", default="outputs/06_02_2026_output/timing_data_qa")
    ap.add_argument(
        "--pu200-only",
        action="store_true",
        help="keep only events with ActualNumOfInt > 100 (drops the PU0 "
        "half of the mixed r16443 / r16638 reco tags)",
    )
    args = ap.parse_args()

    mu_min = 100.0 if args.pu200_only else -1.0
    os.makedirs(args.output_dir, exist_ok=True)
    samples = build_samples()
    print(
        f"Loading {len(samples)} samples, {args.events_per_file} events/file "
        f"(pu200_only={args.pu200_only}) ..."
    )
    data: dict[str, SampleData] = {
        s.label: load_sample(s, args.events_per_file, mu_min) for s in samples
    }

    print("Plotting ...")
    plot_grid(
        TRACK_PLOTS,
        data,
        samples,
        ncols=3,
        title="HL-LHC PU200 with timing — track kinematics & uncertainties",
        out_path=f"{args.output_dir}/track_kinematics.png",
    )
    plot_grid(
        TIMING_PLOTS,
        data,
        samples,
        ncols=2,
        title="HL-LHC PU200 — track timing (timed tracks only, sentinel -1 masked)",
        out_path=f"{args.output_dir}/timing.png",
    )
    plot_timing_fraction(data, samples, f"{args.output_dir}/timing_fraction_vs_eta.png")
    plot_grid(
        EVENT_PLOTS,
        data,
        samples,
        ncols=4,
        title="HL-LHC PU200 with timing — event-level & truth-vertex sanity",
        out_path=f"{args.output_dir}/event_level.png",
    )

    summary = summarize(data, samples)
    with open(f"{args.output_dir}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  wrote {args.output_dir}/summary.json")
    print("Done.")


if __name__ == "__main__":
    main()
