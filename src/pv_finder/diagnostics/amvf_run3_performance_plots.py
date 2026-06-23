"""AMVF resolution and reconstructed-vertex category plots (AMVF-only; no
PV-Finder inference). Reproduces two ATLAS-style figures from a PV-Finder ntuple.

* ``amvf_resolution_delta_z`` — signed ``Delta z`` between all AMVF reco-vertex
  pairs; central dip fit with the eval's sigmoid notch for ``sigma_vtx_vtx`` (no
  truth needed, works on real Run 3 data).
* ``amvf_vertex_categories_vs_mu`` — average AMVF vertices per category (All /
  Matched / Merged / Split / Fake) vs ``ActualNumOfInt`` (needs MC truth).

Definitions match ``run_eval_pvf_run3.py`` by reusing ``compare_res_reco``:
Matched = ``reco_clean`` (won a 1-to-1 greedy closest-first assignment in the
window); Merged = ``reco_merged`` (assigned reco that also absorbs extra
in-window truths); Split = ``reco_split`` (unassigned reco whose in-window truth
a closer reco took); Fake = ``reco_fake`` (no truth in window). Truth =
``TruthVertex_z`` nTracks >= 2 (detector frame, no beam corr.); window =
``sigma_vtx_vtx`` (bins), here fit from the AMVF reco-reco pairwise ``Delta z``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — safe over SSH + tmux
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pv_finder.data.run3_io import Run3Event, load_run3_from_root  # noqa: E402
from pv_finder.evaluation.vertex_finding.efficiency_res_optimized_atlas import (  # noqa: E402
    compare_res_reco,
)

# z-grid and bin width must match the evaluation scripts so that the matching
# window (sigma converted to bins) means the same thing here as it does there.
Z_MIN, Z_MAX = -240.0, 240.0  # mm
N_BINS_FULL = 12000
BIN_WIDTH = (Z_MAX - Z_MIN) / N_BINS_FULL  # 0.04 mm/bin

DEFAULT_ROOT = "data/run3/file_3.root"
DEFAULT_OUT = "outputs/06_23_2026_output/amvf_run3_2500"

# Resolution-fit histogram window/edges (mirrors run_eval_pvf_run3.py: 61 edges
# over [-6, 6]). Widened default range so the flat plateau is visible like the
# ATLAS reference figure; the fitted sigma is insensitive to the range.
RES_DZ_MAX = 8.0
RES_N_BINS = 80


def mm_to_bins(z_mm: np.ndarray) -> np.ndarray:
    """Convert detector-frame z positions (mm) to 12000-bin coordinates."""
    return (z_mm - Z_MIN) / BIN_WIDTH


def sigmoid_notch(
    x_mm: np.ndarray,
    amplitude: float,
    slope: float,
    baseline: float,
    sigma_mm: float,
) -> np.ndarray:
    """Symmetric sigmoid notch (identical form to the eval's ``sigmoid_fit``)."""
    return amplitude / (1.0 + np.exp(slope * (sigma_mm - np.abs(x_mm)))) + baseline


# --------------------------------------------------------------------------
# Resolution: pairwise AMVF Delta z + sigmoid-notch fit
# --------------------------------------------------------------------------


def collect_pairwise_dz(events: list[Run3Event]) -> np.ndarray:
    """Return the symmetric set of pairwise AMVF vertex separations (mm).

    For each event with >= 2 AMVF vertices, every unordered pair contributes
    both ``+Delta z`` and ``-Delta z`` so the distribution is symmetric for the
    fit (matching the eval's convention).
    """
    values: list[float] = []
    for event in events:
        z = np.asarray(event.amvf_z, dtype=np.float64)
        if z.size < 2:
            continue
        dz = z[:, None] - z[None, :]
        upper = dz[np.triu_indices(z.size, k=1)]
        values.extend(upper.tolist())
        values.extend((-upper).tolist())
    return np.asarray(values, dtype=np.float64)


def fit_resolution(
    dz_mm: np.ndarray,
    *,
    dz_max_mm: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[float] | None]:
    """Histogram AMVF pair distances and fit the central dip for sigma_vtx_vtx.

    The fit is performed on raw counts with the same functional form, starting
    point, and bounds as ``run_eval_pvf_run3.py`` so the extracted sigma is
    directly comparable. The returned histogram, errors, and fit parameters are
    all divided by the total count so they share an "Arbitrary units" display
    scale (vertical normalisation does not change the fitted sigma). Returns bin
    centres, the normalised histogram, normalised Poisson errors, the fitted
    sigma (mm), and the normalised fit parameters
    ``[amplitude, slope, baseline, sigma, sigma_err]`` (or ``None`` on failure).
    """
    bins = np.linspace(-dz_max_mm, dz_max_mm, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _ = np.histogram(dz_mm, bins=bins)
    counts_f = counts.astype(np.float64)
    total = max(float(counts_f.sum()), 1.0)

    sigma = 0.5
    popt: list[float] | None = None
    baseline = float(np.median(counts_f))
    dip = float(baseline) - float(counts_f.min())
    p0 = [max(dip, 1.0), 10.0, max(baseline, 1.0), 0.5]
    try:
        fit, pcov = curve_fit(
            sigmoid_notch,
            centers,
            counts_f,
            p0=p0,
            bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, np.inf, dz_max_mm]),
            maxfev=20000,
        )
        sigma = float(abs(fit[3]))
        sigma_err = float(np.sqrt(np.diag(pcov))[3])
        # Normalise the vertical parameters (amplitude, baseline) so the fit
        # overlays the normalised histogram exactly; slope and sigma are
        # horizontal/shape parameters and are left untouched.
        amplitude, slope, base, sig = (float(v) for v in fit)
        popt = [amplitude / total, slope, base / total, sig, sigma_err]
    except (RuntimeError, ValueError):
        pass

    y = counts_f / total
    yerr = np.sqrt(counts_f) / total
    return centers, y, yerr, sigma, popt


def plot_resolution(
    centers: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    sigma_mm: float,
    popt: list[float] | None,
    out_dir: Path,
    label: str,
    sample_desc: str,
) -> None:
    """Draw the AMVF pairwise Delta z resolution figure."""
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    ax.errorbar(
        centers,
        y,
        yerr=yerr,
        fmt="s",
        color="#2f6fab",
        ms=3.4,
        lw=0,
        elinewidth=0.8,
        label="AMVF vertex pairs",
    )
    if popt is not None:
        xf = np.linspace(float(centers.min()), float(centers.max()), 600)
        ax.plot(
            xf,
            sigmoid_notch(xf, *popt[:4]),
            color="#2f6fab",
            lw=1.1,
            alpha=0.9,
            label=rf"sigmoid fit, $\sigma_\mathrm{{vtx-vtx}}={sigma_mm:.3f}$ mm",
        )
    ax.text(
        0.06, 0.93, "ATLAS", transform=ax.transAxes, fontsize=22,
        fontweight="bold", fontstyle="italic", va="top",
    )  # fmt: skip
    ax.text(0.27, 0.93, label, transform=ax.transAxes, fontsize=18, va="top")
    ax.text(0.06, 0.85, sample_desc, transform=ax.transAxes, fontsize=14, va="top")
    ax.set_xlabel(r"$\Delta z$ [mm]", fontsize=16)
    ax.set_ylabel("Arbitrary units", fontsize=16)
    ax.set_xlim(float(centers.min()), float(centers.max()))
    # Headroom above the plateau so the ATLAS label clears the data points.
    ax.set_ylim(0.0, float(y.max()) * 1.30)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.tick_params(axis="both", labelsize=13, direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.legend(loc="center right", fontsize=11, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"amvf_resolution_delta_z.{suffix}", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------
# Categories: AMVF vs truth via compare_res_reco (identical to the eval)
# --------------------------------------------------------------------------


def classify_amvf_events(
    events: list[Run3Event],
    *,
    match_window_mm: float,
) -> list[dict[str, float | int]]:
    """Classify AMVF vertices per event into matched/merged/split/fake.

    Mirrors the ``has_truth`` AMVF block of ``run_eval_pvf_run3.py`` exactly:
    truth and AMVF are taken in the detector frame (no beam correction), events
    with zero truth vertices are skipped, and ``compare_res_reco`` is called
    with a per-reco matching window of ``match_window_mm`` (converted to bins).
    """
    window_bins = match_window_mm / BIN_WIDTH
    per_event: list[dict[str, float | int]] = []
    for event in events:
        if event.truth_z is None:
            raise RuntimeError("TruthVertex_z is required for AMVF categories.")
        truth_bins = mm_to_bins(np.asarray(event.truth_z, dtype=np.float64))
        amvf_bins = mm_to_bins(np.asarray(event.amvf_z, dtype=np.float64))
        n_truth = int(truth_bins.size)
        if n_truth == 0:
            continue  # same as eval: `if nt == 0: continue`
        n_amvf = int(amvf_bins.size)
        if n_amvf > 0:
            res, _, _ = compare_res_reco(
                truth_bins,
                amvf_bins,
                window_bins * np.ones(n_amvf),
                debug=0,
            )
            matched, merged, split, fake = (
                res.reco_clean,
                res.reco_merged,
                res.reco_split,
                res.reco_fake,
            )
        else:
            matched = merged = split = fake = 0
        per_event.append(
            {
                "event_idx": int(event.event_idx),
                "mu": float(event.mu) if event.mu is not None else float(n_truth),
                "n_truth": n_truth,
                "all_reco": n_amvf,
                "matched": int(matched),
                "merged": int(merged),
                "split": int(split),
                "fake": int(fake),
            }
        )
    return per_event


def bucket_means(
    per_event: list[dict[str, float | int]],
) -> dict[int, dict[str, tuple[float, float]]]:
    """Mean and SEM of AMVF category counts in each integer pileup bin."""
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in per_event:
        bucket = int(round(float(row["mu"])))
        for key in ("all_reco", "matched", "merged", "split", "fake"):
            buckets[bucket][key].append(float(row[key]))

    out: dict[int, dict[str, tuple[float, float]]] = {}
    for bucket, values in buckets.items():
        out[bucket] = {}
        for key, vals in values.items():
            arr = np.asarray(vals, dtype=np.float64)
            sem = float(arr.std(ddof=0) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
            out[bucket][key] = (float(arr.mean()), sem)
    return dict(sorted(out.items()))


def plot_categories(
    per_event: list[dict[str, float | int]],
    out_dir: Path,
    label: str,
    sample_desc: str,
) -> dict[int, dict[str, tuple[float, float]]]:
    """Draw mean AMVF reconstructed vertices per category vs interactions."""
    stats = bucket_means(per_event)
    if not stats:
        raise ValueError("No populated pileup bins — cannot draw category plot.")
    x = np.asarray(sorted(stats.keys()), dtype=float)

    # (legend label, internal key, colour, marker)
    curves = [
        ("All reconstructed", "all_reco", "#000000", "*"),
        ("Matched", "matched", "#1f6fb2", "*"),
        ("Merged", "merged", "#1aa37a", "*"),
        ("Split", "split", "#c45a9c", "*"),
        ("Fake", "fake", "#7ec3ec", "*"),
    ]

    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    for legend, key, color, marker in curves:
        y = np.asarray([stats[int(v)][key][0] for v in x], dtype=float)
        err = np.asarray([stats[int(v)][key][1] for v in x], dtype=float)
        ax.errorbar(
            x, y, yerr=err, fmt=f"-{marker}", color=color, ms=6, lw=1.4,
            capsize=0, label=legend,
        )  # fmt: skip

    ax.text(
        0.06, 0.94, "ATLAS", transform=ax.transAxes, fontsize=22,
        fontweight="bold", fontstyle="italic", va="top",
    )  # fmt: skip
    ax.text(0.27, 0.94, label, transform=ax.transAxes, fontsize=18, va="top")
    ax.text(0.06, 0.86, sample_desc, transform=ax.transAxes, fontsize=14, va="top")
    ax.set_xlabel("Number of interactions", fontsize=16)
    ax.set_ylabel("Average number of reconstructed vertices", fontsize=16)
    ax.set_xlim(left=0.0, right=float(x.max()) + 2.0)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="both", labelsize=13, direction="in", top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.05, 0.80), frameon=False, fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"amvf_vertex_categories_vs_mu.{suffix}", dpi=180)
    plt.close(fig)
    return stats


# --------------------------------------------------------------------------
# Bookkeeping
# --------------------------------------------------------------------------


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    n_events: int,
    dz_mm: np.ndarray,
    sigma_mm: float,
    popt: list[float] | None,
    match_window_mm: float,
    per_event: list[dict[str, float | int]] | None,
    category_stats: dict[int, dict[str, tuple[float, float]]] | None,
) -> None:
    """Persist compact numeric outputs for reproducibility."""
    summary: dict = {
        "root_path": args.root,
        "n_events": n_events,
        "n_pairwise_dz": int(dz_mm.size),
        "sigma_vtx_vtx_mm": sigma_mm,
        "resolution_fit_params": popt,
        "match_window_mm": match_window_mm,
        "min_truth_ntrks": 2,
        "min_amvf_ntrks": 2,
        "has_truth": per_event is not None,
    }
    if per_event is not None:
        summary["n_events_with_truth"] = len(per_event)
        summary["per_event_means"] = {
            key: float(np.mean([row[key] for row in per_event]))
            for key in ("n_truth", "all_reco", "matched", "merged", "split", "fake")
        }
    if category_stats is not None:
        summary["category_stats_by_mu"] = {
            str(mu): {k: {"mean": v[0], "sem": v[1]} for k, v in vals.items()}
            for mu, vals in category_stats.items()
        }
    with (out_dir / "summary.json").open("w") as fp:
        json.dump(summary, fp, indent=2)

    np.savez_compressed(
        out_dir / "amvf_arrays.npz",
        pairwise_dz_mm=dz_mm,
        per_event=np.asarray(per_event, dtype=object)
        if per_event is not None
        else np.asarray([], dtype=object),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", default=DEFAULT_ROOT, help="ATLAS ntuple ROOT input")
    parser.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--max-events", type=int, default=2500)
    parser.add_argument("--entry-start", type=int, default=0)
    parser.add_argument("--entry-stop", type=int, default=None)
    parser.add_argument("--min-tracks", type=int, default=1)
    parser.add_argument("--min-amvf-vtx", type=int, default=1)
    parser.add_argument("--dz-max", type=float, default=RES_DZ_MAX, dest="dz_max")
    parser.add_argument("--dz-bins", type=int, default=RES_N_BINS, dest="dz_bins")
    parser.add_argument(
        "--match-window",
        type=float,
        default=0.0,
        help="AMVF/truth matching window in mm. 0 (default) uses the fitted "
        "sigma_vtx_vtx, exactly as run_eval_pvf_run3.py reuses its sigma.",
    )
    parser.add_argument(
        "--plot-label",
        default="Simulation Internal",
        help="ATLAS plot label, e.g. 'Simulation Internal' or 'Internal'.",
    )
    parser.add_argument(
        "--sample-desc",
        default="",
        help="Second-line annotation (e.g. r'$t\\bar{t}$, $\\sqrt{s}=13$ TeV').",
    )
    return parser.parse_args()


def main() -> None:
    """Run the AMVF-only resolution and category diagnostics."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = load_run3_from_root(
        args.root,
        max_events=args.max_events,
        min_tracks=args.min_tracks,
        min_amvf_vtx=args.min_amvf_vtx,
        entry_start=args.entry_start,
        entry_stop=args.entry_stop,
    )
    if not events:
        raise SystemExit("No events loaded after filtering.")

    has_truth = events[0].truth_z is not None
    n_truth_total = (
        sum(int(np.asarray(e.truth_z).size) for e in events) if has_truth else 0
    )
    print(
        f"[amvf] {len(events)} events  has_truth={has_truth}  truth_verts={n_truth_total}"
    )

    # ---- Resolution (no truth needed) -----------------------------------
    dz_mm = collect_pairwise_dz(events)
    centers, y, yerr, sigma_mm, popt = fit_resolution(
        dz_mm, dz_max_mm=args.dz_max, n_bins=args.dz_bins
    )
    res_desc = args.sample_desc or ("Run 3 data" if not has_truth else "")
    plot_resolution(
        centers, y, yerr, sigma_mm, popt, out_dir, args.plot_label, res_desc
    )
    print(f"[amvf] sigma_vtx_vtx = {sigma_mm:.4f} mm  (pairs={dz_mm.size:,})")

    # ---- Categories (truth required) ------------------------------------
    per_event: list[dict[str, float | int]] | None = None
    category_stats: dict[int, dict[str, tuple[float, float]]] | None = None
    match_window = args.match_window if args.match_window > 0 else sigma_mm

    if has_truth and n_truth_total > 0:
        per_event = classify_amvf_events(events, match_window_mm=match_window)
        if per_event:
            cat_desc = args.sample_desc or r"$t\bar{t}$, $\sqrt{s}=13$ TeV"
            category_stats = plot_categories(
                per_event, out_dir, args.plot_label, cat_desc
            )
            means = {
                k: float(np.mean([r[k] for r in per_event]))
                for k in ("all_reco", "matched", "merged", "split", "fake")
            }
            print(f"[amvf] match_window = {match_window:.4f} mm")
            print(f"[amvf] per-event category means = {means}")
        else:
            print(
                "[amvf] WARNING: truth branches present but no event had truth "
                "vertices with nTracks>=2 — skipping the category plot."
            )
    else:
        print(
            "[amvf] No MC truth in this sample (real data). The matched/merged/"
            "split/fake categories are undefined without truth, so only the "
            "resolution plot was produced."
        )

    write_summary(
        out_dir,
        args,
        len(events),
        dz_mm,
        sigma_mm,
        popt,
        match_window,
        per_event,
        category_stats,
    )
    print(f"[amvf] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
