"""Compare the pairwise-Delta-z distribution of PV-Finder, AMVF and truth.

The pairwise-Delta-z distribution of reconstructed vertices is the observable
behind ``sigma_vtx_vtx``: pairs closer than the algorithm can resolve are
merged into one vertex, so the distribution is suppressed near zero and rises
to a flat plateau at large separation. Fitting a sigmoid to it gives the
half-recovery width.

Viewed at fine binning it is not a clean sigmoid: there is an **excess just
outside the dip**. This script quantifies that excess and, critically, plots
the same observable for AMVF and for the truth vertices on the *same events*,
which is what distinguishes the three candidate explanations:

* if the excess is present in **truth**, it is the vertex-spacing physics of
  the sample;
* if it is present in **AMVF as well**, it is a generic property of vertex
  reconstruction, not of PV-Finder;
* if it is unique to PV-Finder, it is a finder artefact (sidelobes).

Run from the repo root with the venv active::

    python -u src/pv_finder/diagnostics/pairwise_dz_comparison.py \\
        --pkl outputs/<date>/eval_.../eval_results.pkl \\
        --root data/run4_all_etas/.../ATLAS_PVFinderData_..._r16443_PU200.root \\
        --mu-min 185 --mu-max 215 --n-events 1500 \\
        --output-dir outputs/<date>/pairwise_dz
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot

TREE = "PVFinderData"
BASELINE_MIN_MM = 3.0  # |dz| beyond which the distribution is flat


def pairwise_profile(
    z_per_event: Sequence[np.ndarray], n_bins: int = 240, half_range: float = 6.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Symmetrised pairwise-Delta-z histogram and its large-|dz| baseline.

    Returns ``(bin_centres, counts, baseline)``. The baseline is the median
    count for ``|dz| > BASELINE_MIN_MM``, where the distribution is flat.
    """
    chunks = []
    for z in z_per_event:
        z = np.asarray(z, dtype=np.float64)
        z = z[np.isfinite(z)]
        if len(z) < 2:
            continue
        d = z[:, None] - z[None, :]
        chunks.append(d[np.triu_indices(len(z), 1)])
    if not chunks:
        raise ValueError("no event had two or more vertices")
    dz = np.concatenate(chunks)
    dz = np.concatenate([dz, -dz])  # symmetrise
    edges = np.linspace(-half_range, half_range, n_bins + 1)
    counts, _ = np.histogram(dz, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    # Median over the positive half only: the histogram is symmetric by
    # construction, so using both halves doubles every value and makes the
    # median land on an arbitrary tie at the 0.5-count level.
    baseline = float(np.median(counts[centres > BASELINE_MIN_MM]))
    return centres, counts, baseline


def summarise(centres: np.ndarray, counts: np.ndarray, baseline: float) -> dict:
    """Dip depth, peak excess and its location, and the 0.3-0.7 mm band mean."""
    rel = (counts - baseline) / baseline
    search = (np.abs(centres) > 0.25) & (np.abs(centres) < 2.5)
    i = int(np.argmax(np.where(search, rel, -np.inf)))
    band = (np.abs(centres) >= 0.3) & (np.abs(centres) <= 0.7)
    dip = np.abs(centres) < 0.1
    return {
        "baseline": baseline,
        "dip_depth_pct": float(100 * rel[dip].mean()),
        "peak_excess_pct": float(100 * rel[i]),
        "peak_excess_at_mm": float(abs(centres[i])),
        "band_0p3_0p7_pct": float(100 * rel[band].mean()),
    }


def load_amvf_and_truth(
    root_path: str,
    n_events: int,
    mu_min: float,
    mu_max: float,
    n_entries: int,
    min_ntrk: int = 2,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """AMVF and truth vertex z per selected event, matching the eval's mu window."""
    # entry_stop MUST NOT exceed the range the pkl covers, or the mu selection
    # walks past it and we silently compare PV-Finder against AMVF/truth from
    # DIFFERENT events. The caller passes n_entries = len(pkl per_event).
    arrs = uproot.open(f"{root_path}:{TREE}").arrays(
        [
            "RecoVertex_z",
            "TruthVertex_z",
            "TruthVertex_nTracks",
            "RecoVertex_nTracks",
            "ActualNumOfInt",
        ],
        entry_stop=n_entries,
        library="np",
    )
    mu = np.asarray(arrs["ActualNumOfInt"], dtype=float)
    keep = np.where((mu >= mu_min) & (mu <= mu_max))[0][:n_events]
    # The eval counts AMVF vertices with nTracks >= 2 (verified: per_event
    # ["n_amvf"] matches that cut exactly, not len(RecoVertex_z) -- ~1.3% of
    # events carry a type==1 dummy with nTrk < 2). Apply the same cut here or
    # the AMVF curve is a different object from the one the eval reports.
    amvf = [
        np.asarray(arrs["RecoVertex_z"][i], dtype=np.float64)[
            np.asarray(arrs["RecoVertex_nTracks"][i]) >= min_ntrk
        ]
        for i in keep
    ]
    truth = [
        np.asarray(arrs["TruthVertex_z"][i], dtype=np.float64)[
            np.asarray(arrs["TruthVertex_nTracks"][i]) >= min_ntrk
        ]
        for i in keep
    ]
    return amvf, truth


def plot(profiles: dict, out_path: Path, title: str) -> None:
    """Two panels: full +-3 mm range, and a zoom on the dip and the excess."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"PV-Finder": "#d62728", "AMVF": "#1f77b4", "Truth": "#2ca02c"}
    for ax, (lo, hi) in zip(axes, [(-3.0, 3.0), (0.0, 1.6)]):
        for name, (ctr, cnt, base) in profiles.items():
            rel = 100 * (cnt - base) / base
            m = (ctr >= lo) & (ctr <= hi)
            ax.step(
                ctr[m],
                rel[m],
                where="mid",
                lw=1.6,
                color=colors.get(name, "k"),
                label=name,
            )
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(r"$\Delta z$ between vertex pairs [mm]", fontsize=11)
        ax.set_ylabel("deviation from large-$|\\Delta z|$ plateau [%]", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylim(-105, 30)
    axes[0].legend(fontsize=10, loc="lower right")
    axes[0].set_title("Full range: the resolution dip", fontsize=11)
    axes[1].set_ylim(-105, 30)
    axes[1].set_title("Zoom: dip edge and the excess beyond it", fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pkl", required=True, help="eval_results.pkl (PV-Finder positions)"
    )
    p.add_argument("--root", required=True, help="matching ROOT file for AMVF + truth")
    p.add_argument("--n-events", type=int, default=1500)
    p.add_argument("--mu-min", type=float, default=185.0)
    p.add_argument("--mu-max", type=float, default=215.0)
    p.add_argument("--n-bins", type=int, default=240)
    p.add_argument("--label", default="HL-LHC PU200")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pkl, "rb") as fh:
        res = pickle.load(fh)
    # The eval stores EVERY event it read, not just the ones inside the mu
    # window it summarises. On a flat-mu held-out file only ~8% of events are
    # PU200-like, so slicing the first N here would compare PV-Finder at
    # <mu>~100 against AMVF/truth at <mu>~192 -- different vertex densities,
    # different plateau, meaningless shape comparison. Select on the stored
    # per-event mu so all three curves use the same events.
    mu_pkl = np.array([e.get("mu", np.nan) for e in res["per_event"]], dtype=float)
    keep_pkl = np.where((mu_pkl >= args.mu_min) & (mu_pkl <= args.mu_max))[0]
    if len(keep_pkl) == 0:
        raise ValueError(
            f"no event in the pkl has mu in [{args.mu_min}, {args.mu_max}]"
        )
    keep_pkl = keep_pkl[: args.n_events]
    print(
        f"  pkl: {len(res['per_event']):,} events read, "
        f"{len(keep_pkl):,} used (mu in [{args.mu_min:.0f},{args.mu_max:.0f}], "
        f"mean mu {np.nanmean(mu_pkl[keep_pkl]):.1f})"
    )
    pvf = [np.asarray(res["pred_pvs_mm"][i], dtype=np.float64) for i in keep_pkl]
    amvf, truth = load_amvf_and_truth(
        args.root,
        len(keep_pkl),
        args.mu_min,
        args.mu_max,
        n_entries=len(res["per_event"]),
    )
    if not (len(amvf) == len(truth) == len(pvf)):
        raise ValueError(
            f"event-count mismatch: pvf={len(pvf)} amvf={len(amvf)} "
            f"truth={len(truth)} -- the three curves would not be the same events"
        )

    profiles, summary = {}, {}
    for name, zs in (("PV-Finder", pvf), ("AMVF", amvf), ("Truth", truth)):
        ctr, cnt, base = pairwise_profile(zs, n_bins=args.n_bins)
        profiles[name] = (ctr, cnt, base)
        summary[name] = summarise(ctr, cnt, base)
        s = summary[name]
        print(
            f"  {name:10s} baseline {s['baseline']:9.0f} | dip {s['dip_depth_pct']:7.1f}% "
            f"| peak excess {s['peak_excess_pct']:6.1f}% at {s['peak_excess_at_mm']:.2f} mm "
            f"| band 0.3-0.7 {s['band_0p3_0p7_pct']:6.1f}%"
        )

    plot(
        profiles,
        outdir / "pairwise_dz_comparison.png",
        f"Pairwise $\\Delta z$: PV-Finder vs AMVF vs truth — {args.label}",
    )
    with open(outdir / "pairwise_dz_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  wrote {outdir}/pairwise_dz_comparison.png")
    print(f"  wrote {outdir}/pairwise_dz_summary.json")


if __name__ == "__main__":
    main()
