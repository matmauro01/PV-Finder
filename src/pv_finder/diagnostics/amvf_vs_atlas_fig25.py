"""AMVF on Run 3 MC against ATLAS IDTR-2021-01 Figure 25(a) and 25(b).

Sanity check requested by LT: do we classify reconstructed vertices the same
way the ATLAS ID performance paper does?

ATLAS "Track and Vertex Reconstruction with the ATLAS Inner Detector"
(IDTR-2021-01, arXiv:2605.07585), Section 6.5, classifies reconstructed
primary vertices into four types using the truth-matching of reconstructed
tracks and the associated weight from the vertex fit:

  * Matched: >= 70% of the total track weight comes from a single simulated
    pp interaction.
  * Merged:  < 70% of the total track weight comes from any single
    simulated pp interaction.
  * Split:   if a single interaction contributes the largest fraction of
    track weights to two or more reconstructed vertices, the vertex with the
    largest track sum-pT^2 is Matched or Merged and the other(s) are Split.
  * Fake:    fake tracks contribute more weight than any simulated pp
    interaction.

That is the same rule set as ``gnn.evaluation.classification`` with
``PURITY_THRESHOLD = 0.7`` (our "Clean" is their "Matched"). The one
difference is the weighting: ATLAS weights each track by its vertex-fit
weight, we count tracks unweighted, because the per-track fit weights are not
stored in this ntuple. See docs/evaluation/atlas_fig25_crosscheck.md.

Figure 25(a) is the average number of reconstructed vertices per category
versus the number of simulated interactions, with two reference lines:
the diagonal ("100% interaction reconstruction efficiency") and the
"reconstruction acceptance", defined as the number of interactions having at
least two reconstructed tracks in the detector.

Figure 25(b) is the distance in z between reconstructed vertices. In the
absence of merging it would be Gaussian with a standard deviation of
sqrt(2) x the beamspot size; the dip at zero is caused by two interactions
being reconstructed as one vertex.

Run inside tmux (shared host):

    tmux new -s fig25
    source venv/bin/activate
    python -u src/pv_finder/diagnostics/amvf_vs_atlas_fig25.py \
        --h5 /share/lazy/qibinlei/recoTracks_incamvfassoc.h5 \
        --root data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root \
        --max-events 12000 \
        --output-dir outputs/08_20_2026_output/amvf_vs_atlas_fig25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import uproot
from tqdm import tqdm

from gnn.evaluation.classification import build_truth_adjacency, classify_assignments

CATEGORIES = ("Matched", "Merged", "Split", "Fake")
# classify_assignments returns [clean, merged, split, fake, n_reco, n_truth];
# ATLAS calls our "Clean" category "Matched".
_RESULT_SLOTS = (0, 1, 2, 3)


def load_mu(root_path: str | Path, n_events: int) -> tuple[np.ndarray, float]:
    """Read per-event pile-up and the mean beamspot z width from the ntuple.

    Args:
        root_path: ATLAS PVFinderData ROOT file.
        n_events: Number of leading events to read.

    Returns:
        Tuple of (ActualNumOfInt per event, mean BeamPosSigmaZ in mm).
    """
    tree = uproot.open(str(root_path))["PVFinderData"]
    mu = tree["ActualNumOfInt"].array(library="np", entry_stop=n_events)
    sigma_z = float(
        np.mean(tree["BeamPosSigmaZ"].array(library="np", entry_stop=n_events))
    )
    return np.asarray(mu, dtype=float), sigma_z


def unpack_assoc(
    ntracks: np.ndarray, assoc: np.ndarray, min_ntracks: int
) -> list[np.ndarray]:
    """Split a flat concatenated association list into per-vertex arrays.

    Args:
        ntracks: Per-vertex track counts (the flat list's segment lengths).
        assoc: Concatenated track indices for all vertices.
        min_ntracks: Drop vertices with fewer than this many tracks.

    Returns:
        List of unique track-index arrays, one per surviving vertex.
    """
    out: list[np.ndarray] = []
    cursor = 0
    for n in ntracks.astype(int):
        tracks = np.unique(assoc[cursor : cursor + n].astype(int))
        cursor += n
        if n >= min_ntracks:
            out.append(tracks)
    return out


def collect(
    h5_path: str | Path,
    root_path: str | Path,
    max_events: int,
    min_reco_ntracks: int,
    dz_range: float,
    dz_bin: float,
) -> dict[str, Any]:
    """Classify AMVF vertices per event and accumulate the Figure 25 inputs.

    Args:
        h5_path: Event-keyed HDF5 with pv_* truth and reco_pv_* AMVF data.
        root_path: Matching ROOT ntuple (EventN <-> entry N) for pile-up.
        max_events: Number of events to process.
        min_reco_ntracks: Minimum tracks for an AMVF vertex to be counted.
        dz_range: Half-width in mm of the pairwise dz histogram.
        dz_bin: Bin width in mm of the pairwise dz histogram.

    Returns:
        Dict with per-event category counts, pile-up, acceptance and the
        pairwise dz histogram.
    """
    mu, beam_sigma_z = load_mu(root_path, max_events)

    n_bins = int(round(2 * dz_range / dz_bin))
    dz_edges = np.linspace(-dz_range, dz_range, n_bins + 1)
    dz_hist = np.zeros(n_bins, dtype=np.int64)
    n_pairs_total = 0

    counts = np.zeros((max_events, len(CATEGORIES)), dtype=np.int64)
    acceptance = np.zeros(max_events, dtype=np.int64)
    n_truth_evt = np.zeros(max_events, dtype=np.int64)

    with h5py.File(str(h5_path), "r") as f:
        pt_ds = f["recoTrk_pt"]
        pv_z_ds = f["pv_loc_z"]
        pv_nt_ds = f["pv_ntracks"]
        pv_at_ds = f["pv_assoc_tracks"]
        rv_z_ds = f["reco_pv_loc_z"]
        rv_nt_ds = f["reco_pv_ntracks"]
        rv_at_ds = f["reco_pv_assoc_tracks"]

        for ev in tqdm(range(max_events), desc="events"):
            key = f"Event{ev}"
            pv_nt = pv_nt_ds[key][:]

            truth_track_idx, truth_pv_idx = build_truth_adjacency(
                pv_z_ds[key][:], pv_nt, pv_at_ds[key][:]
            )
            # ATLAS reconstruction acceptance: interactions with >= 2
            # reconstructed tracks in the detector.
            n_truth = int((pv_nt >= 2).sum())
            acceptance[ev] = n_truth
            n_truth_evt[ev] = n_truth

            reco_tracks = unpack_assoc(
                rv_nt_ds[key][:], rv_at_ds[key][:], min_reco_ntracks
            )
            results, _ = classify_assignments(
                reco_tracks, pt_ds[key][:], truth_track_idx, truth_pv_idx, n_truth
            )
            counts[ev] = [results[i] for i in _RESULT_SLOTS]

            # Pairwise dz between the same surviving AMVF vertices.
            rv_z = rv_z_ds[key][:]
            rv_nt = rv_nt_ds[key][:]
            rv_z = rv_z[rv_nt >= min_reco_ntracks]
            if rv_z.size > 1:
                diff = rv_z[:, None] - rv_z[None, :]
                iu = np.triu_indices(rv_z.size, k=1)
                d = diff[iu]
                # Symmetrize: every unordered pair contributes +d and -d.
                both = np.concatenate([d, -d])
                n_pairs_total += both.size
                dz_hist += np.histogram(both, bins=dz_edges)[0]

    return {
        "mu": mu,
        "counts": counts,
        "acceptance": acceptance,
        "n_truth": n_truth_evt,
        "dz_hist": dz_hist,
        "dz_edges": dz_edges,
        "n_pairs_total": n_pairs_total,
        "beam_sigma_z": beam_sigma_z,
        "min_reco_ntracks": min_reco_ntracks,
    }


def profile_vs_mu(
    data: dict[str, Any], mu_min: int, mu_max: int
) -> dict[str, np.ndarray]:
    """Average the per-category vertex counts in unit-width pile-up bins.

    Args:
        data: Output of :func:`collect`.
        mu_min: Lower edge of the pile-up range.
        mu_max: Upper edge of the pile-up range.

    Returns:
        Dict with bin centres, per-category means, acceptance mean and the
        number of events per bin.
    """
    mu = data["mu"]
    counts = data["counts"]
    acceptance = data["acceptance"]

    edges = np.arange(mu_min, mu_max + 1, 1.0)
    idx = np.digitize(mu, edges) - 1
    valid = (idx >= 0) & (idx < len(edges) - 1)

    centres = 0.5 * (edges[:-1] + edges[1:])
    n_per_bin = np.zeros(len(centres), dtype=np.int64)
    means = np.zeros((len(centres), len(CATEGORIES)))
    errs = np.zeros((len(centres), len(CATEGORIES)))
    acc_mean = np.zeros(len(centres))

    for b in range(len(centres)):
        sel = valid & (idx == b)
        n = int(sel.sum())
        n_per_bin[b] = n
        if n == 0:
            continue
        means[b] = counts[sel].mean(axis=0)
        errs[b] = counts[sel].std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else 0.0
        acc_mean[b] = acceptance[sel].mean()

    return {
        "centres": centres,
        "means": means,
        "errs": errs,
        "acceptance": acc_mean,
        "n_per_bin": n_per_bin,
    }


def summarize(data: dict[str, Any], prof: dict[str, np.ndarray]) -> dict[str, Any]:
    """Build the JSON summary of rates and the pile-up profile.

    Args:
        data: Output of :func:`collect`.
        prof: Output of :func:`profile_vs_mu`.

    Returns:
        JSON-serializable summary dict.
    """
    totals = data["counts"].sum(axis=0)
    n_reco = int(totals.sum())
    rates = {
        c: (float(totals[i]) / n_reco if n_reco else 0.0)
        for i, c in enumerate(CATEGORIES)
    }
    return {
        "n_events": int(len(data["mu"])),
        "min_reco_ntracks": int(data["min_reco_ntracks"]),
        "beam_sigma_z_mm": float(data["beam_sigma_z"]),
        "expected_dz_plateau_sigma_mm": float(np.sqrt(2) * data["beam_sigma_z"]),
        "n_reco_vertices": n_reco,
        "n_truth_vertices": int(data["n_truth"].sum()),
        "totals": {c: int(totals[i]) for i, c in enumerate(CATEGORIES)},
        "rates": rates,
        "mean_reco_per_event": n_reco / max(len(data["mu"]), 1),
        "profile": {
            "mu": prof["centres"].tolist(),
            "n_events_per_bin": prof["n_per_bin"].tolist(),
            "acceptance": prof["acceptance"].tolist(),
            **{c: prof["means"][:, i].tolist() for i, c in enumerate(CATEGORIES)},
        },
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", required=True, help="recoTracks_incamvfassoc.h5")
    p.add_argument("--root", required=True, help="ATLAS_PVFinderData_TruthMatched.root")
    p.add_argument("--max-events", type=int, default=12000)
    p.add_argument(
        "--min-reco-ntracks",
        type=int,
        default=2,
        help="ATLAS requires >=2 tracks for a reconstructed PV (default: 2)",
    )
    p.add_argument("--mu-min", type=int, default=0)
    p.add_argument("--mu-max", type=int, default=80)
    p.add_argument("--dz-range", type=float, default=8.0, help="mm, matches Fig 25(b)")
    p.add_argument("--dz-bin", type=float, default=0.2, help="mm")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = collect(
        args.h5,
        args.root,
        args.max_events,
        args.min_reco_ntracks,
        args.dz_range,
        args.dz_bin,
    )
    prof = profile_vs_mu(data, args.mu_min, args.mu_max)
    summary = summarize(data, prof)

    np.savez_compressed(
        out / "fig25_data.npz",
        mu=data["mu"],
        counts=data["counts"],
        acceptance=data["acceptance"],
        n_truth=data["n_truth"],
        dz_hist=data["dz_hist"],
        dz_edges=data["dz_edges"],
        n_pairs_total=data["n_pairs_total"],
        beam_sigma_z=data["beam_sigma_z"],
    )
    with open(out / "fig25_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"\nEvents: {summary['n_events']}   AMVF vertices: {summary['n_reco_vertices']}"
    )
    print(f"Truth interactions (>=2 reco tracks): {summary['n_truth_vertices']}")
    print(
        f"Beamspot sigma_z = {summary['beam_sigma_z_mm']:.1f} mm  ->  "
        f"expected dz plateau sigma = {summary['expected_dz_plateau_sigma_mm']:.1f} mm"
    )
    for c in CATEGORIES:
        print(f"  {c:<8s} {summary['totals'][c]:>8d}   {summary['rates'][c]:.4f}")
    print(f"\nWrote {out}/fig25_data.npz and fig25_summary.json")


if __name__ == "__main__":
    main()
