"""Re-derive the numerical claims of a slide deck from archived evaluation pickles.

Written for the 2026-08-05 audit of ``presentations/mattia/07_24_2026/slides.tex``.
Every number a deck quotes from an eval run is stored in that run's
``eval_results.pkl``; this script reads them back and, for the resolution
numbers, re-fits ``sigma_vtx_vtx`` at several binnings so that the
coarse-binning bias documented in JOURNAL 2026-07-20 and the incommensurate-
binning comb documented in ``docs/research/resolution_plot_ripple.md`` can be
measured on the exact pair arrays the deck's figures were drawn from.

Nothing here re-runs inference: it is pure post-processing of stored arrays, so
it is cheap and reproducible without a GPU.

Usage::

    PYTHONPATH=src python -u src/pv_finder/diagnostics/deck_claim_verification.py \
        --output-dir outputs/08_05_2026_output/deck_claim_audit
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

# The eval's own model grid.  Reconstructed positions are quantised on it, so a
# plot bin width that is not an integer multiple of it beats against the comb.
BIN_WIDTH_MM: float = 0.04
PAIRWISE_RANGE_MM: float = 6.0

# Binnings to scan.  60 is the historical default of run_eval_pvf.py (and of
# run_eval_pvf_run3.py before 2026-07-20); 240 is the 2026-07-20 fix, which is
# fine but *incommensurate* (0.05 mm); 300 is the current default (0.04 mm).
BINNINGS: tuple[int, ...] = (60, 120, 240, 300, 600)


def sigmoid_fit(x: np.ndarray, a: float, b: float, c: float, rcc: float) -> np.ndarray:
    """The eval's resolution fit function (``run_eval_pvf_run3.sigmoid_fit``)."""
    return a / (1.0 + np.exp(b * (rcc - np.abs(x)))) + c


def fit_sigma(dz: np.ndarray, n_bins: int) -> tuple[float, float]:
    """Fit sigma_vtx_vtx on ``dz`` at ``n_bins`` bins across +-6 mm.

    Reproduces the eval's adaptive initial guess exactly so that the numbers are
    comparable with what the eval printed.  Returns ``(sigma, err)``; ``err`` is
    ``nan`` when the fit does not converge.
    """
    edges = np.linspace(-PAIRWISE_RANGE_MM, PAIRWISE_RANGE_MM, n_bins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    cnts, _ = np.histogram(dz, bins=edges)
    base = float(np.median(cnts))
    p0 = [max(base - float(cnts.min()), 1.0), 10.0, max(base, 1.0), 0.5]
    try:
        popt, pcov = curve_fit(
            sigmoid_fit,
            ctrs,
            cnts.astype(float),
            p0=p0,
            maxfev=10000,
            bounds=([0, 0, 0, 0], [np.inf] * 4),
        )
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return float(abs(popt[3])), float(np.sqrt(np.diag(pcov))[3])


def comb_amplitude(dz: np.ndarray, n_bins: int, period_mm: float = 0.20) -> float:
    """Amplitude of a ``period_mm`` modulation of the plateau, as a fraction.

    This is the observable that is 3.8-4.0 % for the 0.05 mm binning and ~0.04 %
    for 0.04 mm bins (``docs/research/resolution_plot_ripple.md`` section 2).
    Projects the plateau counts onto one sinusoid of that period; the plateau is
    taken beyond 1.2 mm so the dip and the satellite shoulder are excluded.
    """
    edges = np.linspace(-PAIRWISE_RANGE_MM, PAIRWISE_RANGE_MM, n_bins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    cnts, _ = np.histogram(dz, bins=edges)
    sel = np.abs(ctrs) >= 1.2
    x, y = ctrs[sel], cnts[sel].astype(float)
    if len(x) < 8 or y.mean() <= 0:
        return float("nan")
    # Remove the residual slope of the plateau before projecting.
    y = y - np.polyval(np.polyfit(x, y, 1), x)
    omega = 2.0 * np.pi / period_mm
    a = 2.0 * np.mean(y * np.cos(omega * x))
    b = 2.0 * np.mean(y * np.sin(omega * x))
    return float(np.hypot(a, b) / cnts[sel].mean())


@dataclass
class PklClaim:
    """One archived eval, with what the deck quotes from it."""

    tag: str
    path: str
    note: str = ""
    deck_values: dict[str, Any] = field(default_factory=dict)


def summarise(pkl_path: Path) -> dict[str, Any]:
    """Read one eval pickle and re-derive its resolution number at each binning."""
    with pkl_path.open("rb") as fh:
        d = pickle.load(fh)

    out: dict[str, Any] = {
        "stored": {
            k: (
                float(d[k])
                if isinstance(d.get(k), (int, float, np.floating))
                else d.get(k)
            )
            for k in (
                "sigma_vtx_vtx_mm",
                "overall_efficiency",
                "fp_rate_per_evt",
                "n_events",
                "total_truth_pvs",
                "total_clean",
                "total_merged",
                "total_split",
                "total_fake",
                "mode",
                "e2e_checkpoint",
            )
            if k in d
        }
    }
    for k in ("total_clean", "total_merged", "total_split", "total_fake"):
        if k in d:
            out["stored"][k] = float(np.asarray(d[k]))

    dz = np.asarray(d.get("pairwise_dz_mm", np.empty(0)), dtype=float)
    out["n_pairs"] = int(dz.size)
    scan: dict[str, Any] = {}
    for nb in BINNINGS:
        sigma, err = fit_sigma(dz, nb)
        width = 2.0 * PAIRWISE_RANGE_MM / nb
        scan[str(nb)] = {
            "bin_width_mm": round(width, 5),
            "commensurate": bool(
                abs(width / BIN_WIDTH_MM - round(width / BIN_WIDTH_MM)) < 1e-6
                and round(width / BIN_WIDTH_MM) >= 1
            ),
            "sigma_mm": sigma,
            "sigma_err_mm": err,
            "comb_0p20mm": comb_amplitude(dz, nb),
        }
    out["binning_scan"] = scan
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-dir",
        default="outputs/08_05_2026_output/deck_claim_audit",
        help="where to write deck_claim_verification.json",
    )
    ap.add_argument(
        "--repo-root",
        default=".",
        help="repository root that the pickle paths are relative to",
    )
    args = ap.parse_args()

    root = Path(args.repo_root).resolve()
    claims = [
        PklClaim(
            "run2_mc_e2e_ep300",
            "outputs/04_16_2026_output/reproduction_ep300_mc/eval_results.pkl",
            "Deck 'Run 2 Simulation: Resolution vs AMVF' (slides.tex:539-548)",
            {"sigma_mm": 0.42, "efficiency": 0.916, "fake_per_evt": 0.42},
        ),
        PklClaim(
            "run2_data_e2e_ep300",
            "outputs/04_16_2026_output/reproduction_ep300_run2/eval_results.pkl",
            "Deck 'Real Collision Data: Run 2 & Run 3' (slides.tex:616)",
            {"sigma_mm": 0.37, "agreement": 0.953, "fake_per_evt": 1.82},
        ),
        PklClaim(
            "run3_data_e2e_ep300",
            "outputs/04_16_2026_output/reproduction_ep300_run3/eval_results.pkl",
            "Deck 'Real Collision Data: Run 2 & Run 3' (slides.tex:617)",
            {"sigma_mm": 0.35, "agreement": 0.948, "fake_per_evt": 1.96},
        ),
        PklClaim(
            "hllhc_v4b_floor0p03",
            "outputs/06_09_2026_output/eval_v4b_ep3_floor0p03/eval_results.pkl",
            "Deck HL-LHC headline table, v4b row (slides.tex:732)",
            {},
        ),
        PklClaim(
            "hllhc_v4b_gbt_r16633",
            "outputs/06_09_2026_output/eval_v4b_ep3_gbt_r16633/eval_results.pkl",
            "Deck HL-LHC headline table, +GBT row (slides.tex:733)",
            {},
        ),
        PklClaim(
            "hllhc_v4b_baseline_nofloor_0720",
            "outputs/07_20_2026_output/eval_v4b_baseline_nofloor/eval_results.pkl",
            "2026-07-20 re-derivation at the 0.223 mm window",
            {},
        ),
        PklClaim(
            "hllhc_v6_heldout_r16443_0804",
            "outputs/08_04_2026_output/eval_v6_heldout/r16443/eval_results.pkl",
            "v6 held-out, previous operating point",
            {},
        ),
        PklClaim(
            "hllhc_v6_oppoint_r16443_0805",
            "outputs/08_05_2026_output/eval_v6_operating_point/r16443/eval_results.pkl",
            "v6 held-out at the deployed operating point, 300 bins",
            {},
        ),
    ]

    results: dict[str, Any] = {}
    for c in claims:
        p = root / c.path
        if not p.exists():
            results[c.tag] = {"error": "missing", "path": c.path, "note": c.note}
            print(f"[MISS] {c.tag}: {c.path}")
            continue
        print(f"[ OK ] {c.tag}: {c.path}")
        r = summarise(p)
        r["path"] = c.path
        r["note"] = c.note
        r["deck_values"] = c.deck_values
        results[c.tag] = r
        s = r["stored"]
        print(
            f"       stored sigma={s.get('sigma_vtx_vtx_mm')} "
            f"eff={s.get('overall_efficiency')} fp={s.get('fp_rate_per_evt')} "
            f"n_events={s.get('n_events')} pairs={r['n_pairs']:,}"
        )
        for nb, v in r["binning_scan"].items():
            print(
                f"         {nb:>4} bins ({v['bin_width_mm']:.3f} mm, "
                f"{'commensurate' if v['commensurate'] else 'INCOMMENSURATE'}): "
                f"sigma={v['sigma_mm']:.4f} +/- {v['sigma_err_mm']:.4f}  "
                f"comb={100 * v['comb_0p20mm']:.2f}%"
            )

    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "deck_claim_verification.json"
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
