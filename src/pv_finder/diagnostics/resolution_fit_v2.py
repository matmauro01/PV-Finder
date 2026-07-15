"""Background-corrected AMVF z-resolution vs truth nTracks (PU200 ITk).

Re-derives the per-multiplicity-bin resolution from the saved AMVF-truth
residual pairs (vertex_data.npz of amvf_resolution_vs_ntracks.py), fitting
each bin with a Gaussian core plus a flat background so that wrong matches
inside the 2 mm matching window no longer inflate the widths. The power law
sigma(n) = A n^-B + C is refit to the corrected widths, and the figure shows
both the refined fit and the preset actually used to build the training
targets.

Run from the repo root:
    python -u src/pv_finder/diagnostics/resolution_fit_v2.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from gnn.diagnostics.plot_style import atlas_label, use_atlas_style

NPZ_DEFAULT = "outputs/06_01_2026_output/amvf_resolution_residuals/vertex_data.npz"
PRESET = (0.17898, 0.7274, 0.0)  # hllhc preset used to build the targets

EDGES = (
    *range(2, 30),
    30,
    32,
    34,
    36,
    38,
    40,
    42,
    45,
    48,
    51,
    55,
    60,
    65,
    70,
    78,
    88,
    100,
    115,
    135,
    170,
)


def gauss_flat(x: np.ndarray, ng: float, s: float, nb: float) -> np.ndarray:
    return ng * np.exp(-0.5 * (x / s) ** 2) + nb


def sigma_gauss_flat(dz: np.ndarray) -> tuple[float, float] | None:
    """Binned Gaussian+flat fit over the full +-2 mm match window."""
    cnt, be = np.histogram(dz, bins=200, range=(-2, 2))
    ctr = 0.5 * (be[:-1] + be[1:])
    err = np.sqrt(np.maximum(cnt, 1))
    s0 = 1.4826 * np.median(np.abs(dz - np.median(dz)))
    try:
        popt, pcov = curve_fit(
            gauss_flat,
            ctr,
            cnt,
            sigma=err,
            absolute_sigma=True,
            p0=[cnt.max(), max(s0, 1e-3), max(cnt[np.abs(ctr) > 1.5].mean(), 0.1)],
            bounds=([0, 1e-4, 0], [np.inf, 1, np.inf]),
            maxfev=20000,
        )
    except RuntimeError:
        return None
    return abs(popt[1]), float(np.sqrt(np.diag(pcov))[1])


def per_bin_sigmas(ntrk: np.ndarray, dz: np.ndarray) -> dict[str, np.ndarray]:
    centers, sigmas, errs, counts = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (ntrk >= lo) & (ntrk < hi)
        if m.sum() < 200:
            continue
        res = sigma_gauss_flat(dz[m].astype(np.float64))
        if res is None:
            continue
        s, serr = res
        centers.append(float(ntrk[m].mean()))
        sigmas.append(s)
        errs.append(max(serr, 0.002 * s))
        counts.append(int(m.sum()))
    return dict(
        centers=np.array(centers),
        sigmas=np.array(sigmas),
        errs=np.array(errs),
        counts=np.array(counts),
    )


def power_law(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(x, -b) + c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=NPZ_DEFAULT)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/07_15_2026_note_figs")
    )
    args = ap.parse_args()

    d = np.load(args.npz)
    agg = per_bin_sigmas(d["truth_ntrks"], d["dz_mm"])
    c, sg, se = agg["centers"], agg["sigmas"], agg["errs"]

    popt, pcov = curve_fit(
        power_law,
        c,
        sg,
        sigma=se,
        p0=[0.15, 0.5, 0.0],
        maxfev=20000,
        absolute_sigma=True,
    )
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((sg - power_law(c, *popt)) / se) ** 2))
    print(
        f"refined fit: A={popt[0]:.4f}+-{perr[0]:.4f} mm  "
        f"B={popt[1]:.4f}+-{perr[1]:.4f}  C={1e3 * popt[2]:.2f}+-{1e3 * perr[2]:.2f} um  "
        f"chi2/ndf={chi2 / (len(c) - 3):.1f}"
    )

    use_atlas_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(
        c,
        sg,
        yerr=se,
        fmt="o",
        ms=5,
        capsize=2.5,
        elinewidth=1.2,
        color="#0072B2",
        label="AMVF$-$truth residual width (bkg-corrected)",
    )
    xs = np.linspace(2, c.max(), 500)
    ax.plot(
        xs,
        power_law(xs, *popt),
        color="#D55E00",
        lw=2,
        label=(rf"Fit: $A\,n^{{-B}}+C$, $A={popt[0]:.3f}$ mm, $B={popt[1]:.3f}$"),
    )
    ax.plot(
        xs,
        power_law(xs, *PRESET),
        color="#009E73",
        lw=2,
        ls="--",
        label=rf"Target preset: $A={PRESET[0]:.3f}$ mm, $B={PRESET[1]:.3f}$",
    )
    ax.set_xlabel(r"Truth-vertex $n_\mathrm{trk}$")
    ax.set_ylabel(r"Vertex $z$ resolution [mm]")
    ax.set_xlim(0, 1.05 * c.max())
    ax.set_ylim(0, 1.45 * sg.max())
    ax.legend(loc="upper right", fontsize="small")
    atlas_label(ax, desc=r"HL-LHC $t\bar{t}$, $\langle\mu\rangle=200$, ITk layout")
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "amvf_resolution_vs_ntracks_v2.png"
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")

    with open(args.out_dir / "resolution_fit_v2.json", "w") as fp:
        json.dump(
            dict(
                method="Gaussian core + flat background per bin, full +-2 mm window",
                fit=dict(
                    A_mm=popt[0],
                    B=popt[1],
                    C_mm=popt[2],
                    A_err_mm=perr[0],
                    B_err=perr[1],
                    C_err_mm=perr[2],
                    chi2_ndf=chi2 / (len(c) - 3),
                ),
                preset=dict(A_mm=PRESET[0], B=PRESET[1], C_mm=PRESET[2]),
                per_bin=dict(
                    centers=c.tolist(),
                    sigmas_mm=sg.tolist(),
                    errs_mm=se.tolist(),
                    counts=agg["counts"].tolist(),
                ),
            ),
            fp,
            indent=2,
        )
    print(f"wrote {args.out_dir / 'resolution_fit_v2.json'}")


if __name__ == "__main__":
    main()
