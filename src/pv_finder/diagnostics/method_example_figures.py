"""Example figures for the technical-note Method chapter (fig:examples).

Panel (a): an analytical KDE (KDE-A) for one mu~60 sub-event with the truth
PVs it encodes. Panel (b): an end-to-end predicted histogram at PU200 with
the peak-finder vertices, using the cached v4b histograms from the gap
decomposition (events 28500+ of the SingleLep file).

Run from the repo root:
    python -u src/pv_finder/diagnostics/method_example_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import uproot

from gnn.diagnostics.plot_style import atlas_label, use_atlas_style
from pv_finder.utils.peak_finding import pv_locations_updated_res

H5_MU60 = (
    "/share/lazy/qibinlei/"
    "recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5"
)
HISTS_NPZ = "outputs/07_14_2026_ttva_gap/histograms_300ev.npz"
ROOT_PU200 = (
    "data/run4/Run4_MC21_ITk/"
    "ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root"
)
ROOT_START = 28500

Z_MIN, BIN_W, N_BINS = -240.0, 0.04, 12000
PAD = -999999.0


def pick_mu60_subevent(
    h5f: h5py.File, event: int, target: int = 7
) -> tuple[int, np.ndarray]:
    """Return the sub-event of `event` with a PV count nearest `target`."""
    pvs = h5f["pv"][event]
    pvs = pvs[pvs > PAD + 1]
    best_j, best_pv = 0, np.array([])
    for j in range(12):
        lo, hi = Z_MIN + 40.0 * j, Z_MIN + 40.0 * (j + 1)
        sel = pvs[(pvs >= lo) & (pvs < hi)]
        if abs(len(sel) - target) < abs(len(best_pv) - target):
            best_j, best_pv = j, sel
    return best_j, best_pv


def panel_kde(h5f: h5py.File, event: int, out: Path) -> None:
    j, pv_z = pick_mu60_subevent(h5f, event)
    kde = h5f["kde_split"][12 * event + j, 0, :]
    lo = Z_MIN + 40.0 * j
    z = lo + BIN_W * (np.arange(1000) + 0.5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z, kde, color="#0072B2", lw=1.6, label="Analytical KDE (KDE-A)")
    for i, v in enumerate(pv_z):
        ax.axvline(
            v,
            color="#D55E00",
            ls="--",
            lw=1.3,
            label="Truth PV" if i == 0 else None,
        )
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("KDE amplitude [a.u.]")
    ax.set_xlim(lo, lo + 40.0)
    ax.set_ylim(0, 1.45 * kde.max())
    ax.legend(loc="upper right")
    atlas_label(ax, desc=r"$t\bar{t}$ MC, $\langle\mu\rangle\approx60$")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"panel a: event {event} subevent {j}, {len(pv_z)} PVs -> {out}")


def panel_e2e(idx: int, z_window: tuple[float, float], out: Path) -> None:
    hist = np.load(HISTS_NPZ)["hists"][idx]
    tree = uproot.open(ROOT_PU200)["PVFinderData"]
    arr = tree.arrays(
        ["TruthVertex_z", "TruthVertex_nTracks"],
        entry_start=ROOT_START + idx,
        entry_stop=ROOT_START + idx + 1,
    )
    tz = np.asarray(arr["TruthVertex_z"][0])
    tn = np.asarray(arr["TruthVertex_nTracks"][0])
    tz = tz[tn >= 2]

    pz, _, _, _ = pv_locations_updated_res(hist, 0.01, 0.40, 3, 0.03)
    z = Z_MIN + BIN_W * (np.arange(N_BINS) + 0.5)

    lo, hi = z_window
    m = (z >= lo) & (z <= hi)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z[m], hist[m], color="#0072B2", lw=1.6, label="Predicted histogram")
    for i, v in enumerate(tz[(tz >= lo) & (tz <= hi)]):
        ax.axvline(
            v,
            color="#D55E00",
            ls="--",
            lw=1.3,
            label=r"Truth PV ($n_\mathrm{trk}\geq2$)" if i == 0 else None,
        )
    sel = pz[(pz >= lo) & (pz <= hi)]
    heights = hist[np.clip(((sel - Z_MIN) / BIN_W).astype(int), 0, N_BINS - 1)]
    ax.plot(
        sel,
        heights,
        "o",
        ms=8,
        mfc="none",
        mew=1.8,
        color="#009E73",
        label="Peak-finder vertex",
    )
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("Predicted vertex density")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.45 * hist[m].max())
    ax.legend(loc="upper right")
    atlas_label(ax, desc=r"HL-LHC $t\bar{t}$, $\langle\mu\rangle=200$, ITk layout")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(
        f"panel b: event {ROOT_START + idx}, {len(sel)} peaks / "
        f"{len(tz[(tz >= lo) & (tz <= hi)])} truth in window -> {out}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mu60-event", type=int, default=48460)
    ap.add_argument("--pu200-idx", type=int, default=0)
    ap.add_argument("--z-lo", type=float, default=-12.0)
    ap.add_argument("--z-hi", type=float, default=12.0)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("outputs/07_15_2026_note_figs")
    )
    args = ap.parse_args()

    use_atlas_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(H5_MU60, "r") as h5f:
        panel_kde(h5f, args.mu60_event, args.out_dir / "method_example_kde.png")
    panel_e2e(
        args.pu200_idx,
        (args.z_lo, args.z_hi),
        args.out_dir / "method_example_e2e.png",
    )


if __name__ == "__main__":
    main()
