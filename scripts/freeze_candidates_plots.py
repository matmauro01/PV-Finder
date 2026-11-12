"""Plots for the HL-LHC density-model freeze comparison (step 2a).

1. Pairwise Delta-z density of reconstructed vertices, per candidate at the
   common HL-LHC cuts and AMVF, from pair_hists.npz written by
   scripts/freeze_candidates_compare.py (same events, same peaks as the
   counting).
2. A spot check of five PU200 events: the predicted vertex density of every
   candidate in a 12 mm window around the true hard-scatter vertex, with
   the truth vertices (n_trk >= 2) and each model's peaks marked.

Example:
    python scripts/freeze_candidates_plots.py --tag r16443 \
        --compare outputs/11_12_2026_output/freeze_candidates/compare_r16443 \
        --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
        --model v6=outputs/11_12_2026_output/freeze_candidates/v6/r16443 \
        --model s5=outputs/11_12_2026_output/freeze_candidates/s5/r16443 \
        --model s8=outputs/11_12_2026_output/freeze_candidates/s8/r16443
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import matplotlib.pyplot as plt  # noqa: E402
from freeze_candidates_compare import read_root  # noqa: E402
from matched_fp_scan import load  # noqa: E402

from gnn.diagnostics.plot_style import (  # noqa: E402
    atlas_label,
    save_figure,
    use_atlas_style,
)

Z_MIN, BIN_WIDTH = -240.0, 0.04
COLORS = {"v6": "black", "s5": "tab:blue", "s5_2p2": "tab:cyan", "s8": "tab:red",
          "AMVF": "tab:orange"}  # fmt: skip
DESC = r"$t\bar{t}$, $\sqrt{s}$ = 14 TeV, $185 \leq \mu \leq 215$"


def plot_pairs(compare_dir: Path, tag: str) -> None:
    """Pairwise Delta-z density per event, |Delta z| < 3 mm."""
    d = np.load(compare_dir / "pair_hists.npz")
    res = json.loads((compare_dir / "compare.json").read_text())
    n = res["n_events"]
    edges = d["edges"]
    ctr = 0.5 * (edges[:-1] + edges[1:])
    bw = edges[1] - edges[0]
    fig, ax = plt.subplots(figsize=(8, 6))
    for key in d.files:
        if key == "edges" or key.endswith("_matched"):
            continue
        name = key.replace("_common", "")
        arm = res["arms"][
            key.replace("_common", "@common") if name != "AMVF" else "AMVF"
        ]
        y = d[key] / (n * bw)
        ax.step(ctr, y, where="mid", color=COLORS.get(name, "gray"),
                label=rf"{name}: $\sigma_{{vtx-vtx}}$ = {1e3 * arm['sigma_mm']:.0f} "
                      rf"$\pm$ {1e3 * arm['sigma_err_mm']:.0f} $\mu$m")  # fmt: skip
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.set_xlabel(r"$\Delta z$ between reconstructed vertices [mm]")
    ax.set_ylabel(r"Vertex pairs / (event $\cdot$ mm)")
    atlas_label(ax, desc=f"{DESC}\n{n} events ({tag})")
    ax.legend(loc="upper right", fontsize="small")
    save_figure(fig, compare_dir, f"pair_dz_{tag}")
    plt.close(fig)


def plot_spot_check(runs: dict, root: str, compare_dir: Path, tag: str) -> None:
    """Five events, every model's density near the HS, truth and peaks marked."""
    names = list(runs)
    ref = runs[names[0]]
    n = len(ref["idx"])
    picks = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    hs_z, _ = read_root(root, ref["idx"][picks])
    fig, axes = plt.subplots(
        len(picks), 1, figsize=(10, 2.6 * len(picks)), sharex=False
    )
    for row, (k, z0) in enumerate(zip(picks, hs_z)):
        ax = axes[row]
        lo, hi = z0 - 6.0, z0 + 6.0
        b0, b1 = int((lo - Z_MIN) / BIN_WIDTH), int((hi - Z_MIN) / BIN_WIDTH)
        zc = Z_MIN + (np.arange(b0, b1) + 0.5) * BIN_WIDTH
        top = 0.0
        for name in names:
            h = runs[name]["hist"]
            if h is None:
                continue
            y = np.asarray(h[k])[b0:b1]
            top = max(top, float(y.max()))
            ax.plot(zc, y, color=COLORS.get(name, "gray"), lw=1, label=name)
        for j, name in enumerate(names):
            p = runs[name]["pvs"][k]
            p = p[(p > lo) & (p < hi)]
            ax.plot(p, np.full(len(p), -0.08 * (j + 1) * max(top, 1e-3)), "|",
                    color=COLORS.get(name, "gray"), ms=10, mew=2)  # fmt: skip
        t = ref["truth"][k]
        t = t[(t > lo) & (t < hi)]
        ax.vlines(t, 0, max(top, 1e-3), color="green", alpha=0.35, lw=1,
                  label="truth ($n_{trk} \\geq 2$)" if row == 0 else None)  # fmt: skip
        ax.axvline(
            z0, color="green", ls="--", lw=1.5, label="truth HS" if row == 0 else None
        )
        ax.set_xlim(lo, hi)
        ax.set_ylabel("density")
        ax.text(0.01, 0.92, f"event {ref['idx'][k]}", transform=ax.transAxes,
                fontsize="small", va="top")  # fmt: skip
        if row == 0:
            ax.legend(loc="upper right", fontsize="x-small", ncol=3)
    axes[-1].set_xlabel("z [mm] (ticks below zero: peaks, one row per model)")
    fig.tight_layout()
    save_figure(fig, compare_dir, f"spot_check_{tag}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=dir")
    args = ap.parse_args()
    use_atlas_style()
    cdir = Path(args.compare)
    plot_pairs(cdir, args.tag)
    runs = {}
    for spec in args.model:
        name, d = spec.split("=", 1)
        runs[name] = load(Path(d))
    plot_spot_check(runs, args.root, cdir, args.tag)


if __name__ == "__main__":
    main()
