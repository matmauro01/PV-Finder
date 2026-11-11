#!/usr/bin/env python3
"""ATLAS-style HS resolution vs HS track multiplicity for the position estimators.

Reads ``results.json`` from ``scripts/hs_position_estimators.py``.

    PYTHONPATH=src venv/bin/python scripts/hs_position_estimators_plot.py \
        --dir outputs/11_11_2026_output/hs_position_estimators --best robseed_d0_t098
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from gnn.diagnostics.plot_style import (  # noqa: E402
    ALGO_COLORS,
    atlas_label,
    save_figure,
    use_atlas_style,
)


def _col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([np.nan if r[key] is None else r[key] for r in rows], float)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", required=True, type=Path)
    p.add_argument("--best", required=True)
    args = p.parse_args()
    res = json.loads((args.dir / "results.json").read_text())
    use_atlas_style()
    fig, (ax, rx) = plt.subplots(
        2, 1, figsize=(8, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    series = (
        ("centroid", "PV-Finder, histogram peak", ALGO_COLORS["retrained"], "o", False),
        (
            args.best,
            "PV-Finder + GNN, track refit",
            ALGO_COLORS["retrained"],
            "o",
            True,
        ),
        ("amvf", "AMVF", ALGO_COLORS["amvf"], "s", True),
    )
    ref = res["vs_ntrk"]["amvf"]
    ry, re_ = 1e3 * _col(ref, "rms"), 1e3 * _col(ref, "rms_err")
    for n, (key, label, color, marker, filled) in enumerate(series):
        rows = res["vs_ntrk"][key]
        lo, hi = _col(rows, "lo"), _col(rows, "hi")
        x = 0.5 * (lo + hi) + (n - 1) * 0.08 * (hi - lo)
        y, e = 1e3 * _col(rows, "rms"), 1e3 * _col(rows, "rms_err")
        ax.errorbar(
            x, y, yerr=e, fmt=marker, color=color, mfc=color if filled else "white",
            label=label, capsize=0, lw=1.5, ms=7,
        )  # fmt: skip
        if key != "amvf":
            ratio = y / ry
            rx.errorbar(
                x, ratio, yerr=ratio * np.hypot(e / y, re_ / ry), fmt=marker,
                color=color, mfc=color if filled else "white", lw=1.5, ms=6,
            )  # fmt: skip
    rx.axhline(1.0, color="grey", ls="--", lw=1)
    rx.set_ylim(0.6, 1.6)
    rx.set_ylabel("PVF / AMVF", fontsize="small")
    rx.set_xlabel("Number of tracks of the hard-scatter vertex")
    ax.set_ylabel(r"HS long. position resolution [$\mu$m]", fontsize="small")
    ax.set_ylim(0, 35)
    ax.set_xlim(0, 130)
    fig.align_ylabels([ax, rx])
    n = res["n_common"]
    atlas_label(
        ax,
        desc=rf"$t\bar{{t}}$, $\sqrt{{s}}=14$ TeV, ITk, $\langle\mu\rangle=200$"
        f"\n{n} events, HS reconstructed by both",
        desc_xy=(0.05, 0.88),
    )
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    save_figure(fig, args.dir, "hs_resolution_vs_ntrk_estimators")


if __name__ == "__main__":
    main()
