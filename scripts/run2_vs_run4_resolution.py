"""Run 2 MC vs HL-LHC PU200: why is the PV-Finder / AMVF sigma_vtx-vtx gap smaller?

Two stages, both CPU only, no inference:

``build``
    Aligned per-event caches (``run2_cache.npz``, ``hl_cache.npz``) plus a
    track sample (``tracks_*.npz``).  Proves the Run 2 h5 -> ROOT map and the
    HL-LHC pkl/ROOT alignment, and recovers the unsaved integral-0.5 HL-LHC
    peak lists (``pv_finder.diagnostics.run2_vs_run4.recover``).
``analyze``
    Every metric on identical events with identical code: sigma_vtx-vtx
    (``amvf_fair_comparison.scan.fit_sigma_vtx_vtx`` and verified histogram
    re-implementations for the variations), core p68 position resolution,
    ITk-style HS resolution (``itk_resolution``), methodological scans, and
    the figures.  Writes ``results.json``.

Usage (from the repo root, venv active, inside tmux)::

    python -u scripts/run2_vs_run4_resolution.py build
    python -u scripts/run2_vs_run4_resolution.py analyze
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pv_finder.diagnostics.run2_vs_run4 import samples  # noqa: E402

OUT = Path("outputs/11_09_2026_output/run2_vs_run4_resolution")
RUN2_PKL = "outputs/04_16_2026_output/reproduction_ep300_mc/eval_results.pkl"
RUN2_ROOT = "data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root"
QIBIN = "configs/qibin_test_main_indices_v2.p"
HL_ROOT = (
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"
)
HL_PKL_CEN = "outputs/08_05_2026_output/eval_v6_mu_window/r16443/eval_results.pkl"
HL_PKL_FULL = "outputs/08_04_2026_output/eval_v6_heldout/r16443/eval_results.pkl"


def build() -> None:
    """Write the three caches."""
    print("[build] Run 2", flush=True)
    m = samples.build_run2(RUN2_PKL, RUN2_ROOT, QIBIN, OUT / "run2_cache.npz")
    print(m["alignment_proof"], flush=True)
    print("[build] HL-LHC", flush=True)
    m = samples.build_hl(HL_PKL_CEN, HL_PKL_FULL, HL_ROOT, 25000, OUT / "hl_cache.npz")
    print(m["recovery"], flush=True)
    print("[build] tracks", flush=True)
    for tag, root, n in (("run2", RUN2_ROOT, 2000), ("hl", HL_ROOT, 2000)):
        t = samples.read_tracks(root, n)
        np.savez_compressed(OUT / f"tracks_{tag}.npz", **t)
        print(f"  {tag}: {len(t['pt']):,} tracks", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("build", "analyze"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        build()
    else:
        from pv_finder.diagnostics.run2_vs_run4.analysis import run_all

        run_all(OUT)


if __name__ == "__main__":
    main()
