"""Regenerate every paper figure from saved evaluation outputs (one command).

    source venv/bin/activate
    python -u src/pv_finder/diagnostics/paper_figures/make_paper_figures.py \
        --out-dir outputs/11_09_2026_output/paper_figures

Run by path from the repository root. No inference is run; the whole job reads
pickles, one npz and two ROOT branches pairs, and takes about a minute on one
core. ``--timing-csv`` adds Figure 7 once per-event Athena timing exists.
A ``summary.json`` with every plotted number is written next to the figures.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pv_finder.diagnostics.paper_figures import categories as cat  # noqa: E402
from pv_finder.diagnostics.paper_figures import (  # noqa: E402
    deltaz,
    diagrams,
    sources,
    style,
    timing,  # noqa: E402
)

NOTE_TIMING_PNG = (
    Path.home() / "codice/ANA-IDTR-2026-01-INT1/figures/athena_cpu_gpu_amvf.png"
)
HLLHC_LINE = "$\\sqrt{s}$ = 14 TeV, $t\\bar{t}$, ITk layout"
LOWPU_LINE = "$\\sqrt{s}$ = 13 TeV, $t\\bar{t}$, ID"
RUN3_LINE = "Data, $\\sqrt{s}$ = 13.6 TeV"
SECTIONS = ("hllhc", "lowpu", "run3", "timing")


def _fits(sample: sources.Sample) -> dict[str, Any]:
    return {arm: {"sigma_mm": r.sigma_mm, "sigma_err_mm": r.sigma_err_mm,
                  "n_pairs_in_range": int(np.sum(np.abs(r.dz_mm) <= 6.0)),
                  "n_bins": r.n_bins, "params": r.params.tolist(), "note": r.note}
            for arm, r in sample.resolution.items()}  # fmt: skip


def hllhc(
    out: Path,
    summary: dict[str, Any],
    pkl: Path = sources.HLLHC_PKL,
    cache: Path = sources.HLLHC_CACHE,
) -> None:
    """Figures 3 and 4 (HL-LHC ITk simulation)."""
    full, win = sources.load_hllhc(pkl_path=pkl, cache_path=cache)
    w = win.window_mm
    every = np.ones(len(win.mu), dtype=bool)
    pvf = cat.summarize(cat.per_event_counts(win.truth_z, win.pvf_z, w), every)
    amvf = cat.summarize(cat.per_event_counts(win.truth_z, win.amvf_z, w), every)
    lines = [HLLHC_LINE, f"$185 \\leq \\mu \\leq 200$, {len(win.mu):,} events",
             f"Truth matching within {w:.2f} mm"]  # fmt: skip
    cat.plot_category_bars(pvf, amvf, lines, style.SIM_STATUS, out,
                           "fig3a_hllhc_categories_pu200",
                           reference_label="Truth vertices, $\\geq 2$ tracks")  # fmt: skip
    counts_all = cat.per_event_counts(full.truth_z, full.pvf_z, w)
    cat.plot_categories_vs_mu(
        full, counts_all, np.arange(0, 211, 10),
        [HLLHC_LINE, "PV-Finder", f"Truth matching within {w:.2f} mm"],
        style.SIM_STATUS, out, "fig3b_hllhc_categories_vs_mu",
    )  # fmt: skip
    deltaz.plot_deltaz(win.resolution, [HLLHC_LINE, "$185 \\leq \\mu \\leq 200$"],
                       style.SIM_STATUS, out, "fig4_hllhc_deltaz")  # fmt: skip
    summary["hllhc"] = {
        "sources": win.meta, "window_mm": w, "n_events_pu200": int(len(win.mu)),
        "mean_mu_pu200": float(win.mu.mean()), "mu_range_pu200": [float(win.mu.min()),
        float(win.mu.max())], "pvf_pu200": pvf, "amvf_pu200": amvf,
        "n_events_all_mu": int(len(full.mu)), "fits": _fits(win),
    }  # fmt: skip


def lowpu_mc(out: Path, summary: dict[str, Any]) -> None:
    """Substitute for Figures 5 and 6: 13 TeV simulation (no Run 3 MC exists here)."""
    s = sources.load_lowpu_mc()
    w = s.window_mm
    lo, hi = sources.LOWPU_MU_RANGE
    sel = (np.round(s.mu) >= lo) & (np.round(s.mu) <= hi)
    c_pvf = cat.per_event_counts(s.truth_z, s.pvf_z, w)
    c_amvf = cat.per_event_counts(s.truth_z, s.amvf_z, w)
    pvf, amvf = cat.summarize(c_pvf, sel), cat.summarize(c_amvf, sel)
    lines = [LOWPU_LINE, f"${lo:.0f} \\leq \\mu \\leq {hi:.0f}$, {int(sel.sum())} events",
             f"Truth matching within {w:.2f} mm"]  # fmt: skip
    cat.plot_category_bars(pvf, amvf, lines, style.SIM_STATUS, out,
                           "fig5a_lowpu_mc_categories",
                           reference_label="Truth vertices, $\\geq 2$ tracks")  # fmt: skip
    cat.plot_categories_vs_mu(
        s, c_pvf, np.arange(0, 83, 2),
        [LOWPU_LINE, "PV-Finder", f"Truth matching within {w:.2f} mm"],
        style.SIM_STATUS, out, "fig5b_lowpu_mc_categories_vs_mu", min_events=10,
    )  # fmt: skip
    deltaz.plot_deltaz(s.resolution, [LOWPU_LINE, f"$0 \\leq \\mu \\leq 80$, "
                       f"{len(s.mu):,} events"], style.SIM_STATUS, out,
                       "fig6_lowpu_mc_deltaz")  # fmt: skip
    summary["lowpu_mc"] = {
        "sources": s.meta, "window_mm": w, "n_events": int(len(s.mu)),
        "n_events_mu_window": int(sel.sum()), "mean_mu": float(s.mu.mean()),
        "pvf_mu_window": pvf, "amvf_mu_window": amvf,
        "pvf_all_events": cat.summarize(c_pvf, np.ones(len(s.mu), bool)),
        "fits": _fits(s),
    }  # fmt: skip


def run3_data(out: Path, summary: dict[str, Any]) -> None:
    """Data alternative for Figure 6: Run 3 collision data, AMVF on the same events."""
    s = sources.load_run3_data()
    lines = [RUN3_LINE, f"$\\langle\\mu\\rangle$ = {s.mu.mean():.1f}, "
             f"{len(s.mu):,} events"]  # fmt: skip
    deltaz.plot_deltaz(s.resolution, lines, style.DATA_STATUS, out,
                       "fig6alt_run3_data_deltaz")  # fmt: skip
    summary["run3_data"] = {"sources": s.meta, "n_events": int(len(s.mu)),
                            "mean_mu": float(s.mu.mean()), "fits": _fits(s)}  # fmt: skip


def main(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    if not args.skip_diagrams:
        diagrams.make_diagrams(out)
        print("  diagrams done")
    sections = set(args.sections)
    if "hllhc" in sections:
        hllhc(out, summary, Path(args.hllhc_pkl), Path(args.hllhc_cache))
        print("  HL-LHC done")
    if "lowpu" in sections:
        lowpu_mc(out, summary)
        print("  low-pile-up MC done")
    if "run3" in sections:
        run3_data(out, summary)
        print("  Run 3 data done")
    if "timing" in sections:
        _timing(out, args.timing_csv)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"  wrote {out}")


def _timing(out: Path, timing_csv: str | None) -> None:
    """Figure 7: per-event timing if given, else a copy of the note's PNG."""
    if timing_csv:
        series = timing.read_timing_csv(Path(timing_csv))
        timing.plot_timing(series, np.arange(0, 211, 10), ["Athena, ttbar and data"],
                           out, "fig7_timing")  # fmt: skip
    elif NOTE_TIMING_PNG.exists():
        # Only the colleague's rendered figure exists; keep it for reference.
        shutil.copyfile(NOTE_TIMING_PNG, out / "fig7_timing_existing_note_figure.png")
        print("  Figure 7: no per-event timing; copied the existing note PNG")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", default="outputs/11_09_2026_output/paper_figures")
    p.add_argument(
        "--timing-csv",
        default=None,
        help="per-event timing table (see timing.py for the columns)",
    )
    p.add_argument("--skip-diagrams", action="store_true")
    p.add_argument("--sections", nargs="+", default=list(SECTIONS),
                   choices=SECTIONS, help="figure groups to make")  # fmt: skip
    p.add_argument("--hllhc-pkl", default=str(sources.HLLHC_PKL),
                   help="HL-LHC eval_results.pkl (default: pre-freeze v6)")  # fmt: skip
    p.add_argument("--hllhc-cache", default=str(sources.HLLHC_CACHE),
                   help="fair-comparison cache built from --hllhc-pkl")  # fmt: skip
    return p.parse_args()


if __name__ == "__main__":
    main(_cli())
