"""Write numbers.tex for the EB deck from the saved result files.

Every result number on a slide of presentations/mattia/11_13_2026_EB/ is a
LaTeX macro defined here, with its source file and key. Regenerate after any
recompute or model freeze:

    python scripts/eb_numbers.py \
        --out presentations/mattia/11_13_2026_EB/numbers.tex

To switch checkpoints or working points, edit SOURCES: one block per group
(density model, GNN, low pile-up). Values not yet in a result file are in
MANUAL with the document they come from, written with a "% MANUAL" comment.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs"

# SOURCES: one block per group. Change these when a model is re-frozen.
FIN = OUT / "11_12_2026_output/hllhc_final"
SOURCES: dict[str, dict] = {
    "density": {
        "checkpoint": "s8 (hllhc_alleta_v6_s8_1p1ep_phase2_step_450000), FROZEN 12 Nov 2026",
        "window": "0.17 mm (HS 0.1 mm), regime hllhc",
        "peaks": "amplitude 0.01, width >= 3 bins, integral 0.184, height 0.0184, centroid +-3",
        "split": FIN / "h1_vertex_finding/split_inclusive/results.json",
        "sigma": FIN / "h1_vertex_finding/compare_r16443/compare.json",
        "small": OUT
        / "11_12_2026_output/freeze_candidates/compare_r16443/compare.json",
        "small_b": OUT
        / "11_12_2026_output/freeze_candidates/compare_r16638/compare.json",
        "s8v6": FIN / "s8_vs_v6/compare_r16443.json",
        "s8v6_b": FIN / "s8_vs_v6/compare_r16638.json",
        "cpu": FIN / "cpu_timing/cpu_timing.json",
        "cpu_nov9": OUT / "11_09_2026_output/cpu_bench_packed/cpu_bench.json",
    },
    "gnn": {
        "checkpoint": "GNN v2 epoch 527 (ttva_gat_alleta_k20_fullmu_v2_150k), FROZEN",
        "t": "0.97; vertices with >= 2 tracks for both algorithms",
        "chain": FIN / "chain/chain_headline.json",
        "extras": FIN / "chain_extras/chain_extras_summary.json",
        "recon": OUT
        / "11_09_2026_output/hs_selection_reconciliation/reconciliation.json",
    },
    "lowpu": {
        "checkpoint": "Run 2 reproduction model (reproduction_ep300), integral 0.50",
        "datamc": OUT / "11_09_2026_output/data_mc_validation/results.json",
        "r2r4": OUT / "11_09_2026_output/run2_vs_run4_resolution/results.json",
        "target": OUT / "11_11_2026_output/target_readout/results.json",
    },
}


class Num(NamedTuple):
    macro: str
    group: str
    path: str  # key into SOURCES[group]
    keys: str  # "/"-separated key path; integers index lists
    scale: float = 1.0
    fmt: str = "{:.1f}"
    note: str = ""


@lru_cache(maxsize=None)
def _load(path: Path) -> object:
    with open(path) as f:
        return json.load(f)


def _get(obj: object, keys: str) -> float:
    for k in keys.split("/"):
        obj = obj[int(k)] if isinstance(obj, list) else obj[k]  # type: ignore[index]
    return float(obj)  # type: ignore[arg-type]


def numbers() -> list[Num]:
    w, h = "windows/r16443@0.17", "windows/r16443@0.1"
    p = "headline/PU200 (185-215)"
    s8, am = "arms/s8@common", "arms/AMVF"
    ms = "MEASURED/density_onnx/ms_per_event"
    z0 = "bins/PU200 (185-215)/arms/min2/clean_per_truth/value"
    mt = "matching/fitted_sample"
    # fmt: off
    return [
        # HL-LHC vertex finding (density model s8, before the GNN)
        Num("effPVF", "density", "split", f"{w}/pvf/eff", 100),
        Num("effAMVF", "density", "split", f"{w}/amvf/eff", 100),
        Num("effDiff", "density", "split", f"{w}/diff/eff", 100, "{:.2f}"),
        Num("effDiffErr", "density", "split", f"{w}/diff/eff_err", 100, "{:.2f}"),
        Num("effHSwPVF", "density", "split", f"{h}/pvf/eff", 100),
        Num("effHSwAMVF", "density", "split", f"{h}/amvf/eff", 100),
        Num("effHSwDiff", "density", "split", f"{h}/diff/eff", 100, "{:.2f}"),
        Num("fakePVF", "density", "split", f"{w}/pvf/fake_per_evt"),
        Num("fakeAMVF", "density", "split", f"{w}/amvf/fake_per_evt"),
        Num("splitPVF", "density", "split", f"{w}/pvf/split_per_evt", 1, "{:.2f}"),
        Num("splitAMVF", "density", "split", f"{w}/amvf/split_per_evt", 1, "{:.2f}"),
        Num("truthPerEvt", "density", "split", f"{w}/pvf/truth_per_evt"),
        Num("recoPVF", "density", "split", f"{w}/pvf/reco_per_evt"),
        Num("recoAMVF", "density", "split", f"{w}/amvf/reco_per_evt"),
        Num("nEvtPU", "density", "split", f"{w}/n_events", 1, "{:,.0f}"),
        Num("nTruthPU", "density", "split", f"{w}/totals/pvf/n_truth", 1, "{:,.0f}"),
        Num("fakeFracPVF", "density", "sigma", f"{s8}/fake_frac", 100),
        Num("fakeFracAMVF", "density", "sigma", f"{am}/fake_frac", 100),
        Num("sigPVF", "density", "sigma", f"{s8}/sigma_mm", 1, "{:.3f}",
            "fitted on the counted peaks (one threshold set)"),
        Num("sigPVFErr", "density", "sigma", f"{s8}/sigma_fit_err_mm", 1, "{:.3f}"),
        Num("sigAMVF", "density", "sigma", f"{am}/sigma_mm", 1, "{:.3f}"),
        Num("sigAMVFErr", "density", "sigma", f"{am}/sigma_fit_err_mm", 1, "{:.3f}"),
        Num("pSixEightPVF", "density", "sigma", f"{s8}/p68_um", 1, "{:.1f}"),
        Num("pSixEightAMVF", "density", "sigma", f"{am}/p68_um", 1, "{:.1f}"),
        Num("hsRecoPVF", "density", "sigma", f"{s8}/hs_reco_eff", 100),
        Num("hsRecoAMVF", "density", "sigma", f"{am}/hs_reco_eff", 100),
        Num("meanMu", "gnn", "chain", f"{p}/mean_mu"),
        # small models (CPU) against v6, at matched fake rate; r16443 = A, r16638 = B
        Num("sFiveDEffA", "density", "small", "arms/s5@matched/d_eff_pt", 1, "{:.2f}"),
        Num("sFiveDEffB", "density", "small_b", "arms/s5@matched/d_eff_pt", 1, "{:.2f}"),
        Num("sFiveDFake", "density", "small", "arms/s5@matched/d_fp", 1, "{:.2f}"),
        Num("sEightDEffA", "density", "s8v6", "s8/d_eff_pt", 1, "{:.2f}",
            "frozen s8 cuts 0.184/0.0184 vs v6, same events"),
        Num("sEightDEffB", "density", "s8v6_b", "s8/d_eff_pt", 1, "{:.2f}"),
        Num("sEightDFake", "density", "s8v6", "s8/d_fp", 1, "{:.2f}"),
        Num("cpuVsixPad", "density", "cpu", f"{ms}/v6_padded/median", 1, "{:,.0f}",
            "12 Nov, interleaved, 1 thread"),
        Num("cpuVsixPack", "density", "cpu", f"{ms}/v6_packed/median", 1, "{:,.0f}"),
        Num("cpuSEightPack", "density", "cpu", f"{ms}/s8_packed/median", 1, "{:,.0f}"),
        Num("cpuSFivePack", "density", "cpu_nov9", "3/median_ms", 1, "{:,.0f}",
            "9 Nov invocation (s5 not re-measured)"),
        Num("cpuVsixPadNovNine", "density", "cpu_nov9", "0/median_ms", 1, "{:,.0f}"),
        Num("spdSEight", "density", "cpu",
            "MEASURED/density_onnx/speedup_vs_v6_padded_paired/s8_packed/median", 1, "{:.1f}"),
        Num("spdVsixPack", "density", "cpu",
            "MEASURED/density_onnx/speedup_vs_v6_padded_paired/v6_packed/median", 1, "{:.1f}"),
        Num("cpuGNNsneezy", "density", "cpu",
            "MEASURED/chain_stages_torch/ms_per_event/gnn_v2e527", 1, "{:.0f}"),
        Num("projSEight", "density", "cpu",
            "PROJECTED/athena_equivalent_chain/s8_packed/central/total_ms", 1, "{:.0f}",
            "Athena v6 CPU inference / measured s8 speed-up + other steps + GNN"),
        Num("projSEightPad", "density", "cpu",
            "PROJECTED/athena_equivalent_chain/s8_padded/central/total_ms", 1, "{:.0f}"),
        # after the GNN: t = 0.97, vertices with >= 2 tracks for both algorithms
        Num("cleanTruthGNN", "gnn", "chain", f"{p}/clean_per_truth/gnn", 100),
        Num("cleanTruthAMVF", "gnn", "chain", f"{p}/clean_per_truth/amvf", 100),
        Num("cleanTruthZnear", "gnn", "extras", f"{z0}/2", 100),
        Num("cleanTruthZsig", "gnn", "extras", f"{z0}/3", 100),
        Num("cleanHSGNN", "gnn", "chain", f"{p}/hs_clean/gnn", 100),
        Num("cleanHSAMVF", "gnn", "chain", f"{p}/hs_clean/amvf", 100),
        Num("hsTrkGNN", "gnn", "chain", f"{p}/hs_track_frac/gnn", 100),
        Num("hsTrkAMVF", "gnn", "chain", f"{p}/hs_track_frac/amvf", 100),
        Num("hsPurGNN", "gnn", "chain", f"{p}/hs_vertex_purity/gnn", 100),
        Num("hsPurAMVF", "gnn", "chain", f"{p}/hs_vertex_purity/amvf", 100),
        Num("hsSelGNN", "gnn", "chain", f"{p}/hs_selection/gnn_rule", 100, "{:.1f}",
            "Athena 24.0 rule, 3D adaptive-fit positions"),
        Num("hsSelAMVF", "gnn", "chain", f"{p}/hs_selection/amvf_stored", 100, "{:.1f}",
            "AMVF stored primary"),
        Num("hsSelAMVFrule", "gnn", "chain", f"{p}/hs_selection/amvf_rule", 100, "{:.1f}",
            "AMVF with the identical rule (our code)"),
        Num("hsSelAMVFrerank", "gnn", "recon", "pu200_185_200/flags/rerank_plurality/value", 100),
        Num("hsResGNN", "gnn", "chain", f"{p}/hs_resolution_um/gnn", 1, "{:.1f}",
            "3D adaptive beam-spot fit on the GNN tracks at t"),
        Num("hsResGNNErr", "gnn", "chain", f"{p}/hs_resolution_um/gnn_err", 1, "{:.1f}"),
        Num("hsResAMVF", "gnn", "chain", f"{p}/hs_resolution_um/amvf_common", 1, "{:.1f}"),
        Num("hsResAMVFErr", "gnn", "chain", f"{p}/hs_resolution_um/amvf_own_err", 1, "{:.1f}"),
        Num("hsResNcommon", "gnn", "chain", f"{p}/hs_resolution_um/n_common", 1, "{:,.0f}"),
        Num("trkFOneGNN", "gnn", "chain", f"{p}/trk_f1/gnn", 100, "{:.1f}"),
        Num("trkFOneAMVF", "gnn", "chain", f"{p}/trk_f1/amvf", 100, "{:.1f}"),
        Num("nOneTrkDropped", "gnn", "chain", f"{p}/table4/one_track_candidates_dropped", 1 / 1920, "{:.1f}",
            "PV-Finder candidates with exactly one GNN track, per PU200 event"),
        # low pile-up (Run 2 reproduction model)
        Num("sigMC", "lowpu", "datamc", "sigma/mc/fit", 1, "{:.3f}"),
        Num("sigRunTwo", "lowpu", "datamc", "sigma/run2/fit", 1, "{:.3f}"),
        Num("sigRunThree", "lowpu", "datamc", "sigma/run3/fit", 1, "{:.3f}"),
        Num("effMC", "lowpu", "datamc", f"{mt}/mc_truth_ref/recovery", 100),
        Num("noRefMCtruth", "lowpu", "datamc", f"{mt}/mc_truth_ref/no_ref_per_evt", 1, "{:.2f}"),
        Num("noRefMCamvf", "lowpu", "datamc", f"{mt}/mc_amvf_ref/no_ref_per_evt", 1, "{:.2f}"),
        Num("noRefRunTwo", "lowpu", "datamc", f"{mt}/run2_data_amvf_ref/no_ref_per_evt", 1, "{:.2f}"),
        Num("noRefRunThree", "lowpu", "datamc", f"{mt}/run3_data_amvf_ref/no_ref_per_evt", 1, "{:.2f}"),
        Num("sigMCAMVF", "lowpu", "r2r4", "sigma/run2/amvf/sigma_mm", 1, "{:.3f}"),
        Num("sigMCPVFr", "lowpu", "r2r4", "sigma/run2/pvf_int050_h000_full/sigma_mm", 1, "{:.3f}"),
        Num("ratioRunTwo", "lowpu", "r2r4", "sigma/run2/ratio_amvf_over_pvf/ratio", 1, "{:.2f}"),
        Num("tgtSigRunTwo", "lowpu", "target", "run2/target_run2cfg/sigma/sigma_mm", 1, "{:.3f}"),
        Num("tgtSigHL", "lowpu", "target", "hl_pu200/target_notecfg/sigma/sigma_mm", 1, "{:.3f}"),
        Num("tgtSigHLwone", "lowpu", "target", "hl_pu200/target_notecfg_w1/sigma/sigma_mm", 1, "{:.3f}"),
        Num("tgtSigHLfine", "lowpu", "target", "hl_pu200/target_notecfg_w1_g002/sigma/sigma_mm", 1, "{:.3f}"),
        Num("idealPURunTwo", "lowpu", "target", "ideal_precision_um/run2/pu_median"),
        Num("idealPUHL", "lowpu", "target", "ideal_precision_um/hl/pu_median"),
    ]


# Values with no result file yet: (macro, value, source, group).
MANUAL: list[tuple[str, str, str, str]] = [
    (
        "nTrainEvt",
        "5.17",
        "note Sec. 3 / v6 config: million training events (s8: same data)",
        "density",
    ),
    (
        "nParamVsix",
        "3.55",
        "v6 parameter count 3,547,993 (cpu_bench.json), millions",
        "density",
    ),
    (
        "athCPUms",
        "2401",
        "note Table tab:athena-timing-breakdown, CPU total, Xeon 4216 (v6)",
        "density",
    ),
    (
        "athGPUms",
        "76",
        "note Table tab:athena-timing-breakdown, GPU total, T4 (v6)",
        "density",
    ),
    (
        "athCPUinf",
        "2189",
        "note Table tab:athena-timing-breakdown, CPU inference (v6 padded)",
        "density",
    ),
    (
        "athGNNcpu",
        "182",
        "note Table tab:athena-timing-breakdown, CPU track association",
        "gnn",
    ),
    (
        "athAMVF",
        "650",
        "read off note Fig. athena_cpu_gpu_amvf at mu = 200 (600-750 ms)",
        "density",
    ),
    (
        "athGPUfactor",
        "9",
        "note Sec. 5.1.2: GPU chain vs AMVF CPU at mu = 200",
        "density",
    ),
    (
        "projSFive",
        "555",
        "9 Nov projection 2189/6.4 + 31 + 182 ms (cpu_small_model_2026-11-09.md)",
        "density",
    ),
    (
        "gnnParams",
        "39,521",
        "GNN v2 e527 parameter count (counted from the checkpoint)",
        "gnn",
    ),
    (
        "gnnTrainEvt",
        "149,820",
        "GNN v2 training events (gnn_full_pileup_training_v2_2026-11-11.md)",
        "gnn",
    ),
]
# fmt: on


def _regime_macros() -> dict[str, str]:
    """Working-point macros straight from pv_finder.utils.working_points."""
    import sys

    sys.path.insert(0, str(REPO / "src"))
    from pv_finder.utils.working_points import get_regime

    r = get_regime("hllhc")
    r.require_frozen()
    return {
        "windowMM": f"{r.match_window_mm:g}",
        "hsWindowMM": f"{r.hs_match_window_mm:g}",
        "gnnT": f"{r.gnn.t:g}",
        "intThr": f"{r.peaks.integral:g}",
        "heightThr": f"{r.peaks.min_height:g}",
        "ampThr": f"{r.peaks.amplitude:g}",
        "minWidth": f"{r.peaks.min_width:d}",
        "centroidHW": f"{r.peaks.centroid_halfwidth:d}",
    }


def _derived(v: dict[str, float]) -> dict[str, str]:
    return {
        # s5 was not re-measured on 12 Nov: its speed-up is from the 9 Nov invocation
        "spdSFive": f"{v['cpuVsixPadNovNine'] / v['cpuSFivePack']:.1f}",
        "cleanRatio": f"{v['cleanTruthGNN'] / v['cleanTruthAMVF']:.1f}",
        "sigRatioHL": f"{v['sigAMVF'] / v['sigPVF']:.2f}",
        "pvfOverTgtHL": f"{v['sigPVF'] / v['tgtSigHL']:.1f}",
        "pvfOverTgtRunTwo": f"{v['sigMCPVFr'] / v['tgtSigRunTwo']:.1f}",
    }


def _tex(val: str) -> str:
    return val.replace(",", "{,}")


def build() -> str:
    out = [
        "% numbers.tex: GENERATED by scripts/eb_numbers.py. Do not edit by hand.",
        "%",
    ]
    for grp, src in SOURCES.items():
        out.append(f"% [{grp}] checkpoint: {src['checkpoint']}")
        out += [f"%     {k}: {src[k]}" for k in ("window", "peaks", "t") if k in src]
    out.append("%")
    raw: dict[str, float] = {}
    for n in numbers():
        path = SOURCES[n.group][n.path]
        val = _get(_load(path), n.keys) * n.scale
        raw[n.macro] = val
        note = f" ({n.note})" if n.note else ""
        out.append(f"% {n.group}: {Path(path).relative_to(REPO)} :: {n.keys}{note}")
        out.append(f"\\newcommand{{\\{n.macro}}}{{{_tex(n.fmt.format(val))}}}")
    for macro, val in _derived(raw).items():
        out.append(f"% derived from the macros above ({macro})")
        out.append(f"\\newcommand{{\\{macro}}}{{{val}}}")
    for macro, val in _regime_macros().items():
        out.append(f"% REGIME hllhc (pv_finder.utils.working_points): {macro}")
        out.append(f"\\newcommand{{\\{macro}}}{{{val}}}")
    for macro, val, src, grp in MANUAL:
        out.append(f"% MANUAL [{grp}]: {src}")
        out.append(f"\\newcommand{{\\{macro}}}{{{_tex(val)}}}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.write_text(build())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
