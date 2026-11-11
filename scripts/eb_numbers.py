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
SOURCES: dict[str, dict] = {
    "density": {
        "checkpoint": "hllhc_alleta_v6_mse_2ep_phase2_epoch_2 (v6 production)",
        "window": "0.1694 mm (to become 0.17 mm)",
        "peaks": "amplitude 0.01, width >= 3 bins, integral 0.30, height 0.03, centroid",
        "split": OUT / "11_09_2026_output/split_inclusive_efficiency/results.json",
        "sigma": OUT / "11_09_2026_output/run2_vs_run4_resolution/results.json",
        "small": OUT / "11_10_2026_output/small_model_eval/compare3_r16443.json",
        "small_b": OUT / "11_10_2026_output/small_model_eval/compare3_r16638.json",
        "cpu": OUT / "11_09_2026_output/cpu_bench_packed/cpu_bench.json",
    },
    "gnn": {
        "checkpoint": "ttva_gnn_hllhc_v4_noaug epoch 107 (to be replaced by full-mu v2)",
        "key": "gnn_v4",  # key stem in hs_chain_summary.json
        "t": "t098",  # GNN score cut used for every after-GNN number
        "chain": OUT / "11_09_2026_output/hs_chain_study/hs_chain_summary.json",
        "hsres": OUT / "11_11_2026_output/hs_position_estimators/results.json",
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
    g = f"{SOURCES['gnn']['key']}_{SOURCES['gnn']['t']}"  # e.g. gnn_v4_t098
    a3 = f"ada3d_{SOURCES['gnn']['t']}"  # adaptive fit at the same score cut
    w, h = "windows/r16443@0.1694", "windows/r16443@0.1"
    p = "reco_min_tracks_1/pu200"
    fs = f"{p}/fractions"
    s30 = "sigma/hl_pu200/pvf_int030_h003_cen"
    mt = "matching/fitted_sample"
    # fmt: off
    return [
        # HL-LHC vertex finding (density model)
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
        Num("sigPVF", "density", "sigma", f"{s30}/sigma_mm", 1, "{:.3f}",
            "integral 0.30, same as counting (one threshold set)"),
        Num("sigPVFErr", "density", "sigma", f"{s30}/fit_err_mm", 1, "{:.3f}"),
        Num("sigAMVF", "density", "sigma", "sigma/hl_pu200/amvf/sigma_mm", 1, "{:.3f}"),
        Num("sigAMVFErr", "density", "sigma", "sigma/hl_pu200/amvf/fit_err_mm", 1, "{:.3f}"),
        Num("hsRecoPVF", "gnn", "chain", f"{fs}/a_reco_pvf_peaks/value", 100),
        Num("hsRecoAMVF", "gnn", "chain", f"{fs}/a_reco_amvf/value", 100),
        Num("meanMu", "gnn", "chain", f"{p}/mean_mu"),
        # small models (CPU); r16443 = A, r16638 = B
        Num("sFiveDEffA", "density", "small", "s5/d_eff_pt", 1, "{:.2f}"),
        Num("sFiveDEffB", "density", "small_b", "s5/d_eff_pt", 1, "{:.2f}"),
        Num("sFiveDFake", "density", "small", "s5/d_fp"),
        Num("sEightDEffA", "density", "small", "s8/d_eff_pt", 1, "{:.2f}"),
        Num("sEightDEffB", "density", "small_b", "s8/d_eff_pt", 1, "{:.2f}"),
        Num("sEightDFake", "density", "small", "s8/d_fp"),
        Num("cpuVsixPad", "density", "cpu", "0/median_ms", 1, "{:,.0f}"),
        Num("cpuVsixPack", "density", "cpu", "1/median_ms", 1, "{:,.0f}"),
        Num("cpuSFivePack", "density", "cpu", "3/median_ms", 1, "{:,.0f}"),
        Num("cpuSEightPack", "density", "cpu", "5/median_ms", 1, "{:,.0f}"),
        # after the GNN
        Num("cleanTruthGNN", "gnn", "chain", f"{p}/clean_per_truth/{g}/value", 100),
        Num("cleanTruthAMVF", "gnn", "chain", f"{p}/clean_per_truth/amvf/value", 100),
        Num("cleanTruthZnear", "gnn", "chain", f"{p}/clean_per_truth/z0_nearest/value", 100),
        Num("cleanTruthZsig", "gnn", "chain", f"{p}/clean_per_truth/z0_3sigma/value", 100),
        Num("cleanHSGNN", "gnn", "chain", f"{fs}/b_clean_hs_{g}/value", 100),
        Num("cleanHSAMVF", "gnn", "chain", f"{fs}/b_clean_hs_amvf/value", 100),
        Num("hsTrkGNN", "gnn", "chain", f"{p}/c_hs_track_fraction/{g}/value", 100),
        Num("hsTrkAMVF", "gnn", "chain", f"{p}/c_hs_track_fraction/amvf/value", 100),
        Num("hsPurGNN", "gnn", "chain", f"{p}/c_hs_vertex_purity/{g}/value", 100),
        Num("hsPurAMVF", "gnn", "chain", f"{p}/c_hs_vertex_purity/amvf/value", 100),
        Num("hsSelGNN", "gnn", "chain", f"{fs}/d_select_{g}_itk_0p1/value", 100),
        Num("hsSelAMVF", "gnn", "chain", f"{fs}/d_native_amvf_itk_0p1/value", 100, "{:.1f}",
            "AMVF stored primary, same code as the GNN number"),
        Num("hsSelAMVFrerank", "gnn", "recon", "pu200_185_200/flags/rerank_plurality/value", 100),
        Num("hsResGNN", "gnn", "hsres", f"estimators/{a3}/rms_um", 1, "{:.1f}",
            "3D adaptive fit with beam spot on the GNN tracks at the same t"),
        Num("hsResGNNErr", "gnn", "hsres", f"estimators/{a3}/boot_err_um", 1, "{:.1f}"),
        Num("hsResAMVF", "gnn", "hsres", "estimators/amvf/rms_um", 1, "{:.1f}"),
        Num("hsResAMVFErr", "gnn", "hsres", "estimators/amvf/boot_err_um", 1, "{:.1f}"),
        Num("hsResPeak", "gnn", "hsres", "estimators/centroid/rms_um", 1, "{:.1f}"),
        Num("hsResTruth", "gnn", "hsres", "estimators/ada3d_truth/rms_um", 1, "{:.1f}"),
        Num("hsResNcommon", "gnn", "hsres", "n_common", 1, "{:,.0f}"),
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
        "windowMM",
        "0.17",
        "PI decision 2026-11-11; 3 x 56.45 um, matching_window_audit_2026-11-09.md",
        "density",
    ),
    ("hsWindowMM", "0.1", "ITk paper Sec. 5.5 hard-scatter criterion", "density"),
    (
        "pSixEightPVF",
        "55.1",
        "matching_window_audit_2026-11-09.md (55.05 +- 0.23 um)",
        "density",
    ),
    (
        "pSixEightAMVF",
        "56.5",
        "matching_window_audit_2026-11-09.md (56.45 +- 0.26 um)",
        "density",
    ),
    (
        "nTrainEvt",
        "5.17",
        "note Sec. 3 / v6 config: million training events",
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
        "note Table tab:athena-timing-breakdown, CPU total, Xeon 4216",
        "density",
    ),
    (
        "athGPUms",
        "76",
        "note Table tab:athena-timing-breakdown, GPU total, T4",
        "density",
    ),
    (
        "athCPUinf",
        "2189",
        "note Table tab:athena-timing-breakdown, CPU inference",
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
        "projection 2189/6.4 + 31 + 182 ms (cpu_small_model_2026-11-09.md)",
        "density",
    ),
    ("projSEight", "455", "projection 2189/9.0 + 31 + 182 ms", "density"),
    (
        "gnnParams",
        "39,521",
        "note Sec. 7 GNN parameter count (v4); update for v2",
        "gnn",
    ),
    (
        "gnnTrainEvt",
        "180,000",
        "note Table tab:gnn_training (v4); update for v2",
        "gnn",
    ),
    (
        "gnnT",
        "0.98",
        "GNN score cut used for every after-GNN number (SOURCES gnn t)",
        "gnn",
    ),
]
# fmt: on


def _derived(v: dict[str, float]) -> dict[str, str]:
    pad = v["cpuVsixPad"]
    return {
        "spdVsixPack": f"{pad / v['cpuVsixPack']:.1f}",
        "spdSFive": f"{pad / v['cpuSFivePack']:.1f}",
        "spdSEight": f"{pad / v['cpuSEightPack']:.1f}",
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
