"""Collect the HL-LHC CPU timing measurements and project them to Athena.

Reads the outputs of one timing run directory (see its ``run_all.sh``):
``density_rep*/cpu_bench.json`` (``scripts/cpu_inference_benchmark.py``,
ONNX Runtime) and ``chain_stages.json`` (``scripts/cpu_chain_timing.py``,
torch), plus ``loadavg.log``. Writes ``cpu_timing.json`` with every measured
number, the provenance (CPU, load, git, checkpoint sha256, regime) and the
projection to the Athena timing of the note.

MEASURED numbers are on this machine. PROJECTED numbers combine measured
ratios with the Athena table of the note (``ATHENA`` below, quoted verbatim
from ANA-IDTR-2026-01-INT1 sections/04b_timing.tex) and the AMVF time read
off the note's figure; AMVF cannot be run here.

Usage:
    PYTHONPATH=src python -u scripts/hllhc_cpu_timing_summary.py \\
        --dir outputs/11_12_2026_output/hllhc_final/cpu_timing
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from gnn.evaluation.chain_inference import REPO, git_state, sha256
from pv_finder.utils.working_points import V6, get_regime

# Note table tab:athena-timing-breakdown, CPU column (Intel Xeon Silver 4216,
# 597 events, Run 4 PU200), sections/04b_timing.tex lines 148-158.
ATHENA = {
    "track_selection": 30.54,
    "tensor_building": 0.10,
    "inference": 2188.67,
    "peak_finding": 0.05,
    "vertex_building": 0.29,
    "ttva": 181.65,
    "ttva_graph_building": 2.83,
    "ttva_gnn_inference": 177.89,
    "ttva_edge_selection": 0.68,
    "total": 2401.28,
}
# Read off figures/athena_cpu_gpu_amvf.png (note Fig. athena_timing_gpu): the
# two Run 4 MC points at mu = 200 sit at about 620 and 690 ms.
AMVF_MS = {"central": 650.0, "low": 600.0, "high": 750.0}
NOISE = 0.20  # run-to-run spread of absolute times on the shared machine


KEY = {
    ("280:4", "fp32"): "v6_padded",
    ("280:4", "fp32_packed"): "v6_packed",
    ("80:1", "fp32"): "s8_padded",
    ("80:1", "fp32_packed"): "s8_packed",
}
REF_0911 = "outputs/11_09_2026_output/cpu_bench_packed/cpu_bench.json"


def density_run(p: Path) -> dict:
    """One invocation: median ms/event of v6 and s8, padded and packed.

    The speed-ups are ratios of the per-model medians; for an interleaved run
    the median of the per-event paired ratios is added (``*_paired``).
    """
    rows = [x for x in json.loads(p.read_text()) if x["threads"] == 1]
    rows = [x for x in rows if (x["spec"], x["variant"]) in KEY]
    r: dict = {KEY[(x["spec"], x["variant"])]: x["median_ms"] for x in rows}
    r["order"] = [x["spec"] for x in rows][::2]
    r["interleaved"] = all(x.get("interleaved", False) for x in rows)
    r["file"] = str(p.relative_to(REPO)) if p.is_absolute() else str(p)
    for m in ("s8_packed", "s8_padded", "v6_packed"):
        r[f"speedup_{m}_vs_v6_padded"] = r["v6_padded"] / r[m]
    if r["interleaved"]:
        t = {KEY[(x["spec"], x["variant"])]: np.asarray(x["times_ms"]) for x in rows}
        for m in ("s8_packed", "s8_padded", "v6_packed"):
            ratio = t["v6_padded"] / t[m]
            r[f"speedup_{m}_vs_v6_padded_paired"] = float(np.median(ratio))
    return r


def density_runs(d: Path, pattern: str) -> list[dict]:
    """All invocations in ``d`` matching ``pattern``/cpu_bench.json."""
    runs = [density_run(p) for p in sorted(d.glob(f"{pattern}/cpu_bench.json"))]
    if not runs:
        raise FileNotFoundError(f"no {pattern}/cpu_bench.json in {d}")
    return runs


def agg(values: list[float]) -> dict[str, float]:
    """Median and range over repeats."""
    return {"median": float(np.median(values)), "min": float(min(values)),
            "max": float(max(values)), "n": len(values)}  # fmt: skip


def project(speedup: float, gnn_scale: float) -> dict[str, float]:
    """Athena-equivalent chain time for a density model ``speedup`` x faster."""
    other = sum(ATHENA[k] for k in ("track_selection", "tensor_building",
                                    "peak_finding", "vertex_building"))  # fmt: skip
    dens = ATHENA["inference"] / speedup
    ttva = ATHENA["ttva"] * gnn_scale
    total = dens + other + ttva
    return {
        "density_ms": dens, "other_ms": other, "ttva_ms": ttva, "total_ms": total,
        "amvf_over_chain": {k: v / total for k, v in AMVF_MS.items()},
        "margin_vs_amvf_ms": {k: v - total for k, v in AMVF_MS.items()},
    }  # fmt: skip


def required_speedup() -> dict[str, float]:
    """Density speed-up vs v6 padded needed to tie AMVF (GNN as in Athena)."""
    rest = ATHENA["total"] - ATHENA["inference"]
    return {k: ATHENA["inference"] / (v - rest) for k, v in AMVF_MS.items()}


def cpu_model() -> str:
    """CPU model name from /proc/cpuinfo."""
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True, type=Path)
    a = ap.parse_args()
    d = a.dir.resolve()
    runs = density_runs(d, "density_interleaved_rep*")  # primary
    seq = density_runs(d, "density_rep*")  # one model after the other
    ref = density_run(REPO / REF_0911)  # 9 Nov 2026 run quoted in the docs
    chain = json.loads((d / "chain_stages.json").read_text())
    st = chain["stages"]
    med = {k: v["median_of_pass_medians_ms"] for k, v in st.items()}
    sp = {
        m: agg([r[f"speedup_{m}_vs_v6_padded_paired"] for r in runs])
        for m in ("s8_packed", "s8_padded", "v6_packed")
    }
    # Every same-invocation estimate of each speed-up (interleaved paired,
    # interleaved and sequential medians, 9 Nov): its spread is the systematic.
    sp_all = {
        m: agg(
            [r[f"speedup_{m}_vs_v6_padded_paired"] for r in runs]
            + [r[f"speedup_{m}_vs_v6_padded"] for r in [*runs, *seq, ref]]
        )
        for m in sp
    }
    gnn_ratio = med["gnn_v2e527"] / med["gnn_v4e107"]

    s8_pk = agg([r["s8_packed"] for r in runs])["median"]
    measured_chain = {
        "tensor_building_ms": med["tensor"],  # padded builder; packing not timed
        "s8_packed_onnx_ms": s8_pk,
        "peaks_ms": med["peaks"],
        "graph_ms": med["graph"],
        "gnn_v2e527_forward_torch_ms": med["gnn_v2e527"],
        "edge_selection_ms": med["select_v2e527"],
    }
    measured_chain["total_ms"] = sum(measured_chain.values())

    projections = {}
    for name in ("s8_packed", "s8_padded"):
        c, lo, hi = sp[name]["median"], sp_all[name]["min"], sp_all[name]["max"]
        projections[name] = {
            "speedup_used": c,
            "central": project(c, 1.0),
            "range_from_all_speedup_estimates_ms": [
                project(hi, 1.0)["total_ms"],
                project(lo, 1.0)["total_ms"],
            ],
            # The density term moved +-20% (ratio not transferring exactly to
            # the Xeon 4216, or load) as a deliberately generous allowance.
            "density_term_pm20pct_ms": [
                project(c * (1 + NOISE), 1.0)["total_ms"],
                project(c * (1 - NOISE), 1.0)["total_ms"],
            ],
            "with_measured_gnn_v2_over_v4_scale": project(c, gnn_ratio),
        }
    rp = get_regime("hllhc")
    loads = (d / "loadavg.log").read_text().splitlines()
    out = {
        "question": "Is s8 + GNN v2 e527 (frozen hllhc) faster than AMVF at mu=200 on CPU?",
        "machine": {
            "host": platform.node(),
            "cpu_model": cpu_model(),
            "n_cpu": len(
                [
                    1
                    for x in Path("/proc/cpuinfo").read_text().splitlines()
                    if x.startswith("processor")
                ]
            ),
            "loadavg_log": loads,
            "loadavg_start": loads[0],
            "loadavg_end": loads[-1],
        },  # fmt: skip
        "threads": {
            "onnxruntime_intra_op": 1,
            "onnxruntime_inter_op": 1,
            "torch": chain["threads"],
        },  # fmt: skip
        "git": git_state(),
        "checkpoints": {
            "v6": {"path": V6.checkpoint, "sha256": sha256(REPO / V6.checkpoint)},
            "s8": {
                "path": chain["density"]["checkpoint"],
                "sha256": chain["density"]["sha256"],
            },
            **{
                f"gnn_{k}": {
                    "path": v["path"],
                    "sha256": v["sha256"],
                    "params": v["params"],
                }
                for k, v in chain["gnn"].items()
            },
        },  # fmt: skip
        "regime": rp.to_dict(),
        "MEASURED": {
            "density_onnx": {
                "protocol": "scripts/cpu_inference_benchmark.py --interleave, ONNX "
                "Runtime opset 17, 1 intra-op thread, one event = 12 sub-events, "
                "first 40 real events of the r16438 PU200 h5, 80 rounds (each event "
                "twice) after 2 warm-up, the four models run round-robin on each "
                "event; padded = cropped to the widest sub-event",
                "per_invocation": runs,
                "ms_per_event": {m: agg([r[m] for r in runs]) for m in KEY.values()},
                "speedup_vs_v6_padded_paired": sp,
                "speedup_vs_v6_padded_all_estimates": sp_all,
                "sequential_invocations": seq,
                "reference_2026_11_09": ref,
            },
            "chain_stages_torch": {
                "protocol": "scripts/cpu_chain_timing.py, torch 1 thread, "
                f"{chain['n_events']} real r16438 PU200 events (entries "
                f"{chain['entries']}), {chain['n_pass']} passes, median over events "
                "per pass, then median of the pass medians",
                "ms_per_event": med,
                "per_pass": {k: v["per_pass"] for k, v in st.items()},
                "gnn_v2_over_v4_forward_ratio": gnn_ratio,
                "n_tracks_mean": chain["n_tracks_mean"],
                "n_peaks_mean": chain["n_peaks_mean"],
                "n_edges_track_to_pv_mean": chain["n_edges_track_to_pv_mean"],
                "mu_min_max": chain["mu_min_max"],
            },  # fmt: skip
            "sneezy_chain_total": measured_chain,
        },
        "QUOTED": {
            "athena_cpu_ms": ATHENA,
            "athena_source": "ANA-IDTR-2026-01-INT1 sections/04b_timing.tex, "
            "tab:athena-timing-breakdown (CPU column), Intel Xeon Silver 4216",
            "amvf_ms_mu200": AMVF_MS,
            "amvf_source": "read off figures/athena_cpu_gpu_amvf.png at mu = 200",
        },
        "PROJECTED": {
            "formula": "T = 2188.67 / speedup(s8 vs v6 padded, measured) + "
            "(30.54 + 0.10 + 0.05 + 0.29) + 181.65 * g, g = 1 (same GNN architecture)",
            "athena_equivalent_chain": projections,
            "required_speedup_to_tie_amvf": required_speedup(),
        },
    }
    (d / "cpu_timing.json").write_text(json.dumps(out, indent=1) + "\n")
    brief = {
        "density_ms": out["MEASURED"]["density_onnx"]["ms_per_event"],
        "speedup_paired": sp,
        "speedup_all": sp_all,
        "stages": med,
        "sneezy_total": measured_chain,
        "projection": {
            k: [v["central"]["total_ms"], v["range_from_all_speedup_estimates_ms"]]
            for k, v in projections.items()
        },
        "required": required_speedup(),
    }
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
