"""Collect the H1/H2 numbers into h1_summary.json and Table 3 (markdown + LaTeX).

Reads only the result files written by ``scripts/run_h1_vertex_finding.sh``
under ``--out``:

- ``compare_<tag>/compare.json``  (freeze_candidates_compare.py: Eq. (1),
  fakes, splits, HS, sigma_vtx-vtx, p68, paired differences);
- ``fair_<tag>/results_<tag>.json`` (run_fair_comparison.py: one-to-one
  efficiency, window scan, chance matches);
- ``split_inclusive/results.json`` (split_inclusive_efficiency.py);
- ``h1_derived.json`` (h1_vertex_finding.py: per-mu numbers, residuals,
  3 x p68, peak-consistency gate);
- ``plots/paper_figures_r16443/summary.json`` (make_paper_figures.py).

Cross-checks between the files are asserted, and the AMVF anchors on r16443
are reported as pass/fail. Differences are PV-Finder minus AMVF with paired
event-bootstrap errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_finder.utils.result_provenance import result_provenance  # noqa: E402
from pv_finder.utils.working_points import get_regime, window_tag  # noqa: E402

# Previously published AMVF values on r16443 PU200 (same 1,920 events).
ANCHORS = {"eff_pct": 79.85, "sigma_mm": 0.305, "p68_um": 56.45, "hs": (1916, 1920)}
SCAN_POINTS = (0.10, 0.15, 0.17, 0.20, 0.30, 0.50, 1.00)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _close(a: float, b: float, rtol: float = 1e-9) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b), 1e-12)


def _count_diff(cmp_arm: dict, other: dict, key: str, n_events: int) -> dict:
    """Vertex-count difference of one metric between compare.json and another file."""
    norm = cmp_arm["n_truth"] if key == "eff" else n_events
    okey = {"fp_per_evt": "fake_per_evt"}.get(key, key)
    diff = int(round((cmp_arm[key] - other[okey]) * norm))
    return {"compare_minus_other_vertices": diff, "pass_edge_tolerance": abs(diff) <= 5}


def collect(out: Path, tag: str, split: dict, derived: dict) -> dict[str, Any]:
    """All Table 3 numbers of one file, with cross-checks between sources."""
    w = get_regime("hllhc").match_window_mm
    cpath = out / f"compare_{tag}" / "compare.json"
    fpath = out / f"fair_{tag}" / f"results_{tag}.json"
    comp, fair = _read(cpath), _read(fpath)
    p, a = comp["arms"][comp["ref"]], comp["arms"]["AMVF"]
    head = fair["headlines"][f"{w:g}"]
    sp = split["windows"][f"{tag}@{w:g}"]
    d = derived["tags"][tag]
    checks = {
        "compare_checks": comp["checks"],
        # compare.json decides window-edge pairs in the matcher's float32 bin
        # units; the fair-comparison cache and split script in float32 mm. A
        # few vertices within ~1e-6 mm of the edge flip; counted, not hidden.
        "counts_compare_vs_fair_mm": {
            arm: {key: _count_diff(cmp_arm, head[arm], key, comp["n_events"])
                  for key in ("eff", "fp_per_evt", "split_per_evt")}
            for arm, cmp_arm in (("pvf", p), ("amvf", a))},
        "counts_compare_vs_split_mm": {
            arm: _count_diff(cmp_arm, sp[arm], "eff", comp["n_events"])
            for arm, cmp_arm in (("pvf", p), ("amvf", a))},
        "p68_compare_eq_fair_core_pvf": _close(
            p["p68_um"], 1e3 * fair["core_position_resolution_mm"]["pvf"], 1e-6),
        "p68_compare_eq_fair_core_amvf": _close(
            a["p68_um"], 1e3 * fair["core_position_resolution_mm"]["amvf"], 1e-6),
        "n_events_agree": comp["n_events"] == fair["meta"]["n_events"] == sp["n_events"],
        "h1_derived_checks": d["checks"],
    }  # fmt: skip

    def row(pv: float, pe: Any, av: float, ae: Any, dv: float, de: float) -> dict:
        return {"pvf": pv, "pvf_err": pe, "amvf": av, "amvf_err": ae, "diff": dv,
                "diff_err": de}  # fmt: skip

    hd = head["diff"]
    table = {
        "eff_pct": row(100 * p["eff"], 100 * p["eff_err"], 100 * a["eff"],
                       100 * a["eff_err"], -a["d_eff_pt"], a["d_eff_pt_err"]),
        "eff_one_to_one_pct": row(
            100 * head["pvf"]["eff_strict"], 100 * head["pvf"]["eff_strict_err"],
            100 * head["amvf"]["eff_strict"], 100 * head["amvf"]["eff_strict_err"],
            100 * hd["eff_strict"], 100 * hd["eff_strict_err"]),
        "fake_fraction_pct": row(100 * p["fake_frac"], 100 * p["fake_frac_err"],
                                 100 * a["fake_frac"], 100 * a["fake_frac_err"],
                                 -a["d_fake_frac_pt"], a["d_fake_frac_pt_err"]),
        "fakes_per_event": row(p["fp_per_evt"], p["fp_per_evt_err"], a["fp_per_evt"],
                               a["fp_per_evt_err"], -a["d_fp"], a["d_fp_err"]),
        "splits_per_event": row(p["split_per_evt"], p["split_per_evt_err"],
                                a["split_per_evt"], a["split_per_evt_err"],
                                -a["d_split"], a["d_split_err"]),
        "sigma_vtx_vtx_mm": row(p["sigma_mm"], p["sigma_fit_err_mm"], a["sigma_mm"],
                                a["sigma_fit_err_mm"], -1e-3 * a["d_sigma_um"],
                                1e-3 * a["d_sigma_um_err"]),
        "p68_um": row(p["p68_um"], p["p68_err_um"], a["p68_um"], a["p68_err_um"],
                      -a["d_p68_um"], a["d_p68_um_err"]),
        "hs_reco_pct": row(100 * p["hs_reco_eff"], [100 * x for x in p["hs_reco_cp68"]],
                           100 * a["hs_reco_eff"], [100 * x for x in a["hs_reco_cp68"]],
                           -a["d_hs_pt"], a["d_hs_pt_err"]),
    }  # fmt: skip
    table["hs_reco_pct"]["counts"] = {"pvf": p["hs_reco_n"], "amvf": a["hs_reco_n"],
                                      "n_events": comp["n_events"]}  # fmt: skip
    rows = {round(r["window_mm"], 4): r for r in fair["scan"]["rows"]}
    scan = {f"{x:.2f}": {
        "pvf_eff": rows[x]["pvf"]["eff"], "amvf_eff": rows[x]["amvf"]["eff"],
        "d_eff_pt": 100 * rows[x]["diff"]["eff"], "d_eff_pt_err": 100 * rows[x]["diff"]["eff_err"],
        "d_eff_strict_pt": 100 * rows[x]["diff"]["eff_strict"],
        "d_eff_strict_pt_err": 100 * rows[x]["diff"]["eff_strict_err"],
        "chance_eff_pvf": rows[x]["pvf"]["eff_accidental"],
        "chance_eff_amvf": rows[x]["amvf"]["eff_accidental"]} for x in SCAN_POINTS}  # fmt: skip
    d_all = [100 * r["diff"]["eff"] for r in fair["scan"]["rows"]]
    d_str = [100 * r["diff"]["eff_strict"] for r in fair["scan"]["rows"]]
    return {
        "n_events": comp["n_events"], "event_idx_sha1": comp["event_idx_sha1"],
        "table3": table, "checks": checks,
        "chance_match_3mm_at_window": head["accidental"],
        "window_scan": {"points": scan, "d_eff_pt_range": [min(d_all), max(d_all)],
                        "d_eff_strict_pt_range": [min(d_str), max(d_str)],
                        "crossings": fair["crossings"],
                        "pvf_above_amvf_everywhere": min(d_all) > 0},
        "core_p68_window_basis": {"three_x_p68_mm": d.get("three_x_p68_mm"),
                                  "three_x_larger_p68_mm": d.get("three_x_larger_p68_mm"),
                                  "rounds_to_window": d.get("three_x_larger_p68_rounds_to_window"),
                                  "fair_primary_window_mm": fair["headline_window_mm"]},
        "split_inclusive": {f"@{x:g}": split["windows"][f"{tag}@{x:g}"]
                            for x in (w, 0.1) if f"{tag}@{x:g}" in split["windows"]},
        "sources": {"compare": str(cpath), "fair": str(fpath),
                    "split": str(out / "split_inclusive" / "results.json"),
                    "derived": str(out / "h1_derived.json")},
    }  # fmt: skip


def anchors(r: dict[str, Any]) -> dict[str, Any]:
    """AMVF r16443 PU200 anchors: rounded values must equal the published ones."""
    t = r["table3"]
    got = {"eff_pct": round(t["eff_pct"]["amvf"], 2),
           "sigma_mm": round(t["sigma_vtx_vtx_mm"]["amvf"], 3),
           "p68_um": round(t["p68_um"]["amvf"], 2),
           "hs": (int(t["hs_reco_pct"]["counts"]["amvf"]), t["hs_reco_pct"]["counts"]["n_events"])}  # fmt: skip
    exact = {"eff_pct": t["eff_pct"]["amvf"], "sigma_mm": t["sigma_vtx_vtx_mm"]["amvf"],
             "p68_um": t["p68_um"]["amvf"]}  # fmt: skip
    res = {k: {"expected": ANCHORS[k], "got": got[k], "exact": exact.get(k),
               "pass": tuple(got[k]) == ANCHORS[k] if k == "hs" else got[k] == ANCHORS[k]}
           for k in ANCHORS}  # fmt: skip
    res["all_pass"] = all(v["pass"] for v in res.values())
    return res


LABELS = (
    ("eff_pct", "Vertex-finding efficiency, Eq. (1) [%]", 2),
    ("eff_one_to_one_pct", "One-to-one efficiency [%]", 2),
    ("fake_fraction_pct", "Fake fraction of reconstructed vertices [%]", 2),
    ("fakes_per_event", "Fake vertices / event", 2),
    ("splits_per_event", "Split vertices / event", 3),
    ("sigma_vtx_vtx_mm", "Vertex-vertex resolution sigma_vtx-vtx [mm]", 3),
    ("p68_um", "Single-vertex position error (68%) [um]", 2),
    ("hs_reco_pct", "Hard-scatter reconstruction efficiency (0.1 mm) [%]", 2),
)


def _cell(v: float, e: Any, nd: int, tex: bool) -> str:
    if isinstance(e, list):  # Clopper-Pearson 68% interval
        up, dn = e[1] - v, v - e[0]
        if tex:
            return f"${v:.{nd}f}^{{+{up:.{nd}f}}}_{{-{dn:.{nd}f}}}$"
        return f"{v:.{nd}f} +{up:.{nd}f} / -{dn:.{nd}f}"
    pm = r" \pm " if tex else " ± "
    s = f"{v:.{nd}f}{pm}{e:.{nd}f}"
    return f"${s}$" if tex else s


def tables(per_tag: dict[str, dict]) -> tuple[str, str]:
    """Table 3 as markdown and as a LaTeX tabular, one block per file."""
    w = get_regime("hllhc").match_window_mm
    md, tex = [], []
    for tag, r in per_tag.items():
        n = r["n_events"]
        md += [f"### {tag}: {n:,} events, 185 <= mu <= 215, window {w:g} mm", "",
               "| Quantity | PV-Finder | AMVF | PV-Finder - AMVF |", "|---|---:|---:|---:|"]  # fmt: skip
        tex += [f"% {tag}: {n} events, 185 <= mu <= 215, window {w:g} mm",
                r"\begin{tabular}{lccc}", r"\toprule",
                r" & PV-Finder & AMVF & Difference \\", r"\midrule"]  # fmt: skip
        for key, label, nd in LABELS:
            x = r["table3"][key]
            d = _cell(x["diff"], x["diff_err"], nd, False)
            md.append(f"| {label} | {_cell(x['pvf'], x['pvf_err'], nd, False)} | "
                      f"{_cell(x['amvf'], x['amvf_err'], nd, False)} | {d} |")  # fmt: skip
            lt = (label.replace("sigma_vtx-vtx", r"$\sigma_{\mathrm{vtx\text{-}vtx}}$")
                  .replace("[um]", r"[$\mu$m]").replace("%", r"\%"))  # fmt: skip
            tex.append(f"{lt} & {_cell(x['pvf'], x['pvf_err'], nd, True)} & "
                       f"{_cell(x['amvf'], x['amvf_err'], nd, True)} & "
                       f"{_cell(x['diff'], x['diff_err'], nd, True)} \\\\")  # fmt: skip
        md += ["", "Errors: event bootstrap (differences paired); sigma: fit error; "
               "HS: 68% Clopper-Pearson.", ""]  # fmt: skip
        tex += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(md), "\n".join(tex)


def main(args: argparse.Namespace) -> None:
    out = Path(args.out)
    split = _read(out / "split_inclusive" / "results.json")
    derived = _read(out / "h1_derived.json")
    per_tag = {tag: collect(out, tag, split, derived) for tag in derived["tags"]}
    summary: dict[str, Any] = {
        "provenance": result_provenance(derived["provenance"]["density_checkpoint"]),
        "window_mm": get_regime("hllhc").match_window_mm,
        "window_tag": window_tag(get_regime("hllhc").match_window_mm),
        "headline_tag": "r16443", "files": per_tag,
        "amvf_anchors_r16443": anchors(per_tag["r16443"]) if "r16443" in per_tag else None,
        "per_mu": {t: {"yield_vs_mu": v["yield_vs_mu"], "vs_mu": v["vs_mu"],
                       "n_events_all": v["n_events_all"]}
                   for t, v in derived["tags"].items()},
        "gnn_peak_gate": {t: v.get("gnn_peak_gate") for t, v in derived["tags"].items()},
        "residual_shape": {t: v.get("residuals") for t, v in derived["tags"].items()},
    }  # fmt: skip
    pf = out / "plots" / "paper_figures_r16443" / "summary.json"
    if pf.exists():
        fits = _read(pf)["hllhc"]["fits"]
        summary["fig7a_fits"] = {"source": str(pf), **fits}
        t3 = per_tag["r16443"]["table3"]["sigma_vtx_vtx_mm"]
        summary["fig7a_sigma_eq_table"] = {
            "pvf": _close(fits["pvf"]["sigma_mm"], t3["pvf"], 1e-6),
            "amvf": _close(fits["amvf"]["sigma_mm"], t3["amvf"], 1e-6)}  # fmt: skip
    md, tex = tables(per_tag)
    (out / "table3.md").write_text("# Table 3 (HL-LHC, pre-GNN)\n\n" + md)
    (out / "table3.tex").write_text(tex)
    (out / "h1_summary.json").write_text(json.dumps(summary, indent=1))
    print(md)
    print(json.dumps(summary["amvf_anchors_r16443"], indent=1))
    print(f"wrote {out / 'h1_summary.json'}, table3.md, table3.tex")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", required=True, help="h1_vertex_finding output directory")
    main(p.parse_args())
