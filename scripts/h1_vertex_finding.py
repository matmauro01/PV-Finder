"""HL-LHC pre-GNN vertex-finding figures and per-mu numbers (paper items H1/H2).

Reads, for each file tag, the saved ``run_eval_pvf_run3.py --regime hllhc``
evaluation (all events, not only PU200), the source ROOT file (AMVF
positions for every event) and the fair-comparison cache and scan JSON of
``run_fair_comparison.py``. No inference and no peak finding is run.

Per tag it writes into ``--out``:

- Fig. 6: reconstructed vertices per event vs mu (PV-Finder, AMVF, truth with
  n_trk >= 2), and the AMVF category panel vs mu (the PV-Finder one comes
  from ``make_paper_figures.py``);
- Eq. (1) efficiency at the regime window vs mu, PV-Finder vs AMVF, with
  event-bootstrap errors per mu bin and the paired difference;
- on the PU200 events of the cache: signed and absolute single-vertex
  residuals (optimal one-to-one within 1 mm), the window-justification plot
  and the window-scan plot;
- ``h1_derived.json`` with every plotted number and the checks below.

Checks (each raises on failure):

- the per-event Eq. (1) numerator recomputed from positions equals the saved
  merged-credit count (PV-Finder ``tc + tm``, AMVF ``amvf_t_found``) in every
  event, and the ROOT AMVF multiplicity equals the saved ``n_amvf``;
- the signed residuals' absolute values equal
  ``matching.matched_residuals_optimal`` event by event, and their 68%
  equals ``scan.core_position_resolution``.

Optionally (``--gnn-chunks``) the peak lists the post-GNN inference used are
compared with the eval's ``pred_pvs_mm`` (a consistency gate, reported, not
raised).

    python -u scripts/h1_vertex_finding.py \\
        --eval r16443=<dir> --root r16443=<file.root> --fair r16443=<dir> \\
        --eval r16638=<dir> --root r16638=<file.root> --fair r16638=<dir> \\
        --gnn-chunks r16443=<dir with chunk_*.npz> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import uproot
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pv_finder.diagnostics.amvf_fair_comparison.cache import load_cache  # noqa: E402
from pv_finder.diagnostics.amvf_fair_comparison.matching import (  # noqa: E402
    _UNMATCHABLE,
    match_vertices,
    matched_residuals_optimal,
)
from pv_finder.diagnostics.amvf_fair_comparison.scan import (  # noqa: E402
    core_position_resolution,
)
from pv_finder.diagnostics.paper_figures import categories as cat  # noqa: E402
from pv_finder.diagnostics.paper_figures import h1_plots, sources  # noqa: E402
from pv_finder.diagnostics.paper_figures.style import SIM_STATUS  # noqa: E402
from pv_finder.evaluation.vertex_finding.freeze_compare import (  # noqa: E402
    truth_found_any,
)
from pv_finder.utils.result_provenance import result_provenance  # noqa: E402
from pv_finder.utils.working_points import (  # noqa: E402
    RESIDUAL_ASSIGNMENT_RADIUS_MM,
    get_regime,
)

Z_MIN, BIN_WIDTH = -240.0, 0.04  # bin-unit arithmetic of compare_res_reco
MU_EDGES = np.arange(0.0, 201.0, 10.0)  # last bin [190, 200] inclusive
N_BOOT, SEED = 1000, 20261112
LINE = "$\\sqrt{s}$ = 14 TeV, $t\\bar{t}$, ITk layout"


def load_eval(path: Path) -> dict[str, Any]:
    """Per-event arrays of one saved evaluation (histograms dropped)."""
    with open(path / "eval_results.pkl", "rb") as fh:
        d = pickle.load(fh)
    d.pop("histograms", None)
    pe = d["per_event"]
    if not (len(pe) == len(d["pred_pvs_mm"]) == len(d["truth_pvs_mm"])):
        raise ValueError(f"{path}: per_event does not align with the event lists")
    pred = [np.asarray(z) for z in d["pred_pvs_mm"]]
    for r, p in zip(pe, pred):
        if r["n_pred"] != len(p):
            raise ValueError(f"event {r['event_idx']}: n_pred != number of peaks")
    return {
        "idx": np.array([r["event_idx"] for r in pe], dtype=np.int64),
        "mu": np.array([r["mu"] for r in pe], dtype=np.float64),
        "n_truth": np.array([r["n_truth"] for r in pe], dtype=np.int64),
        "n_pred": np.array([r["n_pred"] for r in pe], dtype=np.int64),
        "n_amvf": np.array([r["n_amvf"] for r in pe], dtype=np.int64),
        "found": np.array([r["tc"] + r["tm"] for r in pe], dtype=np.int64),
        "split": np.array([r["split"] for r in pe], dtype=np.int64),
        "fake": np.array([r["fake"] for r in pe], dtype=np.int64),
        "amvf_found": np.array([r["amvf_t_found"] for r in pe], dtype=np.int64),
        "in_window": np.asarray(d["in_mu_window"], dtype=bool),
        "pred": pred,
        "truth": [np.asarray(z) for z in d["truth_pvs_mm"]],
        "window_mm": float(d["match_window_mm"]),
        "checkpoint": d.get("e2e_checkpoint") or d.get("k2h_checkpoint"),
    }


def amvf_positions(root: str, idx: np.ndarray) -> list[np.ndarray]:
    """AMVF z (n_trk >= 2, float32 as run3_io stores it) for the given entries."""
    a = uproot.open(root)["PVFinderData"].arrays(
        ["RecoVertex_z", "RecoVertex_nTracks"], entry_stop=int(idx.max()) + 1,
        library="np",
    )  # fmt: skip
    out = []
    for i in idx:
        z, n = np.asarray(a["RecoVertex_z"][i]), np.asarray(a["RecoVertex_nTracks"][i])
        out.append(np.asarray(z[n >= 2], dtype=np.float32))
    return out


def eq1_found(
    truth: list[np.ndarray], reco: list[np.ndarray], w_mm: float
) -> np.ndarray:
    """Eq. (1) numerator per event, in the matcher's bin-unit arithmetic."""
    w = w_mm / BIN_WIDTH
    return np.array([truth_found_any((t - Z_MIN) / BIN_WIDTH, (r - Z_MIN) / BIN_WIDTH, w)
                     for t, r in zip(truth, reco)], dtype=np.int64)  # fmt: skip


def mu_bins(mu: np.ndarray) -> np.ndarray:
    """Bin index of each event on MU_EDGES; -1 outside [0, 200]."""
    b = np.searchsorted(MU_EDGES, mu, side="right") - 1
    b[mu == MU_EDGES[-1]] = len(MU_EDGES) - 2
    b[(mu < MU_EDGES[0]) | (mu > MU_EDGES[-1])] = -1
    return b


def yields_vs_mu(ev: dict, b: np.ndarray) -> dict[str, dict[str, list[float]]]:
    """Mean and SEM of reconstructed / truth vertices per event in each mu bin."""
    out: dict[str, dict[str, list[float]]] = {}
    for key, arr in (("pvf", ev["n_pred"]), ("amvf", ev["n_amvf"]), ("truth", ev["n_truth"])):  # fmt: skip
        x, y, e, n = [], [], [], []
        for k in range(len(MU_EDGES) - 1):
            sel = b == k
            m, s = cat.mean_sem(arr[sel])
            x.append(float(ev["mu"][sel].mean()))
            y.append(m)
            e.append(s)
            n.append(int(sel.sum()))
        out[key] = {"x": x, "y": y, "e": e, "n_events": n}
    return out


def match_counts(truth: list[np.ndarray], reco: list[np.ndarray], w_mm: float) -> dict:
    """Per-event production-matcher counts (merged credit, one-to-one, split, fake).

    Positions go in as the bin units of ``compare_res_reco`` (the counts are
    scale invariant), so a vertex pair on the window edge is decided with the
    same float32 arithmetic as the saved evaluation; in mm a few pairs within
    ~1e-6 mm of the edge flip.
    """
    keys = ("matched", "primary", "split", "fake", "n_reco")
    out = {k: np.zeros(len(reco), dtype=np.int64) for k in keys}
    wb = w_mm / BIN_WIDTH
    for i, (t, r) in enumerate(zip(truth, reco)):
        tb, rb = (
            (np.asarray(t) - Z_MIN) / BIN_WIDTH,
            (np.asarray(r) - Z_MIN) / BIN_WIDTH,
        )
        m = match_vertices(tb, rb, wb)
        out["matched"][i], out["primary"][i] = m.n_truth_matched, m.n_truth_primary
        out["split"][i], out["fake"][i] = m.reco_split, m.reco_fake
        out["n_reco"][i] = m.n_reco
    return out


def ratio_vs_mu(
    num: dict[str, np.ndarray],
    den: dict[str, np.ndarray],
    b: np.ndarray,
    rng: np.random.Generator,
    scale: float = 100.0,
) -> dict[str, Any]:
    """sum(num) / sum(den) per mu bin and arm; event bootstrap, paired difference.

    With ``den`` all ones this is a per-event mean (fakes or splits / event).
    """
    out: dict[str, Any] = {"x": [], "xerr": [], "edges": MU_EDGES.tolist(), "n_events": [],
                           "pvf": {"val": [], "err": []}, "amvf": {"val": [], "err": []},
                           "diff": [], "diff_err": [], "scale": scale}  # fmt: skip
    for k in range(len(MU_EDGES) - 1):
        j = np.flatnonzero(b == k)
        boot = rng.integers(0, len(j), size=(N_BOOT, len(j)))
        val, rep = {}, {}
        for arm in ("pvf", "amvf"):
            n, d = num[arm][j], den[arm][j]
            val[arm] = scale * n.sum() / d.sum()
            rep[arm] = scale * n[boot].sum(1) / d[boot].sum(1)
            out[arm]["val"].append(float(val[arm]))
            out[arm]["err"].append(float(np.std(rep[arm], ddof=1)))
        out["x"].append(0.5 * (MU_EDGES[k] + MU_EDGES[k + 1]))
        out["xerr"].append(0.5 * (MU_EDGES[k + 1] - MU_EDGES[k]))
        out["n_events"].append(len(j))
        out["diff"].append(float(val["pvf"] - val["amvf"]))
        out["diff_err"].append(float(np.std(rep["pvf"] - rep["amvf"], ddof=1)))
    return out


# name: (numerator key, denominator key or None for per event, scale, y label,
#        difference label, paper role)
VS_MU = {
    "eff": ("matched", "n_truth", 100.0, "Vertex-finding efficiency [%]", "[pt]", "main"),
    "eff_one_to_one": ("primary", "n_truth", 100.0, "One-to-one efficiency [%]", "[pt]",
                       "backup"),
    "fake_fraction": ("fake", "n_reco", 100.0, "Fake / reconstructed vertices [%]",
                      "[pt]", "main"),
    "fakes_per_event": ("fake", None, 1.0, "Fake vertices / event", "", "main"),
    "splits_per_event": ("split", None, 1.0, "Split vertices / event", "", "backup"),
}  # fmt: skip


def signed_residuals(truth: np.ndarray, reco: np.ndarray, w_mm: float) -> np.ndarray:
    """z_reco - z_truth of matched_residuals_optimal's assignment (signed)."""
    if len(truth) == 0 or len(reco) == 0:
        return np.zeros(0)
    d = reco[:, None].astype(np.float64) - truth[None, :].astype(np.float64)
    dist = np.abs(d)
    ri, tj = linear_sum_assignment(np.where(dist <= w_mm, dist, _UNMATCHABLE))
    s = d[ri, tj]
    return s[np.abs(s) <= w_mm]


def residuals(cache: dict) -> dict[str, Any]:
    """Signed residuals per arm on the cache events, checked against the reference."""
    w = RESIDUAL_ASSIGNMENT_RADIUS_MM
    out: dict[str, Any] = {}
    for arm in ("pvf", "amvf"):
        chunks = []
        for t, r in zip(cache["truth_ge2"], cache[arm]):
            s = signed_residuals(t, r, w)
            ref = (
                matched_residuals_optimal(t, r, w) if len(t) and len(r) else np.zeros(0)
            )
            if not np.array_equal(np.sort(np.abs(s)), np.sort(ref)):
                raise AssertionError(f"{arm}: signed residuals != optimal assignment")
            chunks.append(s)
        s = np.concatenate(chunks)
        p68 = float(np.percentile(np.abs(s), 68.0))
        if p68 != core_position_resolution(cache["truth_ge2"], cache[arm]):
            raise AssertionError(f"{arm}: p68 != core_position_resolution")
        out[arm] = {"signed_mm": s, "p68_mm": p68, "n_pairs": int(len(s)),
                    "mean_um": 1e3 * float(s.mean()), "median_um": 1e3 * float(np.median(s))}  # fmt: skip
    return out


def gnn_gate(ev: dict, chunk_dir: Path) -> dict[str, Any]:
    """Compare the post-GNN inference peak lists with the eval's peaks."""
    entries, peaks = [], []
    for f in sorted(chunk_dir.glob("chunk_*.npz")):
        z = np.load(f)
        off = z["peak_offsets"]
        entries.extend(z["entries"].tolist())
        peaks.extend(z["peak_z"][off[i] : off[i + 1]] for i in range(len(off) - 1))
    pos = {int(e): k for k, e in enumerate(ev["idx"])}
    res: dict[str, Any] = {"chunk_dir": str(chunk_dir), "n_chunk_events": len(entries)}
    for name, keep in (("pu200", ev["in_window"]), ("all", np.ones(len(ev["idx"]), bool))):  # fmt: skip
        n = n_count = n_val = 0
        max_dz = 0.0
        missing = 0
        for e, p in zip(entries, peaks):
            k = pos.get(int(e))
            if k is None:
                missing += 1
                continue
            if not keep[k]:
                continue
            n += 1
            a, q = (
                np.sort(np.asarray(ev["pred"][k], np.float64)),
                np.sort(p.astype(np.float64)),
            )
            if len(a) != len(q):
                n_count += 1
                continue
            dz = float(np.max(np.abs(a - q))) if len(a) else 0.0
            max_dz = max(max_dz, dz)
            n_val += dz > 0.0
        res[name] = {"n_events_compared": n, "n_count_differs": n_count,
                     "n_positions_differ": int(n_val), "max_abs_dz_mm_same_count": max_dz,
                     "n_chunk_entries_not_in_eval": missing}  # fmt: skip
    pu = int(ev["in_window"].sum())
    res["pu200_all_covered"] = res["pu200"]["n_events_compared"] == pu
    return res


def run_tag(tag: str, args: argparse.Namespace, rng: np.random.Generator) -> dict:
    """Every number and figure of one file."""
    ev = load_eval(Path(args.eval[tag]))
    w = ev["window_mm"]
    if w != get_regime("hllhc").match_window_mm:
        raise ValueError(f"{tag}: eval window {w} is not the regime window")
    amvf = amvf_positions(args.root[tag], ev["idx"])
    if not np.array_equal([len(a) for a in amvf], ev["n_amvf"]):
        raise AssertionError(f"{tag}: ROOT AMVF multiplicity != saved n_amvf")
    if not np.array_equal(eq1_found(ev["truth"], ev["pred"], w), ev["found"]):
        raise AssertionError(f"{tag}: PV-Finder Eq. (1) != saved merged credit")
    if not np.array_equal(eq1_found(ev["truth"], amvf, w), ev["amvf_found"]):
        raise AssertionError(f"{tag}: AMVF Eq. (1) != saved amvf_t_found")
    print(f"[{tag}] {len(ev['idx'])} events: Eq. (1) identity holds for both arms")
    b = mu_bins(ev["mu"])
    out_dir = Path(args.out) / "plots"
    n_all = len(ev["idx"])
    lines = [LINE, f"$0 \\leq \\mu \\leq 200$, {n_all:,} events ({tag})"]
    yv = yields_vs_mu(ev, b)
    h1_plots.plot_yield_vs_mu(yv, lines, out_dir, f"fig6_vertices_vs_mu_{tag}")
    mc = {"pvf": match_counts(ev["truth"], ev["pred"], w),
          "amvf": match_counts(ev["truth"], amvf, w)}  # fmt: skip
    for key, mine in (("matched", "found"), ("split", "split"), ("fake", "fake")):
        bad = np.flatnonzero(mc["pvf"][key] != ev[mine])
        if len(bad):
            raise AssertionError(f"{tag}: matcher {key} != saved {mine} in {len(bad)} "
                                 f"events, e.g. entry {ev['idx'][bad[0]]}")  # fmt: skip
    if not np.array_equal(mc["amvf"]["matched"], ev["amvf_found"]):
        raise AssertionError(f"{tag}: AMVF matcher count != saved amvf_t_found")
    ones = np.ones(n_all, dtype=np.int64)
    vs_mu: dict[str, Any] = {}
    for name, (nk, dk, scale, ylab, dunit, role) in VS_MU.items():
        den = {a: ev["n_truth"] if dk == "n_truth" else mc[a]["n_reco"] if dk else ones
               for a in mc}  # fmt: skip
        vs_mu[name] = ratio_vs_mu({a: mc[a][nk] for a in mc}, den, b, rng, scale)
        vs_mu[name]["paper_role"] = role
        h1_plots.plot_ratio_vs_mu(
            vs_mu[name], ylab, f"PVF $-$ AMVF {dunit}".strip(),
            [*lines, f"Truth matching within {w:g} mm"], out_dir,
            f"{name}_vs_mu_{tag}", bounded=scale == 100.0 and name.startswith("eff"),
        )  # fmt: skip
    amvf_counts = cat.per_event_counts(ev["truth"], amvf, w)
    smp = sources.Sample(tag, ev["mu"], amvf, ev["truth"], amvf, w, False)
    cat.plot_categories_vs_mu(smp, amvf_counts, np.arange(0, 211, 10),
                              [LINE, "AMVF", f"Truth matching within {w:.2f} mm"],
                              SIM_STATUS, out_dir, f"fig6_categories_vs_mu_amvf_{tag}")  # fmt: skip
    out: dict[str, Any] = {
        "eval_dir": args.eval[tag], "root": args.root[tag], "n_events_all": n_all,
        "density_checkpoint": ev["checkpoint"],
        "n_events_pu200": int(ev["in_window"].sum()), "window_mm": w,
        "checks": {"eq1_identity_all_events_pvf": True, "eq1_identity_all_events_amvf": True,
                   "root_amvf_multiplicity": True,
                   "matcher_split_fake_equal_saved_pvf": True},
        "per_event_mean_all_mu": {k: float(ev[k].mean()) for k in ("n_pred", "n_amvf", "n_truth")},
        "yield_vs_mu": yv, "vs_mu": vs_mu,
        "amvf_categories_all_mu": {k: float(v.mean()) for k, v in amvf_counts.items()},
    }  # fmt: skip
    if tag in args.fair:
        comp = Path(args.compare[tag]) / "compare.json"
        out.update(pu200_part(tag, Path(args.fair[tag]), comp, ev, out_dir))
    if tag in args.gnn_chunks:
        out["gnn_peak_gate"] = gnn_gate(ev, Path(args.gnn_chunks[tag]))
    return out


def pu200_part(
    tag: str, fair: Path, compare_json: Path, ev: dict, out_dir: Path
) -> dict[str, Any]:
    """Residuals, window justification and window scan on the PU200 cache."""
    cache = load_cache(fair / f"cache_{tag}.npz")
    if not np.array_equal(cache["event_idx"], ev["idx"][ev["in_window"]]):
        raise ValueError(f"{tag}: cache events != eval PU200 events")
    payload = json.loads((fair / f"results_{tag}.json").read_text())
    w = ev["window_mm"]
    res = residuals(cache)
    n = len(cache["event_idx"])
    lines = [LINE, f"$185 \\leq \\mu \\leq 215$, {n:,} events ({tag})"]
    comp = json.loads(compare_json.read_text())
    ref = comp["ref"]
    p68 = {"pvf": (1e3 * res["pvf"]["p68_mm"], comp["arms"][ref]["p68_err_um"]),
           "amvf": (1e3 * res["amvf"]["p68_mm"], comp["arms"]["AMVF"]["p68_err_um"])}  # fmt: skip
    for arm, key in (("pvf", ref), ("amvf", "AMVF")):
        if not np.isclose(p68[arm][0], comp["arms"][key]["p68_um"], rtol=1e-6, atol=0):
            raise AssertionError(f"{tag} {arm}: p68 differs from compare.json")
    signed_um = {a: 1e3 * res[a]["signed_mm"] for a in ("pvf", "amvf")}
    h1_plots.plot_residuals(signed_um, p68, [*lines, "One-to-one within 1 mm"], out_dir,
                            f"fig7b_residuals_{tag}")  # fmt: skip
    h1_plots.plot_window_justification(
        {a: np.abs(s) for a, s in signed_um.items()}, p68, w,
        [*lines, "One-to-one within 1 mm"], out_dir, f"window_justification_{tag}",
    )  # fmt: skip
    rows = payload["scan"]["rows"]
    if not any(abs(r["window_mm"] - w) < 1e-9 for r in rows):
        raise ValueError(f"{tag}: the scan has no point at {w} mm")
    h1_plots.plot_window_scan(rows, w, lines, out_dir, f"window_scan_{tag}")
    three = {a: 3e-3 * p68[a][0] for a in p68}
    return {
        "fair_dir": str(fair),
        "residuals": {a: {**{k: v for k, v in res[a].items() if k != "signed_mm"},
                          "p68_um": p68[a][0], "p68_err_um": p68[a][1]} for a in res},
        "three_x_p68_mm": three,
        "three_x_larger_p68_mm": max(three.values()),
        "three_x_larger_p68_rounds_to_window": round(max(three.values()), 2) == round(w, 2),
        "check_residuals_vs_reference": True,
    }  # fmt: skip


def _pairs(spec: list[str] | None) -> dict[str, str]:
    return dict(s.split("=", 1) for s in spec or [])


def main(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    result: dict[str, Any] = {"n_boot_per_mu_bin": N_BOOT, "seed": SEED, "tags": {}}
    for tag in args.eval:
        result["tags"][tag] = run_tag(tag, args, rng)
    ckpts = {t["density_checkpoint"] for t in result["tags"].values()}
    if len(ckpts) != 1:
        raise ValueError(f"evaluations use different density checkpoints: {ckpts}")
    result["provenance"] = result_provenance(ckpts.pop())
    (out / "h1_derived.json").write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {out / 'h1_derived.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--eval", "--root", "--fair", "--compare", "--gnn-chunks"):
        p.add_argument(flag, action="append", metavar="TAG=PATH")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    for k in ("eval", "root", "fair", "compare", "gnn_chunks"):
        setattr(a, k, _pairs(getattr(a, k)))
    main(a)
