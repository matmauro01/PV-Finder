#!/usr/bin/env python3
"""Final HL-LHC full-chain numbers at the frozen working point (H4, H5, H6).

Replays the per-event full-chain metrics of ``gnn.evaluation.chain_t_metrics``
(s8 peaks -> GNN v2 e527 top-1 edges, every t of its grid, AMVF on the same
events) and extracts the frozen t of ``get_regime("hllhc")``:

* H4: post-GNN categories, track-purity taxonomy (``hs_metrics``, the chain
  code) and the positional one (greedy closest-first at 0.17 mm, labelled
  "pos."), at PU200 (185 <= mu <= 215), inclusive and in mu bins;
* H5: HS selection with the identical Athena 24.0 rule for both algorithms,
  the AMVF stored primary, the clean-HS fraction and the HS-track fraction;
* H6: HS longitudinal resolution (3D adaptive beam-spot fit on the
  GNN-assigned tracks, iterative 3-RMS) on the events where both algorithms
  reconstruct the HS within 0.1 mm, also against the local pile-up density.

Before anything is read, the inference metadata is checked against the
frozen regime (peak cuts, density and GNN checkpoints with sha256, windows,
t), so a result from a placeholder configuration cannot be produced. Ratios
are ratios of sums; errors are event bootstraps shared by both algorithms
(``chain_t_summary``). No inference.

    PYTHONPATH=src venv/bin/python -u scripts/hllhc_final_chain.py \
        --metrics outputs/11_12_2026_output/gnn_v2_freeze/metrics/test_s8_v2e527.npz \
        --out outputs/11_12_2026_output/hllhc_final/chain
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.diagnostics.plot_style import (  # noqa: E402
    atlas_label,
    save_figure,
    use_atlas_style,
)
from gnn.evaluation.chain_t_summary import (  # noqa: E402
    MU_BINS,
    ChainRecord,
    load,
    summarize,
)
from pv_finder.evaluation.vertex_finding.itk_resolution import (  # noqa: E402
    iterative_rms,
)
from pv_finder.utils.working_points import get_regime  # noqa: E402

PU200 = "PU200 (185-215)"
MU10 = {f"{lo}-{lo + 10}": (lo, lo + 10 if lo < 190 else 200.5) for lo in range(0, 200, 10)}  # fmt: skip
DENSITY_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 4.0))
N_BOOT = 200
SEED = 20261112
PVF_COLOR, AMVF_COLOR, RULE_COLOR = "#0072B2", "#D55E00", "#E69F00"
# Vertex rule of the headline: a vertex counts only with >= this many tracks
# (PV-Finder: GNN-assigned at t; AMVF: associated). Set from --vertex-min-tracks.
VTX_MIN = {"n": 2}


def sha256(path: Path) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen(meta: dict, gnn_key: str) -> dict:
    """Assert the metrics were made at the frozen regime; return provenance."""
    regime = get_regime("hllhc")
    regime.require_frozen()
    inf = meta["infer_metadata"]
    dens = inf["density"]
    checks = {
        "density_checkpoint": dens["checkpoint"] == regime.density.checkpoint,
        "density_peaks": dens["peaks"] == asdict(regime.peaks),
        "density_sha256": sha256(REPO / regime.density.checkpoint) == dens["sha256"],
        "gnn_key": meta["gnn"] == gnn_key,
        "gnn_checkpoint": inf["gnn"][gnn_key]["path"] == regime.gnn.checkpoint,
        "gnn_sha256": sha256(REPO / regime.gnn.checkpoint) == inf["gnn"][gnn_key]["sha256"],
        "t_in_grid": regime.gnn.t in meta["t_grid"],
        "window": meta["window_mm"] == regime.match_window_mm,
        "hs_window": meta["hs_window_mm"] == regime.hs_match_window_mm,
        "entries": inf["entries_spec"] == "0:25000" and "r16443" in inf["root"],
        "events_hash": meta["events_sha256"] == inf["entries_sha256"],
    }  # fmt: skip
    if not all(checks.values()):
        raise RuntimeError(f"metrics are not at the frozen working point: {checks}")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO)  # fmt: skip
    return {
        "regime": regime.to_dict(),
        "checks": checks,
        "git_head_now": git.stdout.strip(),
        "metrics_git": meta["git"],
        "inference_git": inf["git"],
        "density_sha256": dens["sha256"],
        "gnn_sha256": inf["gnn"][gnn_key]["sha256"],
        "entries_sha256": inf["entries_sha256"],
        "root": inf["root"],
    }


def subset(rec: ChainRecord, ts: tuple[float, ...]) -> ChainRecord:
    """Keep the arms at the given t (plus AMVF) so every table has three columns."""
    grid = list(rec.meta["t_grid"])
    idx = [grid.index(t) for t in ts]
    n_t = len(grid)
    meta = dict(rec.meta, t_grid=list(ts))
    meta["sel_arms"] = [f"t{t}" for t in ts] + ["amvf_stored", "amvf_rule"]
    return replace(
        rec,
        counts=rec.counts[:, idx + [n_t]],
        sel=rec.sel[:, idx + [n_t, n_t + 1]],
        res=rec.res[:, idx + [n_t]],
        fallback=rec.fallback[:, idx],
        meta=meta,
    )


def totals(rec: ChainRecord, mask: np.ndarray) -> dict:
    """Summed counts per arm (min tracks 1 and 2), for the Table 4 rows."""
    out: dict = {}
    for m, mt in enumerate(rec.meta["min_tracks"]):
        out[f"min{mt}"] = {
            arm: {f: int(rec.cf(f, m)[mask, a].sum()) for f in rec.meta["count_fields"]}
            for a, arm in enumerate(rec.arms)
        }
    out["n_truth_ge2"] = int(rec.ev("n_truth_ge2")[mask].sum())
    out["n_peaks"] = int(rec.ev("n_peaks")[mask].sum())
    out["n_events"] = int(mask.sum())
    out["n_tracks"] = int(rec.ev("n_tracks")[mask].sum())
    out["n_labelled"] = int(rec.ev("n_labelled")[mask].sum())
    out["n_hs_tracks"] = int(rec.ev("n_hs_tracks")[mask].sum())
    return out


def hs_res_vs_density(rec: ChainRecord, a: int) -> list[dict]:
    """Iterative-RMS HS width per local-density bin, common events, all mu."""
    dens = rec.ev("local_density")
    r, ra = rec.res[:, a], rec.res[:, -1]
    common = np.isfinite(r) & np.isfinite(ra)
    rng = np.random.default_rng(SEED)
    rows = []
    for lo, hi in DENSITY_BINS:
        sel = common & (dens >= lo) & (dens < hi)
        x, y = r[sel], ra[sel]
        row = {"lo": lo, "hi": hi, "n": int(sel.sum())}
        if len(x) >= 20:
            row["pvf_um"] = 1e3 * iterative_rms(x, 0.1).rms
            row["amvf_um"] = 1e3 * iterative_rms(y, 0.1).rms
            bx, by = [], []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(x), len(x))
                bx.append(iterative_rms(x[i], 0.1).rms)
                by.append(iterative_rms(y[i], 0.1).rms)
            row["pvf_err_um"] = 1e3 * float(np.std(bx, ddof=1))
            row["amvf_err_um"] = 1e3 * float(np.std(by, ddof=1))
        rows.append(row)
    return rows


def headline(s: dict, tot: dict, a: int, label: str) -> dict:
    """Flat, named headline numbers of one bin for arm ``a`` and AMVF."""
    b = s["bins"][label]
    r = b[f"ratios_min{VTX_MIN['n']}"]

    def pair(key: str) -> dict:
        v, e, d = r[key]["value"], r[key]["err"], r[key]["diff_err"]
        return {"gnn": v[a], "gnn_err": e[a], "amvf": v[-1], "amvf_err": e[-1],
                "diff": v[a] - v[-1], "diff_err": d[a]}  # fmt: skip

    out = {k: pair(k) for k in r}
    eff, pur = out["trk_eff"], out["trk_pur"]
    out["trk_f1"] = {
        w: 2 * eff[w] * pur[w] / (eff[w] + pur[w]) for w in ("gnn", "amvf")
    }
    out["trk_wrong_rate"] = {w: 1 - pur[w] for w in ("gnn", "amvf")}
    hsel = b["hs_selection"]
    n_t = len(s["arms"]) - 1
    out["hs_selection"] = {
        "gnn_rule": hsel["value"][a], "gnn_rule_k": hsel["k"][a], "gnn_rule_err": hsel["err"][a],
        "amvf_stored": hsel["value"][n_t], "amvf_stored_k": hsel["k"][n_t],
        "amvf_stored_err": hsel["err"][n_t],
        "amvf_rule": hsel["value"][n_t + 1], "amvf_rule_k": hsel["k"][n_t + 1],
        "amvf_rule_err": hsel["err"][n_t + 1],
        "gnn_minus_stored_err": hsel["diff_stored_err"][a], "n": hsel["n"],
    }  # fmt: skip
    hr = b["hs_resolution"]
    out["hs_resolution_um"] = {
        "gnn": hr["rms_um"][a], "gnn_err": hr["err_um"][a],
        "amvf_common": hr["amvf_um"][a], "amvf_own_events": hr["rms_um"][-1],
        "amvf_own_err": hr["err_um"][-1], "diff_err": hr["diff_err_um"][a],
        "n_common": hr["n"][a],
    }  # fmt: skip
    arm = s["arms"][a]
    t1 = tot[f"min{VTX_MIN['n']}"]
    g, am = t1[arm], t1["amvf"]
    out["table4"] = {
        "vertices_with_tracks": {"gnn": g["n_reco"], "amvf": am["n_reco"]},
        **{c: {"gnn": g[c], "amvf": am[c]} for c in ("clean", "merged", "split", "fake")},
        "pos": {c: {"gnn": g[f"p_{c}"], "amvf": am[f"p_{c}"]}
                for c in ("clean", "merged", "split", "fake", "truth_found")},
        "n_truth_ge2": tot["n_truth_ge2"], "n_peaks": tot["n_peaks"],
        "empty_candidates_dropped": tot["n_peaks"] - tot["min1"][arm]["n_reco"],
        "one_track_candidates_dropped": tot["min1"][arm]["n_reco"] - tot["min2"][arm]["n_reco"],
        "amvf_one_track_dropped": tot["min1"]["amvf"]["n_reco"] - tot["min2"]["amvf"]["n_reco"],
        "vertex_min_tracks": VTX_MIN["n"],
        "fake_rate_all_peaks": (g["fake"] + tot["n_peaks"] - g["n_reco"]) / tot["n_peaks"],
        "n_events": tot["n_events"],
    }  # fmt: skip
    out["hs_vertex_purity"] = {
        "gnn": g["hs_match_hs_tracks"] / g["hs_match_tracks"],
        "amvf": am["hs_match_hs_tracks"] / am["hs_match_tracks"],
    }
    out["mean_mu"] = b["mean_mu"]
    return out


def _series(
    s: dict, key: str, a: int, kind: str = "ratio"
) -> tuple[np.ndarray, np.ndarray]:
    v, e = [], []
    for b in s["bins"].values():
        if kind == "sel":
            x, y = b["hs_selection"]["value"][a], b["hs_selection"]["err"][a]
        elif kind == "res":
            x, y = b["hs_resolution"]["rms_um"][a], b["hs_resolution"]["err_um"][a]
        else:
            r = b[f"ratios_min{VTX_MIN['n']}"]
            x, y = r[key]["value"][a], r[key]["err"][a]
        v.append(np.nan if x is None else x)
        e.append(np.nan if y is None else y)
    return np.array(v, float), np.array(e, float)


def _mu_axes(  # noqa: ANN001
    ax, ylabel: str, t: float | None, desc_extra: str = "", desc_y: float = 0.86
) -> None:
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 200)
    gnn = f"GNN t = {t:g}" if t is not None else "before the GNN"
    desc = r"ITk $t\bar{t}$, $\sqrt{s}$ = 14 TeV, r16443" + f"\n{gnn}{desc_extra}"
    atlas_label(ax, desc=desc, desc_xy=(0.05, desc_y))
    ax.legend(loc="lower right", frameon=False, fontsize="small")


def plot_all(s10: dict, dens_rows: list[dict], t: float, out: Path) -> list[str]:
    """ATLAS-style figures vs mu (10-unit bins) and vs local density."""
    import matplotlib.pyplot as plt

    use_atlas_style()
    x = np.array([(lo + min(hi, 200)) / 2 for lo, hi in MU10.values()])
    xe = 5.0
    a, n = 1, len(s10["arms"]) - 1  # arm order: t0, t, amvf
    paths: list[str] = []
    single = (
        ("clean_per_truth", "Clean / truth vertices [%]", "categories_clean_per_truth_vs_mu"),
        ("clean_per_reco", "Clean / reconstructed [%]", "categories_clean_per_reco_vs_mu"),
        ("merged_per_reco", "Merged / reconstructed [%]", "categories_merged_per_reco_vs_mu"),
        ("split_per_reco", "Split / reconstructed [%]", "categories_split_per_reco_vs_mu"),
        ("fake_per_reco", "Fake / reconstructed [%]", "categories_fake_per_reco_vs_mu"),
        ("pos_eff", "Truth found within 0.17 mm [%]", "positional_eff_vs_mu"),
        ("pos_fake_per_reco", "Positional fake / reconstructed [%]", "positional_fake_vs_mu"),
        ("hs_clean", "Events with a Clean HS vertex [%]", "hs_clean_fraction_vs_mu"),
        ("hs_track_frac", "HS tracks on the HS-matched vertex [%]", "hs_track_fraction_vs_mu"),
        ("trk_eff", "Track-assignment efficiency [%]", "track_efficiency_vs_mu"),
        ("trk_pur", "Track-assignment purity [%]", "track_purity_vs_mu"),
    )  # fmt: skip
    for key, ylab, name in single:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for arm, lab, c, mk in ((a, "PV-Finder + GNN", PVF_COLOR, "o"), (n, "AMVF", AMVF_COLOR, "s")):  # fmt: skip
            v, e = _series(s10, key, arm)
            ax.errorbar(x, 100 * v, xerr=xe, yerr=100 * e, color=c, marker=mk, ms=4, ls="none", label=lab)  # fmt: skip
        ymax = np.nanmax(ax.get_ylim())
        ax.set_ylim(0, 115 if ymax > 60 else ymax * 1.35)
        extra = ", positional taxonomy" if key.startswith("pos_") else ""
        _mu_axes(ax, ylab, t, extra)
        paths += [str(p) for p in save_figure(fig, out, name)]
    # HS selection: GNN rule, AMVF stored primary, AMVF with the same rule
    sel_n = len(s10["sel_arms"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for arm, lab, c, mk in ((a, "PV-Finder + GNN (Athena rule)", PVF_COLOR, "o"),
                            (sel_n - 2, "AMVF stored primary", AMVF_COLOR, "s"),
                            (sel_n - 1, "AMVF (same rule, our code)", RULE_COLOR, "^")):  # fmt: skip
        v, e = _series(s10, "", arm, "sel")
        ax.errorbar(x, 100 * v, xerr=xe, yerr=100 * e, color=c, marker=mk, ms=4, ls="none", label=lab)  # fmt: skip
    ax.set_ylim(80, 106)
    _mu_axes(ax, "HS reconstructed and selected [%]", t, ", HS window 0.1 mm")
    paths += [str(p) for p in save_figure(fig, out, "hs_selection_vs_mu")]
    # HS resolution vs mu (common events per bin) and vs local density
    fig, ax = plt.subplots(figsize=(7, 5.5))
    v, e = _series(s10, "", a, "res")
    va = np.array([b["hs_resolution"]["amvf_um"][a] or np.nan for b in s10["bins"].values()], float)  # fmt: skip
    ax.errorbar(x, v, xerr=xe, yerr=e, color=PVF_COLOR, marker="o", ms=4, ls="none", label="PV-Finder + GNN (3D adaptive fit)")  # fmt: skip
    ax.errorbar(
        x, va, xerr=xe, color=AMVF_COLOR, marker="s", ms=4, ls="none", label="AMVF"
    )
    ax.set_ylim(0, 20)
    _mu_axes(ax, r"HS longitudinal resolution [$\mu$m]", t)
    paths += [str(p) for p in save_figure(fig, out, "hs_resolution_vs_mu")]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    rows = [r for r in dens_rows if "pvf_um" in r]
    xc = np.array([(r["lo"] + r["hi"]) / 2 for r in rows])
    xw = np.array([(r["hi"] - r["lo"]) / 2 for r in rows])
    ax.errorbar(xc, [r["pvf_um"] for r in rows], xerr=xw, yerr=[r["pvf_err_um"] for r in rows],
                color=PVF_COLOR, marker="o", ls="none", label="PV-Finder + GNN (3D adaptive fit)")  # fmt: skip
    ax.errorbar(xc, [r["amvf_um"] for r in rows], xerr=xw, yerr=[r["amvf_err_um"] for r in rows],
                color=AMVF_COLOR, marker="s", ls="none", label="AMVF")  # fmt: skip
    ax.set_xlabel("Local pile-up density [vertices / mm]")
    ax.set_ylabel(r"HS longitudinal resolution [$\mu$m]")
    ax.set_ylim(0, 20)
    atlas_label(ax, desc=r"ITk $t\bar{t}$, $\sqrt{s}$ = 14 TeV, r16443, 0 $\leq\mu\leq$ 200" + f"\nGNN t = {t:g}")  # fmt: skip
    ax.legend(loc="lower right", frameon=False, fontsize="small")
    paths += [str(p) for p in save_figure(fig, out, "hs_resolution_vs_local_density")]
    # Track F1 (no error bars: the statistical error is below 0.1 point per bin)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for arm, lab, c, mk in ((a, "PV-Finder + GNN", PVF_COLOR, "o"), (n, "AMVF", AMVF_COLOR, "s")):  # fmt: skip
        e, _ = _series(s10, "trk_eff", arm)
        p, _ = _series(s10, "trk_pur", arm)
        ax.errorbar(x, 100 * 2 * e * p / (e + p), xerr=xe, color=c, marker=mk, ms=4, ls="none", label=lab)  # fmt: skip
    ax.set_ylim(0, 115)
    _mu_axes(ax, "Track F1 [%]", t)
    paths += [str(p) for p in save_figure(fig, out, "track_f1_vs_mu")]
    return paths


def plot_hs_reco(rec: ChainRecord, t: float, out: Path) -> list[str]:
    """HS reconstruction efficiency (any vertex within 0.1 mm) vs mu, Clopper-Pearson."""
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    use_atlas_style()
    mu, has = rec.ev("mu"), rec.ev("has_hs") > 0
    x = np.array([(lo + min(hi, 200)) / 2 for lo, hi in MU10.values()])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    rows = {}
    for field, lab, c, mk in (("pvf_hs_within", "PV-Finder", PVF_COLOR, "o"),
                              ("amvf_hs_within", "AMVF", AMVF_COLOR, "s")):  # fmt: skip
        flag = rec.ev(field) > 0
        v, lo_e, hi_e, ks, ns = [], [], [], [], []
        for lo, hi in MU10.values():
            m = has & (mu >= lo) & (mu < hi)
            k, nn = int(flag[m].sum()), int(m.sum())
            lo_cp = beta.ppf(0.15865, k, nn - k + 1) if k > 0 else 0.0
            hi_cp = beta.ppf(0.84135, k + 1, nn - k) if k < nn else 1.0
            v.append(k / nn)
            lo_e.append(k / nn - lo_cp)
            hi_e.append(hi_cp - k / nn)
            ks.append(k)
            ns.append(nn)
        rows[field] = {"k": ks, "n": ns}
        ax.errorbar(x, 100 * np.array(v), xerr=5, yerr=[100 * np.array(lo_e), 100 * np.array(hi_e)],
                    color=c, marker=mk, ms=4, ls="none", label=lab)  # fmt: skip
    ax.set_ylim(95, 101)
    _mu_axes(ax, "HS reconstruction efficiency [%]", None, ", HS window 0.1 mm", 0.45)
    paths = [str(p) for p in save_figure(fig, out, "hs_reco_eff_vs_mu")]
    (out / "hs_reco_eff_vs_mu.json").write_text(json.dumps(rows) + "\n")
    return paths


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", required=True, type=Path)
    ap.add_argument("--gnn", default="v2e527")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--vertex-min-tracks", type=int, default=2, choices=(1, 2),
        help="headline vertex rule (identical for both algorithms); must equal "
        "the HS rule the metrics were made with (chain_t_metrics --hs-min-tracks)",
    )  # fmt: skip
    a = ap.parse_args()
    VTX_MIN["n"] = a.vertex_min_tracks
    a.out.mkdir(parents=True, exist_ok=True)
    full = load(a.metrics)  # asserts the per-event bookkeeping identities
    prov = verify_frozen(full.meta, a.gnn)
    if full.meta.get("hs_min_tracks", 1) != a.vertex_min_tracks:
        raise RuntimeError("HS quantities were made with another vertex rule")
    prov["vertex_min_tracks"] = a.vertex_min_tracks
    t = get_regime("hllhc").gnn.t
    rec = subset(full, (0.0, t))  # arms: t = 0 (no cut, reference), frozen t, AMVF
    mu = rec.ev("mu")
    masks = {lab: (mu >= lo) & (mu < hi) for lab, (lo, hi) in MU_BINS.items()}
    if int(masks[PU200].sum()) != 1920:
        raise RuntimeError("PU200 event count is not 1,920")
    s = summarize(rec, n_boot=N_BOOT, seed=SEED)
    s10 = summarize(rec, n_boot=N_BOOT, seed=SEED, bins=MU10)
    tots = {lab: totals(rec, m) for lab, m in masks.items()}
    dens_rows = hs_res_vs_density(rec, 1)
    result = {
        "description": __doc__.splitlines()[0],
        "provenance": dict(
            prov, metrics=str(a.metrics), metrics_sha256=sha256(a.metrics)
        ),
        "t": t,
        "arms": rec.arms,
        "headline": {
            lab: headline(s, tots[lab], 1, lab) for lab in ("inclusive", PU200)
        },
        "headline_t0": {
            lab: headline(s, tots[lab], 0, lab) for lab in ("inclusive", PU200)
        },
        "totals": tots,
        "hs_resolution_vs_local_density": dens_rows,
        "summary_coarse": s,
        "summary_mu10": s10,
    }
    result["plots"] = plot_all(s10, dens_rows, t, a.out / "plots")
    result["plots"] += plot_hs_reco(rec, t, a.out / "plots")
    (a.out / "chain_headline.json").write_text(json.dumps(result, indent=1) + "\n")
    h = result["headline"][PU200]
    print(f"PU200 Clean/truth {h['clean_per_truth']['gnn']:.4f} vs AMVF {h['clean_per_truth']['amvf']:.4f}")  # fmt: skip
    print(f"HS sel {h['hs_selection']['gnn_rule']:.4f} vs AMVF stored {h['hs_selection']['amvf_stored']:.4f}")  # fmt: skip
    print(f"HS res {h['hs_resolution_um']['gnn']:.2f} vs {h['hs_resolution_um']['amvf_common']:.2f} um")  # fmt: skip
    print(f"wrote {a.out / 'chain_headline.json'}")


if __name__ == "__main__":
    main()
