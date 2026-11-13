#!/usr/bin/env python3
"""Tables and ATLAS-style figures for the supplementary HL-LHC studies (analysis pass).

Reads ``per_event.npz`` of ``hllhc_final_supplementary.py`` and the
``chain_t_metrics`` record of the same 25,000 r16443 events, and writes

* ``hs_vs_density.json``: HS reconstruction efficiency (0.1 mm), HS selection
  efficiency (identical Athena rule; AMVF stored primary), HS longitudinal
  resolution (iterative 3-RMS from +-0.1 mm) in bins of the local pile-up
  density, ITk definition (all truth vertices, new file) and old stored-truth
  definition, all mu and PU200 (185 <= mu <= 215);
* ``classes.json``: Clean / Merged / Split / Fake vertices per event vs mu
  (10-unit bins) and at PU200, PV-Finder + GNN (t = 0.97, drop-empty) and
  AMVF, same-rule unweighted (primary) and weighted (cross-check);
* ``supplementary_summary.json`` and the figures under ``plots/main`` and
  ``plots/backup``: one quantity per figure, PDF + PNG.

Efficiencies carry Clopper-Pearson 68.27% intervals; differences and widths
use event bootstraps shared by both algorithms (paired).

    PYTHONPATH=src venv/bin/python -u scripts/hllhc_final_supplementary_plots.py \
        --sup outputs/11_12_2026_output/hllhc_final/supplementary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.diagnostics.plot_style import (  # noqa: E402
    ALGO_COLORS,
    atlas_label,
    save_figure,
    use_atlas_style,
)
from gnn.evaluation.chain_t_summary import boot_weights, load  # noqa: E402
from pv_finder.evaluation.vertex_finding.itk_resolution import (  # noqa: E402
    iterative_rms,
)
from pv_finder.utils.working_points import get_regime  # noqa: E402

REGIME = get_regime("hllhc")
T, HS_WIN = REGIME.gnn.t, REGIME.hs_match_window_mm
METRICS = REPO / "outputs/11_12_2026_output/gnn_v2_freeze/metrics/test_s8_v2e527.npz"
N_BOOT, SEED, CL = 200, 20261112, 0.6827
PU200 = (185.0, 215.0)  # inclusive
MU10 = [(lo, lo + 10) for lo in range(0, 200, 10)]  # last bin includes mu = 200
# Densities are multiples of 0.25 / mm (a count over 4 mm); bins [lo, hi).
# Chosen so that every all-mu bin holds >= MIN_BIN events (max density 3.75).
DENS_EDGES = {
    ("itk", "all_mu"): (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0),
    ("old", "all_mu"): (0.0, 0.5, 1.0, 1.5, 2.0, 4.0),
    ("itk", "pu200"): (0.0, 1.0, 1.5, 2.0, 2.5, 4.0),
    ("old", "pu200"): (0.0, 0.5, 1.0, 1.5, 4.0),
}
MIN_BIN = 100  # events per density bin (all mu), asserted
CLASSES = ("clean", "merged", "split", "fake")
PVF_C, AMVF_C, RULE_C = ALGO_COLORS["pvf_gnn"], ALGO_COLORS["amvf"], "#E69F00"
SAMPLE = r"ITk $t\bar{t}$, $\sqrt{s}$ = 14 TeV, r16443"
DENS_LABEL = {
    "itk": "Local pile-up density [vertices / mm]",
    "old": "Local pile-up density, stored truth [vertices / mm]",
}


def clopper_pearson(k: int, n: int, cl: float = CL) -> tuple[float, float]:
    """Central Clopper-Pearson interval of k successes in n trials."""
    a = (1 - cl) / 2
    lo = beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def eff(x: np.ndarray) -> dict:
    """Efficiency of a boolean per-event array with its CP interval."""
    k, n = int(x.sum()), len(x)
    lo, hi = clopper_pearson(k, n) if n else (np.nan, np.nan)
    return {"k": k, "n": n, "value": k / n if n else np.nan, "lo": lo, "hi": hi}


def paired_err(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Bootstrap error of mean(x) - mean(y) with shared event weights."""
    d = (w @ (x.astype(float) - y.astype(float))) / w.sum(1)
    return float(d.std(ddof=1))


def resolution(r: np.ndarray, ra: np.ndarray, rng: np.random.Generator) -> dict:
    """Iterative-RMS HS width [um], PV-Finder and AMVF on common events, paired bootstrap."""
    out: dict = {"n": len(r)}
    if len(r) < 20:
        return out
    f = lambda x: 1e3 * iterative_rms(x, HS_WIN).rms  # noqa: E731
    b = np.array([[f(r[i]), f(ra[i])] for i in (rng.integers(0, len(r), len(r)) for _ in range(N_BOOT))])  # fmt: skip
    out.update(
        pvf_um=f(r), amvf_um=f(ra), pvf_err_um=float(b[:, 0].std(ddof=1)),
        amvf_err_um=float(b[:, 1].std(ddof=1)), diff_err_um=float((b[:, 0] - b[:, 1]).std(ddof=1)),
    )  # fmt: skip
    return out


def hs_vs_density(
    sup: dict, rec, dens: np.ndarray, mask: np.ndarray, edges: tuple
) -> list[dict]:  # noqa: ANN001
    """One row per density bin: HS reco, selection and resolution, both algorithms."""
    sf = rec.meta["sel_fields"].index("reco_sel")
    arms = rec.meta["sel_arms"]
    a_t = rec.meta["t_grid"].index(T)
    sel = {k: rec.sel[:, arms.index(a), sf] > 0 for k, a in
           (("pvf_rule", f"t{T}"), ("amvf_stored", "amvf_stored"), ("amvf_rule", "amvf_rule"))}  # fmt: skip
    reco = {k: sup[k] > 0 for k in ("pvf_any", "pvf_ge1", "pvf_ge2", "amvf_ge1", "amvf_ge2")}  # fmt: skip
    rng = np.random.default_rng(SEED)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = mask & (sup["has_hs"] > 0) & (dens >= lo) & (dens < hi)
        w = boot_weights(int(m.sum()), N_BOOT, SEED)
        row: dict = {"lo": lo, "hi": hi, "n_events": int(m.sum()),
                     "mean_density": float(dens[m].mean()) if m.any() else np.nan,
                     "mean_mu": float(sup["mu"][m].mean()) if m.any() else np.nan}  # fmt: skip
        row["reco"] = {k: eff(v[m]) for k, v in reco.items()}
        row["reco"]["diff_err_pvf_any_minus_amvf_ge2"] = paired_err(reco["pvf_any"][m], reco["amvf_ge2"][m], w)  # fmt: skip
        row["sel"] = {k: eff(v[m]) for k, v in sel.items()}
        for k in ("amvf_rule", "amvf_stored"):
            row["sel"][f"diff_err_pvf_rule_minus_{k}"] = paired_err(sel["pvf_rule"][m], sel[k][m], w)  # fmt: skip
        r, ra = rec.res[m, a_t], rec.res[m, -1]
        common = np.isfinite(r) & np.isfinite(ra)
        row["res"] = resolution(r[common], ra[common], rng)
        rows.append(row)
    return rows


def class_table(cls: np.ndarray, arms: list[str], fields: list[str], mask: np.ndarray) -> dict:  # fmt: skip
    """Mean vertices per event per class and arm, error std / sqrt(N)."""
    x = cls[mask]
    n = len(x)
    out: dict = {"n_events": n}
    for a, arm in enumerate(arms):
        out[arm] = {}
        for c in (*CLASSES, "n_reco", "n_zero_weight"):
            v = x[:, a, fields.index(c)].astype(float)
            out[arm][c] = {"mean": v.mean(), "err": v.std(ddof=1) / np.sqrt(n), "sum": int(v.sum())}  # fmt: skip
    return out


def class_change(cls: np.ndarray, arms: list[str], fields: list[str], mask: np.ndarray, new: str, ref: str) -> dict:  # fmt: skip
    """Per-event change new - ref per class: absolute, paired error, relative."""
    x = cls[mask]
    out = {}
    for c in (*CLASSES, "n_reco"):
        k = fields.index(c)
        d = (x[:, arms.index(new), k] - x[:, arms.index(ref), k]).astype(float)
        base = x[:, arms.index(ref), k].mean()
        out[c] = {"abs_per_event": d.mean(), "abs_err": d.std(ddof=1) / np.sqrt(len(d)),
                  "relative": d.mean() / base if base else np.nan,
                  "events_changed": float((d != 0).mean())}  # fmt: skip
    return out


def fit_track_stats(sup: dict, mask: np.ndarray) -> dict:
    """How AMVF fit tracks differ from the TTVA links (vertex-track pairs)."""
    s = {k: float(sup[k][mask].sum()) for k in ("n_fit_pairs", "n_fit_not_assoc", "w_fit_not_assoc", "w_fit",
                                                 "n_assoc_pairs", "n_assoc_not_fit", "n_fit_tracks",
                                                 "n_fit_tracks_shared", "n_trk")}  # fmt: skip
    return {
        "frac_fit_pairs_not_in_assoc": s["n_fit_not_assoc"] / s["n_fit_pairs"],
        "frac_fit_weight_not_in_assoc": s["w_fit_not_assoc"] / s["w_fit"],
        "frac_assoc_pairs_not_in_fit": s["n_assoc_not_fit"] / s["n_assoc_pairs"],
        "frac_fit_tracks_shared": s["n_fit_tracks_shared"] / s["n_fit_tracks"],
        "fit_pairs_per_event": s["n_fit_pairs"] / mask.sum(),
        "assoc_pairs_per_event": s["n_assoc_pairs"] / mask.sum(),
        "frac_tracks_in_any_fit": s["n_fit_tracks"] / s["n_trk"], "sums": s,
    }  # fmt: skip


def dump(x: object, path: Path) -> None:
    """Write JSON with numpy types converted and non-finite floats as null."""

    def clean(v: object) -> object:
        if isinstance(v, np.ndarray):
            v = v.tolist()
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, dict):
            return {k: clean(u) for k, u in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(u) for u in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    path.write_text(json.dumps(clean(x), indent=1, allow_nan=False) + "\n")


# ----------------------------------------------------------------- plotting
def _axes(ax, xlabel: str, ylabel: str, desc: str, loc: str = "lower left") -> None:  # noqa: ANN001
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    atlas_label(ax, desc=desc, desc_xy=(0.05, 0.86))
    ax.legend(loc=loc, frameon=False, fontsize="small")


def _xbins(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.array([r["lo"] for r in rows]), np.array([r["hi"] for r in rows])
    return (lo + hi) / 2, (hi - lo) / 2


def _eff_series(
    ax,
    rows: list[dict],
    path: tuple[str, str],
    label: str,
    c: str,
    mk: str,
    dx: float = 0.0,
) -> None:  # noqa: ANN001
    x, xe = _xbins(rows)
    e = [r[path[0]][path[1]] for r in rows]
    v = np.array([q["value"] for q in e])
    lo, hi = np.array([q["lo"] for q in e]), np.array([q["hi"] for q in e])
    ax.errorbar(x + dx, 100 * v, xerr=xe, yerr=100 * np.array([v - lo, hi - v]), color=c, marker=mk, ms=5, ls="none", label=label)  # fmt: skip


def plot_density(
    rows: list[dict], kind: str, desc: str, out: Path, tag: str
) -> list[str]:
    """HS reco, selection, resolution vs density: three standalone figures."""
    import matplotlib.pyplot as plt

    xl = DENS_LABEL[kind]
    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _eff_series(ax, rows, ("reco", "pvf_any"), "PV-Finder + GNN", PVF_C, "o")
    _eff_series(ax, rows, ("reco", "amvf_ge2"), "AMVF", AMVF_C, "s")
    ax.set_ylim(96, 102)
    ax.set_xlim(0, 4)
    _axes(
        ax, xl, "HS reconstruction efficiency [%]", desc + f"\nHS within {HS_WIN:g} mm"
    )
    paths += map(str, save_figure(fig, out, f"hs_reco_eff_vs_density_{tag}"))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _eff_series(ax, rows, ("sel", "pvf_rule"), "PV-Finder + GNN", PVF_C, "o")
    _eff_series(ax, rows, ("sel", "amvf_rule"), "AMVF", AMVF_C, "s")
    _eff_series(ax, rows, ("sel", "amvf_stored"), "AMVF (stored primary)", RULE_C, "^", 0.03)  # fmt: skip
    ax.set_ylim(82, 106)
    ax.set_xlim(0, 4)
    _axes(ax, xl, "HS selection efficiency [%]", desc + "\nidentical Athena selection rule")  # fmt: skip
    paths += map(str, save_figure(fig, out, f"hs_sel_eff_vs_density_{tag}"))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ok = [r for r in rows if "pvf_um" in r["res"]]
    x, xe = _xbins(ok)
    for k, lab, c, mk in (("pvf", "PV-Finder + GNN", PVF_C, "o"), ("amvf", "AMVF", AMVF_C, "s")):  # fmt: skip
        ax.errorbar(x, [r["res"][f"{k}_um"] for r in ok], xerr=xe, yerr=[r["res"][f"{k}_err_um"] for r in ok],
                    color=c, marker=mk, ms=5, ls="none", label=lab)  # fmt: skip
    ax.set_ylim(0, 25)
    ax.set_xlim(0, 4)
    _axes(ax, xl, r"HS longitudinal resolution [$\mu$m]", desc + "\nevents with both HS vertices", "lower right")  # fmt: skip
    paths += map(str, save_figure(fig, out, f"hs_resolution_vs_density_{tag}"))
    return paths


def plot_hs_reco_variants(rows: list[dict], desc: str, out: Path) -> list[str]:
    """Backup: HS reco efficiency with PV-Finder peaks required to hold GNN tracks."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _eff_series(ax, rows, ("reco", "pvf_any"), "PV-Finder + GNN (any peak)", PVF_C, "o")
    _eff_series(ax, rows, ("reco", "pvf_ge2"), "PV-Finder + GNN (peak with >= 2 tracks)", "#56B4E9", "v", 0.03)  # fmt: skip
    _eff_series(ax, rows, ("reco", "amvf_ge2"), "AMVF (>= 2 tracks)", AMVF_C, "s")
    ax.set_ylim(96, 102)
    ax.set_xlim(0, 4)
    _axes(ax, DENS_LABEL["itk"], "HS reconstruction efficiency [%]", desc + f"\nHS within {HS_WIN:g} mm")  # fmt: skip
    return list(map(str, save_figure(fig, out, "hs_reco_eff_vs_density_itk_track_requirement")))  # fmt: skip


def plot_classes_vs_mu(rows: list[dict], arms: tuple[str, str], labels: tuple[str, str], desc: str, out: Path, tag: str) -> list[str]:  # fmt: skip
    """One figure per class: vertices per event vs mu."""
    import matplotlib.pyplot as plt

    x = np.array([(lo + hi) / 2 for lo, hi in MU10])
    paths: list[str] = []
    for c in CLASSES:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for arm, lab, col, mk in zip(arms, labels, (PVF_C, AMVF_C), ("o", "s")):
            ax.errorbar(x, [r[arm][c]["mean"] for r in rows], xerr=5, yerr=[r[arm][c]["err"] for r in rows],
                        color=col, marker=mk, ms=4, ls="none", label=lab)  # fmt: skip
        ax.set_xlim(0, 200)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
        _axes(ax, r"$\mu$", f"{c.capitalize()} vertices / event", desc, "upper right")
        paths += map(str, save_figure(fig, out, f"{c}_per_event_vs_mu_{tag}"))
    return paths


def plot_bars(tab: dict, arms: list[str], labels: list[str], styles: list[tuple[str, str]], desc: str, out: Path, name: str, loc: str = "upper right") -> list[str]:  # fmt: skip
    """Grouped bars: vertices per event per class at PU200."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    width = 0.8 / len(arms)
    xs = np.arange(len(CLASSES))
    for j, (arm, lab, (col, hatch)) in enumerate(zip(arms, labels, styles)):
        v = np.array([tab[arm][c]["mean"] for c in CLASSES])
        e = np.array([tab[arm][c]["err"] for c in CLASSES])
        pos = xs - 0.4 + width * (j + 0.5)
        ax.bar(pos, v, width, yerr=e, color=col if not hatch else "white", edgecolor=col, hatch=hatch, label=lab, capsize=2)  # fmt: skip
        for p, q in zip(pos, v):
            ax.text(p, q + 1, f"{q:.1f}", ha="center", va="bottom", fontsize=7 if len(arms) > 2 else 9)  # fmt: skip
    ax.set_xticks(xs)
    ax.set_xticklabels([c.capitalize() for c in CLASSES])
    ax.set_ylim(0, max(tab[a]["clean"]["mean"] for a in arms) * 1.55)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    _axes(ax, "", "Vertices / event", desc, loc)
    return list(map(str, save_figure(fig, out, name)))


def plot_backup_distributions(sup: dict, whist: np.ndarray, wb: np.ndarray, out: Path) -> list[str]:  # fmt: skip
    """Backup: the two density definitions, and the AMVF fit-weight distribution."""
    import matplotlib.pyplot as plt

    m = sup["has_hs"] > 0
    fig, ax = plt.subplots(figsize=(7, 5.5))
    b = np.arange(0, 5.01, 0.25)
    for k, lab, c in (("dens_itk", "All truth vertices (new ntuple)", PVF_C), ("dens_old", "Stored truth vertices (old ntuple)", AMVF_C)):  # fmt: skip
        x = np.clip(sup[k][m], 0, 4.99)
        w = np.full(len(x), 1 / len(x))
        ax.hist(x, b, weights=w, histtype="step", lw=2, color=c, label=lab)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    _axes(ax, "Local pile-up density [vertices / mm]", "Fraction of events / 0.25 per mm", SAMPLE + r", 0 $\leq\mu\leq$ 200", "upper right")  # fmt: skip
    paths = list(map(str, save_figure(fig, out, "local_density_definitions")))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    h = whist.reshape(-1, 20).sum(1)
    ax.stairs(h / h.sum(), wb[::20], color=AMVF_C, lw=2, label="AMVF fit tracks")
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 30)
    _axes(ax, "AMVF track fit weight", "Fraction / 0.02", SAMPLE + r", 0 $\leq\mu\leq$ 200", "upper center")  # fmt: skip
    return paths + list(map(str, save_figure(fig, out, "amvf_fit_weight_distribution")))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--sup", type=Path, required=True, help="compute-pass output directory"
    )
    a = ap.parse_args()
    with np.load(a.sup / "per_event.npz") as f:
        z = {k: f[k] for k in f.files}
    pm = json.loads((a.sup / "per_event.json").read_text())
    ident = json.loads((a.sup / "identity_check.json").read_text())
    rec = load(METRICS)  # asserts the chain bookkeeping
    if len(z["ev"]) != len(rec.events) or not np.array_equal(
        z["ev"][:, 0], rec.ev("entry")
    ):
        raise RuntimeError("per-event table and chain metrics are not the same events")
    if pm["provenance"]["regime"] != json.loads(json.dumps(REGIME.to_dict())):
        raise RuntimeError("compute pass was run at a different regime")
    sup = {k: z["ev"][:, i] for i, k in enumerate(pm["ev_fields"])}
    mu = sup["mu"]
    masks = {"all_mu": np.ones(len(mu), bool), "pu200": (mu >= PU200[0]) & (mu <= PU200[1])}  # fmt: skip
    prov = dict(
        pm["provenance"],
        analysis_script=str(Path(__file__).resolve().relative_to(REPO)),
    )
    # ---- HS vs density
    hs: dict = {"provenance": prov, "definitions": {
        "itk": "truth vertices of the NEW ntuple (all, incl. 0 selected tracks) within +-2 mm of the true HS z, HS excluded, / 4 mm",
        "old": "same with the OLD ntuple's stored truth list (vertices with >= 1 selected track)",
        "reco_pvf": f"any s8 peak within {HS_WIN} mm of the true HS (chain_t_metrics pvf_hs_within)",
        "reco_amvf": f"AMVF vertex with >= 2 TTVA tracks within {HS_WIN} mm (chain_t_metrics amvf_hs_within)",
        "sel": "reco_sel of chain_t_metrics: selected vertex (identical Athena rule) is the HS-matched one and within 0.1 mm",
        "res": "PV-Finder 3D adaptive fit of GNN tracks at t; AMVF closest >= 2-track vertex; both within 0.1 mm; iterative 3-RMS from +-0.1 mm",
        "events": "has_hs == 1 only", "errors": "efficiencies CP 68.27%; widths and differences paired event bootstrap (200)",
    }, "bins": {}}  # fmt: skip
    for kind in ("itk", "old"):
        for mk, m in masks.items():
            edges = DENS_EDGES[(kind, mk)]
            if np.nanmax(sup[f"dens_{kind}"][m]) >= edges[-1]:
                raise RuntimeError(f"{kind} density beyond the last bin edge")
            rows = hs_vs_density(sup, rec, sup[f"dens_{kind}"], m, edges)
            hs["bins"][f"{kind}_{mk}"] = rows
            if mk == "all_mu" and min(r["n_events"] for r in rows) < MIN_BIN:
                raise RuntimeError(f"a {kind} density bin has < {MIN_BIN} events: {[r['n_events'] for r in rows]}")  # fmt: skip
    dump(hs, a.sup / "hs_vs_density.json")
    # ---- vertex classes
    arms, fields = pm["arms"], pm["cls_fields"]
    cls = z["cls"]
    mu_rows = [class_table(cls, arms, fields, (mu >= lo) & ((mu < hi) if hi < 200 else (mu <= hi))) for lo, hi in MU10]  # fmt: skip
    for k, r in zip(MU10, mu_rows):
        r["mu_bin"] = list(k)
    ct: dict = {"provenance": prov, "arms": {
        "pvf_unit": f"PRIMARY PV-Finder + GNN: top-1 tracks at t = {T}, drop-empty, unweighted same rule",
        "amvf_assoc": "PRIMARY AMVF: stored TTVA assocTracks (np.unique), unweighted same rule",
        "pvf_score": "CROSS-CHECK PV-Finder + GNN: tracks weighted by their GNN score",
        "amvf_fitw": "CROSS-CHECK AMVF: fitTracks weighted by fitWeights, Split ranked by sum(w pT^2)",
        "amvf_fit_unit": "diagnostic AMVF: fitTracks, unit weights", "amvf_fitw_unwpt2": "diagnostic AMVF: fit weights, Split ranked by unweighted sum(pT^2)",
    }, "vs_mu": mu_rows}  # fmt: skip
    changes = {("amvf_fitw", "amvf_assoc"): "amvf_fitweights_vs_ttva", ("amvf_fit_unit", "amvf_assoc"): "amvf_fit_list_only",
               ("amvf_fitw", "amvf_fit_unit"): "amvf_weighting_only", ("pvf_score", "pvf_unit"): "pvf_score_weights",
               ("amvf_fitw_unwpt2", "amvf_fitw"): "amvf_split_ranking_unweighted"}  # fmt: skip
    for mk, m in masks.items():
        ct[mk] = class_table(cls, arms, fields, m)
        ct[f"{mk}_changes"] = {lab: class_change(cls, arms, fields, m, *pair) for pair, lab in changes.items()}  # fmt: skip
        ct[f"{mk}_fit_tracks"] = fit_track_stats(sup, m)
    dump(ct, a.sup / "classes.json")
    # ---- figures
    use_atlas_style()
    main_d, back_d = a.sup / "plots/main", a.sup / "plots/backup"
    d_all = SAMPLE + "\n" + r"0 $\leq\mu\leq$ 200" + f", GNN t = {T:g}"
    d_pu = SAMPLE + "\n" + r"185 $\leq\mu\leq$ 215" + f", GNN t = {T:g}"
    figs = {"main": [], "backup": []}
    figs["main"] += plot_density(hs["bins"]["itk_all_mu"], "itk", d_all, main_d, "itk")
    figs["backup"] += plot_density(
        hs["bins"]["old_all_mu"], "old", d_all, back_d, "old"
    )
    figs["backup"] += plot_density(
        hs["bins"]["itk_pu200"], "itk", d_pu, back_d, "itk_pu200"
    )
    figs["backup"] += plot_density(
        hs["bins"]["old_pu200"], "old", d_pu, back_d, "old_pu200"
    )
    figs["backup"] += plot_hs_reco_variants(hs["bins"]["itk_all_mu"], d_all, back_d)
    lab2 = ("PV-Finder + GNN", "AMVF")
    figs["main"] += plot_classes_vs_mu(mu_rows, ("pvf_unit", "amvf_assoc"), lab2, (SAMPLE + f", GNN t = {T:g}") + "\ntrack-content classification", main_d, "primary")  # fmt: skip
    wlab = ("PV-Finder + GNN (GNN-score weights)", "AMVF (fit weights)")
    figs["backup"] += plot_classes_vs_mu(mu_rows, ("pvf_score", "amvf_fitw"), wlab, (SAMPLE + f", GNN t = {T:g}") + "\nweighted classification", back_d, "weighted")  # fmt: skip
    pu = ct["pu200"]
    figs["main"] += plot_bars(pu, ["pvf_unit", "amvf_assoc"], list(lab2), [(PVF_C, ""), (AMVF_C, "")], d_pu + "\ntrack-content classification", main_d, "classes_per_event_pu200_primary")  # fmt: skip
    figs["backup"] += plot_bars(pu, ["pvf_score", "amvf_fitw"], list(wlab), [(PVF_C, ""), (AMVF_C, "")], d_pu + "\nweighted classification", back_d, "classes_per_event_pu200_weighted")  # fmt: skip
    figs["backup"] += plot_bars(
        pu, ["pvf_unit", "pvf_score", "amvf_assoc", "amvf_fitw"],
        ["PV-Finder + GNN (unweighted)", "PV-Finder + GNN (GNN-score weights)", "AMVF (TTVA links, unweighted)", "AMVF (fit tracks, fit weights)"],
        [(PVF_C, ""), (PVF_C, "//"), (AMVF_C, ""), (AMVF_C, "//")], d_pu, back_d, "classes_per_event_pu200_all_variants", "center right",
    )  # fmt: skip
    figs["backup"] += plot_backup_distributions(sup, z["whist"], z["w_bins"], back_d)
    summary = {
        "provenance": prov, "identity_check": ident["result"], "gates_vs_chain_t_metrics": ident["gates_vs_chain_t_metrics"],
        "hs_vs_density": {k: [{"bin": f"{r['lo']}-{r['hi']}", "n": r["n_events"],
                               "reco_pvf": r["reco"]["pvf_any"]["value"], "reco_amvf": r["reco"]["amvf_ge2"]["value"],
                               "sel_pvf": r["sel"]["pvf_rule"]["value"], "sel_amvf_rule": r["sel"]["amvf_rule"]["value"],
                               "sel_amvf_stored": r["sel"]["amvf_stored"]["value"], "res_pvf_um": r["res"].get("pvf_um"),
                               "res_amvf_um": r["res"].get("amvf_um"), "n_res": r["res"]["n"]} for r in rows]
                          for k, rows in hs["bins"].items()},
        "classes_pu200": {arm: {c: pu[arm][c]["mean"] for c in (*CLASSES, "n_reco")} for arm in arms},
        "classes_inclusive": {arm: {c: ct["all_mu"][arm][c]["mean"] for c in (*CLASSES, "n_reco")} for arm in arms},
        "changes_pu200": ct["pu200_changes"], "changes_inclusive": ct["all_mu_changes"],
        "fit_tracks_pu200": ct["pu200_fit_tracks"], "fit_tracks_inclusive": ct["all_mu_fit_tracks"],
        "figures": figs,
    }  # fmt: skip
    dump(summary, a.sup / "supplementary_summary.json")  # fmt: skip
    print((a.sup / "supplementary_summary.json").read_text()[:4000])
    print("done", flush=True)


if __name__ == "__main__":
    main()
