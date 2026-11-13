#!/usr/bin/env python3
"""Aggregate the chain-extras replay (H3, H7) into ``chain_extras_summary.json``.

Every ratio is a ratio of sums over events; errors are the standard deviation
over 200 event-bootstrap replicas (``chain_t_summary.boot_weights``, same seed
and the same replicas for every arm and for the GNN t scan of
``chain_t_metrics``, so differences are paired). mu bins: ``MU_BINS`` of
``chain_t_summary`` plus 10-unit bins.

    PYTHONPATH=src python -u scripts/hllhc_final_chain_extras_summary.py \
        --events outputs/11_12_2026_output/hllhc_final/chain_extras/chain_extras_events.npz \
        --ref outputs/11_12_2026_output/gnn_v2_freeze/metrics/test_s8_v2e527.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.data.root_to_graphs import compute_pv_sigma_preset  # noqa: E402
from gnn.evaluation.chain_inference import git_state  # noqa: E402
from gnn.evaluation.chain_t_summary import (  # noqa: E402
    MU_BINS,
    boot_weights,
    load,
)
from pv_finder.data.resolution_presets import RESOLUTION_PRESETS  # noqa: E402

PU200 = "PU200 (185-215)"
SAMPLES = ("inclusive", PU200)
MU10 = {f"{lo}-{lo + 10}": (lo, lo + 10 if lo < 190 else 200.5) for lo in range(0, 200, 10)}  # fmt: skip
N_BOOT, SEED = 200, 20261112
VAR_BINS: dict[str, list[float]] = {
    "n_trk": [2, 3, 4, 5, 6, 8, 11, 16, 21, 31, 51, 81, np.inf],
    "density": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf],
    "dz_nearest": [
        0.0,
        0.1,
        0.17,
        0.25,
        0.35,
        0.5,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        5.0,
        np.inf,
    ],  # fmt: skip
}
ARM_RATIOS: dict[str, tuple[str, str]] = {
    # (numerator, denominator); event fields are broadcast over arms
    "clean_per_truth": ("clean", "n_truth_ge2"),
    "clean_ge2_per_truth": ("clean_dom_ge2", "n_truth_ge2"),
    "clean_per_reco": ("clean", "n_reco"),
    "merged_per_reco": ("merged", "n_reco"),
    "split_per_reco": ("split", "n_reco"),
    "fake_per_reco": ("fake", "n_reco"),
    "reco_per_truth": ("n_reco", "n_truth_ge2"),
    "trk_eff": ("n_correct", "n_labelled"),
    "trk_pur": ("n_correct", "n_assigned_truth"),
    "wrong_rate": ("wrong", "n_assigned_truth"),
    "assigned_truth_frac": ("n_assigned_truth", "n_labelled"),
    "labelled_unassigned_frac": ("n_labelled_unassigned", "n_labelled"),
    "lost_unassigned_frac": ("n_lost_unassigned", "n_labelled"),
    "lost_share_of_unassigned": ("n_lost_unassigned", "n_labelled_unassigned"),
    "unlabelled_frac": ("n_unlabelled", "n_tracks"),
    "unlabelled_left_unassigned": ("n_unlabelled_unassigned", "n_unlabelled"),
    "assigned_frac": ("n_assigned", "n_tracks"),
}
EVENT_RATIOS: dict[str, tuple[str, str]] = {
    "finder_cap_pvf": ("o2o_pvf", "n_truth_ge2"),
    "finder_cap_amvf": ("o2o_amvf", "n_truth_ge2"),
    "o2o_gnn_dropempty": ("o2o_gnn", "n_truth_ge2"),
    "eff_pvf": ("mv_found_pvf", "n_truth_ge2"),
    "eff_amvf": ("mv_found_amvf", "n_truth_ge2"),
}
PAIRS = (("finder_cap_pvf", "finder_cap_amvf"), ("eff_pvf", "eff_amvf"), ("o2o_gnn_dropempty", "finder_cap_amvf"))  # fmt: skip


def _jsonable(x):  # noqa: ANN001, ANN202
    """JSON-safe copy (recurses into lists; non-finite floats -> None)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _ratio(num: np.ndarray, den: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(value, bootstrap replicas) of sum(num) / sum(den) over axis 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum(0) / den.sum(0), (w @ num) / (w @ den)


def _pack(value: np.ndarray, reps: np.ndarray, ref: dict[str, int]) -> dict:
    """value, err and paired errors of (column - column ref[k]) for each k."""
    out = {"value": value, "err": np.nanstd(reps, 0, ddof=1)}
    for k, c in ref.items():
        out[f"diff_{k}_err"] = np.nanstd(reps - reps[:, c : c + 1], 0, ddof=1)
    return out


def arm_tables(ev: dict, counts: np.ndarray, cf: list[str], mask: np.ndarray, w: np.ndarray, arms: list[str]) -> dict:  # fmt: skip
    """ARM_RATIOS, F1 and totals per arm for min-tracks 1 and 2."""
    ref = {"amvf": arms.index("amvf"), "gnn_t": arms.index("gnn_t")}
    out = {}
    for m, mt in enumerate((1, 2)):
        c = counts[mask, :, m, :].astype(np.float64)

        def get(f: str, c: np.ndarray = c) -> np.ndarray:
            if f in cf:
                return c[..., cf.index(f)]
            return np.repeat(ev[f][mask, None], len(arms), 1)

        tab, reps = {}, {}
        for name, (a, b) in ARM_RATIOS.items():
            v, r = _ratio(get(a), get(b), w)
            tab[name], reps[name] = _pack(v, r, ref), r
        e, p = tab["trk_eff"]["value"], tab["trk_pur"]["value"]
        f1r = 2 * reps["trk_eff"] * reps["trk_pur"] / (reps["trk_eff"] + reps["trk_pur"])  # fmt: skip
        tab["trk_f1"] = _pack(2 * e * p / (e + p), f1r, ref)
        tab["totals"] = {f: c[..., cf.index(f)].sum(0) for f in cf}
        out[f"min{mt}"] = tab
    return out


def event_tables(ev: dict, mask: np.ndarray, w: np.ndarray, gnn_reco: np.ndarray, gnn_fake: np.ndarray) -> dict:  # fmt: skip
    """Finder caps, Eq. (1) efficiencies, empty candidates (paired PVF - AMVF)."""
    names = list(EVENT_RATIOS)
    num = np.stack([ev[EVENT_RATIOS[k][0]][mask] for k in names], 1)
    den = np.stack([ev[EVENT_RATIOS[k][1]][mask] for k in names], 1)
    v, r = _ratio(num, den, w)
    out = {k: {"value": v[i], "err": np.std(r[:, i], ddof=1)} for i, k in enumerate(names)}  # fmt: skip
    for a, b in PAIRS:
        d = r[:, names.index(a)] - r[:, names.index(b)]
        out[f"{a}-minus-{b}"] = {"value": v[names.index(a)] - v[names.index(b)], "err": np.std(d, ddof=1)}  # fmt: skip
    peaks = ev["n_peaks"][mask]
    empty = peaks - gnn_reco[mask]
    v, r = _ratio(np.stack([empty, gnn_fake[mask] + empty], 1), np.stack([peaks, peaks], 1), w)  # fmt: skip
    out["empty_candidates"] = {"total": empty.sum(), "n_peaks": peaks.sum(), "per_event": empty.mean()}  # fmt: skip
    out["empty_frac_of_peaks"] = {"value": v[0], "err": np.std(r[:, 0], ddof=1)}
    out["fake_rate_all_peaks"] = {"value": v[1], "err": np.std(r[:, 1], ddof=1)}
    return out


def truth_tables(truth: np.ndarray, flags: np.ndarray, fnames: list[str], tf: list[str], mask: np.ndarray, w: np.ndarray) -> dict:  # fmt: skip
    """Per-truth-vertex efficiencies vs n_trk, density, separation (paired diffs)."""
    row = truth[:, tf.index("row")].astype(np.int64)
    sel = mask[row]
    inv = np.cumsum(mask) - 1
    r_ev, fl = inv[row[sel]], flags[sel]
    ne = int(mask.sum())
    pairs = {"found_pvf": "found_amvf", "o2o_pvf": "o2o_amvf", "o2o_gnn": "o2o_amvf"}
    for pre in ("clean_", "clean2_"):  # >= 1 / >= 2 track vertex lists
        pairs.update({f: f"{pre}amvf" for f in fnames if f.startswith(pre) and f != f"{pre}amvf"})  # fmt: skip
    out = {}
    for var, edges in VAR_BINS.items():
        x = truth[sel, tf.index(var)]
        b = np.digitize(x, edges) - 1
        ok = (b >= 0) & (b < len(edges) - 1)
        nb = len(edges) - 1
        den = np.zeros((ne, nb))
        np.add.at(den, (r_ev[ok], b[ok]), 1.0)
        entry = {"edges": np.asarray(edges, float), "n": den.sum(0), "mean_x": np.bincount(b[ok], weights=x[ok], minlength=nb) / den.sum(0)}  # fmt: skip
        reps = {}
        for k, f in enumerate(fnames):
            num = np.zeros((ne, nb))
            np.add.at(num, (r_ev[ok], b[ok]), fl[ok, k])
            v, r = _ratio(num, den, w)
            entry[f] = {"value": v, "err": np.nanstd(r, 0, ddof=1), "k": num.sum(0)}
            reps[f] = r
        for a, c in pairs.items():
            entry[f"{a}-minus-{c}_err"] = np.nanstd(reps[a] - reps[c], 0, ddof=1)
        out[var] = entry
    return out


def tscan(rec, mask: np.ndarray, w: np.ndarray) -> dict:  # noqa: ANN001
    """GNN t scan (chain_t_metrics record): Clean/truth, track eff/pur/F1."""
    g = lambda f: rec.cf(f, 0)[mask].astype(np.float64)  # noqa: E731
    n_lab = np.repeat(rec.ev("n_labelled")[mask, None], g("clean").shape[1], 1)
    n_tru = np.repeat(rec.ev("n_truth_ge2")[mask, None], g("clean").shape[1], 1)
    out = {"t": list(rec.meta["t_grid"]) + ["amvf"]}
    re = {}
    for name, (a, b) in {"clean_per_truth": (g("clean"), n_tru), "trk_eff": (g("n_correct"), n_lab), "trk_pur": (g("n_correct"), g("n_assigned_truth")), "fake_per_reco": (g("fake"), g("n_reco"))}.items():  # fmt: skip
        v, r = _ratio(a, b, w)
        out[name], re[name] = {"value": v, "err": np.std(r, 0, ddof=1)}, r
    e, p = out["trk_eff"]["value"], out["trk_pur"]["value"]
    f1r = 2 * re["trk_eff"] * re["trk_pur"] / (re["trk_eff"] + re["trk_pur"])
    out["trk_f1"] = {"value": 2 * e * p / (e + p), "err": np.std(f1r, 0, ddof=1)}
    return out


def sigma_pv_check(truth: np.ndarray, tf: list[str], ev_mu: np.ndarray, mask: np.ndarray) -> dict:  # fmt: skip
    """Peak width (the chain graph's sigma_PV) vs the Appendix B n_trk parameterisation."""
    row = truth[:, tf.index("row")].astype(np.int64)
    sel = mask[row] & np.isfinite(truth[:, tf.index("o2o_peak_sigma")])
    n, s = truth[sel, tf.index("n_trk")], truth[sel, tf.index("o2o_peak_sigma")]
    edges = VAR_BINS["n_trk"]
    b = np.digitize(n, edges) - 1
    rows = []
    for k in range(len(edges) - 1):
        nk, sk = n[b == k], s[b == k]
        if len(nk) == 0:
            continue
        param = compute_pv_sigma_preset(nk, *RESOLUTION_PRESETS["hllhc_alleta"])
        rows.append({"bin": np.asarray(edges[k : k + 2], float), "n": len(nk), "peak_sigma_median_mm": np.median(sk), "peak_sigma_q16_q84_mm": np.quantile(sk, [0.16, 0.84]), "param_sigma_median_mm": np.median(param)})  # fmt: skip
    return {"matched_peaks": int(sel.sum()), "peak_sigma_quantiles_mm": {q: np.quantile(s, q) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}, "by_ntrk": rows, "preset": "hllhc_alleta"}  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", required=True, type=Path)
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    meta = json.loads(a.events.with_suffix(".json").read_text())
    with np.load(a.events) as f:
        z = {k: f[k] for k in f.files}
    rec = load(a.ref)  # asserts its bookkeeping
    arms, cf, ef, tf = meta["arms"], meta["count_fields"], meta["event_fields"], meta["truth_fields"]  # fmt: skip
    ev = {k: z["events"][:, i] for i, k in enumerate(ef)}
    if not np.array_equal(ev["entry"], rec.ev("entry")):
        raise RuntimeError("event order differs from the chain_t_metrics record")
    counts, truth = z["counts"], z["truth"]
    fnames = ["found_pvf", "found_amvf", "o2o_pvf", "o2o_amvf", "o2o_gnn", *(f"clean_{x}" for x in arms), *(f"clean2_{x}" for x in arms)]  # fmt: skip
    flags = np.column_stack([truth[:, tf.index(k)] for k in fnames[:5]] + [z["truth_clean"].astype(np.float64), z["truth_clean2"].astype(np.float64)])  # fmt: skip
    gi = arms.index("gnn_t")
    gnn_reco, gnn_fake = counts[:, gi, 0, cf.index("n_reco")], counts[:, gi, 0, cf.index("fake")]  # fmt: skip
    mu = ev["mu"]
    out: dict = {"meta": {"events": str(a.events), "ref": str(a.ref), "n_boot": N_BOOT, "seed": SEED, "git": git_state(), "replay_meta": {k: meta[k] for k in ("checkpoints", "regime", "window_mm", "t", "s_max", "knn", "arms", "gnn", "events_sha256", "git")}}, "gates": meta["gates"], "bins": {}, "per_truth": {}, "tscan": {}}  # fmt: skip
    for label, (lo, hi) in {
        **MU_BINS,
        **{f"mu {k}": v for k, v in MU10.items()},
    }.items():
        mask = (mu >= lo) & (mu < hi)
        w = boot_weights(int(mask.sum()), N_BOOT, SEED)
        out["bins"][label] = {
            "n_events": int(mask.sum()), "mean_mu": mu[mask].mean(),
            "n_truth_ge2": ev["n_truth_ge2"][mask].sum(),
            "arms": arm_tables(ev, counts, cf, mask, w, arms),
            "event": event_tables(ev, mask, w, gnn_reco, gnn_fake),
        }  # fmt: skip
        out["tscan"][label] = tscan(rec, mask, w)
        if label in SAMPLES:
            out["per_truth"][label] = truth_tables(truth, flags, fnames, tf, mask, w)
    # Eq. (4): best s_max by Clean/truth at PU200 (optimistic for the baseline)
    eq4 = [x for x in arms if x.startswith("eq4_")]
    pu = out["bins"][PU200]["arms"]["min1"]
    vals = [pu["clean_per_truth"]["value"][arms.index(x)] for x in eq4]
    out["eq4_best"] = {"arm": eq4[int(np.argmax(vals))], "by": "clean_per_truth, PU200, min1", "scan": dict(zip(eq4, vals))}  # fmt: skip
    pu_mask = (mu >= MU_BINS[PU200][0]) & (mu < MU_BINS[PU200][1])
    out["sigma_pv_check"] = {s: sigma_pv_check(truth, tf, mu, (mu >= MU_BINS[s][0]) & (mu < MU_BINS[s][1])) for s in SAMPLES}  # fmt: skip
    out["amvf_tracks_in_several_vertices"] = int(ev["n_amvf_multi"].sum())
    b = out["bins"][PU200]
    t4 = {}
    for arm in ("gnn_t", "amvf"):
        k, tot, r = arms.index(arm), b["arms"]["min1"]["totals"], b["arms"]["min1"]
        t4[arm] = {"vertices_with_tracks": tot["n_reco"][k], **{f: tot[f][k] for f in ("clean", "merged", "split", "fake", "clean_dom_ge2")}, **{f: r[f]["value"][k] for f in ("clean_per_reco", "clean_per_truth", "clean_ge2_per_truth", "merged_per_reco", "split_per_reco", "fake_per_reco")}}  # fmt: skip
    out["table4_pu200"] = {"n_events": b["n_events"], "n_truth_ge2": b["n_truth_ge2"], **t4, "empty_candidates": b["event"]["empty_candidates"], "fake_rate_all_peaks": b["event"]["fake_rate_all_peaks"], "n_events_check": int(pu_mask.sum())}  # fmt: skip
    path = a.out or a.events.parent / "chain_extras_summary.json"
    path.write_text(json.dumps(_jsonable(out), indent=1, default=float) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
