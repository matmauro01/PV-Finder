"""HL-LHC peak-finding consolidation: position estimator and cut configuration.

Replays peak finding on saved s8 densities (run_eval_pvf_run3 --save-histograms
at the frozen hllhc working point); no inference. Off the final test set
(r16443 entries 0-24,999) by construction:

* r16638 entries 0-24,999 (density models never saw the file): even entries
  are the tuning half (matched cut values are derived there), odd entries the
  report half;
* r16443 GNN-validation entries inside 25,000-49,999 (confirmation, another file).

Part 1: position estimators at the frozen cuts.
Part 2: one-knob cut families, each matched to the fakes per event of the
frozen configuration on the PU200 events of the tuning half.

docs/evaluation/hllhc_peak_finding_consolidation_2026-11-13.md

Usage (repo root, venv, inside tmux):
    python -u scripts/peak_consolidation.py --out outputs/11_13_2026_output/peak_finding
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import uproot

ROOT = Path(__file__).resolve().parents[1]
for p in (
    ROOT / "src/pv_finder/evaluation/vertex_finding",
    ROOT / "src",
    ROOT / "scripts",
):
    sys.path.insert(0, str(p))
import peak_consolidation_lib as L  # noqa: E402

from gnn.evaluation.chain_t_summary import MU_BINS  # noqa: E402
from pv_finder.diagnostics.run2_vs_run4.metrics import (  # noqa: E402
    event_hists,
    fit_counts,
    per_event_residuals,
)
from pv_finder.evaluation.vertex_finding.freeze_compare import hs_found  # noqa: E402
from pv_finder.utils.working_points import get_regime  # noqa: E402

REG = get_regime("hllhc")
WIN, HS_WIN = REG.match_window_mm, REG.hs_match_window_mm
FROZEN = {"min_width": REG.peaks.min_width, "integral": REG.peaks.integral,
          "min_height": REG.peaks.min_height}  # fmt: skip
AMP = REG.peaks.amplitude
N_BOOT, N_BOOT_FIT, SEED = 2000, 200, 20261113
DATA = ROOT / "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026"
SAMPLES = {
    "r16638": ("outputs/11_12_2026_output/hllhc_final/pre_gnn/s8/r16638",
               DATA / "ATLAS_PVFinderData_601229_e8481_s4494_r16638_PU200.root"),
    "r16443val": ("outputs/11_13_2026_output/peak_finding/eval_s8_r16443_e25000_50000",
                  DATA / "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"),
}  # fmt: skip
GNN_VAL = (
    ROOT / "data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_entries.npy"
)
PU200 = "PU200 (185-215)"


def load_sample(name: str, max_events: int = 0) -> dict:
    """Events with truth of one saved eval, restricted to the allowed entries."""
    d_path, root = SAMPLES[name]
    with open(ROOT / d_path / "eval_results.pkl", "rb") as f:
        d = pickle.load(f)
    pk = d["regime"]["density"]["peaks"] if "regime" in d else None
    got = (d["peak_threshold"], d["integral_threshold"], d["min_height"],
           d["centroid_halfwidth"], d["match_window_mm"])  # fmt: skip
    want = (AMP, FROZEN["integral"], FROZEN["min_height"], 3, WIN)
    if got != want:
        raise ValueError(
            f"{name}: saved eval not at the frozen working point {got} {pk}"
        )
    with_truth = [i for i, t in enumerate(d["truth_pvs_mm"]) if len(t) > 0]
    recs = dict(zip(with_truth, d["per_event"]))
    idx = np.array([recs[i]["event_idx"] for i in with_truth])
    if np.any(idx < 25000) and "r16443" in name:
        raise ValueError("r16443 entries 0-24,999 are the final test set")
    keep = np.ones(len(with_truth), bool)
    if name == "r16443val":
        keep = np.isin(idx, np.load(GNN_VAL))
    if max_events:
        keep &= np.cumsum(keep) <= max_events
    sel = [i for i, k in zip(with_truth, keep) if k]
    ev = {
        "name": name,
        "idx": idx[keep],
        "mu": np.array([recs[i]["mu"] for i in sel], float),
        "truth": [np.asarray(d["truth_pvs_mm"][i], np.float32) for i in sel],
        "hists": [np.asarray(d["histograms"][i]) for i in sel],
        "saved_pvs": [np.asarray(d["pred_pvs_mm"][i]) for i in sel],
        "saved": [recs[i] for i in sel],
    }
    ev["truth_b"] = [L.to_bins(t) for t in ev["truth"]]
    ev["amvf"], ev["hs_z"] = read_root(root, ev["idx"])
    del d
    return ev


def read_root(path: Path, idx: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """AMVF z (n_trk >= 2, float32 as run3_io) and true HS z (nan if not unique)."""
    br = ["TruthVertex_z", "TruthVertex_type", "RecoVertex_z", "RecoVertex_nTracks"]
    lo, hi = int(idx.min()), int(idx.max()) + 1
    a = uproot.open(path)["PVFinderData"].arrays(
        br, entry_start=lo, entry_stop=hi, library="np"
    )
    amvf, hs = [], np.full(len(idx), np.nan)
    for k, i in enumerate(idx - lo):
        z, n = np.asarray(a["RecoVertex_z"][i]), np.asarray(a["RecoVertex_nTracks"][i])
        amvf.append(np.asarray(z[n >= 2], dtype=np.float32))
        h = np.flatnonzero(np.asarray(a["TruthVertex_type"][i]) == 1)
        if len(h) == 1:
            hs[k] = float(a["TruthVertex_z"][i][h[0]])
    return amvf, hs


def arm(ev: dict, preds: list[np.ndarray], matcher: bool = True) -> dict:
    """Per-event quantities of one configuration (or AMVF) on one sample."""
    c: dict = {}
    c["found"], c["fake"], c["npred"] = L.found_fake(ev["truth_b"], preds, WIN)
    c["truth"] = np.array([len(t) for t in ev["truth"]], float)
    if matcher:
        m = L.matcher_counts(ev["truth_b"], preds, WIN)
        if not (
            np.array_equal(m["found"], c["found"])
            and np.array_equal(m["fake"], c["fake"])
        ):
            raise AssertionError("matcher identities broken")
        c["split"], c["t_merged"] = m["split"], m["t_merged"]
    c["res"] = per_event_residuals(ev["truth"], preds)
    c["H"], c["edges"] = event_hists(preds)
    ok = ~np.isnan(ev["hs_z"])
    c["hs"] = np.array([hs_found(p, z, HS_WIN) if o else np.nan
                        for p, z, o in zip(preds, ev["hs_z"], ok)])  # fmt: skip
    # signed residual of the nearest reco within the window (bias check)
    c["bias"] = [_signed(t, p) for t, p in zip(ev["truth"], preds)]
    return c


def _signed(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    if len(p) == 0:
        return np.zeros(0)
    ps = np.sort(p.astype(np.float64))
    i = (
        np.clip(np.searchsorted(ps, t), 1, len(ps) - 1)
        if len(ps) > 1
        else np.zeros(len(t), int)
    )
    cand = np.stack([ps[np.maximum(i - 1, 0)], ps[i]]) - t[None, :].astype(np.float64)
    d = cand[np.argmin(np.abs(cand), axis=0), np.arange(len(t))]
    return d[np.abs(d) <= WIN]


def _stats(c: dict, j: np.ndarray) -> dict:
    p68 = np.concatenate([c["res"][i] for i in j]) if len(j) else np.zeros(0)
    out = {
        "eff": c["found"][j].sum() / c["truth"][j].sum(),
        "fp_per_evt": c["fake"][j].mean(),
        "reco_per_evt": c["npred"][j].mean(),
        "p68_um": 1e3 * float(np.percentile(p68, 68.0)) if len(p68) else np.nan,
        "sigma_um": 1e3 * fit_counts(c["edges"], c["H"][j].sum(0))[0],
    }
    if "split" in c:
        out["split_per_evt"] = c["split"][j].mean()
        out["truth_merged_per_evt"] = c["t_merged"][j].mean()
    hs = c["hs"][j]
    out["hs_eff"] = float(np.nanmean(hs))
    return out


def summarize(
    c: dict,
    ref: dict | None,
    j: np.ndarray,
    rng: np.random.Generator,
    n_fit: int = N_BOOT_FIT,
) -> dict:
    """Values on events j; paired bootstrap differences to ref on the same events."""
    out = {k: float(v) for k, v in _stats(c, j).items()}
    out["n_events"], out["n_truth"] = int(len(j)), int(c["truth"][j].sum())
    b = [rng.choice(j, len(j)) for _ in range(N_BOOT)]
    eff_b = [c["found"][x].sum() / c["truth"][x].sum() for x in b]
    out["eff_err"] = float(np.std(eff_b, ddof=1))
    out["fp_err"] = float(np.std([c["fake"][x].mean() for x in b], ddof=1))
    if ref is None:
        return out
    r = {k: float(v) for k, v in _stats(ref, j).items()}
    out["d_eff_pt"] = 100 * (out["eff"] - r["eff"])
    out["d_fp"] = out["fp_per_evt"] - r["fp_per_evt"]
    de = [c["found"][x].sum() / c["truth"][x].sum() - ref["found"][x].sum() / ref["truth"][x].sum()
          for x in b]  # fmt: skip
    out["d_eff_pt_err"] = 100 * float(np.std(de, ddof=1))
    out["d_fp_err"] = float(
        np.std([c["fake"][x].mean() - ref["fake"][x].mean() for x in b], ddof=1)
    )
    if "split" in c and "split" in ref:
        out["d_split"] = out["split_per_evt"] - r["split_per_evt"]
        out["d_split_err"] = float(np.std([c["split"][x].mean() - ref["split"][x].mean()
                                           for x in b], ddof=1))  # fmt: skip
    dp, ds = [], []
    for x in b[:n_fit]:
        a1, a2 = _stats(c, x), _stats(ref, x)
        dp.append(a1["p68_um"] - a2["p68_um"])
        ds.append(a1["sigma_um"] - a2["sigma_um"])
    out["d_p68_um"], out["d_p68_um_err"] = (
        out["p68_um"] - r["p68_um"],
        float(np.std(dp, ddof=1)),
    )
    out["d_sigma_um"] = out["sigma_um"] - r["sigma_um"]
    out["d_sigma_um_err"] = float(np.nanstd(ds, ddof=1))
    return out


# --- cut families: (name, description, function knob -> config) ---------------
def families(int0: float, h0: float) -> dict:
    return {
        "current_scaled": ("integral 0.184k + height 0.0184k, width 3",
                           lambda k: dict(amp=AMP, min_width=3, integral=int0 * k, min_height=h0 * k)),
        "integral_only": ("integral X, width 3, no height",
                          lambda x: dict(amp=AMP, min_width=3, integral=x, min_height=0.0)),
        "integral_only_w1": ("integral X, no width, no height",
                             lambda x: dict(amp=AMP, min_width=1, integral=x, min_height=0.0)),
        "height_only": ("height X, width 3, no integral",
                        lambda x: dict(amp=AMP, min_width=3, integral=0.0, min_height=x)),
        "height_only_w1": ("height X, no width, no integral",
                           lambda x: dict(amp=AMP, min_width=1, integral=0.0, min_height=x)),
        "amplitude_only": ("amplitude X, no width, integral or height",
                           lambda x: dict(amp=x, min_width=1, integral=0.0, min_height=0.0)),
        "amplitude_w3": ("amplitude X, width 3, no integral or height",
                         lambda x: dict(amp=x, min_width=3, integral=0.0, min_height=0.0)),
    }  # fmt: skip


class Peaks:
    """Cached candidate extraction per amplitude for one sample."""

    def __init__(self, ev: dict):
        self.ev, self.cache = ev, {}

    def cand(self, amp: float) -> dict:
        if amp not in self.cache:
            if len(self.cache) > 4:
                self.cache.pop(next(k for k in self.cache if k != AMP))
            self.cache[amp] = L.extract(self.ev["hists"], amp)
        return self.cache[amp]

    def preds(
        self, cfg: dict, est: str, j: np.ndarray | None = None
    ) -> list[np.ndarray]:
        cand = self.cand(cfg["amp"])
        m = L.select(cand, min_width=cfg["min_width"], integral=cfg["integral"],
                     min_height=cfg["min_height"])  # fmt: skip
        p = L.peaks_by_event(cand, m, est)
        return p if j is None else [p[i] for i in j]


def fp_eff(pk: Peaks, cfg: dict, est: str, j: np.ndarray) -> tuple[float, float]:
    ev = pk.ev
    tb = [ev["truth_b"][i] for i in j]
    f, fk, _ = L.found_fake(tb, pk.preds(cfg, est, j), WIN)
    return float(fk.mean()), float(f.sum() / sum(len(t) for t in tb))


def match_knob(
    pk: Peaks, fam, est: str, j: np.ndarray, target: float, lo: float, hi: float
):
    """Bisect the knob (fakes fall as it rises) to the target fakes per event."""
    f_lo, f_hi = fp_eff(pk, fam(lo), est, j)[0], fp_eff(pk, fam(hi), est, j)[0]
    if not f_hi <= target <= f_lo:
        raise ValueError(f"not bracketed: {f_lo} {f_hi} {target}")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if fp_eff(pk, fam(mid), est, j)[0] > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6 * max(hi, 1e-3):
            break
    return 0.5 * (lo + hi)


KNOB_RANGE = {"current_scaled": (0.2, 5.0), "integral_only": (0.01, 2.0),
              "integral_only_w1": (0.01, 2.0), "height_only": (0.005, 1.0),
              "height_only_w1": (0.005, 1.0), "amplitude_only": (0.01, 0.5),
              "amplitude_w3": (0.01, 0.5)}  # fmt: skip
CURVE = {k: np.geomspace(*v, 40) for k, v in KNOB_RANGE.items()}


FIT_BINS = ("inclusive", PU200)  # bootstrap the refits only here (cost)


def by_mu_fast(c: dict, ref: dict | None, ev: dict, j: np.ndarray, rng) -> dict:
    """by_mu, with refit errors only for the inclusive and PU200 bins."""
    out = {}
    for lab, (lo, hi) in MU_BINS.items():
        jj = j[(ev["mu"][j] >= lo) & (ev["mu"][j] < hi)]
        out[lab] = summarize(c, ref, jj, rng, N_BOOT_FIT if lab in FIT_BINS else 2)
    return out


EV: dict = {}


def _arm_task(task: tuple[str, list[np.ndarray], bool]) -> dict:
    name, preds, matcher = task
    return arm(EV[name], preds, matcher)


def run_arms(
    tasks: list[tuple[str, list[np.ndarray], bool]], workers: int
) -> list[dict]:
    import multiprocessing as mp

    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_arm_task, tasks)


def round_sig(x: float, n: int = 3) -> float:
    return float(f"{x:.{n}g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", nargs="+", default=list(SAMPLES))
    ap.add_argument("--estimator", default="centroid_3", help="estimator for part 2")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=0, help="smoke test only")
    ap.add_argument(
        "--knobs-from", help="consolidation.json whose matched knobs to reuse"
    )
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    result: dict = {"working_point": REG.to_dict()["density"]["peaks"], "window_mm": WIN,
                    "samples": {}, "estimators": {}, "cuts": {}, "curves": {}}  # fmt: skip

    pks = {}
    for s in a.samples:
        ev = load_sample(s, a.max_events)
        EV[s] = ev
        amvf_saved = np.array([r["amvf_t_found"] for r in ev["saved"]], float)
        amvf_found = L.found_fake(ev["truth_b"], ev["amvf"], WIN)[0]
        if not np.array_equal(amvf_found, amvf_saved):
            raise AssertionError(f"{s}: AMVF found differs from the saved eval")
        pk = Peaks(ev)
        ver = L.verify_against_production(
            ev["hists"], pk.cand(AMP), FROZEN, ev["saved_pvs"], ev["truth_b"], WIN
        )
        m = ver.pop("matcher")
        for key, sk in (("split", "split"), ("fake", "fake"), ("found", None)):
            ref = np.array(
                [(r["tc"] + r["tm"]) if sk is None else r[sk] for r in ev["saved"]]
            )
            if not np.array_equal(m[key], ref):
                raise AssertionError(f"{s}: replay {key} differs from saved eval")
        pu = (ev["mu"] >= 185) & (ev["mu"] < 215.5)
        result["samples"][s] = {**ver, "n_events": len(ev["idx"]), "n_pu200": int(pu.sum()),
                                "entries": [int(ev["idx"].min()), int(ev["idx"].max())],
                                "amvf_found_matches_saved": True,
                                "replay_matches_saved_counts": True}  # fmt: skip
        print(
            f"[{s}] {len(ev['idx'])} events ({pu.sum()} PU200): verification passed",
            flush=True,
        )
        pks[s] = pk

    # --- Part 1: estimators at the frozen cuts -----------------------------------
    mask = {s: L.select(pks[s].cand(AMP), **FROZEN) for s in pks}
    for s, pk in pks.items():
        ev = EV[s]
        tasks = [
            (s, L.peaks_by_event(pk.cand(AMP), mask[s], e), True) for e in L.ESTIMATORS
        ]
        arms = dict(zip(L.ESTIMATORS, run_arms(tasks, a.workers)))
        arms["AMVF"] = arm(ev, ev["amvf"])
        allj = np.arange(len(ev["idx"]))
        res = {}
        for e, c in arms.items():
            res[e] = by_mu_fast(c, arms["centroid_3"], ev, allj, rng)
            bias = np.concatenate(c["bias"])
            res[e]["bias_um"] = 1e3 * float(np.mean(bias))
            res[e]["median_signed_um"] = 1e3 * float(np.median(bias))
            print(f"[{s}] {e:13s} PU200 p68 {res[e][PU200]['p68_um']:.2f} "
                  f"({res[e][PU200].get('d_p68_um', 0):+.2f}±{res[e][PU200].get('d_p68_um_err', 0):.2f})"
                  f" sigma {res[e][PU200]['sigma_um']:.1f} eff {100 * res[e][PU200]['eff']:.2f} "
                  f"fp {res[e][PU200]['fp_per_evt']:.2f}", flush=True)  # fmt: skip
        result["estimators"][s] = res

    # --- Part 2: one-knob cut families at the frozen fake rate -------------------
    est = a.estimator
    fams = families(FROZEN["integral"], FROZEN["min_height"])
    frozen_cfg = dict(amp=AMP, **FROZEN)
    cfgs = {"frozen": frozen_cfg}
    if a.knobs_from:
        # Confirmation run: reuse the knobs matched on the r16638 tuning half.
        prev = json.loads(Path(a.knobs_from).read_text())["cuts"]
        result["cuts"]["target"], result["cuts"]["knobs"] = (
            prev["target"],
            prev["knobs"],
        )
        for f, kn in prev["knobs"].items():
            cfgs[f] = kn["config"]
    else:
        ev0 = EV["r16638"]
        pu0 = (ev0["mu"] >= 185) & (ev0["mu"] < 215.5)
        tune = np.flatnonzero(pu0 & (ev0["idx"] % 2 == 0))
        target = fp_eff(pks["r16638"], frozen_cfg, est, tune)[0]
        result["cuts"]["target"] = {"fp_per_evt": target, "events": "r16638 even entries, PU200",
                                    "n_events": int(len(tune)), "estimator": est}  # fmt: skip
        print(
            f"target fakes/event (tune half, PU200, frozen cuts): {target:.3f}",
            flush=True,
        )
        for f, (desc, fam) in fams.items():
            x = match_knob(pks["r16638"], fam, est, tune, target, *KNOB_RANGE[f])
            xr = round_sig(x)
            cfgs[f] = fam(xr)
            result["cuts"].setdefault("knobs", {})[f] = {"desc": desc, "matched": x,
                                                         "rounded": xr, "config": fam(xr)}  # fmt: skip
            print(f"  {f}: matched knob {x:.5g} -> {xr}  cfg {fam(xr)}", flush=True)
    # Mattia's literal suggestion: height 0.5 (units of the s8 output)
    cfgs["height_0p5_literal"] = dict(amp=AMP, min_width=3, integral=FROZEN["integral"],
                                      min_height=0.5)  # fmt: skip

    for s, pk in pks.items():
        ev = EV[s]
        names = list(cfgs)
        tasks = [(s, pk.preds(cfgs[n], est), True) for n in names]
        arms = dict(zip(names, run_arms(tasks, a.workers)))
        arms["AMVF"] = arm(ev, ev["amvf"])
        sets = {"all": np.arange(len(ev["idx"]))}
        if s == "r16638":
            sets["report"] = np.flatnonzero(ev["idx"] % 2 == 1)
            sets["tune"] = np.flatnonzero(ev["idx"] % 2 == 0)
        res = {}
        for setname, j in sets.items():
            res[setname] = {
                n: by_mu_fast(c, arms["frozen"], ev, j, rng) for n, c in arms.items()
            }
        result["cuts"][s] = res
        for n in arms:
            r = res["report" if s == "r16638" else "all"][n][PU200]
            print(f"[{s}] {n:20s} PU200 eff {100 * r['eff']:.2f} ({r.get('d_eff_pt', 0):+.2f}"
                  f"±{r.get('d_eff_pt_err', 0):.2f}) fp {r['fp_per_evt']:.2f} "
                  f"({r.get('d_fp', 0):+.2f}±{r.get('d_fp_err', 0):.2f}) p68 {r['p68_um']:.2f} "
                  f"sigma {r['sigma_um']:.1f} split {r.get('split_per_evt', 0):.3f}", flush=True)  # fmt: skip
        # trade-off curves on the PU200 events of the sample (all its entries)
        jpu = np.flatnonzero((ev["mu"] >= 185) & (ev["mu"] < 215.5))
        curves = {}
        for f, (_, fam) in fams.items():
            pts = [fp_eff(pk, fam(float(x)), est, jpu) for x in CURVE[f]]
            curves[f] = {"knob": CURVE[f].tolist(), "fp": [p[0] for p in pts],
                         "eff": [p[1] for p in pts]}  # fmt: skip
        f_a, e_a = L.found_fake(
            [ev["truth_b"][i] for i in jpu], [ev["amvf"][i] for i in jpu], WIN
        )[:2]
        curves["AMVF"] = {
            "fp": float(e_a.mean()),
            "eff": float(f_a.sum() / sum(len(ev["truth"][i]) for i in jpu)),
        }
        result["curves"][s] = curves

    (out_dir / "consolidation.json").write_text(
        json.dumps(result, indent=1, default=float)
    )
    print("wrote", out_dir / "consolidation.json")


if __name__ == "__main__":
    main()
