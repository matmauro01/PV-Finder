"""Low pile-up vertex-finding metrics at the frozen working point (R1, R3).

Everything is computed from saved evaluation pickles (``run_eval_pvf.py`` and
``run_eval_pvf_run3.py`` at ``--regime lowpu``) plus, for simulation, the
truth and AMVF vertices re-read from the MC ROOT file through the h5 -> ROOT
index map. No network is run.

Definitions (docs/evaluation/metric_definitions.md):

* efficiency / reference recovery: reference vertices with at least one reco
  vertex inside the window (clean + merged), over all reference vertices;
  split-inclusive by construction (§3.3a); one-to-one also given;
* fake fraction: reco vertices with no reference vertex in the window, over
  all reco vertices (no reference per event also given);
* sigma_vtx-vtx: sigmoid fit of all within-event pairwise dz of the counted
  vertices, 300 bins over +-6 mm, data-driven start values;
* 68% residual: 68th percentile of |z_reco - z_truth| after an optimal
  one-to-one assignment within 1 mm;
* HS efficiency: events with a reco vertex within 0.1 mm of the truth HS
  (``TruthVertex_type == 1``).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta

from pv_finder.diagnostics.amvf_fair_comparison.scan import core_position_resolution
from pv_finder.diagnostics.data_mc_validation import reference
from pv_finder.diagnostics.paper_figures.sources import Resolution, fit_resolution
from pv_finder.utils.working_points import Regime

MC_BRANCHES = (
    "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_type",
    "RecoVertex_z", "RecoVertex_nTracks", "ActualNumOfInt", "BeamPosZ",
)  # fmt: skip
N_BOOT = 2000
SEED = 20261112


@dataclass
class Arm:
    """One algorithm's vertices on one sample, aligned with the reference."""

    name: str
    reco: list[np.ndarray]
    dz: np.ndarray = field(repr=False)


@dataclass
class LowPuSample:
    """Reference and per-arm vertex lists of one low pile-up sample."""

    name: str
    ref: list[np.ndarray]  # truth (nTrk >= 2) in MC, AMVF (beam-corrected) in data
    mu: np.ndarray
    arms: dict[str, Arm]
    hs_z: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def load_pkl(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def pairs_dz(lists: list[np.ndarray], seed: int = SEED) -> np.ndarray:
    """All within-event pairwise dz, each with a random sign (seeded)."""
    rng = np.random.default_rng(seed)
    out = []
    for z in lists:
        z = np.asarray(z, dtype=np.float64)
        if len(z) < 2:
            continue
        i, j = np.triu_indices(len(z), 1)
        d = z[i] - z[j]
        out.append(d * rng.choice((-1.0, 1.0), size=len(d)))
    return np.concatenate(out) if out else np.zeros(0)


def vf_part(regime_dict: dict[str, Any]) -> dict[str, Any]:
    """The regime without its GNN entries (frozen after the vertex finding ran)."""
    return {k: v for k, v in regime_dict.items() if k not in ("gnn", "pending")}


def check_regime(res: dict[str, Any], regime: Regime) -> None:
    """The pkl must come from this regime with no override."""
    prov = res["provenance"]
    if vf_part(prov["regime"]) != vf_part(regime.to_dict()) or prov["regime_overrides"]:
        raise ValueError("pkl was not produced at the regime without overrides")


def load_mc(
    pkl_path: Path, root_vertices: dict[str, list], qibin: list[int]
) -> LowPuSample:
    """Run 2 MC: PVF peaks from the pkl, truth / AMVF / HS from ROOT entries qibin[i]."""
    res = load_pkl(pkl_path)
    pe = res["per_event"]
    n = len(pe)
    if [e["event_idx"] for e in pe] != list(range(48450, 48450 + n)):
        raise ValueError("MC pkl is not the h5 test events 48450.. in order")
    ent = qibin[:n]
    tz, tn, tt = (root_vertices[k] for k in MC_BRANCHES[:3])
    rz, rn = root_vertices["RecoVertex_z"], root_vertices["RecoVertex_nTracks"]
    truth = [tz[e][tn[e] >= 2] for e in ent]
    amvf = [rz[e][rn[e] >= 2] for e in ent]
    hs = []
    for e in ent:
        h = np.nonzero(tt[e] == 1)[0]
        if len(h) != 1:
            raise ValueError(f"entry {e}: {len(h)} HS vertices")
        hs.append(tz[e][h[0]])
    mu = np.asarray(root_vertices["ActualNumOfInt"])[ent]
    beam = np.asarray(root_vertices["BeamPosZ"])[ent]
    if np.any(beam != 0.0):
        raise ValueError("MC BeamPosZ is not zero")
    # Alignment proof against what the eval itself stored.
    checks = {
        "n_truth": all(len(t) == e["n_truth"] for t, e in zip(truth, pe)),
        "n_amvf": all(len(a) == e["n_amvf"] for a, e in zip(amvf, pe)),
        "mu": bool(np.all(np.abs(mu - np.array([e["mu"] for e in pe])) < 1e-6)),
    }
    if not all(checks.values()):
        raise ValueError(f"MC alignment failed: {checks}")
    pred = [np.asarray(p, dtype=np.float64) for p in res["pred_pvs_mm"]]
    arms = {
        "pvf": Arm("pvf", pred, np.asarray(res["pairwise_dz_mm"], dtype=np.float64)),
        "amvf": Arm("amvf", amvf, pairs_dz(amvf)),
    }
    return LowPuSample("Run 2 MC", truth, mu, arms, np.asarray(hs), {
        "pkl": str(pkl_path), "alignment": checks, "provenance": res.get("provenance"),
        "entries_first_last": [int(ent[0]), int(ent[-1])],
    })  # fmt: skip


def load_data(name: str, pkl_path: Path) -> LowPuSample:
    """Collision data: PVF peaks and the beam-corrected AMVF reference from the pkl."""
    res = load_pkl(pkl_path)
    if res.get("has_truth"):
        raise ValueError(f"{pkl_path}: data pkl claims MC truth")
    ref = [np.asarray(t, dtype=np.float64) for t in res["truth_pvs_mm"]]
    pred = [np.asarray(p, dtype=np.float64) for p in res["pred_pvs_mm"]]
    pe = res["per_event"]
    dz_all = res.get("pairwise_dz_mm_all_events", res["pairwise_dz_mm"])
    arms = {
        "pvf": Arm("pvf", pred, np.asarray(dz_all, dtype=np.float64)),
        "amvf_ref": Arm("amvf_ref", ref, pairs_dz(ref)),
    }
    return LowPuSample(name, ref, np.array([e["mu"] for e in pe], dtype=float), arms,
                       meta={"pkl": str(pkl_path), "provenance": res.get("provenance"),
                             "entries": [int(e["event_idx"]) for e in pe]})  # fmt: skip


def clopper_pearson(k: int, n: int, cl: float = 0.6827) -> tuple[float, float]:
    """Equal-tailed binomial interval."""
    a = (1 - cl) / 2
    lo = beta.ppf(a, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def hs_efficiency(reco: list[np.ndarray], hs_z: np.ndarray, window: float) -> dict:
    """Fraction of events with a reco vertex within ``window`` of the truth HS."""
    ok = np.array([len(r) > 0 and np.min(np.abs(r - h)) < window
                   for r, h in zip(reco, hs_z)])  # fmt: skip
    k, n = int(ok.sum()), len(ok)
    lo, hi = clopper_pearson(k, n)
    return {"passed": k, "events": n, "eff": k / n, "ci68": [lo, hi]}


def fake_fraction(c: reference.EventCounts, n_boot: int = N_BOOT) -> dict[str, float]:
    """No-reference reco vertices over all reco vertices, event bootstrap."""
    v = c.valid
    no_ref, n_reco = c.no_ref[v].astype(float), c.n_reco[v].astype(float)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(no_ref), len(no_ref))
        boot.append(no_ref[i].sum() / n_reco[i].sum())
    return {"value": float(no_ref.sum() / n_reco.sum()), "err": float(np.std(boot))}


def resolution(dz: np.ndarray, n_bins: int) -> Resolution:
    return fit_resolution(dz, n_bins, None, note=f"{n_bins} bins over +-6 mm")


def arm_metrics(s: LowPuSample, arm: Arm, regime: Regime) -> dict[str, Any]:
    """Every vertex-finding number of one arm on one sample at the regime window."""
    w = regime.match_window_mm
    c = reference.match_events(s.ref, arm.reco, w)
    out: dict[str, Any] = reference.summary_with_errors(c, n_boot=N_BOOT, seed=SEED)
    out["fake_fraction"] = fake_fraction(c)
    out["reco_per_event"] = float(np.mean([len(r) for r in arm.reco]))
    out["ref_per_event"] = float(np.mean([len(r) for r in s.ref]))
    fit = resolution(arm.dz, regime.pairwise_bins)
    out["sigma_vtx_vtx_mm"] = {"value": fit.sigma_mm, "err": fit.sigma_err_mm,
                               "n_pairs": int(len(arm.dz))}  # fmt: skip
    if s.hs_z is not None:
        p68 = core_position_resolution(s.ref, arm.reco)
        out["p68_residual_um"] = 1e3 * p68
        out["window_over_p68"] = w / p68
        out["hs_eff"] = hs_efficiency(arm.reco, s.hs_z, regime.hs_match_window_mm)
    return out


def sample_metrics(s: LowPuSample, regime: Regime) -> dict[str, Any]:
    """All arms of one sample, plus the paired PV-Finder minus other-arm deltas."""
    out: dict[str, Any] = {"n_events": len(s.ref), "mean_mu": float(s.mu.mean()),
                           "meta": s.meta}  # fmt: skip
    counts = {}
    for name, arm in s.arms.items():
        if name == "amvf_ref":  # data: AMVF is the reference, only its sigma is new
            fit = resolution(arm.dz, regime.pairwise_bins)
            out[name] = {"sigma_vtx_vtx_mm": {"value": fit.sigma_mm,
                         "err": fit.sigma_err_mm, "n_pairs": int(len(arm.dz))},
                         "per_event": float(np.mean([len(r) for r in arm.reco]))}  # fmt: skip
            continue
        out[name] = arm_metrics(s, arm, regime)
        counts[name] = reference.match_events(s.ref, arm.reco, regime.match_window_mm)
    if "amvf" in counts:
        out["delta_pvf_minus_amvf"] = reference.paired_delta(
            counts["pvf"], counts["amvf"], n_boot=N_BOOT, seed=SEED
        )
    return out


def pvf_vs_amvf_reference(s: LowPuSample, window: float) -> dict[str, Any]:
    """MC only: PV-Finder matched to AMVF instead of truth (the data convention)."""
    c = reference.match_events(s.arms["amvf"].reco, s.arms["pvf"].reco, window)
    out = reference.summary_with_errors(c, n_boot=N_BOOT, seed=SEED)
    out["fake_fraction"] = fake_fraction(c)
    return out
