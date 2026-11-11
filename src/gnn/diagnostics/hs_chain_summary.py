"""Summaries of the per-event HS / vertex-composition study (hs_chain_study).

Reads ``hs_chain_events.npz`` and writes ``hs_chain_summary.json`` with every
metric integrated at PU200 (185 <= mu <= 215, the note's 1,920 events) and in
bins of mu over the 25,000 flat-mu entries.

Uncertainties:
  event fractions (a, b, d): equal-tailed 68.27% Clopper-Pearson intervals;
  ratios of sums (c, clean/truth): standard deviation of an event bootstrap
  (1,000 resamples, fixed seed). Differences between methods are paired:
  the same resampled events enter both terms.

Usage:
    PYTHONPATH=src python -u src/gnn/diagnostics/hs_chain_summary.py \\
        --study outputs/11_09_2026_output/hs_chain_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

CONFIDENCE = 0.682689492137
MU_EDGES = np.arange(0, 201, 20)
PU200 = (185.0, 215.0)
N_BOOT = 1000
SEED = 20261109


def clopper_pearson(k: int, n: int) -> dict[str, float | int | None]:
    """Efficiency with an equal-tailed 68.27% Clopper-Pearson interval."""
    if not 0 <= k <= n:
        raise ValueError("invalid binomial counts")
    if n == 0:
        return {"k": 0, "n": 0, "value": None, "low": None, "high": None}
    alpha = 1 - CONFIDENCE
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {"k": int(k), "n": int(n), "value": k / n, "low": low, "high": high}


def bootstrap_ratio(
    num: np.ndarray, den: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT
) -> dict[str, float | None]:
    """Ratio of sums over events, with its event-bootstrap standard deviation.

    ``num`` and ``den`` may be 2-D (events x methods); the result then also
    carries the paired bootstrap error of method 0 minus each other method.
    """
    num = np.atleast_2d(np.asarray(num, dtype=np.float64).T).T
    den = np.atleast_2d(np.asarray(den, dtype=np.float64).T).T
    n = len(num)
    if n == 0 or den.sum() == 0:
        return {"value": None, "error": None}
    value = num.sum(0) / den.sum(0)
    boot = np.empty((n_boot, num.shape[1]))
    for b in range(n_boot):  # loop: a (n_boot, n, methods) gather is GBs
        w = np.bincount(rng.integers(0, n, size=n), minlength=n)
        boot[b] = (w @ num) / (w @ den)
    out: dict = {"value": value.tolist(), "error": boot.std(0, ddof=1).tolist()}
    if num.shape[1] > 1:
        diff = boot[:, :1] - boot[:, 1:]
        out["diff_vs_first"] = (value[0] - value[1:]).tolist()
        out["diff_error"] = diff.std(0, ddof=1).tolist()
    if num.shape[1] == 1:
        out = {"value": float(value[0]), "error": float(out["error"][0])}
    return out


class Study:
    """Accessor for the saved per-event arrays."""

    def __init__(self, study_dir: Path) -> None:
        d = np.load(study_dir / "hs_chain_events.npz")
        self.meta = json.loads((study_dir / "hs_chain_metadata.json").read_text())
        self.counts, self.hs, self.events = d["counts"], d["hs"], d["events"]
        self.mu = self.ev("mu_x100") / 100.0

    def ev(self, name: str) -> np.ndarray:
        return self.events[:, self.meta["event_fields"].index(name)]

    def h(self, arm: str, name: str, min_tracks: int = 1) -> np.ndarray:
        a = self.meta["arms"].index(arm)
        m = self.meta["min_tracks"].index(min_tracks)
        return self.hs[:, a, m, self.meta["hs_fields"].index(name)]

    def c(self, arm: str, name: str, min_tracks: int = 1) -> np.ndarray:
        a = self.meta["arms"].index(arm)
        m = self.meta["min_tracks"].index(min_tracks)
        return self.counts[:, a, m, self.meta["count_fields"].index(name)]


def event_flags(s: Study, min_tracks: int = 1) -> dict[str, np.ndarray]:
    """Event-level pass/fail flags for metrics (a), (b), (d)."""
    f: dict[str, np.ndarray] = {
        "a_reco_pvf_peaks": s.ev("peaks_hs_within"),
        "a_reco_amvf": s.ev("amvf_hs_within"),
        "d_native_amvf_itk": s.ev("native_is_match"),
        "d_native_amvf_itk_0p1": s.ev("native_is_match") * s.ev("native_within"),
        "d_native_amvf_plurality": s.ev("native_dominant_hs"),
    }
    for arm in s.meta["arms"]:
        f[f"a_reco_{arm}_populated"] = s.h(arm, "any_within", min_tracks)
        f[f"b_clean_hs_{arm}"] = s.h(arm, "clean_hs", min_tracks)
        f[f"b_hs_dominated_{arm}"] = s.h(arm, "hs_dominated", min_tracks)
        sel = s.h(arm, "selected_is_match", min_tracks)
        f[f"d_select_{arm}_itk"] = sel
        f[f"d_select_{arm}_itk_0p1"] = sel * s.h(arm, "match_within", min_tracks)
        f[f"d_select_{arm}_plurality"] = s.h(arm, "selected_dominant_hs", min_tracks)
    return f


def summarize_mask(
    s: Study, mask: np.ndarray, rng: np.random.Generator, min_tracks: int = 1
) -> dict:
    """All metrics for the events in ``mask``."""
    out: dict = {"events": int(mask.sum()), "mean_mu": float(s.mu[mask].mean())}
    out["fractions"] = {
        k: clopper_pearson(int(v[mask].sum()), int(mask.sum()))
        for k, v in event_flags(s, min_tracks).items()
    }
    arms = s.meta["arms"]
    nhs = np.stack([s.h(a, "n_hs_tracks", min_tracks)[mask] for a in arms], 1)
    got = np.stack([s.h(a, "match_hs_tracks", min_tracks)[mask] for a in arms], 1)
    size = np.stack([s.h(a, "match_tracks", min_tracks)[mask] for a in arms], 1)
    order = [arms.index("amvf")] + [i for i, a in enumerate(arms) if a != "amvf"]
    frac = bootstrap_ratio(got[:, order], nhs[:, order], rng)
    pur = bootstrap_ratio(got[:, order], size[:, order], rng)
    names = [arms[i] for i in order]
    out["c_hs_track_fraction"] = _unpack(frac, names)
    out["c_hs_vertex_purity"] = _unpack(pur, names)
    clean = np.stack([s.c(a, "clean", min_tracks)[mask] for a in arms], 1)[:, order]
    truth = np.stack([s.c(a, "n_truth_ge2", min_tracks)[mask] for a in arms], 1)[
        :, order
    ]
    out["clean_per_truth"] = _unpack(bootstrap_ratio(clean, truth, rng), names)
    t = s.c("amvf", "n_truth_ge2")[mask]
    stages = {}
    for key in (
        "peaks_1to1_0p5",
        "peaks_1to1_0p1694",
        "amvf_1to1_0p5",
        "amvf_1to1_0p1694",
    ):
        stages[key] = bootstrap_ratio(s.ev(key)[mask], t, rng)
    out["one_to_one_per_truth"] = stages
    out["n_truth_ge2"] = int(t.sum())
    out["n_hs_tracks"] = int(nhs[:, 0].sum())
    return out


def _unpack(res: dict, names: list[str]) -> dict:
    """Name the columns of a multi-method bootstrap result (first = AMVF)."""
    if res["value"] is None:
        return {}
    rows = {
        n: {"value": v, "error": e}
        for n, v, e in zip(names, res["value"], res["error"])
    }
    for n, d, e in zip(
        names[1:], res.get("diff_vs_first", []), res.get("diff_error", [])
    ):
        rows[n]["minus_amvf"] = -d
        rows[n]["minus_amvf_error"] = e
    return rows


def summarize(s: Study) -> dict:
    """Integrated PU200 summary plus mu bins, for reco >= 1 and >= 2 tracks."""
    rng = np.random.default_rng(SEED)
    out: dict = {
        "mu_edges": MU_EDGES.tolist(),
        "pu200_window": PU200,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED},
        "confidence": CONFIDENCE,
    }
    for mt in s.meta["min_tracks"]:
        pu = (s.mu >= PU200[0]) & (s.mu <= PU200[1])
        block = {
            "pu200": summarize_mask(s, pu, rng, mt),
            "all": summarize_mask(s, s.mu >= 0, rng, mt),
        }
        bins = []
        for j, (lo, hi) in enumerate(zip(MU_EDGES[:-1], MU_EDGES[1:])):
            upper = s.mu <= hi if j == len(MU_EDGES) - 2 else s.mu < hi
            row = summarize_mask(s, (s.mu >= lo) & upper, rng, mt)
            row["mu_low"], row["mu_high"] = float(lo), float(hi)
            bins.append(row)
        block["mu_bins"] = bins
        out[f"reco_min_tracks_{mt}"] = block
    return out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True, type=Path)
    args = p.parse_args()
    summary = summarize(Study(args.study))
    (args.study / "hs_chain_summary.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print("wrote", args.study / "hs_chain_summary.json")


if __name__ == "__main__":
    main()
