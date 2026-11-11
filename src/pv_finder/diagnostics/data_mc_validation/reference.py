"""Data-style matching against an arbitrary reference, with a paired bootstrap.

This reproduces exactly what ``run_eval_pvf_run3.py`` does on collision data:

* positions converted to model-bin units, ``(z - Z_MIN) / 0.04``, and the
  window to ``sigma / 0.04`` before matching (the production matcher works in
  bins; doing the same keeps every ``<=`` boundary decision identical);
* events with an empty reference are dropped from every total;
* reference recovery = (clean + merged reference vertices) / all reference
  vertices, i.e. merged reference vertices count as recovered;
* "no reference / event" = reco peaks with no reference vertex inside the
  window (the matcher's *fake* class) per kept event; split peaks are a
  separate class and excluded from it.

The matcher is ``amvf_fair_comparison.matching.match_vertices``, asserted
count-for-count equal to the production ``compare_res_reco`` by
``tests/test_amvf_fair_matching.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pv_finder.diagnostics.amvf_fair_comparison.matching import match_vertices
from pv_finder.diagnostics.data_mc_validation.samples import BIN_WIDTH_MM, Z_MIN_MM

METRICS: tuple = ("recovery", "no_ref_per_evt", "split_per_evt")


@dataclass
class EventCounts:
    """Per-event match outcome; all arrays have one entry per input event."""

    valid: np.ndarray  # reference non-empty (event enters the totals)
    n_ref: np.ndarray
    n_reco: np.ndarray
    ref_matched: np.ndarray  # clean + merged on the reference side
    ref_primary: np.ndarray  # one-to-one only, no merge credit
    clean: np.ndarray
    merged: np.ndarray
    split: np.ndarray
    no_ref: np.ndarray  # matcher "fake" class

    def summary(self) -> dict[str, float]:
        v = self.valid
        n_evt = int(v.sum())
        n_ref = int(self.n_ref[v].sum())
        out = {
            "n_events": n_evt,
            "n_ref": n_ref,
            "n_reco": int(self.n_reco[v].sum()),
            "ref_matched": int(self.ref_matched[v].sum()),
            "clean": int(self.clean[v].sum()),
            "merged": int(self.merged[v].sum()),
            "split": int(self.split[v].sum()),
            "no_ref": int(self.no_ref[v].sum()),
        }
        out["recovery"] = out["ref_matched"] / n_ref if n_ref else float("nan")
        out["recovery_one_to_one"] = (
            float(self.ref_primary[v].sum()) / n_ref if n_ref else float("nan")
        )
        out["no_ref_per_evt"] = out["no_ref"] / n_evt if n_evt else float("nan")
        out["split_per_evt"] = out["split"] / n_evt if n_evt else float("nan")
        return out


def _to_bins(z: np.ndarray) -> np.ndarray:
    return (np.asarray(z) - Z_MIN_MM) / BIN_WIDTH_MM


def match_events(
    refs: Sequence[np.ndarray], recos: Sequence[np.ndarray], window_mm: float
) -> EventCounts:
    """Match every event's reco list to its reference list at one window."""
    if len(refs) != len(recos):
        raise ValueError(f"{len(refs)} reference events vs {len(recos)} reco events")
    n = len(refs)
    cols = {k: np.zeros(n, dtype=np.int64) for k in (
        "n_ref", "n_reco", "ref_matched", "ref_primary", "clean", "merged",
        "split", "no_ref")}  # fmt: skip
    win_bins = window_mm / BIN_WIDTH_MM
    for i, (ref, reco) in enumerate(zip(refs, recos)):
        cols["n_ref"][i], cols["n_reco"][i] = len(ref), len(reco)
        if len(ref) == 0:
            continue
        m = match_vertices(_to_bins(ref), _to_bins(reco), win_bins)
        cols["ref_matched"][i] = m.n_truth_matched
        cols["ref_primary"][i] = m.n_truth_primary
        cols["clean"][i], cols["merged"][i] = m.reco_clean, m.reco_merged
        cols["split"][i], cols["no_ref"][i] = m.reco_split, m.reco_fake
    return EventCounts(valid=cols["n_ref"] > 0, **cols)


def _metrics_from_sums(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "recovery": s["ref_matched"] / s["n_ref"],
        "no_ref_per_evt": s["no_ref"] / s["valid"],
        "split_per_evt": s["split"] / s["valid"],
    }


def bootstrap(
    counts: Sequence[EventCounts],
    n_boot: int = 10_000,
    seed: int = 20261109,
    chunk: int = 500,
) -> list[dict[str, np.ndarray]]:
    """Event bootstrap of the three metrics, SAME draws for every input.

    All inputs must describe the same events in the same order (paired); an
    event that is invalid for one input simply contributes nothing to it.
    Returns, per input, ``{metric: array of n_boot replicas}``.
    """
    n = len(counts[0].valid)
    if any(len(c.valid) != n for c in counts):
        raise ValueError("paired bootstrap needs the same events for every input")
    keys = ("valid", "n_ref", "ref_matched", "no_ref", "split")
    cols = [
        {k: getattr(c, k).astype(np.float64) * c.valid for k in keys} for c in counts
    ]
    rng = np.random.default_rng(seed)
    out: list[dict[str, list[np.ndarray]]] = [{m: [] for m in METRICS} for _ in counts]
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        for c, o in zip(cols, out):
            sums = {k: v[idx].sum(axis=1) for k, v in c.items()}
            for m, val in _metrics_from_sums(sums).items():
                o[m].append(val)
        done += b
    return [{m: np.concatenate(v) for m, v in o.items()} for o in out]


def paired_delta(
    a: EventCounts, b: EventCounts, n_boot: int = 10_000, seed: int = 20261109
) -> dict[str, dict[str, float]]:
    """delta = metric(a) - metric(b), with paired-bootstrap standard errors."""
    sa, sb = a.summary(), b.summary()
    ba, bb = bootstrap([a, b], n_boot=n_boot, seed=seed)
    res: dict[str, dict[str, float]] = {}
    for m in METRICS:
        d = ba[m] - bb[m]
        res[m] = {
            "a": sa[m],
            "a_err": float(np.std(ba[m])),
            "b": sb[m],
            "b_err": float(np.std(bb[m])),
            "delta": sa[m] - sb[m],
            "delta_err": float(np.std(d)),
            "delta_ci68": [float(np.percentile(d, 16)), float(np.percentile(d, 84))],
            "corr": float(np.corrcoef(ba[m], bb[m])[0, 1]),
        }
    return res


def summary_with_errors(
    c: EventCounts, n_boot: int = 10_000, seed: int = 20261109
) -> dict[str, float]:
    """``EventCounts.summary`` plus event-bootstrap standard errors."""
    s = c.summary()
    (bs,) = bootstrap([c], n_boot=n_boot, seed=seed)
    for m in METRICS:
        s[m + "_err"] = float(np.std(bs[m]))
    return s
