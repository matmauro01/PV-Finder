"""Resolution versus pile-up on the flat-mu HL-LHC samples.

Two resolution measures per mu bin, for PV-Finder and AMVF on the same events:

- ``sigma_vtx-vtx``: half-width of the dip in the signed pairwise Delta z of an
  algorithm's own vertices, from the production sigmoid fit
  (``run2_vs_run4.metrics.event_hists`` / ``sigma_summary``, which are verified
  bit for bit against ``amvf_fair_comparison.scan.fit_sigma_vtx_vtx``).
- the 68th percentile of |z_reco - z_truth| over the optimal one-to-one
  assignment within 1 mm, truth vertices with nTracks >= 2
  (``run2_vs_run4.metrics.per_event_residuals``).

The statistical error of both is an event bootstrap.  This module holds the
event selection and the per-bin bookkeeping; inference and I/O live in
``scripts/resolution_vs_mu.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pv_finder.diagnostics.run2_vs_run4.metrics import (
    SEED,
    event_hists,
    per_event_residuals,
    sigma_summary,
)


def mu_bin_index(mu: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin index of each mu, with the last bin closed on the right.

    Bins are ``[e_k, e_k+1)`` except the last, ``[e_n-1, e_n]``, so that the
    integer mu = 200 of the flat samples falls in the top bin.  Outside the
    edges the index is -1.
    """
    mu = np.asarray(mu, dtype=float)
    edges = np.asarray(edges, dtype=float)
    k = np.searchsorted(edges, mu, side="right") - 1
    k[mu == edges[-1]] = len(edges) - 2
    k[(mu < edges[0]) | (mu > edges[-1])] = -1
    return k


def select_entries(
    mu_by_file: dict[str, np.ndarray],
    edges: np.ndarray,
    target_per_bin: np.ndarray,
) -> dict[str, np.ndarray]:
    """Choose entries so that each mu bin holds at most ``target_per_bin`` events.

    Deterministic and outcome-blind: files are taken in the given order and,
    within a file, entries in increasing entry number.  A bin takes events
    from the first file until it is full, then from the next one.
    """
    target = np.asarray(target_per_bin, dtype=int)
    if len(target) != len(edges) - 1:
        raise ValueError("target_per_bin must have one entry per mu bin")
    still = target.copy()
    chosen: dict[str, np.ndarray] = {}
    for name, mu in mu_by_file.items():
        k = mu_bin_index(mu, edges)
        keep = np.zeros(len(mu), dtype=bool)
        for b in range(len(target)):
            cand = np.flatnonzero(k == b)[: max(int(still[b]), 0)]
            keep[cand] = True
            still[b] -= len(cand)
        chosen[name] = np.flatnonzero(keep)
    return chosen


def bin_summary(
    pvf: list[np.ndarray],
    amvf: list[np.ndarray],
    truth: list[np.ndarray],
    mu: np.ndarray,
    edges: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Per-mu-bin sigma_vtx-vtx and p68 for both algorithms.

    Returns the JSON-able rows and the per-event pairwise histograms (for the
    per-bin fit plots).  The two algorithms use the same events in every bin.
    """
    if not (len(pvf) == len(amvf) == len(truth) == len(mu)):
        raise ValueError("pvf, amvf, truth and mu must describe the same events")
    hists: dict[str, np.ndarray] = {}
    residuals: dict[str, list[np.ndarray]] = {}
    for name, reco in (("pvf", pvf), ("amvf", amvf)):
        hists[name], hz_edges = event_hists(reco, seed=seed)
        residuals[name] = per_event_residuals(truth, reco)
    hists["edges"] = hz_edges
    k = mu_bin_index(mu, edges)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for b in range(len(edges) - 1):
        sel = np.flatnonzero(k == b)
        row: dict[str, Any] = {
            "mu_lo": float(edges[b]), "mu_hi": float(edges[b + 1]),
            "mu_mean": float(np.mean(mu[sel])) if len(sel) else float("nan"),
            "n_events": int(len(sel)),
        }  # fmt: skip
        boots = [rng.choice(sel, len(sel)) for _ in range(n_boot)] if len(sel) else []
        for name in ("pvf", "amvf"):
            s = sigma_summary(hists[name], hz_edges, sel, n_boot=n_boot, seed=seed)
            row[name] = {**s, **_p68(residuals[name], sel, boots)}
        rows.append(row)
    return rows, hists


def _p68(res: list[np.ndarray], sel: np.ndarray, boots: list[np.ndarray]) -> dict:
    """68th percentile of |dz| in um over the selected events, bootstrap error."""
    if len(sel) == 0:
        return {"p68_um": float("nan"), "p68_err_um": float("nan"), "n_matched": 0}
    cat = np.concatenate([res[i] for i in sel])
    if len(cat) == 0:
        return {"p68_um": float("nan"), "p68_err_um": float("nan"), "n_matched": 0}
    bs = []
    for idx in boots:
        c = np.concatenate([res[i] for i in idx])
        if len(c):
            bs.append(np.percentile(c, 68.0))
    err = float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan")
    return {"p68_um": 1e3 * float(np.percentile(cat, 68.0)), "p68_err_um": 1e3 * err,
            "n_matched": int(len(cat))}  # fmt: skip


def fit_chi2(
    edges: np.ndarray, cnts: np.ndarray, popt: np.ndarray
) -> tuple[float, int]:
    """Poisson-Neyman chi2 and ndf of the sigmoid on the pairwise histogram.

    Bins with zero counts use a variance of 1.  ndf = bins - 4 parameters.
    """
    from pv_finder.diagnostics.amvf_fair_comparison.scan import sigmoid_fit

    ctrs = 0.5 * (edges[:-1] + edges[1:])
    pred = sigmoid_fit(ctrs, *popt)
    var = np.maximum(np.asarray(cnts, dtype=float), 1.0)
    return float(np.sum((cnts - pred) ** 2 / var)), int(len(cnts) - 4)
