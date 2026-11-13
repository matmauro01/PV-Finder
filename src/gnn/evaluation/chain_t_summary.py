"""Aggregate ``chain_t_metrics`` per-event records into mu bins, with bootstrap errors.

Every ratio is a ratio of sums over events (never a mean of per-event
ratios). Errors are the standard deviation over event-bootstrap replicas;
the same replicas serve every arm, so differences against AMVF are paired.
Bookkeeping identities are asserted on load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pv_finder.evaluation.vertex_finding.itk_resolution import iterative_rms

MU_BINS: dict[str, tuple[float, float]] = {
    "0-20": (0, 20), "20-50": (20, 50), "50-100": (50, 100),
    "100-150": (100, 150), "150-200": (150, 200.5),
    "inclusive": (0, 1e9), "PU200 (185-215)": (185, 215.5),
}  # fmt: skip
RATIOS: dict[str, tuple[str, str]] = {
    # name: (numerator field, denominator field); "n_truth_ge2" / "n_labelled"
    # are event fields, the rest count fields.
    "clean_per_truth": ("clean", "n_truth_ge2"),
    "clean_per_reco": ("clean", "n_reco"),
    "merged_per_reco": ("merged", "n_reco"),
    "split_per_reco": ("split", "n_reco"),
    "fake_per_reco": ("fake", "n_reco"),
    "reco_per_truth": ("n_reco", "n_truth_ge2"),
    "pos_eff": ("p_truth_found", "n_truth_ge2"),
    "pos_clean_per_reco": ("p_clean", "n_reco"),
    "pos_merged_per_reco": ("p_merged", "n_reco"),
    "pos_split_per_reco": ("p_split", "n_reco"),
    "pos_fake_per_reco": ("p_fake", "n_reco"),
    "trk_eff": ("n_correct", "n_labelled"),
    "trk_pur": ("n_correct", "n_assigned_truth"),
    "assigned_frac": ("n_assigned", "n_tracks"),
    "hs_clean": ("hs_clean", "has_hs"),
    "hs_track_frac": ("hs_match_hs_tracks", "n_hs_tracks"),
    "hs_purity": ("hs_match_hs_tracks", "hs_match_tracks"),
    "merged_per_event": ("merged", "one"),
    "split_per_event": ("split", "one"),
    "fake_per_event": ("fake", "one"),
}


@dataclass
class ChainRecord:
    """One ``chain_t_metrics`` output: per-event arrays and their field names."""

    counts: np.ndarray  # (n, arms, min_tracks, fields)
    sel: np.ndarray  # (n, sel_arms, 4)
    res: np.ndarray  # (n, arms) residual [mm]
    fallback: np.ndarray
    events: np.ndarray
    meta: dict

    @property
    def arms(self) -> list[str]:
        """Arm labels: ``t0.98`` ... then ``amvf``."""
        return [f"t{t:g}" for t in self.meta["t_grid"]] + ["amvf"]

    def ev(self, name: str) -> np.ndarray:
        """One event field."""
        return self.events[:, self.meta["event_fields"].index(name)]

    def cf(self, name: str, m: int = 0) -> np.ndarray:
        """One count field, all arms: (n, arms)."""
        return self.counts[:, :, m, self.meta["count_fields"].index(name)]


def load(path: str | Path) -> ChainRecord:
    """Load a metrics npz (+ json) and assert the bookkeeping identities."""
    path = Path(path)
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    rec = ChainRecord(z["counts"], z["sel"], z["res"], z["fallback"], z["events"], meta)
    check_bookkeeping(rec)
    return rec


def check_bookkeeping(rec: ChainRecord) -> None:
    """Category sums equal vertex counts; sub-counts never exceed totals."""
    for m in range(rec.counts.shape[2]):
        tp = sum(rec.cf(k, m) for k in ("clean", "merged", "split", "fake"))
        pos = sum(rec.cf(k, m) for k in ("p_clean", "p_merged", "p_split", "p_fake"))
        if not (np.array_equal(tp, rec.cf("n_reco", m)) and np.array_equal(pos, rec.cf("n_reco", m))):  # fmt: skip
            raise AssertionError("category bookkeeping fails")
        if np.any(rec.cf("p_truth_found", m) > rec.ev("n_truth_ge2")[:, None]):
            raise AssertionError("more truth found than truth vertices")
        if np.any(rec.cf("n_correct", m) > rec.cf("n_assigned_truth", m)):
            raise AssertionError("more correct than assigned tracks")
        if np.any(rec.cf("n_assigned_truth", m) > rec.ev("n_labelled")[:, None]):
            raise AssertionError("more assigned than labelled tracks")
    if np.any(rec.cf("n_reco", 1) > rec.cf("n_reco", 0)):
        raise AssertionError(">= 2 track vertices exceed >= 1 track vertices")


def _field(rec: ChainRecord, name: str, m: int) -> np.ndarray:
    """(n, arms) array of a count field or a broadcast event field."""
    n_arms = rec.counts.shape[1]
    if name == "one":
        return np.ones((len(rec.events), n_arms))
    if name in rec.meta["event_fields"]:
        return np.repeat(rec.ev(name)[:, None], n_arms, axis=1)
    return rec.cf(name, m).astype(np.float64)


def boot_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) multiplicities of an event bootstrap."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(n_boot)])  # fmt: skip


def ratio_table(
    rec: ChainRecord, mask: np.ndarray, w: np.ndarray, m: int = 0
) -> dict[str, dict[str, np.ndarray]]:
    """{ratio: {'value': (arms,), 'err': (arms,), 'diff_err': (arms,)}} on ``mask``.

    ``w`` are bootstrap weights over the masked events; ``diff_err`` is the
    paired error of (arm - AMVF).
    """
    out = {}
    for name, (num, den) in RATIOS.items():
        a, b = _field(rec, num, m)[mask], _field(rec, den, m)[mask]
        value = a.sum(0) / b.sum(0)
        reps = (w @ a) / (w @ b)
        out[name] = {
            "value": value,
            "err": reps.std(0, ddof=1),
            "diff_err": (reps - reps[:, -1:]).std(0, ddof=1),
        }
    return out


def selection_table(
    rec: ChainRecord, mask: np.ndarray, w: np.ndarray
) -> dict[str, np.ndarray]:
    """HS reconstructed-and-selected fraction per selection arm (t grid, AMVF stored, AMVF rule)."""
    has = rec.ev("has_hs")[mask] > 0  # events without a stored HS are excluded
    w, mask = w[:, has], mask & (rec.ev("has_hs") > 0)
    x = rec.sel[mask, :, rec.meta["sel_fields"].index("reco_sel")].astype(np.float64)
    reps = (w @ x) / w.sum(1, keepdims=True)
    stored = len(rec.meta["t_grid"])
    return {
        "value": x.mean(0), "k": x.sum(0), "n": int(mask.sum()),
        "err": reps.std(0, ddof=1),
        "diff_stored_err": (reps - reps[:, stored : stored + 1]).std(0, ddof=1),
    }  # fmt: skip


def resolution_table(
    rec: ChainRecord, mask: np.ndarray, n_boot: int, seed: int, halfrange: float = 0.1
) -> dict[str, np.ndarray]:
    """Iterative-RMS HS width [um] per arm on the events where both the arm and AMVF exist."""
    res = rec.res[mask]
    rng = np.random.default_rng(seed)
    n_arms = res.shape[1]
    out = {k: np.full(n_arms, np.nan) for k in ("rms_um", "err_um", "amvf_um", "diff_err_um", "n")}  # fmt: skip
    for a in range(n_arms):
        common = np.isfinite(res[:, a]) & np.isfinite(res[:, -1])
        r, ra = res[common, a], res[common, -1]
        if len(r) < 20:
            continue
        out["rms_um"][a] = 1e3 * iterative_rms(r, halfrange).rms
        out["amvf_um"][a] = 1e3 * iterative_rms(ra, halfrange).rms
        out["n"][a] = len(r)
        b, d = [], []
        for _ in range(n_boot):
            i = rng.integers(0, len(r), len(r))
            x, y = (
                iterative_rms(r[i], halfrange).rms,
                iterative_rms(ra[i], halfrange).rms,
            )
            b.append(x)
            d.append(x - y)
        out["err_um"][a] = 1e3 * np.std(b, ddof=1)
        out["diff_err_um"][a] = 1e3 * np.std(d, ddof=1)
    return out


def summarize(
    rec: ChainRecord, n_boot: int = 200, seed: int = 20261112, bins: dict | None = None
) -> dict:
    """All tables for every mu bin (JSON-safe)."""
    mu = rec.ev("mu")
    out: dict = {"arms": rec.arms, "sel_arms": rec.meta["sel_arms"], "bins": {}}
    for label, (lo, hi) in (bins or MU_BINS).items():
        mask = (mu >= lo) & (mu < hi)
        w = boot_weights(int(mask.sum()), n_boot, seed)
        entry = {
            "n_events": int(mask.sum()),
            "mean_mu": float(mu[mask].mean()),
            "ratios_min1": ratio_table(rec, mask, w, 0),
            "ratios_min2": ratio_table(rec, mask, w, 1),
        }
        if rec.meta["hs_computed"]:
            entry["hs_selection"] = selection_table(rec, mask, w)
            entry["hs_resolution"] = resolution_table(rec, mask, n_boot, seed)
        out["bins"][label] = entry
    return _jsonable(out)


def _jsonable(x):  # noqa: ANN001, ANN202
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return [None if (isinstance(v, float) and not np.isfinite(v)) else v for v in x.tolist()]  # fmt: skip
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x
