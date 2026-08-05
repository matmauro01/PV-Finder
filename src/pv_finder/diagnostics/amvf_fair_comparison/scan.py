"""The matching-window scan and the four-cell headline table.

Everything here is measured on one event list with one window applied to both
algorithms.  The window is the free parameter and is *scanned*, because any
single choice can be argued with and a curve cannot.

Two truth definitions are carried side by side:

``standard``
    ATLAS convention.  Truth = TruthVertex with nTracks >= 2.  A reco vertex
    matching none of them is a fake.  This stays the primary definition.
``corrected``
    Same efficiency denominator, but a reco vertex is only a fake if it matches
    no truth interaction with nTracks >= 1.  The difference is the population
    that is real but not a reconstructible vertex by convention.

The accidental floor of the corrected definition is measured, not assumed: the
same excusal is re-run with each event's nTracks == 1 truth list replaced by a
neighbouring event's, which keeps the z profile and the multiplicity but
destroys the association.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from pv_finder.diagnostics.amvf_fair_comparison.matching import (
    excused_by_low_ntrk,
    match_vertices,
    matched_residuals,
    matched_residuals_optimal,
)
from pv_finder.utils.pairwise_dz import DEFAULT_PAIRWISE_BINS, PAIRWISE_RANGE_MM

# Pre-registered common window for the headline table.  Chosen before looking at
# any result, on three grounds, none of which is either algorithm's fit:
#   * round physical value, conventional for a distance-based vertex match;
#   * >= 1.6x the fitted sigma_vtx-vtx of *both* algorithms (0.22 and 0.30 mm),
#     so neither has its own core cut into;
#   * about half the mean inter-vertex spacing at peak PU200 density
#     (~1.0 vertex/mm), so accidental matches stay subdominant.
HEADLINE_WINDOW_MM: float = 0.5

# Displacement for the second accidental-floor control, in mm.  Large compared
# with any matching window, small compared with the ~45 mm beam-spot sigma, so
# the local density of nTrk == 1 interactions is essentially unchanged.
CONTROL_SHIFT_MM: float = 10.0

ARMS = ("pvf", "amvf")


def sigmoid_fit(x: np.ndarray, a: float, b: float, c: float, rcc: float) -> np.ndarray:
    """Sigmoid used for the pairwise-dz dip. Identical to the production eval."""
    return a / (1.0 + np.exp(b * (rcc - np.abs(x)))) + c


def fit_sigma_vtx_vtx(
    z_by_event: list[np.ndarray],
    *,
    n_bins: int = DEFAULT_PAIRWISE_BINS,
    seed: int = 20260805,
) -> tuple[float, float, int]:
    """Fit sigma_vtx-vtx from the pairwise-dz dip of one algorithm's own output.

    Same procedure as ``run_eval_pvf_run3.py``: shuffle each event's positions,
    take signed differences over i < j, histogram on the commensurate 0.04 mm
    binning, fit the sigmoid.  Applying it to AMVF as well as to PV-Finder is
    what makes the "own sigma" view symmetric.
    """
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for z in z_by_event:
        if len(z) < 2:
            continue
        zs = np.asarray(z, dtype=np.float64).copy()
        rng.shuffle(zs)
        d = zs[:, None] - zs[None, :]
        iu = np.triu_indices(len(zs), k=1)
        dz = d[iu]
        chunks.append(dz[np.abs(dz) <= PAIRWISE_RANGE_MM])
    dz_all = np.concatenate(chunks) if chunks else np.zeros(0)
    edges = np.linspace(-PAIRWISE_RANGE_MM, PAIRWISE_RANGE_MM, n_bins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    cnts, _ = np.histogram(dz_all, bins=edges)
    base = float(np.median(cnts))
    p0 = [max(base - float(cnts.min()), 1.0), 10.0, max(base, 1.0), 0.5]
    popt, pcov = curve_fit(
        sigmoid_fit, ctrs, cnts.astype(float), p0=p0, maxfev=20000,
        bounds=([0, 0, 0, 0], [np.inf] * 4),
    )  # fmt: skip
    return float(abs(popt[3])), float(np.sqrt(np.diag(pcov))[3]), int(len(dz_all))


def per_event_counts(
    truth_ge2: list[np.ndarray],
    reco: list[np.ndarray],
    truth_eq1: list[np.ndarray],
    window_mm: float,
    *,
    control_shift: int = 1,
) -> dict[str, np.ndarray]:
    """Match one algorithm against truth at one window, event by event.

    ``excused`` counts fakes sitting on a real nTracks == 1 interaction;
    ``excused_ctrl`` is the same count with the nTracks == 1 list taken from
    another event, which is the accidental floor.
    """
    n = len(reco)
    out = {
        k: np.zeros(n, dtype=np.int64)
        for k in ("n_truth", "n_reco", "matched", "primary", "clean", "merged",
                  "split", "fake", "excused", "excused_ctrl",
                  "excused_ctrl_shift")
    }  # fmt: skip
    for i in range(n):
        m = match_vertices(truth_ge2[i], reco[i], window_mm)
        out["n_truth"][i] = m.n_truth
        out["n_reco"][i] = m.n_reco
        out["matched"][i] = m.n_truth_matched
        out["primary"][i] = m.n_truth_primary
        out["clean"][i] = m.reco_clean
        out["merged"][i] = m.reco_merged
        out["split"][i] = m.reco_split
        out["fake"][i] = m.reco_fake
        fake_z = reco[i][m.fake_idx] if len(m.fake_idx) else reco[i][:0]
        out["excused"][i] = excused_by_low_ntrk(fake_z, truth_eq1[i], window_mm)
        # Accidental floor: the same fakes against another event's nTrk == 1
        # list.  Same z profile and multiplicity, no association.
        out["excused_ctrl"][i] = excused_by_low_ntrk(
            fake_z, truth_eq1[(i + control_shift) % n], window_mm
        )
        # Second, independent control: this event's own list, displaced far
        # enough to break the association but not the local density.
        out["excused_ctrl_shift"][i] = excused_by_low_ntrk(
            fake_z, truth_eq1[i] + CONTROL_SHIFT_MM, window_mm
        )
    return out


def _metrics(c: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, float]:
    """Efficiency and fake rates on a (possibly resampled) set of events."""
    n_truth = c["n_truth"][idx].sum()
    n_evt = len(idx)
    fake = c["fake"][idx].sum()
    split = c["split"][idx].sum()
    excused = c["excused"][idx].sum()
    return {
        "eff": float(c["matched"][idx].sum() / n_truth) if n_truth else 0.0,
        # Strict one-to-one efficiency: no reco is credited with more than one
        # truth vertex.  The gap to "eff" is the merge convention, and it widens
        # with the window, so the absolute efficiency must never be quoted
        # without saying which of the two it is.
        "eff_strict": float(c["primary"][idx].sum() / n_truth) if n_truth else 0.0,
        "fake_per_evt": float(fake / n_evt),
        "fake_corr_per_evt": float((fake - excused) / n_evt),
        "excused_per_evt": float(excused / n_evt),
        "excused_ctrl_per_evt": float(c["excused_ctrl"][idx].sum() / n_evt),
        "excused_ctrl_shift_per_evt": float(c["excused_ctrl_shift"][idx].sum() / n_evt),
        # Genuine excess over the accidental floor: the part of the correction
        # that is an association rather than a coincidence.
        "excused_net_per_evt": float((excused - c["excused_ctrl"][idx].sum()) / n_evt),
        "reco_per_evt": float(c["n_reco"][idx].sum() / n_evt),
        "split_per_evt": float(split / n_evt),
        # Surplus = fake + split: every reco vertex that did not win a truth
        # vertex of its own.  Immune to the fake/split relabelling that a wider
        # window causes, so it is the honest way to read the window scan.
        "surplus_per_evt": float((fake + split) / n_evt),
        "truth_per_evt": float(n_truth / n_evt),
    }


METRIC_KEYS = (
    "eff", "eff_strict", "fake_per_evt", "fake_corr_per_evt", "surplus_per_evt",
    "excused_per_evt", "excused_ctrl_per_evt", "excused_ctrl_shift_per_evt",
    "excused_net_per_evt", "reco_per_evt", "split_per_evt",
)  # fmt: skip


def bootstrap_point(
    counts: dict[str, dict[str, np.ndarray]],
    *,
    n_boot: int = 1000,
    seed: int = 20260805,
) -> dict[str, Any]:
    """Paired bootstrap over events for one window.

    ``counts`` maps arm name -> per-event count arrays.  The *same* resampled
    event indices are used for every arm in every replica, so the PV-Finder minus
    AMVF differences carry a paired error, which is much smaller than either
    absolute error and is the number the comparison rests on.
    """
    n = len(next(iter(counts.values()))["fake"])
    rng = np.random.default_rng(seed)
    point = {a: _metrics(c, np.arange(n)) for a, c in counts.items()}

    reps: dict[str, dict[str, list[float]]] = {
        a: {k: [] for k in METRIC_KEYS} for a in counts
    }
    diffs: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = {a: _metrics(c, idx) for a, c in counts.items()}
        for a in counts:
            for k in METRIC_KEYS:
                reps[a][k].append(m[a][k])
        if set(counts) >= {"pvf", "amvf"}:
            for k in METRIC_KEYS:
                diffs[k].append(m["pvf"][k] - m["amvf"][k])

    out: dict[str, Any] = {}
    for a in counts:
        out[a] = dict(point[a])
        for k in METRIC_KEYS:
            out[a][k + "_err"] = float(np.std(reps[a][k], ddof=1))
    if diffs["eff"]:
        out["diff"] = {}
        for k in METRIC_KEYS:
            out["diff"][k] = float(point["pvf"][k] - point["amvf"][k])
            out["diff"][k + "_err"] = float(np.std(diffs[k], ddof=1))
    return out


def run_scan(
    data: dict[str, Any],
    windows: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 20260805,
    verbose: bool = True,
) -> dict[str, Any]:
    """Match both algorithms against truth at every window in ``windows``."""
    reco = {"pvf": data["pvf"], "amvf": data["amvf"]}
    t2, t1 = data["truth_ge2"], data["truth_eq1"]
    rows: list[dict[str, Any]] = []
    for w in windows:
        counts = {a: per_event_counts(t2, reco[a], t1, float(w)) for a in ARMS}
        row = bootstrap_point(counts, n_boot=n_boot, seed=seed)
        row["window_mm"] = float(w)
        # Accidental floor, per arm, at this window — the overlay that tells the
        # reader where the efficiency stops meaning "found the vertex".
        for a in ARMS:
            acc = accidental_efficiency(t2, reco[a], float(w))
            row[a]["eff_accidental"] = acc["eff"]
            row[a]["eff_strict_accidental"] = acc["eff_strict"]
        rows.append(row)
        if verbose:
            p, m = row["pvf"], row["amvf"]
            print(
                f"  w={w:5.3f} | PVF eff={p['eff']:.4f} fake={p['fake_per_evt']:6.2f}"
                f" corr={p['fake_corr_per_evt']:5.2f} surp={p['surplus_per_evt']:5.2f}"
                f" | AMVF eff={m['eff']:.4f} fake={m['fake_per_evt']:6.2f}"
                f" corr={m['fake_corr_per_evt']:5.2f} surp={m['surplus_per_evt']:5.2f}"
                f" | acc={100 * p['eff_accidental']:4.1f}/{100 * m['eff_accidental']:4.1f}%",
                flush=True,
            )
    return {"rows": rows, "windows": [float(w) for w in windows]}


def self_sigma_view(
    data: dict[str, Any], *, n_boot: int = 400, seed: int = 20260805
) -> dict[str, Any]:
    """Each algorithm judged by a window equal to its OWN fitted sigma.

    This is the convention the production eval uses for PV-Finder, except that
    the production eval then applies PV-Finder's sigma to AMVF as well.  Both
    variants are reported here: the circular-but-symmetric one (each by its own)
    and the actually-published one (both by PV-Finder's).
    """
    out: dict[str, Any] = {"sigma": {}}
    for arm, key in (("pvf", "pvf"), ("amvf", "amvf")):
        s, err, npairs = fit_sigma_vtx_vtx(data[key])
        out["sigma"][arm] = {"sigma_mm": s, "err_mm": err, "n_pairs": npairs}
    t2, t1 = data["truth_ge2"], data["truth_eq1"]

    def one(arm: str, window: float) -> dict[str, Any]:
        c = per_event_counts(t2, data[arm], t1, window)
        r = bootstrap_point({arm: c}, n_boot=n_boot, seed=seed)[arm]
        r["window_mm"] = window
        return r

    s_pvf = out["sigma"]["pvf"]["sigma_mm"]
    s_amvf = out["sigma"]["amvf"]["sigma_mm"]
    out["each_by_own"] = {"pvf": one("pvf", s_pvf), "amvf": one("amvf", s_amvf)}
    out["both_by_pvf"] = {"pvf": one("pvf", s_pvf), "amvf": one("amvf", s_pvf)}
    return out


def core_position_resolution(
    truth_ge2: list[np.ndarray],
    reco: list[np.ndarray],
    *,
    window_mm: float = 1.0,
    pct: float = 68.0,
    optimal: bool = True,
) -> float:
    """Percentile of |Δz| over one-to-one matched pairs, in mm.

    This — not sigma_vtx-vtx — is the quantity a matching window should be built
    on.  sigma_vtx-vtx measures two-vertex *separation* and differs by 40 %
    between the two algorithms; the core position resolution is nearly identical
    for both, so a window derived from it is fair to both by construction.

    ``optimal`` selects a globally optimal assignment rather than the production
    greedy one.  It is the default here because greedy biases the width low (see
    ``matched_residuals_optimal``), and because a *smaller* core resolution
    yields a *tighter* window, which is the direction that flatters PV-Finder —
    its lead grows as the window tightens.  Taking the conservative estimator is
    the point.
    """
    fn = matched_residuals_optimal if optimal else matched_residuals
    chunks = [fn(t, r, window_mm) for t, r in zip(truth_ge2, reco) if len(t) and len(r)]
    res = np.concatenate([c for c in chunks if len(c)]) if chunks else np.zeros(0)
    return float(np.percentile(res, pct)) if len(res) else float("nan")


def accidental_efficiency(
    truth_ge2: list[np.ndarray],
    reco: list[np.ndarray],
    window_mm: float,
    *,
    shift_mm: float = 3.0,
) -> dict[str, float]:
    """Efficiency obtained by a reco list displaced far enough to be wrong.

    At PU200 the local truth density is ~0.9 vertices/mm, so at a wide window a
    reconstructed vertex lands within the window of *some* truth vertex whether
    or not it found it.  Displacing the whole list by ``shift_mm`` — far larger
    than any window, far smaller than the ~45 mm beam spot — leaves the density
    unchanged and the association destroyed, so the efficiency it still scores
    is pure coincidence.  This is the floor below which the efficiency stops
    meaning "found the vertex", and it is what tells the reader that a 0.5 mm
    window is too generous at this pileup.

    Returned for both conventions, because the merge credit inflates the
    accidental floor just as it inflates the efficiency.
    """
    n_truth = n_hit = n_strict = 0
    for t, r in zip(truth_ge2, reco):
        m = match_vertices(t, r + np.float32(shift_mm), window_mm)
        n_truth += m.n_truth
        n_hit += m.n_truth_matched
        n_strict += m.n_truth_primary
    if not n_truth:
        return {"eff": 0.0, "eff_strict": 0.0}
    return {"eff": n_hit / n_truth, "eff_strict": n_strict / n_truth}


def matched_multiplicity_view(
    data: dict[str, Any],
    window: float,
    *,
    n_boot: int = 400,
    seed: int = 20260805,
) -> dict[str, Any]:
    """PV-Finder re-cut so it emits as many candidates per event as AMVF.

    PV-Finder reports ~101 candidates/event against AMVF's ~98 at the deployed
    operating point, so a sceptic can argue its efficiency advantage is bought
    by simply emitting more.  This raises PV-Finder's peak-height floor — the
    single knob the operating point already uses — until the two candidate
    multiplicities agree, and re-measures against AMVF with a paired bootstrap.

    Filtering the stored peak heights after the fact is *exactly* equivalent to
    re-running the peak finder with a higher ``--min-height``: that threshold
    enters ``pv_locations_updated_res`` only as ``targets[currentmax] >=
    min_height``, which decides whether a region is recorded and never moves a
    region boundary or a position.  ``pred_heights`` stores that same
    ``targets[currentmax]``.  So no inference and no peak finding is re-run.
    """
    heights = np.concatenate([h for h in data["pvf_h"] if len(h)])
    n_evt = len(data["pvf"])
    target_total = int(round(sum(len(v) for v in data["amvf"])))
    if target_total >= len(heights):
        return {"applicable": False, "reason": "AMVF emits at least as many"}
    # Keep the target_total tallest peaks: threshold is the corresponding
    # order statistic of the pooled height distribution.
    thr = float(
        np.partition(heights, len(heights) - target_total)[len(heights) - target_total]
    )
    cut = [z[h >= thr] for z, h in zip(data["pvf"], data["pvf_h"])]
    t2, t1 = data["truth_ge2"], data["truth_eq1"]
    # Named "pvf"/"amvf" so bootstrap_point emits the paired difference.
    counts = {
        "pvf": per_event_counts(t2, cut, t1, window),
        "amvf": per_event_counts(t2, data["amvf"], t1, window),
    }
    res = bootstrap_point(counts, n_boot=n_boot, seed=seed)
    res.update(
        applicable=True,
        height_threshold=thr,
        window_mm=window,
        candidates_per_evt=sum(len(v) for v in cut) / n_evt,
        amvf_candidates_per_evt=target_total / n_evt,
    )
    return res


def crossings(rows: list[dict[str, Any]], arm_metric: str) -> list[float]:
    """Windows at which PV-Finder crosses AMVF on ``arm_metric``.

    Linear interpolation between scan points on the sign change of the paired
    difference.  Returns an empty list if the difference never changes sign.
    """
    ws = np.array([r["window_mm"] for r in rows])
    d = np.array([r["pvf"][arm_metric] - r["amvf"][arm_metric] for r in rows])
    out: list[float] = []
    for i in range(len(d) - 1):
        if d[i] == 0.0:
            out.append(float(ws[i]))
        elif d[i] * d[i + 1] < 0:
            t = d[i] / (d[i] - d[i + 1])
            out.append(float(ws[i] + t * (ws[i + 1] - ws[i])))
    return out
