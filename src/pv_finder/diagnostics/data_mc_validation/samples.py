"""Inputs for the Section 6.2 data/MC study: pkls, MC ROOT vertices, sigma fit.

Conventions reproduced from the April 2026 evaluations (commit 121d00f for
``run_eval_pvf_run3.py``, 93bff27 for ``run_eval_pvf.py``), which produced the
note's numbers:

* pairwise-dz histogram: 60 bins over +-6 mm (0.2 mm), integral threshold 0.5;
* sigmoid ``a / (1 + exp(b (rcc - |x|))) + c`` fitted to bin counts with bounds
  ``[0, inf)``; sigma = |rcc|;
* start values: adaptive on collision data, ``FIT_P0 = [1000, 10, 30, 0.8]`` on
  MC.  Both are kept so the stored sigmas reproduce; ``fit_sigma`` takes either.

The current eval defaults (300 bins, min height 0.03) are deliberately NOT
used: they would change the quoted numbers.

MC event mapping: the MC eval iterates the h5 test events
``event_idx = 48450 + i`` and reads ROOT entry ``qibin[i]`` from
``configs/qibin_test_main_indices_v2.p``.  It is a permutation, not the
identity; ``verify_mc_mapping`` checks it against the stored truth.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import curve_fit

Z_MIN_MM: float = -240.0
BIN_WIDTH_MM: float = 0.04
FIT_RANGE_MM: float = 6.0
N_FIT_BINS: int = 60
MC_FIT_P0: tuple = (1000.0, 10.0, 30.0, 0.8)

MC_ROOT_BRANCHES: tuple = (
    "TruthVertex_z",
    "TruthVertex_nTracks",
    "RecoVertex_z",
    "RecoVertex_nTracks",
    "ActualNumOfInt",
    "BeamPosZ",
)


@dataclass
class Sample:
    """One evaluated sample, as stored in its ``eval_results.pkl``.

    ``ref_pkl`` is what the pkl calls ``truth_pvs_mm``: on collision data the
    beam-corrected AMVF nTrk >= 2 vertices (the reference actually matched); on
    MC the h5 ``pv`` truth list *without* the nTrk >= 2 filter (the MC eval
    matched against ROOT truth instead, see ``load_mc_root``).
    """

    name: str
    pkl_path: Path
    pred: list[np.ndarray]
    ref_pkl: list[np.ndarray]
    mu: np.ndarray
    beam_z: np.ndarray
    event_idx: np.ndarray
    n_pred: np.ndarray
    n_ref: np.ndarray
    n_amvf: np.ndarray | None
    dz: np.ndarray
    raw: dict

    @property
    def n_events(self) -> int:
        return len(self.pred)


def load_sample(name: str, pkl_path: Path) -> Sample:
    """Load an eval pkl and check its per-event records line up with the lists."""
    with open(pkl_path, "rb") as fh:
        r = pickle.load(fh)
    pe = r["per_event"]
    pred = [np.asarray(p) for p in r["pred_pvs_mm"]]
    ref = [np.asarray(t) for t in r["truth_pvs_mm"]]
    if not (len(pe) == len(pred) == len(ref) == r["n_events"]):
        raise ValueError(
            f"{pkl_path}: per_event={len(pe)} pred={len(pred)} ref={len(ref)} "
            f"n_events={r['n_events']} do not line up"
        )
    n_pred = np.array([e["n_pred"] for e in pe])
    if not np.array_equal(n_pred, [len(p) for p in pred]):
        raise ValueError(f"{pkl_path}: n_pred disagrees with pred_pvs_mm lengths")
    has_amvf = pe[0].get("n_amvf") is not None
    return Sample(
        name=name,
        pkl_path=Path(pkl_path),
        pred=pred,
        ref_pkl=ref,
        mu=np.array([e["mu"] for e in pe], dtype=np.float64),
        beam_z=np.array([e.get("beam_z", 0.0) for e in pe], dtype=np.float64),
        event_idx=np.array([e["event_idx"] for e in pe], dtype=np.int64),
        n_pred=n_pred,
        n_ref=np.array([e["n_truth"] for e in pe]),
        n_amvf=np.array([e["n_amvf"] for e in pe]) if has_amvf else None,
        dz=np.asarray(r["pairwise_dz_mm"], dtype=np.float64),
        raw=r,
    )


# --------------------------------------------------------------------------
# sigma_vtx-vtx fit
# --------------------------------------------------------------------------


def sigmoid(x: np.ndarray, a: float, b: float, c: float, rcc: float) -> np.ndarray:
    """The dip model used by every PV-Finder eval script."""
    return a / (1.0 + np.exp(b * (rcc - np.abs(x)))) + c


@dataclass
class FitResult:
    sigma: float
    sigma_err: float
    popt: np.ndarray
    edges: np.ndarray
    centres: np.ndarray
    counts: np.ndarray


def dz_edges(n_bins: int = N_FIT_BINS) -> np.ndarray:
    """Pairwise-dz bin edges over +-6 mm."""
    return np.linspace(-FIT_RANGE_MM, FIT_RANGE_MM, n_bins + 1)


def fit_sigma(dz: np.ndarray, p0_mode: str, n_bins: int = N_FIT_BINS) -> FitResult:
    """Sigmoid fit of the pairwise-dz dip with the historical conventions.

    ``p0_mode`` is ``"adaptive"`` (collision-data scripts) or ``"fixed"`` (MC
    script, ``MC_FIT_P0``).  Raises if the fit does not converge: a silent
    default sigma would silently become a matching window.
    """
    edges = dz_edges(n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(dz, bins=edges)
    if p0_mode == "adaptive":
        base = float(np.median(counts))
        p0 = [max(base - float(counts.min()), 1.0), 10.0, max(base, 1.0), 0.5]
    elif p0_mode == "fixed":
        p0 = list(MC_FIT_P0)
    else:
        raise ValueError(f"unknown p0_mode {p0_mode!r}")
    popt, pcov = curve_fit(sigmoid, centres, counts.astype(float), p0=p0,
                           maxfev=10000, bounds=([0, 0, 0, 0], [np.inf] * 4))  # fmt: skip
    return FitResult(
        sigma=float(abs(popt[3])),
        sigma_err=float(np.sqrt(np.diag(pcov))[3]),
        popt=np.asarray(popt),
        edges=edges,
        centres=centres,
        counts=counts,
    )


# --------------------------------------------------------------------------
# MC ROOT vertices
# --------------------------------------------------------------------------


def _jagged_to_npz(arrs: Sequence[np.ndarray]) -> tuple:
    lens = np.array([len(a) for a in arrs], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    flat = np.concatenate([np.asarray(a, dtype=np.float64) for a in arrs])
    return flat, off


def load_mc_root(root_path: Path, cache_path: Path) -> dict[str, object]:
    """Read the MC vertex branches for all entries (cached as flat npz).

    Returns per-entry lists (float64) for the jagged branches and arrays for
    ``ActualNumOfInt`` / ``BeamPosZ``.
    """
    if not cache_path.exists():
        import uproot

        tree = uproot.open(str(root_path))["PVFinderData"]
        a = tree.arrays(list(MC_ROOT_BRANCHES), library="np")
        payload = {"n_entries": np.int64(tree.num_entries)}
        for b in MC_ROOT_BRANCHES[:4]:
            payload[b], payload[b + "_off"] = _jagged_to_npz(a[b])
        payload["ActualNumOfInt"] = np.asarray(a["ActualNumOfInt"], dtype=np.float64)
        payload["BeamPosZ"] = np.array(
            [float(np.asarray(v).ravel()[0]) for v in a["BeamPosZ"]]
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **payload)
    d = np.load(cache_path)
    out: dict[str, object] = {
        "n_entries": int(d["n_entries"]),
        "ActualNumOfInt": d["ActualNumOfInt"],
        "BeamPosZ": d["BeamPosZ"],
    }
    for b in MC_ROOT_BRANCHES[:4]:
        flat, off = d[b], d[b + "_off"]
        out[b] = [flat[off[i] : off[i + 1]] for i in range(len(off) - 1)]
    return out


def mc_vertices_for_events(root: dict[str, object], entries: Sequence[int]) -> dict:
    """Truth (nTrk >= 2) and AMVF (nTrk >= 2) vertices for the given ROOT entries.

    MC ``BeamPosZ`` is checked to be exactly zero, so the detector frame and the
    beam frame coincide and the data-style beam correction is the identity.
    """
    tz, tn = root["TruthVertex_z"], root["TruthVertex_nTracks"]
    rz, rn = root["RecoVertex_z"], root["RecoVertex_nTracks"]
    beam = np.asarray(root["BeamPosZ"])[list(entries)]
    return {
        "truth_all": [tz[e] for e in entries],
        "truth_ge2": [tz[e][tn[e] >= 2] for e in entries],
        "amvf_ge2": [rz[e][rn[e] >= 2] for e in entries],
        "mu": np.asarray(root["ActualNumOfInt"])[list(entries)],
        "beam_z": beam,
    }


def _same_set(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    if len(a) != len(b):
        return False
    return bool(np.all(np.abs(np.sort(a) - np.sort(b)) <= tol))


def verify_mc_mapping(
    mc: Sample, root: dict[str, object], qibin: Sequence[int], tol_mm: float = 1e-3
) -> dict:
    """Prove that h5 test event i is ROOT entry ``qibin[i]``, and not entry i.

    Checks per event: the full truth-z set (pkl ``truth_pvs_mm`` is the h5
    ``pv`` list, unfiltered) equals ROOT ``TruthVertex_z``; the stored
    ``n_truth`` equals ROOT nTrk >= 2 truth count; the stored ``n_amvf`` equals
    ROOT nTrk >= 2 reco count; the stored mu equals ``ActualNumOfInt``.  The same
    tests are run for the identity hypothesis (entry = h5 event index) as a
    control.
    """
    n = mc.n_events
    if len(qibin) < n:
        raise ValueError(f"qibin has {len(qibin)} entries, need {n}")

    def _score(entries: Sequence[int]) -> dict:
        v = mc_vertices_for_events(root, entries)
        tz_ok = [_same_set(v["truth_all"][i], mc.ref_pkl[i], tol_mm) for i in range(n)]
        nt_ok = [len(v["truth_ge2"][i]) == mc.n_ref[i] for i in range(n)]
        na_ok = [len(v["amvf_ge2"][i]) == mc.n_amvf[i] for i in range(n)]
        mu_ok = np.abs(v["mu"] - mc.mu) < 1e-6
        maxdiff = max(
            (float(np.max(np.abs(np.sort(v["truth_all"][i]) - np.sort(mc.ref_pkl[i]))))
             for i in range(n) if len(v["truth_all"][i]) == len(mc.ref_pkl[i])
             and len(mc.ref_pkl[i])), default=float("nan"))  # fmt: skip
        return {
            "truth_set_equal": int(np.sum(tz_ok)),
            "n_truth_ge2_equal": int(np.sum(nt_ok)),
            "n_amvf_ge2_equal": int(np.sum(na_ok)),
            "mu_equal": int(np.sum(mu_ok)),
            "max_abs_dz_mm_when_len_equal": maxdiff,
            "n_events": n,
        }

    q = [int(x) for x in qibin[:n]]
    ident = [int(e) for e in mc.event_idx]
    return {
        "qibin_is_permutation_not_identity": q != ident,
        "n_distinct_entries": len(set(q)),
        "qibin_mapping": _score(q),
        "identity_mapping_control": _score(ident),
    }
