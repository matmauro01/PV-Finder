"""Load the saved evaluation outputs behind the paper figures.

Nothing here runs a network or a peak finder. Each loader reads stored peak,
truth and AMVF vertex lists, checks that they are aligned event by event, and
returns them in one common layout (:class:`Sample`). Paths are fixed so that a
figure can always be traced back to one file; see
``docs/evaluation/paper_figures.md``.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]

# HL-LHC ITk, flat-mu held-out reconstruction r16443 (never used in training).
HLLHC_PKL = REPO / "outputs/08_05_2026_output/eval_v6_mu_window/r16443/eval_results.pkl"
HLLHC_CACHE = REPO / "outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz"
# Fixed truth-matching window of the main comparison: 3 x the larger 68%
# position error (56.5 um, AMVF), technical note Section 4.1.
HLLHC_WINDOW_MM = 0.1694
HLLHC_MU_RANGE = (185.0, 215.0)

# Low pile-up 13 TeV ttbar simulation used for the Run 2 reproduction model.
LOWPU_MC_PKL = REPO / "outputs/04_16_2026_output/reproduction_ep300_mc/eval_results.pkl"
LOWPU_MC_ROOT = REPO / "data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root"
LOWPU_MC_INDEX = REPO / "configs/qibin_test_main_indices_v2.p"
LOWPU_MU_RANGE = (55.0, 65.0)

# Run 3 collision data, AMVF (nTracks >= 2, beam-spot corrected) as reference.
RUN3_DATA_PKL = (
    REPO / "outputs/04_16_2026_output/reproduction_ep300_run3/eval_results.pkl"
)


@dataclass
class Resolution:
    """Pairwise-dz sample and sigmoid fit for one algorithm."""

    dz_mm: np.ndarray
    n_bins: int
    params: np.ndarray  # (a, b, c, sigma) of a / (1 + exp(b (sigma - |dz|))) + c
    sigma_mm: float
    sigma_err_mm: float
    note: str = ""


@dataclass
class Sample:
    """Per-event vertex lists for one evaluation sample.

    ``truth_z`` is MC truth with nTracks >= 2 on simulation and AMVF on data.
    """

    name: str
    mu: np.ndarray
    pvf_z: list[np.ndarray]
    truth_z: list[np.ndarray]
    amvf_z: list[np.ndarray] | None
    window_mm: float
    is_data: bool
    resolution: dict[str, Resolution] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def sigmoid(x: np.ndarray, a: float, b: float, c: float, sigma: float) -> np.ndarray:
    """Pairwise-dz fit function, identical to the production evaluation."""
    return a / (1.0 + np.exp(b * (sigma - np.abs(x)))) + c


def pairwise_dz(z_by_event: list[np.ndarray], seed: int) -> np.ndarray:
    """Signed differences over all vertex pairs, with the eval's random sign.

    The evaluation shuffles each event's positions before taking z_i - z_j for
    i < j, so the sign of each pair is random. A fixed seed keeps it repeatable.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for z in z_by_event:
        if len(z) < 2:
            continue
        zs = np.asarray(z, dtype=np.float64).copy()
        rng.shuffle(zs)
        iu = np.triu_indices(len(zs), k=1)
        chunks.append((zs[:, None] - zs[None, :])[iu])
    return np.concatenate(chunks) if chunks else np.zeros(0)


def fit_resolution(
    dz: np.ndarray, n_bins: int, p0: list[float] | None = None, note: str = ""
) -> Resolution:
    """Fit the sigmoid to dz histogrammed on [-6, 6] mm, as the evaluation does.

    ``p0=None`` uses the data-driven start of ``run_eval_pvf_run3.py``; the
    low-pile-up evaluation used the fixed start ``[1000, 10, 30, 0.8]``.
    """
    from scipy.optimize import curve_fit

    edges = np.linspace(-6.0, 6.0, n_bins + 1)
    ctrs = 0.5 * (edges[:-1] + edges[1:])
    cnts, _ = np.histogram(dz, bins=edges)
    if p0 is None:
        base = float(np.median(cnts))
        p0 = [max(base - float(cnts.min()), 1.0), 10.0, max(base, 1.0), 0.5]
    popt, pcov = curve_fit(
        sigmoid, ctrs, cnts.astype(float), p0=p0, maxfev=20000,
        bounds=([0, 0, 0, 0], [np.inf] * 4),
    )  # fmt: skip
    return Resolution(
        dz_mm=dz,
        n_bins=n_bins,
        params=np.asarray(popt),
        sigma_mm=float(abs(popt[3])),
        sigma_err_mm=float(np.sqrt(pcov[3, 3])),
        note=note,
    )


def _load_pkl(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _unflatten(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def load_hllhc(with_amvf_fit: bool = True) -> tuple[Sample, Sample]:
    """HL-LHC r16443: (all 25 000 events, PVF only) and (1 920 PU200 events).

    The first sample feeds the yield-vs-mu panel; the second carries AMVF and
    both resolution fits. PVF peaks use the production 0.30 integral cut; the
    PVF resolution uses the 0.50 cut, as in the note (Section 4.2).
    """
    res = _load_pkl(HLLHC_PKL)
    events = res["per_event"]
    mu = np.array([e["mu"] for e in events], dtype=np.float64)
    pvf = [np.asarray(z) for z in res["pred_pvs_mm"]]
    truth = [np.asarray(z) for z in res["truth_pvs_mm"]]
    full = Sample("hllhc_all", mu, pvf, truth, None, HLLHC_WINDOW_MM, False,
                  meta={"source": str(HLLHC_PKL.relative_to(REPO))})  # fmt: skip

    cache = np.load(HLLHC_CACHE)
    idx = cache["event_idx"]
    lo, hi = HLLHC_MU_RANGE
    in_window = np.array([lo <= round(m) <= hi for m in mu])
    # Alignment checks: the cache must be exactly the mu-window events of the
    # pkl, with identical PVF peaks and truth lists.
    if not np.array_equal(in_window, np.asarray(res["in_mu_window"], dtype=bool)):
        raise ValueError("mu-window selection differs from the stored one")
    if not np.array_equal(np.flatnonzero(in_window), idx):
        raise ValueError("cache event order does not match the pkl mu window")
    c_pvf = _unflatten(cache["pvf_z"], cache["pvf_off"])
    c_truth = _unflatten(cache["truth_ge2_z"], cache["truth_ge2_off"])
    for k, i in enumerate(idx):
        if not (np.allclose(c_pvf[k], pvf[i], atol=1e-5, rtol=0)
                and np.allclose(c_truth[k], truth[i], atol=1e-5, rtol=0)):  # fmt: skip
            raise ValueError(f"cache and pkl disagree for event {i}")
    if not np.allclose(cache["mu"], mu[idx]):
        raise ValueError("cache mu differs from pkl mu")
    amvf = _unflatten(cache["amvf_z"], cache["amvf_off"])
    win = Sample(
        "hllhc_pu200", mu[idx], c_pvf, c_truth, amvf, HLLHC_WINDOW_MM, False,
        meta={"source": str(HLLHC_PKL.relative_to(REPO)),
              "amvf_source": str(HLLHC_CACHE.relative_to(REPO))},
    )  # fmt: skip
    win.resolution["pvf"] = Resolution(
        dz_mm=np.asarray(res["pairwise_dz_mm"]),
        n_bins=300,
        params=np.asarray(res["fit_params"]),
        sigma_mm=float(res["sigma_vtx_vtx_mm"]),
        sigma_err_mm=float(res["sigma_vtx_vtx_err_mm"]),
        note="stored fit, integral threshold 0.50, 0.04 mm bins",
    )
    if with_amvf_fit:
        # Same procedure as amvf_fair_comparison.scan.fit_sigma_vtx_vtx
        # (seed 20260805), which gave 0.3047 mm.
        dz = pairwise_dz(amvf, seed=20260805)
        win.resolution["amvf"] = fit_resolution(
            dz[np.abs(dz) <= 6.0], 300, note="refit, 0.04 mm bins"
        )
    return full, win


def _lowpu_root_lists(order: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Truth (nTracks >= 2) and AMVF (nTracks >= 2) z for the given ROOT entries."""
    import uproot

    branches = ["TruthVertex_z", "TruthVertex_nTracks", "RecoVertex_z",
                "RecoVertex_nTracks"]  # fmt: skip
    tree = uproot.open(LOWPU_MC_ROOT)["PVFinderData"]
    arr = tree.arrays(branches, library="np")
    truth = [np.asarray(arr["TruthVertex_z"][r])[arr["TruthVertex_nTracks"][r] >= 2]
             for r in order]  # fmt: skip
    amvf = [np.asarray(arr["RecoVertex_z"][r])[arr["RecoVertex_nTracks"][r] >= 2]
            for r in order]  # fmt: skip
    return truth, amvf


def load_lowpu_mc() -> Sample:
    """13 TeV ttbar simulation, 2 550 test events, Run 2 reproduction model.

    The stored pkl keeps unfiltered HDF5 truth, so truth (nTracks >= 2) and AMVF
    lists are re-read from the ROOT file through the evaluation's index map and
    checked against the stored per-event counts.
    """
    res = _load_pkl(LOWPU_MC_PKL)
    events = res["per_event"]
    with LOWPU_MC_INDEX.open("rb") as handle:
        order = [int(r) for r in pickle.load(handle)]
    if len(order) != len(events):
        raise ValueError("index map and pkl have different lengths")
    truth, amvf = _lowpu_root_lists(order)
    for k, e in enumerate(events):
        if len(truth[k]) != e["n_truth"] or len(amvf[k]) != e["n_amvf"]:
            raise ValueError(f"ROOT lists do not match stored counts in event {k}")
    mu = np.array([e["mu"] for e in events], dtype=np.float64)
    pvf = [np.asarray(z) for z in res["pred_pvs_mm"]]
    s = Sample("lowpu_mc", mu, pvf, truth, amvf, float(res["sigma_vtx_vtx_mm"]), False,
               meta={"source": str(LOWPU_MC_PKL.relative_to(REPO)),
                     "root": str(LOWPU_MC_ROOT.relative_to(REPO))})  # fmt: skip
    p0 = [1000.0, 10.0, 30.0, 0.8]  # FIT_P0 of run_eval_pvf.py
    stored = fit_resolution(np.asarray(res["pairwise_dz_mm"]), 60, p0,
                            note="refit of stored dz; 0.2 mm bins")  # fmt: skip
    if abs(stored.sigma_mm - res["sigma_vtx_vtx_mm"]) > 1e-6:
        raise ValueError("refit does not reproduce the stored PVF sigma")
    s.resolution["pvf"] = stored
    s.resolution["amvf"] = fit_resolution(pairwise_dz(amvf, seed=20261109), 60, p0,
                                          note="same events, 0.2 mm bins")  # fmt: skip
    return s


def load_run3_data() -> Sample:
    """Run 3 collision data, 2 500 events; AMVF is the only reference."""
    res = _load_pkl(RUN3_DATA_PKL)
    events = res["per_event"]
    mu = np.array([e["mu"] for e in events], dtype=np.float64)
    pvf = [np.asarray(z) for z in res["pred_pvs_mm"]]
    amvf = [np.asarray(z) for z in res["truth_pvs_mm"]]
    s = Sample("run3_data", mu, pvf, amvf, amvf, float(res["sigma_vtx_vtx_mm"]), True,
               meta={"source": str(RUN3_DATA_PKL.relative_to(REPO))})  # fmt: skip
    stored = fit_resolution(np.asarray(res["pairwise_dz_mm"]), 60,
                            list(np.asarray(res["fit_params"])),
                            note="refit of stored dz; 0.2 mm bins")  # fmt: skip
    if abs(stored.sigma_mm - res["sigma_vtx_vtx_mm"]) > 1e-4:
        raise ValueError("refit does not reproduce the stored PVF sigma")
    s.resolution["pvf"] = stored
    s.resolution["amvf"] = fit_resolution(pairwise_dz(amvf, seed=20261109), 60,
                                          [1000.0, 10.0, 30.0, 0.8],
                                          note="same events, 0.2 mm bins")  # fmt: skip
    return s
