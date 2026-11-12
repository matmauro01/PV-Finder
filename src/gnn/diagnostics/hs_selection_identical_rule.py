"""HS vertex selection with one identical rule for AMVF and PV-Finder+GNN.

The rule is Athena 24.0's (``gnn.evaluation.pv_selection``):
score(v) = sum of pT^2 over the tracks assigned to v and compatible with it
(chi2 <= 18.42), the selected vertex = argmax, and the HS match is
``bestHardScatterMatch`` with ties to the first vertex of the score-sorted
container. The same function scores both algorithms. Only the inputs differ:
- AMVF: its stored track lists and its fitted vertex position.
- PV-Finder+GNN: the GNN-assigned tracks and the PV-Finder vertex position.
  The position is either the histogram peak z on the beam spot, or the 3D
  adaptive fit of the assigned tracks.

Per event and arm the output gives has_match, sel_is_match, sel_near and
reco_sel (= sel_is_match and |z_sel - z_HS| < 0.1 mm, ITk Fig. 27), plus
diagnostics on the plain sum(pT^2) leader.

Inputs: the same replay as ``hs_chain_study`` (r16443 entries 0-24,999, saved
histograms and top-1 GNN edges). No inference.

Run: python -u -m gnn.diagnostics.hs_selection_identical_rule --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from gnn.diagnostics import hs_chain_study as chain
from gnn.evaluation import hs_metrics as hm
from gnn.evaluation import pv_selection as ps
from pv_finder.evaluation.vertex_finding.hs_position_estimators import (
    adaptive_beamline_fit,
)
from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast
from pv_finder.utils.working_points import HS_MATCH_WINDOW_MM

HS_WINDOW = HS_MATCH_WINDOW_MM  # 0.1 mm (working_points)
GNN_ARMS = (("v4", 0.98), ("v4", 0.95), ("v4", 0.0), ("fullmu", 0.98), ("fullmu", 0.0))
# rule variants: plain (no compatibility), beam (peak z on the beam spot), fit (3D fit)
ARMS = (
    "amvf_stored", "amvf_rule", "amvf_plain",
    *(f"gnn_{c}_t{round(t * 100):03d}_{v}" for c, t in GNN_ARMS for v in ("rule_fit", "rule_beam", "plain")),
)  # fmt: skip
FIELDS = (
    "has_match",
    "sel_is_match",
    "sel_near",
    "reco_sel",
    "lead_incompat_pt_max_mev",
)
BRANCHES = [
    "ActualNumOfInt", "BeamPosX", "BeamPosY", "BeamPosSigmaX",
    "RecoTrack_pT", "RecoTrack_d0", "RecoTrack_z0", "RecoTrack_phi", "RecoTrack_theta",
    "RecoTrack_ErrD0", "RecoTrack_ErrZ0", "RecoTrack_ErrD0Z0",
    "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type",
    "RecoVertex_x", "RecoVertex_y", "RecoVertex_z", "RecoVertex_type", "RecoVertex_assocTracks",
]  # fmt: skip


class Tracks:
    """Per-event track arrays (mm, rad, MeV)."""

    def __init__(self, ev: ak.Record) -> None:
        g = lambda k: ak.to_numpy(ev[k]).astype(np.float64)  # noqa: E731
        self.pt = g("RecoTrack_pT")
        self.d0, self.z0 = g("RecoTrack_d0"), g("RecoTrack_z0")
        self.phi, self.theta = g("RecoTrack_phi"), g("RecoTrack_theta")
        self.var_d0, self.var_z0 = g("RecoTrack_ErrD0") ** 2, g("RecoTrack_ErrZ0") ** 2
        self.cov = g("RecoTrack_ErrD0Z0")

    def chi2(self, idx: np.ndarray, vertex: tuple[float, float, float]) -> np.ndarray:
        """Track-vertex compatibility chi2 for the tracks ``idx``."""
        return ps.track_vertex_chi2(
            self.d0[idx], self.z0[idx], self.phi[idx], self.theta[idx],
            self.var_d0[idx], self.var_z0[idx], self.cov[idx], vertex,
        )  # fmt: skip

    def fit(
        self, idx: np.ndarray, seed_z: float, beam: tuple[float, float], sig: float
    ):  # noqa: ANN201
        """3D adaptive vertex position (x, y, z) of the tracks ``idx``."""
        return adaptive_beamline_fit(
            self.d0[idx], self.z0[idx], self.phi[idx], self.theta[idx],
            self.var_d0[idx], self.var_z0[idx], self.cov[idx], seed_z,
            beam_xy=beam, beam_sigma_xy=sig,
        )  # fmt: skip


def rule_scores(
    trk: Tracks,
    lists: list[np.ndarray],
    position,  # noqa: ANN001 - callable v -> (x, y, z)
    candidates: np.ndarray | None = None,
) -> np.ndarray:
    """Athena score (compatible sum pT^2, GeV^2) per vertex; NaN where not computed."""
    out = np.full(len(lists), np.nan)
    todo = range(len(lists)) if candidates is None else candidates
    for v in todo:
        idx = lists[v]
        if len(idx) == 0:
            out[v] = 0.0
            continue
        ok = trk.chi2(idx, position(v)) <= ps.MAX_VERTEX_CHI2
        out[v] = ps.sumpt2_score(trk.pt[idx] / 1000.0, ok)
    return out


def lazy_rule_selection(
    trk: Tracks,
    lists: list[np.ndarray],
    plain: np.ndarray,
    position,
    nhs: np.ndarray,  # noqa: ANN001
) -> tuple[int, int]:
    """(selected, HS match) under the rule, computing positions only where needed.

    The compatible score never exceeds the plain one, so vertices are scored
    in descending plain order until no remaining vertex can win. Vertices
    tied for the most HS tracks are also scored, because their container order
    decides the HS match.
    """
    scores = np.full(len(lists), np.nan)
    best = -1.0
    for v in np.argsort(-plain, kind="stable"):
        if plain[v] < best:
            break
        scores[v] = rule_scores(trk, lists, position, np.array([v]))[v]
        best = max(best, scores[v])
    tied = (
        np.flatnonzero(nhs == nhs.max())
        if len(nhs) and nhs.max() > 0
        else np.zeros(0, int)
    )
    missing = [v for v in tied if np.isnan(scores[v])]
    if missing:
        scores[missing] = rule_scores(trk, lists, position, np.array(missing))[missing]
    # vertices never scored cannot win or tie: rank them after all scored ones
    filled = np.where(np.isnan(scores), -1.0, scores)
    return ps.select_primary(filled), ps.best_hs_match(
        nhs, ps.sorted_container_order(filled)
    )


def arm_record(
    trk: Tracks,
    lists: list[np.ndarray],
    sel: int,
    match: int,
    vz: np.ndarray,
    hz: float,
    plain_lead: int,
    position,  # noqa: ANN001
) -> list[int]:
    """FIELDS for one arm."""
    near = sel >= 0 and abs(vz[sel] - hz) < HS_WINDOW
    pmax = 0
    if plain_lead >= 0 and len(lists[plain_lead]):
        idx = lists[plain_lead]
        bad = trk.chi2(idx, position(plain_lead)) > ps.MAX_VERTEX_CHI2
        pmax = int(trk.pt[idx][bad].max()) if bad.any() else 0
    return [
        int(match >= 0),
        int(match >= 0 and sel == match),
        int(near),
        int(match >= 0 and sel == match and near),
        pmax,
    ]


def process_chunk(start: int, stop: int) -> dict:
    """Evaluate r16443 entries [start, stop)."""
    data = uproot.open(chain.ROOT)["PVFinderData"].arrays(
        BRANCHES, entry_start=start, entry_stop=stop, library="ak"
    )
    with np.load(chain.HIST_DIR / f"chunk_{start:05d}_{start + 500:05d}.npz") as f:
        hists = f["histograms"]
        np.testing.assert_array_equal(
            f["event_idx"][: stop - start], np.arange(start, stop)
        )
    edge_files = {k: chain._load_edges(d, start) for k, d in chain.EDGES.items()}
    n = stop - start
    out = np.zeros((n, len(ARMS), len(FIELDS)), dtype=np.int64)
    mu = np.zeros(n)
    for loc in range(n):
        ev = data[loc]
        trk = Tracks(ev)
        n_trk = len(trk.pt)
        mu[loc] = float(ev["ActualNumOfInt"])
        beam = (float(ev["BeamPosX"]), float(ev["BeamPosY"]))
        sig = float(ev["BeamPosSigmaX"])
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        truth = hm.track_truth_labels(
            ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, n_trk
        )
        h = int(np.flatnonzero(ak.to_numpy(ev["TruthVertex_type"]) == 1)[0])
        hz = float(ak.to_numpy(ev["TruthVertex_z"])[h])
        rec = []
        # ---- AMVF: stored lists, stored (fitted) positions, populated vertices
        alists = [
            np.unique(ak.to_numpy(x).astype(np.int64))
            for x in ev["RecoVertex_assocTracks"]
        ]
        axyz = np.stack(
            [
                ak.to_numpy(ev[k]).astype(np.float64)
                for k in ("RecoVertex_x", "RecoVertex_y", "RecoVertex_z")
            ],
            1,
        )
        native = int(np.flatnonzero(ak.to_numpy(ev["RecoVertex_type"]) == 1)[0])
        keep = np.flatnonzero([len(x) > 0 for x in alists])
        native = int(np.flatnonzero(keep == native)[0])
        alists = [alists[i] for i in keep]
        axyz = axyz[keep]
        apos = lambda v: tuple(axyz[v])  # noqa: E731
        acls = hm.classify_vertices(alists, trk.pt, truth)
        anhs = hm.hs_track_counts(alists, acls, truth, h)
        aplain = np.array([ps.sumpt2_score(trk.pt[x] / 1000.0) for x in alists])
        stored_match = ps.best_hs_match(anhs, np.arange(len(alists)))
        plain_lead = ps.select_primary(aplain)
        rec.append(
            arm_record(
                trk, alists, native, stored_match, axyz[:, 2], hz, plain_lead, apos
            )
        )
        sel, match = lazy_rule_selection(trk, alists, aplain, apos, anhs)
        rec.append(
            arm_record(trk, alists, sel, match, axyz[:, 2], hz, plain_lead, apos)
        )
        rec.append(
            arm_record(
                trk,
                alists,
                plain_lead,
                ps.best_hs_match(anhs, ps.sorted_container_order(aplain)),
                axyz[:, 2],
                hz,
                plain_lead,
                apos,
            )
        )
        # ---- PV-Finder peaks + GNN lists
        peaks = pv_locations_updated_res_fast(hists[loc], *chain.PEAK_ARGS)[0].astype(
            np.float64
        )
        for ckpt, t in GNN_ARMS:
            off, sc, tg = edge_files[ckpt]
            lo, hi = off[loc], off[loc + 1]
            if hi - lo != n_trk:
                raise ValueError(f"entry {start + loc}: edge count mismatch ({ckpt})")
            tgt = hm.assign_from_scores(tg[lo:hi], sc[lo:hi], t)
            lists_all = hm.lists_from_targets(tgt, len(peaks))
            pk = np.flatnonzero([len(x) > 0 for x in lists_all])
            lists = [lists_all[i] for i in pk]
            pz = peaks[pk]
            cls = hm.classify_vertices(lists, trk.pt, truth)
            nhs = hm.hs_track_counts(lists, cls, truth, h)
            plain = np.array([ps.sumpt2_score(trk.pt[x] / 1000.0) for x in lists])
            plead = ps.select_primary(plain)
            fits: dict[int, tuple[float, float, float]] = {}

            def fpos(v: int) -> tuple[float, float, float]:
                if v not in fits:
                    fits[v] = trk.fit(lists[v], pz[v], beam, sig)
                return fits[v]

            bpos = lambda v: (beam[0], beam[1], pz[v])  # noqa: E731
            if len(lists) == 0:
                rec += [[0, 0, 0, 0, 0]] * 3
                continue
            s_f, m_f = lazy_rule_selection(trk, lists, plain, fpos, nhs)
            # the reported position of the selected vertex is its fitted z
            fz = pz.copy()
            if s_f >= 0:
                fz[s_f] = fpos(s_f)[2]
            rec.append(arm_record(trk, lists, s_f, m_f, fz, hz, plead, fpos))
            s_b, m_b = lazy_rule_selection(trk, lists, plain, bpos, nhs)
            rec.append(arm_record(trk, lists, s_b, m_b, pz, hz, plead, bpos))
            m_p = ps.best_hs_match(nhs, ps.sorted_container_order(plain))
            rec.append(arm_record(trk, lists, plead, m_p, pz, hz, plead, bpos))
        out[loc] = np.array(rec, dtype=np.int64)
    return {"start": start, "out": out, "mu": mu}


def summarize(
    out: np.ndarray, mu: np.ndarray, n_boot: int = 2000, seed: int = 20261111
) -> dict:
    """reco_sel fractions (PU200 and mu bins) and paired differences vs AMVF."""
    rs = out[:, :, FIELDS.index("reco_sel")].astype(np.float64)
    pu = (mu >= 185) & (mu <= 215)
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(pu)
    boots = rng.integers(0, len(idx), size=(n_boot, len(idx)))
    res: dict = {"n_pu200": int(pu.sum()), "arms": {}}
    for a, arm in enumerate(ARMS):
        x = rs[idx, a]
        d_st = x - rs[idx, ARMS.index("amvf_stored")]
        d_ru = x - rs[idx, ARMS.index("amvf_rule")]
        bins = {}
        for lo in range(0, 200, 20):
            m = (mu >= lo) & (mu < lo + 20)
            bins[f"{lo}-{lo + 20}"] = [int(rs[m, a].sum()), int(m.sum())]
        res["arms"][arm] = {
            "k": int(x.sum()), "frac": float(x.mean()), "err": float(x[boots].mean(1).std()),
            "minus_amvf_stored": [float(d_st.mean()), float(d_st[boots].mean(1).std())],
            "minus_amvf_rule": [float(d_ru.mean()), float(d_ru[boots].mean(1).std())],
            "has_match": float(out[idx, a, 0].mean()), "sel_is_match": float(out[idx, a, 1].mean()),
            "sel_near": float(out[idx, a, 2].mean()), "mu_bins_k_n": bins,
        }  # fmt: skip
    # AMVF rule vs Athena's stored primary: selection agreement needs per-event
    # selected indices, which arm_record does not keep; reco_sel agreement:
    st, ru = rs[:, ARMS.index("amvf_stored")], rs[:, ARMS.index("amvf_rule")]
    res["amvf_rule_vs_stored_reco_sel_agree"] = {
        "pu200": float((st[pu] == ru[pu]).mean()),
        "all": float((st == ru).mean()),
    }
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-events", type=int, default=25000)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, a.n_events, 500))
    with ProcessPoolExecutor(a.workers) as pool:
        parts = sorted(
            pool.map(process_chunk, starts, [min(s + 500, a.n_events) for s in starts]),
            key=lambda d: d["start"],
        )
    out = np.concatenate([d["out"] for d in parts])
    mu = np.concatenate([d["mu"] for d in parts])
    np.savez_compressed(a.out / "hs_selection_events.npz", out=out, mu=mu)
    summ = summarize(out, mu)
    summ["meta"] = {
        "arms": ARMS, "fields": FIELDS, "root": str(chain.ROOT), "edges": {k: str(v) for k, v in chain.EDGES.items()},
        "checkpoints": chain.CHECKPOINTS, "peak_args": chain.PEAK_ARGS, "max_vertex_chi2": ps.MAX_VERTEX_CHI2,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rule_sha256": hashlib.sha256(Path(ps.__file__).read_bytes()).hexdigest(),
    }  # fmt: skip
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    for arm in ARMS:
        r = summ["arms"][arm]
        print(f"{arm:28s} {r['k']:5d}/{summ['n_pu200']} = {100 * r['frac']:6.2f} +- {100 * r['err']:.2f}  "
              f"vs stored {100 * r['minus_amvf_stored'][0]:+6.2f}  vs rule {100 * r['minus_amvf_rule'][0]:+6.2f}")  # fmt: skip


if __name__ == "__main__":
    main()
