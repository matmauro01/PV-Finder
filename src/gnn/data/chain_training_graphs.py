"""TTVA training graphs on PV-Finder candidates with truth labels (GNN v3).

The v1/v2 networks were trained on truth-vertex candidates and deployed on
PV-Finder peaks. Here the training graph is the deployment graph itself: the
density model of ``pv_finder.utils.working_points`` (by name, peak cuts read at
run time, optionally overridden) finds the peaks of each event and the kNN
inference graph is built exactly as ``gnn.evaluation.chain_inference`` builds
it (``create_inference_graph``, k = 20, float32 track features, pT / PT_SCALE).

Labels come from a one-to-one truth matching of the candidates:
truth vertices with >= 2 tracks, in descending track count (index on ties),
each take the nearest still-free candidate within the regime window
(0.17 mm). An edge is positive iff its candidate is the one owned by the
track's truth vertex. Tracks of unmatched truth vertices (finder misses, or
vertices merged into a candidate owned by a larger vertex), of 1-track truth
vertices and unlabelled tracks have no positive edge: the network is taught to
leave them unassigned. The track-count priority keeps the HS (and any large
vertex) from losing its peak to a nearby small pile-up vertex.

Each graph also stores ``track.truth_pv`` (truth index, -1 none),
``pv.owner_truth`` (matched truth index, -1 none), ``pv.y_hs``,
``pv.sigma`` (peak width used in the edge significance), ``graph.mu`` and
``graph.entry``.

Optional pile-up thinning (``--thin-mu-max``): before anything else, each
event keeps a random subset of its pile-up truth vertices (and their tracks)
so that its pile-up count is drawn uniformly in [0, mu_actual]; unlabelled
tracks are kept with the same fraction. The HS vertex is always kept. This
turns PU200 events into extra flat-mu events.

Usage (repo root, inside tmux):
    PYTHONPATH=src python -u -m gnn.data.chain_training_graphs \\
        --root data/.../ATLAS_PVFinderData_601229_e8481_s4494_r16638_PU200.root \\
        --entries data/.../r16638_train_shard_00.entry_indices.npy \\
        --out data/.../chain_s8_v3/r16638_train_shard_00.pt -d 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from gnn.data.event_selection import iterate_entries
from gnn.data.graph_construction import create_inference_graph
from pv_finder.data.run3_io import Run3Event
from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import (
    build_subevent_inputs,
    run_inference,
)
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast
from pv_finder.utils.working_points import HLLHC, S8, V6, DensityModel, PeakFinding

DENSITY: dict[str, DensityModel] = {"v6": V6, "s8": S8}
FEATURES = (
    "RecoTrack_d0", "RecoTrack_z0", "RecoTrack_ErrD0", "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0", "RecoTrack_theta", "RecoTrack_phi", "RecoTrack_pT",
)  # fmt: skip
TRUTH = ("TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type")  # fmt: skip
KNN = 20
COVERAGE_K = (5, 10, 15, 20)


def priority_match(
    cand_z: np.ndarray, truth_z: np.ndarray, truth_n: np.ndarray, window: float
) -> np.ndarray:
    """Candidate owned by each truth vertex (-1 if none), see module docstring."""
    owner = np.full(len(truth_z), -1, dtype=np.int64)
    if len(cand_z) == 0:
        return owner
    free = np.ones(len(cand_z), bool)
    for v in np.lexsort((np.arange(len(truth_z)), -truth_n)):
        if truth_n[v] < 2:
            break
        d = np.where(free, np.abs(cand_z - truth_z[v]), np.inf)
        k = int(np.argmin(d))
        if d[k] < window:
            owner[v] = k
            free[k] = False
    return owner


def thin_event(
    feats: list[np.ndarray], tz: np.ndarray, tn: np.ndarray, assoc: list[np.ndarray],
    ttype: np.ndarray, mu: float, rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, float]:  # fmt: skip
    """Keep a random subset of pile-up vertices so the pile-up count is U[0, mu]."""
    pu = np.flatnonzero(ttype != 1)
    target = rng.uniform(0.0, mu)
    frac = target / mu if mu > 0 else 1.0
    keep_v = np.ones(len(tz), bool)
    keep_v[pu] = rng.random(len(pu)) < frac
    n = len(feats[0])
    owned = np.zeros(n, bool)
    for x in assoc:
        owned[x] = True
    keep_t = np.zeros(n, bool)
    for v in np.flatnonzero(keep_v):
        keep_t[assoc[v]] = True
    keep_t |= ~owned & (rng.random(n) < frac)
    new_index = np.cumsum(keep_t) - 1
    feats = [f[keep_t] for f in feats]
    vs = np.flatnonzero(keep_v)
    assoc = [new_index[assoc[v]] for v in vs]
    return feats, tz[vs], tn[vs], assoc, ttype[vs], target


def build_event(
    ev: ak.Record, pvf: torch.nn.Module, peaks_cfg: PeakFinding, device: torch.device,
    window: float, rng: np.random.Generator | None = None,
):  # fmt: skip  # noqa: ANN201
    """One labelled candidate graph and its kNN-coverage counts."""
    f = [ak.to_numpy(ev[k]).astype(np.float32) for k in FEATURES]
    tz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
    tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
    assoc = [ak.to_numpy(x).astype(np.int64) for x in ev["TruthVertex_assocTracks"]]
    ttype = ak.to_numpy(ev["TruthVertex_type"]).astype(np.int64)
    mu = float(ev["ActualNumOfInt"])
    if rng is not None:
        f, tz, tn, assoc, ttype, mu = thin_event(f, tz, tn, assoc, ttype, mu, rng)
        tn = np.array([len(x) for x in assoc], dtype=np.int64)
    d0, z0, ed0, ez0, cov, theta, phi, pt = f
    n = len(z0)
    truth = np.full(n, -1, dtype=np.int64)
    for v, x in enumerate(assoc):
        truth[x] = v
    r3 = Run3Event(
        z0=z0, d0=d0, d0_err=ed0, z0_err=ez0, d0_z0_cov=cov,
        amvf_z=np.array([]), amvf_ntrks=np.array([]), beam_z=0.0,
        mu=None, event_idx=0, n_tracks=n,
    )  # fmt: skip
    hist = run_inference(build_subevent_inputs(r3), device, e2e=pvf)
    p = peaks_cfg
    pz, ph, _, ps = pv_locations_updated_res_fast(
        hist, p.amplitude, p.integral, p.min_width, p.min_height, p.centroid_halfwidth
    )  # fmt: skip
    pz = np.asarray(pz, dtype=np.float64)
    stack = np.stack(f, axis=1)
    stack[:, 7] = pt / PT_SCALE
    g = create_inference_graph(z0, d0, ez0, ed0, stack, pz, ph, ps, knn=KNN)
    owner = priority_match(pz, tz, tn, window)
    track_target = np.where(truth >= 0, owner[np.maximum(truth, 0)], -1)
    ei = g[("track", "to", "pv")].edge_index.numpy()
    y = (ei[1] == track_target[ei[0]]) & (track_target[ei[0]] >= 0)
    g[("track", "to", "pv")].y = torch.from_numpy(y.astype(np.float32))
    owner_of = np.full(len(pz), -1, dtype=np.int64)
    owner_of[owner[owner >= 0]] = np.flatnonzero(owner >= 0)
    hs = np.flatnonzero(ttype == 1)
    g["track"].truth_pv = torch.from_numpy(truth)
    g["pv"].owner_truth = torch.from_numpy(owner_of)
    g["pv"].y_hs = torch.from_numpy(
        (len(hs) == 1) & (owner_of == (hs[0] if len(hs) else -9))
    ).float()
    g["pv"].sigma = torch.from_numpy(np.asarray(ps, dtype=np.float32))
    g.mu = torch.tensor([mu], dtype=torch.float32)
    # kNN coverage: rank of the owned candidate among each track's nearest peaks
    has_t = track_target >= 0
    cov_counts = np.zeros(len(COVERAGE_K) + 1, dtype=np.int64)
    cov_counts[-1] = int(has_t.sum())
    if has_t.any() and len(pz):
        dz = np.abs(z0[has_t, None].astype(np.float64) - pz[None, :])
        own = dz[np.arange(has_t.sum()), track_target[has_t]]
        rank = (dz < own[:, None]).sum(1)  # 0 = nearest
        for j, k in enumerate(COVERAGE_K):
            cov_counts[j] = int((rank < k).sum())
    n_pos_tracks = int(np.unique(ei[0][y]).size)
    return g, cov_counts, n_pos_tracks, len(pz), int((owner >= 0).sum()), int((tn >= 2).sum())  # fmt: skip


def main() -> None:
    """CLI entry point."""
    from gnn.data.pu200_chain_graphs import build_pvf_model

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--entries", required=True, type=Path, help=".npy entry list")
    ap.add_argument("--out", required=True, type=Path, help="output .pt graph list")
    ap.add_argument("--density", default="s8", choices=sorted(DENSITY))
    ap.add_argument("--cuts", default=None, help="amplitude,width,integral,height,centroid override")  # fmt: skip
    ap.add_argument(
        "--thin-seed", type=int, default=None, help="enable pile-up thinning"
    )
    ap.add_argument("-d", "--device", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{a.device}")
    dm = DENSITY[a.density]
    peaks_cfg = dm.peaks
    if a.cuts:
        v = [float(x) for x in a.cuts.split(",")]
        peaks_cfg = replace(dm.peaks, amplitude=v[0], min_width=int(v[1]),
                            integral=v[2], min_height=v[3], centroid_halfwidth=int(v[4]))  # fmt: skip
    entries = np.load(a.entries).astype(np.int64)
    if np.any(np.diff(entries) <= 0):
        raise ValueError("entries must be strictly increasing")
    repo = Path(__file__).resolve().parents[3]
    pvf = build_pvf_model(str(repo / dm.checkpoint), device, dm.unet_channels,
                          dm.latent_channels, list(dm.hidden))  # fmt: skip
    rng = np.random.default_rng(a.thin_seed) if a.thin_seed is not None else None
    tree = uproot.open(a.root)["PVFinderData"]
    branches = [*FEATURES, *TRUTH, "ActualNumOfInt"]
    graphs, cov = [], np.zeros(len(COVERAGE_K) + 1, dtype=np.int64)
    n_pos = n_peaks = n_owned = n_truth2 = 0
    t0 = time.monotonic()
    for i, ev in enumerate(iterate_entries(tree, branches, entries, step_size=500)):
        g, c, npos, npk, nown, nt2 = build_event(ev, pvf, peaks_cfg, device,
                                                 HLLHC.match_window_mm, rng)  # fmt: skip
        g.entry = torch.tensor([int(entries[i])])
        graphs.append(g)
        cov += c
        n_pos, n_peaks, n_owned, n_truth2 = n_pos + npos, n_peaks + npk, n_owned + nown, n_truth2 + nt2  # fmt: skip
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(entries)} events, {time.monotonic() - t0:.0f} s", flush=True)  # fmt: skip
    if len(graphs) != len(entries):
        raise RuntimeError(f"built {len(graphs)} of {len(entries)} graphs")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    torch.save(graphs, tmp)
    tmp.replace(a.out)
    np.save(a.out.with_suffix(".entry_indices.npy"), entries)
    stats = {
        "root": str(a.root.resolve()), "entries": str(a.entries), "n_events": len(graphs),
        "entries_sha256": hashlib.sha256(entries.tobytes()).hexdigest(),
        "density": dm.name, "checkpoint": dm.checkpoint, "peaks": asdict(peaks_cfg),
        "cuts_overridden": bool(a.cuts), "window_mm": HLLHC.match_window_mm, "knn": KNN,
        "thin_seed": a.thin_seed, "n_peaks": n_peaks, "n_owned_peaks": n_owned,
        "n_truth_ge2": n_truth2, "tracks_with_target": int(cov[-1]),
        "tracks_with_positive_edge": n_pos,
        "knn_coverage": {str(k): float(cov[j] / max(cov[-1], 1)) for j, k in enumerate(COVERAGE_K)},
        "seconds": time.monotonic() - t0,
    }  # fmt: skip
    a.out.with_suffix(".json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
