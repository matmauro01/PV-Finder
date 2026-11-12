"""Full-chain inference: density model -> peaks -> top-1 GNN edges, saved per chunk.

For each selected ROOT entry: run a PV-Finder density model from
``pv_finder.utils.working_points`` (by name, with its own peak cuts), find
peaks, build the kNN inference graph exactly as the deck evaluation did
(``create_inference_graph``, k = 20, track pT / PT_SCALE), and run one or more
GNN checkpoints. Per track only the best edge is kept (target peak and
sigmoid score; ties go to the first edge in graph order). Any score cut t is
then a replay (``gnn.evaluation.chain_t_metrics``), so no inference is
repeated per t.

Output per chunk of entries, ``chunk_XXXX.npz``:
  entries, mu, n_tracks, peak_offsets, peak_z, peak_height, peak_sigma,
  [histograms], and per GNN name ``targets_<name>``, ``scores_<name>``
  (flat over tracks; event i owns ``track_offsets[i]:track_offsets[i+1]``).
``metadata.json`` records git, checkpoints with sha256, the regime and the
entry hash.

Usage (repo root, inside tmux):
    PYTHONPATH=src python -u -m gnn.evaluation.chain_inference \\
        --root data/.../ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \\
        --entries 0:25000 --density v6 \\
        --gnn v2e527=model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt \\
        --out outputs/11_12_2026_output/gnn_v2_freeze/infer/test_v6 -d 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from gnn.data.event_selection import iterate_entries
from gnn.data.graph_construction import create_inference_graph
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.data.run3_io import Run3Event
from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import (
    build_subevent_inputs,
    run_inference,
)
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast
from pv_finder.utils.working_points import (
    HLLHC,
    S8,
    V6,
    DensityModel,
)

DENSITY: dict[str, DensityModel] = {"v6": V6, "s8": S8}
FEATURES = (
    "RecoTrack_d0", "RecoTrack_z0", "RecoTrack_ErrD0", "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0", "RecoTrack_theta", "RecoTrack_phi", "RecoTrack_pT",
)  # fmt: skip
KNN = 20
REPO = Path(__file__).resolve().parents[3]


def best_edges(
    scores: np.ndarray, edge_index: np.ndarray, n_tracks: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per track, the target and score of its highest-scoring edge.

    ``edge_index[0]`` must be sorted (track-major, as ``create_inference_graph``
    builds it). Ties go to the first edge in that order. Tracks with no edge
    get target -1 and score 0.
    """
    src = np.asarray(edge_index[0])
    if len(src) and np.any(np.diff(src) < 0):
        raise ValueError("edges are not track-major")
    targets = np.full(n_tracks, -1, dtype=np.int64)
    best = np.zeros(n_tracks, dtype=np.float32)
    if len(src) == 0:
        return targets, best
    uniq, starts = np.unique(src, return_index=True)
    top = np.maximum.reduceat(scores, starts)
    pos = np.arange(len(scores))
    first = np.minimum.reduceat(
        np.where(
            scores == np.repeat(top, np.diff(np.r_[starts, len(src)])), pos, len(src)
        ),
        starts,
    )
    targets[uniq] = edge_index[1][first]
    best[uniq] = scores[first]
    return targets, best


def load_gnn(path: str, device: torch.device) -> TTVAGATModel:
    """The v1/v2 full-mu architecture (GAT, hidden 32, 4 heads) with weights."""
    model = TTVAGATModel(
        track_input_size=8, pv_input_size=2, hidden_dim=32, num_heads=4,
        dropout=0.25, edge_attr_dim=3,
    )  # fmt: skip
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def parse_entries(spec: str) -> np.ndarray:
    """``a:b`` (range) or a ``.npy`` file of entry indices; returned sorted."""
    if spec.endswith(".npy"):
        e = np.load(spec).astype(np.int64)
    else:
        lo, hi = (int(x) for x in spec.split(":"))
        e = np.arange(lo, hi, dtype=np.int64)
    if len(np.unique(e)) != len(e):
        raise ValueError("duplicate entries")
    return np.sort(e)


def sha256(path: Path) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict[str, str]:
    """HEAD and whether tracked files differ from it."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    return {"head": head, "tracked_dirty": dirty}


def process_chunk(
    events: list, entries: np.ndarray, pvf: torch.nn.Module,
    gnns: dict[str, TTVAGATModel], dm: DensityModel, device: torch.device,
    keep_hist: bool,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Inference for one chunk of events (already read, in entry order)."""
    p = dm.peaks
    out: dict[str, list] = {k: [] for k in ("mu", "n_tracks", "pz", "ph", "ps", "hist")}
    edges: dict[str, tuple[list, list]] = {g: ([], []) for g in gnns}
    for ev in events:
        f = [ak.to_numpy(ev[k]).astype(np.float32) for k in FEATURES]
        d0, z0, ed0, ez0, cov, theta, phi, pt = f
        n = len(z0)
        r3 = Run3Event(
            z0=z0, d0=d0, d0_err=ed0, z0_err=ez0, d0_z0_cov=cov,
            amvf_z=np.array([]), amvf_ntrks=np.array([]), beam_z=0.0,
            mu=None, event_idx=0, n_tracks=n,
        )  # fmt: skip
        hist = run_inference(build_subevent_inputs(r3), device, e2e=pvf)
        pz, ph, _, ps = pv_locations_updated_res_fast(
            hist, p.amplitude, p.integral, p.min_width, p.min_height,
            p.centroid_halfwidth,
        )  # fmt: skip
        stack = np.stack(f, axis=1)
        stack[:, 7] = pt / PT_SCALE
        graph = create_inference_graph(z0, d0, ez0, ed0, stack, pz, ph, ps, knn=KNN)
        graph = graph.to(device)
        ei = graph[("track", "to", "pv")].edge_index.cpu().numpy()
        for name, model in gnns.items():
            with torch.no_grad():
                sc = torch.sigmoid(model(graph)).cpu().numpy()
            tg, bs = best_edges(sc, ei, n)
            edges[name][0].append(tg)
            edges[name][1].append(bs)
        out["mu"].append(float(ev["ActualNumOfInt"]))
        out["n_tracks"].append(n)
        out["pz"].append(np.asarray(pz, dtype=np.float32))
        out["ph"].append(np.asarray(ph, dtype=np.float32))
        out["ps"].append(np.asarray(ps, dtype=np.float32))
        if keep_hist:
            out["hist"].append(hist.astype(np.float32))
    res = {
        "entries": entries,
        "mu": np.array(out["mu"]),
        "n_tracks": np.array(out["n_tracks"], dtype=np.int64),
        "track_offsets": np.r_[0, np.cumsum(out["n_tracks"])].astype(np.int64),
        "peak_offsets": np.r_[0, np.cumsum([len(x) for x in out["pz"]])].astype(
            np.int64
        ),
        "peak_z": np.concatenate(out["pz"]),
        "peak_height": np.concatenate(out["ph"]),
        "peak_sigma": np.concatenate(out["ps"]),
    }
    if keep_hist:
        res["histograms"] = np.stack(out["hist"])
    for name, (tg, bs) in edges.items():
        res[f"targets_{name}"] = np.concatenate(tg)
        res[f"scores_{name}"] = np.concatenate(bs)
    return res


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--entries", required=True, help="a:b or file.npy")
    ap.add_argument("--density", required=True, choices=sorted(DENSITY))
    ap.add_argument("--gnn", required=True, nargs="+", help="name=path.pyt")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--save-histograms", action="store_true")
    ap.add_argument("-d", "--device", type=int, default=0)
    a = ap.parse_args()
    from gnn.data.pu200_chain_graphs import build_pvf_model

    torch.set_num_threads(2)
    device = torch.device(f"cuda:{a.device}" if a.device >= 0 else "cpu")
    dm = DENSITY[a.density]
    entries = parse_entries(a.entries)
    gnn_paths = dict(s.split("=", 1) for s in a.gnn)
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "root": str(a.root.resolve()),
        "entries_spec": a.entries,
        "n_entries": int(len(entries)),
        "entries_sha256": hashlib.sha256(entries.tobytes()).hexdigest(),
        "density": {**dm.__dict__, "peaks": dm.peaks.__dict__,
                    "sha256": sha256(REPO / dm.checkpoint)},  # fmt: skip
        "gnn": {k: {"path": v, "sha256": sha256(Path(v))} for k, v in gnn_paths.items()},
        "knn": KNN, "pt_scale": PT_SCALE, "chunk": a.chunk,
        "regime_window_mm": HLLHC.match_window_mm,
        "git": git_state(), "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "script_sha256": sha256(Path(__file__)),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }  # fmt: skip
    meta_path = a.out / "metadata.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        for k in ("entries_sha256", "density", "gnn", "root"):
            if old[k] != meta[k]:
                raise RuntimeError(f"existing output differs in {k}; use a new --out")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    pvf = build_pvf_model(
        str(REPO / dm.checkpoint), device, dm.unet_channels, dm.latent_channels,
        list(dm.hidden),
    )  # fmt: skip
    gnns = {k: load_gnn(v, device) for k, v in gnn_paths.items()}
    tree = uproot.open(a.root)["PVFinderData"]
    branches = [*FEATURES, "ActualNumOfInt"]
    t0 = time.monotonic()
    for k, lo in enumerate(range(0, len(entries), a.chunk)):
        path = a.out / f"chunk_{k:04d}.npz"
        if path.exists():
            continue
        sel = entries[lo : lo + a.chunk]
        events = list(iterate_entries(tree, branches, sel, step_size=500))
        if len(events) != len(sel):
            raise RuntimeError(f"read {len(events)} of {len(sel)} events")
        res = process_chunk(events, sel, pvf, gnns, dm, device, a.save_histograms)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **res)
        tmp.replace(path)
        done = lo + len(sel)
        print(
            f"{done}/{len(entries)} events, {time.monotonic() - t0:.0f} s", flush=True
        )
    (a.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


if __name__ == "__main__":
    main()
