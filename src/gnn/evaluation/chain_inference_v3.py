"""Full-chain inference for GNN v3 networks, in the ``chain_inference`` chunk format.

Same events, density model, peak finding and kNN graph as
``gnn.evaluation.chain_inference`` (whose helpers it imports), but each
``--gnn name=path`` may be a v3-campaign checkpoint (``model_config.json``
beside it, loaded with ``train_ttva_v3.load_v3``) or a legacy v1/v2 ``.pyt``.
Scores are each model's own per-edge scores (sigmoid, or the per-track softmax
with the "no vertex" option); per track only the top-1 edge is stored, so
``gnn.evaluation.chain_t_metrics`` replays any t unchanged.

``--candidates truth`` replaces the peaks by the truth vertex list (the exact
truth-training graphs, ``root_to_graphs.build_graph_from_event``) for the
"GNN on truth candidates" diagnostic.

Peak cuts are read from ``pv_finder.utils.working_points`` at run time and
recorded in ``metadata.json``; an existing output with other cuts is refused.

    PYTHONPATH=src python -u -m gnn.evaluation.chain_inference_v3 \\
        --root data/.../ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \\
        --entries data/.../fullmu_v1_20260924/validation_entries.npy \\
        --gnn r1=model_weights/ttva_gnn_v3/r1/r1_epoch_359.pt --out outputs/.../infer/val_s8 -d 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from gnn.data.event_selection import iterate_entries
from gnn.data.graph_construction import create_inference_graph
from gnn.data.root_to_graphs import (
    TRACK_BRANCHES,
    TRUTH_BRANCHES,
    build_graph_from_event,
    compute_pv_sigma_preset,
)
from gnn.evaluation.chain_inference import (
    DENSITY,
    FEATURES,
    KNN,
    REPO,
    best_edges,
    git_state,
    load_gnn,
    parse_entries,
    sha256,
)
from gnn.training.train_ttva_v3 import load_v3
from pv_finder.data.resolution_presets import RESOLUTION_PRESETS
from pv_finder.data.run3_io import Run3Event
from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import (
    build_subevent_inputs,
    run_inference,
)
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast
from pv_finder.utils.working_points import HLLHC


class _Legacy(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def scores(self, g):  # noqa: ANN001, ANN201
        return torch.sigmoid(self.net(g))


def load_any(path: str, device: torch.device) -> torch.nn.Module:
    """v3 checkpoint (with model_config.json) or legacy v1/v2 .pyt."""
    if (Path(path).parent / "model_config.json").exists():
        return load_v3(path, device)
    return _Legacy(load_gnn(path, device))


def s8_graph(ev, pvf, dm, device):  # noqa: ANN001, ANN201
    """Inference graph on density-model peaks, exactly as chain_inference."""
    f = [ak.to_numpy(ev[k]).astype(np.float32) for k in FEATURES]
    d0, z0, ed0, ez0, cov, theta, phi, pt = f
    n = len(z0)
    r3 = Run3Event(
        z0=z0, d0=d0, d0_err=ed0, z0_err=ez0, d0_z0_cov=cov,
        amvf_z=np.array([]), amvf_ntrks=np.array([]), beam_z=0.0,
        mu=None, event_idx=0, n_tracks=n,
    )  # fmt: skip
    hist = run_inference(build_subevent_inputs(r3), device, e2e=pvf)
    p = dm.peaks
    pz, ph, _, ps = pv_locations_updated_res_fast(
        hist, p.amplitude, p.integral, p.min_width, p.min_height, p.centroid_halfwidth
    )  # fmt: skip
    stack = np.stack(f, axis=1)
    stack[:, 7] = pt / PT_SCALE
    g = create_inference_graph(z0, d0, ez0, ed0, stack, pz, ph, ps, knn=KNN)
    g["pv"].sigma = torch.from_numpy(np.asarray(ps, dtype=np.float32))
    return g, n, np.asarray(pz), np.asarray(ph), np.asarray(ps)


def truth_graph(ev):  # noqa: ANN001, ANN201
    """Truth-training graph of the event (truth vertices as candidates)."""
    res = RESOLUTION_PRESETS["hllhc_alleta"]
    g = build_graph_from_event(ev, KNN, res)
    tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.float64)
    sig = compute_pv_sigma_preset(tn, *res).astype(np.float32)
    g["pv"].sigma = torch.from_numpy(sig)
    pz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
    return g, int(g["track"].x.shape[0]), pz, g["pv"].x[:, 1].numpy().copy(), sig


def main() -> None:
    """CLI entry point."""
    from gnn.data.pu200_chain_graphs import build_pvf_model

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--entries", required=True, help="a:b or file.npy")
    ap.add_argument("--candidates", default="s8", choices=["s8", "v6", "truth"])
    ap.add_argument("--gnn", required=True, nargs="+", help="name=path")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("-d", "--device", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(2)
    device = torch.device(f"cuda:{a.device}")
    entries = parse_entries(a.entries)
    gnn_paths = dict(s.split("=", 1) for s in a.gnn)
    truth = a.candidates == "truth"
    dm = None if truth else DENSITY[a.candidates]
    density = (
        {"name": "truth", "peaks": None, "sha256": None}
        if truth
        else {
            **dm.__dict__,
            "peaks": dm.peaks.__dict__,
            "sha256": sha256(REPO / dm.checkpoint),
        }  # fmt: skip
    )
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {
        "root": str(a.root.resolve()), "entries_spec": a.entries,
        "n_entries": int(len(entries)),
        "entries_sha256": hashlib.sha256(entries.tobytes()).hexdigest(),
        "density": density,
        "gnn": {k: {"path": v, "sha256": sha256(Path(v))} for k, v in gnn_paths.items()},
        "knn": KNN, "pt_scale": PT_SCALE, "chunk": a.chunk,
        "regime_window_mm": HLLHC.match_window_mm, "git": git_state(),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(device),
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
    pvf = None
    if not truth:
        pvf = build_pvf_model(str(REPO / dm.checkpoint), device, dm.unet_channels,
                              dm.latent_channels, list(dm.hidden))  # fmt: skip
    gnns = {k: load_any(v, device) for k, v in gnn_paths.items()}
    tree = uproot.open(a.root)["PVFinderData"]
    branches = sorted({*FEATURES, *TRACK_BRANCHES, *TRUTH_BRANCHES, "ActualNumOfInt"})
    t0 = time.monotonic()
    for k, lo in enumerate(range(0, len(entries), a.chunk)):
        path = a.out / f"chunk_{k:04d}.npz"
        if path.exists():
            continue
        sel = entries[lo : lo + a.chunk]
        cols: dict[str, list] = {c: [] for c in ("mu", "n", "pz", "ph", "ps")}
        edges: dict[str, tuple[list, list]] = {g: ([], []) for g in gnns}
        n_read = 0
        for ev in iterate_entries(tree, branches, sel, step_size=500):
            n_read += 1
            g, n, pz, ph, ps = (
                truth_graph(ev) if truth else s8_graph(ev, pvf, dm, device)
            )
            ei = g[("track", "to", "pv")].edge_index.numpy().copy()
            g = g.to(device)
            for name, model in gnns.items():
                with torch.no_grad():
                    sc = model.scores(g).float().cpu().numpy()
                tg, bs = best_edges(sc, ei, n)
                edges[name][0].append(tg)
                edges[name][1].append(bs)
            cols["mu"].append(float(ev["ActualNumOfInt"]))
            cols["n"].append(n)
            cols["pz"].append(np.asarray(pz, dtype=np.float32))
            cols["ph"].append(np.asarray(ph, dtype=np.float32))
            cols["ps"].append(np.asarray(ps, dtype=np.float32))
        if n_read != len(sel):
            raise RuntimeError(f"read {n_read} of {len(sel)} events")
        res = {
            "entries": sel, "mu": np.array(cols["mu"]),
            "n_tracks": np.array(cols["n"], dtype=np.int64),
            "track_offsets": np.r_[0, np.cumsum(cols["n"])].astype(np.int64),
            "peak_offsets": np.r_[0, np.cumsum([len(x) for x in cols["pz"]])].astype(np.int64),
            "peak_z": np.concatenate(cols["pz"]), "peak_height": np.concatenate(cols["ph"]),
            "peak_sigma": np.concatenate(cols["ps"]),
        }  # fmt: skip
        for name, (tg, bs) in edges.items():
            res[f"targets_{name}"] = np.concatenate(tg)
            res[f"scores_{name}"] = np.concatenate(bs)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **res)
        tmp.replace(path)
        print(f"{lo + len(sel)}/{len(entries)} events, {time.monotonic() - t0:.0f} s", flush=True)  # fmt: skip
    (a.out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()))


if __name__ == "__main__":
    main()
