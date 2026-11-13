"""CPU timing of the non-density stages of the HL-LHC chain, one thread.

For real PU200 ROOT events, times each stage the chain runs after the tracks
are read, exactly as ``gnn.evaluation.chain_inference`` runs it:

  tensor   ``build_subevent_inputs`` (12 padded sub-event tensors)
  density  the density model forward in torch, padded (reference only: the
           CPU density number of record is the ONNX Runtime one from
           ``scripts/cpu_inference_benchmark.py``)
  peaks    ``pv_locations_updated_res_fast`` with the model's frozen cuts
           (numba, compiled before timing; includes the local centroid)
  graph    feature stack + ``create_inference_graph`` (kNN k = 20, undirected)
  gnn      ``model(graph)`` under ``torch.no_grad`` for every GNN given, on
           the same graph, back to back
  select   sigmoid + per-track best edge (``best_edges``)

Every stage is timed on every event in every pass. The GNNs are timed in a
different order in each pass so that load drift does not favour one.

Usage (repo root, inside tmux, CUDA_VISIBLE_DEVICES=""):
    PYTHONPATH=src python -u scripts/cpu_chain_timing.py \\
        --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/...r16438_PU200.root \\
        --n-events 100 --density s8 \\
        --gnn v2e527=model_weights/.../ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt \\
              v4e107=model_weights/.../ttva_gat_alleta_k20_v4_noaug180k_epoch_107.pyt \\
        --out outputs/<date>/cpu_timing/chain_stages.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot

from gnn.data.event_selection import iterate_entries
from gnn.data.graph_construction import create_inference_graph
from gnn.data.pu200_chain_graphs import build_pvf_model
from gnn.evaluation.chain_inference import (
    DENSITY,
    FEATURES,
    KNN,
    REPO,
    best_edges,
    git_state,
    load_gnn,
    sha256,
)
from pv_finder.data.run3_io import Run3Event
from pv_finder.evaluation.vertex_finding.run_eval_pvf_run3 import (
    build_subevent_inputs,
    run_inference,
)
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding_fast import pv_locations_updated_res_fast
from pv_finder.utils.working_points import DensityModel

STAGES = ("tensor", "density_torch", "peaks", "graph")


def _ms(t0: float) -> float:
    """Milliseconds since ``t0`` (perf_counter)."""
    return (time.perf_counter() - t0) * 1e3


def _stats(values: list[float]) -> dict[str, float]:
    """Median, mean, p10 and p90 of a list of times (ms)."""
    a = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(a)),
        "mean_ms": float(a.mean()),
        "p10_ms": float(np.percentile(a, 10)),
        "p90_ms": float(np.percentile(a, 90)),
    }


def read_events(root: Path, start: int, n: int) -> list:
    """The ROOT entries ``start .. start + n - 1`` with the chain's branches."""
    tree = uproot.open(root)["PVFinderData"]
    idx = np.arange(start, start + n, dtype=np.int64)
    events = list(iterate_entries(tree, [*FEATURES, "ActualNumOfInt"], idx))
    if len(events) != n:
        raise RuntimeError(f"read {len(events)} of {n} events")
    return events


def time_event(
    ev: object,
    pvf: torch.nn.Module,
    gnns: dict[str, torch.nn.Module],
    order: list[str],
    dm: DensityModel,
) -> dict[str, float]:
    """Time every stage of the chain on one event; return ms and sizes."""
    cpu = torch.device("cpu")
    p = dm.peaks
    f = [ak.to_numpy(ev[k]).astype(np.float32) for k in FEATURES]
    d0, z0, ed0, ez0, cov, theta, phi, pt = f
    n = len(z0)
    rec: dict[str, float] = {"n_tracks": n, "mu": float(ev["ActualNumOfInt"])}

    t0 = time.perf_counter()
    r3 = Run3Event(
        z0=z0, d0=d0, d0_err=ed0, z0_err=ez0, d0_z0_cov=cov,
        amvf_z=np.array([]), amvf_ntrks=np.array([]), beam_z=0.0,
        mu=None, event_idx=0, n_tracks=n,
    )  # fmt: skip
    subs = build_subevent_inputs(r3)
    rec["tensor"] = _ms(t0)

    t0 = time.perf_counter()
    hist = run_inference(subs, cpu, e2e=pvf)
    rec["density_torch"] = _ms(t0)

    t0 = time.perf_counter()
    pz, ph, _, ps = pv_locations_updated_res_fast(
        hist, p.amplitude, p.integral, p.min_width, p.min_height,
        p.centroid_halfwidth,
    )  # fmt: skip
    rec["peaks"] = _ms(t0)
    rec["n_peaks"] = len(pz)

    t0 = time.perf_counter()
    stack = np.stack(f, axis=1)
    stack[:, 7] = pt / PT_SCALE
    graph = create_inference_graph(z0, d0, ez0, ed0, stack, pz, ph, ps, knn=KNN)
    rec["graph"] = _ms(t0)
    ei = graph[("track", "to", "pv")].edge_index.numpy()
    rec["n_edges_track_to_pv"] = ei.shape[1]

    for name in order:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = gnns[name](graph)
        rec[f"gnn_{name}"] = _ms(t0)
        t0 = time.perf_counter()
        best_edges(torch.sigmoid(logits).numpy(), ei, n)
        rec[f"select_{name}"] = _ms(t0)
    return rec


def check_same_architecture(gnns: dict[str, torch.nn.Module]) -> dict[str, int]:
    """Parameter count per GNN; raise if the tensor shapes differ between them."""
    shapes = {
        k: [(n, tuple(t.shape)) for n, t in m.state_dict().items()]
        for k, m in gnns.items()
    }
    ref = next(iter(shapes.values()))
    for k, s in shapes.items():
        if s != ref:
            raise ValueError(f"GNN {k} has a different architecture")
    return {k: sum(q.numel() for q in m.parameters()) for k, m in gnns.items()}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-events", type=int, default=100)
    ap.add_argument("--n-warmup", type=int, default=5)
    ap.add_argument("--n-pass", type=int, default=3)
    ap.add_argument("--density", default="s8", choices=sorted(DENSITY))
    ap.add_argument("--gnn", required=True, nargs="+", help="name=path.pyt")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    dm = DENSITY[a.density]
    cpu = torch.device("cpu")
    pvf = build_pvf_model(
        str(REPO / dm.checkpoint), cpu, dm.unet_channels, dm.latent_channels,
        list(dm.hidden),
    ).eval()  # fmt: skip
    gnn_paths = dict(s.split("=", 1) for s in a.gnn)
    gnns = {k: load_gnn(v, cpu) for k, v in gnn_paths.items()}
    n_params = check_same_architecture(gnns)

    events = read_events(a.root, a.start, a.n_events)
    names = list(gnns)
    for ev in events[: a.n_warmup]:  # numba compile, allocator and PyG warm-up
        time_event(ev, pvf, gnns, names, dm)

    per_pass: list[list[dict[str, float]]] = []
    for k in range(a.n_pass):
        order = names[k % len(names) :] + names[: k % len(names)]
        t0 = time.perf_counter()
        per_pass.append([time_event(ev, pvf, gnns, order, dm) for ev in events])
        print(f"pass {k}: order {order}, {_ms(t0) / 1e3:.0f} s", flush=True)

    keys = [*STAGES, *(f"gnn_{g}" for g in names), *(f"select_{g}" for g in names)]
    summary = {
        k: {
            "per_pass": [_stats([r[k] for r in recs]) for recs in per_pass],
            "median_of_pass_medians_ms": float(
                np.median([np.median([r[k] for r in recs]) for recs in per_pass])
            ),
        }
        for k in keys
    }
    first = per_pass[0]
    out = {
        "what": "per-event CPU wall time of chain stages, torch, real PU200 events",
        "threads": {"torch_intra": torch.get_num_threads(), "torch_interop": 1,
                    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                    "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS")},  # fmt: skip
        "root": str(a.root.resolve()),
        "entries": [a.start, a.start + a.n_events],
        "n_events": a.n_events, "n_pass": a.n_pass, "n_warmup": a.n_warmup,
        "mu_min_max": [min(r["mu"] for r in first), max(r["mu"] for r in first)],
        "n_tracks_mean": float(np.mean([r["n_tracks"] for r in first])),
        "n_peaks_mean": float(np.mean([r["n_peaks"] for r in first])),
        "n_edges_track_to_pv_mean": float(
            np.mean([r["n_edges_track_to_pv"] for r in first])
        ),
        "knn": KNN,
        "density": {"name": dm.name, "checkpoint": dm.checkpoint,
                    "sha256": sha256(REPO / dm.checkpoint),
                    "peaks": dm.peaks.__dict__},  # fmt: skip
        "gnn": {k: {"path": v, "sha256": sha256(Path(v)), "params": n_params[k]}
                for k, v in gnn_paths.items()},  # fmt: skip
        "stages": summary,
        "torch": torch.__version__,
        "git": git_state(),
        "per_event_pass0": first,
    }  # fmt: skip
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    for k in keys:
        print(f"{k:>16s}: {summary[k]['median_of_pass_medians_ms']:8.2f} ms/event")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
