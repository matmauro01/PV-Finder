#!/usr/bin/env python3
"""CPU time of the GNN forward per PU200 event, one thread, models interleaved.

Step 1 (``--build``, GPU allowed): s8 peaks and the kNN inference graph of the
first ``--n-events`` entries of a PU200 file, exactly as
``chain_inference_v3`` builds them; saved to ``--graphs``.

Step 2 (default, run with ``CUDA_VISIBLE_DEVICES=""``): every model computes
its per-edge scores (forward + sigmoid or softmax, ``torch.no_grad``, one
intra-op thread) on every graph. Within each event the models run round-robin
in an order rotated per event, so load drift on the shared machine affects all
of them alike (see the CPU timing note of 12 Nov). Reported per model: median
and 16/84% quantiles of the per-event time, and the median of the per-event
paired ratio to ``--reference``; load average at start and end.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_cpu_timing.py --build --graphs G.pt -d 1 \
        --root data/.../ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.root --n-events 60
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_cpu_timing.py \
        --graphs G.pt --gnn v2e527=... b=... --out timing.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def build(a: argparse.Namespace) -> None:
    """Step 1: graphs of the first n events."""
    import uproot

    from gnn.data.event_selection import iterate_entries
    from gnn.data.pu200_chain_graphs import build_pvf_model
    from gnn.evaluation.chain_inference import DENSITY, FEATURES
    from gnn.evaluation.chain_inference_v3 import s8_graph

    device = torch.device(f"cuda:{a.device}")
    dm = DENSITY["s8"]
    pvf = build_pvf_model(str(REPO / dm.checkpoint), device, dm.unet_channels,
                          dm.latent_channels, list(dm.hidden))  # fmt: skip
    tree = uproot.open(a.root)["PVFinderData"]
    out = []
    for ev in iterate_entries(tree, [*FEATURES, "ActualNumOfInt"], np.arange(a.n_events), step_size=100):  # fmt: skip
        g = s8_graph(ev, pvf, dm, device)[0]
        g.mu = torch.tensor([float(ev["ActualNumOfInt"])])
        out.append(g)
    torch.save(
        {"graphs": out, "root": str(a.root), "peaks": dm.peaks.__dict__}, a.graphs
    )
    print(f"saved {len(out)} graphs to {a.graphs}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--root", type=Path)
    ap.add_argument("--n-events", type=int, default=60)
    ap.add_argument("--graphs", required=True, type=Path)
    ap.add_argument("--gnn", nargs="+", default=[], help="name=path")
    ap.add_argument("--reference", default="v2e527")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", type=Path)
    ap.add_argument("-d", "--device", type=int, default=0)
    a = ap.parse_args()
    if a.build:
        build(a)
        return
    if torch.cuda.is_available():
        raise RuntimeError('run the timing with CUDA_VISIBLE_DEVICES=""')
    from gnn.evaluation.chain_inference_v3 import load_any

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cpu = torch.device("cpu")
    blob = torch.load(a.graphs, weights_only=False)
    graphs = blob["graphs"]
    models = {k: load_any(v, cpu) for k, v in (s.split("=", 1) for s in a.gnn)}
    names = list(models)
    if a.reference not in names:
        raise ValueError("reference model missing")
    params = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
    load0 = os.getloadavg()
    with torch.no_grad():
        for g in graphs[: a.warmup]:
            for m in models.values():
                m.scores(g)
        times = {k: [] for k in names}
        for p in range(a.passes):
            for i, g in enumerate(graphs):
                k0 = (i + p) % len(names)
                for name in names[k0:] + names[:k0]:
                    t0 = time.perf_counter()
                    models[name].scores(g)
                    times[name].append(1e3 * (time.perf_counter() - t0))
    load1 = os.getloadavg()
    ref = np.array(times[a.reference])
    res = {}
    for k in names:
        t = np.array(times[k])
        res[k] = {
            "params": params[k], "median_ms": float(np.median(t)),
            "q16_ms": float(np.quantile(t, 0.16)), "q84_ms": float(np.quantile(t, 0.84)),
            "ratio_to_ref_median": float(np.median(t / ref)),
            "ratio_q16": float(np.quantile(t / ref, 0.16)), "ratio_q84": float(np.quantile(t / ref, 0.84)),
        }  # fmt: skip
        print(f"{k:10s} {params[k]:>10,d} params  {res[k]['median_ms']:7.1f} ms  "
              f"x{res[k]['ratio_to_ref_median']:.2f} vs {a.reference}", flush=True)  # fmt: skip
    n_tracks = [int(g["track"].x.shape[0]) for g in graphs]
    n_edges = [int(g[("track", "to", "pv")].edge_index.shape[1]) for g in graphs]
    out = {
        "models": res, "gnn": dict(s.split("=", 1) for s in a.gnn), "reference": a.reference,
        "n_events": len(graphs), "passes": a.passes, "threads": 1, "graphs": str(a.graphs),
        "root": blob["root"], "peaks": blob["peaks"],
        "mean_tracks": float(np.mean(n_tracks)), "mean_edges": float(np.mean(n_edges)),
        "mean_mu": float(np.mean([float(g.mu) for g in graphs])),
        "loadavg_start": load0, "loadavg_end": load1, "torch": torch.__version__,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }  # fmt: skip
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
