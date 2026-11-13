"""CPU inference benchmark for PV-Finder v2 models of different widths.

Exports each model to ONNX (opset 17, as in the Athena integration), optionally
quantises it to INT8 with ONNX Runtime, and times one event (a batch of 12
sub-events of 695 padded tracks) on CPU at a fixed number of intra-op threads.

Absolute times depend on the machine and its load; compare models measured in
the same invocation. With ``--interleave`` all models are timed round-robin,
event by event, so load drift during the run affects them alike (the default
times one model after the other).

Example:
    python -u scripts/cpu_inference_benchmark.py --out-dir outputs/11_09_2026_output/cpu_bench \
        --widths 280:4 112:1 80:1 --threads 1 4 \
        --weights 280:4=model_weights/hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth \
        --calib-h5 data/run4_all_etas/PU200_alleta_h5/...r16438_PU200.h5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pv_finder.models.packed_inference import (  # noqa: E402
    PackedTracksToHist,
    pack_event,
)
from pv_finder.training.hllhc_helpers import build_model, load_model_state  # noqa: E402

N_SUB = 12
N_TRK = 695
N_FEAT = 7


def make_model(
    width: int, latent: int, hidden: int, weights: str | None
) -> torch.nn.Module:
    """Build a v2 model; load weights if given, else keep random init."""
    cfg = {
        "models_config": {
            "model_type": "v2",
            "n_input_features": N_FEAT,
            "n_latent_channels": latent,
            "dropout": 0.25,
            "n_unet_channels": width,
            "l_hidden_nodes": [hidden] * 5,
        }
    }
    model = build_model(cfg, torch.device("cpu"))
    if weights:
        load_model_state(weights, model)
    return model.eval()


def export_onnx(model: torch.nn.Module, path: Path) -> None:
    """Export with a dynamic batch axis, opset 17."""
    x = torch.randn(N_SUB, N_FEAT, N_TRK)  # all slots valid: no crop is traced
    torch.onnx.export(
        model,
        x,
        str(path),
        opset_version=17,
        input_names=["tracks"],
        output_names=["hist"],
        dynamic_axes={"tracks": {0: "batch", 2: "ntrk"}, "hist": {0: "batch"}},
        dynamo=False,
    )


def export_packed_onnx(model: torch.nn.Module, path: Path) -> None:
    """Export the packed-track wrapper with a dynamic track axis, opset 17."""
    packed = PackedTracksToHist(model).eval()
    tracks = torch.randn(500, N_FEAT)
    sub_idx = torch.arange(500) % N_SUB
    torch.onnx.export(
        packed,
        (tracks, sub_idx),
        str(path),
        opset_version=17,
        input_names=["tracks", "sub_idx"],
        output_names=["hist"],
        dynamic_axes={"tracks": {0: "ntrk"}, "sub_idx": {0: "ntrk"}},
        dynamo=False,
    )


def packed_feeds(events: list[np.ndarray]) -> list[dict]:
    """Packed ONNX inputs for each padded event."""
    feeds = []
    for e in events:
        tracks, idx = pack_event(torch.from_numpy(e))
        feeds.append({"tracks": tracks.numpy(), "sub_idx": idx.numpy()})
    return feeds


def max_abs_diff(path_a: Path, feeds_a: list, path_b: Path, feeds_b: list) -> float:
    """Largest absolute output difference between two models on the same events."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.enable_cpu_mem_arena = False  # keep virtual memory under the ulimit
    a = ort.InferenceSession(str(path_a), so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(path_b), so, providers=["CPUExecutionProvider"])
    return max(
        float(np.abs(a.run(None, fa)[0] - b.run(None, fb)[0]).max())
        for fa, fb in zip(feeds_a[:10], feeds_b[:10])
    )


def load_real_events(h5_path: str, n_events: int, start: int = 0) -> list[np.ndarray]:
    """Return per-event (12, 7, n_trk) float32 track tensors from an h5 file.

    Each event is cropped to its widest sub-event (z0 > -240 marks a real
    track), as a tensor builder that does not pad to a fixed width would do.
    """
    with h5py.File(h5_path, "r") as f:
        arr = f["tracks"][start * N_SUB : (start + n_events) * N_SUB].astype(np.float32)
    events = []
    for e in arr.reshape(n_events, N_SUB, N_FEAT, -1):
        valid = (e[:, 1, :] > -240.0).any(axis=0)
        n_keep = int(np.nonzero(valid)[0][-1]) + 1
        events.append(np.ascontiguousarray(e[:, :, :n_keep]))
    return events


class _Calib(CalibrationDataReader):
    def __init__(self, events: np.ndarray):
        self._it = iter([{"tracks": e} for e in events])  # padded inputs

    def get_next(self) -> dict | None:
        return next(self._it, None)


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    """CPU session at a fixed number of intra-op threads."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _summary(times: list[float]) -> dict:
    """Median and p90 of per-event times (ms)."""
    return {
        "median_ms": float(np.median(times)),
        "p90_ms": float(np.percentile(times, 90)),
    }


def time_session(path: Path, threads: int, feeds: list[dict], n_rep: int) -> dict:
    """Median and p90 wall time per event (ms)."""
    sess = make_session(path, threads)
    for fd in feeds[:2]:
        sess.run(None, fd)  # warm-up
    times = []
    for i in range(n_rep):
        fd = feeds[i % len(feeds)]
        t0 = time.perf_counter()
        sess.run(None, fd)
        times.append((time.perf_counter() - t0) * 1e3)
    return _summary(times)


def time_interleaved(
    jobs: list[tuple[Path, int, list[dict]]], n_rep: int
) -> list[dict]:
    """Time all sessions round-robin on each event, so load drift hits all alike.

    Each job is (onnx path, threads, feeds); event i is run on every session
    before event i + 1. Returns one summary per job, in job order.
    """
    sessions = [make_session(p, th) for p, th, _ in jobs]
    for sess, (_, _, feeds) in zip(sessions, jobs):
        for fd in feeds[:2]:
            sess.run(None, fd)  # warm-up
    times: list[list[float]] = [[] for _ in jobs]
    for i in range(n_rep):
        for k, (sess, (_, _, feeds)) in enumerate(zip(sessions, jobs)):
            fd = feeds[i % len(feeds)]
            t0 = time.perf_counter()
            sess.run(None, fd)
            times[k].append((time.perf_counter() - t0) * 1e3)
    out = [_summary(t) for t in times]
    for o, t in zip(out, times):
        o["times_ms"] = [round(x, 3) for x in t]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--widths", nargs="+", default=["280:4", "112:1", "80:1"],
                   help="width:latent[:hidden] specs")  # fmt: skip
    p.add_argument("--weights", nargs="*", default=[], help="spec=checkpoint pairs")
    p.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    p.add_argument(
        "--calib-h5", default=None, help="real events for timing/calibration"
    )
    p.add_argument("--n-rep", type=int, default=20)
    p.add_argument("--quant", action="store_true", help="also time INT8 variants")
    p.add_argument(
        "--interleave",
        action="store_true",
        help="time all models round-robin event by event (load-robust ratios)",
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(w.split("=", 1) for w in args.weights)
    if args.calib_h5:
        events = load_real_events(args.calib_h5, 40)
    else:
        events = list(np.random.randn(8, N_SUB, N_FEAT, N_TRK).astype(np.float32))

    pad_feeds = [{"tracks": e} for e in events]
    pk_feeds = packed_feeds(events)
    results = []
    jobs: list[tuple[Path, int, list[dict]]] = []  # used with --interleave
    for spec in args.widths:
        parts = [int(v) for v in spec.split(":")]
        width, latent = parts[0], parts[1]
        hidden = parts[2] if len(parts) > 2 else 128
        model = make_model(width, latent, hidden, weights.get(spec))
        n_par = sum(q.numel() for q in model.parameters())
        base = out / f"pvf_w{width}_l{latent}_h{hidden}.onnx"
        export_onnx(model, base)
        variants = {"fp32": base}
        packed = base.with_suffix(".packed.onnx")
        export_packed_onnx(model, packed)
        variants["fp32_packed"] = packed
        diff = max_abs_diff(base, pad_feeds, packed, pk_feeds)
        print(f"{spec}: packed vs padded max |diff| = {diff:.3g}", flush=True)
        if args.quant:
            dyn = base.with_suffix(".int8dyn.onnx")
            quantize_dynamic(
                str(base),
                str(dyn),
                weight_type=QuantType.QInt8,
                op_types_to_quantize=["MatMul", "Gemm"],
            )
            variants["int8_dynamic"] = dyn
            st = base.with_suffix(".int8static.onnx")
            quantize_static(str(base), str(st), _Calib(events[:16]),
                            quant_format=QuantFormat.QDQ, per_channel=True,
                            activation_type=QuantType.QUInt8,
                            weight_type=QuantType.QInt8)  # fmt: skip
            variants["int8_static"] = st
        for vname, vpath in variants.items():
            for th in args.threads:
                feeds = pk_feeds if "packed" in vname else pad_feeds
                row = {"spec": spec, "params": n_par, "variant": vname, "threads": th}
                if args.interleave:
                    jobs.append((vpath, th, feeds))
                else:
                    row.update(time_session(vpath, th, feeds, args.n_rep))
                    print(json.dumps(row), flush=True)
                results.append(row)
    if args.interleave:
        for row, r in zip(results, time_interleaved(jobs, args.n_rep)):
            row.update(r, interleaved=True)
            print(json.dumps({k: v for k, v in row.items() if k != "times_ms"}))
    ntrk = [e.shape[2] for e in events]
    print(f"track-axis width per event: median {np.median(ntrk):.0f}, max {max(ntrk)}")
    (out / "cpu_bench.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
