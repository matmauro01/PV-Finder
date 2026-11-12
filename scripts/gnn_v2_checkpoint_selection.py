#!/usr/bin/env python3
"""GNN v2 checkpoint choice: learning curves, plateau test, v1 vs v2 on truth graphs.

1. Reads both training ``learning_curve.jsonl`` files (same 1,500-event
   validation monitor, same loss), picks the v2 epoch with the lowest
   validation loss and tests the plateau (per-pass mean validation loss and
   its pass-to-pass change).
2. Re-evaluates the truth-candidate track metrics of the chosen v2 epoch, the
   last v2 epoch and v1's chosen epoch 213 on the identical monitor graphs
   (``gnn_fullmu_campaign.validation_tracks``, the code that wrote
   ``validation_pass_*.json``).
3. Plots the learning curves against event presentations.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_checkpoint_selection.py \
        --out outputs/11_12_2026_output/gnn_v2_freeze/checkpoint -d 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gnn_fullmu_campaign as v1camp  # noqa: E402

from gnn.diagnostics.plot_style import save_figure  # noqa: E402
from gnn.evaluation.chain_inference import load_gnn, sha256  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = {
    "v1": (
        REPO / "outputs/09_24_2026_output/gnn_fullmu_v1/learning_curve.jsonl",
        REPO / "model_weights/ttva_gnn_hllhc_fullmu_v1_20260924/"
        "ttva_gat_alleta_k20_fullmu_v1_60k_epoch_{}.pyt",
        60000,
    ),
    "v2": (
        REPO / "outputs/11_11_2026_output/gnn_fullmu_v2/learning_curve.jsonl",
        REPO / "model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/"
        "ttva_gat_alleta_k20_fullmu_v2_150k_epoch_{}.pyt",
        149820,
    ),
}
MONITOR = (
    REPO / "data/run4_all_etas/ttva_graphs/fullmu_v1_20260924/validation_monitor.pt"
)
N_PASSES = 18


def curve(path: Path) -> dict[str, np.ndarray]:
    """Learning-curve columns as arrays."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def plateau(c: dict[str, np.ndarray]) -> dict:
    """Best epoch, per-pass mean losses and the last pass-to-pass changes."""
    v = c["validation_loss"]
    per = len(v) // N_PASSES
    pm = [float(v[i * per : (i + 1) * per].mean()) for i in range(N_PASSES)]
    tm = [float(c["training_loss"][i * per : (i + 1) * per].mean()) for i in range(N_PASSES)]  # fmt: skip
    best = int(np.argmin(v))
    last_pass = v[-per:]
    return {
        "best_epoch": int(c["shard_epoch"][best]),
        "best_val": float(v[best]),
        "last_epoch": int(c["shard_epoch"][-1]),
        "last_val": float(v[-1]),
        "per_pass_val": pm,
        "per_pass_train": tm,
        "last4_pass_deltas": np.diff(pm)[-4:].tolist(),
        "last_pass_val_std": float(last_pass.std(ddof=1)),
        "best_minus_last": float(v[best] - v[-1]),
        "rel_drop_last_pass": float((pm[-1] - pm[-2]) / pm[-2]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-d", "--device", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    curves = {k: curve(p) for k, (p, _, _) in RUNS.items()}
    res: dict = {k: plateau(c) for k, c in curves.items()}

    from torch_geometric.loader import DataLoader

    graphs = torch.load(MONITOR, weights_only=False)
    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    device = torch.device(f"cuda:{a.device}")
    picks = {
        "v1_e213": str(RUNS["v1"][1]).format(213),
        f"v2_e{res['v2']['best_epoch']}": str(RUNS["v2"][1]).format(
            res["v2"]["best_epoch"]
        ),
        f"v2_e{res['v2']['last_epoch']}": str(RUNS["v2"][1]).format(
            res["v2"]["last_epoch"]
        ),
    }
    res["truth_graph_monitor"] = {}
    for name, path in picks.items():
        tv = v1camp.validation_tracks(load_gnn(str(path), device), loader)
        cnt = np.array(tv["counts"])  # (mu bin, cut, [correct, input, assigned])
        tot = cnt.sum(0)
        eff = tot[:, 0] / tot[:, 1]
        pur = tot[:, 0] / tot[:, 2]
        res["truth_graph_monitor"][name] = {
            "checkpoint": str(path), "sha256": sha256(Path(path)),
            "cuts": tv["cuts"], "eff_all_mu": eff.tolist(), "pur_all_mu": pur.tolist(),
            "f1_all_mu": (2 * eff * pur / (eff + pur)).tolist(),
            "eff_mu0_20": (cnt[:2].sum(0)[:, 0] / cnt[:2].sum(0)[:, 1]).tolist(),
            "eff_mu180_200": (cnt[18:].sum(0)[:, 0] / cnt[18:].sum(0)[:, 1]).tolist(),
            "counts": tv["counts"],
        }  # fmt: skip
    res["monitor"] = {"path": str(MONITOR), "events": len(graphs)}
    (a.out / "checkpoint_selection.json").write_text(json.dumps(res, indent=2) + "\n")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, color in (("v1", "#999999"), ("v2", "#0072B2")):
        c, n_train = curves[k], RUNS[k][2]
        x = (c["shard_epoch"] + 1) * (n_train / len(c["shard_epoch"]) * N_PASSES) / 1e6
        for ax in axes:
            ax.plot(
                x, c["validation_loss"], color=color, lw=1.5, label=f"{k} validation"
            )
            ax.plot(
                x, c["training_loss"], color=color, lw=1, ls="--", label=f"{k} training"
            )
        b = res[k]["best_epoch"]
        axes[1].plot(x[b], c["validation_loss"][b], "o", color=color, ms=8)
    axes[0].set_ylim(0.24, 0.40)
    axes[1].set_xlim(1.5, 2.75)
    axes[1].set_ylim(0.253, 0.262)
    for ax in axes:
        ax.set_xlabel("event presentations [millions]")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].set_title(
        f"v2 end: best epoch {res['v2']['best_epoch']} (dot)", fontsize=10
    )
    save_figure(fig, a.out, "learning_curve_v1_v2")
    print(
        json.dumps(
            {k: v for k, v in res.items() if k != "truth_graph_monitor"}, indent=1
        )
    )
    for name, r in res["truth_graph_monitor"].items():
        print(name, "F1", np.round(r["f1_all_mu"], 4).tolist())


if __name__ == "__main__":
    main()
