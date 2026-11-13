#!/usr/bin/env python3
"""Write a GNN v3 training config from shard sets (phase 2 and later).

Shard sets: ``base`` (the 30 phase-1 s8 shards, 149,820 flat-mu events, in
phase-1 order), ``thin`` (20 r16438 PU200 shards thinned to flat mu, 99,800
events), ``pu200`` (20 r16633 PU200 shards, 99,800 events). Sets are
interleaved proportionally, so every stretch of the round-robin mixes them.

    PYTHONPATH=src venv/bin/python scripts/gnn_v3_make_config.py --run p2a --gpu 1 \
        --sets base thin pu200 --epochs 1080 --model "{arch: ign, hidden: 64, ...}" \
        --comment "data volume: ..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
G = REPO / "data/run4_all_etas/ttva_graphs/chain_s8_v3_20261113"
MONITOR = G / "validation_monitor.pt"


def shard_set(name: str) -> list[str]:
    """Absolute shard paths of one set, in their canonical order."""
    if name == "base":
        cfg = yaml.safe_load(
            (REPO / "configs/gnn/config_gnn_ttva_v3_b.yml").read_text()
        )
        return list(cfg["data_files"])
    pattern = {"thin": "r16438_thin_shard_{:02d}.pt", "pu200": "r16633_pu200_shard_{:02d}.pt"}[name]  # fmt: skip
    return [str(G / "phase2" / pattern.format(i)) for i in range(20)]


def interleave(sets: list[list[str]]) -> list[str]:
    """Merge lists so each appears at its own uniform rate."""
    keyed = [
        ((k + 0.5) / len(s), j, p) for j, s in enumerate(sets) for k, p in enumerate(s)
    ]
    return [p for _, _, p in sorted(keyed)]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument(
        "--sets", required=True, nargs="+", choices=["base", "thin", "pu200"]
    )
    ap.add_argument("--epochs", required=True, type=int)
    ap.add_argument(
        "--model", required=True, help="model_config as a YAML flow mapping"
    )
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--comment", required=True)
    a = ap.parse_args()
    files = interleave([shard_set(s) for s in a.sets])
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} shards missing, e.g. {missing[0]}")
    cfg = {
        "data_files": files, "val_file": str(MONITOR), "monitor_file": str(MONITOR),
        "batch_size": 32, "num_workers": 2, "model_config": yaml.safe_load(a.model),
        "n_epochs": a.epochs, "learning_rate": a.lr, "warmup_epochs": a.warmup,
        "eta_min": 1e-5, "weight_decay": 0.0, "grad_clip": 1.0, "seed": 20261113,
        "device_id": a.gpu,
        "save_folder": str(REPO / f"model_weights/ttva_gnn_v3_20261113/{a.run}"),
        "runname": f"ttva_v3_{a.run}",
    }  # fmt: skip
    head = [
        f"# GNN v3 campaign, run {a.run} (docs/evaluation/gnn_v3_campaign_2026-11-13.md).",
        *(f"# {line}" for line in a.comment.split("\n")),
        f"# Shard sets: {', '.join(a.sets)} ({len(files)} shards, interleaved); "
        f"{a.epochs} shard epochs = {a.epochs / len(files):.1f} passes.",
        "# Written by scripts/gnn_v3_make_config.py.",
    ]
    out = REPO / f"configs/gnn/config_gnn_ttva_v3_{a.run}.yml"
    out.write_text("\n".join(head) + "\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out}: {len(files)} shards")


if __name__ == "__main__":
    main()
