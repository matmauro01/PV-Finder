"""Full-pileup GNN v2: the v1 recipe on 2.5 times more training events.

v1 (``scripts/gnn_fullmu_campaign.py``) trained on 60,000 r16443 events.
v2 keeps those 12 training shards, v1's validation monitor and v1's
validation and reserved-test entries unchanged. It adds training events from
r16638, an independent production of the same flat-mu sample: none of its
hard-scatter vertices (x, y, z), nor any full truth-z fingerprint, occurs in
r16443. 10% of r16638 is reserved as an extra test set and never built.

The graphs use truth-vertex candidates, as in v1, so no PV-Finder density
model enters the training data.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gnn_fullmu_campaign as v1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/11_11_2026_output/gnn_fullmu_v2"
DATA = ROOT / "data/run4_all_etas/ttva_graphs/fullmu_v2_20261111"
SOURCE = (
    ROOT
    / "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16638_PU200.root"
)
CONFIG = ROOT / "configs/gnn/config_gnn_ttva_hllhc_fullmu_v2.yml"
WEIGHTS = ROOT / "model_weights/ttva_gnn_hllhc_fullmu_v2_20261111"
SEED = 20261111
N_NEW_SHARDS = 18
EDGES = list(range(0, 200, 10)) + [201]


def status(stage: str, **kwargs: object) -> None:
    record = {
        "stage": stage,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **kwargs,
    }
    v1.write_json(OUT / "status.json", record)
    print(json.dumps(record), flush=True)


def _fingerprints(path: Path) -> tuple[np.ndarray, list[str], set[tuple]]:
    """mu, full truth-z fingerprints and hard-scatter (x, y, z) of one file."""
    arrays = uproot.open(path)["PVFinderData"].arrays(
        [
            "ActualNumOfInt",
            "TruthVertex_x",
            "TruthVertex_y",
            "TruthVertex_z",
            "TruthVertex_type",
        ]
    )
    mu = ak.to_numpy(arrays.ActualNumOfInt)
    prints = [
        hashlib.blake2b(
            np.asarray(z, dtype=np.float32).tobytes(), digest_size=16
        ).hexdigest()
        for z in ak.to_list(arrays.TruthVertex_z)
    ]
    hs = arrays.TruthVertex_type == 1
    xyz = np.stack(
        [
            np.asarray(
                ak.to_numpy(
                    ak.fill_none(ak.firsts(arrays[f"TruthVertex_{c}"][hs]), np.nan)
                ),
                dtype=np.float64,
            )
            for c in "xyz"
        ],
        1,
    )
    hs_set = {tuple(v) for v in xyz[~np.isnan(xyz).any(1)]}
    return mu, prints, hs_set


def plan() -> dict:
    """Split r16638 (reserved test / new training shards) and record checks."""
    for d in (DATA, OUT, OUT / "build_logs"):
        d.mkdir(parents=True, exist_ok=True)
    if (OUT / "split_manifest.json").exists():
        return json.loads((OUT / "split_manifest.json").read_text())
    mu, prints, hs_new = _fingerprints(SOURCE)
    _, prints_v1, hs_v1 = _fingerprints(v1.SOURCE)
    n = len(mu)
    assert mu.min() == 0 and mu.max() == 200
    assert len(set(prints)) == n, "duplicate truth-z fingerprints in r16638"
    overlap_prints = len(set(prints) & set(prints_v1))
    overlap_hs = len(hs_new & hs_v1)
    assert overlap_prints == 0 and overlap_hs == 0, (overlap_prints, overlap_hs)

    rng = np.random.default_rng(SEED)
    groups = []
    for b in range(20):
        sel = (mu >= EDGES[b]) & (mu < EDGES[b + 1])
        entries = np.flatnonzero(sel)
        rng.shuffle(entries)
        groups.append(entries)
    counts = np.array([len(g) for g in groups])
    n_test_total = n // 10
    ntest = (counts / 10).astype(int)
    order = np.argsort(-(counts / 10 - ntest), kind="stable")
    ntest[order[: n_test_total - ntest.sum()]] += 1
    test = np.sort(np.concatenate([g[:k] for g, k in zip(groups, ntest)]))
    train = np.sort(np.concatenate([g[k:] for g, k in zip(groups, ntest)]))
    assert len(test) == n_test_total and len(test) + len(train) == n
    assert len(np.intersect1d(test, train)) == 0

    train_order = train.copy()
    rng.shuffle(train_order)
    shards = np.array_split(train_order, N_NEW_SHARDS)
    for i, idx in enumerate(shards):
        np.save(DATA / f"r16638_train_shard_{i:02d}.entry_indices.npy", np.sort(idx))
    np.save(DATA / "r16638_train_entries.npy", train)
    np.save(DATA / "r16638_reserved_test_entries.npy", test)

    h = hashlib.sha256()
    with SOURCE.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            h.update(block)
    v1_manifest = json.loads((v1.OUT / "split_manifest.json").read_text())
    result = {
        "v1_manifest": str(v1.OUT / "split_manifest.json"),
        "v1_unchanged": {
            "train_events": v1_manifest["splits"]["train"]["events"],
            "validation_events": v1_manifest["splits"]["validation"]["events"],
            "test_events": v1_manifest["splits"]["test"]["events"],
            "validation_monitor_events": v1_manifest["validation_monitor_events"],
            "source_sha256": v1_manifest["source_sha256"],
        },
        "new_source": str(SOURCE),
        "new_source_sha256": h.hexdigest(),
        "new_source_entries": n,
        "split_seed": SEED,
        "histogram_edges": EDGES,
        "new_splits": {
            name: {
                "events": len(v),
                "mu_histogram": np.histogram(mu[v], EDGES)[0].tolist(),
                "entries_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            }
            for name, v in {"train": train, "reserved_test": test}.items()
        },
        "new_train_shards": [len(s) for s in shards],
        "total_train_events": v1_manifest["splits"]["train"]["events"] + len(train),
        "checks": {
            "r16638_truth_z_fingerprints_unique": True,
            "truth_z_fingerprint_overlap_with_r16443": overlap_prints,
            "hard_scatter_xyz_overlap_with_r16443": overlap_hs,
            "reserved_test_train_disjoint": True,
        },
        "candidates": "truth vertices (no PV-Finder density model in the graphs)",
    }
    v1.write_json(OUT / "split_manifest.json", result)
    return result


def build(name: str) -> None:
    """Build one r16638 shard with v1's graph builder and settings."""
    from gnn.data.event_selection import iterate_entries
    from gnn.data.root_to_graphs import (
        TRACK_BRANCHES,
        TRUTH_BRANCHES,
        build_graph_from_event,
    )
    from pv_finder.data.resolution_presets import RESOLUTION_PRESETS

    start = time.monotonic()
    indices = np.load(DATA / f"{name}.entry_indices.npy")
    output = DATA / f"{name}.pt"
    if output.exists():
        print(f"Reusing completed {output}", flush=True)
        return
    torch.set_num_threads(1)
    tree = uproot.open(SOURCE)["PVFinderData"]
    graphs = []
    for event in iterate_entries(
        tree, TRACK_BRANCHES + TRUTH_BRANCHES, indices, step_size=500
    ):
        graph = build_graph_from_event(event, 20, RESOLUTION_PRESETS["hllhc_alleta"])
        if not torch.isfinite(graph["track"].x).all():
            raise ValueError("Non-finite track features")
        graphs.append(graph)
        if len(graphs) % 500 == 0:
            print(f"{name}: {len(graphs)}/{len(indices)}", flush=True)
    assert len(graphs) == len(indices)
    tmp = output.with_suffix(".pt.tmp")
    torch.save(graphs, tmp)
    tmp.replace(output)
    v1.write_json(
        OUT / "build_logs" / f"{name}.json",
        {
            "graphs": len(graphs),
            "seconds": time.monotonic() - start,
            "bytes": output.stat().st_size,
        },
    )


def prepare() -> None:
    manifest = plan()
    names = [f"r16638_train_shard_{i:02d}" for i in range(N_NEW_SHARDS)]
    status("preparing", new_train_events=manifest["new_splits"]["train"]["events"])

    def run(name: str) -> None:
        with (OUT / "build_logs" / f"{name}.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "build",
                    "--name",
                    name,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(f"Prepared {name}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, names))
    status("prepared", new_train_shards=N_NEW_SHARDS)


def train() -> None:
    from gnn.training import train_ttva

    config = yaml.safe_load(CONFIG.read_text())
    assert all(Path(p).exists() for p in config["data_files"] + [config["val_file"]])
    n_shards = len(config["data_files"])
    if WEIGHTS.exists() and list(WEIGHTS.glob("*.pyt")):
        raise RuntimeError("Checkpoints already exist; refusing to overwrite")
    runtime_config = OUT / "runtime_config.yml"
    runtime_config.write_text(yaml.safe_dump(config))
    config["config_file"] = str(runtime_config)
    torch.set_num_threads(2)
    v1.write_json(
        OUT / "training_provenance.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "code_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__).resolve(),
                    Path(v1.__file__).resolve(),
                    CONFIG,
                    ROOT / "src/gnn/training/train_ttva.py",
                    ROOT / "src/gnn/training/training_loop.py",
                    ROOT / "src/gnn/training/shard_loader.py",
                    ROOT / "src/gnn/models/ttva_gat.py",
                    ROOT / "src/gnn/data/graph_construction.py",
                ]
            },
        },
    )
    status(
        "training", pid=os.getpid(), shard_epochs=config["n_epochs"], shards=n_shards
    )
    original_loop = train_ttva.trainNet
    started = time.monotonic()
    durations: list[float] = []

    def observed_loop(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        for result in original_loop(*args, **kwargs):
            if not np.isfinite(result.cost) or not np.isfinite(result.val):
                raise RuntimeError(f"Non-finite loss at shard epoch {result.epoch}")
            durations.append(result.time)
            record = {
                "shard_epoch": result.epoch,
                "completed_passes": (result.epoch + 1) / n_shards,
                "training_loss": result.cost,
                "validation_loss": result.val,
                "seconds": result.time,
                "elapsed_seconds": time.monotonic() - started,
                "remaining_seconds_estimate": float(
                    np.median(durations[-n_shards:])
                    * (config["n_epochs"] - result.epoch - 1)
                ),
            }
            with (OUT / "learning_curve.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if (result.epoch + 1) % n_shards == 0:
                detailed = {
                    **record,
                    "truth_graph_validation": v1.validation_tracks(args[0], args[3]),
                }
                v1.write_json(
                    OUT / f"validation_pass_{(result.epoch + 1) // n_shards:02d}.json",
                    detailed,
                )
            status("training", pid=os.getpid(), **record)
            yield result

    train_ttva.trainNet = observed_loop
    os.chdir(OUT)  # keep the trainer's run_stats.pyt inside the campaign folder
    train_ttva.main(config)
    status("training_complete", checkpoint_folder=str(WEIGHTS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "prepare", "build", "train"])
    parser.add_argument("--name")
    args = parser.parse_args()
    try:
        if args.action == "build":
            build(args.name)
        elif args.action == "plan":
            print(json.dumps(plan(), indent=2))
        else:
            globals()[args.action]()
    except Exception as exc:
        if OUT.exists():
            status("failed", action=args.action, error=str(exc))
        raise
