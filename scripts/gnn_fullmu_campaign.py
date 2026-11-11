"""Reproducible, isolated full-pileup GNN campaign (September 2026).

The first 25,000 entries used in the presentation are never training data.
This script only writes into the new campaign directories.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import awkward as ak
import numpy as np
import torch
import uproot
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/09_24_2026_output/gnn_fullmu_v1"
DATA = ROOT / "data/run4_all_etas/ttva_graphs/fullmu_v1_20260924"
SOURCE = (
    ROOT
    / "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root"
)
CONFIG = ROOT / "configs/gnn/config_gnn_ttva_hllhc_fullmu_v1.yml"
WEIGHTS = ROOT / "model_weights/ttva_gnn_hllhc_fullmu_v1_20260924"
CACHE = Path("/dev/shm/matmauro_gnn_fullmu_v1_20260924")
SEED = 20260924


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def status(stage, **kwargs):
    record = {
        "stage": stage,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **kwargs,
    }
    write_json(OUT / "status.json", record)
    print(json.dumps(record), flush=True)


def plan():
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "build_logs").mkdir(exist_ok=True)
    if (OUT / "split_manifest.json").exists():
        return json.loads((OUT / "split_manifest.json").read_text())
    tree = uproot.open(SOURCE)["PVFinderData"]
    arrays = tree.arrays(["ActualNumOfInt", "TruthVertex_z"])
    mu = ak.to_numpy(arrays.ActualNumOfInt)
    assert len(mu) == 100000 and mu.min() == 0 and mu.max() == 200
    fingerprints = [
        hashlib.blake2b(
            np.asarray(z, dtype=np.float32).tobytes(), digest_size=16
        ).hexdigest()
        for z in ak.to_list(arrays.TruthVertex_z)
    ]
    assert len(set(fingerprints)) == len(fingerprints), (
        "Duplicate complete truth-z event fingerprints"
    )
    rng = np.random.default_rng(SEED)
    # Split by 10-unit pileup bin, adjusting rounding to exact 80/10/10 totals.
    # Every event stays within this single ROOT production, so no reconstruction
    # tags or alternative productions of the same collision are mixed.
    groups = []
    for bin_id in range(20):
        entries = np.flatnonzero(
            (np.arange(len(mu)) >= 25000)
            & (mu >= 10 * bin_id)
            & ((mu < 10 * (bin_id + 1)) if bin_id < 19 else (mu <= 200))
        )
        rng.shuffle(entries)
        groups.append(entries)
    counts = np.array([len(g) for g in groups])
    nval = (counts / 10).astype(int)
    ntest = nval.copy()
    order = np.argsort(-(counts / 10 - nval), kind="stable")
    nval[order[: 7500 - nval.sum()]] += 1
    ntest[order[: 7500 - ntest.sum()]] += 1
    splits = {"train": [], "validation": [], "test": []}
    for g, nv, nt in zip(groups, nval, ntest):
        splits["validation"].extend(g[:nv])
        splits["test"].extend(g[nv : nv + nt])
        splits["train"].extend(g[nv + nt :])
    splits = {k: np.sort(np.asarray(v, dtype=np.int64)) for k, v in splits.items()}
    assert [len(splits[k]) for k in ["train", "validation", "test"]] == [
        60000,
        7500,
        7500,
    ]
    assert len(set(np.concatenate(list(splits.values())))) == 75000
    assert np.concatenate(list(splits.values())).min() >= 25000
    train_order = splits["train"].copy()
    rng.shuffle(train_order)
    for i, indices in enumerate(np.split(train_order, 12)):
        np.save(DATA / f"train_shard_{i:02d}.entry_indices.npy", np.sort(indices))
    # Fixed, stratified validation monitor. The remaining validation events are
    # reserved for full-chain checkpoint/WP selection, not used for gradients.
    monitor = []
    for bin_id in range(20):
        v = splits["validation"]
        v = v[
            (mu[v] >= 10 * bin_id)
            & ((mu[v] < 10 * (bin_id + 1)) if bin_id < 19 else (mu[v] <= 200))
        ]
        rng.shuffle(v)
        monitor.extend(v[:75])
    monitor = np.sort(np.asarray(monitor, dtype=np.int64))
    assert len(monitor) == 1500
    np.save(DATA / "validation_monitor.entry_indices.npy", monitor)
    for key, indices in splits.items():
        np.save(DATA / f"{key}_entries.npy", indices)
    np.save(DATA / "diagnostic_deck_entries.npy", np.arange(25000, dtype=np.int64))
    hist_edges = list(range(0, 200, 10)) + [201]
    source_hash = hashlib.sha256()
    with SOURCE.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            source_hash.update(block)
    result = {
        "source": str(SOURCE),
        "source_sha256": source_hash.hexdigest(),
        "source_size": SOURCE.stat().st_size,
        "source_entries": len(mu),
        "split_seed": SEED,
        "training_seed": 42,
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "histogram_edges": hist_edges,
        "splits": {
            k: {
                "events": len(v),
                "mu_histogram": np.histogram(mu[v], hist_edges)[0].tolist(),
                "entries_sha256": hashlib.sha256(v.tobytes()).hexdigest(),
            }
            for k, v in splits.items()
        },
        "diagnostic_entries": [0, 25000],
        "validation_monitor_events": len(monitor),
        "checks": {
            "entry_disjoint": True,
            "all_100000_truth_z_fingerprints_unique": True,
            "no_training_entry_below_25000": True,
        },
        "caveat": "No run/event identifiers are stored. Disjoint entries of one ROOT production and unique full truth-z fingerprints are checked. Do not mix alternate reconstruction tags.",
    }
    write_json(OUT / "split_manifest.json", result)
    return result


def build(name):
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
            print(
                f"{name}: {len(graphs)}/{len(indices)} graphs, {time.monotonic() - start:.1f} s",
                flush=True,
            )
    assert len(graphs) == len(indices)
    temporary = output.with_suffix(".pt.tmp")
    torch.save(graphs, temporary)
    temporary.replace(output)
    record = {
        "graphs": len(graphs),
        "seconds": time.monotonic() - start,
        "bytes": output.stat().st_size,
    }
    write_json(OUT / "build_logs" / f"{name}.json", record)
    print(json.dumps(record), flush=True)


def prepare():
    manifest = plan()
    status("preparing", events=manifest["splits"]["train"]["events"], workers=2)
    names = ["validation_monitor"] + [f"train_shard_{i:02d}" for i in range(12)]

    def run(name):
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
    status("prepared", training_shards=12, training_events=60000, monitor_events=1500)


def train():
    from gnn.training import train_ttva

    config = yaml.safe_load(CONFIG.read_text())
    assert all(Path(p).exists() for p in config["data_files"])
    # The original graph files remain on disk. Read byte-identical temporary
    # copies from tmpfs to avoid repeated tiny disk reads in torch.load.
    for p in config["data_files"] + [config["val_file"]]:
        cached = CACHE / Path(p).name
        assert cached.exists() and cached.stat().st_size == Path(p).stat().st_size, (
            cached
        )
    config["data_files"] = [str(CACHE / Path(p).name) for p in config["data_files"]]
    config["val_file"] = str(CACHE / Path(config["val_file"]).name)
    runtime_config = OUT / "runtime_config.yml"
    runtime_config.write_text(yaml.safe_dump(config))
    config["config_file"] = str(runtime_config)
    if WEIGHTS.exists() and list(WEIGHTS.glob("*.pyt")):
        raise RuntimeError(
            "Checkpoints already exist; refusing to overwrite a previous run"
        )
    torch.set_num_threads(2)
    write_json(
        OUT / "training_provenance.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "runtime_graph_cache": str(CACHE),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "code_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__).resolve(),
                    CONFIG,
                    ROOT / "src/gnn/training/train_ttva.py",
                    ROOT / "src/gnn/training/training_loop.py",
                    ROOT / "src/gnn/models/ttva_gat.py",
                    ROOT / "src/gnn/data/graph_construction.py",
                ]
            },
        },
    )
    status(
        "training",
        pid=os.getpid(),
        physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
        shard_epochs=216,
        complete_passes=18,
    )
    original_loop = train_ttva.trainNet
    started = time.monotonic()
    durations = []

    def observed_loop(*args, **kwargs):
        for result in original_loop(*args, **kwargs):
            if not np.isfinite(result.cost) or not np.isfinite(result.val):
                raise RuntimeError(f"Non-finite loss at shard epoch {result.epoch}")
            durations.append(result.time)
            record = {
                "shard_epoch": result.epoch,
                "completed_passes": (result.epoch + 1) / 12,
                "training_loss": result.cost,
                "validation_loss": result.val,
                "seconds": result.time,
                "elapsed_seconds": time.monotonic() - started,
                "remaining_seconds_estimate": float(
                    np.median(durations[-12:]) * (config["n_epochs"] - result.epoch - 1)
                ),
            }
            with (OUT / "learning_curve.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if (result.epoch + 1) % 12 == 0:
                detailed = {
                    **record,
                    "truth_graph_validation": validation_tracks(args[0], args[3]),
                }
                write_json(
                    OUT / f"validation_pass_{(result.epoch + 1) // 12:02d}.json",
                    detailed,
                )
            status(
                "training",
                pid=os.getpid(),
                physical_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                **record,
            )
            yield result

    train_ttva.trainNet = observed_loop
    # Isolate the standard trainer's run_stats.pyt and other relative output.
    os.chdir(OUT)
    train_ttva.main(config)
    status("training_complete", checkpoint_folder=str(WEIGHTS))
    # Only this campaign's known copies are removed; persistent data remain.
    for path in [Path(p) for p in config["data_files"] + [config["val_file"]]]:
        path.unlink()
    CACHE.rmdir()


def cache():
    """Stage the fixed graph files in a bounded, temporary RAM directory."""
    paths = sorted(DATA.glob("*.pt"))
    assert len(paths) == 13
    missing = [p for p in paths if not (CACHE / p.name).exists()]
    required = sum(p.stat().st_size for p in missing)
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    if (
        shutil.disk_usage("/dev/shm").free < required + 10 * 1024**3
        or available < required + 64 * 1024**3
    ):
        raise RuntimeError("Insufficient headroom for this campaign's RAM cache")
    CACHE.mkdir(mode=0o700, exist_ok=True)
    status(
        "staging_graph_cache", required_bytes=required, available_memory_bytes=available
    )
    started = time.monotonic()
    for p in paths:
        target = CACHE / p.name
        if target.exists():
            assert target.stat().st_size == p.stat().st_size
            continue
        temporary = target.with_suffix(".pt.tmp")
        shutil.copyfile(p, temporary)
        assert temporary.stat().st_size == p.stat().st_size
        temporary.replace(target)
        print(
            f"Cached {p.name}: {p.stat().st_size / 1024**3:.2f} GiB, elapsed {time.monotonic() - started:.1f} s",
            flush=True,
        )
    write_json(
        OUT / "graph_cache.json",
        {
            "path": str(CACHE),
            "bytes": sum(p.stat().st_size for p in paths),
            "copy_seconds": time.monotonic() - started,
        },
    )
    status("graph_cache_ready", seconds=time.monotonic() - started)


def validation_tracks(model, loader):
    """Track metrics on *truth-PV candidate* graphs, not full-chain results.

    Correctness uses the known candidate truth-PV identity directly. This is
    intentionally more direct than the deck's dominant-origin labelling and
    must not be presented as an AMVF comparison or PV-Finder-chain efficiency.
    """
    cuts = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    counts = np.zeros((20, len(cuts), 3), dtype=np.int64)  # correct, input, assigned
    mu = uproot.open(SOURCE)["PVFinderData"]["ActualNumOfInt"].array(library="np")
    entries = np.load(DATA / "validation_monitor.entry_indices.npy")
    bins = np.minimum((mu[entries] // 10).astype(int), 19)
    cursor = 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            scores = torch.sigmoid(model(batch)).flatten().cpu().numpy()
            edge = batch["track", "to", "pv"].edge_index.cpu().numpy()
            ntracks = batch["track"].num_nodes
            npvs = batch["pv"].num_nodes
            truth = batch["track"].truth_pv.cpu().numpy()
            tbatch = batch["track"].batch.cpu().numpy()
            pvptr = batch["pv"].ptr.cpu().numpy()
            labelled = truth >= 0
            truth = np.where(labelled, truth + pvptr[tbatch], -1)
            trackbin = bins[cursor : cursor + batch.num_graphs][tbatch]
            cursor += batch.num_graphs
            unique, starts = np.unique(edge[0], return_index=True)
            bestscore = np.full(ntracks, -np.inf)
            bestscore[unique] = np.maximum.reduceat(scores, starts)
            winners = np.minimum.reduceat(
                np.where(
                    scores == bestscore[edge[0]], np.arange(len(scores)), len(scores)
                ),
                starts,
            )
            bestpv = np.full(ntracks, -1, dtype=np.int64)
            bestpv[unique] = edge[1, winners]
            for ci, cut in enumerate(cuts):
                assigned = (bestpv >= 0) & (bestscore >= cut)
                multiplicity = np.bincount(bestpv[assigned], minlength=npvs)
                assigned &= multiplicity[np.maximum(bestpv, 0)] >= 2
                correct = assigned & labelled & (bestpv == truth)
                counts[:, ci, 0] += np.bincount(trackbin[correct], minlength=20)
                counts[:, ci, 1] += np.bincount(trackbin[labelled], minlength=20)
                counts[:, ci, 2] += np.bincount(
                    trackbin[assigned & labelled], minlength=20
                )
    assert cursor == len(entries)
    rates = {}
    for name, numerator, denominator in [
        ("track_efficiency", 0, 1),
        ("assignment_purity", 0, 2),
        ("assigned_fraction", 2, 1),
    ]:
        rates[name] = np.divide(
            counts[:, :, numerator],
            counts[:, :, denominator],
            out=np.zeros(counts.shape[:2]),
            where=counts[:, :, denominator] > 0,
        ).tolist()
    return {
        "candidate_source": "truth vertices; not PV-Finder peaks",
        "reco_min_tracks": 2,
        "correct_definition": "winning candidate truth ID equals track truth ID",
        "cuts": cuts,
        "mu_edges": list(range(0, 200, 10)) + [201],
        "count_columns": ["correct", "input_truth_labelled", "assigned_truth_labelled"],
        "counts": counts.tolist(),
        "rates": rates,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=["plan", "prepare", "build", "cache", "train"]
    )
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
