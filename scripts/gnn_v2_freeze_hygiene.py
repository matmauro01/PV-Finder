#!/usr/bin/env python3
"""Data-hygiene audit for the GNN v2 freeze (step 2b, 12 November 2026).

Answers, from the saved split files and the trained-on shard list only:

1. Which r16443 entries GNN v2 trained on (the v1 shards listed in the
   training ``runtime_config.yml``), and that none of them is in the GNN
   validation split, the reserved test split or entries 0-24,999.
2. That the full-chain test set (r16443 entries 0-24,999) and the t-scan set
   (the GNN validation split) are disjoint from GNN v2 training and from each
   other.
3. That no r16443 hard-scatter truth vertex (x, y, z) or full truth-z
   fingerprint occurs in the density-model training files r16438 / r16633
   (same generator and simulation tags), and none in r16638 (re-checked).

Output: ``hygiene.json`` in ``--out``.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_hygiene.py \
        --out outputs/11_12_2026_output/gnn_v2_freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnn_fullmu_v2_campaign import _fingerprints  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
V1 = REPO / "data/run4_all_etas/ttva_graphs/fullmu_v1_20260924"
V2 = REPO / "data/run4_all_etas/ttva_graphs/fullmu_v2_20261111"
RUNTIME = REPO / "outputs/11_11_2026_output/gnn_fullmu_v2/runtime_config.yml"
RDIR = REPO / "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026"
ROOTS = {
    tag: RDIR / f"ATLAS_PVFinderData_601229_e8481_s4494_{tag}_PU200.root"
    for tag in ("r16443", "r16638", "r16438", "r16633")
}


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(a, dtype=np.int64).tobytes()).hexdigest()


def trained_entries(runtime: Path) -> dict[str, np.ndarray]:
    """Entries of every shard GNN v2 was trained on, grouped by source file."""
    files = yaml.safe_load(runtime.read_text())["data_files"]
    out: dict[str, list[np.ndarray]] = {"r16443": [], "r16638": []}
    for f in files:
        p = Path(f)
        idx = np.load(p.with_suffix(".entry_indices.npy"))
        key = "r16638" if p.name.startswith("r16638_") else "r16443"
        if key == "r16443" and p.parent != V1:
            raise ValueError(f"unexpected r16443 shard location {p}")
        out[key].append(idx)
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--skip-fingerprints", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tr = trained_entries(RUNTIME)
    v1_train = np.load(V1 / "train_entries.npy")
    val = np.load(V1 / "validation_entries.npy")
    test_v1 = np.load(V1 / "test_entries.npy")
    monitor = np.load(V1 / "validation_monitor.entry_indices.npy")
    diag = np.arange(25000)
    r38_train = np.load(V2 / "r16638_train_entries.npy")
    r38_res = np.load(V2 / "r16638_reserved_test_entries.npy")

    t43 = tr["r16443"]
    res: dict = {
        "runtime_config": str(RUNTIME),
        "n_shards": {"r16443": 12, "r16638": 18},
        "trained_r16443_events": int(len(t43)),
        "trained_r16443_unique": int(len(np.unique(t43))),
        "trained_r16443_equals_v1_train_entries": bool(
            np.array_equal(np.sort(t43), v1_train)
        ),
        "trained_r16443_range": [int(t43.min()), int(t43.max())],
        "trained_r16638_events": int(len(tr["r16638"])),
        "trained_r16638_equals_split_train": bool(
            np.array_equal(np.sort(tr["r16638"]), r38_train)
        ),
        "overlaps": {
            "trained_r16443 & entries_0_24999": int(len(np.intersect1d(t43, diag))),
            "trained_r16443 & gnn_validation": int(len(np.intersect1d(t43, val))),
            "trained_r16443 & gnn_reserved_test": int(
                len(np.intersect1d(t43, test_v1))
            ),
            "gnn_validation & entries_0_24999": int(len(np.intersect1d(val, diag))),
            "gnn_validation & gnn_reserved_test": int(
                len(np.intersect1d(val, test_v1))
            ),
            "monitor_subset_of_validation": bool(np.isin(monitor, val).all()),
            "trained_r16638 & r16638_reserved": int(
                len(np.intersect1d(tr["r16638"], r38_res))
            ),
        },
        "sets": {
            "chain_test_r16443_0_24999": {"n": 25000, "sha256": _sha(diag)},
            "t_scan_gnn_validation": {"n": int(len(val)), "sha256": _sha(val)},
            "gnn_reserved_test_untouched": {
                "n": int(len(test_v1)),
                "sha256": _sha(test_v1),
            },
        },
    }
    if not a.skip_fingerprints:
        fp = {tag: _fingerprints(path) for tag, path in ROOTS.items()}
        _, p43, hs43 = fp["r16443"]
        res["fingerprints"] = {
            tag: {
                "entries": int(len(fp[tag][0])),
                "truth_z_fingerprint_overlap_with_r16443": int(
                    len(set(fp[tag][1]) & set(p43))
                ),
                "hs_xyz_overlap_with_r16443": int(len(fp[tag][2] & hs43)),
            }
            for tag in ("r16638", "r16438", "r16633")
        }
        res["fingerprints"]["r16443_unique_truth_z"] = len(set(p43)) == len(p43)
    (a.out / "hygiene.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
