"""Extract AMVF track associations for the cached Run 3 events.

The Run 3 NPZ cache (data/run3/cache_file3_2000ev_seed42.npz) stores tracks
and AMVF vertex z/nTracks but not RecoVertex_assocTracks. This script reads
that branch from the source ROOT file for exactly the cached events
(matched via the cache's ``event_indices``) and saves a companion NPZ,
row-aligned with the cache.

Alignment is verified by comparing RecoVertex_z per event between ROOT and
cache; any mismatch aborts.

Usage:
    python -u -m gnn.data.run3_assoc_cache \\
        --root data/run3/file_3.root \\
        --cache data/run3/cache_file3_2000ev_seed42.npz \\
        --output data/run3/assoc_cache_file3_2000ev_seed42.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import uproot
from tqdm import tqdm

BRANCHES = ["RecoVertex_z", "RecoVertex_nTracks", "RecoVertex_assocTracks"]
STEP_SIZE = 20000


def extract_assoc(root_path: str, cache_path: str) -> dict[str, np.ndarray]:
    """Read assoc lists for the cache's event_indices; verify z alignment."""
    cache = np.load(cache_path, allow_pickle=True)
    event_indices = np.asarray(cache["event_indices"], dtype=np.int64)
    cache_vtx_z = cache["RecoVertex_z"]
    wanted = {int(idx): row for row, idx in enumerate(event_indices)}

    n_events = len(event_indices)
    assoc_flat = np.empty(n_events, dtype=object)
    assoc_counts = np.empty(n_events, dtype=object)
    vtx_ntracks = np.empty(n_events, dtype=object)
    n_found = 0

    tree = uproot.open(root_path)["PVFinderData"]
    entry_start = int(event_indices.min())
    entry_stop = int(event_indices.max()) + 1
    print(f"Scanning entries [{entry_start}, {entry_stop}) of {tree.num_entries}")

    progress = tqdm(total=n_events, desc="events found")
    for chunk in tree.iterate(
        BRANCHES,
        step_size=STEP_SIZE,
        entry_start=entry_start,
        entry_stop=entry_stop,
        report=True,
    ):
        arrays, report = chunk
        for local, entry in enumerate(
            range(report.tree_entry_start, report.tree_entry_stop)
        ):
            row = wanted.get(entry)
            if row is None:
                continue
            root_z = np.asarray(arrays["RecoVertex_z"][local], dtype=np.float32)
            cached_z = np.asarray(cache_vtx_z[row], dtype=np.float32)
            if len(root_z) != len(cached_z) or not np.array_equal(root_z, cached_z):
                msg = (
                    f"RecoVertex_z mismatch at entry {entry} (cache row {row}): "
                    "cache and ROOT file are not aligned"
                )
                raise ValueError(msg)

            per_vertex = arrays["RecoVertex_assocTracks"][local]
            counts = np.array([len(v) for v in per_vertex], dtype=np.int32)
            flat = (
                np.concatenate([np.asarray(v, dtype=np.int32) for v in per_vertex])
                if len(per_vertex)
                else np.array([], dtype=np.int32)
            )
            assoc_flat[row] = flat
            assoc_counts[row] = counts
            vtx_ntracks[row] = np.asarray(
                arrays["RecoVertex_nTracks"][local], dtype=np.float32
            )
            n_found += 1
            progress.update(1)
        if n_found == n_events:
            break
    progress.close()

    if n_found != n_events:
        msg = f"Only found {n_found}/{n_events} cached events in ROOT file"
        raise ValueError(msg)

    # Report how often the assoc-list length differs from the nTracks branch
    n_diff = sum(
        int(not np.array_equal(assoc_counts[i], vtx_ntracks[i].astype(np.int32)))
        for i in range(n_events)
    )
    print(f"Events where assoc counts != RecoVertex_nTracks: {n_diff}/{n_events}")

    return {
        "event_indices": event_indices,
        "assoc_flat": assoc_flat,
        "assoc_counts": assoc_counts,
        "vtx_ntracks": vtx_ntracks,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract AMVF assocTracks for cached Run 3 events"
    )
    parser.add_argument("--root", required=True, type=str, help="Source ROOT file")
    parser.add_argument("--cache", required=True, type=str, help="Run 3 NPZ cache")
    parser.add_argument("--output", required=True, type=str, help="Output NPZ")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    data = extract_assoc(args.root, args.cache)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    print(f"Saved assoc cache for {len(data['event_indices'])} events to {out_path}")


if __name__ == "__main__":
    main()
