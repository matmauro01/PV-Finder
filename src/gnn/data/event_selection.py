"""Shared ROOT event selection for TTVA graph building and evaluation.

The extended-|eta| July-2026 re-production holds out two SingleLep files
(601229 r16443 / r16638) from PV-Finder v6 training, and those two are
**flat-mu**: only ~7.7% of their entries sit in the PU200 window
mu in [185, 215] (measured 2026-08-04; the rest run down to mu = 0).
Anything that wants "PU200 held-out events" must therefore select a
*non-contiguous* subset of ROOT entries.

That breaks the assumption baked into the original chain tooling, where a
graph list was paired with truth by re-reading entries
``[entry_start, entry_start + len(graphs))``. Pairing graph *i* with the
wrong ROOT event does not raise — it silently mismatches truth vertices
against peaks and produces plausible-looking nonsense. So every consumer
now round-trips an explicit ``entry_indices.npy`` written by the builder.

Helpers here:
  - ``select_entries``   : mu-window entry selection over a scan range
  - ``iterate_entries``  : yield exactly those entries, in index order
  - ``save/load_entry_indices``
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import awkward as ak
import numpy as np

MU_BRANCH = "ActualNumOfInt"


def select_entries(
    tree: Any,
    entry_start: int = 0,
    entry_stop: int | None = None,
    mu_min: float | None = None,
    mu_max: float | None = None,
    max_events: int | None = None,
) -> np.ndarray:
    """Entry indices within a scan range, optionally filtered by pileup.

    Args:
        tree: An open uproot TTree (``PVFinderData``).
        entry_start: First ROOT entry of the scan range.
        entry_stop: One past the last entry to scan (None = end of file).
        mu_min: Lower bound on ``ActualNumOfInt`` (inclusive). None = no cut.
        mu_max: Upper bound on ``ActualNumOfInt`` (inclusive). None = no cut.
        max_events: Keep at most this many *selected* entries (not scanned
            entries) — the natural meaning once a filter is applied.

    Returns:
        Sorted int64 array of absolute ROOT entry indices.
    """
    n_total = int(tree.num_entries)
    stop = n_total if entry_stop is None else min(int(entry_stop), n_total)
    start = max(int(entry_start), 0)
    if stop <= start:
        return np.empty(0, dtype=np.int64)

    indices = np.arange(start, stop, dtype=np.int64)
    if mu_min is not None or mu_max is not None:
        mu = ak.to_numpy(
            tree[MU_BRANCH].array(entry_start=start, entry_stop=stop)
        ).astype(np.float64)
        keep = np.ones(len(mu), dtype=bool)
        if mu_min is not None:
            keep &= mu >= mu_min
        if mu_max is not None:
            keep &= mu <= mu_max
        indices = indices[keep]

    if max_events is not None:
        indices = indices[: int(max_events)]
    return indices


def iterate_entries(
    tree: Any,
    branches: list[str],
    indices: np.ndarray,
    step_size: int = 200,
) -> Iterator[Any]:
    """Yield the events at *indices* (ascending), reading the tree once.

    Selected entries are typically sparse (~8% of the scan range on the
    flat-mu files), so grouping them into contiguous runs buys nothing;
    this streams the enclosing range and drops the non-selected events.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return
    wanted = set(int(i) for i in indices)
    start, stop = int(indices[0]), int(indices[-1]) + 1

    cursor = start
    for chunk in tree.iterate(
        branches, entry_start=start, entry_stop=stop, step_size=step_size
    ):
        n = len(chunk)
        for offset in range(n):
            if (cursor + offset) in wanted:
                yield chunk[offset]
        cursor += n


def save_entry_indices(path: str | Path, indices: np.ndarray) -> None:
    """Persist the selected entry indices next to the artefacts they describe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(indices, dtype=np.int64))


def load_entry_indices(path: str | Path) -> np.ndarray:
    """Load entry indices written by :func:`save_entry_indices`."""
    return np.load(Path(path)).astype(np.int64)


def resolve_entry_indices(
    tree: Any,
    entry_indices_path: str | Path | None,
    entry_start: int,
    n_expected: int,
) -> np.ndarray:
    """Entry indices for a saved graph list, preferring an explicit index file.

    Falls back to the legacy contiguous assumption
    ``[entry_start, entry_start + n_expected)`` when no index file is given,
    which is only correct for builds that applied no event filtering.

    Raises:
        ValueError: if the index file does not have one entry per graph.
    """
    if entry_indices_path is not None:
        indices = load_entry_indices(entry_indices_path)
        if len(indices) != n_expected:
            msg = (
                f"entry-indices file has {len(indices)} entries but the graph "
                f"list has {n_expected}; they must correspond one-to-one"
            )
            raise ValueError(msg)
        return indices
    return np.arange(entry_start, entry_start + n_expected, dtype=np.int64)
