"""Shard-cycling data loader for large TTVA graph datasets.

A 180k-graph PU200 training set (~160 GB) cannot be resident in memory on
a shared machine. ShardCyclingLoader keeps ONE shard loaded at a time and
advances to the next shard after each full iteration, so with trainNet's
per-epoch ``for batch in train_loader`` loop every "epoch" trains on the
next shard in round-robin order. len() reports the batch count of the
shard most recently iterated (trainNet divides the summed loss by it after
the epoch).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader


class ShardCyclingLoader:
    """Round-robin, one-shard-resident loader for graph lists on disk."""

    def __init__(
        self,
        shard_paths: list[str],
        batch_size: int,
        num_workers: int = 2,
    ) -> None:
        if not shard_paths:
            msg = "shard_paths must be non-empty"
            raise ValueError(msg)
        self.shard_paths = [Path(p) for p in shard_paths]
        for p in self.shard_paths:
            if not p.exists():
                msg = f"Missing shard: {p}"
                raise FileNotFoundError(msg)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._idx = 0
        self._data: list[HeteroData] | None = None
        self._load(0)
        self._current_len = -(-len(self._data) // batch_size)

    def _load(self, idx: int) -> None:
        self._data = torch.load(self.shard_paths[idx], weights_only=False)
        self._idx = idx

    @property
    def first_graph(self) -> HeteroData:
        """A sample graph (e.g. to materialize lazy model layers)."""
        if self._data is None:
            self._load(self._idx)
        return self._data[0]

    def __len__(self) -> int:
        return self._current_len

    def __iter__(self):
        if self._data is None:
            self._load(self._idx)
        loader = DataLoader(
            self._data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
        self._current_len = len(loader)
        print(
            f"[shards] epoch uses shard {self._idx + 1}/{len(self.shard_paths)}: "
            f"{self.shard_paths[self._idx].name} "
            f"({len(self._data)} graphs, {self._current_len} batches)"
        )
        yield from loader
        # Advance round-robin and free the shard we just consumed
        next_idx = (self._idx + 1) % len(self.shard_paths)
        if next_idx != self._idx:
            self._data = None
            self._idx = next_idx
