"""Recover per-event integral-0.5 peak lists from a stored pairwise-dz array.

The HL-LHC evals stored the peak positions only at the *counting* integral
threshold (0.30 or 0.20), while sigma_vtx-vtx was fitted on a second peak list
at integral 0.50 that was never saved.  What was saved is the flat array of
signed pairwise differences of that second list, concatenated event by event in
event order (``run_eval_pvf_run3.py``: shuffle each event, then all i < j).

Because the integral threshold only decides whether a region is *recorded* and
never moves a region boundary or a position (``pv_locations_updated_res``), the
integral-0.5 list of an event is an exact subset of its stored lower-threshold
list, with bit-identical float32 positions.  So each event's block of
differences can be explained exactly by a subset of the stored peaks, which
recovers the subset and the block length together.

Recovery is proved, not assumed: each block must be reproduced bit for bit by
the recovered subset, the blocks must tile the whole array with nothing left
over, and (when available) the mu-window sub-array the eval also stored must
equal the concatenation of the recovered blocks for the flagged events.
Events with fewer than two integral-0.5 peaks contribute no pair and cannot be
recovered; they are returned as ``None`` and counted.
"""

from __future__ import annotations

import numpy as np

# Tolerance used only to *propose* a block length; acceptance is bit-exact.
_PROPOSE_TOL: float = 1e-4


def _pair_block(order: np.ndarray) -> np.ndarray:
    """Signed differences over i < j in the given order, float32 then float64.

    Mirrors the eval: ``float(p[ii] - p[jj])`` on float32 positions.
    """
    n = len(order)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    return (order[:, None] - order[None, :])[iu].astype(np.float64)


def _try_block(peaks: np.ndarray, dz: np.ndarray, o: int, n: int) -> np.ndarray | None:
    """Return the ordered subset of ``peaks`` producing ``dz[o:o+n(n-1)/2]``."""
    npair = n * (n - 1) // 2
    if o + npair > len(dz):
        return None
    block = dz[o : o + npair]
    head = block[: n - 1]  # s0 - s_j for j = 1 .. n-1
    for s0 in peaks:
        diffs = (s0 - peaks).astype(np.float64)  # float32 subtraction
        srt = np.sort(diffs)
        pos = np.clip(np.searchsorted(srt, head), 0, len(srt) - 1)
        if not np.all(srt[pos] == head):
            continue
        # Map each head difference back to its peak; reject duplicates.
        idx = [int(np.flatnonzero(diffs == h)[0]) for h in head]
        if len(set(idx)) != len(idx):
            continue
        order = np.concatenate([[s0], peaks[idx]]).astype(np.float32)
        if np.array_equal(_pair_block(order), block):
            return order
    return None


def recover_subsets(
    peaks_by_event: list[np.ndarray],
    dz: np.ndarray,
) -> tuple[list[np.ndarray | None], dict[str, int]]:
    """Split ``dz`` into per-event blocks and recover each event's peak subset.

    Parameters
    ----------
    peaks_by_event:
        The stored lower-threshold peak positions per event (float32), in the
        order the eval processed the events.
    dz:
        The stored flat pairwise-dz array of the higher-threshold lists, all
        events, no |dz| cut.

    Returns
    -------
    subsets:
        Per event, the recovered positions (sorted, float32), or ``None`` for an
        event with fewer than two higher-threshold peaks (no pairs; the list
        is then not recoverable, but it contributes nothing to any pair).
    stats:
        Counts of recovered / pairless events and the number of differences
        consumed, which must equal ``len(dz)``.
    """
    dz = np.asarray(dz, dtype=np.float64)
    out: list[np.ndarray | None] = []
    o = 0
    n_rec = n_none = 0
    for e, pk in enumerate(peaks_by_event):
        pk = np.asarray(pk, dtype=np.float32)
        m = len(pk)
        found = None
        if m >= 2 and o < len(dz):
            cands: list[int] = []
            if m >= 3 and o + 1 < len(dz):
                target = dz[o + 1] - dz[o]  # s1 - s2, first entry of row 1
                hi = min(len(dz) - o, m)
                win = dz[o + 2 : o + hi]
                ks = np.flatnonzero(np.abs(win - target) < _PROPOSE_TOL) + 3
                cands = sorted({int(k) for k in ks if 3 <= k <= m}, reverse=True)
            cands.append(2)
            for n in cands:
                found = _try_block(pk, dz, o, n)
                if found is not None:
                    o += n * (n - 1) // 2
                    break
        if found is None:
            out.append(None)
            n_none += 1
        else:
            out.append(np.sort(found))
            n_rec += 1
        if e % 5000 == 0:
            print(f"    [recover] event {e}: consumed {o:,}/{len(dz):,}", flush=True)
    if o != len(dz):
        raise ValueError(
            f"recovery consumed {o} of {len(dz)} differences; the blocks do not "
            "tile the stored array, so the event order or subset assumption fails"
        )
    return out, {"n_recovered": n_rec, "n_pairless": n_none, "n_consumed": o}


def check_window_subarray(
    subsets: list[np.ndarray | None],
    flags: np.ndarray,
    dz_window: np.ndarray,
) -> bool:
    """The eval's mu-window pairwise array must be the flagged events' blocks.

    Order within each block differs (the eval shuffled), so compare sorted.
    """
    blocks = [
        _pair_block(np.asarray(s, dtype=np.float32))
        for s, f in zip(subsets, flags)
        if f and s is not None
    ]
    a = np.sort(np.abs(np.concatenate(blocks))) if blocks else np.zeros(0)
    b = np.sort(np.abs(np.asarray(dz_window, dtype=np.float64)))
    return bool(len(a) == len(b) and np.array_equal(a, b))
