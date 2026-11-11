"""Processing time per event versus pile-up (Figure 7).

No timing measurement exists in this repository. The Athena timing in the
technical note (Section 4b, fig:athena_timing_gpu) was made by the colleague who
did the Athena integration, and only the rendered PNG is available locally.
This module therefore plots a per-event timing table once it is supplied:

    algorithm,device,sample,mu,time_ms
    PV-Finder,GPU,mc21_601229_r16443,193.0,74.2
    AMVF,CPU,mc21_601229_r16443,193.0,612.5
    ...

One row per event and per algorithm configuration, ``time_ms`` the wall time
of the vertexing algorithm (InDetPriVxFinder execute, including track-to-vertex
association), ``mu`` the actual number of interactions of that event.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pv_finder.diagnostics.paper_figures import style

REQUIRED_COLUMNS = ("algorithm", "device", "sample", "mu", "time_ms")
SERIES_STYLE = {
    ("PV-Finder", "GPU"): {"color": style.PVF_COLOR, "marker": "o", "fill": True},
    ("PV-Finder", "CPU"): {"color": style.PVF_COLOR, "marker": "o", "fill": False},
    ("AMVF", "CPU"): {"color": style.AMVF_COLOR, "marker": "s", "fill": True},
}


def read_timing_csv(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Group per-event timing rows by (algorithm, device); validate columns."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks columns {missing}")
        mu: dict[tuple[str, str], list[float]] = defaultdict(list)
        t: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in reader:
            key = (row["algorithm"].strip(), row["device"].strip().upper())
            mu[key].append(float(row["mu"]))
            t[key].append(float(row["time_ms"]))
    return {k: (np.array(mu[k]), np.array(t[k])) for k in mu}


def plot_timing(
    series: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    edges: np.ndarray,
    label_lines: Sequence[str],
    out_dir: Path,
    name: str,
    min_events: int = 20,
) -> list[Path]:
    """Mean time per event and its standard error in bins of mu (log scale)."""
    fig, ax = style.new_figure(8.0, 6.0)
    for key, (mu, t) in sorted(series.items()):
        st = SERIES_STYLE.get(key, {"color": "0.3", "marker": "^", "fill": True})
        xs, ys, es = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (mu >= lo) & (mu < hi)
            if sel.sum() < min_events:
                continue
            xs.append(mu[sel].mean())
            ys.append(t[sel].mean())
            es.append(t[sel].std(ddof=1) / np.sqrt(sel.sum()))
        ax.errorbar(xs, ys, yerr=es, fmt=st["marker"], color=st["color"],
                    markerfacecolor=st["color"] if st["fill"] else "none",
                    label=f"{key[0]} ({key[1]})")  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("Actual number of interactions per crossing, $\\mu$")
    ax.set_ylabel("Mean processing time per event [ms]")
    style.label_block(ax, style.SIM_STATUS, label_lines)
    ax.legend(loc="lower right", frameon=False)
    return style.save(fig, out_dir, name)
