#!/usr/bin/env python3
"""Are the s8 candidates of the phase-2 training graphs realistic? (GNN v3)

s8 was trained on PU200 events of r16438 / r16633, and the thinned events are
PU200 events with pile-up removed. This compares, from the labelled candidate
graphs of ``gnn.data.chain_training_graphs``, the s8 peak quality of each graph
set with the r16443 validation entries (never seen by s8 or any GNN), per mu
bin:

* one-to-one efficiency: truth vertices (>= 2 tracks) that own a candidate
  within 0.17 mm / truth vertices (the label matching);
* unmatched candidates per event (fakes, splits and merge remnants);
* candidates per truth vertex;
* oracle Clean/truth (each track on its labelled candidate, >= 2-track rule);
* with ``--residual``: candidate z minus the z of the truth vertex that owns
  it, for all owned candidates and for the HS candidate: half-width of the
  central 68% interval (vertex bootstrap). Truth z is read from the ROOT file
  named in each shard's ``.json`` sidecar; for thinned shards the thinning RNG
  is replayed (``thin_event``, same seed, same event order) and the replay is
  checked against every graph's track count.

Ratios of sums, 200-replica event bootstrap.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_peak_quality.py \
        --set ref=.../phase2/r16443_validation_all.pt \
        --set thin=.../phase2/r16438_thin_shard_00.pt,.../r16438_thin_shard_01.pt \
        --set pu200=.../phase2/r16633_pu200_shard_00.pt --out .../peak_quality [--residual]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.evaluation import hs_metrics as hm  # noqa: E402
from gnn.evaluation.chain_t_summary import boot_weights  # noqa: E402

BINS = {"0-20": (0, 20), "20-50": (20, 50), "50-100": (50, 100), "100-150": (100, 150),
        "150-185": (150, 185), "PU200 (185-215)": (185, 215.5)}  # fmt: skip
FIELDS = ("mu", "n_truth2", "owned", "n_peaks", "clean_oracle")


def truth_z_lists(shard: str) -> list[np.ndarray]:
    """Per graph of *shard*: truth vertex z in the graph's own vertex indexing.

    Raises if the replayed track count differs from the graph's for any event.
    """
    import awkward as ak
    import uproot

    from gnn.data.chain_training_graphs import thin_event
    from gnn.data.event_selection import iterate_entries

    meta = json.loads(Path(shard).with_suffix(".json").read_text())
    entries = np.load(Path(shard).with_suffix(".entry_indices.npy")).astype(np.int64)
    seed = meta["thin_seed"]
    rng = np.random.default_rng(seed) if seed is not None else None
    tree = uproot.open(meta["root"])["PVFinderData"]
    branches = ["RecoTrack_z0", "TruthVertex_z", "TruthVertex_nTracks",
                "TruthVertex_assocTracks", "TruthVertex_type", "ActualNumOfInt"]  # fmt: skip
    out = []
    for ev in iterate_entries(tree, branches, entries, step_size=500):
        z0 = ak.to_numpy(ev["RecoTrack_z0"]).astype(np.float32)
        tz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        assoc = [ak.to_numpy(x).astype(np.int64) for x in ev["TruthVertex_assocTracks"]]
        ttype = ak.to_numpy(ev["TruthVertex_type"]).astype(np.int64)
        if rng is not None:
            f, tz, _, _, _, _ = thin_event([z0], tz, tn, assoc, ttype, float(ev["ActualNumOfInt"]), rng)  # fmt: skip
            z0 = f[0]
        out.append((len(z0), tz))
    if len(out) != len(entries):
        raise RuntimeError(f"{shard}: read {len(out)} of {len(entries)} events")
    return out


def residuals(g, n_tracks: int, tz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    """(all owned candidates, HS candidate) z residuals in mm for one graph."""
    if int(g["track"].x.shape[0]) != n_tracks:
        raise RuntimeError(f"entry {int(g.entry)}: replay has {n_tracks} tracks, graph {int(g['track'].x.shape[0])}")  # fmt: skip
    owner = g["pv"].owner_truth.numpy()
    cz = g["pv"].x[:, 0].numpy().astype(np.float64)
    k = np.flatnonzero(owner >= 0)
    dz = cz[k] - tz[owner[k]]
    hs = g["pv"].y_hs.numpy()[k] > 0.5
    return dz, dz[hs]


def half68(x: np.ndarray, n_boot: int = 100) -> list[float]:
    """Half-width of the central 68% interval [um] and its vertex-bootstrap error."""
    if len(x) < 20:
        return [float("nan"), float("nan")]
    hw = lambda v: 0.5 * float(np.subtract(*np.quantile(v, [0.84, 0.16])))  # noqa: E731
    rng = np.random.default_rng(20261113)
    reps = [hw(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return [1e3 * hw(x), 1e3 * float(np.std(reps, ddof=1))]


def per_graph(g) -> list[float]:  # noqa: ANN001
    """FIELDS for one labelled graph."""
    truth = g["track"].truth_pv.numpy()
    n_truth2 = (
        int((np.bincount(truth[truth >= 0]) >= 2).sum()) if (truth >= 0).any() else 0
    )
    owner = g["pv"].owner_truth.numpy()
    ei = g[("track", "to", "pv")].edge_index.numpy()
    y = g[("track", "to", "pv")].y.numpy() > 0.5
    tg = np.full(len(truth), -1)
    tg[ei[0][y]] = ei[1][y]
    lists = [x for x in hm.lists_from_targets(tg, len(owner)) if len(x) >= 2]
    cls = hm.classify_vertices(lists, g["track"].x[:, 7].numpy(), truth)
    return [
        float(g.mu),
        n_truth2,
        int((owner >= 0).sum()),
        len(owner),
        int(cls.counts()[0]),
    ]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, action="append", help="name=a.pt,b.pt")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--residual", action="store_true", help="also the z residual (reads ROOT)"
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    for spec in a.set:
        name, files = spec.split("=", 1)
        rows, res_all, res_hs = [], [], []
        for f in files.split(","):
            graphs = torch.load(f, weights_only=False)
            rows += [per_graph(g) for g in graphs]
            if a.residual:
                tzs = truth_z_lists(f)
                if len(tzs) != len(graphs):
                    raise RuntimeError(f"{f}: {len(graphs)} graphs, {len(tzs)} events")
                for g, (nt, tz) in zip(graphs, tzs):
                    dz, dh = residuals(g, nt, tz)
                    res_all.append(dz)
                    res_hs.append(dh)
            del graphs
            print(f"{name}: {f} read", flush=True)
        x = np.array(rows)
        res[name] = {"files": files.split(","), "n_events": len(x), "bins": {}}
        for lab, (lo, hi) in BINS.items():
            m = (x[:, 0] >= lo) & (x[:, 0] < hi)
            if m.sum() < 20:
                continue
            xm = x[m]
            w = boot_weights(int(m.sum()), 200, 20261113)
            out = {"n_events": int(m.sum()), "mean_mu": float(xm[:, 0].mean())}
            for key, (num, den) in {
                "one_to_one_eff": (2, 1), "candidates_per_truth": (3, 1), "oracle_clean_per_truth": (4, 1),
            }.items():  # fmt: skip
                reps = (w @ xm[:, num]) / (w @ xm[:, den])
                out[key] = [
                    float(xm[:, num].sum() / xm[:, den].sum()),
                    float(reps.std(ddof=1)),
                ]
            unm = xm[:, 3] - xm[:, 2]
            out["unmatched_per_event"] = [
                float(unm.mean()),
                float((w @ unm / w.sum(1)).std(ddof=1)),
            ]
            if a.residual:
                idx = np.flatnonzero(m)
                out["res68_all_um"] = half68(np.concatenate([res_all[i] for i in idx]))
                out["res68_hs_um"] = half68(np.concatenate([res_hs[i] for i in idx]))
            res[name]["bins"][lab] = out
    (a.out / "peak_quality.json").write_text(json.dumps(res, indent=1) + "\n")
    lines = ["# s8 candidate quality per graph set (labelled training graphs)\n",
             "| set | mu bin | events | mean mu | one-to-one eff [%] | candidates / truth | unmatched / event | oracle Clean/truth [%] | res68 all [um] | res68 HS [um] |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for lab in BINS:
        for name, r in res.items():
            b = r["bins"].get(lab)
            if b is None:
                continue
            lines.append(
                f"| {name} | {lab} | {b['n_events']} | {b['mean_mu']:.1f} | "
                f"{100 * b['one_to_one_eff'][0]:.2f} ± {100 * b['one_to_one_eff'][1]:.2f} | "
                f"{b['candidates_per_truth'][0]:.3f} ± {b['candidates_per_truth'][1]:.3f} | "
                f"{b['unmatched_per_event'][0]:.2f} ± {b['unmatched_per_event'][1]:.2f} | "
                f"{100 * b['oracle_clean_per_truth'][0]:.2f} ± {100 * b['oracle_clean_per_truth'][1]:.2f} | "
                + " | ".join(
                    f"{b[k][0]:.1f} ± {b[k][1]:.1f}" if k in b else "-"
                    for k in ("res68_all_um", "res68_hs_um")
                )
                + " |"
            )
    (a.out / "peak_quality.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
