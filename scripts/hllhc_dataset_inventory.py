"""HL-LHC dataset inventory: events and pile-up per file and per role.

Reads ``ActualNumOfInt`` from every HL-LHC ROOT file used for PV-Finder
training (six PU200 files, in the order of the v6 config) and evaluation
(r16443, r16638), then reconstructs the v6 train / validation / test split.

The v6 split (``collectdata_poca_KDE.collect_data_poca_ATLAS``) is a seed-42
random permutation of the 62,083,200 *sub-events* of the concatenated pool,
cut at 96% / 3% / 1%. An event's twelve sub-events can therefore fall in
different roles; per-role pile-up histograms below are weighted by sub-event
(each sub-event counts 1/12 of an event).

Usage (inside tmux, from the repo root):
    python -u scripts/hllhc_dataset_inventory.py \
        --out outputs/11_11_2026_output/datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import uproot

ROOT_DIR = Path("data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026")
H5_DIR = Path("data/run4_all_etas/PU200_alleta_h5")
PART = "ATLAS_PVFinderData_601237_e8481_s4494_r16633_PU200"
# Order must match configs/vertex_finding/config_hllhc_alleta_v6_mse.yml.
TRAIN_FILES: list[tuple[str, Path, Path]] = [
    (
        "601229 r16438",
        ROOT_DIR / "ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.root",
        H5_DIR / "ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.h5",
    ),
    (
        "601229 r16633",
        ROOT_DIR / "ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root",
        H5_DIR / "ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.h5",
    ),
] + [
    (
        f"601237 r16633 part {k}",
        ROOT_DIR / PART / f"{PART}_{k}.root",
        H5_DIR / f"{PART}_{k}.h5",
    )
    for k in (1, 2, 3, 4)
]
HELDOUT_FILES: list[tuple[str, Path]] = [
    (
        "601229 r16443",
        ROOT_DIR / "ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root",
    ),
    (
        "601229 r16638",
        ROOT_DIR / "ATLAS_PVFinderData_601229_e8481_s4494_r16638_PU200.root",
    ),
]
SPLIT = (0.96, 0.03, 0.01)
SPLIT_SEED = 42
N_SUB = 12


def read_mu(path: Path) -> np.ndarray:
    """Return ActualNumOfInt for every entry of a PVFinderData tree."""
    tree = uproot.open(path)["PVFinderData"]
    return np.asarray(tree["ActualNumOfInt"].array(library="np"), dtype=np.float64)


def h5_events(path: Path) -> int:
    """Number of events in a training HDF5 file (rows of ``pv``)."""
    with h5py.File(path, "r") as f:
        n_ev = int(f["pv"].shape[0])
        n_sub = int(f["tracks"].shape[0])
    if n_sub != N_SUB * n_ev:
        raise ValueError(f"{path}: {n_sub} sub-events for {n_ev} events")
    return n_ev


def split_roles(n_sub_total: int) -> np.ndarray:
    """Role per sub-event (0 train, 1 val, 2 test), as in collect_data_poca_ATLAS."""
    perm = np.random.default_rng(SPLIT_SEED).permutation(n_sub_total)
    n_tr = int(n_sub_total * SPLIT[0])
    n_va = int(n_sub_total * SPLIT[1])
    role = np.empty(n_sub_total, dtype=np.int8)
    role[perm[:n_tr]] = 0
    role[perm[n_tr : n_tr + n_va]] = 1
    role[perm[n_tr + n_va :]] = 2
    return role


def summarise(mu: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Mean, range and a 1-unit pile-up histogram."""
    w = np.ones_like(mu) if weights is None else weights
    lo, hi = int(np.floor(mu.min())), int(np.ceil(mu.max()))
    edges = np.arange(lo, hi + 2)
    counts, _ = np.histogram(mu, bins=edges, weights=w)
    return {
        "events": float(w.sum()),
        "mu_mean": float(np.average(mu, weights=w)),
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "hist_edges": edges.tolist(),
        "hist_counts": counts.tolist(),
    }


def main() -> None:
    """Build the inventory JSON and the per-role pile-up plots."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    inv: dict = {"files": [], "roles": {}, "heldout": []}
    mu_pool: list[np.ndarray] = []
    for name, root, h5 in TRAIN_FILES:
        mu = read_mu(root)
        n_h5 = h5_events(h5)
        if n_h5 != mu.size:
            raise ValueError(f"{name}: h5 {n_h5} events vs ROOT {mu.size}")
        mu_pool.append(mu)
        inv["files"].append({"name": name, "root": str(root), **summarise(mu)})
        print(f"{name}: {mu.size} events, mu {mu.min():.0f}-{mu.max():.0f}", flush=True)

    mu_ev = np.concatenate(mu_pool)
    n_sub_total = N_SUB * mu_ev.size
    role = split_roles(n_sub_total)
    mu_sub = np.repeat(mu_ev, N_SUB)
    for code, label in enumerate(("train", "validation", "test")):
        sel = role == code
        inv["roles"][label] = {
            "sub_events": int(sel.sum()),
            **summarise(mu_sub[sel], np.full(int(sel.sum()), 1.0 / N_SUB)),
        }
    inv["pool"] = {"events": int(mu_ev.size), "sub_events": int(n_sub_total)}

    for name, root in HELDOUT_FILES:
        mu = read_mu(root)
        inv["heldout"].append({"name": name, "root": str(root), **summarise(mu)})
        print(f"{name}: {mu.size} events, mu {mu.min():.0f}-{mu.max():.0f}", flush=True)

    (out / "hllhc_dataset_inventory.json").write_text(json.dumps(inv, indent=1))
    plot(inv, out)


def plot(inv: dict, out: Path) -> None:
    """ATLAS-style pile-up distributions per role (train/val/test/held-out)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep

    hep.style.use("ATLAS")
    panels = [
        ("Training", inv["roles"]["train"]),
        ("Validation", inv["roles"]["validation"]),
        ("Test", inv["roles"]["test"]),
    ] + [(f"Held-out {h['name']}", h) for h in inv["heldout"]]
    for title, d in panels:
        fig, ax = plt.subplots(figsize=(7, 5))
        edges = np.asarray(d["hist_edges"], dtype=float)
        counts = np.asarray(d["hist_counts"], dtype=float)
        width = 1
        if title.startswith("Held-out"):
            # Flat-mu files: integer mu values are unevenly populated by the
            # production, so show 10-unit bins.
            width = 10
            new_edges = np.arange(0.0, edges[-1] + width, width)
            counts, _ = np.histogram(edges[:-1], bins=new_edges, weights=counts)
            edges = new_edges
        ax.stairs(counts, edges, color="k", lw=1.5)
        ax.set_xlabel(r"Actual number of interactions $\mu$")
        ax.set_ylabel(f"Events / {width}")
        hep.atlas.text("Simulation Internal", loc=1, ax=ax)
        ax.text(
            0.05, 0.86,
            f"{title}\n{d['events']:,.0f} events, " + r"$\langle\mu\rangle$"
            + f" = {d['mu_mean']:.1f}",
            transform=ax.transAxes, fontsize=13, va="top",
        )  # fmt: skip
        ax.set_ylim(0, 1.6 * counts.max())
        tag = title.lower().replace(" ", "_")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"mu_{tag}.{ext}", bbox_inches="tight", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
