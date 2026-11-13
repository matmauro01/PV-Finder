#!/usr/bin/env python3
"""Where the chain loses Clean HS and HS purity against AMVF (GNN v3 diagnosis).

Replays chain-format inference directories (top-1 GNN edges per track; the
candidates are s8 peaks or truth vertices) together with the ROOT truth and
AMVF, for vertices with >= 2 assigned tracks, at a few score cuts t. Per arm:

1. **HS-vertex composition.** On the HS-matched vertex (most HS tracks,
   ``hs_metrics``) every non-HS track is put in one class:
   ``unlab`` (no truth vertex), ``own_cand`` (its truth vertex owns a
   candidate in the multiplicity-priority one-to-one matching below: the
   GNN could have sent it there), ``merged_hs`` (its truth vertex has no
   candidate of its own and its nearest candidate is the HS-matched one:
   a finder merge), ``orphan`` (its truth vertex has no candidate of its own
   and its nearest candidate is elsewhere or beyond 0.17 mm).
   Matching: truth vertices with >= 2 tracks in descending track count, each
   takes the nearest still-free candidate within 0.17 mm.
2. **Lost HS tracks:** unassigned (score < t) or on another vertex.
3. **Post-GNN fit cut** (s8 arm only): every vertex is refitted with the 3D
   adaptive beam-spot fit used for the HS resolution; tracks with chi2 above
   ``c`` at the fitted position are unassigned (c = 9: the fit's own chi2 cut,
   weight 0.5 at T = 1; c = 18.42: Athena maxVertexChi2), then the >= 2-track
   rule is reapplied.

Ratios of sums; 200-replica event bootstrap (paired with AMVF).

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_hs_diagnosis.py \
        --arm s8=outputs/11_12_2026_output/gnn_v2_freeze/infer/val_s8 \
        --arm truth=outputs/11_13_2026_output/gnn_v3_campaign/diag/infer/val_truth \
        --gnn v2e527 --out outputs/11_13_2026_output/gnn_v3_campaign/diag/hs_diag
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.diagnostics.hs_selection_identical_rule import Tracks  # noqa: E402
from gnn.evaluation import hs_metrics as hm  # noqa: E402
from gnn.evaluation.chain_t_summary import boot_weights  # noqa: E402
from pv_finder.utils.working_points import HLLHC  # noqa: E402

T_LIST = (0.5, 0.9, 0.94, 0.97)
FIT_CUTS = (9.0, 18.42)
WINDOW = HLLHC.match_window_mm
COMP = ("hs", "own_cand", "merged_hs", "orphan", "unlab")
FIELDS = (
    "has_hs", "n_hs_tracks", "clean_hs", "match_tracks", *(f"c_{k}" for k in COMP),
    "hs_unassigned", "hs_elsewhere", "n_correct", "n_assigned_truth", "n_labelled",
    "clean", "n_truth_ge2", "n_reco",
)  # fmt: skip
BRANCHES = [
    "ActualNumOfInt", "BeamPosX", "BeamPosY", "BeamPosSigmaX",
    "RecoTrack_pT", "RecoTrack_d0", "RecoTrack_z0", "RecoTrack_phi", "RecoTrack_theta",
    "RecoTrack_ErrD0", "RecoTrack_ErrZ0", "RecoTrack_ErrD0Z0",
    "TruthVertex_z", "TruthVertex_nTracks", "TruthVertex_assocTracks", "TruthVertex_type",
    "RecoVertex_z", "RecoVertex_assocTracks",
]  # fmt: skip


def priority_match(cand_z: np.ndarray, tz: np.ndarray, tn: np.ndarray) -> np.ndarray:
    """Candidate owned by each truth vertex (-1 if none); see module docstring."""
    owner = np.full(len(tz), -1, dtype=np.int64)
    if len(cand_z) == 0:
        return owner
    free = np.ones(len(cand_z), bool)
    for v in np.lexsort((np.arange(len(tz)), -tn)):
        if tn[v] < 2:
            break
        d = np.where(free, np.abs(cand_z - tz[v]), np.inf)
        k = int(np.argmin(d))
        if d[k] < WINDOW:
            owner[v] = k
            free[k] = False
    return owner


def record(
    lists: list[np.ndarray], cand_idx: np.ndarray, cand_z: np.ndarray,
    pt: np.ndarray, truth: np.ndarray, tz: np.ndarray, tn: np.ndarray, h: int,
    owner: np.ndarray, n_tracks: int,
) -> np.ndarray:  # fmt: skip
    """FIELDS for one vertex list (``cand_idx[v]`` = candidate of vertex v)."""
    cls = hm.classify_vertices(lists, pt, truth)
    nhs = hm.hs_track_counts(lists, cls, truth, h)
    m = hm.hs_matched_index(nhs, cls)
    comp = np.zeros(len(COMP), np.int64)
    n_hs = int((truth == h).sum())
    hs_un = hs_else = 0
    on_vertex = np.zeros(n_tracks, bool)
    for x in lists:
        on_vertex[x] = True
    if m >= 0:
        x = lists[m]
        tt = truth[x]
        comp[0] = int((tt == h).sum())
        comp[4] = int((tt < 0).sum())
        k_m = cand_idx[m]
        for v in tt[(tt >= 0) & (tt != h)]:
            if owner[v] >= 0:
                comp[1] += 1
            elif len(cand_z) and int(np.argmin(np.abs(cand_z - tz[v]))) == k_m:
                comp[2] += 1
            else:
                comp[3] += 1
        hs_mask = truth == h
        hs_un = int((hs_mask & ~on_vertex).sum())
        hs_else = n_hs - comp[0] - hs_un
    n_correct = n_at = 0
    for v, x in enumerate(lists):
        tt = truth[x]
        n_at += int((tt >= 0).sum())
        if cls.dominant[v] >= 0:
            n_correct += int((tt == cls.dominant[v]).sum())
    clean_hs = int(np.any((cls.dominant == h) & (cls.labels == hm.CLEAN)))
    return np.array(
        [int(h >= 0), n_hs, clean_hs, len(lists[m]) if m >= 0 else 0, *comp,
         hs_un, hs_else, n_correct, n_at, int((truth >= 0).sum()),
         int(cls.counts()[hm.CLEAN]), int((tn >= 2).sum()), len(lists)],
        dtype=np.int64,
    )  # fmt: skip


def fit_cut(
    trk: Tracks, lists: list[np.ndarray], cand_z: np.ndarray, beam, sig: float,
    cut: float,
) -> list[np.ndarray]:  # fmt: skip
    """Refit each vertex, drop tracks with chi2 > cut (lists stay aligned)."""
    out = []
    for v, x in enumerate(lists):
        if len(x) == 0:
            out.append(x)
            continue
        pos = trk.fit(x, float(cand_z[v]), beam, sig)
        out.append(x[trk.chi2(x, pos) <= cut])
    return out


def process(path: str, root: str, gnn: str, do_fit: bool) -> dict:
    """Per-event records of one chunk: (n, arms, FIELDS); arms = t (+fit cuts) + AMVF."""
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    entries = z["entries"]
    toff, poff = z["track_offsets"], z["peak_offsets"]
    tg_all, sc_all = z[f"targets_{gnn}"], z[f"scores_{gnn}"]
    tree = uproot.open(root)["PVFinderData"]
    lo = int(entries[0])
    data = tree.arrays(BRANCHES, entry_start=lo, entry_stop=int(entries[-1]) + 1)
    n_arm = len(T_LIST) * (1 + (len(FIT_CUTS) if do_fit else 0)) + 1
    out = np.zeros((len(entries), n_arm, len(FIELDS)), np.int64)
    mu = np.zeros(len(entries))
    for i, e in enumerate(entries):
        ev = data[int(e) - lo]
        trk = Tracks(ev)
        nt = len(trk.pt)
        tz = ak.to_numpy(ev["TruthVertex_z"]).astype(np.float64)
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        truth = hm.track_truth_labels(
            ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, nt
        )
        hs_c = np.flatnonzero(ak.to_numpy(ev["TruthVertex_type"]) == 1)
        h = int(hs_c[0]) if len(hs_c) == 1 else -2
        mu[i] = float(ev["ActualNumOfInt"])
        beam = (float(ev["BeamPosX"]), float(ev["BeamPosY"]))
        sig = float(ev["BeamPosSigmaX"])
        peaks = z["peak_z"][poff[i] : poff[i + 1]].astype(np.float64)
        owner = priority_match(peaks, tz, tn)
        tg, sc = tg_all[toff[i] : toff[i + 1]], sc_all[toff[i] : toff[i + 1]]
        a = 0
        for t in T_LIST:
            all_lists = hm.lists_from_targets(
                hm.assign_from_scores(tg, sc, t), len(peaks)
            )
            variants = [all_lists]
            if do_fit:
                variants += [
                    fit_cut(trk, all_lists, peaks, beam, sig, c) for c in FIT_CUTS
                ]
            for lists_v in variants:
                keep = np.flatnonzero([len(x) >= 2 for x in lists_v])
                out[i, a] = record(
                    [lists_v[j] for j in keep], keep, peaks, trk.pt, truth, tz, tn,
                    h, owner, nt,
                )  # fmt: skip
                a += 1
        alists = [np.unique(ak.to_numpy(x).astype(np.int64)) for x in ev["RecoVertex_assocTracks"]]  # fmt: skip
        az = ak.to_numpy(ev["RecoVertex_z"]).astype(np.float64)
        keep = np.flatnonzero([len(x) >= 2 for x in alists])
        a_owner = priority_match(az[keep], tz, tn)
        out[i, a] = record(
            [alists[j] for j in keep], np.arange(len(keep)), az[keep], trk.pt,
            truth, tz, tn, h, a_owner, nt,
        )  # fmt: skip
    return {"rec": out, "mu": mu}


def ratios(rec: np.ndarray, mask: np.ndarray, w: np.ndarray) -> dict:
    """Value, error and paired error vs the last arm (AMVF), per ratio."""
    f = {k: rec[mask][:, :, j].astype(np.float64) for j, k in enumerate(FIELDS)}
    defs = {
        "clean_hs": ("clean_hs", "has_hs"),
        "hs_purity": ("c_hs", "match_tracks"),
        "hs_track_frac": ("c_hs", "n_hs_tracks"),
        **{f"frac_{k}": (f"c_{k}", "match_tracks") for k in COMP[1:]},
        "hs_unassigned": ("hs_unassigned", "n_hs_tracks"),
        "hs_elsewhere": ("hs_elsewhere", "n_hs_tracks"),
        "trk_eff": ("n_correct", "n_labelled"),
        "trk_pur": ("n_correct", "n_assigned_truth"),
        "clean_per_truth": ("clean", "n_truth_ge2"),
        "clean_per_reco": ("clean", "n_reco"),
    }
    out = {}
    for name, (a, b) in defs.items():
        reps = (w @ f[a]) / (w @ f[b])
        out[name] = {
            "value": (f[a].sum(0) / f[b].sum(0)).tolist(),
            "err": reps.std(0, ddof=1).tolist(),
            "diff_err": (reps - reps[:, -1:]).std(0, ddof=1).tolist(),
        }
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, nargs="+", help="name=infer_dir")
    ap.add_argument("--gnn", required=True)
    ap.add_argument("--fit-arms", nargs="*", default=["s8"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"t": T_LIST, "fit_cuts": FIT_CUTS, "fields": FIELDS, "arms": {}}
    bins = {
        "inclusive": (0, 1e9),
        "PU200": (185, 215.5),
        "mu<20": (0, 20),
        "0-10": (0, 10),
    }
    for spec in a.arm:
        name, d = spec.split("=", 1)
        d = Path(d)
        meta = json.loads((d / "metadata.json").read_text())
        do_fit = name in a.fit_arms
        paths = sorted(str(p) for p in d.glob("chunk_*.npz"))
        t0 = time.monotonic()
        with ProcessPoolExecutor(a.workers) as pool:
            parts = list(pool.map(process, paths, [meta["root"]] * len(paths),
                                  [a.gnn] * len(paths), [do_fit] * len(paths)))  # fmt: skip
        rec = np.concatenate([p["rec"] for p in parts])
        mu = np.concatenate([p["mu"] for p in parts])
        np.savez_compressed(a.out / f"{name}.npz", rec=rec, mu=mu)
        labels = []
        for t in T_LIST:
            labels.append(f"t{t:g}")
            if do_fit:
                labels += [f"t{t:g}_fit{c:g}" for c in FIT_CUTS]
        labels.append("amvf")
        arm = {"labels": labels, "bins": {}, "runtime_s": time.monotonic() - t0}
        for lab, (lo, hi) in bins.items():
            mask = (mu >= lo) & (mu < hi) & (rec[:, 0, 0] > 0)
            w = boot_weights(int(mask.sum()), 200, 20261113)
            arm["bins"][lab] = {"n_events": int(mask.sum()), **ratios(rec, mask, w)}
        summary["arms"][name] = arm
        print(f"{name}: {len(mu)} events, {time.monotonic() - t0:.0f} s", flush=True)
    (a.out / "hs_diag.json").write_text(json.dumps(summary, indent=1) + "\n")
    lines = ["# HS diagnosis (>= 2-track vertices), values in %\n"]
    keys = ("clean_hs", "hs_purity", "hs_track_frac", "frac_own_cand", "frac_merged_hs",
            "frac_orphan", "frac_unlab", "hs_unassigned", "hs_elsewhere",
            "trk_eff", "trk_pur", "clean_per_truth", "clean_per_reco")  # fmt: skip
    for lab in bins:
        lines.append(f"\n## {lab}\n")
        lines.append("| arm | " + " | ".join(keys) + " |")
        lines.append("|---|" + "---:|" * len(keys))
        for name, arm in summary["arms"].items():
            b = arm["bins"][lab]
            for j, al in enumerate(arm["labels"]):
                if al == "amvf" and name != next(iter(summary["arms"])):
                    continue
                cells = [f"{100 * b[k]['value'][j]:.2f} ± {100 * b[k]['err'][j]:.2f}" for k in keys]  # fmt: skip
                lines.append(f"| {name}:{al} | " + " | ".join(cells) + " |")
        lines.append(f"\n(n events: {', '.join(str(arm['bins'][lab]['n_events']) for arm in summary['arms'].values())})")  # fmt: skip
    (a.out / "hs_diag.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
