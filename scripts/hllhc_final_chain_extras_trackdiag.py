#!/usr/bin/env python3
"""Track-level diagnosis of the GNN score cut vs mu and truth n_trk (chain extras).

Why is the GNN track efficiency low at low mu? Per truth-labelled track, the
replay records whether it is correct (assigned to a vertex whose dominant
truth vertex is its own) for the GNN with no score cut (t0, >= 1 track), the
GNN at the frozen t (>= 1 and >= 2 track vertices) and AMVF (>= 1, >= 2), and
whether the cut removes it (score < t). Counts are kept per event and per
truth-vertex n_trk bin of the track's own vertex, plus score histograms of
t0-correct and t0-wrong tracks per 10-unit mu bin. The per-event sums are
asserted equal to ``chain_extras_events.npz`` (n_correct, n_assigned_truth).

    PYTHONPATH=src python -u scripts/hllhc_final_chain_extras_trackdiag.py \
        --infer outputs/11_12_2026_output/gnn_v2_freeze/infer/test_s8 \
        --events outputs/11_12_2026_output/hllhc_final/chain_extras/chain_extras_events.npz
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
sys.path.insert(0, str(REPO / "scripts"))
import hllhc_final_chain_extras as ce  # noqa: E402

from gnn.evaluation import hs_metrics as hm  # noqa: E402
from gnn.evaluation.chain_inference import git_state, sha256  # noqa: E402
from gnn.evaluation.chain_t_summary import boot_weights  # noqa: E402

# truth n_trk of the track's own vertex: [1], [2], ..., [81, inf)
NTRK_EDGES = [1, 2, 3, 4, 5, 6, 8, 11, 16, 21, 31, 51, 81, np.inf]
SCORE_EDGES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.0001]
N_MU = 20  # 10-unit mu bins, 190-200 includes 200
# correct (c_*) and assigned truth-labelled (a_*) per arm and rule
ARMS = (("g0", "gnn_t0", 0), ("g1", "gnn_t", 0), ("g2", "gnn_t", 1), ("a1", "amvf", 0), ("a2", "amvf", 1))  # fmt: skip
FIELDS = (
    "n_lab", *(f"c_{k}" for k, _, _ in ARMS), *(f"a_{k}" for k, _, _ in ARMS),
    "c_g0_cut",  # correct at t0, removed by the score cut
    "c_g0_kept_lost",  # correct at t0, kept, but not correct at t (dominant changed)
    "c_g1_new",  # correct at t but not at t0
    "c_g1_1trk",  # correct at t on a single-track vertex (dropped by >= 2)
)  # fmt: skip


def correct_flags(lists: list[np.ndarray], pt: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Per track: correct, assigned, and on a single-track vertex."""
    cls = hm.classify_vertices(lists, pt, truth)
    n = len(truth)
    ok, asg, one = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for v, x in enumerate(lists):
        asg[x] = True
        one[x] = len(x) == 1
        if cls.dominant[v] >= 0:
            ok[x[truth[x] == cls.dominant[v]]] = True
    return ok, asg, one


def process_chunk(path: str, root: str) -> dict:
    """Per-event x n_trk-bin counts and per-mu score histograms for one chunk."""
    keys = ("entries", "mu", "n_tracks", "track_offsets", "peak_offsets", "peak_z", f"targets_{ce.GNN_KEY}", f"scores_{ce.GNN_KEY}")  # fmt: skip
    with np.load(path) as f:
        z = {k: f[k] for k in keys}
    lo_e, hi_e = int(z["entries"][0]), int(z["entries"][-1]) + 1
    data = uproot.open(root)["PVFinderData"].arrays(ce.BRANCHES, entry_start=lo_e, entry_stop=hi_e, library="ak")  # fmt: skip
    toff, poff = z["track_offsets"], z["peak_offsets"]
    nb = len(NTRK_EDGES) - 1
    out = np.zeros((len(z["entries"]), nb, len(FIELDS)), np.int64)
    hist = np.zeros((N_MU, 2, nb, len(SCORE_EDGES) - 1), np.int64)
    for i, e in enumerate(z["entries"]):
        ev = data[int(e) - lo_e]
        pt = ak.to_numpy(ev["RecoTrack_pT"]).astype(np.float64)
        nt = len(pt)
        if nt != z["n_tracks"][i] or float(ev["ActualNumOfInt"]) != z["mu"][i]:
            raise ValueError(f"entry {e}: differs from the inference record")
        tn = ak.to_numpy(ev["TruthVertex_nTracks"]).astype(np.int64)
        truth = hm.track_truth_labels(ak.to_numpy(ak.flatten(ev["TruthVertex_assocTracks"])), tn, nt)  # fmt: skip
        n_pk = int(poff[i + 1] - poff[i])
        tg, sc = z[f"targets_{ce.GNN_KEY}"][toff[i] : toff[i + 1]], z[f"scores_{ce.GNN_KEY}"][toff[i] : toff[i + 1]]  # fmt: skip
        gnn = {t: hm.lists_from_targets(hm.assign_from_scores(tg, sc, t), n_pk) for t in (0.0, ce.T_CUT)}  # fmt: skip
        amvf = [np.unique(ak.to_numpy(x).astype(np.int64)) for x in ev["RecoVertex_assocTracks"]]  # fmt: skip
        flags = {}
        for key, arm, m in ARMS:
            all_lists = (
                amvf if arm == "amvf" else gnn[0.0 if arm == "gnn_t0" else ce.T_CUT]
            )
            lists = [x for x in all_lists if len(x) >= ce.MIN_TRACKS[m]]
            flags[key] = correct_flags(lists, pt, truth)
        lab = truth >= 0
        b = np.digitize(tn[truth[lab]], NTRK_EDGES) - 1
        cut = sc[lab] < ce.T_CUT
        g0, g1 = flags["g0"][0][lab], flags["g1"][0][lab]
        cols = [
            np.ones(len(b), bool), *(flags[k][0][lab] for k, _, _ in ARMS),
            *(flags[k][1][lab] for k, _, _ in ARMS), g0 & cut, g0 & ~cut & ~g1,
            g1 & ~g0, g1 & flags["g1"][2][lab],
        ]  # fmt: skip
        for j, c in enumerate(cols):
            out[i, :, j] = np.bincount(b, weights=c, minlength=nb)[:nb].astype(np.int64)
        mb = min(int(z["mu"][i] // 10), N_MU - 1)
        sb = np.digitize(sc[lab], SCORE_EDGES) - 1
        for w, sel in ((0, g0), (1, ~g0)):
            np.add.at(hist[mb, w], (b[sel], sb[sel]), 1)
    return {"counts": out, "hist": hist}


def summarize(c: np.ndarray, mu: np.ndarray, hist: np.ndarray) -> dict:
    """Ratios of sums per 10-unit mu bin (all n_trk, and per n_trk bin), bootstrap errors."""
    f = {k: c[..., j].astype(np.float64) for j, k in enumerate(FIELDS)}
    ratios = {
        **{f"eff_{k}": (f"c_{k}", "n_lab") for k, _, _ in ARMS},
        **{f"pur_{k}": (f"c_{k}", f"a_{k}") for k, _, _ in ARMS},
        **{f"assigned_{k}": (f"a_{k}", "n_lab") for k, _, _ in ARMS},
        "cut_loss": ("c_g0_cut", "n_lab"), "kept_lost": ("c_g0_kept_lost", "n_lab"),
        "gain": ("c_g1_new", "n_lab"), "one_track_correct": ("c_g1_1trk", "n_lab"),
        "cut_frac_of_t0_correct": ("c_g0_cut", "c_g0"),
    }  # fmt: skip
    bins = {f"mu {lo}-{lo + 10}": (lo, lo + 10 if lo < 190 else 200.5) for lo in range(0, 200, 10)}  # fmt: skip
    bins.update({"inclusive": (0, 1e9), "PU200 (185-215)": (185, 215.5)})
    out = {"ntrk_edges": np.asarray(NTRK_EDGES, float), "bins": {}}
    for label, (lo, hi) in bins.items():
        mask = (mu >= lo) & (mu < hi)
        w = boot_weights(int(mask.sum()), 200, 20261112)
        entry = {"n_events": int(mask.sum()), "n_lab_by_ntrk": f["n_lab"][mask].sum(0)}
        for name, (a, d) in ratios.items():
            num, den = f[a][mask], f[d][mask]
            with np.errstate(invalid="ignore", divide="ignore"):
                tot = (w @ num.sum(1)) / (w @ den.sum(1))
                by = (w @ num) / (w @ den)
                entry[name] = {"value": num.sum() / den.sum(), "err": np.std(tot, ddof=1), "by_ntrk": num.sum(0) / den.sum(0), "by_ntrk_err": np.nanstd(by, 0, ddof=1)}  # fmt: skip
        out["bins"][label] = entry
    out["score_hist"] = {"edges": SCORE_EDGES, "mu_bins": [f"{10 * k}-{10 * k + 10}" for k in range(N_MU)], "t0_correct": hist[:, 0], "t0_wrong": hist[:, 1]}  # fmt: skip
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--infer", required=True, type=Path)
    ap.add_argument(
        "--events", required=True, type=Path, help="chain_extras_events.npz"
    )
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    meta_in = json.loads((a.infer / "metadata.json").read_text())
    ckpts = ce.check_inputs(meta_in)
    paths = sorted(str(p) for p in a.infer.glob("chunk_*.npz"))
    t0 = time.monotonic()
    with ProcessPoolExecutor(min(a.workers, 12)) as pool:
        parts = list(pool.map(process_chunk, paths, [meta_in["root"]] * len(paths)))
    c = np.concatenate([p["counts"] for p in parts])
    hist = sum(p["hist"] for p in parts)
    emeta = json.loads(a.events.with_suffix(".json").read_text())
    with np.load(a.events) as f:
        ref, evs = f["counts"], f["events"]
    gate = {}
    for key, arm, m in ARMS:  # per-event sums vs the main replay
        k = emeta["arms"].index(arm)
        for fld, ref_f in ((f"c_{key}", "n_correct"), (f"a_{key}", "n_assigned_truth")):
            mine = c[..., FIELDS.index(fld)].sum(1)
            gate[fld] = int((mine != ref[:, k, m, emeta["count_fields"].index(ref_f)]).sum())  # fmt: skip
    gate["n_lab"] = int((c[..., 0].sum(1) != evs[:, emeta["event_fields"].index("n_labelled")]).sum())  # fmt: skip
    if any(gate.values()):
        raise AssertionError(f"track counts differ from chain_extras_events: {gate}")
    mu = evs[:, emeta["event_fields"].index("mu")]
    out_dir = a.events.parent
    np.savez_compressed(
        out_dir / "chain_extras_trackdiag.npz", counts=c, hist=hist, mu=mu
    )
    res = {
        "fields": FIELDS, "arms": ARMS, "t": ce.T_CUT, "regime": ce.REGIME.to_dict(),
        "checkpoints": ckpts, "gate_vs_events_npz": gate, "git": git_state(),
        "script_sha256": sha256(Path(__file__)), "runtime_s": time.monotonic() - t0,
        **summarize(c, mu, hist),
    }  # fmt: skip
    import hllhc_final_chain_extras_summary as cs  # JSON-safe converter

    (out_dir / "chain_extras_trackdiag.json").write_text(json.dumps(cs._jsonable(res), indent=1) + "\n")  # fmt: skip
    print(f"done in {time.monotonic() - t0:.0f} s; gate {gate}", flush=True)


if __name__ == "__main__":
    main()
