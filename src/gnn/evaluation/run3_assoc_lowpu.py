"""Run 3 data track-to-vertex association on the vertex-finding events, versus t.

Truth-free comparison of the GNN track assignments with AMVF's
(``RecoVertex_assocTracks``), on exactly the events and peaks of the low
pile-up vertex finding (``run_eval_pvf_run3.py --regime lowpu
--save-histograms``):

* events: the per-event ROOT entries stored in that ``eval_results.pkl``;
* peaks: re-found from the stored histograms with the regime cuts and
  asserted equal to the pkl's peak list in every event;
* AMVF: nTrk >= 2 vertices, beam-spot corrected, asserted equal to the pkl's
  reference list in every event.

The GNN runs once per event; every t is a replay of the top-1 edges. The
agreement metrics are those of ``gnn.evaluation.evaluate_ttva_run3`` (same
helper functions), at several vertex-matching windows.

Run (repo root, inside tmux):
    PYTHONPATH=src python -u -m gnn.evaluation.run3_assoc_lowpu \\
        --vf-pkl outputs/.../vf_run3/eval_results.pkl --root data/run3/file_3.root \\
        --weights model_weights/ttva_gnn_mu60_v2/ttva_gat_mu60_v2_aug_epoch_115.pyt \\
        -d 2 --out outputs/.../run3_assoc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uproot
from tqdm import tqdm

from gnn.data.graph_construction import create_inference_graph
from gnn.evaluation.evaluate_ttva_run3 import (
    agreement_counts,
    amvf_assignments,
    gnn_assignments,
    leading_vertex,
    match_vertices,
)
from gnn.models.ttva_gat import TTVAGATModel
from pv_finder.diagnostics.lowpu_final.metrics import vf_part
from pv_finder.utils.constants import PT_SCALE
from pv_finder.utils.peak_finding import pv_locations_updated_res
from pv_finder.utils.working_points import MATCH_WINDOW_MM, get_regime

T_GRID: tuple[float, ...] = (0.0, 0.5, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995)
# The regime window, plus the 0.5 / 1.0 mm windows of the July 2026 study.
WINDOWS_MM: tuple[float, ...] = (MATCH_WINDOW_MM, 0.5, 1.0)
BRANCHES = (
    "RecoTrack_z0", "RecoTrack_d0", "RecoTrack_ErrD0", "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0", "RecoTrack_theta", "RecoTrack_phi", "RecoTrack_pT",
    "RecoVertex_z", "RecoVertex_nTracks", "RecoVertex_assocTracks", "BeamPosZ",
)  # fmt: skip


def sha256(path: str | Path) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_entries(root: str, entries: np.ndarray) -> dict[int, dict[str, Any]]:
    """The ``BRANCHES`` of the given ROOT entries, keyed by entry."""
    tree = uproot.open(root)["PVFinderData"]
    wanted = set(int(e) for e in entries)
    out: dict[int, dict[str, Any]] = {}
    for chunk, rep in tree.iterate(
        list(BRANCHES), library="np", step_size=1000, report=True,
        entry_start=int(entries.min()), entry_stop=int(entries.max()) + 1,
    ):  # fmt: skip
        for loc, entry in enumerate(range(rep.tree_entry_start, rep.tree_entry_stop)):
            if entry in wanted:
                out[entry] = {b: chunk[b][loc] for b in BRANCHES}
    if len(out) != len(wanted):
        raise ValueError(f"found {len(out)} of {len(wanted)} entries in {root}")
    return out


def event_inputs(ev: dict[str, Any]) -> dict[str, Any]:
    """Track arrays, AMVF lists (nTrk >= 2) and the beam offset of one event."""
    f = {k: np.asarray(ev[k], dtype=np.float32) for k in BRANCHES[:8]}
    ntrk = np.asarray(ev["RecoVertex_nTracks"], dtype=np.float32)
    mask = ntrk >= 2
    assoc = [np.asarray(v, dtype=np.int64) for v in ev["RecoVertex_assocTracks"]]
    beam = float(np.asarray(ev["BeamPosZ"]).ravel()[0])
    return {
        **f,
        "vtx_mask": mask,
        "amvf_z_corr": np.asarray(ev["RecoVertex_z"], dtype=np.float32)[mask] - beam,
        "assoc_flat": np.concatenate(assoc) if assoc else np.zeros(0, np.int64),
        "assoc_counts": np.array([len(v) for v in assoc], dtype=np.int64),
    }


def best_edges(
    model: TTVAGATModel, x: dict[str, Any], peaks: tuple, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """GNN scores and edge index of one event's fully connected graph."""
    z, h, s = peaks
    stack = np.stack(
        [x["RecoTrack_d0"], x["RecoTrack_z0"], x["RecoTrack_ErrD0"],
         x["RecoTrack_ErrZ0"], x["RecoTrack_ErrD0Z0"], x["RecoTrack_theta"],
         x["RecoTrack_phi"], x["RecoTrack_pT"] / PT_SCALE], axis=1,
    ).astype(np.float32)  # fmt: skip
    g = create_inference_graph(
        x["RecoTrack_z0"], x["RecoTrack_d0"], x["RecoTrack_ErrZ0"],
        x["RecoTrack_ErrD0"], stack, z, h, s,
    )  # fmt: skip
    g["track"].batch = torch.zeros(g["track"].num_nodes, dtype=torch.long)
    g["pv"].batch = torch.zeros(g["pv"].num_nodes, dtype=torch.long)
    with torch.no_grad():
        scores = torch.sigmoid(model(g.to(device))).cpu().numpy()
    return scores, g[("track", "to", "pv")].edge_index.cpu().numpy()


def new_totals() -> dict[str, Any]:
    """Empty accumulators for one t."""
    return {
        "n_tracks": 0, "n_gnn_assigned": 0, "n_amvf_assigned": 0,
        "n_populated": 0, "n_lead_events": 0, "n_lead_agree": 0,
        "windows": {str(w): {} for w in WINDOWS_MM},
    }  # fmt: skip


def accumulate(tot: dict[str, Any], x: dict[str, Any], z: np.ndarray, gnn: np.ndarray,
               amvf: np.ndarray, lead_window: float) -> None:  # fmt: skip
    """Add one event's agreement counts at one t."""
    n = len(gnn)
    tot["n_tracks"] += n
    tot["n_gnn_assigned"] += int((gnn >= 0).sum())
    tot["n_amvf_assigned"] += int((amvf >= 0).sum())
    tot["n_populated"] += int(len(np.unique(gnn[gnn >= 0])))
    for w in WINDOWS_MM:
        p2a = match_vertices(z, x["amvf_z_corr"], w)
        for k, v in agreement_counts(gnn, amvf, p2a).items():
            tot["windows"][str(w)][k] = tot["windows"][str(w)].get(k, 0) + v
        if w == lead_window:
            lg = leading_vertex(gnn, x["RecoTrack_pT"], len(z))
            la = leading_vertex(amvf, x["RecoTrack_pT"], len(x["amvf_z_corr"]))
            if lg >= 0 and la >= 0 and (gnn >= 0).any():
                tot["n_lead_events"] += 1
                tot["n_lead_agree"] += int(p2a[lg] == la)


def rates(tot: dict[str, Any], n_events: int, n_peaks: int, n_amvf: int) -> dict:
    """Fractions from the accumulated counts."""
    out = {
        "gnn_assigned_fraction": tot["n_gnn_assigned"] / tot["n_tracks"],
        "amvf_assigned_fraction": tot["n_amvf_assigned"] / tot["n_tracks"],
        "peaks_per_event": n_peaks / n_events,
        "populated_vertices_per_event": tot["n_populated"] / n_events,
        "amvf_vertices_per_event": n_amvf / n_events,
        "leading_vertex_agreement": tot["n_lead_agree"] / max(tot["n_lead_events"], 1),
        "agreement": {},
    }
    for w, c in tot["windows"].items():
        out["agreement"][w] = {
            **c,
            "vs_amvf_assigned": c["n_agree"] / c["n_amvf_assigned"],
            "vs_both_assigned": c["n_agree"] / c["n_both_assigned"],
        }
    return out


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vf-pkl", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lead-window", type=float, default=MATCH_WINDOW_MM)
    ap.add_argument("-d", "--device", type=int, default=0)
    args = ap.parse_args()
    regime = get_regime("lowpu")
    p = regime.peaks
    dev = torch.device(
        f"cuda:{args.device}"
        if args.device >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    with open(args.vf_pkl, "rb") as fh:
        vf = pickle.load(fh)
    prov_vf = vf["provenance"]
    if prov_vf["regime_overrides"] or vf_part(prov_vf["regime"]) != vf_part(
        regime.to_dict()
    ):
        raise ValueError("vertex-finding pkl was not produced at the lowpu regime")
    entries = np.array([e["event_idx"] for e in vf["per_event"]], dtype=np.int64)
    if len(entries) != len(vf["histograms"]):
        raise ValueError("pkl per_event and histograms differ in length")
    raw = read_entries(args.root, entries)

    model = TTVAGATModel(track_input_size=8, pv_input_size=2, edge_attr_dim=3)
    model.load_state_dict(torch.load(args.weights, map_location=dev))
    model.to(dev).eval()

    totals = {t: new_totals() for t in T_GRID}
    self_check = {"n_amvf_assigned": 0, "n_agree": 0}
    n_peaks = n_amvf = 0
    gates = {"peaks_equal_pkl": 0, "amvf_equal_pkl": 0, "events": len(entries)}
    max_scores: list[np.ndarray] = []
    for i, entry in enumerate(tqdm(entries, desc="Run 3 events")):
        x = event_inputs(raw[int(entry)])
        z, h, _, s = pv_locations_updated_res(
            vf["histograms"][i], p.amplitude, p.integral, p.min_width,
            p.min_height, p.centroid_halfwidth,
        )  # fmt: skip
        gates["peaks_equal_pkl"] += int(np.array_equal(z, vf["pred_pvs_mm"][i]))
        ref = np.asarray(vf["truth_pvs_mm"][i], dtype=np.float32)
        gates["amvf_equal_pkl"] += int(np.allclose(x["amvf_z_corr"], ref, atol=1e-5))
        n_peaks += len(z)
        n_amvf += len(x["amvf_z_corr"])
        amvf = amvf_assignments(
            x["assoc_flat"], x["assoc_counts"], x["vtx_mask"], len(x["RecoTrack_z0"])
        )
        ident = match_vertices(x["amvf_z_corr"], x["amvf_z_corr"], args.lead_window)
        sc = agreement_counts(amvf, amvf, ident)
        self_check["n_amvf_assigned"] += sc["n_amvf_assigned"]
        self_check["n_agree"] += sc["n_agree"]
        if len(z) == 0:
            for t in T_GRID:
                none = np.full(len(amvf), -1, dtype=np.int64)
                accumulate(totals[t], x, z, none, amvf, args.lead_window)
            continue
        scores, ei = best_edges(model, x, (z, h, s), dev)
        mx = np.full(len(amvf), -1.0)
        np.maximum.at(mx, ei[0], scores)
        max_scores.append(mx)
        for t in T_GRID:
            gnn = gnn_assignments(scores, ei, len(amvf), t)
            accumulate(totals[t], x, z, gnn, amvf, args.lead_window)

    n_ev = len(entries)
    if gates["peaks_equal_pkl"] != n_ev or gates["amvf_equal_pkl"] != n_ev:
        raise AssertionError(f"event or peak lists differ from the VF pkl: {gates}")
    sc_rate = self_check["n_agree"] / self_check["n_amvf_assigned"]
    if sc_rate != 1.0:
        raise AssertionError(f"AMVF self-check agreement {sc_rate} != 1")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    summary = {
        "provenance": {
            "git_hash": git.stdout.strip(),
            "argv": sys.argv,
            "regime": regime.to_dict(),
            "gnn_weights": {"path": args.weights, "sha256": sha256(args.weights)},
            "density_checkpoint_sha256": sha256(regime.density.checkpoint),
            "vf_pkl": args.vf_pkl,
            "vf_git_hash": prov_vf["git_hash"],
            "entries_sha256": hashlib.sha256(entries.tobytes()).hexdigest(),
            "entries_first_last": [int(entries[0]), int(entries[-1])],
            "lead_window_mm": args.lead_window,
        },
        "gates": {**gates, "amvf_self_check": sc_rate},
        "n_events": n_ev,
        "by_t": {f"{t:g}": rates(totals[t], n_ev, n_peaks, n_amvf) for t in T_GRID},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    ms = np.concatenate(max_scores)
    np.save(args.out / "max_score_per_track.npy", ms.astype(np.float32))
    print(f"{'t':>6} {'GNN asg':>8} {'AMVF asg':>8} {'agree/both':>10} "
          f"{'agree/AMVF':>10} {'lead':>6}  (window {args.lead_window} mm)")  # fmt: skip
    for t, r in summary["by_t"].items():
        a = r["agreement"][f"{args.lead_window:g}"]
        print(f"{t:>6} {r['gnn_assigned_fraction']:8.4f} {r['amvf_assigned_fraction']:8.4f}"
              f" {a['vs_both_assigned']:10.4f} {a['vs_amvf_assigned']:10.4f}"
              f" {r['leading_vertex_agreement']:6.4f}")  # fmt: skip
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
