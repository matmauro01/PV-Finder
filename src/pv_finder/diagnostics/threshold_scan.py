"""Unified peak-finder threshold scan — integral threshold + peak threshold.

Runs inference ONCE on N events (Run 2 MC from the h5 test set, or HLLHC
from ROOT), then sweeps the two knobs of the peak finder independently:

  * integral_threshold (peak area, default 0.5)
      → fix threshold at default (0.01), sweep integral, report
        efficiency, FP/evt, per-category counts.
  * threshold (peak amplitude, default 0.01)
      → fix integral at default (0.5), sweep peak-height threshold,
        report the same metrics.

Saves a 4-panel comparison plot + a pickle with all numbers.

Usage:
    python -u src/pv_finder/diagnostics/threshold_scan.py \\
        --dataset mc|hllhc --max-events 300 --device 0
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

sys.path.insert(0, "src/pv_finder/evaluation/vertex_finding")
sys.path.insert(0, "src")

# fmt: off
from efficiency_res_optimized_atlas import (  # noqa: E402
    compare_res_reco,
    pv_locations_updated_res,
)

from pv_finder.data.run3_io import load_run3_from_root  # noqa: E402
from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000  # noqa: E402

# fmt: on

BIN_WIDTH = 0.04  # mm/bin, matches eval scripts

CANONICAL_RUN2_MC_CKPT = (
    "model_weights/03_24_2026/"
    "reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_300_fullstate.pth"
)
CANONICAL_HLLHC_CKPT = (
    "model_weights/hllhc_pu200_mlp50_e2e400_v2_phase2_epoch_100_fullstate.pth"
)
MC_H5 = (
    "/share/lazy/qibinlei/"
    "recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5"
)
MC_ROOT = (
    "/share/lazy/rocky/ATLAS_data/Latest_Sept2023/ATLAS_PVFinderData_TruthMatched.root"
)
MC_QIBIN = "configs/qibin_test_main_indices_v2.p"
MC_INDICES = "configs/test_main_indices_2550evt.p"
HLLHC_ROOT = (
    "data/run4/Run4_MC21_ITk/"
    "ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root"
)


def mm_to_bins(z):
    return (z + 240.0) / BIN_WIDTH


def _build_model(dataset: str, device: torch.device):
    from run_eval_pvf_run3 import E2E_CONFIG, load_ckpt

    cfg = dict(E2E_CONFIG)
    if dataset == "hllhc":
        cfg.update(n_UNetChannels=96, l_HiddenNodes=[128] * 5)
        ckpt = CANONICAL_HLLHC_CKPT
    else:
        ckpt = CANONICAL_RUN2_MC_CKPT
    model = trackstoHists_UNet_1000(**cfg)
    load_ckpt(ckpt, model, device)
    return model, Path(ckpt).stem


def _run_inference_hllhc(model, device, n_events):
    from run_eval_pvf_run3 import build_subevent_inputs, run_e2e_inference

    events = load_run3_from_root(HLLHC_ROOT, max_events=n_events, min_tracks=1,
        min_amvf_vtx=1, entry_start=0, entry_stop=None)  # fmt: skip
    hists, truths_mm = [], []
    for ev in events:
        subs = build_subevent_inputs(ev)
        hists.append(run_e2e_inference(subs, model, device))
        truths_mm.append(np.asarray(ev.amvf_z, dtype=np.float64))
    return hists, truths_mm, f"HLLHC PU200 ({len(events)} evt)"


def _run_inference_mc(model, device, n_events):
    """Run inference on MC h5 test set using run_eval_pvf.py conventions."""
    import h5py
    import uproot  # noqa: E402
    from run_eval_pvf import N_SUBEVENTS, get_truth_pvs, run_e2e

    with open(MC_INDICES, "rb") as fp:
        event_indices = pickle.load(fp)[:n_events]
    # ROOT truth for nTracks>=2 filter
    tree = uproot.open(MC_ROOT)["PVFinderData"]
    root_z = tree["TruthVertex_z"].array(library="np")
    root_n = tree["TruthVertex_nTracks"].array(library="np")
    with open(MC_QIBIN, "rb") as fp:
        qibin = list(pickle.load(fp))
    hists, truths_mm = [], []
    with h5py.File(MC_H5, "r") as h5f:
        for i, eidx in enumerate(event_indices):
            eidx = int(eidx)
            s0 = eidx * N_SUBEVENTS
            s1 = s0 + N_SUBEVENTS
            ph = run_e2e(h5f["tracks"][s0:s1], model, device)
            hists.append(ph)
            ridx = qibin[i]
            ze = np.asarray(root_z[ridx], dtype=np.float64)
            ne = np.asarray(root_n[ridx], dtype=np.float64)
            truths_mm.append(ze[ne >= 2])
            _ = get_truth_pvs  # keep import used
    return hists, truths_mm, f"Run 2 MC ({len(event_indices)} evt)"


def _one_eval(hist, t_bins, thr, ith, match_win_bins, min_width):
    """Run peak-finder with (thr, ith), match against t_bins, return count dict."""
    p_pvs = pv_locations_updated_res(hist, thr, ith, min_width)[0]
    out = dict(c=0, m=0, s=0, f=0, tc=0, tm=0, npred=len(p_pvs))
    if len(t_bins) == 0 or len(p_pvs) == 0:
        return out
    p_bins = mm_to_bins(p_pvs)
    res, tc_arr, _ = compare_res_reco(
        t_bins, p_bins, match_win_bins * np.ones(len(p_bins)), debug=0
    )
    out["c"] = res.reco_clean
    out["m"] = res.reco_merged
    out["s"] = res.reco_split
    out["f"] = res.reco_fake
    out["tc"] = int(np.sum(tc_arr == "clean"))
    out["tm"] = int(np.sum(tc_arr == "merged"))
    return out


def _scan_1d(hists, truths_mm, *, vary, values, fixed_thr, fixed_ith, match_win_mm):
    """Sweep `vary` (either 'integral' or 'peak'); hold the other threshold fixed.

    Returns (rows, amvf_per_evt). Each row has keys:
        {vary}: swept value
        eff, fp, pred, clean, merged, split, fake
    """
    from run_eval_pvf_run3 import MIN_WIDTH

    win_bins = match_win_mm / BIN_WIDTH
    truths_bins = [mm_to_bins(t) for t in truths_mm]
    total_truth = sum(len(t) for t in truths_bins)
    n = len(hists)
    rows = []
    for v in values:
        if vary == "integral":
            thr, ith = fixed_thr, v
        elif vary == "peak":
            thr, ith = v, fixed_ith
        else:
            raise ValueError(vary)
        tot = dict(c=0, m=0, s=0, f=0, tc=0, tm=0)
        for hist, t_bins in zip(hists, truths_bins):
            r = _one_eval(hist, t_bins, thr, ith, win_bins, MIN_WIDTH)
            for k in tot:
                tot[k] += r[k]
        eff = (tot["tc"] + tot["tm"]) / total_truth if total_truth else 0.0
        rows.append(dict(
            **{vary: v}, eff=eff, fp=tot["f"] / n,
            pred=(tot["c"] + tot["m"] + tot["s"] + tot["f"]) / n,
            clean=tot["c"] / n, merged=tot["m"] / n,
            split=tot["s"] / n, fake=tot["f"] / n,
        ))  # fmt: skip
    return rows, total_truth / n


DEFAULT_THR = 0.01  # peak amplitude default
DEFAULT_ITH = 0.5  # peak integral default
INTEGRAL_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
PEAK_GRID = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100, 0.200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mc", "hllhc"], required=True)
    parser.add_argument("--max-events", type=int, default=300)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--match-window-mm", type=float, default=0.3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out or f"outputs/04_15_2026_output/thr_scan_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    model, ckpt_stem = _build_model(args.dataset, device)

    print(f"\n[{args.dataset}] running inference on {args.max_events} events...")
    t0 = time.time()
    if args.dataset == "hllhc":
        hists, truths_mm, label = _run_inference_hllhc(model, device, args.max_events)
    else:
        hists, truths_mm, label = _run_inference_mc(model, device, args.max_events)
    print(f"  inference done in {time.time() - t0:.1f}s  ({len(hists)} events)")

    print(f"\n[{args.dataset}] INTEGRAL threshold scan "
          f"(peak threshold fixed at {DEFAULT_THR})")  # fmt: skip
    int_rows, amvf = _scan_1d(hists, truths_mm, vary="integral", values=INTEGRAL_GRID,
                              fixed_thr=DEFAULT_THR, fixed_ith=None,
                              match_win_mm=args.match_window_mm)  # fmt: skip
    print(f"  AMVF/truth per evt: {amvf:.2f}")
    print(f"{'ith':>6} | {'eff':>6} {'FP/ev':>6} {'Pred':>6} {'C':>5} {'M':>4} {'S':>4} {'F':>5}")  # fmt: skip
    for r in int_rows:
        print(f"{r['integral']:>6.3f} | {r['eff']:>6.4f} {r['fp']:>6.2f} "
              f"{r['pred']:>6.2f} {r['clean']:>5.2f} {r['merged']:>4.2f} "
              f"{r['split']:>4.2f} {r['fake']:>5.2f}")  # fmt: skip

    print(f"\n[{args.dataset}] PEAK threshold scan "
          f"(integral threshold fixed at {DEFAULT_ITH})")  # fmt: skip
    peak_rows, _ = _scan_1d(hists, truths_mm, vary="peak", values=PEAK_GRID,
                            fixed_thr=None, fixed_ith=DEFAULT_ITH,
                            match_win_mm=args.match_window_mm)  # fmt: skip
    print(f"{'thr':>8} | {'eff':>6} {'FP/ev':>6} {'Pred':>6} {'C':>5} {'M':>4} {'S':>4} {'F':>5}")  # fmt: skip
    for r in peak_rows:
        print(f"{r['peak']:>8.4f} | {r['eff']:>6.4f} {r['fp']:>6.2f} "
              f"{r['pred']:>6.2f} {r['clean']:>5.2f} {r['merged']:>4.2f} "
              f"{r['split']:>4.2f} {r['fake']:>5.2f}")  # fmt: skip

    with open(out_dir / "scan.pkl", "wb") as fp:
        pickle.dump(dict(
            dataset=args.dataset, label=label, ckpt=ckpt_stem,
            match_window_mm=args.match_window_mm, amvf_per_evt=amvf,
            default_thr=DEFAULT_THR, default_ith=DEFAULT_ITH,
            integral_scan=int_rows, peak_scan=peak_rows,
        ), fp)  # fmt: skip
    _plot(out_dir, label, amvf, int_rows, peak_rows)
    print(f"\n✓ scan.pkl + scan.png saved to {out_dir}")


def _two_axis_panel(ax, x, eff, fp, xlabel, title, logx=False):
    (l1,) = ax.plot(x, eff, "o-", color="#2ECC71", lw=2, ms=7, label="Efficiency")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Efficiency", color="#2ECC71")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    if logx:
        ax.set_xscale("log")
    a2 = ax.twinx()
    (l2,) = a2.plot(x, fp, "s-", color="#E74C3C", lw=2, ms=7, label="FP/evt")
    a2.set_ylabel("Fake/evt", color="#E74C3C")
    ax.set_title(title)
    ax.legend(handles=[l1, l2], loc="center right")


def _plot(out_dir, label, amvf, int_rows, peak_rows):
    iv = np.array([r["integral"] for r in int_rows])
    ie = np.array([r["eff"] for r in int_rows])
    ifp = np.array([r["fp"] for r in int_rows])
    ipred = np.array([r["pred"] for r in int_rows])
    pv = np.array([r["peak"] for r in peak_rows])
    pe = np.array([r["eff"] for r in peak_rows])
    pfp = np.array([r["fp"] for r in peak_rows])
    ppred = np.array([r["pred"] for r in peak_rows])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Peak-finder threshold scans — {label}",
                 fontsize=14, fontweight="bold")  # fmt: skip

    a = axes[0, 0]
    a.plot(iv, ipred, "o-", color="#1f77b4", lw=2, ms=7, label="PV-Finder pred")
    a.axhline(amvf, color="#d62728", ls="--", lw=2, label=f"AMVF={amvf:.1f}/evt")
    a.set_xlabel("integral_threshold  (peak area)")
    a.set_ylabel("reco PVs / event")
    a.set_title(f"Total reco vs integral_threshold  (peak thr fixed at {DEFAULT_THR})")
    a.grid(alpha=0.3)
    a.legend()

    _two_axis_panel(axes[0, 1], iv, ie, ifp, "integral_threshold",
                    "Efficiency vs fake rate (integral scan)")  # fmt: skip

    a = axes[1, 0]
    a.plot(pv, ppred, "o-", color="#1f77b4", lw=2, ms=7, label="PV-Finder pred")
    a.axhline(amvf, color="#d62728", ls="--", lw=2, label=f"AMVF={amvf:.1f}/evt")
    a.set_xlabel("threshold  (peak amplitude)")
    a.set_ylabel("reco PVs / event")
    a.set_xscale("log")
    a.set_title(f"Total reco vs threshold  (integral fixed at {DEFAULT_ITH})")
    a.grid(alpha=0.3)
    a.legend()

    _two_axis_panel(axes[1, 1], pv, pe, pfp, "threshold",
                    "Efficiency vs fake rate (peak scan)", logx=True)  # fmt: skip

    plt.tight_layout()
    plt.savefig(out_dir / "scan.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
