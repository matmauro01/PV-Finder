"""Quick threshold scan on HLLHC v2 ep100: runs inference ONCE on 300 events,
then tests multiple integral_threshold values for peak finding + matching.
Saves table + plot to /tmp/hllhc_thr_scan/."""

import pickle
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src/pv_finder/evaluation/vertex_finding")
sys.path.insert(0, "src")

from efficiency_res_optimized_atlas import compare_res_reco, pv_locations_updated_res
from run_eval_pvf_run3 import (
    E2E_CONFIG,
    MIN_WIDTH,
    THRESHOLD,
    build_subevent_inputs,
    load_ckpt,
    mm_to_bins,
    run_e2e_inference,
)

from pv_finder.data.run3_io import load_run3_from_root
from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000

OUT = Path("/tmp/hllhc_thr_scan")
OUT.mkdir(exist_ok=True)
CKPT = "model_weights/hllhc_pu200_mlp50_e2e400_v2_phase2_epoch_100_fullstate.pth"
ROOT = "data/run4/Run4_MC21_ITk/ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root"
N_EVENTS = 300
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
MATCH_WIN_MM = 0.30  # fixed, in mm — avoids refitting sigma per threshold

# --- Setup ---
device = torch.device("cuda:0")
cfg = dict(E2E_CONFIG, n_UNetChannels=96, l_HiddenNodes=[128] * 5)
model = trackstoHists_UNet_1000(**cfg)
load_ckpt(CKPT, model, device)
BIN_WIDTH = (240.0 - (-240.0)) / 12000  # 0.04 mm/bin
win_bins = MATCH_WIN_MM / BIN_WIDTH  # 7.5 bins

print(f"\nLoading {N_EVENTS} HLLHC events...")
t0 = time.time()
events = load_run3_from_root(
    ROOT,
    max_events=N_EVENTS,
    min_tracks=1,
    min_amvf_vtx=1,
    entry_start=0,
    entry_stop=None,
)
print(f"  loaded {len(events)} events in {time.time() - t0:.1f}s")

print("\nRunning inference (one pass, all events)...")
t0 = time.time()
hists = []
for ev in events:
    subevents = build_subevent_inputs(ev)
    hists.append(run_e2e_inference(subevents, model, device))
truths = [mm_to_bins(ev.amvf_z) for ev in events]
n_truth_per_event = [len(t) for t in truths]
total_truth = sum(n_truth_per_event)
print(
    f"  inference done in {time.time() - t0:.1f}s  "
    f"({total_truth} AMVF truth, {total_truth / len(events):.1f}/evt)"
)

print(f"\nMatching window: {MATCH_WIN_MM} mm ({win_bins:.1f} bins)")
print(
    f"\n{'ith':>6} | {'eff':>6} {'FP/evt':>7} {'Pred/evt':>9} "
    f"{'Clean':>6} {'Merged':>7} {'Split':>6} {'Fake':>6} {'close2fake':>11}"
)
print("-" * 80)

results = []
for ith in THRESHOLDS:
    all_pvs = []
    for hist in hists:
        p_pvs = pv_locations_updated_res(hist, THRESHOLD, ith, MIN_WIDTH)[0]
        all_pvs.append(p_pvs)
    tot = dict(c=0, m=0, s=0, f=0, tc=0, tm=0)
    close_pair_fakes = (
        0  # peaks within 0.3-1.0 mm of another predicted peak that are fakes
    )
    for pvs, t_bins in zip(all_pvs, truths):
        if len(t_bins) == 0 or len(pvs) == 0:
            continue
        p_bins = mm_to_bins(pvs)
        res, tc_arr, _ = compare_res_reco(
            t_bins, p_bins, win_bins * np.ones(len(p_bins)), debug=0
        )
        tot["c"] += res.reco_clean
        tot["m"] += res.reco_merged
        tot["s"] += res.reco_split
        tot["f"] += res.reco_fake
        tot["tc"] += int(np.sum(tc_arr == "clean"))
        tot["tm"] += int(np.sum(tc_arr == "merged"))
    n = len(events)
    eff = (tot["tc"] + tot["tm"]) / total_truth if total_truth else 0.0
    fp = tot["f"] / n
    pred_per_evt = (tot["c"] + tot["m"] + tot["s"] + tot["f"]) / n
    results.append(
        dict(
            ith=ith,
            eff=eff,
            fp=fp,
            pred=pred_per_evt,
            clean=tot["c"] / n,
            merged=tot["m"] / n,
            split=tot["s"] / n,
            fake=tot["f"] / n,
        )
    )
    print(
        f"{ith:>6.2f} | {eff:>6.4f} {fp:>7.3f} {pred_per_evt:>9.2f} "
        f"{tot['c'] / n:>6.2f} {tot['m'] / n:>7.2f} {tot['s'] / n:>6.2f} {tot['f'] / n:>6.2f}"
    )

# --- Plots ---
ith_arr = np.array([r["ith"] for r in results])
eff_arr = np.array([r["eff"] for r in results])
fp_arr = np.array([r["fp"] for r in results])
pred_arr = np.array([r["pred"] for r in results])
amvf_per_evt = total_truth / len(events)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    f"HLLHC v2 ep100 — integral_threshold scan on {len(events)} events "
    f"(match window {MATCH_WIN_MM} mm)",
    fontsize=13,
    fontweight="bold",
)

ax[0].plot(ith_arr, pred_arr, "o-", color="#1f77b4", lw=2, ms=7, label="PV-Finder pred")
ax[0].axhline(
    amvf_per_evt,
    color="#d62728",
    ls="--",
    lw=2,
    label=f"AMVF (nTracks≥2) = {amvf_per_evt:.1f}/evt",
)
ax[0].set_xlabel("integral_threshold", fontsize=12)
ax[0].set_ylabel("Mean reconstructed PVs / event", fontsize=12)
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=11)
ax[0].set_title("Total reco count vs threshold")

ax2 = ax[1]
(l1,) = ax2.plot(
    ith_arr, eff_arr, "o-", color="#2ECC71", lw=2, ms=7, label="Efficiency"
)
ax2.set_xlabel("integral_threshold", fontsize=12)
ax2.set_ylabel("Efficiency", color="#2ECC71", fontsize=12)
ax2.tick_params(axis="y", labelcolor="#2ECC71")
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)
ax3 = ax2.twinx()
(l2,) = ax3.plot(ith_arr, fp_arr, "s-", color="#E74C3C", lw=2, ms=7, label="FP / event")
ax3.set_ylabel("Fake rate / event", color="#E74C3C", fontsize=12)
ax3.tick_params(axis="y", labelcolor="#E74C3C")
ax2.set_title("Efficiency vs fake rate")
ax2.legend(handles=[l1, l2], loc="center right", fontsize=11)

plt.tight_layout()
plt.savefig(OUT / "threshold_scan.png", dpi=150)
with open(OUT / "threshold_scan.pkl", "wb") as fp:
    pickle.dump(
        dict(
            events=len(events),
            amvf_per_evt=amvf_per_evt,
            match_window_mm=MATCH_WIN_MM,
            results=results,
        ),
        fp,
    )
print(f"\n✓ Saved {OUT / 'threshold_scan.png'} and threshold_scan.pkl")
