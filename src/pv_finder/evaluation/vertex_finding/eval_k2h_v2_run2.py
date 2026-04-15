#!/usr/bin/env python3
"""K2H v2 evaluation on Run 2 data with analytical KDE input.

Loads Run 2 events from ROOT, computes analytical KDEs, runs K2H v2
inference, and reports resolution + efficiency against AMVF vertices.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from pv_finder.data.run3_io import load_run3_from_root  # noqa: E402
from pv_finder.diagnostics.domain_shift_investigation.kde_study.analytical_kde import (  # noqa: E402
    compute_analytical_kde_event_run3,
)
from pv_finder.models.unet_v2 import UNet_1000_v2  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from efficiency_res_optimized_atlas import (  # noqa: E402
    compare_res_reco,
    pv_locations_updated_res,
)
from plots_pvf import plot_performance, plot_resolution, plot_stats  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Z_MIN, Z_MAX = -240.0, 240.0  # mm
N_BINS_FULL, N_BINS_SUB, N_SUBEVENTS = 12000, 1000, 12
BIN_WIDTH = (Z_MAX - Z_MIN) / N_BINS_FULL  # 0.04 mm/bin
THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH = 1e-2, 0.2, 3
INTEGRAL_THRESHOLD_RES = 0.5
FIT_P0 = [1000.0, 10.0, 30.0, 0.8]

_DEFAULT_ROOT = (
    "data/run2/Run2_Data/"
    "user.rgarg.data18_13TeV.00364076.physics_ZeroBias.AOD1_EXT0/"
    "user.rgarg.49035490.EXT0._000002.ATLAS_PVFinderData_Run3Data.root"
)
_DEFAULT_K2H = "model_weights/K2H_v2_interp_200epochs_epoch_190_fullstate.pth"


def mm_to_bins(z: np.ndarray) -> np.ndarray:
    return (z - Z_MIN) / BIN_WIDTH


def sigmoid_fit(x: np.ndarray, a: float, b: float, c: float, rcc: float) -> np.ndarray:
    return a / (1.0 + np.exp(b * (rcc - np.abs(x)))) + c


def load_ckpt(path: str, model: torch.nn.Module, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = (
        ckpt["model_state"]
        if isinstance(ckpt, dict) and "model_state" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    if isinstance(ckpt, dict):
        ls = f"{ckpt['loss']:.6f}" if ckpt.get("loss") is not None else "N/A"
        n = sum(p.numel() for p in model.parameters())
        print(
            f"  ckpt {Path(path).name}: epoch={ckpt.get('epoch')} "
            f"loss={ls} params={n:,}"
        )
    model.to(device).eval()


def run_k2h(
    kde: np.ndarray, model: torch.nn.Module, device: torch.device
) -> np.ndarray:
    """Run K2H v2 on (12, 1000) KDE -> (12000,) histogram."""
    inp = torch.from_numpy(kde[:, np.newaxis, :]).float().to(device)
    with torch.no_grad():
        out = model(inp)  # (12, 1000)
    return out.cpu().numpy().reshape(-1).astype(np.float32)


def _evt_rec(
    eidx: int,
    nt: int,
    np_: int,
    c: int,
    m: int,
    s: int,
    f: int,
    eff: float,
    tc: int,
    tm: int,
    tmiss: int,
    mu: float | None,
) -> dict:
    return dict(
        event_idx=eidx, n_truth=nt, n_pred=np_, clean=c, merged=m,
        split=s, fake=f, eff=eff, tc=tc, tm=tm, tmiss=tmiss, mu=mu,
    )  # fmt: skip


def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    print("=" * 65)
    print("  K2H v2 Evaluation on Run 2 Data (analytical KDE)")
    print("=" * 65)

    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print(f"\nDevice: GPU {args.device} -- "
              f"{torch.cuda.get_device_name(args.device)}")  # fmt: skip
    else:
        device = torch.device("cpu")
        print("\nDevice: CPU")

    # --- Load model ---
    print("\n--- Loading K2H v2 Model ---")
    k2h = UNet_1000_v2(n=64, n_features=1, dropout_p=0.0)
    load_ckpt(args.k2h_model, k2h, device)
    mode_label = f"K2H v2 + analytical KDE ({Path(args.k2h_model).stem})"

    # --- Load data ---
    print("\n--- Loading Run 2 Data ---")
    events = load_run3_from_root(
        args.root,
        max_events=args.max_events,
        min_tracks=1,
        min_amvf_vtx=1,
    )
    n_events = len(events)
    if n_events == 0:
        print("ERROR: no events loaded.")
        sys.exit(1)

    # --- Inference ---
    print(f"\n--- KDE + Inference ({n_events} events) ---")
    all_pred: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    pairwise_dz: list[float] = []

    for i, event in enumerate(events):
        # Compute analytical KDE (expects dict with track arrays)
        evt_dict = {
            "z0": event.z0,
            "d0": event.d0,
            "d0_err": event.d0_err,
            "z0_err": event.z0_err,
            "d0_z0_cov": event.d0_z0_cov,
        }
        kde = compute_analytical_kde_event_run3(evt_dict)  # (12, 1000)

        # Run K2H v2
        ph = run_k2h(kde, k2h, device)  # (12000,)

        # Peak finding
        p_pvs, p_hts, *_ = pv_locations_updated_res(
            ph, THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH
        )
        p_pvs_r, p_hts_r, *_ = pv_locations_updated_res(
            ph, THRESHOLD, INTEGRAL_THRESHOLD_RES, MIN_WIDTH
        )

        # Pairwise dz for resolution
        p_pvs_r_sh = p_pvs_r.copy()
        np.random.shuffle(p_pvs_r_sh)
        for ii in range(len(p_pvs_r_sh)):
            for jj in range(ii + 1, len(p_pvs_r_sh)):
                pairwise_dz.append(float(p_pvs_r_sh[ii] - p_pvs_r_sh[jj]))

        # Truth: beam-corrected AMVF vertices
        t_pvs = event.amvf_z - event.beam_z

        print(f"  evt {i+1:3d}/{n_events}: truth={len(t_pvs)} "
              f"pred={len(p_pvs)} max={ph.max():.4f}")  # fmt: skip

        all_pred.append(p_pvs)
        all_truth.append(t_pvs)

    tp = sum(len(p) for p in all_pred)
    tt = sum(len(t) for t in all_truth)
    print(f"\n  done: pred={tp:,} ({tp/n_events:.1f}/evt) "
          f"truth={tt:,} ({tt/n_events:.1f}/evt)")  # fmt: skip

    # --- Resolution ---
    print("\n--- Resolution (sigma_vtx_vtx) ---")
    dz_arr = np.array(pairwise_dz, dtype=np.float64)
    bins_r = np.linspace(-6.0, 6.0, 61)
    ctrs = 0.5 * (bins_r[:-1] + bins_r[1:])
    cnts, _ = np.histogram(dz_arr, bins=bins_r)
    sigma, popt = 0.5, None
    try:
        popt, pcov = curve_fit(
            sigmoid_fit,
            ctrs,
            cnts.astype(float),
            p0=FIT_P0,
            maxfev=10000,
            bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]),
        )
        sigma = float(abs(popt[3]))
        serr = float(np.sqrt(np.diag(pcov))[3])
        print(f"  sigma_vtx_vtx = {sigma:.4f} +/- {serr:.4f} mm "
              f"({sigma/BIN_WIDTH:.1f} bins)")  # fmt: skip
    except RuntimeError as exc:
        print(f"  WARNING: fit failed ({exc}). Default sigma={sigma} mm")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_resolution(
        dz_arr,
        sigma,
        popt,
        sigmoid_fit,
        mode_label,
        outdir,
        title=f"PVF Resolution -- Run 2 Data\n({mode_label})",
    )
    print(f"  Saved: {outdir / 'resolution_plot.png'}")

    # --- Performance ---
    sig_bins = sigma / BIN_WIDTH
    print(f"\n--- Performance (window={sigma:.4f} mm={sig_bins:.1f} bins) ---")
    print("  Truth source: AMVF vertices (nTracks >= 2, beam-corrected)")
    tot_c = tot_m = tot_s = tot_f = 0
    tot_tc = tot_tm = tot_tmiss = tot_truth = 0
    per_event: list[dict] = []

    for i, (p_pvs, t_pvs) in enumerate(zip(all_pred, all_truth)):
        event = events[i]
        nt = len(t_pvs)
        if nt == 0:
            continue
        t_bins, p_bins = mm_to_bins(t_pvs), mm_to_bins(p_pvs)
        np_ = len(p_bins)
        mu = event.mu
        if np_ == 0:
            tot_truth += nt
            tot_tmiss += nt
            per_event.append(
                _evt_rec(event.event_idx, nt, 0, 0, 0, 0, 0, 0.0, 0, 0, nt, mu)
            )
            continue

        res, tc_arr, _ = compare_res_reco(
            t_bins, p_bins, sig_bins * np.ones(np_), debug=0
        )
        ntc = int(np.sum(tc_arr == "clean"))
        ntm = int(np.sum(tc_arr == "merged"))
        ntmiss = int(np.sum(tc_arr == "missed"))
        eff = (ntc + ntm) / nt
        tot_c += res.reco_clean
        tot_m += res.reco_merged
        tot_s += res.reco_split
        tot_f += res.reco_fake
        tot_tc += ntc
        tot_tm += ntm
        tot_tmiss += ntmiss
        tot_truth += nt
        per_event.append(
            _evt_rec(
                event.event_idx,
                nt,
                np_,
                res.reco_clean,
                res.reco_merged,
                res.reco_split,
                res.reco_fake,
                eff,
                ntc,
                ntm,
                ntmiss,
                mu,
            )
        )
        if i < 5 or i % 50 == 0:
            print(f"  evt {i:3d}: t={nt} p={np_} C={res.reco_clean} "
                  f"M={res.reco_merged} S={res.reco_split} "
                  f"F={res.reco_fake} eff={eff:.3f}")  # fmt: skip

    nsc = len(per_event)
    overall_eff = (tot_tc + tot_tm) / tot_truth if tot_truth else 0.0
    fp_rate = tot_f / nsc if nsc else 0.0

    # Summary (all events -- Run 2 has no mu/ActualNumOfInt typically)
    has_mu = any(e.mu is not None for e in events)
    sevts, mu_lbl = per_event, "all events"
    ns = len(sevts)

    def avg(k: str) -> float:
        return float(np.mean([r[k] for r in sevts])) if sevts else 0.0

    ac, am, as_, af = avg("clean"), avg("merged"), avg("split"), avg("fake")
    atc, atm, atmiss, ant = avg("tc"), avg("tm"), avg("tmiss"), avg("n_truth")
    ftc = sum(r["tc"] for r in sevts)
    ftm = sum(r["tm"] for r in sevts)
    ft = sum(r["n_truth"] for r in sevts)
    ff = sum(r["fake"] for r in sevts)
    feff = (ftc + ftm) / ft if ft else 0.0
    ffp = ff / ns if ns else 0.0
    print(f"\n  --- Summary ({ns} events, {mu_lbl}) ---")
    for lbl, cnt, ref in [
        ("truth PVs/evt", ant, None),
        ("  tc (clean)", atc, ant),
        ("  tm (merged)", atm, ant),
        ("  missed", atmiss, ant),
        ("reco PVs/evt", ac + am + as_ + af, None),
        ("  clean", ac, ant),
        ("  merged", am, ant),
        ("  split", as_, ant),
        ("  fake", af, ant),
    ]:
        pct = f"{100 * cnt / ref:5.1f}%" if ref and ref > 0 else "  --"
        print(f"  {lbl:<24} {cnt:>7.2f}  {pct}")
    print(f"  Eff={feff:.4f} ({ftc+ftm}/{ft})  FP={ffp:.4f}/evt  "
          f"sigma={sigma:.4f} mm")  # fmt: skip

    # --- Plots ---
    print("\n--- Generating Plots ---")
    plot_performance(
        per_event,
        overall_eff,
        has_mu,
        mode_label,
        outdir,
        title=f"PVF Performance -- Run 2 Data\n({mode_label})",
    )
    print(f"  Saved: {outdir / 'performance_plot.png'}")
    plot_stats(
        per_event,
        has_mu,
        mode_label,
        outdir,
        title=f"PVF Reco PV Categories -- Run 2 Data\n({mode_label})",
    )
    print(f"  Saved: {outdir / 'stats_histogram.png'}")

    # --- Save results ---
    results = dict(
        mode="k2h_v2_analytical_kde",
        sigma_vtx_vtx_mm=sigma,
        overall_efficiency=overall_eff,
        fp_rate_per_evt=fp_rate,
        n_events=nsc,
        total_truth_pvs=tot_truth,
        total_clean=tot_c,
        total_merged=tot_m,
        total_split=tot_s,
        total_fake=tot_f,
        total_truth_clean=tot_tc,
        total_truth_merged=tot_tm,
        total_truth_missed=tot_tmiss,
        per_event=per_event,
        pred_pvs_mm=all_pred,
        truth_pvs_mm=all_truth,
        pairwise_dz_mm=dz_arr,
        fit_params=popt.tolist() if popt is not None else None,
        k2h_checkpoint=args.k2h_model,
        root_file=args.root,
    )
    pkl_path = outdir / "eval_results.pkl"
    with open(pkl_path, "wb") as fp:
        pickle.dump(results, fp)
    print(f"  Saved: {pkl_path}")
    print(f"\n=== Done ===  (output: {args.output_dir})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K2H v2 evaluation on Run 2 data (analytical KDE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", default=_DEFAULT_ROOT, help="ROOT file path")
    parser.add_argument(
        "--k2h-model",
        default=_DEFAULT_K2H,
        dest="k2h_model",
        help="K2H v2 checkpoint path",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        dest="max_events",
        help="Max events to process (0=all)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval_k2h_v2_run2_analytical",
        dest="output_dir",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device (-1=CPU)",
    )
    main(parser.parse_args())
