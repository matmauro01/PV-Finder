#!/usr/bin/env python3
"""run_eval_pvf.py — PV-Finder evaluation on MC test set.

Modes
-----
  default   : KDE_A_z → K2H  (requires --k2h-model)
  --e2e-model: tracks → histogram end-to-end

Defaults
--------
  --h5        /share/lazy/qibinlei/recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5
  --root-truth /share/lazy/rocky/ATLAS_data/Latest_Sept2023/ATLAS_PVFinderData_TruthMatched.root
  --indices   configs/test_main_indices_2550evt.p
  --output-dir outputs/eval_pvf

Minimal invocation
------------------
  python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py --e2e-model model_weights/foo.pth

Outputs: resolution_plot.png, performance_plot.png, stats_histogram.png, eval_results.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import uproot
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from pv_finder.models.autoencoder_models import (  # noqa: E402
    MaskedDNN,
    UNet_1000,
    trackstoHists_UNet_1000,
)
from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from efficiency_res_optimized_atlas import (  # noqa: E402
    compare_res_reco,
    pv_locations_updated_res,
)
from plots_pvf import plot_performance, plot_resolution, plot_stats  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — must match training
# ---------------------------------------------------------------------------
Z_MIN, Z_MAX = -240.0, 240.0  # mm
N_BINS_FULL, N_BINS_SUB, N_SUBEVENTS = 12000, 1000, 12
BIN_WIDTH = (Z_MAX - Z_MIN) / N_BINS_FULL  # 0.04 mm/bin
THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH = 1e-2, 0.2, 3
INTEGRAL_THRESHOLD_RES = 0.5  # stricter threshold for sigma_vtx_vtx
FIT_P0 = [1000.0, 10.0, 30.0, 0.8]  # sigmoid fit initial params [a, b, c, rcc]

_DEFAULT_H5 = (
    "/share/lazy/qibinlei/"
    "recoTrackNPV_jets_pubindices_1000bins_incbounds_Target_Y_split.h5"
)
_DEFAULT_ROOT = (
    "/share/lazy/rocky/ATLAS_data/Latest_Sept2023/ATLAS_PVFinderData_TruthMatched.root"
)
_DEFAULT_QIBIN = "configs/qibin_test_main_indices_v2.p"

T2KDE_CONFIG = dict(
    input_size=7,
    hidden_nodes=[100, 100, 100, 100, 100],
    output_size=N_BINS_SUB,
    leaky_param=0.01,
    use_bn=False,
    use_drop=False,
    maskVal=-240.0,
    predScaleFactor=0.001,
    allow_negative_output=False,
)
K2H_CONFIG = dict(
    n=64,
    sc_mode="concat",
    dropout_p=0.25,
    d_selection="ConvBNrelu",
    u_selection="Up",
    n_features=1,
)
E2E_CONFIG = dict(
    n_InputFeatures=7,
    n_OutputFeatures=N_BINS_SUB,
    l_HiddenNodes=[100, 100, 100, 100, 100],
    n_LatentChannels=1,
    n_UNetChannels=64,
    sc_mode="concat",
    dropout=0.25,
    LeakyReLU_param=0.01,
    predScaleFactor=0.001,
    maskVal=-240.0,
    d_selection="ConvBNrelu",
    u_selection="Up",
)


def mm_to_bins(z: np.ndarray) -> np.ndarray:
    return (z - Z_MIN) / BIN_WIDTH


def load_root_truth(path: str) -> tuple:
    tree = uproot.open(path)["PVFinderData"]
    print(f"  ROOT: {path}  ({tree.num_entries} events)")
    return (
        tree["TruthVertex_z"].array(library="np"),
        tree["TruthVertex_nTracks"].array(library="np"),
        tree["ActualNumOfInt"].array(library="np"),
    )


def sigmoid_fit(x, a, b, c, rcc):
    return a / (1.0 + np.exp(b * (rcc - np.abs(x)))) + c


def load_ckpt(path: str, model: torch.nn.Module, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif hasattr(ckpt, "state_dict"):
        state = ckpt.state_dict()
    else:
        state = ckpt
    model.load_state_dict(state)
    if isinstance(ckpt, dict):
        ls = f"{ckpt['loss']:.6f}" if ckpt.get("loss") is not None else "N/A"
        n = sum(p.numel() for p in model.parameters())
        print(
            f"  ckpt {Path(path).name}: epoch={ckpt.get('epoch')} loss={ls} params={n:,}"
        )
    model.to(device).eval()


def get_truth_pvs(h5f: h5py.File, idx: int) -> np.ndarray:
    pv = h5f["pv"][idx]
    return pv[np.abs(pv) < Z_MAX].astype(np.float32)


def _stitch(out: torch.Tensor) -> np.ndarray:
    a = out.cpu().numpy()
    return (a if a.ndim > 1 else a[np.newaxis, :]).reshape(-1).astype(np.float32)


def run_k2h(
    kde: np.ndarray, model: torch.nn.Module, device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        return _stitch(
            model(torch.from_numpy(kde[:, np.newaxis, :]).float().to(device))
        )


def run_e2e(
    trk: np.ndarray, model: torch.nn.Module, device: torch.device
) -> np.ndarray:
    with torch.no_grad():
        return _stitch(model(torch.from_numpy(trk).float().to(device)))


def _evt_rec(eidx, nt, np_, c, m, s, f, eff, tc, tm, tmiss, mu):
    """Build a per-event result dict."""
    return dict(
        event_idx=eidx,
        n_truth=nt,
        n_pred=np_,
        clean=c,
        merged=m,
        split=s,
        fake=f,
        eff=eff,
        tc=tc,
        tm=tm,
        tmiss=tmiss,
        mu=mu,
    )


def main(args: argparse.Namespace) -> None:
    print("=" * 65)
    print("  PV-Finder MC Evaluation")
    print("=" * 65)

    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print(
            f"\nDevice: GPU {args.device} — {torch.cuda.get_device_name(args.device)}"
        )
    else:
        device = torch.device("cpu")
        print("\nDevice: CPU")

    if not args.e2e_model and not args.k2h_model:
        raise ValueError("Provide --k2h-model or --e2e-model.")

    print("\n--- Loading Models ---")
    k2h = e2e = None
    if args.e2e_model:
        e2e_type = getattr(args, "e2e_type", "v1")
        if e2e_type == "v2":
            t2kde_sub = MaskedDNN(**T2KDE_CONFIG)
            k2h_sub = UNet_1000_v2(n=64, n_features=1, dropout_p=0.0)
            e2e = TracksToHist_v2(t2kde_sub, k2h_sub)
        else:
            e2e = trackstoHists_UNet_1000(**E2E_CONFIG)
        load_ckpt(args.e2e_model, e2e, device)
        mode_label = f"E2E — tracks → histogram ({Path(args.e2e_model).stem})"
    else:
        if args.k2h_type == "v2":
            k2h = UNet_1000_v2(n=64, n_features=1, dropout_p=0.0)
        else:
            k2h = UNet_1000(**K2H_CONFIG)
        load_ckpt(args.k2h_model, k2h, device)
        mode_label = "K2H only (analytical KDE_A_z input)"

    # Load ROOT truth + qibin index map (always uses defaults)
    root_z = root_n = root_mu = qibin = None
    if args.root_truth:
        qibin_path = Path(_DEFAULT_QIBIN)
        if not qibin_path.exists():
            raise FileNotFoundError(
                f"qibin index file not found at '{qibin_path}'. "
                "Run from repo root or ensure configs/qibin_test_main_indices_v2.p exists."
            )
        print("\n--- Loading ROOT Truth (nTracks >= 2 filter) ---")
        root_z, root_n, root_mu = load_root_truth(args.root_truth)
        with open(qibin_path, "rb") as fp:
            qibin = list(pickle.load(fp))
        print(f"  qibin: {len(qibin)} entries  ({qibin_path})")

    with open(args.indices, "rb") as fp:
        event_indices = pickle.load(fp)
    n_events = len(event_indices)
    print(f"\n  {n_events} events  (h5 idx {event_indices[0]}..{event_indices[-1]})")
    print(f"\n--- Inference ({n_events} events × {N_SUBEVENTS} subevents) ---")

    all_pred, all_truth, pairwise_dz = [], [], []
    with h5py.File(args.h5, "r") as h5f:
        for i, eidx in enumerate(event_indices):
            eidx = int(eidx)
            s0, s1 = eidx * N_SUBEVENTS, eidx * N_SUBEVENTS + N_SUBEVENTS
            t_pvs = get_truth_pvs(h5f, eidx)
            if args.e2e_model:
                ph = run_e2e(h5f["tracks"][s0:s1], e2e, device)
            else:
                ph = run_k2h(h5f["kde_split"][s0:s1, 0, :], k2h, device)
            p_pvs = pv_locations_updated_res(
                ph, THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH
            )[0]
            p_pvs_r = pv_locations_updated_res(
                ph, THRESHOLD, INTEGRAL_THRESHOLD_RES, MIN_WIDTH
            )[0]
            p_pvs_r_sh = p_pvs_r.copy()
            np.random.shuffle(p_pvs_r_sh)
            for ii in range(len(p_pvs_r_sh)):
                for jj in range(ii + 1, len(p_pvs_r_sh)):
                    pairwise_dz.append(float(p_pvs_r_sh[ii] - p_pvs_r_sh[jj]))
            if i < 5 or i % 50 == 0:
                print(
                    f"  evt {i:3d}/{n_events}: truth={len(t_pvs)} pred={len(p_pvs)} max={ph.max():.4f}"
                )
            all_pred.append(p_pvs)
            all_truth.append(t_pvs)

    tp = sum(len(p) for p in all_pred)
    tt = sum(len(t) for t in all_truth)
    print(
        f"\n  done: pred={tp:,} ({tp / n_events:.1f}/evt) truth={tt:,} ({tt / n_events:.1f}/evt)"
    )

    # Resolution
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
        print(
            f"  sigma_vtx_vtx = {sigma:.4f} ± {serr:.4f} mm ({sigma / BIN_WIDTH:.1f} bins)"
        )
    except RuntimeError as exc:
        print(f"  WARNING: fit failed ({exc}). Default sigma={sigma} mm")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_resolution(
        dz_arr,
        sigma,
        popt,
        sigmoid_fit,
        mode_label,
        Path(args.output_dir),
        title=args.title,
    )
    print(f"  Saved: {Path(args.output_dir) / 'resolution_plot.png'}")

    # Performance metrics
    sig_bins = sigma / BIN_WIDTH
    src = "ROOT (nTracks>=2)" if root_z is not None else "h5 pv (no ntrks filter)"
    print(f"\n--- Performance Metrics (window={sigma:.4f} mm={sig_bins:.1f} bins) ---")
    print(f"  Truth source: {src}")
    tot_c = tot_m = tot_s = tot_f = tot_tc = tot_tm = tot_tmiss = tot_truth = 0
    per_event = []

    for i, (p_pvs, h5t) in enumerate(zip(all_pred, all_truth)):
        eidx = int(event_indices[i])
        if root_z is not None:
            ridx = qibin[i]
            ze = np.asarray(root_z[ridx], dtype=np.float64)
            ne = np.asarray(root_n[ridx], dtype=np.float64)
            mu = float(root_mu[ridx])
            t_pvs = mm_to_bins(ze[ne >= 2])
            p_cmp = mm_to_bins(p_pvs)
            win = sig_bins
        else:
            t_pvs, p_cmp, win, mu = h5t, p_pvs, sigma, None

        nt = len(t_pvs)
        if nt == 0:
            continue
        np_ = len(p_cmp)
        if np_ == 0:
            tot_truth += nt
            tot_tmiss += nt
            per_event.append(_evt_rec(eidx, nt, 0, 0, 0, 0, 0, 0.0, 0, 0, nt, mu))
            continue

        res, tc_arr, _ = compare_res_reco(t_pvs, p_cmp, win * np.ones(np_), debug=0)
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
                eidx,
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
            print(
                f"  evt {i:3d}: t={nt} p={np_} C={res.reco_clean} M={res.reco_merged} "
                f"S={res.reco_split} F={res.reco_fake} eff={eff:.3f}"
            )

    nsc = len(per_event)
    overall_eff = (tot_tc + tot_tm) / tot_truth if tot_truth else 0.0
    fp_rate = tot_f / nsc if nsc else 0.0

    MU_MIN, MU_MAX = 55, 65
    if root_z is not None:
        sevts = [
            r
            for r in per_event
            if r["mu"] is not None and MU_MIN <= round(r["mu"]) <= MU_MAX
        ]
        mu_lbl = f"μ∈[{MU_MIN},{MU_MAX}] (ActualNumOfInt)"
    else:
        sevts, mu_lbl = per_event, "all pileup"
    ns = len(sevts)
    print(f"\n  Summary filter: {mu_lbl} → {ns}/{nsc} events")

    def avg(k):
        return float(np.mean([r[k] for r in sevts])) if sevts else 0.0

    ac, am, as_, af = avg("clean"), avg("merged"), avg("split"), avg("fake")
    atc, atm, atmiss, ant = avg("tc"), avg("tm"), avg("tmiss"), avg("n_truth")
    ftc = sum(r["tc"] for r in sevts)
    ftm = sum(r["tm"] for r in sevts)
    ft = sum(r["n_truth"] for r in sevts)
    ff = sum(r["fake"] for r in sevts)
    feff = (ftc + ftm) / ft if ft else 0.0
    ffp = ff / ns if ns else 0.0
    print(f"\n  ─── Summary ({ns} events, {mu_lbl}) ───")
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
        pct = f"{100 * cnt / ref:5.1f}%" if ref and ref > 0 else "  —"
        print(f"  {lbl:<24} {cnt:>7.2f}  {pct}")
    print(
        f"  Eff={feff:.4f} ({ftc + ftm}/{ft})  FP={ffp:.4f}/evt  sigma={sigma:.4f} mm  "
        f"(overall eff={overall_eff:.4f} {tot_tc + tot_tm}/{tot_truth})"
    )

    print("\n--- Generating Plots ---")
    plot_performance(
        per_event,
        overall_eff,
        fp_rate,
        sigma,
        root_z is not None,
        mode_label,
        Path(args.output_dir),
        title=args.title,
    )
    print(f"  Saved: {Path(args.output_dir) / 'performance_plot.png'}")
    plot_stats(
        per_event,
        root_z is not None,
        mode_label,
        Path(args.output_dir),
        title=args.title,
    )
    print(f"  Saved: {Path(args.output_dir) / 'stats_histogram.png'}")

    results = dict(
        mode="e2e" if args.e2e_model else "stage2",
        sigma_vtx_vtx_mm=sigma, overall_efficiency=overall_eff, fp_rate_per_evt=fp_rate,
        n_events=nsc, total_truth_pvs=tot_truth,
        total_clean=tot_c, total_merged=tot_m, total_split=tot_s, total_fake=tot_f,
        total_truth_clean=tot_tc, total_truth_merged=tot_tm, total_truth_missed=tot_tmiss,
        per_event=per_event, pred_pvs_mm=all_pred, truth_pvs_mm=all_truth,
        pairwise_dz_mm=dz_arr,
        fit_params=popt.tolist() if popt is not None else None,
        k2h_checkpoint=args.k2h_model, e2e_checkpoint=args.e2e_model,
    )  # fmt: skip
    pkl_path = Path(args.output_dir) / "eval_results.pkl"
    with open(pkl_path, "wb") as fp:
        pickle.dump(results, fp)
    print(f"  Saved: {pkl_path}")
    print(f"\n=== Done ===  (output: {args.output_dir})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PV-Finder MC evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--h5",
        default=_DEFAULT_H5,
        help="HDF5 file path",
    )
    parser.add_argument("--k2h-model", default=None, dest="k2h_model")
    parser.add_argument("--k2h-type", default="v1", choices=["v1", "v2"],
                        dest="k2h_type", help="K2H model class (v1=UNet_1000, v2=UNet_1000_v2)")  # fmt: skip
    parser.add_argument("--e2e-model", default=None, dest="e2e_model")
    parser.add_argument("--e2e-type", default="v1", choices=["v1", "v2"],
                        dest="e2e_type", help="E2E model class (v1=trackstoHists_UNet_1000, v2=TracksToHist_v2)")  # fmt: skip
    parser.add_argument("--root-truth", default=None, dest="root_truth",
        help=f"ROOT truth for nTracks>=2 filter. Default: {_DEFAULT_ROOT}. "
             f"qibin map auto-loaded from {_DEFAULT_QIBIN}.")  # fmt: skip
    parser.add_argument("--indices", default="configs/test_main_indices_2550evt.p",
                        help="Test event indices pickle")  # fmt: skip
    parser.add_argument("--output-dir", default="outputs/eval_pvf", dest="output_dir")
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1=CPU)")
    parser.add_argument("--title", default="", help="Plot title (used for all plots)")
    main(parser.parse_args())
