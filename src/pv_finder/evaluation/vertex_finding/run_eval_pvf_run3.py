#!/usr/bin/env python3
"""PV-Finder evaluation on real/MC data (Run 2 / Run 3 / HLLHC)."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parents[4] / "src"))
from pv_finder.data.feature_loading import (  # noqa: E402
    MASK_VAL,
    N_SUBEVENTS,
    build_run3_subevent_tensor,
)
from pv_finder.data.run3_io import (  # noqa: E402
    Run3Event,
    load_run3_from_npz,
    load_run3_from_root,
)
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
    suppress_neighbor_peaks,
)
from plots_pvf import (  # noqa: E402
    plot_category_counts,
    plot_performance,
    plot_reco_vs_mu,
    plot_resolution,
    plot_stats,
)

Z_MIN, Z_MAX = -240.0, 240.0  # mm
N_BINS_FULL, N_BINS_SUB = 12000, 1000
BIN_WIDTH = (Z_MAX - Z_MIN) / N_BINS_FULL  # 0.04 mm/bin
THRESHOLD, INTEGRAL_THRESHOLD, MIN_WIDTH = 1e-2, 0.5, 3
MODEL_PAD_VAL = -240.0

# fmt: off
T2KDE_CONFIG = dict(input_size=7, hidden_nodes=[100]*5, output_size=N_BINS_SUB,
    leaky_param=0.01, use_bn=False, use_drop=False, maskVal=-240.0,
    predScaleFactor=0.001, allow_negative_output=False)
K2H_CONFIG = dict(n=64, sc_mode="concat", dropout_p=0.25,
    d_selection="ConvBNrelu", u_selection="Up", n_features=1)
E2E_CONFIG = dict(n_InputFeatures=7, n_OutputFeatures=N_BINS_SUB,
    l_HiddenNodes=[100]*5, n_LatentChannels=1, n_UNetChannels=64,
    sc_mode="concat", dropout=0.25, LeakyReLU_param=0.01,
    predScaleFactor=0.001, maskVal=-240.0, d_selection="ConvBNrelu",
    u_selection="Up")
# fmt: on


def mm_to_bins(z: np.ndarray) -> np.ndarray:
    """Convert mm positions to bin indices."""
    return (z - Z_MIN) / BIN_WIDTH


def sigmoid_fit(x: np.ndarray, a: float, b: float, c: float, rcc: float) -> np.ndarray:
    """Sigmoid function for resolution fit."""
    return a / (1.0 + np.exp(b * (rcc - np.abs(x)))) + c


def load_ckpt(path: str, model: torch.nn.Module, device: torch.device) -> None:
    """Load checkpoint into model, handling model_state key and legacy .pyt paths."""
    import types

    import pv_finder.models.autoencoder_models as _am  # noqa: E402

    if "model" not in sys.modules:
        sys.modules["model"] = types.ModuleType("model")
    sys.modules["model.autoencoder_models"] = _am
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


def _repad(tensor: np.ndarray) -> np.ndarray:
    """Replace MASK_VAL (-999999) with MODEL_PAD_VAL (-240.0)."""
    out = tensor.copy().astype(np.float32)
    out[:, out[1, :] <= (MASK_VAL + 1)] = MODEL_PAD_VAL
    return out


def _pad_to_length(tensor: np.ndarray, length: int) -> np.ndarray:
    """Pad (7, N) to (7, length) with MODEL_PAD_VAL, or truncate."""
    _, n_tracks = tensor.shape
    if n_tracks >= length:
        return tensor[:, :length]
    padded = np.full((7, length), MODEL_PAD_VAL, dtype=np.float32)
    padded[:, :n_tracks] = tensor
    return padded


def build_subevent_inputs(event: Run3Event) -> list[np.ndarray]:
    """Build 12 subevent tensors for one Run 3 event, re-padded for model."""
    subevents = []
    for si in range(N_SUBEVENTS):
        tensor, _ = build_run3_subevent_tensor(
            event.z0, event.d0, event.d0_err, event.z0_err, event.d0_z0_cov, si
        )
        subevents.append(_repad(tensor))
    return subevents


def _stitch(out: torch.Tensor) -> np.ndarray:
    """Stitch model output to flat 12000-bin histogram."""
    a = out.cpu().numpy()
    return (a if a.ndim > 1 else a[np.newaxis, :]).reshape(-1).astype(np.float32)


def run_inference(
    subevents: list[np.ndarray],
    device: torch.device,
    t2kde: torch.nn.Module | None = None,
    k2h: torch.nn.Module | None = None,
    e2e: torch.nn.Module | None = None,
) -> np.ndarray:
    """Run model inference on 12 subevent tensors -> (12000,) histogram."""
    mx = max(t.shape[1] for t in subevents)
    padded = np.stack([_pad_to_length(t, mx) for t in subevents])
    inp = torch.from_numpy(padded).float().to(device)
    with torch.no_grad():
        if e2e is not None:
            hist = e2e(inp)
        else:
            hist = k2h(t2kde(inp).unsqueeze(1))
    return _stitch(hist)


def _evt_rec(eidx, nt, np_, c, m, s, f, eff, tc, tm, tmiss, mu, beam_z,  # noqa: PLR0913
             n_amvf=None, amvf_c=None, amvf_m=None, amvf_s=None, amvf_f=None):  # fmt: skip
    """Build per-event result dict."""
    return dict(event_idx=eidx, n_truth=nt, n_pred=np_, clean=c, merged=m,
                split=s, fake=f, eff=eff, tc=tc, tm=tm, tmiss=tmiss,
                mu=mu, beam_z=beam_z, n_amvf=n_amvf,
                amvf_clean=amvf_c, amvf_merged=amvf_m,
                amvf_split=amvf_s, amvf_fake=amvf_f)  # fmt: skip


def main(args: argparse.Namespace) -> None:  # noqa: C901, PLR0912, PLR0915
    print("=" * 65, "\n  PV-Finder Run 3 / HL-LHC Evaluation\n", "=" * 65, sep="\n")
    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
        print(
            f"\nDevice: GPU {args.device} -- {torch.cuda.get_device_name(args.device)}"
        )
    else:
        device = torch.device("cpu")
        print("\nDevice: CPU")
    has_pipeline, has_e2e = (
        bool(args.t2kde_model and args.k2h_model),
        args.e2e_model is not None,
    )
    if not has_pipeline and not has_e2e:
        raise ValueError("Provide --e2e-model OR both --t2kde-model and --k2h-model.")
    print("\n--- Loading Models ---")
    t2kde = k2h = e2e = None
    if has_e2e:
        e2e_type = getattr(args, "e2e_type", "v1")
        if e2e_type == "v2":
            e2e = TracksToHist_v2(
                MaskedDNN(**T2KDE_CONFIG),
                UNet_1000_v2(n=64, n_features=1, dropout_p=0.0),
            )
        else:
            cfg = dict(E2E_CONFIG)
            if getattr(args, "e2e_wide", False):
                cfg.update(n_UNetChannels=96, l_HiddenNodes=[128] * 5)
            e2e = trackstoHists_UNet_1000(**cfg)
        load_ckpt(args.e2e_model, e2e, device)
        mode_label = f"E2E ({Path(args.e2e_model).stem})"
    else:
        t2kde = MaskedDNN(**T2KDE_CONFIG)
        load_ckpt(args.t2kde_model, t2kde, device)
        k2h = (
            UNet_1000_v2(n=64, n_features=1, dropout_p=0.0)
            if args.k2h_type == "v2"
            else UNet_1000(**K2H_CONFIG)
        )
        load_ckpt(args.k2h_model, k2h, device)
        mode_label = (
            f"T2KDE+K2H ({Path(args.t2kde_model).stem} + {Path(args.k2h_model).stem})"
        )
    print("\n--- Loading Data ---")
    load_kw = dict(
        max_events=args.max_events,
        min_tracks=args.min_tracks,
        min_amvf_vtx=args.min_amvf_vtx,
    )
    if args.root:
        events = load_run3_from_root(
            args.root,
            **load_kw,
            entry_start=args.entry_start,
            entry_stop=args.entry_stop,
        )
    else:
        events = load_run3_from_npz(args.npz, **load_kw)
    n_events = len(events)
    if n_events == 0:
        print("ERROR: no events loaded after filtering.")
        sys.exit(1)
    has_mu = any(e.mu is not None for e in events)
    has_truth = events[0].truth_z is not None
    if has_truth:
        print("  MC truth detected — TruthVertex as truth, AMVF evaluated separately")
    else:
        print("  No MC truth — using AMVF (RecoVertex) as truth reference")
    if args.smooth_sigma > 0:
        print(f"\n  Peak-finding smoothing: sigma={args.smooth_sigma} bins "
              f"({args.smooth_sigma * BIN_WIDTH:.3f} mm)")  # fmt: skip
    if args.nms_min_sep > 0:
        print(f"  NMS: min_sep={args.nms_min_sep} mm, "
              f"max_ratio={args.nms_max_ratio}")  # fmt: skip
    print(f"\n--- Inference ({n_events} events x {N_SUBEVENTS} subevents) ---")
    all_pred: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    all_heights: list[np.ndarray] = []
    all_hists: list[np.ndarray] = []
    pairwise_dz: list[float] = []

    for i, event in enumerate(events):
        subevents = build_subevent_inputs(event)
        ph = run_inference(subevents, device, t2kde=t2kde, k2h=k2h, e2e=e2e)
        ph_peaks = (
            gaussian_filter1d(ph, sigma=args.smooth_sigma)
            if args.smooth_sigma > 0
            else ph
        )
        p_pvs, p_hts, *_ = pv_locations_updated_res(
            ph_peaks, args.peak_threshold, args.integral_threshold, MIN_WIDTH
        )
        p_pvs_r, p_hts_r, *_ = pv_locations_updated_res(
            ph_peaks, args.peak_threshold, args.integral_threshold_res, MIN_WIDTH
        )
        if args.nms_min_sep > 0:
            keep = suppress_neighbor_peaks(
                p_pvs, p_hts, args.nms_min_sep, args.nms_max_ratio
            )
            p_pvs, p_hts = p_pvs[keep], p_hts[keep]
            keep_r = suppress_neighbor_peaks(
                p_pvs_r, p_hts_r, args.nms_min_sep, args.nms_max_ratio
            )
            p_pvs_r, p_hts_r = p_pvs_r[keep_r], p_hts_r[keep_r]
        p_pvs_r_sh = p_pvs_r.copy()
        np.random.shuffle(p_pvs_r_sh)
        for ii in range(len(p_pvs_r_sh)):
            for jj in range(ii + 1, len(p_pvs_r_sh)):
                pairwise_dz.append(float(p_pvs_r_sh[ii] - p_pvs_r_sh[jj]))
        if has_truth:
            t_pvs = (
                event.truth_z.copy()
            )  # MC truth (detector frame, no beam correction)
        else:
            t_pvs = event.amvf_z.copy()  # fallback: AMVF as truth
            if not args.no_correct_beam:
                t_pvs = t_pvs - event.beam_z
        if i < 5 or i % 50 == 0:
            print(f"  evt {i:3d}/{n_events}: truth={len(t_pvs)} "
                  f"pred={len(p_pvs)} max={ph.max():.4f}")  # fmt: skip
        all_pred.append(p_pvs)
        all_heights.append(p_hts)
        all_truth.append(t_pvs)
        if args.save_histograms:
            all_hists.append(ph.copy())

    tp = sum(len(p) for p in all_pred)
    tt = sum(len(t) for t in all_truth)
    print(f"\n  done: pred={tp:,} ({tp/n_events:.1f}/evt) "
          f"truth={tt:,} ({tt/n_events:.1f}/evt)")  # fmt: skip

    # --- Resolution (sigma_vtx_vtx) ---
    print("\n--- Resolution (sigma_vtx_vtx) ---")
    dz_arr = np.array(pairwise_dz, dtype=np.float64)
    bins_r = np.linspace(-6.0, 6.0, 61)
    ctrs = 0.5 * (bins_r[:-1] + bins_r[1:])
    cnts, _ = np.histogram(dz_arr, bins=bins_r)
    sigma, popt = 0.5, None
    baseline, dip = float(np.median(cnts)), float(np.median(cnts)) - float(cnts.min())
    p0 = [max(dip, 1.0), 10.0, max(baseline, 1.0), 0.5]
    try:
        popt, pcov = curve_fit(sigmoid_fit, ctrs, cnts.astype(float), p0=p0,
            maxfev=10000, bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))  # fmt: skip
        sigma = float(abs(popt[3]))
        serr = float(np.sqrt(np.diag(pcov))[3])
        print(f"  sigma_vtx_vtx = {sigma:.4f} +/- {serr:.4f} mm "
              f"({sigma/BIN_WIDTH:.1f} bins)")  # fmt: skip
    except RuntimeError as exc:
        print(f"  WARNING: fit failed ({exc}). Default sigma={sigma} mm")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    ds = args.dataset_name or "Real Data"
    plot_resolution(dz_arr, sigma, popt, sigmoid_fit, mode_label, outdir,
        title=args.title or f"PVF Resolution — {ds}\n({mode_label})")  # fmt: skip

    # --- Performance metrics ---
    sig_bins = sigma / BIN_WIDTH
    tsrc = "MC truth (TruthVertex, nTracks>=2)" if has_truth else "AMVF (nTracks>=2)"
    print(f"\n--- Performance (window={sigma:.4f} mm={sig_bins:.1f} bins) ---")
    print(f"  Truth source: {tsrc}")
    tot_c = tot_m = tot_s = tot_f = tot_tc = tot_tm = tot_tmiss = tot_truth = 0
    per_event: list[dict] = []

    for i, (p_pvs, t_pvs) in enumerate(zip(all_pred, all_truth)):
        event = events[i]
        nt = len(t_pvs)
        if nt == 0:
            continue
        t_bins, p_bins = mm_to_bins(t_pvs), mm_to_bins(p_pvs)
        np_ = len(p_bins)
        mu = event.mu if event.mu is not None else float(nt)
        if np_ == 0:
            tot_truth += nt
            tot_tmiss += nt
            # AMVF categories even when PV-Finder has 0 predictions
            ac0 = am0 = as0 = af0 = n_amvf0 = None
            if has_truth:
                amvf_bins = mm_to_bins(event.amvf_z)
                n_amvf0 = len(amvf_bins)
                ac0 = am0 = as0 = af0 = 0
                if n_amvf0 > 0:
                    r0, _, _ = compare_res_reco(
                        t_bins, amvf_bins, sig_bins * np.ones(n_amvf0), debug=0
                    )
                    ac0, am0, as0, af0 = (r0.reco_clean, r0.reco_merged,
                                           r0.reco_split, r0.reco_fake)  # fmt: skip
            per_event.append(_evt_rec(
                event.event_idx, nt, 0, 0, 0, 0, 0, 0.0, 0, 0, nt,
                mu, event.beam_z, n_amvf0, ac0, am0, as0, af0))  # fmt: skip
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
        # AMVF categories: compare AMVF reco against truth (when truth != AMVF)
        # No beam correction on AMVF here — both truth and AMVF are in
        # detector frame, matching the MC eval (run_eval_pvf.py) behavior.
        ac = am = as_ = af = n_amvf = None
        if has_truth:
            amvf_bins = mm_to_bins(event.amvf_z)
            n_amvf = len(amvf_bins)
            ac = am = as_ = af = 0
            if n_amvf > 0:
                res_a, _, _ = compare_res_reco(
                    t_bins, amvf_bins, sig_bins * np.ones(n_amvf), debug=0
                )
                ac, am, as_, af = (res_a.reco_clean, res_a.reco_merged,
                                   res_a.reco_split, res_a.reco_fake)  # fmt: skip
        per_event.append(_evt_rec(
            event.event_idx, nt, np_, res.reco_clean, res.reco_merged,
            res.reco_split, res.reco_fake, eff, ntc, ntm, ntmiss,
            mu, event.beam_z, n_amvf, ac, am, as_, af))  # fmt: skip
        if i < 5 or i % 50 == 0:
            print(f"  evt {i:3d}: t={nt} p={np_} C={res.reco_clean} "
                  f"M={res.reco_merged} S={res.reco_split} "
                  f"F={res.reco_fake} eff={eff:.3f}")  # fmt: skip

    nsc = len(per_event)
    overall_eff = (tot_tc + tot_tm) / tot_truth if tot_truth else 0.0
    fp_rate = tot_f / nsc if nsc else 0.0

    # fmt: off
    MU_MIN, MU_MAX = args.mu_min, args.mu_max
    if has_mu:
        sevts = [r for r in per_event
                 if r["mu"] is not None and MU_MIN <= round(r["mu"]) <= MU_MAX]
        mu_lbl = f"mu in [{MU_MIN},{MU_MAX}] (ActualNumOfInt)"
    else:
        sevts, mu_lbl = per_event, "all pileup"
    ns = len(sevts)
    print(f"\n  Summary filter: {mu_lbl} -> {ns}/{nsc} events")
    def avg(k): return float(np.mean([r[k] for r in sevts])) if sevts else 0.0
    ac, am, as_, af = avg("clean"), avg("merged"), avg("split"), avg("fake")
    atc, atm, atmiss, ant = avg("tc"), avg("tm"), avg("tmiss"), avg("n_truth")
    ftc, ftm = sum(r["tc"] for r in sevts), sum(r["tm"] for r in sevts)
    ft, ff = sum(r["n_truth"] for r in sevts), sum(r["fake"] for r in sevts)
    feff = (ftc + ftm) / ft if ft else 0.0
    ffp = ff / ns if ns else 0.0
    print(f"\n  --- Summary ({ns} events, {mu_lbl}) ---")
    for lbl, cnt, ref in [("truth PVs/evt", ant, None), ("  tc (clean)", atc, ant),
            ("  tm (merged)", atm, ant), ("  missed", atmiss, ant),
            ("reco PVs/evt", ac+am+as_+af, None), ("  clean", ac, ant),
            ("  merged", am, ant), ("  split", as_, ant), ("  fake", af, ant)]:
        pct = f"{100*cnt/ref:5.1f}%" if ref and ref > 0 else "  --"
        print(f"  {lbl:<24} {cnt:>7.2f}  {pct}")
    print(f"  Eff={feff:.4f} ({ftc+ftm}/{ft})  FP={ffp:.4f}/evt  "
          f"sigma={sigma:.4f} mm  (overall eff={overall_eff:.4f} "
          f"{tot_tc+tot_tm}/{tot_truth})")
    if has_truth:
        a_ant = avg("n_amvf")
        aac, aam, aas, aaf = (avg("amvf_clean"), avg("amvf_merged"),
                               avg("amvf_split"), avg("amvf_fake"))
        print(f"\n  --- AMVF Summary ({ns} events) ---")
        for lbl, cnt, ref in [("AMVF PVs/evt", a_ant, None), ("  clean", aac, ant),
                ("  merged", aam, ant), ("  split", aas, ant), ("  fake", aaf, ant)]:
            pct = f"{100*cnt/ref:5.1f}%" if ref and ref > 0 else "  --"
            print(f"  {lbl:<24} {cnt:>7.2f}  {pct}")
    # fmt: on

    print("\n--- Generating Plots ---")
    t = args.title or f"PVF — {ds}\n({mode_label})"
    plot_performance(per_event, overall_eff, fp_rate, sigma, has_mu,
                     mode_label, outdir, title=t)  # fmt: skip
    plot_stats(per_event, has_mu, mode_label, outdir, title=t)
    if has_mu:
        plot_reco_vs_mu(per_event, mode_label, outdir, title=t)
    ckpt = Path(args.e2e_model or args.k2h_model or "").stem
    truth_avg = ant if has_truth else None
    plot_category_counts(per_event, mode_label, outdir, title="",
        eval_label=f"ckpt: {ckpt}\nintegral_threshold = {args.integral_threshold}",
        mu_min=args.mu_min, mu_max=args.mu_max,
        all_events=args.mu_min >= 100,
        truth_pvs_per_evt=truth_avg)  # fmt: skip
    print(f"  Saved plots to: {outdir}")

    # fmt: off
    results = dict(
        mode="e2e" if has_e2e else "pipeline", sigma_vtx_vtx_mm=sigma,
        overall_efficiency=overall_eff, fp_rate_per_evt=fp_rate, has_truth=has_truth,
        n_events=nsc, total_truth_pvs=tot_truth,
        total_clean=tot_c, total_merged=tot_m, total_split=tot_s, total_fake=tot_f,
        total_truth_clean=tot_tc, total_truth_merged=tot_tm, total_truth_missed=tot_tmiss,
        per_event=per_event, pred_pvs_mm=all_pred, pred_heights=all_heights,
        truth_pvs_mm=all_truth, histograms=all_hists if args.save_histograms else None,
        pairwise_dz_mm=dz_arr, fit_params=popt.tolist() if popt is not None else None,
        t2kde_checkpoint=args.t2kde_model, k2h_checkpoint=args.k2h_model,
        e2e_checkpoint=args.e2e_model, data_source="root" if args.root else "npz",
        correct_beam=not args.no_correct_beam, smooth_sigma=args.smooth_sigma,
        nms_min_sep=args.nms_min_sep, nms_max_ratio=args.nms_max_ratio)
    # fmt: on
    pkl_path = outdir / "eval_results.pkl"
    with open(pkl_path, "wb") as fp:
        pickle.dump(results, fp)
    print(f"  Saved: {pkl_path}\n=== Done ===  (output: {args.output_dir})")


def _cli() -> argparse.Namespace:
    """Parse CLI arguments."""
    a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz")
    g.add_argument("--root")
    a.add_argument("--e2e-model", default=None, dest="e2e_model")
    a.add_argument("--e2e-type", default="v1", choices=["v1", "v2"], dest="e2e_type")
    a.add_argument("--t2kde-model", default=None, dest="t2kde_model")
    a.add_argument("--k2h-model", default=None, dest="k2h_model")
    a.add_argument("--k2h-type", default="v1", choices=["v1", "v2"], dest="k2h_type")
    a.add_argument("--max-events", type=int, default=0, dest="max_events")
    a.add_argument("--min-tracks", type=int, default=1, dest="min_tracks")
    a.add_argument("--min-amvf-vtx", type=int, default=1, dest="min_amvf_vtx")
    a.add_argument("--entry-start", type=int, default=0, dest="entry_start")
    a.add_argument("--entry-stop", type=int, default=None, dest="entry_stop")
    a.add_argument("--no-correct-beam", action="store_true")
    a.add_argument("--output-dir", default="outputs/eval_pvf_run3")
    a.add_argument("--device", type=int, default=0)
    a.add_argument("--smooth-sigma", type=float, default=0.0)
    a.add_argument("--nms-min-sep", type=float, default=0.0)
    a.add_argument("--nms-max-ratio", type=float, default=0.3)
    a.add_argument("--mu-min", type=int, default=55)
    a.add_argument("--mu-max", type=int, default=65)
    a.add_argument("--peak-threshold", type=float, default=THRESHOLD)
    a.add_argument("--integral-threshold", type=float, default=INTEGRAL_THRESHOLD)
    a.add_argument("--integral-threshold-res", type=float, default=0.5)
    a.add_argument("--e2e-wide", action="store_true")
    a.add_argument("--save-histograms", action="store_true")
    a.add_argument("--title", default="")
    a.add_argument("--dataset-name", default="", dest="dataset_name")
    return a.parse_args()


if __name__ == "__main__":
    main(_cli())
