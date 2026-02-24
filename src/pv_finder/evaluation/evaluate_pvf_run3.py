"""PV-Finder evaluation on Run 3 data (ROOT input, AMVF truth).

Unlike the MC evaluation (evaluate_pvf.py), this script:
1. Loads raw tracks from a ROOT file via uproot (no H5 subevents).
2. Runs inference event-by-event, building subevent tensors on the fly.
3. Uses AMVF reconstructed vertices as truth (beam-spot-corrected).
4. Overlays PVF and AMVF pairwise distances on the resolution plot.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit
from tqdm import tqdm

from pv_finder.evaluation.vertex_matching import compare_res_reco, fit_func_resolution
from pv_finder.utils.constants import (
    BIN_WIDTH_MM,
    BINS_PER_SUBEVENT,
    N_SUBEVENTS,
    Z_MIN,
)
from pv_finder.utils.peak_finding import pv_locations_updated_res

# ─── Subevent geometry (must match training) ──────────────────────────────────
_SUBEVENT_WIDTH_MM: float = 40.0
_N_FEATURES: int = 7
_N_TRACKS_PER_SUB: int = 100
_MASK_VAL: float = -999999.0
_SUBEVENT_STARTS: list[float] = [
    Z_MIN + i * _SUBEVENT_WIDTH_MM for i in range(N_SUBEVENTS)
]


# ─── Model loading ────────────────────────────────────────────────────────────


def _load_model(model_path: str | Path, device: torch.device) -> torch.nn.Module:
    """Load a PVF model, handling old 'model' namespace checkpoints."""
    import pv_finder.models.autoencoder_models as _ae

    if "model" not in sys.modules:
        _m = types.ModuleType("model")
        _m.autoencoder_models = _ae  # type: ignore[attr-defined]
        sys.modules["model"] = _m
        sys.modules["model.autoencoder_models"] = _ae

    model = torch.load(str(model_path), map_location="cpu", weights_only=False)
    model = model.to(device)
    model.eval()
    return model


# ─── ROOT helpers ─────────────────────────────────────────────────────────────


def _open_tree(root_path: str | Path) -> object:
    """Open PVFinderData tree, trying primary then fallback name."""
    import uproot

    f = uproot.open(str(root_path))
    for name in ("PVFinderData;1", "PVFinderData"):
        if name in f:
            return f[name]
    raise KeyError(f"Tree not found in {root_path}. Keys: {list(f.keys())}")


def _extract_beam_z(raw: object) -> float:
    """Extract scalar BeamPosZ from a scalar or 1-element array."""
    arr = np.atleast_1d(np.asarray(raw, dtype=np.float64))
    return float(arr[0]) if len(arr) > 0 else 0.0


# ─── Subevent tensor building ─────────────────────────────────────────────────


def _build_subevent_tensors(
    z0: np.ndarray,
    d0: np.ndarray,
    d0_err: np.ndarray,
    z0_err: np.ndarray,
    d0_z0_cov: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Build (sub_idx, tensor) pairs from raw event track arrays.

    Each subevent yields one tensor per 100-track chunk (padded with _MASK_VAL).
    Chunks share the same sub_idx; their model outputs are summed.
    """
    _empty = np.full((_N_FEATURES, _N_TRACKS_PER_SUB), _MASK_VAL, dtype=np.float32)
    result: list[tuple[int, np.ndarray]] = []
    for si in range(N_SUBEVENTS):
        z_start = _SUBEVENT_STARTS[si]
        z_end = z_start + _SUBEVENT_WIDTH_MM
        mask = (z0 >= z_start) & (z0 < z_end)
        n_trk = int(np.sum(mask))
        if n_trk == 0:
            result.append((si, _empty.copy()))
            continue
        srt = np.argsort(z0[mask])
        z0_s = z0[mask][srt]
        d0_s = d0[mask][srt]
        d0e_s = d0_err[mask][srt]
        z0e_s = z0_err[mask][srt]
        cv_s = d0_z0_cov[mask][srt]
        for cs in range(0, n_trk, _N_TRACKS_PER_SUB):
            ce = min(cs + _N_TRACKS_PER_SUB, n_trk)
            nc = ce - cs
            t = np.full((_N_FEATURES, _N_TRACKS_PER_SUB), _MASK_VAL, dtype=np.float32)
            t[0, :nc] = d0_s[cs:ce]
            t[1, :nc] = z0_s[cs:ce]
            t[2, :nc] = d0e_s[cs:ce]
            t[3, :nc] = z0e_s[cs:ce]
            t[4, :nc] = cv_s[cs:ce]
            t[5, :] = z_start
            t[6, :] = z_end
            result.append((si, t))
    return result


# ─── Inference ────────────────────────────────────────────────────────────────


def run_inference_event(
    model: torch.nn.Module,
    tensors: list[tuple[int, np.ndarray]],
    device: torch.device,
) -> np.ndarray:
    """Run model on one event; return 12000-bin histogram.

    Chunk outputs for the same subevent are summed.
    """
    sub_out: dict[int, np.ndarray] = {
        si: np.zeros(BINS_PER_SUBEVENT, dtype=np.float32) for si in range(N_SUBEVENTS)
    }
    with torch.no_grad():
        for si, t in tensors:
            inp = torch.from_numpy(t).unsqueeze(0).float().to(device)
            out = model(inp).cpu().numpy().squeeze()
            if out.shape == ():
                out = out.reshape(BINS_PER_SUBEVENT)
            sub_out[si] += out[:BINS_PER_SUBEVENT]
    return np.concatenate([sub_out[si] for si in range(N_SUBEVENTS)], axis=0)


# ─── Resolution helpers ───────────────────────────────────────────────────────


def _pairwise_distances(z_arr: np.ndarray) -> list[float]:
    """Shuffled pairwise distances between all elements of z_arr."""
    z = z_arr.copy()
    np.random.shuffle(z)
    return [float(z[a] - z[b]) for a in range(len(z) - 1) for b in range(a + 1, len(z))]


def _fit_sigma(distances: np.ndarray) -> tuple[float, float]:
    """Fit sigmoid to histogram; fallback to std of |d|<2 mm."""
    if len(distances) < 10:
        return 0.0, 0.0
    counts, edges = np.histogram(distances, bins=61, range=(-6.0, 6.0))
    centers = 0.5 * (edges[:-1] + edges[1:])
    cf = counts.astype(float)
    try:
        p0 = [float(np.max(cf)), 10.0, float(np.min(cf)), 0.5]
        popt, pcov = curve_fit(
            fit_func_resolution, centers[1:], cf[1:], p0=p0, maxfev=10_000
        )
        return float(abs(popt[3])), float(np.sqrt(np.diag(pcov))[3])
    except RuntimeError:
        close = distances[np.abs(distances) < 2.0]
        return (float(np.std(close)) if len(close) > 0 else 0.0), 0.0


# ─── Resolution plot (overlays PVF and AMVF) ──────────────────────────────────


def make_run3_resolution_plot(
    pvf_distances: list[float],
    amvf_distances: list[float],
    output_dir: Path,
) -> tuple[float, float, float, float]:
    """Overlay PVF/AMVF pairwise-distance histograms with sigmoid fits.

    Returns (sigma_pvf, sigma_pvf_err, sigma_amvf, sigma_amvf_err) in mm.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pvf_arr = np.asarray(pvf_distances, dtype=float)
    amvf_arr = np.asarray(amvf_distances, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 8))
    x_fit = np.linspace(-6.0, 6.0, 1000)
    results = []

    for arr, color, edge, fit_color, lbl in (
        (pvf_arr, "steelblue", "darkblue", "navy", "PV-Finder"),
        (amvf_arr, "orange", "darkorange", "darkorange", "AMVF"),
    ):
        counts, edges, _ = ax.hist(
            arr,
            bins=61,
            range=(-6.0, 6.0),
            color=color,
            alpha=0.5,
            edgecolor=edge,
            linewidth=0.8,
            label=lbl,
        )
        centers = 0.5 * (edges[:-1] + edges[1:])
        cf = counts.astype(float)
        try:
            p0 = [float(np.max(cf)), 10.0, float(np.min(cf)), 0.5]
            popt, pcov = curve_fit(
                fit_func_resolution, centers[1:], cf[1:], p0=p0, maxfev=10_000
            )
            sigma, sigma_err = float(abs(popt[3])), float(np.sqrt(np.diag(pcov))[3])
            ax.plot(
                x_fit,
                fit_func_resolution(x_fit, *popt),
                color=fit_color,
                linestyle="-",
                linewidth=2.5,
                label=f"{lbl} fit: σ = {sigma:.2f} mm",
                zorder=10,
            )
        except RuntimeError:
            close = arr[np.abs(arr) < 2.0]
            sigma = float(np.std(close)) if len(close) > 0 else 0.0
            sigma_err = 0.0
        results.extend([sigma, sigma_err])

    ax.set_xlabel(r"$\Delta z_{\mathrm{vtx-vtx}}$ [mm]", fontsize=18)
    ax.set_ylabel("Counts", fontsize=18)
    ax.set_title("Pairwise Vertex Distances — PVF vs AMVF (Run 3)", fontsize=16, pad=15)
    ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=-50)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(
            output_dir / f"deltaz_resolution.{fmt}", dpi=300, bbox_inches="tight"
        )
    plt.close(fig)
    return results[0], results[1], results[2], results[3]


# ─── Bar chart ────────────────────────────────────────────────────────────────


def _make_category_bar(summary: dict, output_dir: Path) -> None:
    """Bar chart of PVF vertex categories with percentages + AMVF efficiency."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cats = ["Total", "Clean", "Merged", "Split", "Fake"]
    counts = [
        summary["n_reco"],
        summary["clean"],
        summary["merged"],
        summary["split"],
        summary["fake"],
    ]
    total = summary.get("n_reco", 1) or 1
    colors = ["#7f8c8d", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(cats, counts, color=colors, edgecolor="black", linewidth=0.8)
    for idx, (bar, count) in enumerate(zip(bars, counts)):
        label = (
            f"{count}\n(100%)"
            if idx == 0
            else f"{count}\n({100.0 * count / total:.1f}%)"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    amvf_eff = summary.get("amvf_efficiency", 0.0)
    n_amvf = summary.get("n_amvf_truth", 0)
    ax.text(
        0.5,
        1.04,
        f"AMVF efficiency (clean+merged)/n_amvf: {amvf_eff:.3f}  (n_amvf={n_amvf})",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=11,
    )
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title("PVF Vertex Classification (Run 3)", fontsize=16, pad=25)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fig.savefig(output_dir / "pvf_category_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─── Main evaluation logic ────────────────────────────────────────────────────


def evaluate_run3(
    model_path: str | Path,
    root_file: str | Path,
    output_dir: str | Path,
    n_events: int = 2000,
    device: torch.device | None = None,
    sigma_vtx_vtx: float | None = None,
    threshold: float = 0.01,
    integral_threshold: float = 0.5,
    min_width: int = 3,
) -> dict:
    """End-to-end Run 3 evaluation. Returns the summary dict."""
    if device is None:
        device = torch.device("cpu")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading model...")
    model = _load_model(model_path, device)

    print("[2/4] Running inference on ROOT data...")
    tree = _open_tree(root_file)
    n_events = min(n_events, tree.num_entries)
    print(f"      Using {n_events} events")

    all_preds: list[np.ndarray] = []
    all_amvf: list[np.ndarray] = []
    _br = [
        "RecoTrack_z0",
        "RecoTrack_d0",
        "RecoTrack_ErrD0",
        "RecoTrack_ErrZ0",
        "RecoTrack_ErrD0Z0",
        "RecoVertex_z",
        "RecoVertex_nTracks",
        "BeamPosZ",
    ]
    event_data = next(
        iter(tree.iterate(_br, step_size=n_events, entry_stop=n_events, library="np"))
    )

    for i in tqdm(range(n_events), desc="Inference"):
        z0 = np.asarray(event_data["RecoTrack_z0"][i], dtype=np.float32)
        d0 = np.asarray(event_data["RecoTrack_d0"][i], dtype=np.float32)
        d0_err = np.asarray(event_data["RecoTrack_ErrD0"][i], dtype=np.float32)
        z0_err = np.asarray(event_data["RecoTrack_ErrZ0"][i], dtype=np.float32)
        cov = np.asarray(event_data["RecoTrack_ErrD0Z0"][i], dtype=np.float32)
        beam_z = _extract_beam_z(event_data["BeamPosZ"][i])
        amvf_ntracks = np.asarray(event_data["RecoVertex_nTracks"][i], dtype=int)
        amvf_z = (np.asarray(event_data["RecoVertex_z"][i], dtype=np.float64) - beam_z)[
            amvf_ntracks >= 2
        ]
        if len(amvf_z) == 0:
            continue
        tensors = _build_subevent_tensors(z0, d0, d0_err, z0_err, cov)
        all_preds.append(run_inference_event(model, tensors, device))
        all_amvf.append(amvf_z.astype(np.float32))

    n_used = len(all_preds)
    print(f"      Events with valid AMVF truth: {n_used}")

    print("[3/4] Computing resolution...")
    pvf_dists: list[float] = []
    amvf_dists: list[float] = []
    peak_kwargs = {
        "threshold": threshold,
        "integral_threshold": integral_threshold,
        "min_width": min_width,
    }
    for pred, amvf_z in tqdm(zip(all_preds, all_amvf), total=n_used, desc="Distances"):
        z_mm, _, _, _ = pv_locations_updated_res(pred, **peak_kwargs)
        pvf_dists.extend(_pairwise_distances(z_mm))
        amvf_dists.extend(_pairwise_distances(amvf_z))

    if sigma_vtx_vtx is not None:
        sigma_pvf = sigma_vtx_vtx
        sigma_amvf, _ = _fit_sigma(np.asarray(amvf_dists))
        print(f"      Using provided sigma = {sigma_pvf:.2f} mm")
    else:
        sigma_pvf, pvf_err, sigma_amvf, amvf_err = make_run3_resolution_plot(
            pvf_dists, amvf_dists, out
        )
        print(f"      sigma_PVF  = {sigma_pvf:.2f} +/- {pvf_err:.2f} mm")
        print(f"      sigma_AMVF = {sigma_amvf:.2f} +/- {amvf_err:.2f} mm")

    print("[4/4] Classifying vertices...")
    sigma_bins = sigma_pvf / BIN_WIDTH_MM
    totals = np.zeros(4, dtype=int)
    n_amvf_truth = 0
    for pred, amvf_z in tqdm(
        zip(all_preds, all_amvf), total=n_used, desc="Classifying"
    ):
        pred_z, _, _, _ = pv_locations_updated_res(pred, **peak_kwargs)
        n_amvf_truth += len(amvf_z)
        pred_bins = (pred_z - Z_MIN) / BIN_WIDTH_MM
        truth_bins = (amvf_z - Z_MIN) / BIN_WIDTH_MM
        perf, _, _ = compare_res_reco(
            truth_bins, pred_bins, sigma_bins * np.ones(len(pred_bins))
        )
        totals += [perf.reco_clean, perf.reco_merged, perf.reco_split, perf.reco_fake]

    n_reco = int(totals.sum())
    clean, merged, split, fake = (
        int(totals[0]),
        int(totals[1]),
        int(totals[2]),
        int(totals[3]),
    )
    amvf_eff = (clean + merged) / max(n_amvf_truth, 1)

    summary: dict = {
        "n_events": n_used,
        "n_amvf_truth": n_amvf_truth,
        "avg_amvf_per_event": n_amvf_truth / max(n_used, 1),
        "clean": clean,
        "merged": merged,
        "split": split,
        "fake": fake,
        "n_reco": n_reco,
        "avg_reco_per_event": n_reco / max(n_used, 1),
        "amvf_efficiency": amvf_eff,
        "sigma_pvf_mm": sigma_pvf,
        "sigma_amvf_mm": sigma_amvf,
    }
    with open(out / "pvf_run3_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    _make_category_bar(summary, out)

    s = summary
    nr = max(n_reco, 1)
    print("\n── Results ────────────────────────────────────────────")
    print(
        f"  Events: {s['n_events']}  |  AMVF truth: {s['n_amvf_truth']} ({s['avg_amvf_per_event']:.1f}/ev)  |  Reco: {s['n_reco']} ({s['avg_reco_per_event']:.1f}/ev)"
    )
    print(
        f"  AMVF efficiency: {s['amvf_efficiency']:.4f}  |  sigma_PVF={s['sigma_pvf_mm']:.2f} mm  sigma_AMVF={s['sigma_amvf_mm']:.2f} mm"
    )
    print(
        f"  Clean {clean:>5d} ({100 * clean / nr:.1f}%)  Merged {merged:>5d} ({100 * merged / nr:.1f}%)  Split {split:>5d} ({100 * split / nr:.1f}%)  Fake {fake:>5d} ({100 * fake / nr:.1f}%)"
    )
    print(f"  Saved: {out}/pvf_run3_results.json  pvf_category_bar.png")
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PV-Finder evaluation on Run 3 ROOT data (AMVF truth)"
    )
    p.add_argument("--model", type=str, required=True, help="Model weights (.pyt)")
    p.add_argument("--root-file", type=str, required=True, help="ROOT file path")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory")
    p.add_argument("--n-events", type=int, default=2000)
    p.add_argument("--device", type=int, default=0, help="GPU index, -1 for CPU")
    p.add_argument(
        "--sigma-vtx-vtx", type=float, default=None, help="Pre-computed sigma (mm)"
    )
    p.add_argument("--threshold", type=float, default=0.01)
    p.add_argument("--integral-threshold", type=float, default=0.5)
    p.add_argument("--min-width", type=int, default=3)
    return p


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()
    device = (
        torch.device("cpu") if args.device < 0 else torch.device(f"cuda:{args.device}")
    )
    evaluate_run3(
        model_path=args.model,
        root_file=args.root_file,
        output_dir=args.output_dir,
        n_events=args.n_events,
        device=device,
        sigma_vtx_vtx=args.sigma_vtx_vtx,
        threshold=args.threshold,
        integral_threshold=args.integral_threshold,
        min_width=args.min_width,
    )


if __name__ == "__main__":
    main()
