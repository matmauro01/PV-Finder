"""AMVF z-resolution vs N_Tracks via AMVF--truth residuals (HL-LHC PU200).

Match each AMVF reco vertex to its closest truth vertex (greedy, within
``--match-window`` mm). Per truth N_Tracks bin, fit a Gaussian to the
Delta z = z_AMVF - z_truth distribution to extract sigma(n). Fit the
power law sigma(n) = a * n^(-b) + c.

Outputs:
  amvf_resolution_vs_ntracks.png  -- ROOT/atlasplots plot
  fit_params.json                  -- a, b, c and per-bin (sigma, sigma_err)
  vertex_data.npz                  -- raw (truth_ntrks, dz_mm) pairs

Style matches the ATLAS efficiency_example.png reference plot: blue filled
circles for data (with error bars), red solid power-law fit, ATLAS label,
"Data" / "Fit" legend, axes in mm.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from array import array
from pathlib import Path

import numpy as np

# Anaconda's site-packages hosts ROOT 6.24 + atlasplots; venv doesn't see it
# by default, so prepend it before importing ROOT.
_ANACONDA_SP = "/usr/local/anaconda3/lib/python3.8/site-packages"
if _ANACONDA_SP not in sys.path:
    sys.path.insert(0, _ANACONDA_SP)

import atlasplots  # noqa: E402
import ROOT  # noqa: E402
from ROOT import TF1, TH1F, TCanvas, TGraphErrors, TLegend  # noqa: E402

from pv_finder.data.run3_io import load_run3_from_root  # noqa: E402

ROOT.gROOT.SetBatch(True)

# --- Marker / colour scheme (matches ATLAS example: blue data, red fit) --
COLOR_DATA = 4  # ROOT kBlue
COLOR_FIT = 2  # ROOT kRed
MARKER_DATA = 20  # filled circle (matches MARKER_UNETPP in sample_plotting_code.py)
MARKER_SIZE = 0.7

DEFAULT_ROOT = (
    "data/run4/Run4_MC21_ITk/"
    "ATLAS_PVFinderData_HLLHC_mc21_14TeV_ttbar_SingleLep_PU200.root"
)
DEFAULT_OUT = "outputs/06_01_2026_output/amvf_resolution_residuals"

# Per-integer bins for n<=30 (where most vertices live), then progressively
# wider edges up to N_Tracks ~ 170 (full dataset max is 168). Bins with
# <30 vertices are dropped at fit time.
DEFAULT_BINS = (
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    32,
    34,
    36,
    38,
    40,
    42,
    45,
    48,
    51,
    55,
    60,
    65,
    70,
    78,
    88,
    100,
    115,
    135,
    170,
)


# --------------------------------------------------------------------------
# Matching: greedy closest-first, 1-to-1
# --------------------------------------------------------------------------


def greedy_match(
    amvf_z: np.ndarray, truth_z: np.ndarray, window_mm: float
) -> list[tuple[int, int, float]]:
    """Greedy closest-first 1-to-1 matching within +/- window_mm.

    Returns ``[(amvf_idx, truth_idx, |dz| mm), ...]`` -- one pair per AMVF
    vertex that could be matched. Order is closest-first.
    """
    if len(amvf_z) == 0 or len(truth_z) == 0:
        return []

    pairs: list[tuple[int, int, float]] = []
    for ai, az in enumerate(amvf_z):
        dists = np.abs(truth_z - float(az))
        for tj in np.where(dists <= window_mm)[0]:
            pairs.append((ai, int(tj), float(dists[tj])))

    pairs.sort(key=lambda p: p[2])

    used_a: set[int] = set()
    used_t: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for ai, tj, d in pairs:
        if ai in used_a or tj in used_t:
            continue
        used_a.add(ai)
        used_t.add(tj)
        matched.append((ai, tj, d))
    return matched


def collect_residuals(
    root_path: str,
    *,
    max_events: int = 0,
    match_window_mm: float = 2.0,
    min_truth_ntrks: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk ROOT, return (truth_ntrks, dz_mm) for every matched AMVF vertex."""
    t0 = time.time()
    events = load_run3_from_root(
        root_path,
        max_events=max_events,
        min_tracks=1,
        min_amvf_vtx=1,
    )
    print(f"[amvf_res] loaded {len(events)} events in {time.time() - t0:.1f}s")

    if not events or events[0].truth_z is None:
        raise RuntimeError(
            "ROOT file has no TruthVertex_z branch -- this script needs MC truth"
        )

    truth_ntrks_list: list[int] = []
    dz_list: list[float] = []
    n_matched = 0
    n_amvf_total = 0
    for ev in events:
        a_z = np.asarray(ev.amvf_z, dtype=np.float64)
        t_z = np.asarray(ev.truth_z, dtype=np.float64)
        t_n = np.asarray(ev.truth_ntrks, dtype=np.int32)
        n_amvf_total += len(a_z)

        matched = greedy_match(a_z, t_z, match_window_mm)
        for ai, tj, _ in matched:
            n_t = int(t_n[tj])
            if n_t < min_truth_ntrks:
                continue
            truth_ntrks_list.append(n_t)
            dz_list.append(float(a_z[ai] - t_z[tj]))

        n_matched += len(matched)

    ntrk_arr = np.asarray(truth_ntrks_list, dtype=np.int32)
    dz_arr = np.asarray(dz_list, dtype=np.float64)
    eff = n_matched / max(n_amvf_total, 1)
    print(
        f"[amvf_res] matched {n_matched}/{n_amvf_total} AMVF vertices "
        f"({eff:.1%})  ->  {ntrk_arr.size} pairs after nTracks>={min_truth_ntrks}"
    )
    return ntrk_arr, dz_arr


# --------------------------------------------------------------------------
# Per-bin Gaussian fit + power-law fit
# --------------------------------------------------------------------------


def fit_gaussian_dz(dz_mm: np.ndarray, name: str) -> tuple[float, float] | None:
    """Fit Gaussian to a dz distribution (mm); return (sigma_um, sigma_err_um)."""
    if dz_mm.size < 30:
        return None

    q05, q95 = np.percentile(dz_mm, [5, 95])
    half = max(abs(q05), abs(q95)) * 1.6
    if half <= 0:
        return None
    h = TH1F(name, "", 80, -half, half)
    for v in dz_mm:
        h.Fill(float(v))

    rms = h.GetRMS()
    mean = h.GetMean()
    if rms <= 0:
        return None
    f = TF1(name + "_g", "gaus", mean - 2.5 * rms, mean + 2.5 * rms)
    fit_status = h.Fit(f, "RQ0SN")
    if int(fit_status) != 0:
        return None
    sigma_mm = abs(f.GetParameter(2))
    sigma_err_mm = abs(f.GetParError(2))
    return float(sigma_mm * 1000.0), float(sigma_err_mm * 1000.0)


def aggregate_and_fit_bins(
    ntrk: np.ndarray,
    dz_mm: np.ndarray,
    bin_edges: tuple[int, ...],
) -> dict:
    """Per bin: bin centre (mean N_Tracks), Gaussian sigma in um."""
    edges = np.asarray(bin_edges, dtype=np.int64)
    centers: list[float] = []
    sigmas: list[float] = []
    sig_errs: list[float] = []
    counts: list[int] = []

    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (ntrk >= lo) & (ntrk < hi)
        n = int(mask.sum())
        if n < 30:
            continue
        res = fit_gaussian_dz(dz_mm[mask], name=f"dz_bin{k}_{lo}_{hi}")
        if res is None:
            continue
        sig_um, sig_err_um = res
        if not (np.isfinite(sig_um) and np.isfinite(sig_err_um)) or sig_um <= 0:
            continue
        centers.append(float(ntrk[mask].mean()))
        sigmas.append(sig_um)
        sig_errs.append(max(sig_err_um, 0.1 * sig_um))  # floor so weighted fit stable
        counts.append(n)

    return dict(
        centers=np.asarray(centers),
        sigmas_um=np.asarray(sigmas),
        sigma_errs_um=np.asarray(sig_errs),
        counts=np.asarray(counts),
    )


def fit_power_law(centers: np.ndarray, sigmas_um: np.ndarray) -> TF1:
    """Build the TF1 = a / x^b + c with sensible starting params + limits."""
    x_min = max(1.5, float(centers.min()) * 0.8)
    x_max = float(centers.max()) * 1.2
    f = TF1("powlaw", "[0]/pow(x,[1])+[2]", x_min, x_max)
    f.SetParameters(float(sigmas_um[0]), 0.5, float(sigmas_um[-1]))
    f.SetParLimits(0, 0.0, 1e4)
    f.SetParLimits(1, 0.0, 5.0)
    f.SetParLimits(2, 0.0, 1e3)
    f.SetParName(0, "a")
    f.SetParName(1, "b")
    f.SetParName(2, "c")
    return f


# --------------------------------------------------------------------------
# Plotting: PyROOT + atlasplots, mimicking Qi Bin's style
# --------------------------------------------------------------------------


def make_plot(
    agg: dict,
    out_path: Path,
    plot_label: str = "Simulation Preliminary",
) -> tuple[float, float, float, float, float, float]:
    """Draw sigma_z(N_Tracks) graph + power-law fit. Returns (a, b, c, da, db, dc)."""
    atlasplots.set_atlas_style()

    centers = agg["centers"]
    sigmas = agg["sigmas_um"]
    sig_errs = agg["sigma_errs_um"]
    n = len(centers)

    c1 = TCanvas("c1", "amvf_resolution", 200, 10, 900, 800)
    c1.SetGrid()
    c1.SetTicks(1, 1)

    # Convert um -> mm for plotting (matches ATL-PHYS-PUB-2023-011 Fig. 12 units)
    sigmas_mm = sigmas / 1000.0
    sig_errs_mm = sig_errs / 1000.0

    x = array("f", centers.tolist())
    y = array("f", sigmas_mm.tolist())
    ex = array("f", [0.0] * n)
    ey = array("f", sig_errs_mm.tolist())

    g = TGraphErrors(n, x, y, ex, ey)
    g.SetMarkerColor(COLOR_DATA)
    g.SetLineColor(COLOR_DATA)
    g.SetMarkerStyle(MARKER_DATA)
    g.SetMarkerSize(MARKER_SIZE)

    # x-axis: extend up to the full N_Tracks range observed in data (rounded
    # up to a round number) so high-multiplicity vertices are always visible.
    raw_x_max = float(max(centers.max(), DEFAULT_BINS[-1]))
    x_max = float(int(np.ceil(raw_x_max / 20.0) * 20))
    y_max = float(max(sigmas_mm) * 1.25)
    g.GetXaxis().SetTitle("Number of Tracks")
    g.GetYaxis().SetTitle("Vertex Resolution (mm)")
    g.GetXaxis().SetLimits(0.0, x_max)
    g.GetYaxis().SetRangeUser(0.0, y_max)

    # Fit on mm-scaled values so reported (a, c) in mm; b is unitless.
    f = fit_power_law(centers, sigmas_mm)
    f.SetLineColor(COLOR_FIT)
    f.SetLineStyle(1)
    f.SetLineWidth(2)
    g.Fit(f, "RQ")

    a_mm = float(f.GetParameter(0))
    b = float(f.GetParameter(1))
    c_mm = float(f.GetParameter(2))
    da_mm = float(f.GetParError(0))
    db = float(f.GetParError(1))
    dc_mm = float(f.GetParError(2))

    # convert to um for return signature (keeps JSON consistent with old format)
    a = a_mm * 1000.0
    c = c_mm * 1000.0
    da = da_mm * 1000.0
    dc = dc_mm * 1000.0

    g.Draw("AP")
    f.Draw("L same")

    # Keep TLegend refs alive so PyROOT GC doesn't drop them
    # before SaveAs flushes the canvas.
    keep_alive: list = []

    atlasplots.atlas_label(text=plot_label, x=0.20, y=0.85, size=28)

    legend = TLegend(0.65, 0.72, 0.90, 0.85)
    legend.AddEntry(g, "Data", "PE")
    legend.AddEntry(f, "Fit", "L")
    legend.SetTextFont(42)
    legend.SetTextSize(0.030)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.Draw()
    keep_alive.append(legend)

    c1.Modified()
    c1.Update()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    c1.SaveAs(str(out_path))
    c1.Close()
    print(f"[amvf_res] wrote {out_path}")

    return a, b, c, da, db, dc


# --------------------------------------------------------------------------
# Main / CLI
# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", default=DEFAULT_ROOT, help="ROOT file path")
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="Output directory")
    p.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Cap on events read from the tree (0 = all).",
    )
    p.add_argument(
        "--match-window",
        type=float,
        default=2.0,
        help="AMVF<->truth matching window in mm (default 2.0).",
    )
    p.add_argument(
        "--min-truth-ntrks",
        type=int,
        default=2,
        help="Drop truth vertices below this N_Tracks before binning.",
    )
    p.add_argument(
        "--plot-label",
        default="Simulation Preliminary",
        help="ATLAS plot-label text (e.g. 'Simulation Internal').",
    )
    p.add_argument(
        "--replot-from-npz",
        default=None,
        help="Skip ROOT IO and re-render from a saved vertex_data.npz.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replot_from_npz:
        npz = np.load(args.replot_from_npz, allow_pickle=True)
        ntrk = np.asarray(npz["truth_ntrks"], dtype=np.int32)
        dz = np.asarray(npz["dz_mm"], dtype=np.float64)
        print(
            f"[amvf_res] replot from {args.replot_from_npz}: {ntrk.size} matched pairs"
        )
    else:
        ntrk, dz = collect_residuals(
            args.root,
            max_events=args.max_events,
            match_window_mm=args.match_window,
            min_truth_ntrks=args.min_truth_ntrks,
        )

    if ntrk.size == 0:
        raise SystemExit("No matched AMVF/truth pairs -- nothing to fit.")

    agg = aggregate_and_fit_bins(ntrk, dz, DEFAULT_BINS)
    if agg["centers"].size < 3:
        raise SystemExit(
            f"Only {agg['centers'].size} usable N_Tracks bins -- fit unreliable."
        )

    a, b, c, da, db, dc = make_plot(
        agg,
        out_dir / "amvf_resolution_vs_ntracks.png",
        plot_label=args.plot_label,
    )
    print(
        f"[amvf_res] fit: a={a:.3f}+/-{da:.3f} um  "
        f"b={b:.4f}+/-{db:.4f}  c={c:.3f}+/-{dc:.3f} um"
    )

    fit_json = out_dir / "fit_params.json"
    with open(fit_json, "w") as fp:
        json.dump(
            dict(
                root_path=str(args.root),
                method="AMVF-truth residuals, greedy match",
                match_window_mm=args.match_window,
                min_truth_ntrks=args.min_truth_ntrks,
                n_matched_pairs=int(ntrk.size),
                fit=dict(
                    a_um=a,
                    b=b,
                    c_um=c,
                    a_err_um=da,
                    b_err=db,
                    c_err_um=dc,
                    formula="sigma_z(n) = a / n^b + c, units um",
                ),
                bin_edges=list(DEFAULT_BINS),
                per_bin=dict(
                    centers=agg["centers"].tolist(),
                    sigmas_um=agg["sigmas_um"].tolist(),
                    sigma_errs_um=agg["sigma_errs_um"].tolist(),
                    counts=agg["counts"].astype(int).tolist(),
                ),
            ),
            fp,
            indent=2,
        )
    print(f"[amvf_res] wrote {fit_json}")

    npz_path = out_dir / "vertex_data.npz"
    np.savez_compressed(npz_path, truth_ntrks=ntrk, dz_mm=dz)
    print(f"[amvf_res] wrote {npz_path}")


if __name__ == "__main__":
    main()
