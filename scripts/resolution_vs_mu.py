"""HL-LHC resolution versus pile-up with capped per-bin errors.

Stages (each restartable; outputs under --out):

  select   read mu from the flat-mu files and choose entries per mu bin
  infer    run the density model on the chosen entries, save histograms
  analyze  peaks at the single operating point, per-bin sigma_vtx-vtx and p68
           for PV-Finder and AMVF, fit plots, summary plot and JSON

Rerunning with another (e.g. the frozen) density model is one command:

  python -u scripts/resolution_vs_mu.py all --out DIR --model CKPT \
      --unet-channels 280 --latent 4 --device 3

The density model never saw r16443 or r16638, so every entry of both files is
usable for this before-GNN study.  (The GNN uses r16443 >= 25,000 and r16638 for
training; that is irrelevant here, and the selection is still outcome-blind.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(REPO / "src"),
    str(REPO / "src/pv_finder/evaluation/vertex_finding"),
]

from pv_finder.diagnostics.resolution_vs_mu import (  # noqa: E402
    bin_summary,
    fit_chi2,
    mu_bin_index,
    select_entries,
)
from pv_finder.evaluation.vertex_finding.regime_cli import (  # noqa: E402
    git_dirty_eval,
    git_hash,
)
from pv_finder.utils.working_points import HLLHC_PEAKS  # noqa: E402

DATA = REPO / "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026"
FILES = {t: DATA / f"ATLAS_PVFinderData_601229_e8481_s4494_{t}_PU200.root"
         for t in ("r16443", "r16638")}  # fmt: skip
# Single HL-LHC operating point from the shared definition (working_points); the
# sigma fit uses the same peaks as counting (no separate 0.50).
_P = HLLHC_PEAKS
PEAKS = dict(threshold=_P.amplitude, integral_threshold=_P.integral,
             min_width=_P.min_width, min_height=_P.min_height,
             centroid_halfwidth=_P.centroid_halfwidth)  # fmt: skip


def sha256(path: Path) -> str:
    """Hex digest of a file, for the provenance record."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def cmd_select(a: argparse.Namespace) -> None:
    """Write selection.json: entries per file for the requested per-bin counts."""
    import uproot

    edges = np.asarray(a.mu_edges, dtype=float)
    target = np.asarray(a.target, dtype=int)
    if len(target) == 1:
        target = np.full(len(edges) - 1, target[0])
    mu = {t: uproot.open(p)["PVFinderData"]["ActualNumOfInt"].array(library="np")
          for t, p in FILES.items()}  # fmt: skip
    chosen = select_entries(mu, edges, target)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for t, e in chosen.items():
        k = mu_bin_index(mu[t][e], edges)
        counts += np.bincount(k, minlength=len(counts))
    out = {"mu_edges": edges.tolist(), "target": target.tolist(),
           "selected_per_bin": counts.tolist(),
           "entries": {t: e.tolist() for t, e in chosen.items()}}  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "selection.json").write_text(json.dumps(out))
    print("selected per bin:", counts.tolist(), flush=True)


def _build_model(a: argparse.Namespace, device):  # noqa: ANN001, ANN202
    import run_eval_pvf_run3 as ev

    from pv_finder.models.autoencoder_models import MaskedDNN
    from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2

    cfg = dict(ev.T2KDE_CONFIG, hidden_nodes=list(a.hidden),
               output_size=1000 * a.latent)  # fmt: skip
    model = TracksToHist_v2(MaskedDNN(**cfg),
                            UNet_1000_v2(n=a.unet_channels, n_features=a.latent,
                                         dropout_p=0.0))  # fmt: skip
    ev.load_ckpt(str(a.model), model, device)
    return model, ev


def cmd_infer(a: argparse.Namespace) -> None:
    """Histograms for the selected entries, one npz per (file, 5000-entry window)."""
    import torch

    from pv_finder.data.run3_io import load_run3_from_root

    sel = json.loads((a.out / "selection.json").read_text())
    device = torch.device(f"cuda:{a.device}" if a.device >= 0 else "cpu")
    model, ev = _build_model(a, device)
    hdir = a.out / "histograms"
    hdir.mkdir(exist_ok=True)
    prov = {"model": str(a.model), "model_sha256": sha256(a.model),
            "unet_channels": a.unet_channels, "latent": a.latent, "hidden": a.hidden,
            "infer_git_hash": git_hash(), "infer_src_dirty": git_dirty_eval(),
            "files": {t: {"path": str(p), "size": p.stat().st_size}
                      for t, p in FILES.items()}}  # fmt: skip
    (a.out / "provenance.json").write_text(json.dumps(prov, indent=1))
    for t, entries in sel["entries"].items():
        want = np.asarray(entries, dtype=np.int64)
        for start in range(0, int(want.max()) + 1 if len(want) else 0, 5000):
            stop = start + 5000
            path = hdir / f"{t}_{start:06d}_{stop:06d}.npz"
            w = want[(want >= start) & (want < stop)]
            if path.exists() or len(w) == 0:
                continue
            wset = set(w.tolist())
            events = [e for e in load_run3_from_root(str(FILES[t]), entry_start=start,
                                                     entry_stop=stop)
                      if e.event_idx in wset]  # fmt: skip
            H = np.stack([ev.run_inference(ev.build_subevent_inputs(e), device, e2e=model)
                          for e in events])  # fmt: skip
            tz = [e.truth_z for e in events]
            az = [e.amvf_z for e in events]
            np.savez(path, histograms=H,
                     event_idx=np.array([e.event_idx for e in events]),
                     mu=np.array([e.mu for e in events]),
                     truth_z=np.concatenate(tz), truth_off=np.cumsum([0] + [len(z) for z in tz]),
                     amvf_z=np.concatenate(az), amvf_off=np.cumsum([0] + [len(z) for z in az]))  # fmt: skip
            print(f"{t} [{start},{stop}): {len(events)}/{len(w)} events", flush=True)


def _load_chunk(f: Path) -> dict:
    """Peaks at the operating point, truth and AMVF lists for one chunk."""
    from pv_finder.utils.peak_finding import pv_locations_updated_res

    d = np.load(f)
    to, ao = d["truth_off"], d["amvf_off"]
    rec: dict[str, list] = {"pvf": [], "amvf": [], "truth": [], "mu": [], "id": []}
    for k in range(len(d["histograms"])):
        p, *_ = pv_locations_updated_res(d["histograms"][k], **PEAKS)
        rec["pvf"].append(np.asarray(p, dtype=float))
        rec["truth"].append(d["truth_z"][to[k] : to[k + 1]].astype(float))
        rec["amvf"].append(d["amvf_z"][ao[k] : ao[k + 1]].astype(float))
        rec["mu"].append(float(d["mu"][k]))
        rec["id"].append(f"{f.name[:6]}:{int(d['event_idx'][k])}")
    return rec


def _load(out: Path, workers: int) -> dict:
    """All chunks, in file-name order, peaks found in parallel."""
    from multiprocessing import Pool

    files = sorted((out / "histograms").glob("*.npz"))
    with Pool(workers) as pool:
        parts = pool.map(_load_chunk, files)
    rec = {k: [x for part in parts for x in part[k]] for k in parts[0]}
    rec["mu"] = np.asarray(rec["mu"])
    return rec


def cmd_analyze(a: argparse.Namespace) -> None:
    """Per-bin numbers, the per-bin fit PDF and the ATLAS-style summary plot."""
    from resolution_vs_mu_plots import plot_fits, plot_summary

    from pv_finder.diagnostics.run2_vs_run4.metrics import fit_counts

    sel = json.loads((a.out / "selection.json").read_text())
    edges = np.asarray(sel["mu_edges"])
    rec = _load(a.out, a.workers)
    if len(set(rec["id"])) != len(rec["id"]):
        raise AssertionError("duplicate events in the histogram chunks")
    rows, hists = bin_summary(rec["pvf"], rec["amvf"], rec["truth"], rec["mu"], edges,
                              n_boot=a.n_boot)  # fmt: skip
    k = mu_bin_index(rec["mu"], edges)
    for b, row in enumerate(rows):
        for name in ("pvf", "amvf"):
            c = hists[name][k == b].sum(0)
            s, e, popt = fit_counts(hists["edges"], c)
            chi2, ndf = fit_chi2(hists["edges"], c, popt)
            row[name].update(chi2=chi2, ndf=ndf, popt=popt.tolist(),
                             rel_err=row[name]["boot_err_mm"] / s if s else np.nan)  # fmt: skip
    prov = json.loads((a.out / "provenance.json").read_text())
    git = {"analyze_git_hash": git_hash(), "analyze_src_dirty": git_dirty_eval()}
    res = {"peaks": PEAKS, "provenance": {**prov, **git}, "n_events": len(rec["mu"]),
           "bins": rows}  # fmt: skip
    (a.out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    plot_fits(hists, k, rows, a.out / "per_bin_fits.pdf")
    plot_summary(rows, a.out, label=Path(prov["model"]).stem)
    for r in rows:
        print(f"mu {r['mu_lo']:.0f}-{r['mu_hi']:.0f} n={r['n_events']} "
              + " ".join(f"{n}: s={r[n]['sigma_mm']:.4f}+-{r[n]['boot_err_mm']:.4f} "
                         f"({100 * r[n]['rel_err']:.1f}%) chi2/ndf={r[n]['chi2']:.0f}/{r[n]['ndf']} "
                         f"ml={r[n]['merge_len_mm']:.3f} p68={r[n]['p68_um']:.1f}+-{r[n]['p68_err_um']:.1f}"
                         for n in ("pvf", "amvf")), flush=True)  # fmt: skip


def cmd_plot(a: argparse.Namespace) -> None:
    """Redraw the summary figure from results.json (no recomputation)."""
    from resolution_vs_mu_plots import plot_summary

    res = json.loads((a.out / "results.json").read_text())
    plot_summary(res["bins"], a.out, label=Path(res["provenance"]["model"]).stem)


def main() -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stage", choices=["select", "infer", "analyze", "plot", "all"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--mu-edges",
        type=float,
        nargs="+",
        default=[0, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 180, 200],
    )
    p.add_argument(
        "--target",
        type=int,
        nargs="+",
        default=[2000],
        help="events per mu bin (one value, or one per bin)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=REPO / "model_weights/"
        "hllhc_alleta_v6_mse_2ep_phase2_epoch_2_fullstate.pth",
    )
    p.add_argument("--unet-channels", type=int, default=280)
    p.add_argument("--latent", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="+", default=[128] * 5)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=8, help="peak-finding processes")
    a = p.parse_args()
    stages = ["select", "infer", "analyze"] if a.stage == "all" else [a.stage]
    for s in stages:
        {"select": cmd_select, "infer": cmd_infer, "analyze": cmd_analyze,
         "plot": cmd_plot}[s](a)  # fmt: skip


if __name__ == "__main__":
    main()
