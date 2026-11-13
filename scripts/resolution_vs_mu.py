"""HL-LHC resolution versus pile-up with capped per-bin errors.

Stages (each restartable; outputs under --out):

  select   read mu from the flat-mu files and choose entries per mu bin, or
           copy an earlier study's selection (--reuse-selection) so that two
           models are compared on identical events
  infer    run the density model on the chosen entries, save histograms
  analyze  peaks at the single operating point, per-bin sigma_vtx-vtx and p68
           for PV-Finder and AMVF, fit plots, summary plot, JSON and table

By default the density model, its architecture and the peak cuts are the frozen
HL-LHC working point, ``working_points.get_regime("hllhc")``.  Explicit flags
(--model/--unet-channels/--latent/--hidden, --peak-threshold/--integral-threshold/
--min-width/--min-height/--centroid-halfwidth) override it; every override is
recorded in provenance.json (inference) and results.json (analysis).

  python -u scripts/resolution_vs_mu.py select --out DIR --reuse-selection OLD/selection.json
  python -u scripts/resolution_vs_mu.py infer --out DIR --device 2
  python -u scripts/resolution_vs_mu.py analyze --out DIR

The density models (v6, s8) never saw r16443 or r16638, so every entry of both
files is usable for this before-GNN study.  (The GNN uses r16443 >= 25,000 and
r16638 for training; that is irrelevant here, and the selection is outcome-blind.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(REPO / "src"),
    str(REPO / "src/pv_finder/evaluation/vertex_finding"),
]

from pv_finder.diagnostics.resolution_vs_mu import (  # noqa: E402
    bin_summary,
    fit_chi2,
    markdown_table,
    mu_bin_index,
    select_entries,
)
from pv_finder.evaluation.vertex_finding.regime_cli import (  # noqa: E402
    git_dirty_eval,
    git_hash,
)
from pv_finder.utils.working_points import get_regime  # noqa: E402

DATA = REPO / "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026"
FILES = {t: DATA / f"ATLAS_PVFinderData_601229_e8481_s4494_{t}_PU200.root"
         for t in ("r16443", "r16638")}  # fmt: skip
REGIME = get_regime("hllhc")
# CLI dest -> attribute of the regime's DensityModel.
MODEL_ARGS = {"model": "checkpoint", "unet_channels": "unet_channels",
              "latent": "latent_channels", "hidden": "hidden"}  # fmt: skip
# CLI dest -> (attribute of PeakFinding, keyword of pv_locations_updated_res).
PEAK_ARGS = {"peak_threshold": ("amplitude", "threshold"),
             "integral_threshold": ("integral", "integral_threshold"),
             "min_width": ("min_width", "min_width"),
             "min_height": ("min_height", "min_height"),
             "centroid_halfwidth": ("centroid_halfwidth", "centroid_halfwidth")}  # fmt: skip


def sha256(path: Path) -> str:
    """Hex digest of a file, for the provenance record."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _regime_value(dest: str) -> Any:
    """The regime's value for a model CLI dest, in the CLI's type."""
    v = getattr(REGIME.density, MODEL_ARGS[dest])
    if dest == "model":
        return REPO / v
    return list(v) if dest == "hidden" else v


def resolve_working_point(a: argparse.Namespace) -> None:
    """Fill unset model/peak arguments from the frozen HL-LHC regime.

    Sets ``a.peaks`` (keywords of ``pv_locations_updated_res``; the sigma fit
    uses the same peaks as counting) and ``a.overrides`` (explicit values that
    differ from the regime, keyed by CLI dest).
    """
    a.overrides = {}
    for dest in MODEL_ARGS:
        ref, val = _regime_value(dest), getattr(a, dest)
        if val is None:
            setattr(a, dest, ref)
            continue
        if dest == "model":
            same = Path(val).resolve() == Path(ref).resolve()
        else:
            same = (list(val) if dest == "hidden" else val) == ref
        if not same:
            a.overrides[dest] = str(val) if dest == "model" else val
    a.peaks = {}
    for dest, (attr, kw) in PEAK_ARGS.items():
        ref, val = getattr(REGIME.peaks, attr), getattr(a, dest)
        if val is None:
            val = ref
        elif val != ref:
            a.overrides[dest] = val
        a.peaks[kw] = val
    if a.overrides:
        print(f"OVERRIDES of regime '{REGIME.name}': {a.overrides}", flush=True)


def cmd_select(a: argparse.Namespace) -> None:
    """Write selection.json: entries per file for the requested per-bin counts."""
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / "selection.json"
    if a.reuse_selection is not None:
        src = a.reuse_selection.resolve()
        if dst.exists() and dst.resolve() == src:
            raise ValueError("--reuse-selection points at the output selection itself")
        shutil.copyfile(src, dst)
        if sha256(src) != sha256(dst):
            raise AssertionError(f"copied selection differs from {src}")
        if (
            json.loads(src.read_text())["entries"]
            != json.loads(dst.read_text())["entries"]
        ):
            raise AssertionError(f"copied entry list differs from {src}")
        (a.out / "selection_source.json").write_text(json.dumps(
            {"source": str(src), "sha256": sha256(src)}, indent=1))  # fmt: skip
        sel = json.loads(dst.read_text())
        print(f"reused {src} (sha256 {sha256(src)[:12]}...); per bin:",
              sel["selected_per_bin"], flush=True)  # fmt: skip
        return
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
    dst.write_text(json.dumps(out))
    print("selected per bin:", counts.tolist(), flush=True)


def _build_model(a: argparse.Namespace, device):  # noqa: ANN001, ANN202
    import run_eval_pvf_run3 as ev

    from pv_finder.models.autoencoder_models import MaskedDNN
    from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2

    if REGIME.density.model_type != "v2" and "model" not in a.overrides:
        raise ValueError(f"model_type {REGIME.density.model_type!r} is not v2")
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
    model_overrides = {k: v for k, v in a.overrides.items() if k in MODEL_ARGS}
    prov = {"model": str(a.model), "model_sha256": sha256(Path(a.model)),
            "unet_channels": a.unet_channels, "latent": a.latent, "hidden": list(a.hidden),
            "regime_overrides": model_overrides,
            "density_from_regime": not model_overrides,
            "selection_sha256": sha256(a.out / "selection.json"),
            "infer_git_hash": git_hash(), "infer_src_dirty": git_dirty_eval(),
            "infer_argv": list(sys.argv),
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


def load_chunk(f: Path, peaks: dict[str, Any]) -> dict:
    """Peaks at the given cuts, truth and AMVF lists for one chunk."""
    from pv_finder.utils.peak_finding import pv_locations_updated_res

    # Read each array once: every d[key] on an NpzFile re-reads the whole array.
    with np.load(f) as npz:
        d = {key: npz[key] for key in npz.files}
    to, ao = d["truth_off"], d["amvf_off"]
    rec: dict[str, list] = {"pvf": [], "amvf": [], "truth": [], "mu": [], "id": []}
    for k in range(len(d["histograms"])):
        p, *_ = pv_locations_updated_res(d["histograms"][k], **peaks)
        rec["pvf"].append(np.asarray(p, dtype=float))
        rec["truth"].append(d["truth_z"][to[k] : to[k + 1]].astype(float))
        rec["amvf"].append(d["amvf_z"][ao[k] : ao[k + 1]].astype(float))
        rec["mu"].append(float(d["mu"][k]))
        rec["id"].append(f"{f.name[:6]}:{int(d['event_idx'][k])}")
    return rec


def load_study(out: Path, peaks: dict[str, Any], workers: int) -> dict:
    """All chunks of a study, in file-name order, peaks found in parallel."""
    from multiprocessing import Pool

    files = sorted((out / "histograms").glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no histogram chunks under {out / 'histograms'}")
    with Pool(workers) as pool:
        parts = pool.map(partial(load_chunk, peaks=peaks), files)
    rec = {k: [x for part in parts for x in part[k]] for k in parts[0]}
    rec["mu"] = np.asarray(rec["mu"])
    return rec


def cmd_analyze(a: argparse.Namespace) -> None:
    """Per-bin numbers, the per-bin fit PDF and the ATLAS-style summary plots."""
    from resolution_vs_mu_plots import plot_fits, plot_summary

    from pv_finder.diagnostics.run2_vs_run4.metrics import fit_counts

    sel = json.loads((a.out / "selection.json").read_text())
    prov = json.loads((a.out / "provenance.json").read_text())
    if prov.get("selection_sha256") not in (None, sha256(a.out / "selection.json")):
        raise AssertionError("selection.json changed after inference")
    edges = np.asarray(sel["mu_edges"])
    rec = load_study(a.out, a.peaks, a.workers)
    if len(set(rec["id"])) != len(rec["id"]):
        raise AssertionError("duplicate events in the histogram chunks")
    n_sel = sum(len(e) for e in sel["entries"].values())
    print(f"{len(rec['id'])} events in the histograms, {n_sel} selected", flush=True)
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
    peak_overrides = {k: v for k, v in a.overrides.items() if k in PEAK_ARGS}
    infer_overrides = prov.get("regime_overrides", {})
    res = {"peaks": a.peaks, "regime": REGIME.to_dict(),
           "regime_overrides": {"infer": infer_overrides, "analyze": peak_overrides},
           "at_frozen_working_point": not (peak_overrides or infer_overrides),
           "model": prov["model"], "model_sha256": prov["model_sha256"],
           "git_hash": git_hash(), "src_dirty": git_dirty_eval(),
           "analyze_argv": list(sys.argv), "provenance": prov,
           "n_events": len(rec["mu"]), "n_selected": n_sel, "bins": rows}  # fmt: skip
    (a.out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (a.out / "table.md").write_text(markdown_table(rows) + "\n")
    plot_fits(hists, k, rows, a.out / "per_bin_fits.pdf")
    plot_summary(rows, a.out, label=Path(prov["model"]).stem)
    print(markdown_table(rows), flush=True)


def cmd_plot(a: argparse.Namespace) -> None:
    """Redraw the summary figures from results.json (no recomputation)."""
    from resolution_vs_mu_plots import plot_summary

    res = json.loads((a.out / "results.json").read_text())
    plot_summary(res["bins"], a.out, label=Path(res["model"]).stem)


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
        "--reuse-selection",
        type=Path,
        default=None,
        help="copy this selection.json instead of selecting (identical events)",
    )
    wp = "(default: from get_regime('hllhc'))"
    p.add_argument("--model", type=Path, default=None, help=f"checkpoint {wp}")
    p.add_argument("--unet-channels", type=int, default=None, help=wp)
    p.add_argument("--latent", type=int, default=None, help=wp)
    p.add_argument("--hidden", type=int, nargs="+", default=None, help=wp)
    p.add_argument("--peak-threshold", type=float, default=None, help=wp)
    p.add_argument("--integral-threshold", type=float, default=None, help=wp)
    p.add_argument("--min-width", type=int, default=None, help=wp)
    p.add_argument("--min-height", type=float, default=None, help=wp)
    p.add_argument("--centroid-halfwidth", type=int, default=None, help=wp)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--workers", type=int, default=8, help="peak-finding processes")
    a = p.parse_args()
    resolve_working_point(a)
    stages = ["select", "infer", "analyze"] if a.stage == "all" else [a.stage]
    for s in stages:
        {"select": cmd_select, "infer": cmd_infer, "analyze": cmd_analyze,
         "plot": cmd_plot}[s](a)  # fmt: skip


if __name__ == "__main__":
    main()
