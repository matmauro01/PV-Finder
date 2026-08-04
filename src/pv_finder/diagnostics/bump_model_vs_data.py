#!/usr/bin/env python3
"""Separate the model and the data contribution to the pairwise-Delta-z bump.

The satellite excess in the pairwise-Delta-z distribution is larger for the
current v6 model on the July-2026 extended-|eta| re-production than it was for
v4b on the old |eta| < 2.5 sample. Two things changed at the same time:

* the **evaluation input** -- the re-production widened the track selection
  from |eta| < 2.5 to |eta| < 4 and applies uniform ITk-paper cuts, giving
  ~47 % more tracks per event;
* the **model** -- v6 was trained on that re-production with the
  ``hllhc_alleta`` target-width preset, v4b on the old pool with the ``hllhc``
  preset, v5 on the old pool with ``hllhc_corrected``.

Architecture and loss are *identical* across v4b / v5 / v6 (``TracksToHist_v2``,
280 UNet channels, 4 latent channels, [128]x5 MLP, plain MSE), so the "model"
axis is entirely *what the model was trained on*, never how it is built.

The two ROOT productions hold the SAME events -- ``ActualNumOfInt`` and
``RecoVertex_z`` are identical entry by entry, only the track collection and the
truth bookkeeping differ -- so a checkpoint x production grid is a genuine 2x2
on one event sample.

Two observables per cell:

``band_excess_pct``
    mean relative excess over the plateau in |dz| = 0.3-0.7 mm. This is the
    headline number, but it is *not* multiplicity invariant: satellite pairs
    grow like n while total pairs grow like n^2, so a cell that simply finds
    more peaks reports a smaller excess for the same per-peak pathology.

``satellites_per_peak``
    plateau-subtracted surplus companions in the 0.25-2.0 mm shell divided by
    the number of reconstructed peaks. Multiplicity invariant and truth free,
    so it is comparable across productions whose truth definitions differ.

Uncertainties are bootstrap over events (pairs inside an event are correlated).

Run from the repo root with the venv active, inside tmux::

    PYTHONPATH=src python -u src/pv_finder/diagnostics/bump_model_vs_data.py \\
        --cell v4b_old=<v4b.pth>@old --cell v6_new=<v6.pth>@new \\
        --root-old data/run4/PU200_withTiming/..._r16443_PU200.root \\
        --root-new data/run4_all_etas/.../..._r16443_PU200.root \\
        --n-events 1920 --device 2 \\
        --output-dir outputs/08_04_2026_output/bump_model_vs_data
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
import uproot

sys.path.insert(0, str(Path(__file__).parents[2]))
from pv_finder.data.feature_loading import (  # noqa: E402
    MASK_VAL,
    N_SUBEVENTS,
    build_run3_subevent_tensor,
)
from pv_finder.diagnostics.pairwise_dz_metrics import (  # noqa: E402
    bootstrap,
    measure,
    observables,
    paired_contrast,
    plot_cells,
    print_table,
)
from pv_finder.models.autoencoder_models import MaskedDNN  # noqa: E402
from pv_finder.models.unet_v2 import TracksToHist_v2, UNet_1000_v2  # noqa: E402
from pv_finder.utils.peak_finding import pv_locations_updated_res  # noqa: E402

TREE = "PVFinderData"
MODEL_PAD_VAL = -240.0

TRACK_BRANCHES = [
    "RecoTrack_z0",
    "RecoTrack_d0",
    "RecoTrack_ErrD0",
    "RecoTrack_ErrZ0",
    "RecoTrack_ErrD0Z0",
]
VERTEX_BRANCHES = [
    "RecoVertex_z",
    "RecoVertex_nTracks",
    "TruthVertex_z",
    "TruthVertex_nTracks",
]


# ---------------------------------------------------------------------------
# Event selection and loading
# ---------------------------------------------------------------------------


def select_entries(
    root_path: str, mu_min: float, mu_max: float, n_events: int, entry_stop: int
) -> np.ndarray:
    """Entry numbers of the first ``n_events`` with mu inside the window."""
    mu = uproot.open(f"{root_path}:{TREE}")["ActualNumOfInt"].array(
        entry_stop=entry_stop, library="np"
    )
    mu = np.asarray(mu, dtype=float)
    keep = np.where((mu >= mu_min) & (mu <= mu_max))[0][:n_events]
    if len(keep) < n_events:
        print(f"  WARNING: only {len(keep)}/{n_events} entries matched the mu window")
    return keep


def load_entries(root_path: str, entries: np.ndarray) -> dict:
    """Track arrays plus AMVF and truth vertex z (nTrk >= 2) for given entries."""
    want = set(int(e) for e in entries)
    stop = int(entries.max()) + 1
    out: dict[int, dict] = {}
    tree = uproot.open(f"{root_path}:{TREE}")
    start = 0
    for chunk in tree.iterate(
        TRACK_BRANCHES + VERTEX_BRANCHES, library="np", step_size=1000, entry_stop=stop
    ):
        n = len(chunk["RecoTrack_z0"])
        for j in range(n):
            e = start + j
            if e not in want:
                continue
            rn = np.asarray(chunk["RecoVertex_nTracks"][j])
            tn = np.asarray(chunk["TruthVertex_nTracks"][j])
            out[e] = {
                "z0": np.asarray(chunk["RecoTrack_z0"][j], dtype=np.float32),
                "d0": np.asarray(chunk["RecoTrack_d0"][j], dtype=np.float32),
                "d0_err": np.asarray(chunk["RecoTrack_ErrD0"][j], dtype=np.float32),
                "z0_err": np.asarray(chunk["RecoTrack_ErrZ0"][j], dtype=np.float32),
                "cov": np.asarray(chunk["RecoTrack_ErrD0Z0"][j], dtype=np.float32),
                "amvf_z": np.asarray(chunk["RecoVertex_z"][j], dtype=np.float64)[
                    rn >= 2
                ],
                "truth_z": np.asarray(chunk["TruthVertex_z"][j], dtype=np.float64)[
                    tn >= 2
                ],
                "truth_ntrk": tn.astype(np.float64),
            }
        start += n
        if start >= stop:
            break
    return {"events": [out[int(e)] for e in entries]}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Load a TracksToHist_v2 checkpoint (280 channels, 4 latent, [128]x5)."""
    t2kde = MaskedDNN(
        input_size=7,
        hidden_nodes=[128] * 5,
        output_size=1000 * 4,
        leaky_param=0.01,
        use_bn=False,
        use_drop=False,
        maskVal=MODEL_PAD_VAL,
        predScaleFactor=0.001,
        allow_negative_output=False,
    )
    model = TracksToHist_v2(t2kde, UNet_1000_v2(n=280, n_features=4, dropout_p=0.0))
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"  ckpt {Path(ckpt_path).name}: epoch={ckpt.get('epoch')}")
    return model.to(device).eval()


def predict_hist(
    model: torch.nn.Module, event: dict, device: torch.device
) -> np.ndarray:
    """Run the 12 sub-events through the model and stitch to 12000 bins."""
    tensors = []
    for si in range(N_SUBEVENTS):
        t, _ = build_run3_subevent_tensor(
            event["z0"], event["d0"], event["d0_err"], event["z0_err"], event["cov"], si
        )
        t = t.astype(np.float32).copy()
        t[:, t[1, :] <= (MASK_VAL + 1)] = MODEL_PAD_VAL
        tensors.append(t)
    mx = max(t.shape[1] for t in tensors)
    padded = np.full((N_SUBEVENTS, 7, mx), MODEL_PAD_VAL, dtype=np.float32)
    for i, t in enumerate(tensors):
        padded[i, :, : t.shape[1]] = t
    with torch.no_grad():
        hist = model(torch.from_numpy(padded).to(device))
    return hist.cpu().numpy().reshape(-1).astype(np.float32)


_SUPPORTS_CENTROID = (
    "centroid_halfwidth" in inspect.signature(pv_locations_updated_res).parameters
)


def find_peaks(
    hist: np.ndarray,
    threshold: float,
    integral_threshold: float,
    min_width: int,
    min_height: float,
    centroid_halfwidth: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Peak finding with the position estimator pinned explicitly.

    The estimator is a moving target: the production finder averaged over the
    whole above-threshold region until 2026-08-04, when a local centroid was
    introduced. That choice changes the measured band excess by more than a
    factor of two, so every number in this study must state which estimator it
    used. ``centroid_halfwidth = 0`` is the historical full-region weighted
    mean, i.e. the estimator behind every published PV-Finder number to date.
    """
    kw = {}
    if _SUPPORTS_CENTROID:
        kw["centroid_halfwidth"] = centroid_halfwidth
    elif centroid_halfwidth != 0:
        raise RuntimeError(
            "the installed pv_locations_updated_res has no centroid_halfwidth "
            "parameter, so only the historical estimator (0) is available"
        )
    z, h, pb, _ = pv_locations_updated_res(
        hist, threshold, integral_threshold, min_width, min_height, **kw
    )
    return (
        np.asarray(z, dtype=np.float64),
        np.asarray(h, dtype=np.float64),
        np.asarray(pb, dtype=np.int64),
    )


def predict_peaks(
    model: torch.nn.Module,
    event: dict,
    device: torch.device,
    threshold: float,
    integral_threshold: float,
    min_width: int,
    min_height: float,
    centroid_halfwidth: int = 0,
) -> np.ndarray:
    """Peak z positions in mm for one event."""
    flat = predict_hist(model, event, device)
    z, _, _ = find_peaks(
        flat,
        threshold,
        integral_threshold,
        min_width,
        min_height,
        centroid_halfwidth,
    )
    return z


def parse_cell(spec: str) -> tuple[str, str, str]:
    """``label=ckpt@prod`` -> (label, ckpt, prod)."""
    label, rest = spec.split("=", 1)
    ckpt, prod = rest.rsplit("@", 1)
    if prod not in ("old", "new"):
        raise ValueError(f"cell {spec!r}: production must be 'old' or 'new'")
    return label, ckpt, prod


def main() -> None:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", action="append", required=True, help="label=ckpt@old|new")
    p.add_argument("--root-old", required=True)
    p.add_argument("--root-new", required=True)
    p.add_argument("--n-events", type=int, default=1920)
    p.add_argument("--entry-stop", type=int, default=25000)
    p.add_argument("--mu-min", type=float, default=185.0)
    p.add_argument("--mu-max", type=float, default=215.0)
    p.add_argument("--peak-threshold", type=float, default=1e-2)
    p.add_argument("--integral-threshold", type=float, default=0.2)
    p.add_argument("--min-width", type=int, default=3)
    p.add_argument("--min-height", type=float, default=0.0)
    p.add_argument("--centroid-halfwidth", type=int, default=0,
                   help="position estimator: 0 = historical full-region weighted "
                        "mean (every published number to date), >0 = local "
                        "centroid of that half-width in bins")  # fmt: skip
    p.add_argument("--contrast", action="append", default=[],
                   help="paired contrast 'B-A', repeatable")  # fmt: skip
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--device", type=int, default=2)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    (outdir / "cells").mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(f"cuda:{args.device}")
        if args.device >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"device: {device}")

    roots = {"old": args.root_old, "new": args.root_new}
    cells = [parse_cell(c) for c in args.cell]
    prods = sorted({c[2] for c in cells})

    # Reading a production costs minutes; only do it for cells that are not
    # already cached. A cache is a per-event pair-histogram matrix, so it is a
    # complete and exact substitute for re-running that cell.
    needed = {
        prod: any(
            not (outdir / "cells" / f"{lab}.npz").exists()
            for lab in [f"AMVF_{prod}", f"Truth_{prod}"]
            + [c[0] for c in cells if c[2] == prod]
        )
        for prod in prods
    }
    entries, data = {}, {}
    ref = None
    for prod in prods:
        if not needed[prod]:
            print(f"  [{prod}] every cell cached, not reading the production")
            continue
        entries[prod] = select_entries(
            roots[prod], args.mu_min, args.mu_max, args.n_events, args.entry_stop
        )
        if ref is None:
            ref = entries[prod]
        elif not np.array_equal(ref, entries[prod]):
            raise ValueError("mu selection differs between productions")
        print(f"  [{prod}] {len(entries[prod])} entries, loading tracks and vertices")
        data[prod] = load_entries(roots[prod], entries[prod])["events"]

    summary: dict[str, dict] = {}
    profiles: dict[str, np.ndarray] = {}
    caches: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def use_cache(label: str) -> None:
        """Load a cached cell and re-derive its summary from it."""
        d = np.load(outdir / "cells" / f"{label}.npz")
        h, npk = d["hists"], d["npeaks"]
        s = observables(h.sum(0), float(npk.sum()))
        s["peaks_per_event"] = float(npk.mean())
        s["n_events"] = int(len(npk))
        s.update(bootstrap(h, npk, args.n_boot))
        summary[label], profiles[label], caches[label] = s, h.sum(0), (h, npk)

    # Fixed-algorithm references: AMVF and truth on each production.
    for prod in prods:
        for ref_name, key in (("AMVF", "amvf_z"), ("Truth", "truth_z")):
            label = f"{ref_name}_{prod}"
            if (outdir / "cells" / f"{label}.npz").exists():
                print(f"  [{label}] cached")
                use_cache(label)
                continue
            z = [e[key] for e in data[prod]]
            s, h, npk = measure(z, args.n_boot)
            summary[label], profiles[label] = s, h.sum(0)
            caches[label] = (h, npk)
            np.savez_compressed(outdir / "cells" / f"{label}.npz", hists=h, npeaks=npk)

    for label, ckpt, prod in cells:
        cache = outdir / "cells" / f"{label}.npz"
        if cache.exists():
            print(f"  [{label}] cached")
            use_cache(label)
            continue
        print(f"  [{label}] {Path(ckpt).name} on {prod} production")
        model = build_model(ckpt, device)
        z = []
        for i, ev in enumerate(data[prod]):
            z.append(
                predict_peaks(
                    model,
                    ev,
                    device,
                    args.peak_threshold,
                    args.integral_threshold,
                    args.min_width,
                    args.min_height,
                    args.centroid_halfwidth,
                )
            )
            if i % 200 == 0:
                print(f"    {i}/{len(data[prod])} peaks={len(z[-1])}", flush=True)
        del model
        torch.cuda.empty_cache()
        s, h, npk = measure(z, args.n_boot)
        np.savez_compressed(cache, hists=h, npeaks=npk)
        summary[label], profiles[label], caches[label] = s, h.sum(0), (h, npk)

    print_table(summary)
    contrasts = {}
    for spec in args.contrast:
        b, a = spec.split("-", 1)
        contrasts[spec] = paired_contrast(caches[a], caches[b], args.n_boot)
        c = contrasts[spec]
        print(
            f"  {spec:<24} d(band) = {c['band_excess_pct']['delta']:+6.2f} "
            f"+- {c['band_excess_pct']['err']:.2f} %   "
            f"d(sat/peak) = {c['satellites_per_peak']['delta']:+.4f} "
            f"+- {c['satellites_per_peak']['err']:.4f}   "
            f"d(peaks/evt) = {c['peaks_per_event']['delta']:+6.2f}"
        )
    meta = {
        "peak_finder": {
            "threshold": args.peak_threshold,
            "integral_threshold": args.integral_threshold,
            "min_width": args.min_width,
            "min_height": args.min_height,
            "centroid_halfwidth": args.centroid_halfwidth,
        },
        "mu_window": [args.mu_min, args.mu_max],
        "n_events": args.n_events,
        "entry_stop": args.entry_stop,
        "roots": roots,
        "cells": {
            label: {"ckpt": ckpt, "production": prod} for label, ckpt, prod in cells
        },
        "n_boot": args.n_boot,
    }
    with open(outdir / "summary.json", "w") as fh:
        json.dump(
            {"meta": meta, "cells": summary, "contrasts": contrasts}, fh, indent=2
        )
    plot_cells(
        profiles,
        outdir / "bump_model_vs_data.png",
        "Pairwise $|\\Delta z|$: checkpoint x production, same "
        f"{args.n_events} $\\mu$-matched held-out events",
    )
    print(f"\n  wrote {outdir}/summary.json and bump_model_vs_data.png")


if __name__ == "__main__":
    main()
