#!/usr/bin/env python3
"""Run 2 MC: save the low pile-up density model's 12,000-bin histograms.

The low pile-up GNN chain (R2) builds its inference graphs from these
histograms with ``gnn.data.pvf_to_graphs --regime lowpu``. Row i is h5 test
event ``indices[i]`` (48450 + i), i.e. TTVA h5 key ``Event{qibin[i]}``, the
same order as ``run_eval_pvf.py`` and ``configs/qibin_test_main_indices_v2.p``.

The model and its architecture come from ``get_regime("lowpu")``; the inference
code is ``run_eval_pvf.py``'s own (``run_e2e``), so the peaks equal the vertex
finding's up to GPU rounding (checked downstream).

Usage (repo root, venv active, inside tmux):
  python -u scripts/lowpu_mc_histograms.py --device 2 \
      --out outputs/11_12_2026_output/lowpu_final/mc_hists
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src/pv_finder/evaluation/vertex_finding"))

from run_eval_pvf import (  # noqa: E402
    _DEFAULT_H5,
    E2E_CONFIG,
    N_SUBEVENTS,
    load_ckpt,
    run_e2e,
)

from pv_finder.models.autoencoder_models import trackstoHists_UNet_1000  # noqa: E402
from pv_finder.utils.working_points import get_regime  # noqa: E402


def sha256(path: Path) -> str:
    """sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--indices", default="configs/test_main_indices_2550evt.p")
    ap.add_argument("--h5", default=_DEFAULT_H5)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    regime = get_regime("lowpu")
    dm = regime.density
    if dm.model_type != "v1" or dm.unet_channels != E2E_CONFIG["n_UNetChannels"]:
        raise ValueError(
            "lowpu density model no longer matches run_eval_pvf's E2E_CONFIG"
        )
    if tuple(E2E_CONFIG["l_HiddenNodes"]) != dm.hidden:
        raise ValueError("lowpu hidden layers differ from E2E_CONFIG")
    use_gpu = args.device >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if use_gpu else "cpu")
    model = trackstoHists_UNet_1000(**E2E_CONFIG)
    load_ckpt(str(REPO / dm.checkpoint), model, device)

    with open(args.indices, "rb") as fh:
        indices = [int(i) for i in pickle.load(fh)]
    hists = np.zeros((len(indices), 12000), dtype=np.float32)
    with h5py.File(args.h5, "r") as h5f:
        for row, eidx in enumerate(indices):
            s0 = eidx * N_SUBEVENTS
            hists[row] = run_e2e(h5f["tracks"][s0 : s0 + N_SUBEVENTS], model, device)
            if row % 250 == 0:
                print(f"  {row}/{len(indices)} (h5 event {eidx})", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "hists_e400.npy", hists)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    meta = {
        "regime": regime.to_dict(),
        "checkpoint_sha256": sha256(REPO / dm.checkpoint),
        "indices": args.indices,
        "h5": args.h5,
        "n_events": len(indices),
        "first_last_h5_event": [indices[0], indices[-1]],
        "device": str(device),
        "git_hash": git.stdout.strip(),
        "argv": sys.argv,
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {hists.shape} to {args.out / 'hists_e400.npy'}")


if __name__ == "__main__":
    main()
