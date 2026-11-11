"""Paired comparison of held-out evaluations against a reference model.

Reads eval_results.pkl files written by run_eval_pvf_run3.py for the same
events, and reports efficiency, fakes per event and sigma_vtx-vtx for each
model, with the paired difference to the reference. Errors on the
differences come from an event bootstrap that resamples the SAME events for
both models, so the common event-to-event fluctuation cancels.

Example:
    python scripts/compare_heldout_evals.py --ref v6 \
        --eval v6=outputs/11_09_2026_output/small_model_eval/v6/r16443 \
        --eval s5=outputs/11_09_2026_output/small_model_eval/s5/r16443
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

N_BOOT = 2000
SEED = 20261109


def load(path: Path) -> dict:
    """Return per-event arrays restricted to the mu window, plus sigma."""
    with open(path / "eval_results.pkl", "rb") as f:
        d = pickle.load(f)
    win = np.asarray(d["in_mu_window"], dtype=bool)
    pe = [e for e, keep in zip(d["per_event"], win) if keep]
    return {
        "idx": np.array([e["event_idx"] for e in pe]),
        "found": np.array([e["tc"] + e["tm"] for e in pe], dtype=float),
        "truth": np.array([e["n_truth"] for e in pe], dtype=float),
        "fake": np.array([e["fake"] for e in pe], dtype=float),
        "split": np.array([e["split"] for e in pe], dtype=float),
        "sigma": float(d["sigma_vtx_vtx_mm"]),
        "sigma_err": float(d["sigma_vtx_vtx_err_mm"]),
    }


def metrics(m: dict, sel: np.ndarray) -> np.ndarray:
    """Efficiency, fakes/event, splits/event on the selected events."""
    return np.array(
        [
            m["found"][sel].sum() / m["truth"][sel].sum(),
            m["fake"][sel].mean(),
            m["split"][sel].mean(),
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--eval", action="append", required=True, help="name=eval_dir")
    p.add_argument("--ref", required=True)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    runs = {k: load(Path(v)) for k, v in (e.split("=", 1) for e in args.eval)}
    ref = runs[args.ref]
    rng = np.random.default_rng(SEED)
    n = len(ref["idx"])
    boots = rng.integers(0, n, size=(N_BOOT, n))
    out = {}
    print(f"{'model':8s} {'Eff':>8s} {'dEff[pt]':>14s} {'FP/evt':>7s} {'dFP':>12s} "
          f"{'sigma[mm]':>16s} {'dsigma[mm]':>17s}")  # fmt: skip
    for name, m in runs.items():
        if not np.array_equal(m["idx"], ref["idx"]):
            raise ValueError(f"{name}: events differ from reference {args.ref}")
        val = metrics(m, np.arange(n))
        diff = val - metrics(ref, np.arange(n))
        bd = np.array([metrics(m, b) - metrics(ref, b) for b in boots])
        err = bd.std(axis=0)
        ds = m["sigma"] - ref["sigma"]
        ds_err = np.hypot(m["sigma_err"], ref["sigma_err"])  # conservative
        out[name] = {
            "eff": val[0], "fp_per_evt": val[1], "split_per_evt": val[2],
            "d_eff_pt": 100 * diff[0], "d_eff_pt_err": 100 * err[0],
            "d_fp": diff[1], "d_fp_err": err[1],
            "sigma_mm": m["sigma"], "sigma_err_mm": m["sigma_err"],
            "d_sigma_mm": ds, "d_sigma_err_mm": ds_err, "n_events": n,
        }  # fmt: skip
        print(
            f"{name:8s} {val[0]:8.4f} {100 * diff[0]:+7.2f}±{100 * err[0]:.2f} "
            f"{val[1]:7.2f} {diff[1]:+6.2f}±{err[1]:.2f} "
            f"{m['sigma']:.4f}±{m['sigma_err']:.4f} {ds:+.4f}±{ds_err:.4f}"
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
