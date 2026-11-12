#!/usr/bin/env python3
"""Note Section 6.2: MC overlays on Fig. 6.2, and AMVF-as-reference bias in MC.

Reads only existing outputs (no inference):
  outputs/04_16_2026_output/reproduction_ep300_{mc,run2,run3}/eval_results.pkl
  data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root  (MC truth + AMVF)
  configs/qibin_test_main_indices_v2.p                    (h5 -> ROOT map)
  outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz (optional)

Writes figures (PDF+PNG) and results.json to --out.

Usage (from repo root, venv active):
  python -u scripts/data_mc_validation.py \
      --out outputs/11_09_2026_output/data_mc_validation
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from pv_finder.diagnostics.amvf_fair_comparison.scan import (  # noqa: E402
    core_position_resolution,
)
from pv_finder.diagnostics.data_mc_validation import (  # noqa: E402
    plots,
    reference,
    samples,
)
from pv_finder.utils.constants import MATCH_WINDOW_MM  # noqa: E402

PKL = "outputs/04_16_2026_output/reproduction_ep300_{}/eval_results.pkl"
MC_ROOT = "data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root"
QIBIN = "configs/qibin_test_main_indices_v2.p"
HLLHC_CACHE = "outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz"
N_BOOT = 10_000


def _yields(mu: np.ndarray, pvf: np.ndarray, amvf: np.ndarray) -> dict[str, object]:
    """Per-event means, the old figure's mean-of-bin-means, and linear slopes."""
    mus, _, prof = plots.mu_profile(mu, {"pvf": pvf, "amvf": amvf})
    out: dict[str, object] = {"mean_mu": float(np.mean(mu)), "n_events": int(len(mu))}
    for k, v in (("pvf", pvf), ("amvf", amvf)):
        out[k] = {
            "mean_per_evt": float(np.mean(v)),
            "sem": float(np.std(v) / np.sqrt(len(v))),
            "mean_of_mu_bin_means": float(np.mean(prof[k][0])),  # old figure box
            "slope_per_mu": [float(c) for c in np.polyfit(mu, v, 1)],
        }
    return out


def _reweighted_mc(data_mu: np.ndarray, mc_mu: np.ndarray,
                   mc_vals: dict[str, np.ndarray]) -> dict[str, object]:  # fmt: skip
    """MC means after reweighting MC to the data integer-mu profile."""
    dr, mr = np.round(data_mu).astype(int), np.round(mc_mu).astype(int)
    common = np.intersect1d(dr, mr)
    dfrac = {m: np.mean(dr == m) for m in common}
    mfrac = {m: np.mean(mr == m) for m in common}
    w = np.array([dfrac[m] / mfrac[m] if m in dfrac else 0.0 for m in mr])
    out: dict[str, object] = {
        "data_event_fraction_covered": float(np.isin(dr, common).mean()),
        "mc_effective_events": float(w.sum() ** 2 / np.sum(w**2)),
    }
    for k, v in mc_vals.items():
        out[k] = float(np.sum(w * v) / np.sum(w))
    return out


def _grid(name_counts: dict[str, reference.EventCounts]) -> dict[str, dict]:
    return {
        k: reference.summary_with_errors(c, n_boot=N_BOOT)
        for k, c in name_counts.items()
    }


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-hllhc", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    fig_dir = out / "figures"
    res: dict[str, object] = {"window_fixed_mm": MATCH_WINDOW_MM}

    # ---- 1. samples and the reproduced sigma fits -----------------------
    S = {
        k: samples.load_sample(k, ROOT_DIR / PKL.format(k))
        for k in ("mc", "run2", "run3")
    }
    fits = {
        "mc": samples.fit_sigma(S["mc"].dz, "fixed"),
        "run2": samples.fit_sigma(S["run2"].dz, "adaptive"),
        "run3": samples.fit_sigma(S["run3"].dz, "adaptive"),
    }
    res["sigma"] = {k: {"fit": f.sigma, "err": f.sigma_err,
                        "pkl": S[k].raw["sigma_vtx_vtx_mm"],
                        "n_pairs": int(len(S[k].dz))} for k, f in fits.items()}  # fmt: skip
    res["sigma_mc_adaptive_p0"] = samples.fit_sigma(S["mc"].dz, "adaptive").sigma
    res["sigma_300bins_info"] = {
        k: samples.fit_sigma(S[k].dz, "adaptive", n_bins=300).sigma for k in S
    }
    for k, f in fits.items():
        print(f"sigma {k}: {f.sigma:.5f} +- {f.sigma_err:.5f} (pkl {S[k].raw['sigma_vtx_vtx_mm']:.5f})")  # fmt: skip

    # ---- 2. MC ROOT vertices and the h5 -> ROOT mapping proof -----------
    root = samples.load_mc_root(
        ROOT_DIR / MC_ROOT, out / "cache" / "mc_root_vertices.npz"
    )
    with open(ROOT_DIR / QIBIN, "rb") as fh:
        qibin = [int(x) for x in pickle.load(fh)]
    res["mc_mapping"] = samples.verify_mc_mapping(S["mc"], root, qibin)
    print("mapping:", json.dumps(res["mc_mapping"]))
    mapm = res["mc_mapping"]["qibin_mapping"]
    if min(mapm[k] for k in ("truth_set_equal", "n_truth_ge2_equal", "n_amvf_ge2_equal",
                             "mu_equal")) != S["mc"].n_events:  # fmt: skip
        raise RuntimeError("h5 -> ROOT mapping not verified on every event")
    mcv = samples.mc_vertices_for_events(root, qibin[: S["mc"].n_events])
    if np.any(mcv["beam_z"] != 0.0):
        raise RuntimeError(
            "MC BeamPosZ != 0: the data-style beam correction is not a no-op"
        )
    truth_ref = mcv["truth_ge2"]
    amvf_ref = [a - b for a, b in zip(mcv["amvf_ge2"], mcv["beam_z"])]  # data-style

    # ---- 3. matching grid ------------------------------------------------
    sig = {k: f.sigma for k, f in fits.items()}
    windows = {"fitted_sample": None, "common_0.416": sig["mc"],
               f"fixed_{MATCH_WINDOW_MM:g}": MATCH_WINDOW_MM}  # fmt: skip
    grid: dict[str, object] = {}
    for wname, wval in windows.items():
        wv = {k: (sig[k] if wval is None else wval) for k in S}
        c = {
            "mc_truth_ref": reference.match_events(truth_ref, S["mc"].pred, wv["mc"]),
            "mc_amvf_ref": reference.match_events(amvf_ref, S["mc"].pred, wv["mc"]),
            "run2_data_amvf_ref": reference.match_events(
                S["run2"].ref_pkl, S["run2"].pred, wv["run2"]
            ),  # noqa: E501
            "run3_data_amvf_ref": reference.match_events(
                S["run3"].ref_pkl, S["run3"].pred, wv["run3"]
            ),  # noqa: E501
            "mc_amvf_vs_truth": reference.match_events(
                truth_ref, mcv["amvf_ge2"], wv["mc"]
            ),  # noqa: E501
        }
        g = _grid(c)
        g["windows_mm"] = wv
        g["mc_delta_amvfref_minus_truthref"] = reference.paired_delta(
            c["mc_amvf_ref"], c["mc_truth_ref"], n_boot=N_BOOT)  # fmt: skip
        grid[wname] = g
        print(f"\n[{wname}]")
        for k in ("mc_truth_ref", "mc_amvf_ref", "run2_data_amvf_ref",
                  "run3_data_amvf_ref", "mc_amvf_vs_truth"):  # fmt: skip
            s = g[k]
            print(f"  {k:20s} rec={s['recovery']:.5f}+-{s['recovery_err']:.5f} "
                  f"noref={s['no_ref_per_evt']:.4f}+-{s['no_ref_per_evt_err']:.4f} "
                  f"split={s['split_per_evt']:.4f}+-{s['split_per_evt_err']:.4f}")  # fmt: skip
        for m, d in g["mc_delta_amvfref_minus_truthref"].items():
            print(f"  delta {m}: {d['delta']:+.5f} +- {d['delta_err']:.5f}")
    res["matching"] = grid
    pkl_mc = S["mc"].raw
    res["mc_truth_ref_reproduction"] = {
        "pkl_eff": pkl_mc["overall_efficiency"], "pkl_fp": pkl_mc["fp_rate_per_evt"],
        "pkl_split": pkl_mc["total_split"], "pkl_merged": pkl_mc["total_merged"],
    }  # fmt: skip

    # ---- 4. core position resolution on MC ------------------------------
    p68 = {
        "pvf": core_position_resolution(truth_ref, S["mc"].pred),
        "amvf": core_position_resolution(truth_ref, mcv["amvf_ge2"]),
    }
    res["mc_core_p68_mm"] = {
        **p68,
        "3xp68_pvf": 3 * p68["pvf"],
        "3xp68_amvf": 3 * p68["amvf"],
    }
    print("\np68:", res["mc_core_p68_mm"])

    # ---- 5. yields and figures ------------------------------------------
    mc_vals = {"pvf": S["mc"].n_pred.astype(float),
               "amvf": np.array([len(a) for a in mcv["amvf_ge2"]], dtype=float)}  # fmt: skip
    res["yields"] = {"mc": _yields(S["mc"].mu, mc_vals["pvf"], mc_vals["amvf"])}
    mc_lab = (r"Sim. $t\bar{t}$, $\sqrt{s}$ = 13 TeV, "
              rf"$\langle\mu\rangle$ = {S['mc'].mu.mean():.1f}")  # fmt: skip
    data_lab = {
        "run2": r"Data 2018, $\sqrt{s}$ = 13 TeV, ",
        "run3": r"Data Run 3, $\sqrt{s}$ = 13.6 TeV, ",
    }
    caveat = {"run2": None,
              "run3": "Run 2 sim. shown (no Run 3 sim.)"}  # fmt: skip
    res["figures"] = {}
    for k in ("run2", "run3"):
        lab = plots.Labels(
            data=data_lab[k] + rf"$\langle\mu\rangle$ = {S[k].mu.mean():.1f}",
            mc=mc_lab, caveat=caveat[k])  # fmt: skip
        tag = "datamc" if k == "run2" else "data_run2mc"
        res["figures"][f"{k}_resolution"] = plots.plot_resolution_overlay(
            fits[k], fits["mc"], lab, fig_dir, f"{k}_resolution_{tag}")  # fmt: skip
        dvals = {"pvf": S[k].n_pred.astype(float), "amvf": S[k].n_ref.astype(float)}
        res["figures"][f"{k}_yield"] = plots.plot_yield_overlay(
            S[k].mu, dvals, S["mc"].mu, mc_vals, lab, fig_dir, f"{k}_yield_{tag}")  # fmt: skip
        y = _yields(S[k].mu, dvals["pvf"], dvals["amvf"])
        y["mc_reweighted_to_this_mu"] = _reweighted_mc(S[k].mu, S["mc"].mu, mc_vals)
        for a in ("pvf", "amvf"):
            y[a]["data_over_mc_reweighted"] = (
                y[a]["mean_per_evt"] / y["mc_reweighted_to_this_mu"][a])  # fmt: skip
        res["yields"][k] = y
    print("\nyields:", json.dumps(res["yields"], indent=1))

    # ---- 6. optional HL-LHC second point --------------------------------
    if not args.skip_hllhc and (ROOT_DIR / HLLHC_CACHE).exists():
        d = np.load(ROOT_DIR / HLLHC_CACHE)

        def _split(key: str) -> list[np.ndarray]:
            z, off = d[f"{key}_z"], d[f"{key}_off"]
            return [z[off[i] : off[i + 1]] for i in range(len(off) - 1)]

        t, p, a = _split("truth_ge2"), _split("pvf"), _split("amvf")
        ct = reference.match_events(t, p, MATCH_WINDOW_MM)
        ca = reference.match_events(a, p, MATCH_WINDOW_MM)
        res[f"hllhc_r16443_{MATCH_WINDOW_MM:g}"] = {
            "meta": json.loads(str(d["meta"])),
            "delta_amvfref_minus_truthref": reference.paired_delta(
                ca, ct, n_boot=N_BOOT
            ),
            "amvf_vs_truth": reference.summary_with_errors(
                reference.match_events(t, a, MATCH_WINDOW_MM), n_boot=N_BOOT
            ),  # fmt: skip
        }
        print(
            "\nHL-LHC:",
            json.dumps(
                res[f"hllhc_r16443_{MATCH_WINDOW_MM:g}"]["delta_amvfref_minus_truthref"]
            ),
        )  # noqa: E501

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out / 'results.json'}")


if __name__ == "__main__":
    main()
