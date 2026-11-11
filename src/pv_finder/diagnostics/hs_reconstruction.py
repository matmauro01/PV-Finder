"""Compare HS position-finding efficiency using saved, aligned evaluation outputs.

The measured condition is min |z_reco - z_HS| < 0.1 mm, as described in
IDTR-2023-01 Section 5.5. This is not hard-scatter selection by sum(pT**2).
The density axis counts stored truth PU vertices, not all generated collisions.
Run this file by path; no inference or training is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import beta, binomtest

ARMS = ("pvf", "amvf")
CONFIDENCE = 0.682689492137
DENSITY_EDGES = np.array([0, 0.5, 1, 1.5, 2, 4.0])


def closest_distance(reco_z, truth_z):
    """An empty reconstructed list is a failure, including in the denominator."""
    reco_z = np.asarray(reco_z, dtype=np.float64)
    if not np.all(np.isfinite(reco_z)) or not np.isfinite(truth_z):
        raise ValueError("Non-finite vertex position")
    return float(np.min(np.abs(reco_z - truth_z))) if len(reco_z) else np.inf


def local_density(truth_z, hs_index, halfwidth=2.0):
    """Other stored truth vertices in |dz| < 2 mm, divided by 4 mm.

    There is no additional track cut here. In the analyzed ROOT files all
    stored truth vertices already have >=1 selected reconstructed track.
    """
    truth_z = np.asarray(truth_z, dtype=np.float64)
    nearby = np.abs(truth_z - truth_z[hs_index]) < halfwidth
    nearby[hs_index] = False
    return float(nearby.sum() / (2 * halfwidth))


def interval(passed, total):
    """Equal-tailed 68.27% Clopper-Pearson interval, including endpoint bins."""
    if not 0 <= passed <= total:
        raise ValueError("Invalid binomial counts")
    if total == 0:
        return {"passed": 0, "total": 0, "efficiency": None, "low": None, "high": None}
    alpha = 1 - CONFIDENCE
    low = 0.0 if passed == 0 else float(beta.ppf(alpha / 2, passed, total - passed + 1))
    high = (
        1.0
        if passed == total
        else float(beta.ppf(1 - alpha / 2, passed + 1, total - passed))
    )
    return {
        "passed": int(passed),
        "total": int(total),
        "efficiency": passed / total,
        "low": low,
        "high": high,
    }


def compare(events, mask, window=0.1):
    flags = {a: events[f"{a}_dz_mm"][mask] < window for a in ARMS}
    n = int(np.count_nonzero(mask))
    out = {a: interval(int(flags[a].sum()), n) for a in ARMS}
    pvf_only = int(np.count_nonzero(flags["pvf"] & ~flags["amvf"]))
    amvf_only = int(np.count_nonzero(flags["amvf"] & ~flags["pvf"]))
    discordant = pvf_only + amvf_only
    diff = flags["pvf"].astype(float) - flags["amvf"].astype(float)
    out["paired"] = {
        "both_pass": int(np.count_nonzero(flags["pvf"] & flags["amvf"])),
        "both_fail": int(np.count_nonzero(~flags["pvf"] & ~flags["amvf"])),
        "pvf_only_pass": pvf_only,
        "amvf_only_pass": amvf_only,
        "difference": float(diff.mean()) if n else None,
        "difference_se": float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
        "mcnemar_exact_p": float(binomtest(pvf_only, discordant).pvalue)
        if discordant
        else 1.0,
    }
    return out


def binned(events, variable, edges, mask):
    rows = []
    for j, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        upper = (
            events[variable] <= right
            if j == len(edges) - 2
            else events[variable] < right
        )
        selected = mask & (events[variable] >= left) & upper
        rows.append(
            {"left": float(left), "right": float(right), **compare(events, selected)}
        )
    return rows


def extract(cache_path, allow_missing_hs=False):
    """Read ROOT truth/AMVF and verify all saved event indices and positions."""
    import uproot

    cache = np.load(cache_path, allow_pickle=False)
    meta = json.loads(str(cache["meta"]))
    with open(meta["pkl"], "rb") as handle:
        saved = pickle.load(handle)
    records = saved["per_event"]
    if len(records) != len(saved["pred_pvs_mm"]) or len(records) != len(
        saved["truth_pvs_mm"]
    ):
        raise ValueError("Saved per-event arrays have different lengths")
    indices = np.array([r["event_idx"] for r in records], dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
        raise ValueError("Invalid or repeated event indices")
    tree = uproot.open(meta["root"])["PVFinderData"]
    branches = [
        "TruthVertex_z",
        "TruthVertex_type",
        "TruthVertex_nTracks",
        "RecoVertex_z",
        "RecoVertex_nTracks",
        "ActualNumOfInt",
        "BeamPosZ",
    ]
    print(
        f"Reading {int(indices.max()) + 1} ROOT entries for {meta['tag']}", flush=True
    )
    root = tree.arrays(branches, entry_stop=int(indices.max()) + 1, library="np")
    rows, invalid_hs = [], []
    n_truth_zero = 0
    n_truth_stored = 0
    type_counts = {}
    for k, e in enumerate(indices):
        rec = records[k]
        tz = np.asarray(root["TruthVertex_z"][e])
        tn = np.asarray(root["TruthVertex_nTracks"][e])
        tt = np.asarray(root["TruthVertex_type"][e])
        rz = np.asarray(root["RecoVertex_z"][e])
        rn = np.asarray(root["RecoVertex_nTracks"][e])
        pvf = np.asarray(saved["pred_pvs_mm"][k])
        amvf = rz[rn >= 2]
        if not (len(tz) == len(tn) == len(tt) and len(rz) == len(rn)):
            raise ValueError(f"ROOT branch alignment failed at entry {e}")
        if not np.array_equal(np.sort(tz[tn >= 2]), np.sort(saved["truth_pvs_mm"][k])):
            raise ValueError(f"Truth position alignment failed at entry {e}")
        if (
            rec["n_truth"] != int(np.sum(tn >= 2))
            or rec["n_amvf"] != len(amvf)
            or rec["n_pred"] != len(pvf)
        ):
            raise ValueError(f"Multiplicity alignment failed at entry {e}")
        if float(root["ActualNumOfInt"][e]) != rec["mu"]:
            raise ValueError(f"Pile-up alignment failed at entry {e}")
        if root["BeamPosZ"][e] != 0 or rec["beam_z"] != 0:
            raise ValueError(
                "Nonzero beam position requires explicit coordinate conversion"
            )
        hs = np.flatnonzero(tt == 1)
        n_truth_zero += int(np.count_nonzero(tn == 0))
        n_truth_stored += len(tz)
        for value, count in zip(*np.unique(tt, return_counts=True)):
            type_counts[str(int(value))] = type_counts.get(str(int(value)), 0) + int(
                count
            )
        if len(hs) != 1:
            invalid_hs.append({"event_idx": int(e), "hs_count": len(hs)})
            continue
        h = int(hs[0])
        rows.append(
            {
                "event_idx": int(e),
                "mu": float(rec["mu"]),
                "hs_z_mm": float(tz[h]),
                "hs_ntracks": int(tn[h]),
                "stored_truth_count": len(tz),
                "stored_pu_density": local_density(tz, h),
                "pvf_dz_mm": closest_distance(pvf, float(tz[h])),
                "amvf_dz_mm": closest_distance(amvf, float(tz[h])),
                "amvf_all_dz_mm": closest_distance(rz, float(tz[h])),
            }
        )
    # Never silently drop events with absent or ambiguous HS truth.
    if invalid_hs and not allow_missing_hs:
        raise ValueError(f"Cannot define unbiased HS denominator: {invalid_hs[:10]}")
    if invalid_hs:
        print(
            f"Explicitly excluding events without unique stored HS truth: {invalid_hs}",
            flush=True,
        )
    events = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    pu = (events["mu"] >= meta["mu_min"]) & (events["mu"] <= meta["mu_max"])
    if not np.array_equal(events["event_idx"][pu], cache["event_idx"]):
        raise ValueError("PU200 event list differs from the note's comparison cache")
    by_entry = {int(e): k for k, e in enumerate(indices)}
    for j, e in enumerate(cache["event_idx"]):
        for arm in ARMS:
            positions = (
                saved["pred_pvs_mm"][by_entry[int(e)]]
                if arm == "pvf"
                else root["RecoVertex_z"][e][root["RecoVertex_nTracks"][e] >= 2]
            )
            expected = cache[f"{arm}_z"][
                cache[f"{arm}_off"][j] : cache[f"{arm}_off"][j + 1]
            ]
            if not np.array_equal(positions, expected):
                raise ValueError(f"{arm} position cache mismatch at entry {e}")
    meta.update(
        {
            "n_evaluated": len(rows),
            "n_invalid_hs": len(invalid_hs),
            "invalid_hs_events": invalid_hs,
            "n_input_events": len(records),
            "stored_truth_vertices": n_truth_stored,
            "zero_track_truth_vertices": n_truth_zero,
            "truth_type_counts": type_counts,
            "hs_ntracks_min": int(events["hs_ntracks"].min()),
            "hs_ntracks_max": int(events["hs_ntracks"].max()),
            "all_events_aligned": True,
            "pu200_cache_positions_equal": True,
            "checkpoint": saved["e2e_checkpoint"],
            "centroid_halfwidth": saved["centroid_halfwidth"],
            "source_pkl_sha256": hashlib.sha256(
                Path(meta["pkl"]).read_bytes()
            ).hexdigest(),
            "cache_sha256": hashlib.sha256(Path(cache_path).read_bytes()).hexdigest(),
        }
    )
    return events, meta


def plot(rows, path, xlabel, subtitle, footnote, xlim):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 14,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.22, top=0.80)
    fig.text(
        0.13, 0.94, "Hard-scatter vertex reconstruction", fontsize=18, weight="bold"
    )
    fig.text(0.13, 0.889, subtitle, fontsize=12.5)
    fig.text(
        0.13,
        0.845,
        r"At least one reconstructed vertex with $|z_{\rm reco}-z_{\rm HS}|<0.1$ mm",
        fontsize=11.5,
    )
    ax.axhline(1, color="#777777", ls=":", lw=1)
    lower_limits = []
    for arm, color, marker, label in [
        ("pvf", "#2466b4", "o", "PV-Finder"),
        ("amvf", "#d45b2c", "s", "AMVF"),
    ]:
        valid = [r for r in rows if r[arm]["total"]]
        x = np.array([(r["left"] + r["right"]) / 2 for r in valid])
        y = np.array([r[arm]["efficiency"] for r in valid])
        lo = np.array([r[arm]["low"] for r in valid])
        lower_limits.extend(lo.tolist())
        hi = np.array([r[arm]["high"] for r in valid])
        xe = np.array([(r["right"] - r["left"]) / 2 for r in valid])
        ax.errorbar(
            x,
            y,
            xerr=xe,
            yerr=[y - lo, hi - y],
            fmt=marker,
            color=color,
            mfc="white" if arm == "amvf" else color,
            ms=8.5 if arm == "pvf" else 6.0,
            capsize=3,
            elinewidth=1.3,
            label=label,
            zorder=3,
        )
    ymin = min(0.94, np.floor((min(lower_limits) - 0.005) * 100) / 100)
    ax.set(
        xlim=xlim,
        ylim=(ymin, 1.035),
        xlabel=xlabel,
        ylabel="HS reconstruction efficiency",
    )
    if ymin == 0.94:
        ax.set_yticks([0.94, 0.96, 0.98, 1.0, 1.02])
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="upper right", frameon=False)
    fig.text(0.13, 0.065, footnote, fontsize=10.5, linespacing=1.5)
    for suffix in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + suffix), dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--reuse-events", action="store_true")
    parser.add_argument(
        "--allow-missing-hs",
        action="store_true",
        help="Explicitly report and omit entries without exactly one stored HS",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    event_path = args.outdir / "events.npz"
    if args.reuse_events:
        with np.load(event_path, allow_pickle=False) as data:
            events = {k: data[k] for k in data.files if k != "metadata"}
            meta = json.loads(str(data["metadata"]))
        if meta["cache_sha256"] != hashlib.sha256(args.cache.read_bytes()).hexdigest():
            raise ValueError("Reuse cache provenance mismatch")
    else:
        events, meta = extract(args.cache, args.allow_missing_hs)
        np.savez_compressed(event_path, **events, metadata=np.array(json.dumps(meta)))
    if meta["n_invalid_hs"] and not args.allow_missing_hs:
        raise ValueError("Reusing incomplete HS truth requires --allow-missing-hs")
    pu = (events["mu"] >= meta["mu_min"]) & (events["mu"] <= meta["mu_max"])
    all_events = np.ones(len(events["mu"]), dtype=bool)
    densities = binned(events, "stored_pu_density", DENSITY_EDGES, pu)
    mu_rows = binned(events, "mu", np.arange(0, 201, 20), all_events)
    if sum(r["pvf"]["total"] for r in densities) != int(pu.sum()):
        raise ValueError("Density bins do not cover all PU200 events")
    if sum(r["pvf"]["total"] for r in mu_rows) != len(events["mu"]):
        raise ValueError("Pile-up bins do not cover all evaluated events")
    result = {
        "metadata": meta,
        "definition": "min |z_reco - z_HS| < 0.1 mm; unweighted events",
        "confidence_level": CONFIDENCE,
        "interval": "equal-tailed Clopper-Pearson",
        "pu200": compare(events, pu),
        "flat_mu": compare(events, all_events),
        "density_bins_pu200": densities,
        "mu_bins": mu_rows,
        "itk_style_density_bins_pu200": binned(
            events, "stored_pu_density", np.array([0, 0.5, 1, 1.5, 2, 2.5, 4]), pu
        ),
        "density_definition": "Other stored truth vertices with |z-z_HS|<2 mm / 4 mm; not all generated interactions",
        "sensitivity_windows_mm": {
            str(w): compare(events, pu, w) for w in (0.05, 0.1, 0.1694, 0.5)
        },
        "amvf_without_track_count_cut_pu200": interval(
            int(np.sum(events["amvf_all_dz_mm"][pu] < 0.1)), int(pu.sum())
        ),
        "pu200_failed_event_indices": {
            a: events["event_idx"][pu & (events[f"{a}_dz_mm"] >= 0.1)]
            .astype(int)
            .tolist()
            for a in ARMS
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.outdir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (args.outdir / "events.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(events.keys())
        writer.writerows(zip(*events.values()))
    n = int(pu.sum())
    subtitle = rf"HL-LHC $t\bar{{t}}$ simulation  |  {n:,} events, $185\leq\mu\leq200$"
    plot(
        densities,
        args.outdir / "hs_efficiency_vs_density",
        "Stored truth pile-up density [vertices/mm]",
        subtitle,
        "Density uses stored truth vertices with at least one selected track, excluding HS.\nA 4 mm window is used; the sparse tail is grouped. Error bars: 68.27% intervals.",
        (0, 4),
    )
    plot(
        mu_rows,
        args.outdir / "hs_efficiency_vs_mu",
        r"Pile-up $\mu$",
        rf"HL-LHC $t\bar{{t}}$ simulation  |  {len(events['mu']):,} validation events with HS truth",
        "Same events and 0.1 mm condition for both methods.\nError bars: 68.27% exact binomial intervals.",
        (0, 200),
    )
    print(
        json.dumps(
            {
                "pu200": result["pu200"],
                "flat_mu": result["flat_mu"],
                "failed_events": result["pu200_failed_event_indices"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
