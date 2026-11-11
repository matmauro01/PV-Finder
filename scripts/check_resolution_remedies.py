"""Diagnostic checks for discussion of the resolution fit; no production edits.

Run from the repository root with venv/bin/python. The first 10,000 events of
the original fit sample are used for the shared-track check. The selection is
illustrative and is not a proposed change to the resolution definition.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from check_resolution_c import BASE, fit_widths, histogram_width, power
from scipy.optimize import curve_fit

OUT = Path("outputs/09_17_2026_output/resolution_remedies")
ROOT = Path(
    "data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/"
    "ATLAS_PVFinderData_601229_e8481_s4494_r16633_PU200.root"
)
N_EVENTS = 10000


def original_matcher():
    # Reuse the original pure function without importing the ROOT plotting code.
    path = Path("src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py")
    module = ast.parse(path.read_text())
    definition = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "greedy_match"
    )
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["greedy_match"], hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    match, source_hash = original_matcher()
    columns = {
        key: []
        for key in [
            "event",
            "n",
            "dz",
            "shared",
            "purity",
            "completeness",
            "n_central",
            "precision_proxy_mm",
            "nearest_truth_mm",
        ]
    }
    branches = [
        "RecoVertex_z",
        "RecoVertex_nTracks",
        "RecoVertex_assocTracks",
        "TruthVertex_z",
        "TruthVertex_nTracks",
        "TruthVertex_assocTracks",
        "RecoTrack_eta",
        "RecoTrack_ErrZ0",
    ]
    tree = uproot.open(ROOT)["PVFinderData"]
    event = 0
    for batch in tree.iterate(
        branches, entry_stop=N_EVENTS, step_size=250, library="ak"
    ):
        for row in ak.to_list(batch):
            eta = np.array(row["RecoTrack_eta"])
            errors = np.array(row["RecoTrack_ErrZ0"])
            rz = np.array(row["RecoVertex_z"], dtype=np.float32)
            tz = np.array(row["TruthVertex_z"], dtype=np.float32)
            rn = np.array(row["RecoVertex_nTracks"])
            tn = np.array(row["TruthVertex_nTracks"])
            ri, ti = np.flatnonzero(rn >= 2), np.flatnonzero(tn >= 2)
            rsets = [
                set(int(i) for i in ids if i >= 0)
                for ids in row["RecoVertex_assocTracks"]
            ]
            tsets = [
                set(int(i) for i in ids if i >= 0)
                for ids in row["TruthVertex_assocTracks"]
            ]
            for ids in rsets + tsets:
                assert all(i < len(eta) for i in ids)
            for ai, tj, _ in match(rz[ri], tz[ti], 2.0):
                r, t = int(ri[ai]), int(ti[tj])
                rs, ts = rsets[r], tsets[t]
                shared = len(rs & ts)
                ids = np.array(sorted(ts), dtype=int)
                assert len(ts) == int(tn[t])
                good = np.isfinite(errors[ids]) & (errors[ids] > 0)
                precision = (
                    float(np.sqrt(1 / np.sum(errors[ids][good] ** -2)))
                    if good.any()
                    else np.nan
                )
                neighbors = np.delete(tz[ti], tj)
                nearest = (
                    float(np.min(np.abs(neighbors - tz[t])))
                    if len(neighbors)
                    else np.inf
                )
                vals = [
                    event,
                    int(tn[t]),
                    float(rz[r] - tz[t]),
                    shared,
                    shared / max(1, len(rs)),
                    shared / max(1, len(ts)),
                    int(np.sum(np.abs(eta[ids]) < 2.5)),
                    precision,
                    nearest,
                ]
                for key, value in zip(columns, vals):
                    columns[key].append(value)
            event += 1
        if event % 2000 == 0:
            print("Read events", event, flush=True)
    result = {key: np.array(values) for key, values in columns.items()}
    original = np.load(BASE / "vertex_data.npz")
    np.testing.assert_array_equal(
        result["n"], original["truth_ntrks"][: len(result["n"])]
    )
    # The 0.1 nm tolerance allows float32 rounding, far below the widths.
    np.testing.assert_allclose(
        result["dz"], original["dz_mm"][: len(result["n"])], rtol=0, atol=1e-7
    )
    np.savez_compressed(OUT / "pairs_with_track_overlap.npz", **result)
    return result, source_hash


def shape_checks():
    saved = json.loads((BASE / "corrected/resolution_fit_v2.json").read_text())
    d = saved["per_bin"]
    x, y, error = [np.array(d[k]) for k in ("centers", "sigmas_mm", "errs_mm")]
    models = [
        ("positive_shift", lambda n, a, b, n0: a * (n + n0) ** -b, [0.25, 0.7, 2]),
        (
            "quadrature_floor",
            lambda n, a, b, s0: np.sqrt((a * n**-b) ** 2 + s0**2),
            [0.15, 0.5, 0.005],
        ),
    ]
    out = {}
    for label, func, p0 in models:
        p, _ = curve_fit(
            func,
            x,
            y,
            sigma=error,
            absolute_sigma=True,
            p0=p0,
            bounds=(0, np.inf),
            maxfev=50000,
        )
        chi2 = float(np.sum(((y - func(x, *p)) / error) ** 2))
        out[label] = {
            "parameters": p.tolist(),
            "chi2": chi2,
            "ndf": len(x) - len(p),
            "widths_um": {
                str(n): float(1000 * func(n, *p)) for n in [2, 10, 30, 50, 100, 214]
            },
        }
    p = out["positive_shift"]["parameters"]
    toy_width = p[0] * (x + p[2]) ** -p[1]
    out["noiseless_positive_curve_fitted_with_old_formula"] = fit_widths(
        x, toy_width, error
    )
    nominal = [saved["fit"][k] for k in ["A_mm", "B", "C_mm"]]
    out["current_widths_um"] = {
        str(n): float(1000 * power(float(n), *nominal))
        for n in [2, 10, 30, 50, 100, 214]
    }
    return out


def track_checks(p):
    clean = (p["purity"] >= 0.9) & (p["completeness"] >= 0.9)
    out = {}
    for lo, hi in [(2, 3), (3, 4), (5, 6), (10, 11), (20, 21), (40, 50), (50, 70)]:
        m = (p["n"] >= lo) & (p["n"] < hi)
        result = {
            "n_low": lo,
            "n_high": hi,
            "matched_pairs": int(m.sum()),
            "no_shared_track_fraction": float(np.mean(p["shared"][m] == 0)),
            "high_overlap_fraction": float(np.mean(clean[m])),
        }
        for label, selected in [
            ("all_positional_matches", m),
            ("high_track_overlap", m & clean),
            ("no_shared_tracks", m & (p["shared"] == 0)),
        ]:
            dz = p["dz"][selected]
            if len(dz) < 200:
                continue
            q16, q84 = np.quantile(dz, [0.16, 0.84])
            result[label] = {
                "pairs": len(dz),
                "half_central_68_percent_um": float((q84 - q16) * 500),
                "tail_fraction_abs_dz_gt_0p3mm": float(np.mean(np.abs(dz) > 0.3)),
                "median_precision_proxy_um": float(
                    1000 * np.nanmedian(p["precision_proxy_mm"][selected])
                ),
                "median_fraction_central_tracks": float(
                    np.median(p["n_central"][selected] / p["n"][selected])
                ),
            }
            if label != "no_shared_tracks":
                result[label]["integrated_gaussian_flat_fit"] = histogram_width(
                    dz, 2, 400, True, poisson=True
                )
        tail = m & (np.abs(p["dz"]) > 0.3)
        result["fraction_of_tail_with_no_shared_tracks"] = (
            float(np.mean(p["shared"][tail] == 0)) if tail.any() else None
        )
        out[f"n{lo}_{hi}"] = result
    return out


def draw(p):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    clean = (p["purity"] >= 0.9) & (p["completeness"] >= 0.9)
    edges = np.linspace(-2, 2, 201)
    for ax, n in zip(axes, [2, 10]):
        for mask, label, color in [
            (p["n"] == n, "All positional matches", "black"),
            ((p["n"] == n) & (p["shared"] == 0), "No shared tracks", "#d45b2c"),
            ((p["n"] == n) & clean, "At least 90% overlap each way", "#2466b4"),
        ]:
            counts, _ = np.histogram(p["dz"][mask], edges)
            ax.stairs(counts, edges, label=label, color=color, lw=1.3)
        ax.set(
            yscale="log",
            ylim=(0.7, None),
            xlim=(-2, 2),
            xlabel="AMVF z - truth z [mm]",
            ylabel="Pairs per 0.02 mm bin",
            title=f"{n} truth tracks",
        )
        ax.legend(frameon=False, fontsize=9)
        ax.tick_params(direction="in", top=True, right=True)
    fig.suptitle(
        "First 10,000 events: shared-track check of positional matches", fontsize=12
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"shared_track_check.{ext}", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, source_hash = collect()
    result = {
        "source_root": str(ROOT),
        "entry_start": 0,
        "entry_stop": N_EVENTS,
        "matched_pairs": len(pairs["n"]),
        "matcher_file_sha256": source_hash,
        "prefix_matches_original_pairs": {
            "truth_counts_exact": True,
            "residual_absolute_tolerance_mm": 1e-7,
        },
        "overlap_definition": "shared / reco >= 0.9 and shared / truth >= 0.9; counts of exported associated tracks",
        "track_checks": track_checks(pairs),
        "shape_checks": shape_checks(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    draw(pairs)
    print("Wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
