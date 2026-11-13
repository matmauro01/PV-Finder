#!/usr/bin/env python3
"""GNN v3 campaign: compare models against AMVF on identical events, versus t.

Input: ``chain_t_metrics`` records (``--hs-min-tracks 2``) of several models on
the same events (asserted). For every model, every t of its grid and every mu
bin (inclusive, PU200 = 185-215, and 10-wide bins 0-10 ... 190-200) it compares
each chain metric with AMVF on the same events:

  higher is better: Clean/truth, Clean/reco, track efficiency / purity / F1,
                    HS selection (AMVF: stored primary), Clean HS, HS purity,
                    HS tracks on the HS vertex
  lower is better:  Merged, Split, Fake per event, HS resolution (adaptive fit)

A (metric, bin) "fails" at t when the model is worse than AMVF (strict), and
"fails significantly" when worse by more than one paired bootstrap standard
deviation (200 event replicas shared with AMVF). For each model it reports the
t with the fewest failures and every t with none. Matched operating points
interpolate along t (linear between grid points, first crossing) to a target
value of one metric and read the others there.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_eval_summary.py \
        --metrics v2e527=.../val_s8_v2e527.npz a=.../val_s8_a.npz --out .../eval/phase1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.evaluation.chain_t_summary import (  # noqa: E402
    _field,
    _jsonable,
    boot_weights,
    load,
    summarize,
)

PU = "PU200 (185-215)"
BINS = {
    "inclusive": (0, 1e9), PU: (185, 215.5),
    **{f"{lo}-{lo + 10}": (lo, lo + 10 if lo < 190 else 200.5) for lo in range(0, 200, 10)},
}  # fmt: skip
# key, label, sense (+1 higher better), source
METRICS = (
    ("clean_per_truth", "Clean/truth", 1, "ratio"),
    ("clean_per_reco", "Clean/reco", 1, "ratio"),
    ("merged_per_event", "Merged/evt", -1, "ratio"),
    ("split_per_event", "Split/evt", -1, "ratio"),
    ("fake_per_event", "Fake/evt", -1, "ratio"),
    ("trk_eff", "trk eff", 1, "ratio"),
    ("trk_pur", "trk pur", 1, "ratio"),
    ("trk_f1", "trk F1", 1, "f1"),
    ("hs_sel", "HS sel", 1, "sel"),
    ("hs_clean", "Clean HS", 1, "ratio"),
    ("hs_purity", "HS purity", 1, "ratio"),
    ("hs_track_frac", "HS trk on HS vtx", 1, "ratio"),
    ("hs_res", "HS res [um]", -1, "res"),
)
MATCHED = (
    # (x metric, target or "amvf", y metrics)
    ("trk_pur", "amvf", ("trk_eff", "trk_f1", "clean_per_truth")),
    ("trk_pur", 0.85, ("trk_eff", "trk_f1", "clean_per_truth")),
    ("hs_track_frac", "amvf", ("hs_purity", "hs_clean")),
    ("hs_track_frac", 0.80, ("hs_purity", "hs_clean")),
    ("clean_per_reco", 0.80, ("clean_per_truth", "trk_eff")),
)
COLORS = {"v2e527": "#000000", "a": "#E69F00", "b": "#56B4E9", "c": "#009E73",
          "d": "#0072B2", "amvf": "#D55E00"}  # fmt: skip  # Okabe-Ito, fixed order


def f1_table(rec, mask: np.ndarray, w: np.ndarray) -> dict:  # noqa: ANN001
    """Track F1 per arm with bootstrap and paired-to-AMVF errors."""
    c, lab, at = (_field(rec, k, 1)[mask] for k in ("n_correct", "n_labelled", "n_assigned_truth"))  # fmt: skip

    def f1(cc, ll, aa):  # noqa: ANN001, ANN202
        e, p = cc / ll, cc / aa
        return 2 * e * p / (e + p)

    reps = f1(w @ c, w @ lab, w @ at)
    return {"value": f1(c.sum(0), lab.sum(0), at.sum(0)),
            "err": reps.std(0, ddof=1), "diff_err": (reps - reps[:, -1:]).std(0, ddof=1)}  # fmt: skip


def metric_rows(rec, s: dict) -> dict:  # noqa: ANN001
    """{bin: {metric: {'value','amvf','diff','diff_err'} arrays over the t arms}}."""
    mu = rec.ev("mu")
    n_t = len(rec.meta["t_grid"])
    out = {}
    for lab, (lo, hi) in BINS.items():
        b = s["bins"][lab]
        mask = (mu >= lo) & (mu < hi)
        f1 = f1_table(rec, mask, boot_weights(int(mask.sum()), 200, 20261112))
        rows = {}
        for key, _, _, src in METRICS:
            if src == "ratio":
                r = b["ratios_min2"][key]
                v, de = np.array(r["value"], float), np.array(r["diff_err"], float)
                val, amvf, derr = v[:n_t], np.full(n_t, v[-1]), de[:n_t]
            elif src == "f1":
                val, amvf, derr = f1["value"][:n_t], np.full(n_t, f1["value"][-1]), f1["diff_err"][:n_t]  # fmt: skip
            elif src == "sel":
                sel = b["hs_selection"]
                v = np.array(sel["value"], float)
                val, amvf = v[:n_t], np.full(n_t, v[n_t])
                derr = np.array(sel["diff_stored_err"], float)[:n_t]
            else:
                hr = b["hs_resolution"]
                val = np.array([np.nan if x is None else x for x in hr["rms_um"]], float)[:n_t]  # fmt: skip
                amvf = np.array([np.nan if x is None else x for x in hr["amvf_um"]], float)[:n_t]  # fmt: skip
                derr = np.array([np.nan if x is None else x for x in hr["diff_err_um"]], float)[:n_t]  # fmt: skip
            rows[key] = {
                "value": val,
                "amvf": amvf,
                "diff": val - amvf,
                "diff_err": derr,
            }
        out[lab] = {"n_events": b["n_events"], "rows": rows}
    return out


def failures(rows: dict, j: int, bins: tuple[str, ...]) -> tuple[list, list]:
    """(strict, significant) failing (metric, bin) pairs at arm j."""
    strict, sig = [], []
    for lab in bins:
        for key, _, sense, _ in METRICS:
            r = rows[lab]["rows"][key]
            d, e = sense * r["diff"][j], r["diff_err"][j]
            if not np.isfinite(d):
                continue
            if d < 0:
                strict.append((key, lab))
                if d < -e:
                    sig.append((key, lab))
    return strict, sig


def interp_at(ts, x: np.ndarray, target: float):  # noqa: ANN001, ANN201
    """Fractional arm index where x first crosses target along t (None if never)."""
    for k in range(len(x) - 1):
        a, b = x[k] - target, x[k + 1] - target
        if a == 0:
            return float(k)
        if a * b < 0:
            return k + a / (a - b)
    return float(len(x) - 1) if x[-1] == target else None


def matched(rows: dict, ts: list[float], lab: str) -> list[dict]:
    """Matched operating points in one bin."""
    out = []
    for xk, target, ys in MATCHED:
        r = rows[lab]["rows"][xk]
        tgt = float(r["amvf"][0]) if target == "amvf" else float(target)
        fi = interp_at(ts, r["value"], tgt)
        rec = {"x": xk, "target": tgt, "anchor": str(target), "t": None, "y": {}}
        if fi is None and r["value"][0] > tgt:
            # already above the target at the lowest t: read out there
            fi, rec["anchor"] = 0.0, f"{target}, above it at the lowest t"
        if fi is not None:
            k, f = int(np.floor(fi)), fi - np.floor(fi)
            k2 = min(k + 1, len(ts) - 1)
            rec["t"] = ts[k] + f * (ts[k2] - ts[k])
            for yk in ys:
                v = rows[lab]["rows"][yk]["value"]
                rec["y"][yk] = {"value": float(v[k] + f * (v[k2] - v[k])),
                                "amvf": float(rows[lab]["rows"][yk]["amvf"][0])}  # fmt: skip
        out.append(rec)
    return out


def fmt(key: str, v: float) -> str:
    """Percent for ratios, raw for per-event counts and um."""
    if key.endswith("per_event") or key == "hs_res":
        return f"{v:.3f}" if key.endswith("per_event") else f"{v:.2f}"
    return f"{100 * v:.2f}"


def plots(all_rows: dict, all_t: dict, out: Path) -> list[str]:
    """Trade-off curves (PU200, inclusive, 0-10) and Clean/truth vs t."""
    import matplotlib.pyplot as plt

    from gnn.diagnostics.plot_style import save_figure, use_atlas_style

    use_atlas_style()
    made = []
    pairs = (("trk_pur", "trk_eff"), ("hs_track_frac", "hs_purity"), ("hs_track_frac", "hs_clean"),
             ("clean_per_reco", "clean_per_truth"))  # fmt: skip
    labels = {k: lab for k, lab, _, _ in METRICS}
    for lab in ("inclusive", PU, "0-10"):
        fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5))
        for ax, (xk, yk) in zip(axes, pairs):
            for name, rows in all_rows.items():
                x = 100 * rows[lab]["rows"][xk]["value"]
                y = 100 * rows[lab]["rows"][yk]["value"]
                ax.plot(
                    x,
                    y,
                    "-o",
                    ms=3,
                    lw=1.5,
                    color=COLORS.get(name, "#999999"),
                    label=name,
                )
            r0 = next(iter(all_rows.values()))[lab]["rows"]
            ax.plot(100 * r0[xk]["amvf"][0], 100 * r0[yk]["amvf"][0], "*", ms=14,
                    color=COLORS["amvf"], label="AMVF")  # fmt: skip
            ax.set_xlabel(f"{labels[xk]} [%]")
            ax.set_ylabel(f"{labels[yk]} [%]")
        axes[0].legend(fontsize=8, title=f"mu {lab}, points = t grid")
        tag = lab.split(" ")[0].replace("-", "_")
        made += [str(p) for p in save_figure(fig, out, f"tradeoff_{tag}")]
        plt.close(fig)
    return made


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", required=True, nargs="+", help="name=metrics.npz")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    all_rows, all_t, res, ev_sha = {}, {}, {}, set()
    md = ["# GNN v3: models vs AMVF on identical events (>= 2-track vertices)\n"]
    for spec in a.metrics:
        name, path = spec.split("=", 1)
        rec = load(path)
        ev_sha.add(rec.meta["events_sha256"])
        if rec.meta.get("hs_min_tracks") != 2:
            raise RuntimeError(f"{path}: expected --hs-min-tracks 2")
        s = summarize(rec, bins=BINS)
        rows = metric_rows(rec, s)
        ts = [float(t) for t in rec.meta["t_grid"]]
        all_rows[name], all_t[name] = rows, ts
        mu_bins = tuple(b for b in BINS if b not in ("inclusive",))
        per_t = []
        for j, t in enumerate(ts):
            st, sg = failures(rows, j, tuple(BINS))
            st_mu, _ = failures(rows, j, mu_bins)
            per_t.append({"t": t, "n_strict": len(st), "n_sig": len(sg), "strict": st, "sig": sg,
                          "n_strict_mu_bins": len(st_mu)})  # fmt: skip
        best = min(per_t, key=lambda r: (r["n_strict"], r["n_sig"]))
        dens = rec.meta["infer_metadata"]["density"]
        res[name] = {
            "metrics": path, "gnn": rec.meta["infer_metadata"]["gnn"][rec.meta["gnn"]],
            "peaks": dens.get("peaks"), "per_t": per_t, "best_t": best["t"],
            "zero_fail_t": [r["t"] for r in per_t if r["n_strict"] == 0],
            "zero_sig_fail_t": [r["t"] for r in per_t if r["n_sig"] == 0],
            "matched": {lab: matched(rows, ts, lab) for lab in ("inclusive", PU, "0-10")},
            "rows": rows,
        }  # fmt: skip
        n_tot = len(METRICS) * len(BINS)
        md.append(f"\n## {name}\n")
        md.append(f"Failures against AMVF out of {n_tot} (metric, bin) pairs; "
                  f"t with none: {res[name]['zero_fail_t'] or 'none'}; "
                  f"t with no significant failure: {res[name]['zero_sig_fail_t'] or 'none'}.\n")  # fmt: skip
        md.append(
            "| t | strict | > 1 sigma | failing (metric: bins) |\n|---:|---:|---:|---|"
        )
        for r in per_t:
            by: dict[str, list[str]] = {}
            for k, lab in r["strict"]:
                by.setdefault(k, []).append(lab.split(" ")[0])
            desc = "; ".join(
                f"{k}: {', '.join(v)}" if len(v) <= 4 else f"{k}: {len(v)} bins"
                for k, v in by.items()
            )
            md.append(f"| {r['t']:g} | {r['n_strict']} | {r['n_sig']} | {desc} |")
        j = ts.index(best["t"])
        md.append(
            f"\nAt t = {best['t']:g} (fewest failures), value / AMVF / paired diff ± err:\n"
        )
        heads = [m[1] for m in METRICS]
        md.append(
            "| bin | n | " + " | ".join(heads) + " |\n|---|---:|" + "---:|" * len(heads)
        )
        for lab in BINS:
            cells = []
            for key, _, sense, _ in METRICS:
                r = rows[lab]["rows"][key]
                mark = (
                    ""
                    if not np.isfinite(r["diff"][j]) or sense * r["diff"][j] >= 0
                    else " **x**"
                )
                cells.append(
                    f"{fmt(key, r['value'][j])} / {fmt(key, r['amvf'][j])}{mark}"
                )
            md.append(
                f"| {lab} | {rows[lab]['n_events']} | " + " | ".join(cells) + " |"
            )
    if len(ev_sha) != 1:
        raise RuntimeError("models were evaluated on different events")
    md.append("\n## Matched operating points (interpolated along t)\n")
    for lab in ("inclusive", PU, "0-10"):
        md.append(
            f"\n### {lab}\n\n| model | at | t | read-out (model / AMVF) |\n|---|---|---:|---|"
        )
        for name in res:
            for m in res[name]["matched"][lab]:
                at = f"{m['x']} = {100 * m['target']:.1f}% ({m['anchor']})"
                if m["t"] is None:
                    md.append(f"| {name} | {at} | not reached | |")
                    continue
                ys = ", ".join(f"{k} {100 * v['value']:.2f} / {100 * v['amvf']:.2f}" for k, v in m["y"].items())  # fmt: skip
                md.append(f"| {name} | {at} | {m['t']:.3f} | {ys} |")
    figs = plots(all_rows, all_t, a.out / "figures")
    (a.out / "summary.json").write_text(json.dumps(_jsonable({"models": res, "figures": figs}), indent=1) + "\n")  # fmt: skip
    (a.out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
