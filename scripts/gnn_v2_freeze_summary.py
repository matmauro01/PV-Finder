#!/usr/bin/env python3
"""Step 2b summary: tables and plots of the GNN t scan and full chain on r16443.

Reads ``metrics/<set>_<density>_<gnn>.npz`` (``gnn.evaluation.chain_t_metrics``),
writes ``summary/<name>.json`` (coarse mu bins, 200 paired bootstraps),
``summary/<name>_mu10.json`` (10-unit mu bins), ``tables.md`` and figures.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_summary.py \
        --dir outputs/11_12_2026_output/gnn_v2_freeze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn.diagnostics.plot_style import save_figure  # noqa: E402
from gnn.evaluation.chain_t_summary import MU_BINS, load, summarize  # noqa: E402

MU10 = {f"{lo}-{lo + 10}": (lo, lo + 10 if lo < 190 else 200.5) for lo in range(0, 200, 10)}  # fmt: skip
SHOW_T = (0.0, 0.9, 0.95, 0.98, 0.99)
COLS = ("#999999", "#56B4E9", "#009E73", "#0072B2", "#CC79A7")


def f1(r: dict) -> np.ndarray:
    """Track F1 from the efficiency and purity of one ratio table."""
    e = np.array(r["trk_eff"]["value"], dtype=float)
    p = np.array(r["trk_pur"]["value"], dtype=float)
    return 2 * e * p / (e + p)


def table(s: dict, title: str) -> str:
    """Markdown: one block per mu bin, one row per arm."""
    out = [f"## {title}\n"]
    arms = s["arms"]
    for label, b in s["bins"].items():
        r = b["ratios_min1"]
        out.append(
            f"### mu {label}: {b['n_events']} events, mean mu {b['mean_mu']:.1f}\n"
        )
        out.append("| arm | Clean/truth | Clean/reco | Merged/reco | Split/reco | Fake/reco | "
                   "pos. eff | pos. Fake/reco | trk eff | trk pur | F1 | HS sel | HS res [um] (AMVF) |")  # fmt: skip
        out.append("|---|" + "---:|" * 12)
        ff = f1(r)
        for a, arm in enumerate(arms):
            pct = lambda k: f"{100 * r[k]['value'][a]:.2f} ± {100 * r[k]['err'][a]:.2f}"  # noqa: E731
            sel = res = ""
            if "hs_selection" in b:
                si = a if arm != "amvf" else len(arms) - 1
                hsel = b["hs_selection"]
                sel = f"{100 * hsel['value'][si]:.2f} ± {100 * hsel['err'][si]:.2f}"
                hr = b["hs_resolution"]
                if hr["rms_um"][a] is not None:
                    res = f"{hr['rms_um'][a]:.2f} ± {hr['err_um'][a]:.2f} ({hr['amvf_um'][a]:.2f})"
            out.append(
                f"| {arm} | {pct('clean_per_truth')} | {pct('clean_per_reco')} | "
                f"{pct('merged_per_reco')} | {pct('split_per_reco')} | {pct('fake_per_reco')} | "
                f"{pct('pos_eff')} | {pct('pos_fake_per_reco')} | {pct('trk_eff')} | "
                f"{pct('trk_pur')} | {100 * ff[a]:.2f} | {sel} | {res} |"
            )
        if "hs_selection" in b:
            h = b["hs_selection"]
            out.append(f"\nAMVF HS selection, identical rule (our code): "
                       f"{100 * h['value'][-1]:.2f} ± {100 * h['err'][-1]:.2f} %\n")  # fmt: skip
        out.append("")
    return "\n".join(out)


def _arm_series(
    s: dict, key: str, arm: int, sel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    vals, errs = [], []
    for b in s["bins"].values():
        if sel:
            v, e = b["hs_selection"]["value"][arm], b["hs_selection"]["err"][arm]
        elif key == "hs_res":
            v, e = b["hs_resolution"]["rms_um"][arm], b["hs_resolution"]["err_um"][arm]
        else:
            v, e = (
                b["ratios_min1"][key]["value"][arm],
                b["ratios_min1"][key]["err"][arm],
            )
        vals.append(np.nan if v is None else v)
        errs.append(np.nan if e is None else e)
    return np.array(vals, float), np.array(errs, float)


PANELS = (
    ("clean_per_truth", "Clean / truth", 100), ("clean_per_reco", "Clean / reco", 100),
    ("merged_per_reco", "Merged / reco", 100), ("split_per_reco", "Split / reco", 100),
    ("fake_per_reco", "Fake / reco", 100), ("pos_fake_per_reco", "positional Fake / reco", 100),
    ("trk_f1", "track F1", 100), ("hs_sel", "HS reco & selected", 100),
    ("hs_res", "HS resolution [um]", 1),
)  # fmt: skip


def plot_vs_mu(s10: dict, name: str, out: Path) -> None:
    """Metrics vs mu (10-unit bins) for a few t and AMVF."""
    import matplotlib.pyplot as plt

    t_grid = [float(a[1:]) for a in s10["arms"][:-1]]
    x = np.arange(5, 200, 10)
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, (key, lab, scale) in zip(axes.flat, PANELS):
        arms = [(t_grid.index(t), f"t = {t:g}", c) for t, c in zip(SHOW_T, COLS)]
        arms.append((len(t_grid), "AMVF", "#D55E00"))
        for a, lbl, c in arms:
            if key == "trk_f1":
                v = np.array([f1(b["ratios_min1"])[a] for b in s10["bins"].values()])
                e = np.zeros_like(v)
            elif key == "hs_sel":
                v, e = _arm_series(s10, key, a, sel=True)
            else:
                v, e = _arm_series(s10, key, a)
            ls = "--" if lbl == "AMVF" else "-"
            ax.errorbar(x, scale * v, yerr=scale * e, color=c, ls=ls, lw=1.5, marker="o", ms=3, label=lbl)  # fmt: skip
        ax.set_xlabel("mu")
        ax.set_ylabel(lab + (" [%]" if scale == 100 else ""))
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{name}: r16443, 10-unit mu bins, reco >= 1 track (positional window 0.17 mm)")  # fmt: skip
    fig.tight_layout()
    save_figure(fig, out, f"vs_mu_{name}")


def plot_t_scan(s: dict, name: str, out: Path) -> None:
    """Metrics vs t, one line per coarse mu bin; AMVF as a dashed level."""
    import matplotlib.pyplot as plt

    t_grid = np.array([float(a[1:]) for a in s["arms"][:-1]])
    bins = list(s["bins"])
    cols = ("#E69F00", "#56B4E9", "#009E73", "#0072B2", "#CC79A7", "#000000", "#D55E00")
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, (key, lab, scale) in zip(axes.flat, PANELS):
        for bl, c in zip(bins, cols):
            b = {"bins": {bl: s["bins"][bl]}, "arms": s["arms"]}
            n = len(t_grid)
            if key == "trk_f1":
                v = f1(s["bins"][bl]["ratios_min1"])
                vals, amvf = v[:n], v[n]
            elif key == "hs_sel":
                h = s["bins"][bl]["hs_selection"]["value"]
                vals, amvf = np.array(h[:n], float), h[n]
            elif key == "hs_res":
                h = s["bins"][bl]["hs_resolution"]
                vals = np.array([np.nan if v is None else v for v in h["rms_um"][:n]])
                amvf = np.nanmean(
                    [np.nan if v is None else v for v in h["amvf_um"][:n]]
                )
            else:
                v, _ = _arm_series(b, key, 0)
                allv = np.array(s["bins"][bl]["ratios_min1"][key]["value"], float)
                vals, amvf = allv[:n], allv[n]
            ax.plot(t_grid, scale * vals, color=c, marker="o", ms=3, lw=1.3, label=f"mu {bl}")  # fmt: skip
            ax.axhline(scale * amvf, color=c, ls=":", lw=1)
        ax.set_xlabel("GNN score cut t")
        ax.set_ylabel(lab + (" [%]" if scale == 100 else ""))
        ax.set_xlim(0.45, 1.0)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"{name}: t scan (points; t = 0 not shown), AMVF dotted, reco >= 1 track")  # fmt: skip
    fig.tight_layout()
    save_figure(fig, out, f"t_scan_{name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    sdir, fdir = a.dir / "summary", a.dir / "figures"
    sdir.mkdir(exist_ok=True)
    md = ["# GNN v2 freeze: t scan and full chain (generated by scripts/gnn_v2_freeze_summary.py)\n",
          "Ratios of sums; errors: 200 event bootstraps. Track-purity categories unless 'pos.'"
          " (positional, greedy closest-first, 0.17 mm). Vertices with >= 1 assigned track.\n"]  # fmt: skip
    for path in sorted((a.dir / "metrics").glob("*.npz")):
        name = path.stem
        if a.only and name not in a.only:
            continue
        rec = load(path)  # asserts bookkeeping
        s = summarize(rec)
        s["source"] = {"metrics": str(path), "git": rec.meta["git"],
                       "infer": rec.meta["infer_metadata"]["gnn"],
                       "density": rec.meta["infer_metadata"]["density"]["checkpoint"]}  # fmt: skip
        (sdir / f"{name}.json").write_text(json.dumps(s, indent=1))
        s10 = summarize(rec, n_boot=100, bins=MU10)
        (sdir / f"{name}_mu10.json").write_text(json.dumps(s10, indent=1))
        md.append(table(s, name))
        if rec.meta["hs_computed"]:
            plot_vs_mu(s10, name, fdir)
            plot_t_scan(s, name, fdir)
        print(f"{name}: done", flush=True)
    (a.dir / "tables.md").write_text("\n".join(md) + "\n")
    print("bins:", list(MU_BINS))


if __name__ == "__main__":
    main()
