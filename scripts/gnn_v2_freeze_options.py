#!/usr/bin/env python3
"""Working-point options for the GNN score cut t, and v6 vs s8 after the GNN.

Options are derived on the GNN validation split (``val_v6_v2e527``) and
quoted on the r16443 test sample (entries 0-24,999, ``test_v6_v2e527``):

  A  argmax inclusive Clean/truth
  B  argmax inclusive HS selection (Athena rule, reconstructed and selected)
  C  argmax inclusive track F1
  D  mu-robust: argmax over t of min over the five mu bins of
     Clean/truth(t) / max_t' Clean/truth(t') in that bin
  E  no cut (t = 0): best HS resolution and best low-mu Clean/truth

v6 vs s8 differences are paired (same events, same bootstrap replicas).
Nothing is chosen here.

    PYTHONPATH=src venv/bin/python -u scripts/gnn_v2_freeze_options.py \
        --dir outputs/11_12_2026_output/gnn_v2_freeze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnn.evaluation.chain_t_summary import (  # noqa: E402
    MU_BINS,
    RATIOS,
    _field,
    boot_weights,
    load,
)
from pv_finder.evaluation.vertex_finding.itk_resolution import (
    iterative_rms,  # noqa: E402
)

COARSE = ("0-20", "20-50", "50-100", "100-150", "150-200")
SHOW = ("inclusive", "0-20", "20-50", "50-100", "100-150", "150-200", "PU200 (185-215)")
METRICS = ("clean_per_truth", "clean_per_reco", "merged_per_reco", "split_per_reco",
           "fake_per_reco", "pos_eff", "pos_fake_per_reco", "trk_f1", "hs_sel", "hs_res_um")  # fmt: skip


def values(s: dict, b: str) -> dict[str, np.ndarray]:
    """Per-arm metric arrays (t grid then AMVF) of one summary bin."""
    r = s["bins"][b]["ratios_min1"]
    g = lambda k: np.array(r[k]["value"], dtype=float)  # noqa: E731
    e, p = g("trk_eff"), g("trk_pur")
    hs = s["bins"][b]["hs_selection"]["value"]
    n = len(s["arms"]) - 1
    res = s["bins"][b]["hs_resolution"]["rms_um"]
    return {**{k: g(k) for k in METRICS[:7]}, "trk_f1": 2 * e * p / (e + p),
            "hs_sel": np.array(hs[: n + 1], float),  # t grid + AMVF stored primary
            "hs_res_um": np.array([np.nan if v is None else v for v in res], float)}  # fmt: skip


def options(val: dict) -> dict[str, float]:
    """The option t values, derived on the validation summary."""
    t = np.array([float(a[1:]) for a in val["arms"][:-1]])
    n = len(t)
    inc = values(val, "inclusive")
    rel = np.stack([values(val, b)["clean_per_truth"][:n] / values(val, b)["clean_per_truth"][:n].max() for b in COARSE])  # fmt: skip
    return {
        "A max Clean/truth": float(t[np.argmax(inc["clean_per_truth"][:n])]),
        "B max HS selection": float(t[np.argmax(inc["hs_sel"][:n])]),
        "C max track F1": float(t[np.argmax(inc["trk_f1"][:n])]),
        "D mu-robust Clean/truth": float(t[np.argmax(rel.min(0))]),
        "E no cut": 0.0,
    }


def per_event(rec, name: str) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN001
    num, den = RATIOS[name]
    return _field(rec, num, 0), _field(rec, den, 0)


def paired_v6_s8(v6, s8, t: float, mask: np.ndarray, w: np.ndarray) -> dict:  # noqa: ANN001
    """s8 minus v6 for each metric at one t, with paired bootstrap errors."""
    a = v6.meta["t_grid"].index(t)
    out = {}
    for name in ("clean_per_truth", "clean_per_reco", "merged_per_reco", "split_per_reco",
                 "fake_per_reco", "pos_eff", "pos_fake_per_reco", "trk_eff", "trk_pur"):  # fmt: skip
        reps, vals = [], []
        for rec in (v6, s8):
            x, y = per_event(rec, name)
            x, y = x[mask, a], y[mask, a]
            vals.append(x.sum() / y.sum())
            reps.append((w @ x) / (w @ y))
        out[name] = [float(vals[1] - vals[0]), float(np.std(reps[1] - reps[0], ddof=1))]
    sf = v6.meta["sel_fields"].index("reco_sel")
    hm = mask & (v6.ev("has_hs") > 0)
    wh = w[:, (v6.ev("has_hs") > 0)[mask]]
    x6, x8 = v6.sel[hm, a, sf].astype(float), s8.sel[hm, a, sf].astype(float)
    out["hs_sel"] = [float(x8.mean() - x6.mean()), float(np.std((wh @ x8 - wh @ x6) / wh.sum(1), ddof=1))]  # fmt: skip
    both = mask & np.isfinite(v6.res[:, a]) & np.isfinite(s8.res[:, a])
    r6, r8 = v6.res[both, a], s8.res[both, a]
    rng = np.random.default_rng(7)
    d = [1e3 * (iterative_rms(r8[i], 0.1).rms - iterative_rms(r6[i], 0.1).rms)
         for i in (rng.integers(0, len(r6), len(r6)) for _ in range(200))]  # fmt: skip
    out["hs_res_um"] = [float(1e3 * (iterative_rms(r8, 0.1).rms - iterative_rms(r6, 0.1).rms)), float(np.std(d, ddof=1))]  # fmt: skip
    return out


def fmt_row(label: str, v: dict, a: int) -> str:
    p = lambda k: f"{100 * v[k][a]:.2f}"  # noqa: E731
    return (f"| {label} | {p('clean_per_truth')} | {p('clean_per_reco')} | {p('merged_per_reco')} | "
            f"{p('split_per_reco')} | {p('fake_per_reco')} | {p('pos_eff')} | {p('pos_fake_per_reco')} | "
            f"{p('trk_f1')} | {p('hs_sel')} | {v['hs_res_um'][a]:.2f} |")  # fmt: skip


def plot_v6_s8(d: Path, ts: tuple[float, ...]) -> None:
    """Clean/truth, Split/reco, HS selection and HS resolution vs mu, v6 and s8."""
    import matplotlib.pyplot as plt

    from gnn.diagnostics.plot_style import save_figure

    s = {
        m: json.loads((d / f"summary/test_{m}_v2e527_mu10.json").read_text())
        for m in ("v6", "s8")
    }
    tg = [float(x[1:]) for x in s["v6"]["arms"][:-1]]
    x = np.arange(5, 200, 10)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    cols = ("#56B4E9", "#009E73", "#0072B2")
    keys = (("clean_per_truth", "Clean / truth [%]", 100), ("split_per_reco", "Split / reco [%]", 100),
            ("hs_sel", "HS reco & selected [%]", 100), ("hs_res_um", "HS resolution [um]", 1))  # fmt: skip
    for ax, (k, lab, sc) in zip(axes, keys):
        for m, ls in (("v6", "-"), ("s8", "--")):
            vals = [values(s[m], b) for b in s[m]["bins"]]
            for t, c in zip(ts, cols):
                ax.plot(x, [sc * v[k][tg.index(t)] for v in vals], ls=ls, color=c, marker="o", ms=3, label=f"{m} t={t:g}")  # fmt: skip
            if m == "v6":
                ax.plot(x, [sc * v[k][len(tg)] for v in vals], ls=":", color="#D55E00", marker="s", ms=3, label="AMVF")  # fmt: skip
        ax.set_xlabel("mu")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(
        "Density model after GNN v2 (r16443 test, 0-24,999): v6 solid, s8 dashed"
    )
    fig.tight_layout()
    save_figure(fig, d / "figures", "v6_vs_s8_after_gnn_vs_mu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    a = ap.parse_args()
    S = lambda n: json.loads((a.dir / f"summary/{n}.json").read_text())  # noqa: E731
    val, test, test8 = S("val_v6_v2e527"), S("test_v6_v2e527"), S("test_s8_v2e527")
    opts = options(val)
    tg = [float(x[1:]) for x in test["arms"][:-1]]
    hdr = ("| t | Clean/truth | Clean/reco | Merged/reco | Split/reco | Fake/reco | pos. eff | "
           "pos. Fake/reco | trk F1 | HS sel | HS res [um] |\n|---|" + "---:|" * 10)  # fmt: skip
    md = [
        "# GNN t options (derived on validation, quoted on test r16443 0-24,999, v6 density)\n"
    ]
    md.append("| option | t |\n|---|---:|")
    md += [f"| {k} | {v:g} |" for k, v in opts.items()]
    for set_name, s in (("validation (t-scan set)", val), ("test", test)):
        for b in SHOW:
            v = values(s, b)
            md.append(f"\n### {set_name}, mu {b} ({s['bins'][b]['n_events']} events)\n")
            md.append(hdr)
            for k, t in sorted(opts.items(), key=lambda x: x[1]):
                md.append(fmt_row(f"{k[0]}: {t:g}", v, tg.index(t)))
            md.append(fmt_row("AMVF", v, len(tg)))
    # v6 vs s8, paired, on test
    v6r, s8r = (
        load(a.dir / "metrics/test_v6_v2e527.npz"),
        load(a.dir / "metrics/test_s8_v2e527.npz"),
    )
    if not np.array_equal(v6r.ev("entry"), s8r.ev("entry")):
        raise AssertionError("v6 and s8 event sets differ")
    pair: dict = {}
    md.append("\n## s8 minus v6 after GNN v2, test, paired (points; HS res in um)\n")
    for b in ("inclusive", "0-20", "PU200 (185-215)"):
        lo, hi = MU_BINS[b]
        mask = (v6r.ev("mu") >= lo) & (v6r.ev("mu") < hi)
        w = boot_weights(int(mask.sum()), 200, 20261112)
        md.append(f"\n### mu {b}\n\n| t | dClean/truth | dClean/reco | dSplit/reco | dFake/reco | dpos.eff | dpos.Fake/reco | dtrk eff | dtrk pur | dHS sel | dHS res |\n|---|" + "---:|" * 10)  # fmt: skip
        for t in sorted({*opts.values(), 0.95, 0.98, 0.99}):
            d = paired_v6_s8(v6r, s8r, t, mask, w)
            pair[f"{b}|{t}"] = d
            q = lambda k, s=100: f"{s * d[k][0]:+.2f} ± {s * d[k][1]:.2f}"  # noqa: E731
            md.append(f"| {t:g} | {q('clean_per_truth')} | {q('clean_per_reco')} | {q('split_per_reco')} | {q('fake_per_reco')} | "
                      f"{q('pos_eff')} | {q('pos_fake_per_reco')} | {q('trk_eff')} | {q('trk_pur')} | {q('hs_sel')} | {q('hs_res_um', 1)} |")  # fmt: skip
    for b in SHOW:
        v8 = values(test8, b)
        md.append(f"\n### s8 absolute, test, mu {b}\n\n" + hdr)
        for t in sorted(set(opts.values())):
            md.append(fmt_row(f"{t:g}", v8, tg.index(t)))
    (a.dir / "options.md").write_text("\n".join(md) + "\n")
    (a.dir / "options.json").write_text(
        json.dumps({"options": opts, "s8_minus_v6": pair}, indent=1)
    )
    plot_v6_s8(
        a.dir,
        (
            opts["B max HS selection"],
            opts["D mu-robust Clean/truth"],
            opts["A max Clean/truth"],
        ),
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
