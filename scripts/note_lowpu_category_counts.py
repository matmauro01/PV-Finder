"""Per-event positional vertex categories on Run 2 MC in a pile-up window (note Fig. 6.1b).

Reads the frozen low pile-up vertex-finding evaluation (e400, integral 0.50,
0.17 mm window) and plots the mean number of reconstructed vertices per event
in each positional category (Total, Clean, Merged, Split, Fake) for PV-Finder
and AMVF on the same events, with the standard error of the mean. No
inference: only the per-event counts stored in ``eval_results.pkl`` are used.

Usage (repo root, venv):
    python scripts/note_lowpu_category_counts.py \
        --pkl outputs/11_12_2026_output/lowpu_final/vf_mc/eval_results.pkl \
        --mu-min 55 --mu-max 65 --out outputs/11_13_2026_output/note_edits/lowpu_categories
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np  # noqa: E402

CATS = ("total", "clean", "merged", "split", "fake")
LABELS = ("Total", "Clean", "Merged", "Split", "Fake")


def counts(per_event: list[dict], prefix: str) -> dict[str, np.ndarray]:
    """Per-event counts for one algorithm; ``prefix`` is "" (PV-Finder) or "amvf_"."""
    out = {
        c: np.array([e[f"{prefix}{c}"] for e in per_event], dtype=float)
        for c in CATS[1:]
    }
    out["total"] = sum(out[c] for c in CATS[1:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--mu-min", type=float, default=55.0)
    ap.add_argument("--mu-max", type=float, default=65.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.pkl, "rb") as fh:
        d = pickle.load(fh)
    sel = [e for e in d["per_event"] if args.mu_min <= e["mu"] <= args.mu_max]
    if not sel:
        raise ValueError("no events in the pile-up window")
    pvf, amvf = counts(sel, ""), counts(sel, "amvf_")
    # Consistency: PV-Finder total equals its stored number of reconstructed vertices.
    if not np.array_equal(
        pvf["total"], np.array([e["n_pred"] for e in sel], dtype=float)
    ):
        raise AssertionError("PV-Finder categories do not add up to n_pred")
    if not np.array_equal(
        amvf["total"], np.array([e["n_amvf"] for e in sel], dtype=float)
    ):
        raise AssertionError("AMVF categories do not add up to n_amvf")

    n = len(sel)
    summ = {
        alg: {
            c: (float(v[c].mean()), float(v[c].std(ddof=1) / np.sqrt(n))) for c in CATS
        }
        for alg, v in (("pvf", pvf), ("amvf", amvf))
    }
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(CATS))
    w = 0.38
    for off, alg, lab, col in ((-w / 2, "pvf", "PV-Finder", "#0072B2"),
                               (w / 2, "amvf", "AMVF", "#D55E00")):  # fmt: skip
        m = np.array([summ[alg][c][0] for c in CATS])
        e = np.array([summ[alg][c][1] for c in CATS])
        ax.bar(x + off, m, w, yerr=e, color=col, label=lab, capsize=3)
        for xi, mi in zip(x + off, m):
            ax.text(xi, mi + 0.6, f"{mi:.1f}", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Reconstructed vertices / event")
    ax.set_ylim(0, 1.75 * max(summ["pvf"]["total"][0], summ["amvf"]["total"][0]))
    ax.minorticks_off()
    hep.atlas.label(ax=ax, label="Internal", data=False, loc=0, rlabel="")
    ax.text(
        0.03,
        0.83,
        "Run 2 MC, $t\\bar{t}$, $\\sqrt{s}$ = 13 TeV",
        transform=ax.transAxes,
    )
    ax.text(0.03, 0.77, f"${args.mu_min:g} \\leq \\mu \\leq {args.mu_max:g}$, {n:,} events",
            transform=ax.transAxes)  # fmt: skip
    ax.text(
        0.03, 0.71, "0.17 mm window, PV-Finder integral 0.50", transform=ax.transAxes
    )
    ax.legend(loc="upper right")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / "mc_category_counts_e400"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    (args.out / "mc_category_counts_e400.json").write_text(
        json.dumps({"pkl": str(args.pkl), "n_events": n, "mu": [args.mu_min, args.mu_max],
                    "mean_sem": summ}, indent=2))  # fmt: skip
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
