"""Pile-up distributions of the three low pile-up evaluation samples.

Reads the per-event ``mu`` stored in the low pile-up evaluation pickles
(``run_eval_pvf.py`` / ``run_eval_pvf_run3.py``) and draws one ATLAS-style
figure with Run 2 MC (test events), Run 2 data and Run 3 data, plus a JSON
with the event counts and mean / min / max mu. Read-only: no inference.

Usage (repo root, venv active):
    PYTHONPATH=src python -u scripts/lowpu_mu_distributions.py \
        --base outputs/11_12_2026_output/lowpu_final \
        --out outputs/11_12_2026_output/paper_deck/lowpu_mu
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

from gnn.diagnostics.plot_style import save_figure  # noqa: E402

SAMPLES = (
    ("mc", "Run 2 MC (test)", "#0072B2"),
    ("run2", "Run 2 data", "#D55E00"),
    ("run3", "Run 3 data", "#009E73"),
)


def read_mu(pkl: Path) -> np.ndarray:
    """Per-event mu of one evaluation pickle."""
    with open(pkl, "rb") as f:
        res = pickle.load(f)
    mu = np.asarray([ev["mu"] for ev in res["per_event"]], dtype=float)
    if mu.size == 0 or not np.all(np.isfinite(mu)):
        raise ValueError(f"{pkl}: missing or non-finite per-event mu")
    return mu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help="lowpu_final output directory")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    edges = np.arange(0.0, 82.0, 2.0)
    for key, label, color in SAMPLES:
        mu = read_mu(base / f"vf_{key}" / "eval_results.pkl")
        summary[key] = {
            "label": label,
            "events": int(mu.size),
            "mu_mean": float(mu.mean()),
            "mu_min": float(mu.min()),
            "mu_max": float(mu.max()),
        }
        counts, _ = np.histogram(mu, bins=edges)
        ax.stairs(
            counts / mu.size,
            edges,
            color=color,
            lw=2,
            label=f"{label}: {mu.size:,} events, "
            + r"$\langle\mu\rangle$"
            + f" = {mu.mean():.1f}",
        )
    ax.set_xlabel(r"Number of interactions $\mu$")
    ax.set_ylabel("Fraction of events / 2")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    hep.atlas.text("Internal", loc=1, ax=ax)
    ax.legend(loc="upper right", fontsize=13)
    save_figure(fig, out, "lowpu_mu_distributions")
    (out / "lowpu_mu_distributions.json").write_text(json.dumps(summary, indent=1))
    for key, s in summary.items():
        print(
            f"{key}: {s['events']} events, mean mu {s['mu_mean']:.2f}, "
            f"range {s['mu_min']:.1f}-{s['mu_max']:.1f}"
        )


if __name__ == "__main__":
    main()
