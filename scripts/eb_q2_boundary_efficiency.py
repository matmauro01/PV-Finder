"""EB question 2: vertex-finding efficiency vs distance to the nearest 40 mm interval boundary.

Reads the frozen-working-point pre-GNN cache (PU200, s8 at 0.184/0.0184, AMVF nTrk >= 2,
truth nTrk >= 2) and computes, in bins of the distance of each truth vertex to the nearest
internal interval edge (-240 + 40 k mm, k = 1..11):
  Eq. (1) efficiency: truth with >= 1 reconstructed vertex within 0.17 mm,
and, for reconstructed vertices, the fake fraction (no truth n_trk >= 2 within 0.17 mm)
in bins of the reconstructed vertex's own distance to the nearest edge.
AMVF has no intervals and is the control. Errors: binomial.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EDGES_MM = -240.0 + 40.0 * np.arange(1, 12)
DIST_BINS = np.array([0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])


def dist_to_edge(z: np.ndarray) -> np.ndarray:
    """Absolute distance (mm) of each z to the nearest internal interval edge."""
    return np.min(np.abs(z[:, None] - EDGES_MM[None, :]), axis=1)


def has_neighbour(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """For each element of a, whether some element of b lies within w (both 1-D)."""
    if a.size == 0:
        return np.zeros(0, bool)
    if b.size == 0:
        return np.zeros(a.size, bool)
    bs = np.sort(b)
    i = np.searchsorted(bs, a)
    lo = np.abs(a - bs[np.clip(i - 1, 0, bs.size - 1)])
    hi = np.abs(a - bs[np.clip(i, 0, bs.size - 1)])
    return np.minimum(lo, hi) <= w


def per_vertex(
    cache: np.lib.npyio.NpzFile, reco: str, w: float
) -> dict[str, np.ndarray]:
    """Per-truth found flags and per-reco fake flags with their edge distances."""
    tz, toff = cache["truth_ge2_z"], cache["truth_ge2_off"]
    rz, roff = cache[f"{reco}_z"], cache[f"{reco}_off"]
    found, fake = [], []
    for e in range(len(toff) - 1):
        t = tz[toff[e] : toff[e + 1]].astype(np.float64)
        r = rz[roff[e] : roff[e + 1]].astype(np.float64)
        found.append(has_neighbour(t, r, w))
        fake.append(~has_neighbour(r, t, w))
    return {
        "t_dist": dist_to_edge(tz.astype(np.float64)),
        "found": np.concatenate(found),
        "r_dist": dist_to_edge(rz.astype(np.float64)),
        "fake": np.concatenate(fake),
    }


def binned(dist: np.ndarray, flag: np.ndarray) -> list[dict[str, float]]:
    """Fraction of flag in DIST_BINS of dist, with binomial errors."""
    out = []
    for lo, hi in zip(DIST_BINS[:-1], DIST_BINS[1:]):
        m = (dist >= lo) & (dist < hi)
        n, k = int(m.sum()), int(flag[m].sum())
        p = k / n if n else float("nan")
        err = float(np.sqrt(p * (1 - p) / n)) if n else float("nan")
        out.append(
            {"lo": float(lo), "hi": float(hi), "n": n, "k": k, "frac": p, "err": err}
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache", nargs="+", required=True, help="fair_<tag>/cache_<tag>.npz"
    )
    ap.add_argument("--window", type=float, default=0.17)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res: dict[str, dict] = {}
    for path in a.cache:
        c = np.load(path, allow_pickle=True)
        tag = path.split("cache_")[-1].replace(".npz", "")
        res[tag] = {}
        for reco in ("pvf", "amvf"):
            pv = per_vertex(c, reco, a.window)
            res[tag][reco] = {
                "eff_total": float(pv["found"].mean()),
                "fake_total": float(pv["fake"].mean()),
                "eff_vs_dist": binned(pv["t_dist"], pv["found"]),
                "fake_vs_dist": binned(pv["r_dist"], pv["fake"]),
            }
    with open(f"{a.out}/boundary_efficiency.json", "w") as f:
        json.dump(res, f, indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, d in res.items():
        for reco, mk, lab in (("pvf", "o", "PV-Finder s8"), ("amvf", "s", "AMVF")):
            for ax, key in zip(axes, ("eff_vs_dist", "fake_vs_dist")):
                b = d[reco][key]
                x = [np.sqrt(max(r["lo"], 0.02) * r["hi"]) for r in b]
                ax.errorbar(
                    x,
                    [100 * r["frac"] for r in b],
                    yerr=[100 * r["err"] for r in b],
                    marker=mk,
                    ls="-",
                    ms=4,
                    capsize=2,
                    label=f"{lab} ({tag})",
                )
    axes[0].set_ylabel("Eq. (1) efficiency at 0.17 mm [%]")
    axes[0].set_xlabel("truth vertex distance to nearest interval edge [mm]")
    axes[1].set_ylabel("fake fraction of reconstructed [%]")
    axes[1].set_xlabel("reco vertex distance to nearest interval edge [mm]")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "HL-LHC PU200 (185 <= mu <= 215), frozen s8 cuts; edges at -240 + 40k mm"
    )
    fig.tight_layout()
    fig.savefig(f"{a.out}/boundary_efficiency.png", dpi=130)
    fig.savefig(f"{a.out}/boundary_efficiency.pdf")
    for tag, d in res.items():
        for reco in ("pvf", "amvf"):
            print(
                tag,
                reco,
                f"eff {100 * d[reco]['eff_total']:.2f} fake {100 * d[reco]['fake_total']:.2f}",
            )
            for r in d[reco]["eff_vs_dist"]:
                print(
                    f"   eff [{r['lo']:.1f},{r['hi']:.1f}) n={r['n']:7d} {100 * r['frac']:.2f} +- {100 * r['err']:.2f}"
                )
            for r in d[reco]["fake_vs_dist"]:
                print(
                    f"   fake[{r['lo']:.1f},{r['hi']:.1f}) n={r['n']:7d} {100 * r['frac']:.2f} +- {100 * r['err']:.2f}"
                )


if __name__ == "__main__":
    main()
