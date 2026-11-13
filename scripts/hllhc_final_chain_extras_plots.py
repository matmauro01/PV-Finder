#!/usr/bin/env python3
"""ATLAS-style figures of the chain extras (H3, H7) from ``chain_extras_summary.json``.

One standalone figure per quantity (PNG + PDF), PV-Finder(+GNN) and AMVF on
the same events, "Simulation Internal". See the README of the output folder
for the list and captions.

    PYTHONPATH=src python -u scripts/hllhc_final_chain_extras_plots.py \
        --summary outputs/11_12_2026_output/hllhc_final/chain_extras/chain_extras_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from gnn.diagnostics.plot_style import (  # noqa: E402
    atlas_label,
    save_figure,
    use_atlas_style,
)

PU200 = "PU200 (185-215)"
SAMPLE = r"$\sqrt{s}=14$ TeV, $t\bar{t}$, ITk, $\langle\mu\rangle$ flat 0-200"
SEL = {PU200: r"$185 \leq \mu \leq 215$ (1 920 events)", "inclusive": r"$0 \leq \mu \leq 200$ (25 000 events)"}  # fmt: skip
TAG = {PU200: "pu200", "inclusive": "inclusive"}
PVF, AMVF = "#0072B2", "#D55E00"
# arm: (label, colour, marker, filled)
STYLE: dict[str, tuple[str, str, str, bool]] = {
    "gnn_t": ("PV-Finder + GNN ($t=0.97$)", PVF, "o", True),
    "gnn_t0": ("PV-Finder + GNN (no score cut)", "#CC79A7", "v", True),
    "z0_nearest": (r"PV-Finder + nearest $z_0$", "#009E73", "^", True),
    "z0_3sigma": (
        r"PV-Finder + nearest $z_0$ ($<3\sigma(z_0)$)",
        "#E69F00",
        "<",
        False,
    ),  # fmt: skip
    "eq4": (r"PV-Finder + significance, Eq. (4)", "#56B4E9", "D", True),
    "truth_guided": ("PV-Finder + truth-guided (MC only)", "0.35", "P", True),
    "amvf": ("AMVF", AMVF, "s", True),
    "pvf": ("PV-Finder", PVF, "o", True),
    "finder_pvf": ("PV-Finder candidates, one-to-one", "0.55", "X", True),
    "finder_amvf": ("AMVF vertices, one-to-one", AMVF, "X", False),
}


def _label(ax: plt.Axes, sample: str, extra: str = "") -> None:
    atlas_label(ax, desc=f"{SAMPLE}\n{SEL[sample]}" + (f"\n{extra}" if extra else ""), desc_xy=(0.05, 0.88))  # fmt: skip


def _pt(
    ax: plt.Axes, x, y, e, arm: str, label: str | None = None, xerr=None, **kw
) -> None:  # noqa: ANN001
    lab, col, mk, fill = STYLE[arm]
    ax.errorbar(x, y, yerr=e, xerr=xerr, fmt=mk, color=col, mfc=col if fill else "none", label=label or lab, ms=6, capsize=0, lw=1.2, **kw)  # fmt: skip


class Plots:
    """Figure factory bound to one summary."""

    def __init__(self, s: dict, out: Path) -> None:
        self.s, self.out = s, out
        self.arms = s["meta"]["replay_meta"]["arms"]
        self.eq4 = s["eq4_best"]["arm"]
        if (
            s["meta"]["replay_meta"]["t"] != 0.97
        ):  # the legend strings below say t = 0.97
            raise ValueError("summary is not at t = 0.97; update the labels")
        STYLE["eq4_s3"] = (
            STYLE["eq4"][0] + ", $s_{max}=3$",
            STYLE["eq4"][1],
            "d",
            False,
        )
        STYLE[self.eq4] = (STYLE["eq4"][0] + f", $s_{{max}}={self.eq4[5:]}$", *STYLE["eq4"][1:])  # fmt: skip
        self.mu = sorted((k for k in s["bins"] if k.startswith("mu ")), key=lambda k: float(k[3:].split("-")[0]))  # fmt: skip
        lo = np.array([float(k[3:].split("-")[0]) for k in self.mu])
        self.mu_x, self.mu_xe = lo + 5, np.full(len(lo), 5.0)
        self.made: list[str] = []

    def save(self, fig: plt.Figure, name: str) -> None:
        save_figure(fig, self.out, name)
        self.made.append(name)

    def arm(self, b: str, ratio: str, arm: str, m: str = "min1") -> tuple[float, float]:
        t = self.s["bins"][b]["arms"][m][ratio]
        k = self.arms.index(arm)
        return t["value"][k], t["err"][k]

    def arm_mu(
        self, ratio: str, arm: str, m: str = "min1"
    ) -> tuple[np.ndarray, np.ndarray]:
        v = np.array([self.arm(b, ratio, arm, m) for b in self.mu], dtype=float)
        return v[:, 0], v[:, 1]

    def ev_mu(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        v = np.array([[self.s["bins"][b]["event"][key][x] for x in ("value", "err")] for b in self.mu], float)  # fmt: skip
        return v[:, 0], v[:, 1]

    # ---------------------------------------------------------------- vs mu
    def vs_mu(
        self,
        series: list[tuple[str, np.ndarray, np.ndarray, str | None]],
        ylabel: str,
        name: str,
        ylim=None,
    ) -> None:  # noqa: ANN001
        fig, ax = plt.subplots(figsize=(8, 6))
        for arm, v, e, lab in series:
            _pt(ax, self.mu_x, v, e, arm, lab, xerr=self.mu_xe)
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 200)
        lo, hi = np.nanmin([np.nanmin(v) for _, v, _, _ in series]), np.nanmax([np.nanmax(v) for _, v, _, _ in series])  # fmt: skip
        ax.set_ylim(*(ylim or (lo - 0.15 * (hi - lo + 1e-3), hi + 0.6 * (hi - lo + 1e-3))))  # fmt: skip
        atlas_label(ax, desc=f"{SAMPLE}\n25 000 events, 0.17 mm window", desc_xy=(0.05, 0.88))  # fmt: skip
        ax.legend(loc="upper right", fontsize="small", frameon=False)
        self.save(fig, name)

    def all_vs_mu(self) -> None:
        a = lambda r, x: (x, *self.arm_mu(r, x), None)  # noqa: E731
        self.vs_mu([("pvf", *self.ev_mu("eff_pvf"), None), ("amvf", *self.ev_mu("eff_amvf"), None)], "Vertex-finding efficiency, Eq. (1)", "eff_pregnn_vs_mu")  # fmt: skip
        self.vs_mu([a("clean_per_truth", "gnn_t"), a("clean_per_truth", "amvf")], "Clean vertices / truth vertices", "clean_per_truth_postgnn_vs_mu")  # fmt: skip

        def a2(r: str, x: str) -> tuple:
            return (x, *self.arm_mu(r, x, "min2"), STYLE[x][0] + r", $\geq 2$ tracks")

        self.vs_mu([a2("clean_per_truth", "gnn_t"), a2("clean_per_truth", "amvf")], r"Clean vertices ($\geq 2$ tracks) / truth vertices", "clean_per_truth_min2_postgnn_vs_mu")  # fmt: skip
        self.vs_mu([a("clean_ge2_per_truth", "gnn_t"), a("clean_ge2_per_truth", "amvf")], r"Clean vertices, dominant $n_{trk} \geq 2$ / truth vertices", "clean_ge2_per_truth_postgnn_vs_mu")  # fmt: skip
        self.vs_mu([("finder_pvf", *self.ev_mu("finder_cap_pvf"), None), ("finder_amvf", *self.ev_mu("finder_cap_amvf"), None)], "One-to-one efficiency (finder cap)", "finder_cap_vs_mu")  # fmt: skip
        self.vs_mu([a("clean_per_truth", "truth_guided"), a("clean_per_truth", "gnn_t"), a("clean_per_truth", "amvf")], "Clean vertices / truth vertices", "truth_guided_clean_vs_mu")  # fmt: skip
        self.vs_mu([a("clean_per_truth", x) for x in dict.fromkeys([self.eq4, "eq4_s3", "gnn_t", "amvf"])], "Clean vertices / truth vertices", "eq4_clean_vs_mu")  # fmt: skip
        self.vs_mu([a("clean_per_truth", "z0_nearest"), a("clean_per_truth", "z0_3sigma"), a("clean_per_truth", "gnn_t"), a("clean_per_truth", "amvf")], "Clean vertices / truth vertices", "z0_baselines_clean_vs_mu")  # fmt: skip
        stages = [("finder_pvf", *self.ev_mu("finder_cap_pvf"), None)]
        self.vs_mu([*stages, a("clean_per_truth", "truth_guided"), a("clean_per_truth", "gnn_t"), a("clean_per_truth", self.eq4), a("clean_per_truth", "z0_nearest"), a("clean_per_truth", "amvf")], "Fraction of truth vertices", "clean_yield_stages_vs_mu")  # fmt: skip
        trk = {"trk_eff": "Track-assignment efficiency", "trk_pur": "Track-assignment purity", "trk_f1": r"Track-assignment $F_1$", "wrong_rate": "Wrong-assignment rate", "labelled_unassigned_frac": "Unassigned fraction of truth-matched tracks", "unlabelled_left_unassigned": "Unmatched tracks left unassigned", "lost_unassigned_frac": "Unassigned, no vertex near truth / truth-matched"}  # fmt: skip
        for r, lab in trk.items():
            self.vs_mu([a(r, "gnn_t"), a(r, self.eq4), a(r, "z0_nearest"), a(r, "amvf")], lab, f"{r}_vs_mu")  # fmt: skip

    # ------------------------------------------------- per truth vertex bins
    def vs_var(self, sample: str, var: str, pairs: list[tuple[str, str, str | None]], ylabel: str, name: str, ratio: tuple[str, str] | None) -> None:  # fmt: skip
        t = self.s["per_truth"][sample][var]
        edges = np.array([np.inf if x is None else x for x in t["edges"]], float)
        nb = len(edges) - 1
        x = np.arange(nb)
        if ratio:
            fig, (ax, rx) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})  # fmt: skip
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
        for key, arm, lab in pairs:
            v = np.array(t[key]["value"], float)
            e = np.array(t[key]["err"], float)
            _pt(ax, x, v, e, arm, lab, xerr=0.5)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.45)
        _label(ax, sample, "0.17 mm window")
        ax.legend(loc="upper right", fontsize="small", frameon=False)
        if ratio:
            a, b = (np.array(t[k]["value"], float) for k in ratio)
            de = np.array(
                t.get(f"{ratio[0]}-minus-{ratio[1]}_err", [np.nan] * nb), float
            )
            # paired error of the difference, propagated to the ratio a/b
            rx.errorbar(x, a / b, yerr=de / b, xerr=0.5, fmt="o", color="k", ms=4)
            rx.axhline(1, color="0.5", lw=1, ls="--")
            rx.set_ylabel("PV-Finder / AMVF", fontsize="small")
            hi = np.nanmax(a / b)
            if hi > 2.5:  # wide range (low-n_trk AMVF Clean ~ 0): log axis
                rx.set_yscale("log")
                rx.set_ylim(0.8, 1.5 * hi)
            else:
                rx.set_ylim(min(0.9, np.nanmin(a / b) - 0.05), max(1.1, hi + 0.1))
            axl = rx
        else:
            axl = ax
        lab = [self._bin_label(edges[i], edges[i + 1], var) for i in range(nb)]
        axl.set_xticks(x)
        axl.set_xticklabels(lab, rotation=45 if nb > 7 else 0, fontsize="small")
        axl.set_xlabel({"n_trk": r"Truth vertex $n_{trk}$", "density": r"Local pile-up density [vertices/mm]", "dz_nearest": r"Distance to nearest truth vertex [mm]"}[var])  # fmt: skip
        self.save(fig, name)

    @staticmethod
    def _bin_label(lo: float, hi: float, var: str) -> str:
        if not np.isfinite(hi):
            return f">{lo - 1:g}" if var == "n_trk" else f">{lo:g}"
        if var == "n_trk":
            return f"{lo:g}" if hi - lo == 1 else f"{lo:g}-{hi - 1:g}"
        return f"{lo:g}-{hi:g}"

    def all_vs_var(self) -> None:
        for sample in (PU200, "inclusive"):
            tg = TAG[sample]
            for var in ("n_trk", "density", "dz_nearest"):
                self.vs_var(sample, var, [("found_pvf", "pvf", None), ("found_amvf", "amvf", None)], "Vertex-finding efficiency, Eq. (1)", f"eff_vs_{var}_pregnn_{tg}", ("found_pvf", "found_amvf"))  # fmt: skip
                self.vs_var(sample, var, [("clean_gnn_t", "gnn_t", None), ("clean_amvf", "amvf", None)], "Fraction with a Clean vertex", f"eff_vs_{var}_postgnn_{tg}", ("clean_gnn_t", "clean_amvf"))  # fmt: skip
                self.vs_var(sample, var, [("clean2_gnn_t", "gnn_t", None), ("clean2_amvf", "amvf", None)], r"Fraction with a Clean vertex ($\geq 2$ tracks)", f"eff_vs_{var}_postgnn_min2_{tg}", ("clean2_gnn_t", "clean2_amvf"))  # fmt: skip
                self.vs_var(sample, var, [("o2o_pvf", "finder_pvf", None), ("clean_truth_guided", "truth_guided", None), ("clean_gnn_t", "gnn_t", None), (f"clean_{self.eq4}", self.eq4, None), ("clean_amvf", "amvf", None)], "Fraction of truth vertices", f"stages_vs_{var}_{tg}", None)  # fmt: skip

    # ------------------------------------------------------------ PU200 only
    def stages(self) -> None:
        """Fig. 7.2: Clean yield stages at PU200, horizontal points with errors."""
        b = self.s["bins"][PU200]
        rows = [
            ("One-to-one finder match, PV-Finder", b["event"]["finder_cap_pvf"], "0.6"),
            (
                "Truth-guided assignment (MC only)",
                self._d(PU200, "truth_guided"),
                "0.35",
            ),
            ("PV-Finder + GNN ($t=0.97$)", self._d(PU200, "gnn_t"), PVF),
            (
                r"PV-Finder + GNN ($t=0.97$), $\geq 2$ tracks",
                self._d(PU200, "gnn_t", "min2"),
                "#7FB8DA",
            ),
            ("PV-Finder + GNN (no score cut)", self._d(PU200, "gnn_t0"), "#CC79A7"),
            (STYLE[self.eq4][0], self._d(PU200, self.eq4), "#56B4E9"),
            (r"PV-Finder + Eq. (4), $s_{max}=3$", self._d(PU200, "eq4_s3"), "#56B4E9"),
            (
                r"PV-Finder + nearest $z_0$ ($<3\sigma(z_0)$)",
                self._d(PU200, "z0_3sigma"),
                "#E69F00",
            ),  # fmt: skip
            (r"PV-Finder + nearest $z_0$", self._d(PU200, "z0_nearest"), "#009E73"),
            ("One-to-one finder match, AMVF", b["event"]["finder_cap_amvf"], "#E8A07A"),
            ("AMVF", self._d(PU200, "amvf"), AMVF),
        ]
        fig, ax = plt.subplots(figsize=(9, 6.5))
        y = np.arange(len(rows))[::-1]
        for yi, (lab, d, col) in zip(y, rows):
            ax.barh(yi, d["value"], xerr=d["err"], color=col, height=0.65)
            ax.text(d["value"] + 0.01, yi, f"{100 * d['value']:.1f}%", va="center", fontsize="small")  # fmt: skip
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize="small")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.7, len(rows) + 1.4)
        ax.set_xlabel("Fraction of truth vertices (Clean, or one-to-one matched)")
        atlas_label(
            ax, desc=f"{SAMPLE}\n{SEL[PU200]}\n0.17 mm window", desc_xy=(0.03, 0.975)
        )
        self.save(fig, "clean_yield_stages_pu200")

    def _d(self, b: str, arm: str, m: str = "min1") -> dict:
        v, e = self.arm(b, "clean_per_truth", arm, m)
        return {"value": v, "err": e}

    def scan(self, sample: str) -> None:
        """Eq. (4) s_max scan vs GNN t scan: track eff vs purity; Clean/truth vs purity."""
        t = self.s["tscan"][sample]
        tg = TAG[sample]
        eq4 = [x for x in self.arms if x.startswith("eq4_")]
        for yk, ylabel, name in (("trk_eff", "Track-assignment efficiency", "trk_eff_vs_pur_scan"), ("clean_per_truth", "Clean vertices / truth vertices", "clean_vs_pur_scan")):  # fmt: skip
            fig, ax = plt.subplots(figsize=(8, 6))
            gx, gy = np.array(t["trk_pur"]["value"][:-1]), np.array(t[yk]["value"][:-1])
            ax.plot(gx, gy, "-o", color=PVF, ms=4, label="PV-Finder + GNN, $t$ scan")
            t_cut = self.s["meta"]["replay_meta"]["t"]
            k = t["t"].index(t_cut)
            ax.plot(gx[k], gy[k], "*", color=PVF, ms=16, label=f"Frozen working point $t={t_cut:g}$")  # fmt: skip
            ex = [self.arm(sample, "trk_pur", x)[0] for x in eq4]
            ey = [self.arm(sample, yk, x)[0] for x in eq4]
            ax.plot(ex, ey, "-D", color="#56B4E9", ms=5, label=r"PV-Finder + Eq. (4), $s_{max}$ scan")  # fmt: skip
            for x, a, bb in zip(eq4, ex, ey):
                ax.annotate(x[5:], (a, bb), textcoords="offset points", xytext=(4, -10), fontsize="x-small", color="#2a7fb0")  # fmt: skip
            for arm in ("z0_nearest", "z0_3sigma", "amvf"):
                _pt(ax, self.arm(sample, "trk_pur", arm)[0], self.arm(sample, yk, arm)[0], None, arm)  # fmt: skip
            ax.set_xlabel("Track-assignment purity")
            ax.set_ylabel(ylabel)
            lo = min(min(ey), min(gy), self.arm(sample, yk, "amvf")[0])
            hi = max(max(ey), max(gy), self.arm(sample, yk, "amvf")[0])
            ax.set_ylim(lo - 0.05, hi + 0.5 * (hi - lo) + 0.05)
            _label(ax, sample)
            ax.legend(loc="upper right", fontsize="x-small", frameon=False)
            self.save(fig, f"{name}_{tg}")

    def sigma_pv(self) -> None:
        """Peak width (the chain graph's sigma_PV) vs the n_trk parameterisation."""
        for sample in (PU200, "inclusive"):
            rows = self.s["sigma_pv_check"][sample]["by_ntrk"]
            x = np.arange(len(rows))
            fig, ax = plt.subplots(figsize=(8, 6))
            med = np.array([r["peak_sigma_median_mm"] for r in rows])
            q = np.array([r["peak_sigma_q16_q84_mm"] for r in rows])
            ax.errorbar(x, med, yerr=[med - q[:, 0], q[:, 1] - med], fmt="o", color=PVF, label="Peak width (GNN edge feature), median, 16-84%")  # fmt: skip
            ax.plot(x, [r["param_sigma_median_mm"] for r in rows], "s--", color=AMVF, mfc="none", label=r"$\sigma(n_{trk})$, Appendix B (hllhc_alleta)")  # fmt: skip
            lab = [self._bin_label(*(np.inf if v is None else v for v in r["bin"]), "n_trk") for r in rows]  # fmt: skip
            ax.set_xticks(x)
            ax.set_xticklabels(lab, rotation=45, fontsize="small")
            ax.set_xlabel(r"Truth vertex $n_{trk}$ (one-to-one matched peak)")
            ax.set_ylabel(r"$\sigma_{PV}$ [mm]")
            ax.set_ylim(0, 0.3)
            _label(ax, sample)
            ax.legend(loc="upper right", fontsize="x-small", frameon=False)
            self.save(fig, f"sigma_pv_peakwidth_vs_param_{TAG[sample]}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    use_atlas_style()
    p = Plots(s, a.out or a.summary.parent / "plots")
    p.stages()
    p.all_vs_var()
    p.all_vs_mu()
    for sample in (PU200, "inclusive"):
        p.scan(sample)
    p.sigma_pv()
    print("\n".join(p.made))


if __name__ == "__main__":
    main()
