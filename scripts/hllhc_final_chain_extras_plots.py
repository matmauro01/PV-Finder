#!/usr/bin/env python3
"""ATLAS-style figures of the chain extras (H3, H7) from ``chain_extras_summary.json``.

One standalone figure per quantity (PNG + PDF), PV-Finder(+GNN) and AMVF on
the same events, "Simulation Internal". Post-GNN quantities use one vertex
rule for every arm: the PRIMARY figures count reconstructed vertices with
>= 2 associated tracks (tracks on dropped single-track vertices count as
unassigned); the same figures with >= 1 track (drop-empty) carry the suffix
``_min1`` and are backup. Rule-independent figures (pre-GNN efficiency, finder
cap, sigma_PV) have no suffix. With ``chain_extras_trackdiag.json`` next to the
summary, the track-level diagnosis figures (``trackdiag_*``) are drawn too.

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
# vertex rule -> (file suffix, description line, per-truth flag prefix)
RULES = {
    "min2": ("", r"vertices with $\geq 2$ tracks", "clean2_"),
    "min1": ("_min1", r"vertices with $\geq 1$ track", "clean_"),
}
PVF, AMVF, EQ4 = "#0072B2", "#D55E00", "#56B4E9"
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
    "truth_guided": ("PV-Finder + truth-guided (MC only)", "0.35", "P", True),
    "amvf": ("AMVF", AMVF, "s", True),
    "pvf": ("PV-Finder", PVF, "o", True),
    "finder_pvf": ("PV-Finder candidates, one-to-one", "0.55", "X", True),
    "finder_amvf": ("AMVF vertices, one-to-one", AMVF, "X", False),
}
MU_SERIES = (("mu 0-10", "#000000"), ("mu 50-60", "#009E73"), ("mu 100-110", "#E69F00"), ("mu 190-200", "#D55E00"))  # fmt: skip


def _pt(ax: plt.Axes, x, y, e, arm: str, label: str | None = None, xerr=None) -> None:  # noqa: ANN001
    lab, col, mk, fill = STYLE[arm]
    ax.errorbar(x, y, yerr=e, xerr=xerr, fmt=mk, color=col, mfc=col if fill else "none", label=label or lab, ms=6, capsize=0, lw=1.2)  # fmt: skip


def _desc(ax: plt.Axes, *lines: str) -> None:
    atlas_label(ax, desc="\n".join([SAMPLE, *lines]), desc_xy=(0.05, 0.88))


def _ylim(ax: plt.Axes, vals: list[np.ndarray]) -> None:
    lo = np.nanmin([np.nanmin(v) for v in vals])
    hi = np.nanmax([np.nanmax(v) for v in vals])
    ax.set_ylim(lo - 0.15 * (hi - lo + 1e-3), hi + 0.65 * (hi - lo + 1e-3))


def _ntrk_label(lo: float, hi: float) -> str:
    if not np.isfinite(hi):
        return f">{lo - 1:g}"
    return f"{lo:g}" if hi - lo == 1 else f"{lo:g}-{hi - 1:g}"


class Plots:
    """Figure factory bound to one summary."""

    def __init__(self, s: dict, out: Path) -> None:
        if s["meta"]["replay_meta"]["t"] != 0.97:  # legend strings say t = 0.97
            raise ValueError("summary is not at t = 0.97; update the labels")
        self.s, self.out = s, out
        self.arms = s["meta"]["replay_meta"]["arms"]
        self.eq4 = {m: s["eq4_best"][m]["arm"] for m in RULES}
        for x in (a for a in self.arms if a.startswith("eq4_")):
            STYLE[x] = (f"PV-Finder + Eq. (4), $s_{{max}}={x[5:]}$", EQ4, "D" if x != "eq4_s3" else "d", x != "eq4_s3")  # fmt: skip
        self.mu = sorted((k for k in s["bins"] if k.startswith("mu ")), key=lambda k: float(k[3:].split("-")[0]))  # fmt: skip
        self.mu_x = np.array([float(k[3:].split("-")[0]) for k in self.mu]) + 5
        self.made: list[str] = []

    def save(self, fig: plt.Figure, name: str) -> None:
        save_figure(fig, self.out, name)
        self.made.append(name)

    def arm(self, b: str, ratio: str, arm: str, m: str) -> tuple[float, float]:
        t = self.s["bins"][b]["arms"][m][ratio]
        k = self.arms.index(arm)
        return t["value"][k], t["err"][k]

    def arm_mu(self, ratio: str, arm: str, m: str) -> tuple[np.ndarray, np.ndarray]:
        v = np.array([self.arm(b, ratio, arm, m) for b in self.mu], dtype=float)
        return v[:, 0], v[:, 1]

    def ev_mu(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        v = np.array([[self.s["bins"][b]["event"][key][x] for x in ("value", "err")] for b in self.mu], float)  # fmt: skip
        return v[:, 0], v[:, 1]

    # ---------------------------------------------------------------- vs mu
    def vs_mu(self, series: list[tuple], ylabel: str, name: str, rule: str | None) -> None:  # fmt: skip
        fig, ax = plt.subplots(figsize=(8, 6))
        for arm, v, e, lab in series:
            _pt(ax, self.mu_x, v, e, arm, lab, xerr=5.0)
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 200)
        _ylim(ax, [v for _, v, _, _ in series])
        _desc(ax, "25 000 events, 0.17 mm window", *([RULES[rule][1]] if rule else []))
        ax.legend(loc="upper right", fontsize="small", frameon=False)
        self.save(fig, name + (RULES[rule][0] if rule else ""))

    def all_vs_mu(self) -> None:
        ev = lambda k, a: (a, *self.ev_mu(k), None)  # noqa: E731
        self.vs_mu([ev("eff_pvf", "pvf"), ev("eff_amvf", "amvf")], "Vertex-finding efficiency, Eq. (1)", "eff_pregnn_vs_mu", None)  # fmt: skip
        self.vs_mu([ev("finder_cap_pvf", "finder_pvf"), ev("finder_cap_amvf", "finder_amvf")], "One-to-one efficiency (finder cap)", "finder_cap_vs_mu", None)  # fmt: skip
        trk = {"trk_eff": "Track-assignment efficiency", "trk_pur": "Track-assignment purity", "trk_f1": r"Track-assignment $F_1$", "wrong_rate": "Wrong-assignment rate", "labelled_unassigned_frac": "Unassigned fraction of truth-matched tracks", "unlabelled_left_unassigned": "Unmatched tracks left unassigned", "lost_unassigned_frac": "Unassigned, no vertex near truth / truth-matched"}  # fmt: skip
        cats = {"clean_per_reco": "Clean / reconstructed vertices", "merged_per_reco": "Merged / reconstructed vertices", "split_per_reco": "Split / reconstructed vertices", "fake_per_reco": "Fake / reconstructed vertices", "reco_per_truth": "Reconstructed / truth vertices"}  # fmt: skip
        for m in RULES:
            a = lambda r, x, m=m: (x, *self.arm_mu(r, x, m), None)  # noqa: E731
            cpt, e4 = "clean_per_truth", self.eq4[m]
            ylab = "Clean vertices / truth vertices"
            self.vs_mu([a(cpt, "gnn_t"), a(cpt, "amvf")], ylab, "clean_per_truth_postgnn_vs_mu", m)  # fmt: skip
            self.vs_mu([a(cpt, "truth_guided"), a(cpt, "gnn_t"), a(cpt, "amvf")], ylab, "truth_guided_clean_vs_mu", m)  # fmt: skip
            self.vs_mu([a(cpt, x) for x in dict.fromkeys([e4, "eq4_s3", "gnn_t", "amvf"])], ylab, "eq4_clean_vs_mu", m)  # fmt: skip
            self.vs_mu([a(cpt, "z0_nearest"), a(cpt, "z0_3sigma"), a(cpt, "gnn_t"), a(cpt, "amvf")], ylab, "z0_baselines_clean_vs_mu", m)  # fmt: skip
            self.vs_mu([ev("finder_cap_pvf", "finder_pvf"), a(cpt, "truth_guided"), a(cpt, "gnn_t"), a(cpt, e4), a(cpt, "z0_nearest"), a(cpt, "amvf")], "Fraction of truth vertices", "clean_yield_stages_vs_mu", m)  # fmt: skip
            for r, lab in {**trk, **cats}.items():
                self.vs_mu([a(r, "gnn_t"), a(r, e4), a(r, "z0_nearest"), a(r, "amvf")], lab, f"{r}_vs_mu", m)  # fmt: skip
        a1 = lambda r, x: (x, *self.arm_mu(r, x, "min1"), None)  # noqa: E731
        self.vs_mu([a1("clean_ge2_per_truth", "gnn_t"), a1("clean_ge2_per_truth", "amvf")], r"Clean vertices, dominant $n_{trk} \geq 2$ / truth vertices", "clean_ge2_per_truth_postgnn_vs_mu", "min1")  # fmt: skip

    # ------------------------------------------------- per truth vertex bins
    def vs_var(self, sample: str, var: str, pairs: list[tuple], ylabel: str, name: str, ratio: tuple[str, str] | None, rule: str | None) -> None:  # fmt: skip
        t = self.s["per_truth"][sample][var]
        edges = np.array([np.inf if x is None else x for x in t["edges"]], float)
        nb = len(edges) - 1
        x = np.arange(nb)
        if ratio:
            fig, (ax, rx) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})  # fmt: skip
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
        for key, arm, lab in pairs:
            _pt(ax, x, np.array(t[key]["value"], float), np.array(t[key]["err"], float), arm, lab, xerr=0.5)  # fmt: skip
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.45)
        _desc(
            ax, SEL[sample], "0.17 mm window" + (f", {RULES[rule][1]}" if rule else "")
        )
        ax.legend(loc="upper right", fontsize="small", frameon=False)
        axl = ax
        if ratio:
            a, b = (np.array(t[k]["value"], float) for k in ratio)
            de = np.array(t.get(f"{ratio[0]}-minus-{ratio[1]}_err", [np.nan] * nb), float)  # fmt: skip
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
        if var == "n_trk":
            lab = [_ntrk_label(edges[i], edges[i + 1]) for i in range(nb)]
        else:
            lab = [f">{edges[i]:g}" if not np.isfinite(edges[i + 1]) else f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(nb)]  # fmt: skip
        axl.set_xticks(x)
        axl.set_xticklabels(lab, rotation=45 if nb > 7 else 0, fontsize="small")
        axl.set_xlabel({"n_trk": r"Truth vertex $n_{trk}$", "density": r"Local pile-up density [vertices/mm]", "dz_nearest": r"Distance to nearest truth vertex [mm]"}[var])  # fmt: skip
        self.save(fig, name + (RULES[rule][0] if rule else ""))

    def all_vs_var(self) -> None:
        for sample in (PU200, "inclusive"):
            tg = TAG[sample]
            for var in ("n_trk", "density", "dz_nearest"):
                self.vs_var(sample, var, [("found_pvf", "pvf", None), ("found_amvf", "amvf", None)], "Vertex-finding efficiency, Eq. (1)", f"eff_vs_{var}_pregnn_{tg}", ("found_pvf", "found_amvf"), None)  # fmt: skip
                for m, (_, _, pre) in RULES.items():
                    self.vs_var(sample, var, [(f"{pre}gnn_t", "gnn_t", None), (f"{pre}amvf", "amvf", None)], "Fraction with a Clean vertex", f"eff_vs_{var}_postgnn_{tg}", (f"{pre}gnn_t", f"{pre}amvf"), m)  # fmt: skip
                    self.vs_var(sample, var, [("o2o_pvf", "finder_pvf", None), (f"{pre}truth_guided", "truth_guided", None), (f"{pre}gnn_t", "gnn_t", None), (f"{pre}{self.eq4[m]}", self.eq4[m], None), (f"{pre}amvf", "amvf", None)], "Fraction of truth vertices", f"stages_vs_{var}_{tg}", None, m)  # fmt: skip

    # ------------------------------------------------------------ PU200 only
    def stages(self, m: str) -> None:
        """Fig. 7.2: Clean yield stages at PU200 (finder caps are rule independent)."""
        b = self.s["bins"][PU200]

        def d(arm: str) -> dict:
            return dict(
                zip(("value", "err"), self.arm(PU200, "clean_per_truth", arm, m))
            )

        rows = [
            ("One-to-one finder match, PV-Finder", b["event"]["finder_cap_pvf"], "0.6"),
            ("Truth-guided assignment (MC only)", d("truth_guided"), "0.35"),
            ("PV-Finder + GNN ($t=0.97$)", d("gnn_t"), PVF),
            ("PV-Finder + GNN (no score cut)", d("gnn_t0"), "#CC79A7"),
            (STYLE[self.eq4[m]][0] + " (best)", d(self.eq4[m]), EQ4),
            (STYLE["eq4_s3"][0], d("eq4_s3"), EQ4),
            (STYLE["z0_3sigma"][0], d("z0_3sigma"), "#E69F00"),
            (STYLE["z0_nearest"][0], d("z0_nearest"), "#009E73"),
            ("One-to-one finder match, AMVF", b["event"]["finder_cap_amvf"], "#E8A07A"),
            ("AMVF", d("amvf"), AMVF),
        ]
        fig, ax = plt.subplots(figsize=(9, 6.5))
        y = np.arange(len(rows))[::-1]
        for yi, (_, v, col) in zip(y, rows):
            ax.barh(yi, v["value"], xerr=v["err"], color=col, height=0.65)
            ax.text(v["value"] + 0.01, yi, f"{100 * v['value']:.1f}%", va="center", fontsize="small")  # fmt: skip
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize="small")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.7, len(rows) + 1.4)
        ax.set_xlabel("Fraction of truth vertices (Clean, or one-to-one matched)")
        atlas_label(ax, desc=f"{SAMPLE}\n{SEL[PU200]}\n0.17 mm window, Clean among {RULES[m][1]}", desc_xy=(0.03, 0.975))  # fmt: skip
        self.save(fig, "clean_yield_stages_pu200" + RULES[m][0])

    def scan(self, sample: str, m: str) -> None:
        """Eq. (4) s_max scan vs GNN t scan: track eff vs purity; Clean/truth vs purity."""
        t = self.s["tscan"][sample][m]
        eq4 = [x for x in self.arms if x.startswith("eq4_")]
        for yk, ylabel, name in (("trk_eff", "Track-assignment efficiency", "trk_eff_vs_pur_scan"), ("clean_per_truth", "Clean vertices / truth vertices", "clean_vs_pur_scan")):  # fmt: skip
            fig, ax = plt.subplots(figsize=(8, 6))
            gx, gy = np.array(t["trk_pur"]["value"][:-1]), np.array(t[yk]["value"][:-1])
            ax.plot(gx, gy, "-o", color=PVF, ms=4, label="PV-Finder + GNN, $t$ scan")
            t_cut = self.s["meta"]["replay_meta"]["t"]
            k = t["t"].index(t_cut)
            ax.plot(gx[k], gy[k], "*", color=PVF, ms=16, label=f"Frozen working point $t={t_cut:g}$")  # fmt: skip
            ex = [self.arm(sample, "trk_pur", x, m)[0] for x in eq4]
            ey = [self.arm(sample, yk, x, m)[0] for x in eq4]
            ax.plot(ex, ey, "-D", color=EQ4, ms=5, label=r"PV-Finder + Eq. (4), $s_{max}$ scan")  # fmt: skip
            for x, a, bb in zip(eq4, ex, ey):
                ax.annotate(x[5:], (a, bb), textcoords="offset points", xytext=(4, -10), fontsize="x-small", color="#2a7fb0")  # fmt: skip
            for arm in ("z0_nearest", "z0_3sigma", "amvf"):
                _pt(ax, self.arm(sample, "trk_pur", arm, m)[0], self.arm(sample, yk, arm, m)[0], None, arm)  # fmt: skip
            ax.set_xlabel("Track-assignment purity")
            ax.set_ylabel(ylabel)
            _ylim(ax, [np.array(ey), gy])
            _desc(ax, SEL[sample], RULES[m][1])
            ax.legend(loc="upper right", fontsize="x-small", frameon=False)
            self.save(fig, f"{name}_{TAG[sample]}{RULES[m][0]}")

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
            ax.set_xticks(x)
            ax.set_xticklabels([_ntrk_label(*(np.inf if v is None else v for v in r["bin"])) for r in rows], rotation=45, fontsize="small")  # fmt: skip
            ax.set_xlabel(r"Truth vertex $n_{trk}$ (one-to-one matched peak)")
            ax.set_ylabel(r"$\sigma_{PV}$ [mm]")
            ax.set_ylim(0, 0.3)
            _desc(ax, SEL[sample])
            ax.legend(loc="upper right", fontsize="x-small", frameon=False)
            self.save(fig, f"sigma_pv_peakwidth_vs_param_{TAG[sample]}")

    # ------------------------------------------------------- track diagnosis
    def trackdiag(self, td: dict) -> None:
        """Track efficiency by rule vs mu; what the score cut removes vs truth n_trk."""
        edges = np.array([np.inf if v is None else v for v in td["ntrk_edges"]], float)
        nb, x = len(edges) - 1, np.arange(len(edges) - 1)
        lab = [_ntrk_label(edges[i], edges[i + 1]) for i in range(nb)]
        series = {"g0": ("gnn_t0", None), "g1": ("gnn_t", r"PV-Finder + GNN ($t=0.97$), $\geq 1$ track"), "g2": ("gnn_t", r"PV-Finder + GNN ($t=0.97$), $\geq 2$ tracks"), "a2": ("amvf", None)}  # fmt: skip
        for q, ylabel in (("eff", "Track-assignment efficiency"), ("pur", "Track-assignment purity")):  # fmt: skip
            fig, ax = plt.subplots(figsize=(8, 6))
            vals = []
            for k, (arm, lb) in series.items():
                v = np.array([td["bins"][b][f"{q}_{k}"]["value"] for b in self.mu])
                e = np.array([td["bins"][b][f"{q}_{k}"]["err"] for b in self.mu])
                st = STYLE[arm]
                ax.errorbar(self.mu_x, v, yerr=e, xerr=5, fmt=st[2], color=st[1] if k != "g1" else "#7FB8DA", mfc=st[1] if (st[3] and k != "g1") else "none", label=lb or st[0], ms=6, lw=1.2)  # fmt: skip
                vals.append(v)
            ax.set_xlabel(r"$\mu$")
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 200)
            _ylim(ax, vals)
            _desc(ax, "25 000 events")
            ax.legend(loc="upper right", fontsize="x-small", frameon=False)
            self.save(fig, f"trackdiag_trk_{q}_vs_mu_rules")
        fig, ax = plt.subplots(figsize=(8, 6))
        for b, col in MU_SERIES:
            r = td["bins"][b]["cut_frac_of_t0_correct"]
            lo, hi = b[3:].split("-")
            ax.errorbar(x, r["by_ntrk"], yerr=r["by_ntrk_err"], xerr=0.5, fmt="o", color=col, ms=5, label=f"${lo} \\leq \\mu < {hi}$")  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=45, fontsize="small")
        ax.set_xlabel(r"Truth $n_{trk}$ of the track's vertex")
        ax.set_ylabel(r"Correct at $t=0$, removed by $t=0.97$")
        ax.set_ylim(0, 1.0)
        _desc(ax, "PV-Finder + GNN, 25 000 events")
        ax.legend(loc="upper right", fontsize="small", frameon=False)
        self.save(fig, "trackdiag_cut_loss_vs_ntrk")
        for b in ("mu 0-10", PU200):
            fig, ax = plt.subplots(figsize=(8, 6))
            for k, (arm, lb) in series.items():
                st = STYLE[arm]
                r = td["bins"][b][f"eff_{k}"]
                ax.errorbar(x, r["by_ntrk"], yerr=r["by_ntrk_err"], xerr=0.5, fmt=st[2], color=st[1] if k != "g1" else "#7FB8DA", mfc=st[1] if (st[3] and k != "g1") else "none", label=lb or st[0], ms=6)  # fmt: skip
            ax.set_xticks(x)
            ax.set_xticklabels(lab, rotation=45, fontsize="small")
            ax.set_xlabel(r"Truth $n_{trk}$ of the track's vertex")
            ax.set_ylabel("Track-assignment efficiency")
            ax.set_ylim(0, 1.45)
            _desc(ax, SEL.get(b, r"$0 \leq \mu < 10$"))
            ax.legend(loc="upper right", fontsize="x-small", frameon=False)
            self.save(fig, f"trackdiag_trk_eff_vs_ntrk_{TAG.get(b, 'mu0-10')}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    a = ap.parse_args()
    s = json.loads(a.summary.read_text())
    use_atlas_style()
    p = Plots(s, a.out or a.summary.parent / "plots")
    for m in RULES:
        p.stages(m)
        for sample in (PU200, "inclusive"):
            p.scan(sample, m)
    p.all_vs_var()
    p.all_vs_mu()
    p.sigma_pv()
    td = a.summary.parent / "chain_extras_trackdiag.json"
    if td.exists():
        p.trackdiag(json.loads(td.read_text()))
    print("\n".join(p.made))


if __name__ == "__main__":
    main()
