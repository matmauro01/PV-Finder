"""Figures for the Run 2 vs HL-LHC resolution comparison (ATLAS style)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from gnn.diagnostics.plot_style import atlas_label, save_figure, use_atlas_style

R2_DESC = r"Run 2 MC $t\bar{t}$, $\sqrt{s}=13$ TeV"
HL_DESC = r"HL-LHC ITk $t\bar{t}$, $\sqrt{s}=14$ TeV"
COL = {
    "r2_pvf": "#0072B2", "r2_amvf": "#D55E00",
    "hl_pvf": "#56B4E9", "hl_pvfn": "#009E73", "hl_amvf": "#E69F00", "truth": "#555555",
}  # fmt: skip
PK, PN = "pvf_int050_h000_full", "pvf_int050_h003_cen"


def _folded(H: np.ndarray, idx: np.ndarray, rebin: int = 2) -> tuple[np.ndarray, ...]:
    """|dz| density normalised to its 3-6 mm plateau, with Poisson errors."""
    c = H[idx].sum(0).astype(float)
    n = len(c) // 2
    f = c[n:] + c[:n][::-1]  # bins symmetric about 0
    f = f.reshape(-1, rebin).sum(1)
    x = (np.arange(len(f)) + 0.5) * (6.0 / len(f))
    plat = f[x > 3.0].mean()
    return x, f / plat, np.sqrt(f) / plat


def fig_pair_shapes(H2: dict, HH: dict, sel: dict, out: Path) -> None:
    """Pair-separation density relative to plateau: Run 2, HL rho-matched, HL PU200."""
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.5), sharey=True)
    panels = (
        (
            H2,
            sel["run2"],
            R2_DESC + "\n" + r"$\langle\mu\rangle\approx40$, 2550 evt",
            "r2",
        ),
        (HH, sel["hl_rho_match"], HL_DESC + "\nRun 2 density matched, 2550 evt", "hl"),
        (HH, sel["hl_pu200"], HL_DESC + "\n" + r"$185\leq\mu\leq215$, 1920 evt", "hl"),
    )
    for ax, (H, idx, desc, tag) in zip(axs, panels):
        series = [
            (PK, "PV-Finder", COL[f"{tag}_pvf"]),
            ("amvf", "AMVF", COL[f"{tag}_amvf"]),
        ]
        if tag == "hl":
            series.insert(1, (PN, "PV-Finder (note config)", COL["hl_pvfn"]))
        series.append(("truth", r"Truth $n_\mathrm{trk}\geq2$", COL["truth"]))
        for key, lab, c in series:
            x, y, e = _folded(H[key], idx)
            ax.errorbar(x, y, yerr=e, fmt="o", ms=3, color=c, label=lab)
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2.1)
        ax.set_xlabel(r"$|\Delta z|$ between vertices [mm]")
        atlas_label(ax, desc=desc, desc_xy=(0.05, 0.88))
        ax.legend(loc="lower right", fontsize="small")
    axs[0].set_ylabel("Pair density / plateau (3-6 mm)")
    save_figure(fig, out, "pair_density_shapes")


def _rows_xy(
    rows: list[dict], key: str, val: str = "sigma_mm", err: str = "boot_err_mm"
):
    x = np.array([0.5 * (r["lo"] + r["hi"]) for r in rows])
    xe = np.array([0.5 * (r["hi"] - r["lo"]) for r in rows])
    y = np.array([r[key][val] for r in rows])
    ye = np.array([r[key][err] for r in rows])
    return x, xe, y, ye


def fig_sigma_vs(res: dict, var: str, xlabel: str, out: Path) -> None:
    """sigma_vtx-vtx vs event density or mu, both samples, with AMVF/PVF ratio panel."""
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})  # fmt: skip
    R = res[f"sigma_vs_{var}"]
    for tag, key, lab, c, m in (
        ("run2", PK, "Run 2: PV-Finder", COL["r2_pvf"], "o"),
        ("run2", "amvf", "Run 2: AMVF", COL["r2_amvf"], "o"),
        ("hl", PK, "HL-LHC: PV-Finder (Run 2 config)", COL["hl_pvf"], "s"),
        ("hl", PN, "HL-LHC: PV-Finder (note config)", COL["hl_pvfn"], "^"),
        ("hl", "amvf", "HL-LHC: AMVF", COL["hl_amvf"], "s"),
    ):
        x, xe, y, ye = _rows_xy(R[tag], key)
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=m, color=c, label=lab, ms=5)
    for tag, c, m in (("run2", COL["r2_amvf"], "o"), ("hl", COL["hl_amvf"], "s")):
        rows = R[tag]
        x = np.array([0.5 * (r["lo"] + r["hi"]) for r in rows])
        y = np.array([r["ratio"]["ratio"] for r in rows])
        ye = np.array([r["ratio"]["err"] for r in rows])
        rx.errorbar(x, y, yerr=ye, fmt=m, color=c, ms=5,
                    label="Run 2" if tag == "run2" else "HL-LHC")  # fmt: skip
    ax.set_ylabel(r"$\sigma_\mathrm{vtx-vtx}$ [mm]")
    ax.set_ylim(0, 1.25)
    atlas_label(
        ax, desc=r"$t\bar{t}$; Run 2 13 TeV / HL-LHC 14 TeV", desc_xy=(0.05, 0.88)
    )
    ax.legend(loc="upper right", fontsize="x-small", ncol=1)
    rx.axhline(1.0, color="k", lw=0.6, ls=":")
    rx.set_ylabel("AMVF / PVF")
    rx.set_ylim(0.6, 3.0)
    rx.set_xlabel(xlabel)
    rx.legend(fontsize="x-small", loc="upper right", ncol=2)
    save_figure(fig, out, f"sigma_vs_{var}")


def fig_hs_vs_density(res: dict, out: Path) -> None:
    """ITk-style HS longitudinal resolution vs local PU density."""
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})  # fmt: skip
    D = res["hs_vs_local_density"]
    for tag, key, lab, c, m in (
        ("run2", PK, "Run 2: PV-Finder", COL["r2_pvf"], "o"),
        ("run2", "amvf", "Run 2: AMVF", COL["r2_amvf"], "o"),
        ("hl_all", PK, "HL-LHC: PV-Finder (Run 2 config)", COL["hl_pvf"], "s"),
        ("hl_all", PN, "HL-LHC: PV-Finder (note config)", COL["hl_pvfn"], "^"),
        ("hl_all", "amvf", "HL-LHC: AMVF", COL["hl_amvf"], "s"),
    ):
        rows = [r for r in D[tag][key] if r["n_in"] >= 30]
        x = np.array([0.5 * (r["lo"] + r["hi"]) for r in rows])
        xe = np.array([0.5 * (r["hi"] - r["lo"]) for r in rows])
        y = 1e3 * np.array([r["rms"] for r in rows])
        ye = 1e3 * np.array([r["rms_err"] for r in rows])
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt=m, color=c, label=lab, ms=5)
    for tag, c, m in (("run2", COL["r2_pvf"], "o"), ("hl_all", COL["hl_pvf"], "s")):
        p, a = D[tag][PK], D[tag]["amvf"]
        ok = [i for i in range(len(p)) if p[i]["n_in"] >= 30 and a[i]["n_in"] >= 30]
        x = np.array([0.5 * (p[i]["lo"] + p[i]["hi"]) for i in ok])
        y = np.array([p[i]["rms"] / a[i]["rms"] for i in ok])
        ye = y * np.hypot([p[i]["rms_err"] / p[i]["rms"] for i in ok],
                          [a[i]["rms_err"] / a[i]["rms"] for i in ok])  # fmt: skip
        rx.errorbar(x, y, yerr=ye, fmt=m, color=c, ms=5,
                    label="Run 2" if tag == "run2" else "HL-LHC (Run 2 config)")  # fmt: skip
    ax.set_ylabel(r"HS $\Delta z$ iterative RMS [$\mu$m]")
    ax.set_ylim(0, None)
    atlas_label(ax, desc=r"$t\bar{t}$, closest reco within 1 mm; Run 2 13 TeV ($\mu$ 0-80)"
                "\nHL-LHC 14 TeV (flat $\\mu$ 0-200)", desc_xy=(0.05, 0.88))  # fmt: skip
    ax.legend(loc="lower right", fontsize="x-small")
    rx.axhline(1.0, color="k", lw=0.6, ls=":")
    rx.set_ylabel("PVF / AMVF")
    rx.set_xlabel("Local PU density [vertices/mm]")
    rx.legend(fontsize="x-small")
    save_figure(fig, out, "hs_resolution_vs_local_density")


def fig_detector_inputs(res: dict, out: Path) -> None:
    """Track z0 error vs pT (two |eta| cells), ideal PU vertex precision, target widths."""
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.5))
    tr = res["tracks"]
    pe = np.array(tr["pt_edges"], dtype=float)
    pe[-1] = 30.0
    xc = np.sqrt(pe[:-1] * pe[1:])
    for i, ls in ((0, "-"), (3, "--")):
        lab_eta = f"{tr['eta_edges'][i]}<|$\\eta$|<{tr['eta_edges'][i + 1]}"
        for tag, c in (("run2", COL["r2_pvf"]), ("hl", COL["hl_amvf"])):
            y = np.array(tr[tag]["median_errz0_um"][i], dtype=float)
            nm = "Run 2 ID" if tag == "run2" else "ITk"
            axs[0].plot(xc, y, ls, marker="o", color=c, label=f"{nm}, {lab_eta}")
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"Track $p_\mathrm{T}$ [GeV]")
    axs[0].set_ylabel(r"Median track $\sigma(z_0)$ [$\mu$m]")
    for k, key, ylab, title in (
        (
            1,
            "vertex_precision",
            r"Median ideal PU-vertex $\sigma_z$ [$\mu$m]",
            "PU vertices",
        ),
        (
            2,
            "target_widths",
            r"Median training-target RMS [$\mu$m]",
            "isolated vertices",
        ),
    ):
        tb = res[key]
        nb = np.array(tb["ntrk_edges"], dtype=float)
        nb[-1] = 100.0
        xm = np.sqrt(nb[:-1] * nb[1:])
        for tag, c, nm in (
            ("run2", COL["r2_pvf"], "Run 2"),
            ("hl", COL["hl_amvf"], "HL-LHC"),
        ):
            vals = (
                tb[tag]["pu_median_um_by_ntrk"]
                if key == "vertex_precision"
                else tb[tag]
            )
            y = np.array([np.nan if v is None else v for v in vals], dtype=float)
            axs[k].plot(xm, y, "o-", color=c, label=f"{nm} ({title})")
        axs[k].set_xscale("log")
        axs[k].set_xlabel(r"Truth $n_\mathrm{trk}$")
        axs[k].set_ylabel(ylab)
    for ax, d in zip(axs, ("track", "from associated tracks", "PV-Finder targets")):
        atlas_label(ax, desc=r"$t\bar{t}$, Run 2 vs HL-LHC: " + d, desc_xy=(0.05, 0.88))
        ax.set_ylim(0, None)
        ax.legend(fontsize="x-small", loc="upper right")
    save_figure(fig, out, "detector_inputs")


def fig_summary(res: dict, out: Path) -> None:
    """sigma for PVF and AMVF on each event list, with the AMVF/PVF ratio below."""
    S = res["sigma"]
    cols = [("run2", "Run 2\nall"), ("hl_rho_match", "HL\nRun 2 dens."),
            ("hl_mu_match", r"HL Run 2 $\mu$"), ("hl_mu80", r"HL $\mu\leq80$"),
            ("hl_pu200", "HL PU200")]  # fmt: skip
    x = np.arange(len(cols))
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.05})  # fmt: skip
    for key, lab, c, dx in ((PK, "PV-Finder (Run 2 config)", COL["r2_pvf"], -0.12),
                            ("amvf", "AMVF", COL["r2_amvf"], 0.12)):  # fmt: skip
        y = [S[s][key]["sigma_mm"] for s, _ in cols]
        e = [S[s][key]["boot_err_mm"] for s, _ in cols]
        ax.errorbar(x + dx, y, yerr=e, fmt="o", color=c, label=lab, ms=7)
    yn = [S[s][PN]["sigma_mm"] if PN in S[s] else np.nan for s, _ in cols]
    en = [S[s][PN]["boot_err_mm"] if PN in S[s] else np.nan for s, _ in cols]
    ax.errorbar(x, yn, yerr=en, fmt="^", color=COL["hl_pvfn"], ms=7,
                label="PV-Finder (note config)")  # fmt: skip
    r = [S[s]["ratio_amvf_over_pvf"]["ratio"] for s, _ in cols]
    re = [S[s]["ratio_amvf_over_pvf"]["err"] for s, _ in cols]
    rx.errorbar(x, r, yerr=re, fmt="o", color="k", label="Run 2 config")
    rn = [
        S[s].get("ratio_amvf_over_pvf_note", {}).get("ratio", np.nan) for s, _ in cols
    ]
    rne = [S[s].get("ratio_amvf_over_pvf_note", {}).get("err", np.nan) for s, _ in cols]
    rx.errorbar(
        x + 0.1, rn, yerr=rne, fmt="^", color=COL["hl_pvfn"], label="note config"
    )
    ax.set_ylabel(r"$\sigma_\mathrm{vtx-vtx}$ [mm]")
    ax.set_ylim(0, 1.1)
    atlas_label(
        ax,
        desc=r"$t\bar{t}$; identical fit (300 bins, $\pm6$ mm)",
        desc_xy=(0.05, 0.88),
    )
    ax.legend(loc="upper right", fontsize="small")
    rx.set_ylabel("AMVF / PVF")
    rx.axhline(1.0, color="k", lw=0.6, ls=":")
    rx.set_ylim(0.8, 2.2)
    rx.set_xticks(x)
    rx.set_xticklabels([c[1] for c in cols], fontsize="small")
    rx.legend(fontsize="x-small", ncol=2)
    save_figure(fig, out, "sigma_summary")


def fig_ntrk_scan(res: dict, out: Path) -> None:
    """AMVF sigma_vtx-vtx vs the reco nTracks cut."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for s, lab, c in (("run2", "Run 2 MC", COL["r2_amvf"]),
                      ("hl_rho_match", "HL-LHC, Run 2 density", COL["hl_amvf"]),
                      ("hl_pu200", "HL-LHC PU200", COL["hl_pvfn"])):  # fmt: skip
        d = res["ntrk_scan"][s]
        k = sorted(int(v) for v in d)
        ax.plot(k, [d[v] if v in d else d[str(v)] for v in k], "o-", color=c, label=lab)
    ax.set_xlabel(r"AMVF vertex $n_\mathrm{trk}$ cut ($\geq$)")
    ax.set_ylabel(r"AMVF $\sigma_\mathrm{vtx-vtx}$ [mm]")
    ax.set_ylim(0, 1.2)
    atlas_label(
        ax, desc=r"$t\bar{t}$; Run 2 13 TeV / HL-LHC 14 TeV", desc_xy=(0.05, 0.88)
    )
    ax.legend(fontsize="small", loc="upper right")
    save_figure(fig, out, "amvf_ntrk_cut_scan")


def make_all(res: dict, arr: dict, H2: dict, HH: dict, edges: np.ndarray, sel: dict,
             r2: Any, hl: Any, out: Path) -> None:  # fmt: skip
    """Write every figure into ``out/figures``."""
    use_atlas_style()
    fo = out / "figures"
    fig_pair_shapes(H2, HH, sel, fo)
    fig_sigma_vs(res, "rho", r"Event truth density at $z=0$ [vertices/mm]", fo)
    fig_sigma_vs(res, "mu", r"$\mu$ (ActualNumOfInt)", fo)
    fig_hs_vs_density(res, fo)
    fig_detector_inputs(res, fo)
    fig_summary(res, fo)
    fig_ntrk_scan(res, fo)
    print(f"[plots] wrote figures to {fo}", flush=True)
