"""The window-scan figure — the headline deliverable of the fair comparison.

Six panels against one shared x axis (the common matching window).  Never a
dual-axis chart: efficiency and fake rate are different measures on different
scales and get their own panels.  The accidental floor is drawn *on* the
efficiency panels rather than on a second axis because it is itself an
efficiency, in the same units — and the point of showing it is that the reader
can see how much of the curve above it is coincidence.

Colour is the reference categorical palette's first two slots in fixed order
(blue = PV-Finder, orange = AMVF), a validated adjacent pair in light mode.
Identity is additionally carried by line style and marker, so the figure survives
colour-blind readers, greyscale printing and photocopied proceedings.  The
accidental floor is drawn in muted text ink, not a third series colour, because
it is a reference level rather than an algorithm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots 1 and 2, light mode.
C_PVF, C_AMVF = "#2a78d6", "#eb6834"
INK, INK_2, INK_3, GRID = "#0b0b0b", "#52514e", "#8a8880", "#dcdbd6"

STYLE = {
    "pvf": dict(color=C_PVF, ls="-", marker="o", label="PV-Finder"),
    "amvf": dict(color=C_AMVF, ls="--", marker="s", label="AMVF"),
}

# key, title, y label, scale, accidental-floor key (or None)
PANELS = (
    ("eff", "Efficiency — merged-credit convention (as published)",
     "efficiency  [%]", 100.0, "eff_accidental"),
    ("eff_strict", "Efficiency — strict one-to-one",
     "efficiency  [%]", 100.0, "eff_strict_accidental"),
    ("fake_per_evt", "Fake rate — standard truth (nTrk $\\geq$ 2)",
     "fakes / event", 1.0, None),
    ("fake_corr_per_evt", "Fake rate — corrected truth (nTrk $\\geq$ 1)",
     "fakes / event", 1.0, None),
    ("surplus_per_evt", "Fake + split — immune to relabelling",
     "surplus / event", 1.0, None),
    (None, "PV-Finder $-$ AMVF, efficiency", "difference  [points]", 100.0, None),
)  # fmt: skip


def _series(rows: list[dict[str, Any]], arm: str, key: str, scale: float):
    """Point estimate and paired-bootstrap error for one arm and metric."""
    y = np.array([r[arm][key] for r in rows]) * scale
    e = np.array([r[arm][key + "_err"] for r in rows]) * scale
    return y, e


def _diff_panel(ax, rows: list[dict[str, Any]], w: np.ndarray) -> None:
    """Paired PV-Finder minus AMVF efficiency, both conventions, with zero line."""
    ax.axhline(0.0, color=INK_2, lw=1.2, zorder=1)
    for key, ls, mk, lab in (
        ("eff", "-", "o", "merged credit"),
        ("eff_strict", ":", "^", "strict 1-to-1"),
    ):
        d = np.array([r["diff"][key] for r in rows]) * 100.0
        e = np.array([r["diff"][key + "_err"] for r in rows]) * 100.0
        ax.fill_between(w, d - e, d + e, color=C_PVF, alpha=0.15, lw=0)
        ax.plot(w, d, color=C_PVF, ls=ls, lw=2.0, marker=mk, ms=4.5, mew=0,
                label=lab)  # fmt: skip
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)


def plot_window_scan(payload: dict[str, Any], out_path: str | Path) -> None:
    """Draw and save the six-panel window scan."""
    rows = payload["scan"]["rows"]
    w = np.array([r["window_mm"] for r in rows])
    meta = payload["meta"]
    sig = payload["self_sigma_view"]["sigma"]
    core = payload["core_position_resolution_mm"]
    w_prim = payload["headline_window_mm"]
    w_sens = list(payload["headline_windows"][1:])

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.4), sharex=True)
    for ax, (key, title, ylab, scale, acc_key) in zip(axes.ravel(), PANELS):
        if key is None:
            _diff_panel(ax, rows, w)
        else:
            for arm in ("pvf", "amvf"):
                y, e = _series(rows, arm, key, scale)
                st = STYLE[arm]
                ax.fill_between(w, y - e, y + e, color=st["color"], alpha=0.18, lw=0)
                ax.plot(w, y, color=st["color"], ls=st["ls"], lw=2.0,
                        marker=st["marker"], ms=4.5, mew=0, label=st["label"])  # fmt: skip
            if acc_key is not None:
                a1 = np.array([r["pvf"][acc_key] for r in rows]) * scale
                a2 = np.array([r["amvf"][acc_key] for r in rows]) * scale
                # One band, not two lines: the two algorithms' accidental floors
                # agree to a few tenths of a point, and a band says exactly that.
                ax.fill_between(w, np.minimum(a1, a2), np.maximum(a1, a2),
                                color=INK_3, alpha=0.35, lw=0)  # fmt: skip
                ax.plot(w, 0.5 * (a1 + a2), color=INK_2, ls=(0, (1, 1.6)), lw=1.6,
                        label="accidental floor")  # fmt: skip

        # Reference windows: the two sigma_vtx-vtx values (far apart, which is
        # the problem), the primary resolution-derived window, and 0.5 mm.
        ax.axvline(w_prim, color=INK, lw=1.5, alpha=0.75, zorder=0)
        for ws in w_sens:
            ax.axvline(ws, color=INK_2, lw=1.2, ls=(0, (5, 3)), alpha=0.6, zorder=0)
        for a in ("pvf", "amvf"):
            ax.axvline(sig[a]["sigma_mm"], color=STYLE[a]["color"], lw=1.0, ls=":",
                       alpha=0.75, zorder=0)  # fmt: skip

        ax.set_title(title, fontsize=10.0, color=INK, pad=7, loc="left")
        ax.set_ylabel(ylab, fontsize=9.5, color=INK_2)
        ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9, length=3)
        ax.set_xlim(w.min(), w.max())
        if key is not None and key.startswith("eff"):
            # An efficiency axis that runs past 100 % reads as an error.
            ax.set_ylim(top=100.0)

    for ax in axes[1]:
        ax.set_xlabel("common matching window  [mm]", fontsize=9.5, color=INK_2)

    # Label the reference windows once, on the first panel, running upward from
    # the bottom axis through the empty band between the accidental floor and
    # the efficiency curves.
    ax0 = axes[0, 0]
    marks = [
        (w_prim, f"{w_prim:.2f} mm = 3 $\\times$ core res.  (primary)", INK),
        (sig["pvf"]["sigma_mm"],
         f"$\\sigma^{{vtx-vtx}}_{{PVF}}$ {sig['pvf']['sigma_mm']:.3f}", C_PVF),
        (sig["amvf"]["sigma_mm"],
         f"$\\sigma^{{vtx-vtx}}_{{AMVF}}$ {sig['amvf']['sigma_mm']:.3f}", C_AMVF),
    ] + [(ws, f"{ws:g} mm (sensitivity)", INK_2) for ws in w_sens]  # fmt: skip
    # Stagger the vertical start so labels on nearby guides (0.17 and 0.218 mm
    # sit only 0.05 mm apart) cannot overlap.
    marks.sort(key=lambda m: m[0])
    for i, (x, txt, col) in enumerate(marks):
        ax0.annotate(txt, xy=(x, ax0.get_ylim()[0]), xytext=(3, 5 + 26 * (i % 2)),
                     textcoords="offset points", fontsize=7.6, color=col,
                     rotation=90, va="bottom", ha="left")  # fmt: skip

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.997, 0.998),
               frameon=False, fontsize=10, ncol=3, labelcolor=INK)  # fmt: skip

    fig.suptitle(
        "PV-Finder vs AMVF at a common matching window",
        x=0.006, y=0.988, ha="left", fontsize=14.5, color=INK, weight="bold",
    )  # fmt: skip
    fig.text(
        0.006, 0.956,
        f"HL-LHC PU200 {meta['tag']}, {meta['n_events']} events, "
        f"$\\mu \\in$ [{meta['mu_min']:g}, {meta['mu_max']:g}], "
        f"$\\langle\\mu\\rangle$ = {meta['mu_mean']:.1f}.  Identical events and "
        f"identical window for both algorithms; bands are $\\pm 1\\sigma$ paired "
        f"bootstrap over events.",
        ha="left", fontsize=9.0, color=INK_2,
    )  # fmt: skip
    fig.text(
        0.006, 0.930,
        f"Window basis is the core position resolution — PV-Finder "
        f"{core['pvf'] * 1000:.1f} $\\mu$m, AMVF {core['amvf'] * 1000:.1f} $\\mu$m, "
        f"nearly equal — not $\\sigma_{{vtx-vtx}}$, which differs by 40 % between "
        f"the two and is what the production eval uses.  The accidental floor is "
        f"the efficiency still scored with the reco list displaced by 3 mm.",
        ha="left", fontsize=9.0, color=INK_2,
    )  # fmt: skip

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.915))
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cells(h: dict[str, Any], arm: str, name: str) -> str:
    """One algorithm's row of the headline table."""
    x = h[arm]
    return (
        f"| {name} | {100 * x['eff']:.2f} ± {100 * x['eff_err']:.2f} "
        f"| {100 * x['eff_strict']:.2f} ± {100 * x['eff_strict_err']:.2f} "
        f"| {x['fake_per_evt']:.2f} ± {x['fake_per_evt_err']:.2f} "
        f"| {x['fake_corr_per_evt']:.2f} ± {x['fake_corr_per_evt_err']:.2f} "
        f"| {x['surplus_per_evt']:.2f} ± {x['surplus_per_evt_err']:.2f} |"
    )


def _table_block(h: dict[str, Any], w: float, label: str) -> list[str]:
    """One window's table, with its accidental floor stated underneath."""
    d = h["diff"]
    acc = h.get("accidental", {})
    lines = [
        f"### Common window {w:g} mm — {label}",
        "",
        "| | efficiency [%] | efficiency, strict 1-to-1 [%] "
        "| fake/evt (nTrk>=2) | fake/evt (nTrk>=1) | fake+split/evt |",
        "|---|---|---|---|---|---|",
        _cells(h, "pvf", "**PV-Finder**"),
        _cells(h, "amvf", "**AMVF**"),
        f"| **PVF - AMVF** (paired) | {100 * d['eff']:+.2f} ± "
        f"{100 * d['eff_err']:.2f} | {100 * d['eff_strict']:+.2f} ± "
        f"{100 * d['eff_strict_err']:.2f} | {d['fake_per_evt']:+.2f} ± "
        f"{d['fake_per_evt_err']:.2f} | {d['fake_corr_per_evt']:+.2f} ± "
        f"{d['fake_corr_per_evt_err']:.2f} | {d['surplus_per_evt']:+.2f} ± "
        f"{d['surplus_per_evt_err']:.2f} |",
    ]
    if acc:
        lines += [
            "",
            "Accidental floor at this window (reco list displaced 3 mm): "
            f"PV-Finder {100 * acc['pvf']['eff']:.1f} % merged / "
            f"{100 * acc['pvf']['eff_strict']:.1f} % strict, "
            f"AMVF {100 * acc['amvf']['eff']:.1f} % / "
            f"{100 * acc['amvf']['eff_strict']:.1f} %.",
        ]
    return lines + [""]


def headline_table_md(payload: dict[str, Any]) -> str:
    """The headline tables plus the asymmetry decomposition, as markdown."""
    m = payload["meta"]
    core = payload["core_position_resolution_mm"]
    a1 = payload["asymmetry_1_window"]
    a2 = payload["asymmetry_2_low_ntrk_truth"]
    heads = payload["headlines"]
    wins = payload["headline_windows"]

    lines = [
        f"# Fair comparison — {m['tag']}",
        "",
        f"{m['n_events']} events, mu in [{m['mu_min']:g}, {m['mu_max']:g}], "
        f"<mu> = {m['mu_mean']:.1f}. Errors are bootstrap over events; every "
        "difference is a paired bootstrap.",
        "",
        f"Window basis: core position resolution (p{payload['core_pct']:g} of "
        f"|dz| over one-to-one matched pairs) = **{core['pvf'] * 1000:.1f} um** "
        f"(PV-Finder) and **{core['amvf'] * 1000:.1f} um** (AMVF). Nearly equal, "
        f"so {payload['core_multiple']:g}x the worse of the two — "
        f"**{wins[0]:.2f} mm** — is fair to both by construction. This is NOT "
        f"sigma_vtx-vtx ({a1['sigma_pvf_mm']:.4f} / {a1['sigma_amvf_mm']:.4f} mm), "
        "which measures two-vertex separation, differs by 40% between the "
        "algorithms, and is what the production eval uses.",
        "",
    ]
    for i, w in enumerate(wins):
        label = "PRIMARY" if i == 0 else "sensitivity point"
        lines += _table_block(heads[f"{w:g}"], w, label)

    h = heads[f"{wins[0]:g}"]
    lines += ["## Control — PV-Finder re-cut to AMVF's candidate multiplicity", ""]
    mm = payload.get("matched_multiplicity_control", {})
    if mm.get("applicable"):
        c, am, dm = mm["pvf"], mm["amvf"], mm["diff"]
        lines += [
            f"PV-Finder emits {h['pvf']['reco_per_evt']:.2f} candidates/event "
            f"against AMVF's {mm['amvf_candidates_per_evt']:.2f}. Raising the "
            f"peak-height floor to {mm['height_threshold']:.4f} equalises them "
            f"({mm['candidates_per_evt']:.2f}/evt), which is exactly equivalent "
            "to re-running the peak finder at that `--min-height`.",
            "",
            "| | efficiency [%] | strict [%] | fake/evt | fake+split/evt |",
            "|---|---|---|---|---|",
            f"| PV-Finder, re-cut | {100 * c['eff']:.2f} "
            f"| {100 * c['eff_strict']:.2f} | {c['fake_per_evt']:.2f} "
            f"| {c['surplus_per_evt']:.2f} |",
            f"| AMVF | {100 * am['eff']:.2f} | {100 * am['eff_strict']:.2f} "
            f"| {am['fake_per_evt']:.2f} | {am['surplus_per_evt']:.2f} |",
            f"| **difference** (paired) | {100 * dm['eff']:+.2f} ± "
            f"{100 * dm['eff_err']:.2f} | {100 * dm['eff_strict']:+.2f} ± "
            f"{100 * dm['eff_strict_err']:.2f} | {dm['fake_per_evt']:+.2f} ± "
            f"{dm['fake_per_evt_err']:.2f} | {dm['surplus_per_evt']:+.2f} ± "
            f"{dm['surplus_per_evt_err']:.2f} |",
            "",
        ]
    lines += [
        "## Asymmetry 1 — window taken from PV-Finder's own sigma_vtx-vtx",
        "",
        f"- sigma_PVF = {a1['sigma_pvf_mm']:.4f} mm, "
        f"sigma_AMVF = {a1['sigma_amvf_mm']:.4f} mm "
        f"({a1['window_tighter_by_pct']:.1f}% tighter than AMVF's own).",
        f"- AMVF fake rate {a1['amvf_fake_at_own_sigma']:.2f} -> "
        f"{a1['amvf_fake_at_pvf_sigma']:.2f}/evt when judged at PV-Finder's "
        f"sigma ({a1['d_amvf_fake']:+.2f}); AMVF efficiency "
        f"{a1['d_amvf_eff_pts']:+.2f} pts.",
        f"- Efficiency gap: published {a1['gap_eff_pts_published']:+.2f} pts -> "
        f"common window {a1['gap_eff_pts_common']:+.2f} pts.",
        f"- Fake gap: published {a1['gap_fake_published']:+.2f}/evt -> "
        f"common window {a1['gap_fake_common']:+.2f}/evt.",
        "",
        "## Asymmetry 2 — nTrk == 1 truth interactions counted as fakes",
        "",
        "| | excused/evt | accidental floor | genuine | % of fakes |",
        "|---|---|---|---|---|",
    ]
    for arm, name in (("pvf", "PV-Finder"), ("amvf", "AMVF")):
        x = a2[arm]
        lines.append(
            f"| {name} | {x['excused_per_evt']:.2f} "
            f"| {x['accidental_floor_per_evt']:.2f} "
            f"(shift {x['accidental_floor_shift_per_evt']:.2f}) "
            f"| {x['genuine_per_evt']:.2f} ± {x['genuine_err_per_evt']:.2f} "
            f"| {100 * x['frac_of_fakes_genuine']:.1f}% |"
        )
    lines += [
        "",
        f"Net in PV-Finder's favour: **{a2['net_in_our_favour_per_evt']:.2f}/evt**"
        f", not the {a2['naive_if_amvf_immune_per_evt']:.2f}/evt it would be if "
        "AMVF were immune. AMVF is penalised by this convention too.",
        "",
        "## Crossings",
        "",
    ]
    for k, v in payload["crossings"].items():
        got = ", ".join(f"{x:.3f} mm" for x in v) if v else "none in 0.1-1.0 mm"
        lines.append(f"- `{k}`: {got}")
    return "\n".join(lines) + "\n"
