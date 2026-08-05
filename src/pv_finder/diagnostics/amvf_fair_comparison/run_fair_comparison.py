#!/usr/bin/env python3
"""Entry point for the fair PV-Finder / AMVF comparison.

Run it **by path**, not with ``python -m`` (see docs/evaluation/vertex_association.md,
"Known quirk" — the ``-m`` form has hung unkillably on sneezy):

    source venv/bin/activate
    python -u src/pv_finder/diagnostics/amvf_fair_comparison/run_fair_comparison.py \
        --cache outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz \
        --outdir outputs/08_05_2026_output/amvf_fair_comparison --tag r16443

Build the cache first with ``--build-cache --pkl ... --root ...``.  Nothing here
re-runs inference: PV-Finder's peak list is read out of an existing
``eval_results.pkl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[3]))

from pv_finder.diagnostics.amvf_fair_comparison.cache import (  # noqa: E402
    build_cache,
    load_cache,
)
from pv_finder.diagnostics.amvf_fair_comparison.scan import (  # noqa: E402
    HEADLINE_WINDOW_MM,
    accidental_efficiency,
    core_position_resolution,
    crossings,
    matched_multiplicity_view,
    per_event_counts,
    run_scan,
    self_sigma_view,
)
from pv_finder.diagnostics.amvf_fair_comparison.scan import (  # noqa: E402
    bootstrap_point as _bootstrap,
)


def _headline(data: dict, window: float, n_boot: int, seed: int) -> dict:
    """The four-cell table: {PV-Finder, AMVF} x {standard, corrected} truth."""
    counts = {
        a: per_event_counts(data["truth_ge2"], data[a], data["truth_eq1"], window)
        for a in ("pvf", "amvf")
    }
    res = _bootstrap(counts, n_boot=n_boot, seed=seed)
    res["window_mm"] = window
    return res


def _asymmetries(data: dict, scan: dict, self_view: dict, head: dict) -> dict:
    """What each asymmetry was worth, in efficiency points and fake/event."""
    s_pvf = self_view["sigma"]["pvf"]["sigma_mm"]
    s_amvf = self_view["sigma"]["amvf"]["sigma_mm"]
    pub = self_view["both_by_pvf"]  # what the eval publishes today
    own = self_view["each_by_own"]  # symmetric self-consistent windows

    # Asymmetry 1: AMVF judged at PV-Finder's sigma instead of its own.
    a1 = {
        "sigma_pvf_mm": s_pvf,
        "sigma_amvf_mm": s_amvf,
        "window_tighter_by_pct": 100.0 * (1.0 - s_pvf / s_amvf),
        "amvf_fake_at_pvf_sigma": pub["amvf"]["fake_per_evt"],
        "amvf_fake_at_own_sigma": own["amvf"]["fake_per_evt"],
        "amvf_eff_at_pvf_sigma": pub["amvf"]["eff"],
        "amvf_eff_at_own_sigma": own["amvf"]["eff"],
        "d_amvf_fake": pub["amvf"]["fake_per_evt"] - own["amvf"]["fake_per_evt"],
        "d_amvf_eff_pts": 100.0 * (pub["amvf"]["eff"] - own["amvf"]["eff"]),
        # The quantity that actually matters: how the *gap* moves between the
        # published presentation and the fixed common window.
        "gap_eff_pts_published": 100.0 * (pub["pvf"]["eff"] - pub["amvf"]["eff"]),
        "gap_eff_pts_common": 100.0 * (head["pvf"]["eff"] - head["amvf"]["eff"]),
        "gap_eff_strict_pts_published": 100.0
        * (pub["pvf"]["eff_strict"] - pub["amvf"]["eff_strict"]),
        "gap_eff_strict_pts_common": 100.0
        * (head["pvf"]["eff_strict"] - head["amvf"]["eff_strict"]),
        "gap_fake_published": pub["pvf"]["fake_per_evt"] - pub["amvf"]["fake_per_evt"],
        "gap_fake_common": head["pvf"]["fake_per_evt"] - head["amvf"]["fake_per_evt"],
        "gap_surplus_published": (
            pub["pvf"]["surplus_per_evt"] - pub["amvf"]["surplus_per_evt"]
        ),
        "gap_surplus_common": (
            head["pvf"]["surplus_per_evt"] - head["amvf"]["surplus_per_evt"]
        ),
    }

    # Asymmetry 2: reco vertices on real nTrk == 1 truth interactions.
    a2 = {}
    for arm in ("pvf", "amvf"):
        h = head[arm]
        a2[arm] = {
            "excused_per_evt": h["excused_per_evt"],
            "accidental_floor_per_evt": h["excused_ctrl_per_evt"],
            "accidental_floor_shift_per_evt": h["excused_ctrl_shift_per_evt"],
            "genuine_per_evt": h["excused_net_per_evt"],
            "genuine_err_per_evt": h["excused_net_per_evt_err"],
            "frac_of_fakes_genuine": (
                h["excused_net_per_evt"] / h["fake_per_evt"]
                if h["fake_per_evt"]
                else 0.0
            ),
        }
    a2["net_in_our_favour_per_evt"] = (
        a2["pvf"]["genuine_per_evt"] - a2["amvf"]["genuine_per_evt"]
    )
    a2["naive_if_amvf_immune_per_evt"] = a2["pvf"]["genuine_per_evt"]

    return {
        "asymmetry_1_window": a1,
        "asymmetry_2_low_ntrk_truth": a2,
        "crossings": {
            "eff": crossings(scan["rows"], "eff"),
            "eff_strict": crossings(scan["rows"], "eff_strict"),
            "fake_per_evt": crossings(scan["rows"], "fake_per_evt"),
            "fake_corr_per_evt": crossings(scan["rows"], "fake_corr_per_evt"),
            "surplus_per_evt": crossings(scan["rows"], "surplus_per_evt"),
        },
    }


def main(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.build_cache:
        build_cache(
            args.pkl, args.root, args.cache,
            mu_min=args.mu_min, mu_max=args.mu_max, tag=args.tag,
        )  # fmt: skip

    print(f"\n=== Fair comparison: {args.tag} ===")
    data = load_cache(args.cache)
    meta = data["meta"]
    print(
        f"  {meta['n_events']} events, <mu>={meta['mu_mean']:.1f}; per event: "
        f"PVF {meta['pvf_per_evt']:.2f}, AMVF {meta['amvf_per_evt']:.2f}, "
        f"truth(nTrk>=2) {meta['truth_ge2_per_evt']:.2f}, "
        f"truth(nTrk==1) {meta['truth_eq1_per_evt']:.2f}"
    )

    # --- Window basis: core POSITION resolution, not sigma_vtx-vtx ---
    # sigma_vtx-vtx is a two-vertex separation scale and differs by 40 % between
    # the two algorithms, so any window built on it is unfair by construction.
    # The core position resolution is what "did it find the vertex" needs, and
    # it is nearly equal for both, so a window built on it is fair to both.
    print("\n--- Core position resolution (the window basis) ---")
    core = {
        a: core_position_resolution(data["truth_ge2"], data[a], pct=args.core_pct)
        for a in ("pvf", "amvf")
    }
    # The greedy estimator is recorded too: it is what the production matcher
    # would give, and it is ~10 % tighter because closest-first creams off the
    # core.  A tighter window flatters PV-Finder, so the conservative (optimal)
    # value is the one used.
    core_greedy = {
        a: core_position_resolution(
            data["truth_ge2"], data[a], pct=args.core_pct, optimal=False
        )
        for a in ("pvf", "amvf")
    }
    for a in ("pvf", "amvf"):
        print(
            f"  {a.upper():5s} core p{args.core_pct:g} |dz| = {core[a]:.4f} mm "
            f"(optimal assignment; greedy would give {core_greedy[a]:.4f})"
        )
    w_res = args.core_multiple * max(core.values())
    print(
        f"  -> resolution-derived common window = {args.core_multiple:g} x "
        f"{max(core.values()):.4f} = {w_res:.4f} mm  (primary)"
    )

    windows = np.round(np.arange(args.w_min, args.w_max + 1e-9, args.w_step), 4)
    print(f"\n--- Window scan ({len(windows)} points, common window both arms) ---")
    scan = run_scan(data, windows, n_boot=args.n_boot, seed=args.seed)

    # Primary headline at the resolution-derived window; 0.5 mm kept as a
    # labelled sensitivity point, not as the primary number.
    head_windows = [round(w_res, 4)] + list(args.sensitivity_windows)
    heads = {}
    for i, w in enumerate(head_windows):
        label = "PRIMARY (resolution-derived)" if i == 0 else "sensitivity"
        print(f"\n--- Headline at {w:g} mm — {label} ---")
        hh = _headline(data, float(w), args.n_boot_headline, args.seed)
        hh["accidental"] = {
            a: accidental_efficiency(data["truth_ge2"], data[a], float(w))
            for a in ("pvf", "amvf")
        }
        heads[f"{w:g}"] = hh
        for arm in ("pvf", "amvf"):
            h = hh[arm]
            print(
                f"  {arm.upper():5s} eff={h['eff']:.4f}+-{h['eff_err']:.4f}  "
                f"eff_strict={h['eff_strict']:.4f}+-{h['eff_strict_err']:.4f}  "
                f"fake={h['fake_per_evt']:.3f}+-{h['fake_per_evt_err']:.3f}  "
                f"fake_corr={h['fake_corr_per_evt']:.3f}  "
                f"surplus={h['surplus_per_evt']:.3f}"
                f"+-{h['surplus_per_evt_err']:.3f}  "
                f"accidental_eff={100 * hh['accidental'][arm]['eff']:.1f}%"
                f"/{100 * hh['accidental'][arm]['eff_strict']:.1f}% strict"
            )
        dd = hh["diff"]
        print(
            f"  PVF-AMVF (paired): d_eff={100 * dd['eff']:+.3f}"
            f"+-{100 * dd['eff_err']:.3f} pts  d_eff_strict="
            f"{100 * dd['eff_strict']:+.3f}+-{100 * dd['eff_strict_err']:.3f} pts  "
            f"d_fake={dd['fake_per_evt']:+.3f}+-{dd['fake_per_evt_err']:.3f}  "
            f"d_surplus={dd['surplus_per_evt']:+.3f}"
            f"+-{dd['surplus_per_evt_err']:.3f}"
        )
    head = heads[f"{head_windows[0]:g}"]

    print("\n--- Self-consistent (own-sigma) view, circular; secondary only ---")
    self_view = self_sigma_view(data, n_boot=args.n_boot_headline, seed=args.seed)
    for arm in ("pvf", "amvf"):
        s = self_view["sigma"][arm]
        print(
            f"  sigma_{arm} = {s['sigma_mm']:.4f} +- {s['err_mm']:.4f} mm "
            f"({s['n_pairs']:,} pairs)"
        )

    print("\n--- Control: PV-Finder re-cut to AMVF's candidate multiplicity ---")
    mm_view = matched_multiplicity_view(
        data, float(head_windows[0]), n_boot=args.n_boot_headline, seed=args.seed
    )
    if mm_view.get("applicable"):
        c, a, dm = mm_view["pvf"], mm_view["amvf"], mm_view["diff"]
        print(
            f"  height floor {mm_view['height_threshold']:.4f} -> "
            f"{mm_view['candidates_per_evt']:.2f} cand/evt "
            f"(AMVF {mm_view['amvf_candidates_per_evt']:.2f})"
        )
        print(
            f"  PVF(cut) eff={c['eff']:.4f} fake={c['fake_per_evt']:.3f} "
            f"surplus={c['surplus_per_evt']:.3f}   vs AMVF eff={a['eff']:.4f} "
            f"fake={a['fake_per_evt']:.3f} surplus={a['surplus_per_evt']:.3f}"
        )
        print(
            f"  -> at equal multiplicity (paired): "
            f"d_eff={100 * dm['eff']:+.2f}+-{100 * dm['eff_err']:.2f} pts, "
            f"d_fake={dm['fake_per_evt']:+.2f}+-{dm['fake_per_evt_err']:.2f}, "
            f"d_surplus={dm['surplus_per_evt']:+.2f}"
            f"+-{dm['surplus_per_evt_err']:.2f}"
        )

    asym = _asymmetries(data, scan, self_view, head)
    a1, a2 = asym["asymmetry_1_window"], asym["asymmetry_2_low_ntrk_truth"]
    print("\n--- Asymmetry 1: AMVF judged at PV-Finder's sigma ---")
    print(
        f"  window is {a1['window_tighter_by_pct']:.1f}% tighter than AMVF's own; "
        f"AMVF fake {a1['amvf_fake_at_own_sigma']:.2f} -> "
        f"{a1['amvf_fake_at_pvf_sigma']:.2f}/evt "
        f"({a1['d_amvf_fake']:+.2f}), AMVF eff {a1['d_amvf_eff_pts']:+.2f} pts"
    )
    print(
        f"  efficiency gap  published {a1['gap_eff_pts_published']:+.2f} pts "
        f"-> common window {a1['gap_eff_pts_common']:+.2f} pts"
    )
    print(
        f"  fake gap        published {a1['gap_fake_published']:+.2f}/evt "
        f"-> common window {a1['gap_fake_common']:+.2f}/evt"
    )
    print("\n--- Asymmetry 2: reco on real nTrk==1 truth ---")
    for arm in ("pvf", "amvf"):
        x = a2[arm]
        print(
            f"  {arm.upper():5s} excused={x['excused_per_evt']:.3f}/evt  "
            f"accidental floor={x['accidental_floor_per_evt']:.3f} "
            f"(shift control {x['accidental_floor_shift_per_evt']:.3f})  "
            f"genuine={x['genuine_per_evt']:.3f}"
            f"+-{x['genuine_err_per_evt']:.3f}/evt "
            f"({100 * x['frac_of_fakes_genuine']:.1f}% of fakes)"
        )
    print(
        f"  net in our favour = {a2['net_in_our_favour_per_evt']:.3f}/evt "
        f"(NOT {a2['naive_if_amvf_immune_per_evt']:.3f} — AMVF is penalised too)"
    )
    print("\n--- Crossings (window where PVF and AMVF are equal) ---")
    for k, v in asym["crossings"].items():
        where = ", ".join(f"{x:.3f} mm" for x in v) if v else "none in range"
        print(f"  {k:20s}: {where}")

    payload = {
        "meta": meta,
        "headline_window_mm": float(head_windows[0]),
        "headline_windows": [float(x) for x in head_windows],
        "core_position_resolution_mm": core,
        "core_position_resolution_greedy_mm": core_greedy,
        "core_multiple": args.core_multiple,
        "core_pct": args.core_pct,
        "headlines": heads,
        "n_boot": args.n_boot,
        "n_boot_headline": args.n_boot_headline,
        "seed": args.seed,
        "headline": head,
        "matched_multiplicity_control": mm_view,
        "self_sigma_view": self_view,
        "scan": scan,
        **asym,
    }
    jpath = outdir / f"results_{args.tag}.json"
    with open(jpath, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\n  wrote {jpath}")

    if not args.no_plot:
        from pv_finder.diagnostics.amvf_fair_comparison.plots import (
            headline_table_md,
            plot_window_scan,
        )

        fig = outdir / f"window_scan_{args.tag}.png"
        plot_window_scan(payload, fig)
        tbl = outdir / f"headline_table_{args.tag}.md"
        tbl.write_text(headline_table_md(payload))
        print(f"  wrote {fig}\n  wrote {tbl}")


def _cli() -> argparse.Namespace:
    a = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    a.add_argument("--cache", required=True)
    a.add_argument("--outdir", default="outputs/08_05_2026_output/amvf_fair_comparison")
    a.add_argument("--tag", default="r16443")
    a.add_argument("--build-cache", action="store_true")
    a.add_argument("--pkl", default=None)
    a.add_argument("--root", default=None)
    a.add_argument("--mu-min", type=float, default=185.0)
    a.add_argument("--mu-max", type=float, default=215.0)
    a.add_argument("--w-min", type=float, default=0.10)
    a.add_argument("--w-max", type=float, default=1.00)
    a.add_argument("--w-step", type=float, default=0.05)
    a.add_argument("--core-multiple", type=float, default=3.0,
                   help="common window = this x the worse core position "
                        "resolution of the two algorithms")  # fmt: skip
    a.add_argument("--core-pct", type=float, default=68.0)
    a.add_argument("--sensitivity-windows", type=float, nargs="*",
                   default=[HEADLINE_WINDOW_MM],
                   help="extra windows to tabulate as labelled sensitivity "
                        "points (0.5 mm is the round-number choice)")  # fmt: skip
    a.add_argument("--n-boot", type=int, default=400, help="replicas per scan point")
    a.add_argument("--n-boot-headline", type=int, default=2000)
    a.add_argument("--seed", type=int, default=20260805)
    a.add_argument("--no-plot", action="store_true")
    return a.parse_args()


if __name__ == "__main__":
    main(_cli())
