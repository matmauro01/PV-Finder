#!/usr/bin/env python3
"""Independent re-derivation of the PV-Finder vs AMVF comparison.

Audit of ``evaluation/vertex_finding/run_eval_pvf_run3.py`` (2026-08-05).  The
numbers here are built from the ROOT ntuple and the stored peak lists with a
matcher written from scratch (``diagnostics.fair_matching``); no part of the
evaluation's matching or category code is imported, because that code is the
subject of the audit.

What it measures, for PV-Finder and AMVF on the *same* events, at the *same*
window, against the *same* truth list:

1. a scan of efficiency and fake rate against matching window, so no
   conclusion depends on one window choice;
2. both truth definitions, nTracks>=2 (what the evaluation uses) and
   nTracks>=1 (which is what PV-Finder's fakes are actually scored against);
3. both efficiency conventions -- strict one-to-one, and the evaluation's
   "one reco may be credited with several truths" rule -- so the published
   number is reproduced *and* given the AMVF counterpart it never had;
4. controls: accidental-match rate, equal-reco-count operating point,
   greedy vs optimal matching, and AMVF without its nTracks>=2 grooming.

Run::

    python -u src/pv_finder/diagnostics/amvf_fairness_audit.py \
        --root  <ntuple.root> --pkl <eval_results.pkl> \
        --out   outputs/08_05_2026_output/amvf_fairness_audit
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))
from pv_finder.diagnostics.fair_matching import (  # noqa: E402
    MatchResult,
    Summary,
    bootstrap_paired,
    match_greedy,
    match_optimal,
    summarise,
)

WINDOWS = [0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00, 1.50, 2.00]
ACCIDENTAL_SHIFT_MM = 3.0  # >> resolution, << beam sigma_z, so density is preserved


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_root(path: str, n_events: int) -> dict:
    """Read the vertex-level branches straight from the ntuple."""
    import uproot

    tree = uproot.open(path)["PVFinderData"]
    scal = ["ActualNumOfInt", "BeamPosZ"]
    jag = ["RecoVertex_z", "RecoVertex_nTracks", "TruthVertex_z", "TruthVertex_nTracks"]
    out: dict[str, list] = {k: [] for k in scal + jag}
    n = 0
    for chunk in tree.iterate(
        scal + jag, library="np", step_size=500, entry_start=0, entry_stop=n_events
    ):
        m = len(chunk["ActualNumOfInt"])
        for k in scal + jag:
            out[k].extend(chunk[k][i] for i in range(m))
        n += m
    res: dict = {k: np.asarray(out[k]) for k in scal}
    for k in jag:
        arr = np.empty(len(out[k]), dtype=object)
        for i, v in enumerate(out[k]):
            arr[i] = np.asarray(v, dtype=np.float64)
        res[k] = arr
    print(f"[audit] ROOT: {n} entries read from {Path(path).name}")
    return res


def load_peaks(path: str) -> tuple[list[np.ndarray], list[np.ndarray], list[dict]]:
    """Pull PV-Finder's stored peak positions and heights out of the eval pkl."""
    with open(path, "rb") as f:
        r = pickle.load(f)
    pred = [np.asarray(a, dtype=np.float64) for a in r["pred_pvs_mm"]]
    hts = [np.asarray(a, dtype=np.float64) for a in r["pred_heights"]]
    print(
        f"[audit] pkl: {len(pred)} peak lists, sigma_used={r['sigma_vtx_vtx_mm']:.4f} mm"
    )
    return pred, hts, r["per_event"]


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select_events(R: dict, mu_min: float, mu_max: float) -> np.ndarray:
    """Events in the mu window with at least one nTracks>=2 truth vertex.

    Reproduces ``utils.pairwise_dz.in_summary_window`` from its documented
    definition rather than by importing it, so a change there cannot silently
    move the audit's event list.
    """
    mu = R["ActualNumOfInt"]
    n_t2 = np.array([int((x >= 2).sum()) for x in R["TruthVertex_nTracks"]])
    keep = np.array([mu_min <= round(float(m)) <= mu_max for m in mu]) & (n_t2 > 0)
    return np.where(keep)[0]


def truth_of(R: dict, i: int, min_ntrk: int) -> np.ndarray:
    """Truth vertex z for event ``i`` at a given nTracks threshold."""
    return R["TruthVertex_z"][i][R["TruthVertex_nTracks"][i] >= min_ntrk]


def amvf_of(R: dict, i: int, min_ntrk: int = 2) -> np.ndarray:
    """AMVF vertex z for event ``i``; ``min_ntrk=1`` is the ungroomed list."""
    return R["RecoVertex_z"][i][R["RecoVertex_nTracks"][i] >= min_ntrk]


# --------------------------------------------------------------------------
# the evaluation's efficiency convention, re-implemented
# --------------------------------------------------------------------------


def absorb_efficiency(
    truth_z: np.ndarray, reco_z: np.ndarray, window: float, m: MatchResult
) -> int:
    """Truths counted as found under the evaluation's 'merged' rule.

    ``run_eval_pvf_run3.py`` reports efficiency as ``(clean + merged)/n_truth``
    where a reco already assigned to one truth additionally *claims* every
    other unmatched truth inside its own window.  One reco can therefore be
    credited with several truth vertices.  Re-implemented here so the published
    PV-Finder number can be reproduced and, more to the point, so the identical
    rule can be applied to AMVF -- which the evaluation never does.
    """
    if m.n_matched == 0:
        return 0
    claimed = set(m.pairs[:, 0].tolist())
    for _ti, ri in m.pairs:
        near = np.where(np.abs(truth_z - reco_z[ri]) <= window)[0]
        claimed.update(int(t) for t in near)
    return len(claimed)


# --------------------------------------------------------------------------
# core sweep
# --------------------------------------------------------------------------


def run_algo(
    truth: list[np.ndarray], reco: list[np.ndarray], window: float, greedy: bool = False
) -> list[MatchResult]:
    """Match every event for one algorithm at one window."""
    fn = match_greedy if greedy else match_optimal
    return [fn(t, r, window) for t, r in zip(truth, reco)]


def fmt(s: Summary) -> str:
    """One-line rendering of a Summary."""
    return (
        f"eff={s.efficiency:.4f} fake={s.fake_per_evt:6.3f}/evt "
        f"(excl-split {s.fake_per_evt_excl_split:6.3f}, split {s.split_per_evt:.3f}) "
        f"truth={s.truth_per_evt:6.2f} reco={s.reco_per_evt:6.2f}"
    )


def core_sigma(results: list[MatchResult]) -> float:
    """68th percentile of |reco - truth| over matched pairs, in mm."""
    r = np.concatenate([m.residuals for m in results if len(m.residuals)])
    return float(np.percentile(np.abs(r), 68))


def main(args: argparse.Namespace) -> None:  # noqa: PLR0915, C901
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = load_root(args.root, args.n_events)
    pred, hts, per_event = load_peaks(args.pkl)
    idx = select_events(R, args.mu_min, args.mu_max)
    print(f"[audit] selected {len(idx)} events, mu in [{args.mu_min},{args.mu_max}]")

    # --- alignment: the pkl is indexed by ROOT entry; prove it before using it
    bad = 0
    for i in idx:
        if int(per_event[i]["n_truth"]) != len(truth_of(R, i, 2)):
            bad += 1
        if int(per_event[i]["event_idx"]) != int(i):
            bad += 1
    print(f"[audit] alignment mismatches (pkl n_truth vs ROOT nTrk>=2): {bad}")
    if bad:
        raise SystemExit("pkl and ROOT are not aligned; aborting")

    reco_sets = {
        "PVF": [pred[i] for i in idx],
        "AMVF": [amvf_of(R, i, 2) for i in idx],
    }
    record: dict = {"n_events": len(idx), "scan": {}, "headline": {}, "controls": {}}

    # --- validation: reproduce the published run from first principles -----
    # Greedy matching + the absorb convention + fake-excluding-split, i.e. every
    # convention the evaluation uses, at the window it actually used.  If these
    # do not land on the published numbers then the audit is measuring something
    # else and nothing below can be trusted.
    truth_v = [truth_of(R, i, 2) for i in idx]
    tot_v = sum(len(t) for t in truth_v)
    print(
        f"\n{'=' * 78}\n  VALIDATION -- evaluation's own conventions at its own "
        f"window ({args.published_window:.4f} mm)\n{'=' * 78}"
    )
    record["validation"] = {}
    for k, v in reco_sets.items():
        ms = [match_greedy(t, r, args.published_window) for t, r in zip(truth_v, v)]
        s = summarise(ms)
        got = sum(
            absorb_efficiency(t, r, args.published_window, m)
            for t, r, m in zip(truth_v, v, ms)
        )
        print(
            f"  {k:<5} eff(absorb)={got / tot_v:.4f}  eff(strict)={s.efficiency:.4f}  "
            f"fake(excl-split)={s.fake_per_evt_excl_split:6.3f}  "
            f"fake(all)={s.fake_per_evt:6.3f}  split={s.split_per_evt:.3f}  "
            f"reco={s.reco_per_evt:6.2f}"
        )
        record["validation"][k] = dict(absorb_eff=got / tot_v, **s._asdict())
    print(
        "  published log for comparison: PVF eff=0.8648 FP=16.597/evt; "
        "AMVF fake=17.78/evt (no AMVF efficiency is printed anywhere)"
    )

    # --- window scan, both truth definitions ------------------------------
    for min_ntrk in (2, 1):
        truth = [truth_of(R, i, min_ntrk) for i in idx]
        print(
            f"\n{'=' * 78}\n  WINDOW SCAN -- truth = TruthVertex nTracks>={min_ntrk}"
            f"  ({np.mean([len(t) for t in truth]):.2f}/evt)\n{'=' * 78}"
        )
        print(
            f"{'win/mm':>7} | {'PVF eff':>8} {'PVF fake':>9} | {'AMVF eff':>8} "
            f"{'AMVF fake':>9} | {'dEff':>7} {'dFake':>7}"
        )
        rows = []
        for w in WINDOWS:
            s = {k: summarise(run_algo(truth, v, w)) for k, v in reco_sets.items()}
            p, a = s["PVF"], s["AMVF"]
            print(
                f"{w:7.2f} | {p.efficiency:8.4f} {p.fake_per_evt:9.3f} | "
                f"{a.efficiency:8.4f} {a.fake_per_evt:9.3f} | "
                f"{p.efficiency - a.efficiency:+7.4f} "
                f"{p.fake_per_evt - a.fake_per_evt:+7.3f}"
            )
            rows.append(dict(window=w, pvf=p._asdict(), amvf=a._asdict()))
        record["scan"][f"ntrk>={min_ntrk}"] = rows

    # --- window choice, derived from both algorithms' residuals -----------
    truth2 = [truth_of(R, i, 2) for i in idx]
    sig = {k: core_sigma(run_algo(truth2, v, 1.0)) for k, v in reco_sets.items()}
    w_head = round(3.0 * max(sig.values()), 2)
    print(
        "\n[audit] core sigma (68th pct |dz| of matched pairs, window 1.0 mm): "
        + ", ".join(f"{k}={v:.4f} mm" for k, v in sig.items())
    )
    print(f"[audit] headline window = 3 x max = {w_head:.2f} mm")
    record["core_sigma_mm"] = sig
    record["headline_window_mm"] = w_head

    # --- headline, both truth definitions, with paired bootstrap ----------
    for min_ntrk in (2, 1):
        truth = [truth_of(R, i, min_ntrk) for i in idx]
        res = {k: run_algo(truth, v, w_head) for k, v in reco_sets.items()}
        print(
            f"\n{'=' * 78}\n  HEADLINE -- window {w_head:.2f} mm, "
            f"truth nTracks>={min_ntrk}, {len(idx)} events\n{'=' * 78}"
        )
        for k, v in res.items():
            print(f"  {k:<5} {fmt(summarise(v))}")
        bs = bootstrap_paired(res, n_boot=args.n_boot)
        for k in ("PVF", "AMVF", "__diff__"):
            e, ee = bs[k]["efficiency"]
            f, fe = bs[k]["fake_per_evt"]
            print(
                f"  {k:<9} eff = {e:+.4f} +/- {ee:.4f}   "
                f"fake = {f:+.3f} +/- {fe:.3f} /evt"
            )
        # evaluation's inflating convention, applied to BOTH
        print("  -- evaluation's 'merged counts as found' convention --")
        for k, v in reco_sets.items():
            tot = sum(len(t) for t in truth)
            got = sum(
                absorb_efficiency(t, r, w_head, m) for t, r, m in zip(truth, v, res[k])
            )
            print(
                f"  {k:<5} eff(absorb) = {got / tot:.4f}   "
                f"(strict {summarise(res[k]).efficiency:.4f})"
            )
            record["headline"].setdefault(f"ntrk>={min_ntrk}", {})[k] = dict(
                strict=summarise(res[k])._asdict(), absorb_eff=got / tot
            )

    # --- controls ---------------------------------------------------------
    print(
        f"\n{'=' * 78}\n  CONTROLS (truth nTracks>=2, window {w_head:.2f} mm)"
        f"\n{'=' * 78}"
    )
    truth = truth2

    # accidental matching: displace reco far enough to break the association
    acc = {}
    for k, v in reco_sets.items():
        shifted = [r + ACCIDENTAL_SHIFT_MM for r in v]
        acc[k] = summarise(run_algo(truth, shifted, w_head)).efficiency
        print(
            f"  accidental eff ({k}, reco shifted +{ACCIDENTAL_SHIFT_MM} mm): {acc[k]:.4f}"
        )
    record["controls"]["accidental_eff"] = acc

    # greedy (the evaluation's rule) vs optimal
    for k, v in reco_sets.items():
        g = summarise(run_algo(truth, v, w_head, greedy=True))
        o = summarise(run_algo(truth, v, w_head))
        print(
            f"  greedy vs optimal ({k}): eff {g.efficiency:.4f} -> {o.efficiency:.4f} "
            f"({o.efficiency - g.efficiency:+.4f})"
        )

    # AMVF without its nTracks>=2 grooming
    ung = [amvf_of(R, i, 1) for i in idx]
    s_g = summarise(run_algo(truth, reco_sets["AMVF"], w_head))
    s_u = summarise(run_algo(truth, ung, w_head))
    print(f"  AMVF groomed   : {fmt(s_g)}")
    print(f"  AMVF ungroomed : {fmt(s_u)}")
    record["controls"]["amvf_ungroomed"] = s_u._asdict()

    # equal-reco-count operating point: raise PV-Finder's height floor until it
    # emits as many vertices as AMVF, so the two are compared at equal yield
    target = s_g.reco_per_evt
    lo, hi = 0.0, 5.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        n = np.mean([float((hts[i] >= mid).sum()) for i in idx])
        if n > target:
            lo = mid
        else:
            hi = mid
    cut = 0.5 * (lo + hi)
    pv_eq = [pred[i][hts[i] >= cut] for i in idx]
    s_eq = summarise(run_algo(truth, pv_eq, w_head))
    print(f"  PVF at equal reco count (height floor {cut:.4f}): {fmt(s_eq)}")
    print(f"  AMVF                                            : {fmt(s_g)}")
    record["controls"]["pvf_equal_count"] = dict(height_cut=cut, **s_eq._asdict())

    # sub-event boundary: PV-Finder stitches 12 x 40 mm blocks
    print("\n  -- sub-event boundary (PV-Finder only, 40 mm blocks) --")

    def edge(phase: np.ndarray) -> np.ndarray:
        """True for positions within 1 mm of a 40 mm sub-event boundary."""
        return np.minimum(phase, 40.0 - phase) < 1.0

    phase_t = np.concatenate([(t + 240.0) % 40.0 for t in truth])
    res_pv = run_algo(truth, reco_sets["PVF"], w_head)
    missed_ph = np.concatenate(
        [
            ((t + 240.0) % 40.0)[np.setdiff1d(np.arange(len(t)), m.pairs[:, 0])]
            for t, m in zip(truth, res_pv)
        ]
    )
    print(f"  truth within 1 mm of a block edge: {100 * edge(phase_t).mean():.2f}%")
    print(f"  missed truth within 1 mm of edge : {100 * edge(missed_ph).mean():.2f}%")
    eff_edge = 1 - edge(missed_ph).sum() / max(edge(phase_t).sum(), 1)
    eff_bulk = 1 - (~edge(missed_ph)).sum() / max((~edge(phase_t)).sum(), 1)
    print(
        f"  PVF eff at edges {eff_edge:.4f}  vs bulk {eff_bulk:.4f} "
        f"({eff_edge - eff_bulk:+.4f})"
    )
    record["controls"]["subevent_edge"] = dict(eff_edge=eff_edge, eff_bulk=eff_bulk)

    with open(out / "audit_results.json", "w") as f:
        json.dump(record, f, indent=2, default=float)
    print(f"\n[audit] wrote {out / 'audit_results.json'}")


def _cli() -> argparse.Namespace:
    a = argparse.ArgumentParser()
    a.add_argument("--root", required=True)
    a.add_argument("--pkl", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n-events", type=int, default=25000)
    a.add_argument("--mu-min", type=float, default=185)
    a.add_argument("--mu-max", type=float, default=215)
    a.add_argument("--n-boot", type=int, default=400)
    a.add_argument(
        "--published-window",
        type=float,
        default=0.2200,
        help="the sigma_vtx-vtx the audited run fed back as its "
        "matching window, used only to reproduce it",
    )
    return a.parse_args()


if __name__ == "__main__":
    main(_cli())
