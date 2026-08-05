# Is the PV-Finder vs AMVF comparison apples-to-apples?

An adversarial audit of the comparison produced by
`src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py`, done on
2026-08-05 against the r16443 held-out slice (1920 events, μ ∈ [185, 215]).

This page answers one question: **which parts of that comparison favour one
algorithm over the other, and by how much.** It complements
[`metric_definitions.md`](metric_definitions.md), which defines what the
metrics *are*; this page asks whether they are applied symmetrically.

Everything here was re-derived from the ROOT ntuple and the stored peak lists
with a matcher written from scratch
(`src/pv_finder/diagnostics/fair_matching.py`), deliberately not reusing
`compare_res_reco`, since that code is the subject of the audit.

Reproduce with:

```bash
python -u src/pv_finder/diagnostics/amvf_fairness_audit.py \
  --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
  --pkl  outputs/08_05_2026_output/eval_v6_operating_point/r16443/eval_results.pkl \
  --out  outputs/08_05_2026_output/amvf_fairness_audit
```

## 0. The audit reproduces the published run exactly

Applying the evaluation's own conventions (greedy matching, "merged counts as
found", fake-excluding-split) at its own window of 0.2200 mm:

| | eff (published convention) | fake/evt (published convention) | reco/evt |
|---|---|---|---|
| PV-Finder, this audit | **0.8648** | **16.598** | 101.01 |
| PV-Finder, published log | 0.8648 | 16.597 | 101.01 |
| AMVF, this audit | **0.8349** | 17.777 | 97.87 |
| AMVF, published log | *never printed* | 17.78 | 97.87 |

Agreement to four decimals on an independent implementation. Every number
below rests on that.

## 1. The ranked table

"Worth" is measured at the audit's common window unless stated. Directions are
relative to the published comparison.

| # | Asymmetry | Where | Favours | Worth |
|---|---|---|---|---|
| 1 | **Efficiency credits one reco with several truths.** `eff = (tc+tm)/n_truth`, and an assigned reco *claims* every unmatched truth in its window. Not a bug — it is close to the ATLAS "merged" convention — but it is a large inflation of the absolute number. | `run_eval_pvf_run3.py:532`; rule at `efficiency_res_optimized_atlas.py:292-296` | inflates the absolute number for both; only PV-Finder's is published | **+11.5 eff pts** (PVF 0.7501 → 0.8648); AMVF +11.9 (0.7161 → 0.8349) |
| 2 | **No AMVF efficiency exists.** The AMVF call discards the truth-side classification, so only reco-side categories are printed. The headline efficiency has no counterpart in any log. | `run_eval_pvf_run3.py:550` (`res_a, _, _`) | PV-Finder, by omission | the missing number: **AMVF 0.8349** vs PVF 0.8648 → margin **+2.99 pts**, not "86.5 % vs nothing" |
| 3 | **Matching window is PV-Finder's σ, applied to AMVF.** σ_vtx-vtx = 0.2200 mm from our own peaks; AMVF's own σ is 0.3048 mm. | `run_eval_pvf_run3.py:492`, used at `:518,527,551` | PV-Finder | AMVF at its own σ: eff 0.7165 → 0.7324 (**+1.59 pts**), fake 18.10 → 16.33 (**−1.77/evt**). Does **not** reverse the ranking — see §2 |
| 4 | **nTrk=1 truth removed from truth, but peaks there still count as fakes.** 23.13 such vertices/evt exist. | `run3_io.py:157-163` via `_filter_amvf:46-55` | AMVF, relatively | moving to nTrk≥1 truth: PVF fake −4.78/evt, AMVF fake −3.54/evt → costs PVF **1.24 fake/evt more** than AMVF |
| 5 | **"Split" reco excluded from the published fake rate.** PV-Finder produces 2.4× more splits than AMVF, so the exclusion is worth more to us. | `run_eval_pvf_run3.py:536` (sums `reco_fake` only) | PV-Finder | split 0.913/evt (PVF) vs 0.373 (AMVF). Published gap 1.18/evt → honest gap **0.64/evt**: about **46 % of our fake-rate advantage is accounting** |
| 6 | **σ is fitted on a different peak list than the one scored.** The σ list uses `--integral-threshold-res` (0.5); scoring used `--integral-threshold 0.30`. | `run_eval_pvf_run3.py:352-359` | neutral | refit on the scored list + μ window: σ 0.2200 → 0.2182, ≈ **0.05 eff pts** |
| 7 | **The published artefact predates the μ-window σ fix now in the tree.** The stored pkl lacks `sigma_fit_selection`/`in_mu_window` and the log line carries no selection label, so σ was fitted over all 25 000 events read (⟨truth⟩ = 58/evt, mixed pileup) while the summary is quoted on the 1920 PU200 events. | stored pkl vs `run_eval_pvf_run3.py:446-466` | neutral, small here | ≈ **0.05 eff pts**; but the published result is not reproducible from the current tree |
| 8 | **AMVF groomed to nTracks ≥ 2, PV-Finder not.** Real asymmetry *of kind* — AMVF is filtered on track count, we are filtered on amplitude. Numerically AMVF has almost nothing to groom. | `run3_io.py:46-55,149-152` | AMVF | 52 vertices in 1920 events = **0.03/evt**. Ungroomed: fake 19.819 → 19.834, eff 0.7011 → 0.7012. **Checked, negligible** |

### Checked and clean

| Candidate | Result |
|---|---|
| `--min-amvf-vtx` / `--min-tracks` event selection | **0 of 25 000 events removed**; 0 events with zero truth. No differential selection. |
| Beam-spot correction | In the MC-truth path no correction is applied to truth, PV-Finder *or* AMVF; `--no-correct-beam` only affects the no-truth fallback. `BeamPosZ ≡ 0.0000` in every event of this file. (The pkl records `correct_beam: True`, which is misleading but inert.) |
| z-range acceptance | max \|z\| = 199.7 mm (truth), 199.5 mm (AMVF), against a ±240 mm range. **Zero** vertices lost by either side. |
| Greedy vs optimal matching | Optimal (Hungarian) vs the evaluation's greedy: PVF 0.7346 → 0.7351, AMVF 0.7010 → 0.7011. Symmetric and negligible. |
| Tie-breaking / duplicate vertices | 0 exact duplicates in 187,908 AMVF and 193,945 PV-Finder vertices (1 in truth). No tie ambiguity to exploit. |
| Pairwise-Δz shuffle (`run_eval_pvf_run3.py:375`, unseeded) | 25 independent sign randomisations over 612,192 pairs: σ = 0.2182 **± 0.0000** mm. No run-to-run variance. |
| Sub-event boundary (12 × 40 mm stitching, PV-Finder only) | efficiency at block edges 0.7363 vs bulk 0.7351 (**+0.0012**). No boundary penalty. |
| Category rules applied asymmetrically | Same function, same window, same truth list for both. The asymmetry is in what is *printed* (row 2), not in how it is computed. |

## 2. Independent numbers

**Window choice.** Set from *both* algorithms, taking the worse: the core
position resolution (68th percentile of \|Δz\| over matched pairs at a generous
1.0 mm window) is 0.0551 mm for PV-Finder and 0.0565 mm for AMVF, so the
common window is 3 × 0.0565 ≈ **0.17 mm**. Note this is 4× *tighter* than
σ_vtx-vtx: that σ measures two-vertex *separation*, not single-vertex
position, and the two are not the same quantity.

Strict one-to-one matching, 1920 events, errors from a 400-draw paired
bootstrap over events:

**Truth = TruthVertex nTracks ≥ 2 (111.32/evt)**

| | efficiency | fake/evt |
|---|---|---|
| PV-Finder | 0.7351 ± 0.0008 | 19.179 ± 0.106 |
| AMVF | 0.7011 ± 0.0008 | 19.819 ± 0.103 |
| **difference** | **+0.0340 ± 0.0006** | **−0.640 ± 0.098** |

**Truth = TruthVertex nTracks ≥ 1 (134.45/evt)**

| | efficiency | fake/evt |
|---|---|---|
| PV-Finder | 0.6442 ± 0.0008 | 14.403 ± 0.093 |
| AMVF | 0.6068 ± 0.0008 | 16.280 ± 0.098 |
| **difference** | **+0.0373 ± 0.0005** | **−1.877 ± 0.090** |

**Robustness.** PV-Finder's efficiency exceeds AMVF's at every one of 12
windows from 0.10 to 2.00 mm, under both truth definitions, by +0.018 to
+0.038. The fake-rate ordering is *not* window-independent: PV-Finder has
fewer fakes for windows ≤ 0.5 mm (nTrk ≥ 2) or ≤ 1.0 mm (nTrk ≥ 1), and
**more** above that. Any fake-rate claim must quote its window.

**Equal-yield control.** PV-Finder emits 101.01 vertices/evt against AMVF's
97.87, and its operating point (`--min-height`, `--integral-threshold`) is
tuned while AMVF's is not. Raising the height floor to 0.0444 so both emit
97.87/evt: PV-Finder eff 0.7305, fake 16.548 — still **+2.94 pts** and
**−3.27 fake/evt** against AMVF. The margin is not an artefact of the tuning.

## 3. Two caveats on the absolute numbers

Neither is an asymmetry — both apply equally to both algorithms — but both
matter for how the efficiency should be read.

- **Accidental matching is large.** Displacing each reco list by +3 mm and
  rematching still "finds" 16.6 % (PV-Finder) and 16.4 % (AMVF) of truth at
  0.17 mm. At 111 truth/evt inside a 50 mm beam spot the local truth density
  is ≈ 0.9/mm, so a sizeable part of the absolute efficiency is coincidence.
  The two rates are equal to 0.2 pts, so the *comparison* is unaffected.
- **Some truth is unresolvable by anyone.** 11.26 adjacent truth pairs/evt sit
  closer than 0.17 mm (14.30/evt closer than 0.22 mm) — 10.1 % of all truth
  vertices. No algorithm reporting one vertex per peak can separate them.
  This is the physical reason the "merged" convention of row 1 exists.

## 4. Bottom line

The comparison is **not** apples-to-apples, and the corrections run in both
directions, but **none of them reverses the result**. Correcting everything
that can be corrected on the stored artefacts, PV-Finder is ahead of AMVF by
**+3.4 ± 0.1 efficiency points** and **0.64 ± 0.10 fewer fakes/event** at a
neutral common window — and still ahead at equal vertex yield.

What does change is the *absolute* headline. The published 86.5 % is 11.5
points of convention above a strict one-to-one count of 73.5 %, and the
published fake-rate advantage of 1.18/evt is 0.64/evt once splits are counted
as the unmatched vertices they are. **Quote the margin, not the absolute
efficiency, and always quote the window.**
