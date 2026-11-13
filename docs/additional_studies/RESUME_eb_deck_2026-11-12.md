# RESUME HERE (EB deck feedback, 12-13 Nov 2026)

If a session restarts: read this file, `meeting_2026-11-11_plan.md`, and the
"Status" table below; continue the first item not marked done. Update the
status table as items finish, and commit + push this file each time.

## State at 12 Nov 2026, ~23:00 EST
- Frozen (working_points.py, commit a7cb5fa): HL-LHC s8 (0.01/3/0.184/0.0184/centroid ±3),
  GNN v2 e527, **t = 0.94 both regimes** (changed from 0.97 after the >= 2-track rescan,
  outputs/11_12_2026_output/hllhc_final/t_rescan_min2/t_rescan_min2.md). Low pile-up e400
  (0.01/3/0.50/none/±3), mu60 v2 e115. Window 0.17 mm, HS 0.1 mm. Post-GNN vertex rule:
  >= 2 tracks, both algorithms.
- HL-LHC numbers: docs/evaluation/hllhc_final_numbers_2026-11-12.md (at t = 0.97; being
  redone at 0.94). Low pile-up: docs/evaluation/lowpu_final_numbers_2026-11-12.md (t = 0.97
  headline; 0.94 values in outputs/11_12_2026_output/lowpu_final/gnn_mc_ntrk2/summary.json).
- EB deck: presentations/mattia/11_13_2026_EB/ (numbers.tex from scripts/eb_numbers.py,
  build.sh). An agent was regenerating it at t = 0.94; check `git log -- presentations/mattia/11_13_2026_EB`
  and `git status` for uncommitted deck changes before redoing anything.
- Paper deck: presentations/mattia/11_13_2026_paper/ (TBD macros to fill after EB).
- Note edits on GitLab (push works via https credential store): last.
- EB talk: Friday 13 Nov 2026. Mattia's priority: EB deck first.

## Mattia's EB-deck comments (12 Nov, slide numbers of the t = 0.97 working-copy deck)
1. Slide 5 (working point): make sure integral/min height are the updated values. If
   low pile-up uses very different cuts, justify (e.g. narrower peaks at HL-LHC); if
   results are very similar with one set, use the same values for all.
2. Remind why the 40 mm intervals are not an issue, how vertices near interval
   boundaries were tested, and why we split at all.
3. Did we try a larger GNN, or is it large enough? (39,521 params)
4. Slide 7 (samples): why the density model trained only at PU200; would uniform-mu
   training help; the right plot (mu distribution) maybe remove; update: GNN v2 trained on
   149,820 events at all mu, evaluation at all mu. Fix "185 <= mu <= 200" (should be 215).
8. Slide 9 (categories): maybe not useful; keep the plot only if it matches the GNN
   definitions exactly, else drop. Slide 11: give headline efficiency before TTVA for
   AMVF and PV-Finder; remove the left plot; replace with the paper definition (Eq. 1).
9. Slide 12 (vertex yield by category) goes after the TTVA section (uses their categories).
10. Before slide 16: a slide defining exactly Clean/Merged/Split/Fake, where sigma(z0)
    comes from, how one-to-one and truth-guided are computed, and WHY AMVF IS WORSE THAN
    NEAREST-z0. The gap is huge: explain it, check consistency with ATLAS (heard AMVF not
    optimized for HL-LHC?). Use Trey's fit weights (supplementary) as evidence.
11. Slide 17: explain why PV-Finder is mediocre at low pile-up; add error bars.
12. Slide 18 (HS after TTVA): explain the different HS definitions; small notes on slide.
13. Slide 19 (HS resolution): explain what it means and why; define terms precisely.
14. Slide 21: add sigma for AMVF too.
15. Check the HS claim (conclusion "96.3% vs 92.9%"): does it match ATLAS published values,
    do we follow their definition, is it a fair comparison? Current same-rule: 95.0 vs
    89.1 (rule) / 92.9 (AMVF stored primary). Fix the conclusion slide.

## Status
| item | status |
|---|---|
| EB rebuild at t = 0.94 | in progress (HL-LHC agent resumed 00:51 13 Nov after usage reset) |
| investigations 1,2,3,4,10,11,15 | in progress -> docs/additional_studies/eb_deck_answers_2026-11-13.md |
| deck edits 1-15 | todo |
| paper deck fill | todo |
| note edits + MR | todo |
