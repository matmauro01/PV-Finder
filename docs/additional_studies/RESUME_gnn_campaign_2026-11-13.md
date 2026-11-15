# RESUME HERE (GNN v3 campaign and post-EB work, from 13 Nov 2026)

If a session restarts: read this file and the Status table, check running
agents/tmux jobs (`tmux ls`, `nvidia-smi`, `git log -10`), and continue the
first item not marked done. Commit + push this file whenever the table changes.

## Goal (Mattia, 13 Nov 2026, after the EB talk)
- Every metric better than AMVF, Clean HS and HS purity included.
- Train the GNN for the highest performance the data allows: larger models, a few
  runs, think the architecture through. Among the top performers, take the fastest.
  Explain the trade-offs afterwards. Working points are discussed after that.
- Consolidate the HL-LHC peak finding (position estimator, height/integral cuts;
  Mattia asked whether a single cut such as the low pile-up style is enough). Rerun.
- Plot/slide fixes from his EB comments:
  - slide 16 plot without the "PV-Finder + Eq. 4" arms (keep nearest z0);
  - one reference only: keep truth-guided, one-to-one to backup;
  - simpler, shorter slide language (he will give more comments).
- Then the paper deck fill and the internal-note edits (MR), with the new numbers.

## Context at the start
- Frozen before the talk: s8 (0.01/3/0.184/0.0184/centroid ±3), GNN v2 e527 (hidden 32,
  2 GAT layers, 39,521 params, truth-vertex candidates, no augmentation, 149,820 events),
  t = 0.94, >= 2-track rule. PU200: Clean HS 88.3 vs AMVF 92.5, HS purity 82.3 vs 88.4
  (below AMVF); HS tracks on HS vertex 87.9 vs 74.1; Clean/truth 50.6 vs 30.9; F1 70.2 vs
  68.5; HS sel 95.1 vs 92.9; HS res 9.5 vs 10.2 um. mu 0-10 Clean/truth 2.5 pt below AMVF.
- Test set r16443 entries 0-24,999: final evaluation only, never for tuning or selection.

## Status
| item | owner | status |
|---|---|---|
| A. HL-LHC peak-finding consolidation | agent peakfind | DONE (06add98). Adopted 13 Nov in working_points.py: s8 0.01/3/integral 0.186/no height/centroid ±3. Chain numbers need one recompute |
| B. GNN v3 campaign: diagnostics + phase-1 runs | agent gnn-v3 | DONE. a-d trained and evaluated (13 Nov cuts; `eval/phase1_all/summary/summary.md`, doc section 5). No single t beats AMVF everywhere (c and d: 30 failures at t = 0.55; track eff vs HS purity). d (h128 l4, 3.3x c's CPU) is within 0.1-0.3 pt of c everywhere: capacity buys nothing on phase-1 data |
| C. GNN v3: scale best recipe, speed, selection | gnn-v3 | PRELIMINARY COMPARISON DONE 14 Nov 12:00: `docs/evaluation/gnn_v3_preliminary_comparison_2026-11-14.md` (doc sections 7-9). p2a/p2c evaluated (one cut 28 / 31 failures, two cuts 3 / 4 of 286); all of c, d, p2a, p2c within 0.3 pt (paired). Recommendation: p2c (h32 l3, 1.3x v2 CPU) with two cuts (t 0.1, t_HS 0.65-0.7). p2b (h128 l4) training DONE 14 Nov 20:56 (1,260 epochs). Evaluated 21:58: within 0.3 pt of p2a/c/p2c (comparison doc, update 14 Nov 22:30); then Mattia picks network + (t, t_HS); then one test-set evaluation |
| C2. Control run e (truth-candidate training, c's recipe; Mattia) | gnn-v3 | DONE 14 Nov 06:06 (doc section 9): e is worse than c at every matched point (PU200 trk eff at 85% purity 56.9 vs 60.1, HS purity at AMVF's HS fraction 93.2 vs 96.6); single t 50 failures (c 30), two-threshold 35 (c 3). e's s8-monitor loss is best at epoch 2 and degrades with training (truth sigma/height/no-missed-vertex mismatch). Not a replacement for v2 as the note's truth-trained reference. Last-epoch eval queued (tmux gnnv3_evalq0, eval/phase1_e539). Optional: sigma-free e + c pair; e_aug |
| D. Working-point options for Mattia | me | Preliminary comparison sent 14 Nov noon (docs/evaluation/gnn_v3_preliminary_comparison_2026-11-14.md). DECISION Mattia 14 Nov 13:00: keep ONE score cut (no two-threshold), simple and sensible as before. PROPOSED 14 Nov 22:30: p2c at t = 0.5 (comparison doc, last section). NEXT: Mattia approves, then one test-set evaluation with final peak cuts (0.186, no height) |
| E. Plot/slide fixes + EB deck refresh | me / agent | after D (numbers change) |
| F. Paper deck fill | agent | after D |
| G. Note edits + MR | agent noteedits | DONE 13 Nov: Draft MR !1 (branch mm-edits-2026-11-13, be5bf7d), PDF presentations/mattia/11_13_2026_paper/note_with_MM_edits_2026-11-13.pdf, summary docs/additional_studies/note_edits_2026-11-13.md. GNN v2 numbers as final; update after GNN v3 |
