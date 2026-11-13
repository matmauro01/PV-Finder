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
| B. GNN v3 campaign: diagnostics + phase-1 runs | agent gnn-v3 | diagnosis DONE. a, b, c trained (540 epochs) and EVALUATED 13 Nov 17:40 (13 Nov cuts; `outputs/11_13_2026_output/gnn_v3_campaign/eval/phase1_abc/summary/summary.md`; doc section 5). No run beats AMVF everywhere at one t: global track eff vs HS purity conflict (c best: 30 failures at t = 0.55, trk eff in all bins). d still training (tmux gnnv3_d, GPU 0, ETA about 01:00 14 Nov; resume `RUN=d bash scripts/run_gnn_v3_train.sh --resume`); its eval starts by itself (tmux gnnv3_eval_d -> eval/phase1_d and combined eval/phase1_all/summary). NEXT: when `eval_phase1_d.log` says `EVAL phase1_all DONE`, add d to doc section 5 |
| C. GNN v3: scale best recipe, speed, selection | gnn-v3 | phase 2 RUNNING since 13 Nov 17:44 (doc section 7): p2a h64 l3 (tmux gnnv3_p2a, GPU 1, ETA 14 Nov about 10:00), p2b h128 l4 (gnnv3_p2b, GPU 2, ETA about 21:30), p2c h32 l3 (gnnv3_p2c, GPU 3, ETA about 06:00); softmax loss, 70 shards (v2 events + r16438 thinned + r16633 PU200), 1,260 shard epochs. Resume: `VMEM_KB=134217728 RUN=p2x bash scripts/run_gnn_v3_train.sh --resume`. Peak-quality check DONE (doc section 6: training candidates realistic). CPU timing DONE (section 5). NEXT per finished run: `TAG=phase2_<run> GPU=<g> bash scripts/run_gnn_v3_eval.sh <run>`, then one gnn_v3_eval_summary over v2e527, c, d, p2a-c; pick the fastest run within errors of the best; consider the two-threshold point (section 5) |
| D. Working-point options for Mattia | me | after C |
| E. Plot/slide fixes + EB deck refresh | me / agent | after D (numbers change) |
| F. Paper deck fill | agent | after D |
| G. Note edits + MR | agent noteedits | STARTED 13 Nov evening (Mattia asked to read them): branch mm-edits-2026-11-13 + Draft MR, summary docs/additional_studies/note_edits_2026-11-13.md; GNN-dependent numbers flagged pending |
