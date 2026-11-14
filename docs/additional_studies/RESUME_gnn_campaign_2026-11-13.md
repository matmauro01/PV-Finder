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
| C. GNN v3: scale best recipe, speed, selection | gnn-v3 | phase 2 RUNNING (doc section 7; at 01:20 14 Nov: p2a e580, p2b e340, p2c e740 of 1260; ETA p2c about 06:00, p2a about 10:00, p2b about 21:30 14 Nov). Resume: `VMEM_KB=134217728 RUN=p2x bash scripts/run_gnn_v3_train.sh --resume`. Two-threshold DONE for b, c, d (doc section 8): c 3/286 at (0.1, 0.7), d 4/286 (0 beyond 1 sigma) at (0.2, 0.65), b 7. NEXT per finished phase-2 run: `TAG=phase2_<run> GPU=<g> bash scripts/run_gnn_v3_eval.sh <run>` + `chain_inference_v3 --top2` + `scripts/run_gnn_v3_two_threshold.sh`; pick the fastest within errors of the best |
| C2. Control run e (truth-candidate training, c's recipe; Mattia) | gnn-v3 | TRAINING since 13 Nov about 23:30 (epoch 154 at 01:20; ETA about 06:00, eval done about 08:00) (doc section 9): tmux gnnv3_e waits for d's DONE (about 23:30), trains e on GPU 0 (about 6 h; resume `RUN=e bash scripts/run_gnn_v3_train.sh --resume`), then evaluates it like c. Log `outputs/11_13_2026_output/gnn_v3_campaign/pipeline_e.log`; done when it says `PIPELINE e DONE` (about 08:00 14 Nov). NEXT: compare e vs c (`eval/phase1_ce/summary/summary.md` matched points, `eval/twothr_ce/summary/summary.md`) and vs AMVF; write doc section 9 results; report to coordinator by noon 14 Nov. e_aug deferred (needs s8 flat-mu augmentation statistics + rebuild) |
| D. Working-point options for Mattia | me | after C |
| E. Plot/slide fixes + EB deck refresh | me / agent | after D (numbers change) |
| F. Paper deck fill | agent | after D |
| G. Note edits + MR | agent noteedits | DONE 13 Nov: Draft MR !1 (branch mm-edits-2026-11-13, be5bf7d), PDF presentations/mattia/11_13_2026_paper/note_with_MM_edits_2026-11-13.pdf, summary docs/additional_studies/note_edits_2026-11-13.md. GNN v2 numbers as final; update after GNN v3 |
