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
| A. HL-LHC peak-finding consolidation | agent peakfind | launched 13 Nov |
| B. GNN v3 campaign: diagnostics + phase-1 runs | agent gnn-v3 | launched 13 Nov |
| C. GNN v3: scale best recipe, speed, selection | gnn-v3 | after B |
| D. Working-point options for Mattia | me | after C |
| E. Plot/slide fixes + EB deck refresh | me / agent | after D (numbers change) |
| F. Paper deck fill | agent | after D |
| G. Note edits + MR | agent | after F |
