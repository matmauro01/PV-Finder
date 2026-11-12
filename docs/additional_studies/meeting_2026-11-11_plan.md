# Meeting of 11 November 2026: decisions and plan

Source: Mattia's meeting with the professors, 11 November 2026, and the paper
draft `final_paper/current_paper/paper/PVFinderPaper_11_11_2026.pdf`, which
contains 29 questions for Mattia. Overarching goal: **a clear, simple paper
that draws as few reviewer comments as possible.** Every number must be
precise and traceable.

## Priorities

1. **Final word on each issue, then update the internal note.** Changes go in
   red as `[MM: ...]` so that the colleague finalising the note can see them.
2. **A paper deck for the colleague writing the paper.** It answers her 29
   questions and gives every plot and number she needs to finish the draft.

## Decisions

### D1. Thresholds: one set per regime
At most two configurations: one for Run 2/3 (low pile-up), one for HL-LHC.
This covers every working threshold:
- peak finding (amplitude, minimum width, integral, height, position estimator)
- GNN score cut `t`
- evaluation (matching window)

State the configuration once in the paper, with a table, and use it for every
result. Any result made with other settings is recomputed or dropped.

### D2. Matching window: 0.17 mm everywhere
- Justified by a plot of the single-vertex position residual
  |z_reco - z_truth|: 0.17 mm = 3 x the 68% width (56 um). The plot goes in
  the paper deck and in the internal note.
- One sentence: "for consistency with ITk performance studies, which quote
  only the hard-scatter vertex, we also give the hard-scatter result with a
  0.1 mm window."

### D3. Freeze the HL-LHC density model
Candidates: v6 production, s5 and s8 (smaller CPU models). Compare on:
1. **Vertex-finding efficiency before the GNN**, as defined in the paper:
   truth vertices with at least one reconstructed vertex within the window,
   divided by all truth vertices.
2. **Resolution plots.**
3. **Vertex categories after the GNN**, and their distribution with pile-up.

Recommend one model and freeze it.

### D4. Finish training the GNN and freeze it
Single checkpoint and single working point `t` for HL-LHC.

### D5. Datasets table in the paper deck
For the five or six HL-LHC datasets, give:
- the total number of events
- roughly how many events per pile-up value
- the number or fraction held out for validation and test
- the pile-up distribution of the validation and test sets

### D6. Performance metrics and paper plots
**Before the GNN (vertex finding):**
- truth-matching efficiency, one number as in the paper
- resolution, from events sampled uniformly in pile-up, shown as a function
  of pile-up (expected to be stable)
- hard-scatter reconstruction efficiency (0.1 mm)

**After the GNN (full chain):**
- vertex categories
- efficiency as a function of number of tracks, as in the PUB note
- categories as a function of pile-up (the current plot is fine)
- hard-scatter studies
- local pile-up density

### D7. HS longitudinal resolution
- Make sure it works: 3D adaptive fit on the GNN-assigned tracks with the
  beam-spot constraint, 9.27 um vs AMVF 10.25 um.
- Explain why in the paper deck.
- In the note, say only how it is computed, without the details.

### D8. Access
- Trey Warner (`twarner`) gets read access to the models on sneezy. Done
  11 November 2026: ACL r-x on `model_weights/`, inherited by new files.

## Execution order
1. Freeze the working points: D1, D2, and the GNN `t` from D4.
2. Model comparison and freeze: D3.
3. GNN training and freeze: D4. Long; runs in parallel.
4. Recompute every paper number and plot with the frozen configuration: D6, D7.
5. Internal-note edits in red `[MM: ...]`, on GitLab via lxplus.
6. Paper deck: answers to the 29 questions, datasets table (D5), all plots.

## Answers to the paper-draft questions
Tracked in `paper_questions_2026-11-11.md`.

## Added 11 November 2026 (later)
- **Note edits (lean, mostly finished work):**
  1. 0.17 mm window with justification plot; HS also given at 0.1 mm.
  2. Split-inclusive efficiency: one sentence.
  3. HS longitudinal resolution from the 3D adaptive fit: one paragraph and one figure.
  4. Timing claim restricted to GPU.
  5. MC on the Fig. 6.2 panels, and the AMVF-reference bias.
  6. z0 baselines in Fig. 7.2.
  7. AMVF HS selection 92.9% plus HS numbers.
  8. AMVF-weights paragraph.
  Edits are in red `[MM: ...]` on a GitLab branch with a merge request.
- **Small model:** best performance within a 0.2-point drop (0.5 at most) that is faster than AMVF on CPU at mu = 200, full chain including the GNN. Distillation from v6 if needed.
- **Data access:** Trey has read access to `data/`, including the Run 3 link targets.
- **EB talk, Friday 13 November 2026:** 20-30 min, up to date, no history. Folder `presentations/mattia/11_13_2026_EB/`. All slide numbers come from `numbers.tex`, regenerated after the recompute and freeze.
- **Timing:** GNN v2 trains in an estimated 3-6 h (v1 took 67 min for 60k events; v2 is 2.5x the data, read from disk). Target: frozen GNN v2 and frozen density model in the EB talk and the paper deck. Extend training if the validation loss is still falling at the end.
- **GNN working point (Mattia, 11 Nov):** no rule fixed yet. After GNN v2 is trained:
  - scan t on the validation events in mu bins, using frozen-density-model vertices
  - metrics per bin: Clean/truth, Clean/reco, Merged, Split and Fake, HS selection, HS resolution (3D adaptive fit), track F1
  - check the low-mu behaviour Mattia saw for oddities and explain it
  - present a few options with their trade-offs; Mattia decides
- **Quality gates for every shipped number:**
  - category bookkeeping
  - identical event sets
  - git hash and model recorded
  - physics trends vs mu and n_trk
  - AMVF anchors to the ITk paper (HS selection ~93%, HS resolution ~10 um, Fig. 25 counts)
  - independent recomputation by a second agent
- **Low pile-up decisions (Mattia, 11 Nov):**
  - density model: PVF-e400 (two-phase, no KDE) for every low pile-up result
  - local centroid (±3 bins) everywhere
  - 0.17 mm window everywhere, with one sentence that at low pile-up it is 2.5x the 68% residual
  - samples: Run 2 MC, Run 2 data, Run 3 data only; the MC is called "Run 2 MC"
- **Found 11 Nov:** the sigma_vtx-vtx fit used a hidden second peak finding (`--integral-threshold-res`, default 0.50), so 0.229 mm is at 0.50 and about 0.218 mm at the single threshold 0.30. The recompute makes the fit follow the main threshold.

## Frozen 12 November 2026 (Mattia)
- HL-LHC: density model s8 for every result, GNN v2 epoch 527, t = 0.97 (commit 4f04b53).
- Resolution vs mu in the paper: the 68% residual |z_reco - z_true|; sigma_vtx-vtx quoted at PU200 (or per bin for mu >= 100) only.
- Low pile-up GNN t: options pending from the recompute; Mattia decides.

## Paper deck contents (one PDF for the colleague, built after the recompute)
1. The frozen configuration: one table per regime (D1).
2. Datasets: training, validation and test samples for HL-LHC and low pile-up, with event counts, mu distributions and which file is used for what (D5).
3. Metric definitions: exactly how each number is computed (efficiency, fake fraction, sigma_vtx-vtx, 68% residual, categories, HS selection, HS resolution, track F1, local density), with the code that computes it.
4. Every paper plot and number, PV-Finder vs AMVF on the same events (D6, D7).
5. Answers to the 29 questions, each with its page and line in the draft.
6. Answers to Mattia's own questions from the meeting (window justification, HS 0.1 mm, model freeze, GNN t, low-mu behaviour, CPU timing, Run 2 vs HL-LHC binning).
7. The proposed internal-note edits, each with page and line, the old text and the new [MM: ...] text.
