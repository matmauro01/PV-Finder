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
