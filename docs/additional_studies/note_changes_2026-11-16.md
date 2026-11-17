# Changes to the internal note, 13-16 November 2026

Base: `main` at 91b6f49 (9 Oct 2026). The edits were made in four rounds
on the branch `mm-edits-2026-11-13`:

- 13 Nov: be5bf7d, the first round of edits;
- 13 Nov: 08ceda2, Mattia's comments;
- 13 Nov: 612fa6c, the low pile-up and unmatched-vertex recomputations;
- 16 Nov: e6c125a, one configuration everywhere and the old studies removed.

`main` now holds the clean version, with every edit accepted.

**How to see the changes.**
- **Marked PDF:** `ANA-IDTR-2026-01-INT1.pdf` on the branch `mm-edits-2026-11-13`.
  - Added text is red, in the form `[MM: ...]`.
  - Removed text is red and struck through.
  - `[MM note: ...]` marks a short editorial note, including where a whole
    passage was removed. These notes are dropped in the clean version.
- **Diff against the original note:** `git diff 91b6f49 main -- sections/`.

## 1. One configuration per regime (the main change)

Every result in the note now uses one of the two columns of the new table
"Frozen configuration" (Sec. 2, `tab:frozen`).

| | HL-LHC (PU200) | Low pile-up (Run 2/3) |
|---|---|---|
| Vertex-finding model | s8 (439,593 parameters) | e400 |
| Peak finding | amplitude 0.01, width >= 3 bins, integral >= 0.186, no height cut, local centroid ±3 bins | same, integral >= 0.50 |
| Matching window | 0.17 mm (hard scatter 0.1 mm) | same |
| GNN, score cut | v2 epoch 527, t = 0.94 | mu = 60 v2 epoch 115, t = 0.94 |
| Vertex after the GNN | >= 2 tracks, PV-Finder+GNN and AMVF | same |

The HL-LHC results were computed with an integral cut of 0.184 plus a height
cut of 0.0184. On the same 1,920 PU200 test events, the integral cut alone
(0.186) gives:
- the same efficiency (83.06%);
- 0.04 more fakes per event;
- the same hard-scatter efficiency (99.90%);
- sigma_vtx-vtx within 0.3 um and the 68% residual within 0.02 um.

The table caption says this.

### What this replaces

| | Before (main, 9 Oct) | Now |
|---|---|---|
| HL-LHC model | 3.55M parameters | s8, 439,593 parameters (-0.15 ± 0.04 pt efficiency) |
| HL-LHC peak cuts | integral 0.30 and height 0.03; integral 0.50 for the resolution plot | integral 0.186 only; sigma fitted on the same peaks |
| Matching window | 0.1694 mm at HL-LHC; 0.416 mm or a per-sample sigma at low pile-up; 0.229 / 0.5 mm in several checks | 0.17 mm everywhere (3 x the larger 68% residual at PU200); 0.1 mm for the hard scatter |
| Low pile-up model | Run 2 reproduction | e400 |
| GNN working point | v4, t = 0.98 (0.95 / 0.99 in places), vertices with >= 1 track, 0.5 mm one-to-one | v2, t = 0.94 in both regimes, >= 2 tracks for both algorithms, 0.17 mm |

Checks that vary a cut on purpose are kept and labelled as checks:
- the window scan (0.1-1.0 mm);
- the matched-multiplicity check (height raised to 0.035);
- the 1 mm assignment radius and the 2 mm window of the resolution study.

## 2. Removed

Everything that used other cuts or the earlier model:
- **Old Appendix A, reproduction:** the settings matrix, commands, checkpoints
  and paths of the earlier configuration.
- **Old Appendix D, PU200 checks with the earlier model:**
  - the wider window;
  - the integral cut of the resolution plot;
  - single-track vertices;
  - the reconstruction version;
  - the examples;
  - the fake-peak cuts and GBT;
  - satellite peaks.
- **Sec. 2:** the table of the old peak cuts.
- **Sec. 4.3.2:** the height-cut check (0.03, 0.229 mm) and the GBT classifier.
  The section keeps the statement that the frozen configuration has no height
  cut.
- **Sec. 7.7:** the GNN v4 studies at t = 0.98, 0.5 mm and >= 1 track:
  - the truth-graph evaluation;
  - the augmentation study and its table;
  - the GPU timing of the earlier chain.
- **Sec. 7.5:** the GNN v4 median dominant-track fraction.
- **All sentences that pointed to the removed parts.**

The appendices are now: hyperparameters (A), model history (B) and model
studies (C).

## 3. Numbers and text updated, by chapter

- **Abstract, introduction, conclusions:**
  - PU200 efficiency 83.06 vs 79.85% (+3.20 ± 0.06) at 0.17 mm;
  - 0.63 ± 0.09 fewer fake and split vertices per event;
  - sigma_vtx-vtx 0.217 ± 0.003 vs 0.305 ± 0.006 mm;
  - Clean/truth after the GNN 50.6 vs 30.9%;
  - hard scatter reconstructed and selected in 95.1 vs 92.9% of events, with
    resolution 9.5 vs 10.2 um;
  - low pile-up sigma 0.39 vs 0.79 mm and efficiency 81.6 vs 75.7%;
  - timing: faster than AMVF on the GPU (about 9x at mu = 200), slower on the
    CPU for the timed model, plus a CPU projection for the frozen chain.
- **Sec. 2:**
  - the frozen-configuration table;
  - s8 in the training table;
  - the window definition;
  - split vertices included in the efficiency (no number changes).
- **Sec. 4:**
  - s8 numbers and figures;
  - a new figure justifying the 0.17 mm window;
  - fakes and splits given separately (18.64 vs 19.65 fakes per event);
  - one-to-one 73.36 vs 70.10%;
  - the s8 window scan;
  - close pairs 12.8% below 0.217 mm.
- **Sec. 5 (timing):**
  - the GPU and CPU statements made precise;
  - the CPU projection for s8 + GNN v2.
- **Sec. 6 (low pile-up):**
  - e400 at 0.17 mm, with AMVF on the same events;
  - Run 2 simulation overlaid on every data/simulation panel;
  - the AMVF-as-reference paragraph.
- **Sec. 7 (GNN):**
  - v2 at t = 0.94, >= 2 tracks, for every number and figure;
  - a new hard-scatter resolution figure;
  - AMVF with its stored fit weights as a cross-check;
  - z0 baselines in the clean-yield figure;
  - low pile-up chain, Truth+GNN (78.06%) and oracle (73.90%) recomputed at
    the frozen configuration;
  - Run 3 association at t = 0.94.

The page-by-page list of the first three rounds, with the old and new value of
every number, is in the PV-Finder repository,
`docs/additional_studies/note_edits_2026-11-13.md`.
