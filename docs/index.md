# Docs

Project wiki. Each page describes the current state of one topic. Rewrite (don't append) when things change.

- [Presentations](presentations.md) — September 14, 2026 talk and reading decks, with the factual review against the current technical note.

## Models

- [vertex_finding](models/vertex_finding.md) — UNet (PV-Finder)
- [vertex_association](models/vertex_association.md) — GNN post-filter

## Data

- [monte_carlo](data/monte_carlo.md) — Simulated collisions
- [run_2](data/run_2.md) — Real ATLAS Run 2 data
- [run_3](data/run_3.md) — Real ATLAS Run 3 data
- [run_4](data/run_4.md) — HL-LHC PU200 simulation (with timing)
- [run_4_all_etas](data/run_4_all_etas.md) — **current** HL-LHC PU200, full ITk acceptance |eta|<4 (July 2026 re-production)

## Training

- [vertex_finding](training/vertex_finding.md) — PVF training
- [vertex_association](training/vertex_association.md) — GNN training
- [Full-pileup GNN run, 2026-09-24](evaluation/gnn_full_pileup_training_2026-09-24.md) — split, recipe, tmux monitoring and validation plan

## Evaluation

- [**metric_definitions**](evaluation/metric_definitions.md) — **start here.** What every number means: formulas, denominators, code paths, the two taxonomies, and the traps
- [amvf_fairness_audit](evaluation/amvf_fairness_audit.md) — is the PVF vs AMVF comparison apples-to-apples, and by how much
- [atlas_fig25_crosscheck](evaluation/atlas_fig25_crosscheck.md) — our vertex categories vs ATLAS IDTR-2021-01 Fig. 25: same 70% rule, AMVF reproduces their plot
- [hs_chain_studies_2026-11-09](evaluation/hs_chain_studies_2026-11-09.md) — HS reco / Clean-HS / HS-track fraction / HS selection vs mu for PV-Finder+GNN, z0 baseline and AMVF; 80.7% vs ITk ~92% reconciled (stored primary vs re-ranked links)
- [matching_window_audit_2026-11-09](evaluation/matching_window_audit_2026-11-09.md) — the 0.1694 mm / 0.1 mm (HS) window rule, its justification, every inconsistency and the note edits
- [data_mc_validation_2026-11-09](evaluation/data_mc_validation_2026-11-09.md) — no Run 3 MC exists; Fig. 6.2 with MC; AMVF-as-reference bias measured in MC
- [run2_vs_run4_resolution_2026-11-09](evaluation/run2_vs_run4_resolution_2026-11-09.md) — why the AMVF/PVF σ_vtx-vtx gap shrinks from 1.9 (Run 2) to 1.3 (PU200)
- [itk_hs_resolution_2026-11-09](evaluation/itk_hs_resolution_2026-11-09.md) — HS longitudinal resolution with the ITk-paper definition: PVF 12.4 vs AMVF 10.2 µm
- [vertex_finding](evaluation/vertex_finding.md) — PVF metrics
- [cpu_small_model_2026-11-09](evaluation/cpu_small_model_2026-11-09.md) — smaller PV-Finder for CPU: where CPU time goes, exact packed-track inference, s5/s8 runs and held-out comparison
- [additional studies summary (PDF)](additional_studies/additional_studies_summary.pdf) — one-page status of the reviewer-requested studies
- [paper_figures](evaluation/paper_figures.md) — figures and Table 1 for the short ATLAS paper: sources, conventions, open items
- [vertex_association](evaluation/vertex_association.md) — GNN metrics

## Diagnostics

- [vertex_finding](diagnostics/vertex_finding.md) — PVF plots & analysis
- [vertex_association](diagnostics/vertex_association.md) — GNN plots & analysis

## Research

- [pairwise_dz_bump](research/pairwise_dz_bump.md) — **the Δz bump: satellite peaks, model vs data, how to reduce it**
- [peak_classification_study](research/peak_classification_study.md) — post-hoc fake rejection (GBT)
- [resolution_bump_analysis](research/resolution_bump_analysis.md) — *superseded* by pairwise_dz_bump; kept only to record what it got wrong
