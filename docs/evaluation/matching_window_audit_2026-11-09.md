# Matching-window consistency audit, 9 November 2026

**Rule (PI decision):** every non-hard-scatter vertex study uses a fixed
**0.1694 mm** window, identical for all algorithms
(`pv_finder.utils.constants.MATCH_WINDOW_MM`). Hard-scatter studies use
**0.1 mm** (`HS_MATCH_WINDOW_MM`), from the ITk paper (arXiv:2412.15090) §5.5.
This page justifies the value, lists every place that broke the rule on
9 November 2026, and records what was fixed. The technical note was not
edited; its required edits are listed with replacement numbers.

## Why 0.1694 mm

The window answers one question: did the algorithm put a vertex where the
truth vertex is? It is therefore set by how precisely a single vertex is
located, and it must be the same for both algorithms.

- **Derivation.** On the 1,920 PU200 events, reconstructed and truth vertices
  (nTrk ≥ 2) are assigned one to one within 1 mm, maximising the number of
  matches and then minimising the total distance (Hungarian). The 68th
  percentile of |z_reco − z_truth| is 55.05 ± 0.23 µm for PV-Finder and
  56.45 ± 0.26 µm for AMVF. The window is 3 × the larger value,
  3 × 0.05645 mm = 0.1694 mm. It is not a Gaussian 3σ cut; the residuals have
  non-Gaussian tails. The optimal assignment is used because a greedy one
  takes the closest pairs first and biases the width about 10 % low, which
  would tighten the window and favour PV-Finder.
- **Why not σ_vtx-vtx.** σ_vtx-vtx (0.229 mm PV-Finder, 0.305 mm AMVF) is the
  half-width of the dip in the pairwise distance between *reconstructed*
  vertices. It measures how close two vertices can be and still be reported
  separately, not how well one vertex is placed. It differs by 33 % between
  the algorithms while their position precision differs by 2.5 %, and using
  one algorithm's σ for both made AMVF's measured fake rate depend on
  PV-Finder's resolution. Taking a window from a fit also makes the
  efficiency move whenever the fit moves, even with identical peaks
  (0.66 points for a 6 % change in σ, metric_definitions.md §5.1).
- **Why not 0.1 mm for all vertices.** 0.1 mm is the ITk criterion for the
  hard-scatter vertex, which has tens of tracks and a resolution of about
  10 µm (see [itk_hs_resolution_2026-11-09.md](itk_hs_resolution_2026-11-09.md)).
  The all-vertex population is dominated by low-multiplicity pile-up vertices
  whose 68 % residual is 55-56 µm; 0.1 mm is less than 2 × that and would
  reject correctly found vertices in the residual tail, mixing position
  precision into the efficiency.
- **Sensitivity (window scan, same events, paired bootstrap).** PV-Finder's
  efficiency exceeds AMVF's at every window from 0.10 to 1.00 mm. With merged
  credit the difference is +3.12 (0.10 mm), +3.37 (0.15), +3.36 (0.1694),
  +3.17 (0.20), +2.23 (0.30), +0.86 (0.50) and +0.45 points (1.00 mm); the
  one-to-one difference stays between +2.26 and +3.41 points. Fake plus split
  per event is lower for PV-Finder below 0.596 mm only. Chance matching grows
  with the window: displacing every reconstructed vertex by 3 mm still
  "finds" 11.1 % of truth at 0.10 mm, 18.6 % at 0.17 mm and 45.4 % at 0.5 mm.
  0.1694 mm sits on the plateau of the margin and keeps chance matches well
  below the wide-window values.

Draft LaTeX for the note (§4.1, replacing the paragraph that starts "We
compare the two methods using a fixed matching window"):

```latex
We compare the two methods using one fixed matching window,
$w=0.1694$~mm, for all vertex categories and both methods. The window is
set by the precision of a single reconstructed position. We assign
reconstructed and truth vertices one to one within $1$~mm, choosing the
assignment with the most matches and then the smallest total distance.
For $68\%$ of the assigned pairs, the position difference is below
$55.05\pm0.23~\mu$m for \pvf{} and $56.45\pm0.26~\mu$m for AMVF
(Section~\ref{sec:hllhc:res}). The window is three times the larger value.
It is not a Gaussian three-standard-deviation cut, since the residuals have
non-Gaussian tails. We do not use the vertex-vertex resolution \svtx{}: it
measures how close two reconstructed vertices can be and still be
separated, not where one vertex is placed, and it differs by a third
between the two methods. We also do not use the $0.1$~mm hard-scatter
criterion of Ref.~\cite{IDTR-2023-01} for all vertices. It is designed for
a vertex with many tracks; most pile-up vertices have few tracks and a
$68\%$ position error of about $56~\mu$m. The efficiency difference between
the methods is between $+2.2$ and $+3.4$ percentage points for windows from
$0.1$ to $0.3$~mm (Figure~\ref{fig:windowscan}). The window stays fixed
when the \pvf{} resolution fit changes.
```

## Inconsistencies found

### Repository code (fixed in the working tree unless noted)

| Place | Found | Status |
|---|---|---|
| `src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py:492` (old numbering) | window = fitted PV-Finder σ_vtx-vtx, also applied to AMVF (`:518`, `:551`) | **fixed**: `--match-window-mm`, default 0.1694; `0` restores the legacy σ window; pkl records `match_window_mm`, `match_window_source` |
| `src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py:402,426` | same, window = fitted σ (h5 and ROOT paths) | **fixed**: same flag and default |
| `src/pv_finder/diagnostics/amvf_fair_comparison/scan.py:45` | `HEADLINE_WINDOW_MM = 0.5` with a comment calling it the pre-registered headline window | **fixed**: renamed `SENSITIVITY_WINDOW_MM` (old name kept as alias), comment corrected; the primary 0.1694 mm is derived in `run_fair_comparison.py` |
| `src/pv_finder/diagnostics/amvf_run3_performance_plots.py:402,457` | `--match-window 0` default = fitted AMVF σ | not changed (AMVF-only diagnostic for the Fig. 25 cross-check); pass `--match-window 0.1694` |
| `src/pv_finder/diagnostics/metric_worked_example.py:40` | `WINDOW_MM = 0.22` | not changed (pedagogical example) |
| `src/pv_finder/diagnostics/peak_operating_point.py:187` | `--match-window-mm` default 0.5 | not changed (labels for the operating-point study); flagged |
| `src/pv_finder/diagnostics/visualize_fakes.py:56,184` | default 0.3 mm | not changed (event displays) |
| `src/pv_finder/diagnostics/amvf_resolution_vs_ntracks.py:150,396` | 2.0 mm | correct as is: residual-fit window for target widths, not truth matching |
| `src/gnn/diagnostics/chain_gap_decomposition.py:43` | finder cap at `WINDOWS_MM = (0.3, 0.5, 1.0)`, quoted at 0.5 mm | **needs rerun** at 0.1694 mm (chain inputs not re-evaluated here) |
| `src/gnn/diagnostics/chain_gap_decomposition.py:153` | oracle default `window = 1.0` mm | **needs rerun**; also the note says 0.5 mm for the oracle (below) |
| `src/gnn/evaluation/evaluate_ttva_run3.py:57,410` | data agreement windows 0.5/1.0/2.0, default 1.0 mm | flagged; data-vs-AMVF agreement study |

### Documentation

`metric_definitions.md` now has §5.0 (current rule) and §3.3a
(split-inclusive efficiency); §5.1 and §8.1 are labelled legacy.
`vertex_finding.md` and `research/*.md` quote 0.22-0.233 mm windows for
historical results; those are records of past runs and were left as they are.
`metric_definitions.md` §4.3 says the oracle uses < 1 mm; the note says
0.5 mm (below).

### Technical note (not edited; required edits)

| Location | Found | Required edit |
|---|---|---|
| `04_hllhc_results.tex:282-288` | height-floor check "uses the same $0.229$~mm matching window ... wider than the one used in the main comparison" | see edit 1 |
| `D_pu200_checks.tex:286-289, 323-330, 337-350` (tab:fakefilters) | windows 0.228983 and 0.227832 mm | see edit 2 |
| `D_pu200_checks.tex:160-182` (tab:pu200_tags) | each tag at its own fitted σ: r16443 86.9 % / 16.3, r16638 86.95 % / 16.10 | see edit 3 |
| `D_pu200_checks.tex:108-150` (app:pu200:standard, tab:hllhc_standard) | full results at 0.229 mm | keep only as a labelled sensitivity check (title "Results with the wider matching window" is accurate), or drop since Fig. windowscan covers it |
| `07_gnn.tex:314-316` | finder one-to-one match "within $0.5$~mm. This gives an efficiency of $78.1\%$" | at 0.1694 mm the one-to-one efficiency is 73.43 % (same events): edit 4; the decomposition (21.9 / 4.4 / 5.2 points) and oracle 73.7 % must be recomputed at 0.1694 mm before the paragraph can be finalised |
| `07_gnn.tex:322`, `08_conclusions.tex:31-32` | oracle "within $0.5$~mm" | code default is 1.0 mm (`chain_gap_decomposition.py:153`); check which value produced 73.7 %, then rerun at 0.1694 mm |
| `D_pu200_checks.tex:202` | failure examples "use a $0.5$~mm matching window" (earlier model) | illustrative; acceptable if labelled, or regenerate |
| `D_pu200_checks.tex:291-297` | real candidate = truth within 0.5 mm (height-cut and classifier labels) | classifier training label; changing it needs retraining. Keep, labelled as a label definition |
| `05_validation.tex:30, 86-87`, tab:lowpu caption | Run 2 MC, Run 2/3 data windows = fitted PV-Finder σ (0.416/0.368/0.348 mm) | move to 0.1694 mm; numbers from the low-pile-up study. Note that 0.1694 mm was derived on ITk PU200; see caveat below |
| `A_reproduction.tex:108` (tab:eta-comparison caption) | "Efficiency uses the model's fitted \svtx{} as the matching window" | recompute at 0.1694 mm (not done here) |
| `A_reproduction.tex:253-259, 283-287` | "this script ... derives its matching window from \pvf{}'s own fitted \svtx{}" | edit 5 |

**Caveat on the low-pile-up chapter.** The 0.1694 mm value is derived from
ITk PU200 residuals. Applying the same number to Run 2 follows the PI's rule
literally; applying the same *prescription* (3 × the larger 68 % residual on
that sample) would give a different number for the Run 2 Inner Detector. The
note should say which it does.

### Replacement numbers (measured 9 November 2026)

Fake-filter replay at 0.1694 mm, same events and stored peak lists as the
note (`scripts/fixed_window_replay.py`,
`outputs/11_09_2026_output/matching_window_audit/fake_filter_replay.json`).
The old-window rows reproduce the note's table exactly.

| Check (events) | Window | Eff before → after [%] | Fake/evt before → after | Surplus/evt before → after |
|---|---|---|---|---|
| height 0 → 0.03 (384, r16443) | 0.2290 (note) | 87.25 → 87.04 | 17.02 → 15.96 | 18.04 → 16.98 |
| height 0 → 0.03 (384, r16443) | **0.1694** | **83.23 → 83.08** | **19.68 → 18.55** | 20.22 → 19.09 |
| + GBT ≥ 0.3 (268, r16638) | 0.2278 (note) | 87.29 → 86.95 | 15.99 → 14.62 | 16.96 → 15.58 |
| + GBT ≥ 0.3 (268, r16638) | **0.1694** | **83.62 → 83.32** | **18.35 → 16.93** | 18.90 → 17.49 |

At 0.1694 mm the height floor removes 1.13 fakes/event for 0.15 points and
the GBT 1.42 fakes/event for 0.30 points (note: 1.06 / 0.21 and 1.37 / 0.34).

Reconstruction-tag check at 0.1694 mm (PV-Finder, merged credit; from
`outputs/11_09_2026_output/split_inclusive_efficiency/`): r16443 83.16 % and
18.73 fakes/event (1,920 events); r16638 83.25 % and 18.50 (1,888 events).

### Required note edits (old → new)

1. `04_hllhc_results.tex`, §sec:hllhc:fakes:
   old: "On a subset of $384$ PU200 events, this cut removes $1.06$ fake vertices per event while reducing efficiency by $0.21$ percentage points. This check uses the same $0.229$~mm matching window before and after the cut. The window is wider than the one used in the main comparison."
   new: "On a subset of $384$ PU200 events, this cut removes $1.13$ fake vertices per event while reducing efficiency by $0.15$ percentage points. This check uses the same fixed $0.1694$~mm matching window as the main comparison, before and after the cut."
2. `D_pu200_checks.tex`, §app:pu200:fakecuts and Table tab:fakefilters:
   "removes $1.06$ fake vertices per event. The efficiency falls by $0.21$ percentage points ... We use the same $0.229$~mm matching window before and after the cut." → "removes $1.13$ fake vertices per event. The efficiency falls by $0.15$ percentage points ... We use the fixed $0.1694$~mm matching window before and after the cut."
   "and a fixed $0.228$~mm matching window. It removes another $1.38$ fake vertices per event, or $8.6\%$. The efficiency falls by $0.34$ percentage points." → "and the fixed $0.1694$~mm matching window. It removes another $1.42$ fake vertices per event, or $7.7\%$. The efficiency falls by $0.30$ percentage points."
   Caption: "$w=0.228983$~mm" and "$w=0.227832$~mm" → "the fixed $w=0.1694$~mm" (both).
   Table rows: `$87.25$ & $17.02$` → `$83.23$ & $19.68$`; `$87.04$ & $15.96$` → `$83.08$ & $18.55$`; `$87.29$ & $15.99$` → `$83.62$ & $18.35$`; `$86.95$ & $14.62$` → `$83.32$ & $16.93$`.
3. `D_pu200_checks.tex`, Table tab:pu200_tags: "Each uses its fitted separation scale as the matching window for this check." → "Both use the fixed $0.1694$~mm matching window."; caption "The windows are $0.228983$~mm for \texttt{r16443} and $0.227832$~mm for \texttt{r16638}. The main comparison uses only \texttt{r16443}, with the fixed $0.1694$~mm window." → "Both use the fixed $0.1694$~mm window; the main comparison uses only \texttt{r16443}."; rows `$86.9$ & $16.3$` → `$83.16$ & $18.73$`, `$86.95$ & $16.10$` → `$83.25$ & $18.50$`.
4. `07_gnn.tex:314-316`: "each peak can be matched to at most one truth vertex within $0.5$~mm. This gives an efficiency of $78.1\%$" → "each peak can be matched to at most one truth vertex within the $0.1694$~mm window of Section~\ref{sec:hllhc:perf}. This gives an efficiency of $73.4\%$" (and recompute the following decomposition and oracle at the same window).
5. `A_reproduction.tex:253-255`: "The fixed-window numbers of Table~\ref{tab:hllhc_results} are not produced by this script, which derives its matching window from \pvf{}'s own fitted \svtx{} and applies it to AMVF as well." → "This script uses the fixed $0.1694$~mm matching window by default (\path{--match-window-mm}); the bootstrap errors of Table~\ref{tab:hllhc_results} come from the dedicated audit below." and `:283` "This command fits a new resolution and uses it as the matching window." → "This command fits a new resolution but keeps the fixed matching window."

## Reproduction

```bash
venv/bin/python -u scripts/fixed_window_replay.py
venv/bin/python -u scripts/split_inclusive_efficiency.py
venv/bin/python -m pytest -q tests/test_amvf_fair_matching.py
```
