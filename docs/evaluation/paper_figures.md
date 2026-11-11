# Paper figures (short ATLAS paper, November 2026)

Figures and Table 1 for the short PV-Finder paper. All plots are regenerated
from saved evaluation outputs; no inference is run.

```bash
tmux new -s paperfig
source venv/bin/activate
python -u src/pv_finder/diagnostics/paper_figures/make_paper_figures.py \
    --out-dir outputs/11_09_2026_output/paper_figures
# optional, once per-event Athena timing exists:  --timing-csv <file.csv>
```

About one minute on one core. Outputs are PDF and PNG, plus `summary.json`
with every plotted number and fit. Code: `src/pv_finder/diagnostics/paper_figures/`
(`sources.py` loaders and alignment checks, `categories.py`, `deltaz.py`,
`timing.py`, `diagrams.py` with TikZ sources in `diagrams/`).

The "Section 7.x" references in the PI's list are sections of the planned
paper. The note has no such sections (note chapter 7 is the GNN).

## Inputs

| Figure | Sample | File |
|---|---|---|
| 3, 4 | ITk tt̄, 14 TeV, flat-μ r16443 (held out) | `outputs/08_05_2026_output/eval_v6_mu_window/r16443/eval_results.pkl` (PVF peaks, truth, PVF Δz and fit) and `outputs/08_05_2026_output/amvf_fair_comparison/cache_r16443.npz` (AMVF, same 1 920 events) |
| 5, 6 (substitute) | 13 TeV tt̄, ID, μ = 0–80, 2 550 test events, Run 2 reproduction model | `outputs/04_16_2026_output/reproduction_ep300_mc/eval_results.pkl`; truth and AMVF re-read from `data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root` via `configs/qibin_test_main_indices_v2.p` |
| 6 alt. | Run 3 collision data, 2 500 events, ⟨μ⟩ = 50.0 | `outputs/04_16_2026_output/reproduction_ep300_run3/eval_results.pkl` (AMVF, nTracks ≥ 2, beam-spot corrected, is stored as the reference) |
| 7 | none locally | only `~/codice/ANA-IDTR-2026-01-INT1/figures/athena_cpu_gpu_amvf.png` (copied unchanged) |

`sources.py` checks the alignment before plotting: the cache equals the
pkl μ-window events with identical PVF peaks and truth lists; the low-pile-up
ROOT lists reproduce the stored per-event truth and AMVF counts; the stored PVF
fits are reproduced by refitting.

## Conventions

- Categories are positional, from `amvf_fair_comparison.matching.match_vertices`
  (identical to `compare_res_reco`). Truth vertices need ≥ 2 selected tracks.
- HL-LHC window: fixed 0.1694 mm (3 × the larger 68% position error), the
  note's main comparison. Reproduces the note exactly: PVF efficiency 83.16%,
  fake 18.73/event; AMVF 79.80%, 19.69/event.
- Low pile-up window: 0.416 mm, the PVF σ_vtx-vtx, as in the note. This is not
  the HL-LHC convention.
- Δz: all pairs of one algorithm's vertices in an event, random sign, fitted
  on ±6 mm with Eq. `eq:sigmoid`. HL-LHC uses 0.04 mm bins and the PVF list
  with integral threshold 0.50; low pile-up uses the historical 0.2 mm bins.
- Error bars: standard error of the per-event mean; √N on Δz counts.

## Numbers in the figures

| | PVF σ_vtx-vtx [mm] | AMVF σ_vtx-vtx [mm] |
|---|---|---|
| ITk, 185 ≤ μ ≤ 200 | 0.229 ± 0.003 | 0.305 ± 0.006 |
| 13 TeV sim., same 2 550 events | 0.416 ± 0.011 | 0.790 ± 0.021 (note: 0.793 on a different 2 500) |
| Run 3 data, same 2 500 events | 0.348 ± 0.009 | 0.816 ± 0.023 (08-07 study: 0.820) |

Mean vertices per event (PVF / AMVF): ITk PU200 all 101.0 / 97.9, clean
71.8 / 68.0, merged 9.99 / 9.97, split 0.53 / 0.18, fake 18.7 / 19.7, truth
111.3. 13 TeV sim. with 55 ≤ μ ≤ 65 (373 events): all 35.1 / 32.9, clean
29.6 / 26.2, merged 4.81 / 5.40, split 0.11 / 0.00, fake 0.61 / 1.32,
truth 43.3.

The note's `fig:mc` category panel used the matcher of April 2026, before the
greedy-matching and merged-classification changes (2026-04-23, 2026-06-08).
Its clean/merged split differs (28.6 / 5.8 vs 29.6 / 4.8 here). Efficiency
and fake rate are unchanged.

## Open items

- **No Run 3 MC.** Figures 5 and 6 as captioned need a Run 3 (mc23) tt̄ sample
  with truth and AMVF, and a PV-Finder model evaluated on it. The 13 TeV
  sample's campaign is not recorded in the ROOT file (the note calls it Run 2
  simulation; some docs call it Run 3 MC). Confirm with whoever produced
  `/share/lazy/rocky/ATLAS_data/Latest_Sept2023/`.
- **Figure 7.** Needs per-event Athena timing (columns in `timing.py`) from
  the Athena-integration colleague. The existing PNG has per-sample points,
  and its legend says "Run 2/3" while the note text says Run 3 data.
- Table 1 is `outputs/11_09_2026_output/paper_figures/table1_amvf_track_selection.tex`,
  taken from arXiv:2412.15090 Table 6 and checked against Athena 25.0.79 / 24.0.150.
