# Low-pile-up data/MC validation and the AMVF-reference bias, 9 November 2026

For note §6.2 (Fig. 6.2, `fig:realdata`). No inference; all predictions come
from `outputs/04_16_2026_output/reproduction_ep300_{mc,run2,run3}/eval_results.pkl`
(Run 2 reproduction model, integral 0.5, amplitude 0.01, no height cut). The
inputs reproduce the note: σ_vtx-vtx 0.416 / 0.368 / 0.348 mm, MC 91.59 ± 0.10 %
and 0.419 ± 0.014, data 95.31 % / 1.821 and 94.78 % / 1.959.

## Is there any Run 3 MC? No.

Searched `data/` (following links), `/share/lazy` (rocky, qibinlei, ekauffma
and other readable directories), `docs/data/`, the journal and configs
(`outputs/11_09_2026_output/data_mc_validation/inventory/`). Every readable
ATLAS file with filled truth is one of two samples:

- `ATLAS_PVFinderData_TruthMatched.root` (rocky `Latest_Sept2023`, 51k tt̄,
  μ 0-80, beam-spot length 35 mm). The `Latest_July2023` file and the ekauffma
  `NewXY`, `FullCovariance_dz`, `POCAApproximation`,
  `TruthMatched_FullCovariance` files are reprocessings of the same events
  (identical truth z in the first events).
- `rocky/ATLAS_data/ATLAS_PVFinderData.root` (2021) and
  `ekauffma/..._TruthMatched_3000evts.root` (μ = 60, beam-spot length 42 mm).

All Run 3 files (`data/run3/file_{2,3}.root`, `run_3_raw/file_1.root`,
`Run3Data_clean/_initial/_low_pileup`) are collision data with no truth. Other
candidates are an LHCb file, an ACTS-format file, two truncated files and three
unreadable (permission) ekauffma `BigData*` files that by name and size are
copies of the 51k sample. The HL-LHC flat-μ ITk sample is a different detector.

**Caveat.** No file carries a campaign or √s tag. "Run 2, 13 TeV" for the MC
comes from our docs. `atlas_fig25_crosscheck.md` and JOURNAL 2026-08-20 call
the same file "Run 3 MC", contradicting the note's "Run 2 simulation". Only
the producer (Rocky) can settle this; it should be asked before the note goes
out.

## Fig. 6.2 with MC

All four panels showed data only. New figures (PDF and PNG) in
`outputs/11_09_2026_output/data_mc_validation/figures/`:
`run2_resolution_datamc`, `run2_yield_datamc`, `run3_resolution_data_run2mc`,
`run3_yield_data_run2mc`. The Run 3 panels use the Run 2 MC and say so.

- Pairwise Δz: MC σ = 0.416 ± 0.011 mm against data 0.368 ± 0.009 (Run 2) and
  0.348 ± 0.009 (Run 3). The plateaus agree (χ²/ndf 44/40 and 48/40); data
  has 1.2-1.5 times more pairs at |Δz| = 0.3-0.5 mm, a narrower dip.
- Yields: after reweighting MC to each data μ profile, data/MC = 0.904
  (PV-Finder) and 0.891 (AMVF) for Run 2, 0.915 and 0.929 for Run 3. Slopes
  (vertices per unit μ): PV-Finder 0.517 / 0.514 on data vs 0.534 on MC;
  AMVF 0.435 / 0.449 vs 0.495.
- The boxes in the current yield panels (21.21 and 19.98 for Run 2) are means
  of per-μ-bin means; per-event means are 23.79 and 22.51.
- Split totals differ slightly from the April pkls (Run 2 data 1277 → 1280,
  MC 131 → 158) because of the 2026-06-08 matcher fix; recovery and
  no-reference numbers are unchanged.
- MC mapping: h5 event 48450 + i is ROOT entry `qibin[i]`, verified on all
  2,550 events (truth set, nTrk ≥ 2 counts, AMVF counts, μ).

## AMVF as the reference, in simulation

Same 2,550 MC events; 10,000 paired event bootstraps.
Delta = (AMVF reference) − (truth reference).

| Window | Sample / reference | Recovery [%] | No reference / evt | Split / evt |
|---|---|---|---|---|
| 0.416 mm (MC fitted σ) | MC, truth | 91.59 ± 0.10 | 0.419 ± 0.014 | 0.062 ± 0.005 |
| | MC, AMVF | 93.95 ± 0.12 | 1.715 ± 0.036 | 0.552 ± 0.018 |
| | MC, delta | +2.36 ± 0.13 | +1.296 ± 0.032 | +0.490 ± 0.017 |
| 0.368 / 0.348 (own σ) | Run 2 / Run 3 data, AMVF | 95.31 ± 0.09 / 94.78 ± 0.08 | 1.821 ± 0.029 / 1.959 ± 0.029 | 0.512 / 0.474 |
| 0.416 mm common | Run 2 / Run 3 data, AMVF | 95.35 / 94.87 | 1.657 ± 0.027 / 1.692 ± 0.027 | 0.666 / 0.715 |
| **0.1694 mm** | MC, truth | 80.85 ± 0.14 | 2.013 ± 0.036 | 0.002 |
| | MC, AMVF | 92.45 ± 0.13 | 2.591 ± 0.053 | 0.013 |
| | MC, delta | +11.60 ± 0.17 | +0.578 ± 0.040 | +0.011 |
| | Run 2 / Run 3 data, AMVF | 93.45 ± 0.10 / 93.21 ± 0.10 | 2.726 ± 0.038 / 2.823 ± 0.038 | 0.025 / 0.028 |

AMVF against truth on MC: 88.18 ± 0.12 % and 0.962 per event at 0.416 mm;
75.67 % and 2.50 at 0.1694 mm. Core position residual (68th percentile,
optimal one-to-one within 1 mm) on Run 2 MC: PV-Finder 69.5 µm, AMVF
66.8 µm, so 3 × p68 = 0.208 mm, not 0.1694 mm. HL-LHC r16443 at 0.1694 mm:
recovery 89.43 % with AMVF as reference vs 83.16 % with truth
(+6.27 ± 0.10); no-reference peaks 12.83 vs 18.73 (−5.90 ± 0.09).

**Facts.** On the same MC events, switching the reference from truth to AMVF
raises recovery by 2.4 points and no-reference peaks by 1.30 per event at
0.416 mm. At the common 0.416 mm window, data minus MC-with-AMVF-reference is
−0.06 ± 0.05 (Run 2) and −0.02 ± 0.05 (Run 3) no-reference peaks per event.

**Interpretation** (assumes the MC models the PV-Finder/AMVF relationship):
most of the gap between the data rows (1.8-2.0) and the MC row (0.42) of
Table tab:lowpu is the choice of reference, not extra fakes in data. The bias
depends on window and pile-up and changes sign at HL-LHC for no-reference
peaks, so no single correction transfers between samples.

Code: `scripts/data_mc_validation.py`, `scripts/data_mc_inventory.py`,
`src/pv_finder/diagnostics/data_mc_validation/`, tests in
`tests/test_data_mc_reference.py`. All numbers in `results.json` there.
