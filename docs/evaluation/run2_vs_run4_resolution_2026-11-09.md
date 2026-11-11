# Run 2 vs HL-LHC sigma_vtx-vtx gap, 9 November 2026

Question (note §6.1): AMVF/PV-Finder sigma_vtx-vtx is ~1.9 on Run 2 MC but
~1.3 at PU200. Same fit code (`fit_sigma_vtx_vtx`, 300 bins, ±6 mm),
identical events, no inference. Outputs and figures:
`outputs/11_09_2026_output/run2_vs_run4_resolution/`; code
`scripts/run2_vs_run4_resolution.py`, `src/pv_finder/diagnostics/run2_vs_run4/`.

| Sample (events) | PVF σ [mm] | AMVF σ [mm] | AMVF/PVF | p68 PVF/AMVF [µm] | HS PVF/AMVF [µm] |
|---|---|---|---|---|---|
| Run 2 MC (2550) | 0.410 ± 0.008 | 0.789 ± 0.015 | 1.93 ± 0.03 | 69.5 / 66.8 | 28.9 / 18.7 |
| HL PU200 (1920), Run 2 peak config | 0.241 ± 0.003 | 0.305 ± 0.006 | 1.26 ± 0.01 | 61.1 / 56.5 | 13.5 / 10.2 |
| HL PU200, note config | 0.229 ± 0.003 | 0.305 | 1.33 ± 0.01 | 52.2 / 56.5 | 12.4 / 10.2 |
| HL, Run 2 density matched | 0.226 | 0.258 | 1.14 ± 0.01 | 53.7 / 51.1 | 13.2 / 9.6 |
| HL, mu <= 80 (10,600) | 0.205 | 0.219 | 1.07 ± 0.01 | 50.5 / 48.1 | 13.0 / 9.6 |

(HS: closest vertex within 1 mm, iterative 3-RMS.)

**Measured.** Published 0.416 / 0.793 / 0.229 / 0.305 reproduce exactly with
their original code. Binning, fit range, pair statistics, event choice, AMVF
nTrk cut and PV-Finder peak settings each move a ratio by <= 0.06. Pile-up
density does not explain it: at equal truth density (0.5-0.8 /mm) the ratio
is 1.9 in Run 2 and 1.2 at HL-LHC. Single-vertex position precision is at
parity in both samples. AMVF's σ falls by 2.6x, PV-Finder's by 1.7x; the
track-based ideal PU-vertex precision improves 1.7x (75.6 → 44.4 µm) and the
training-target widths ~1.6x. σ / ideal precision: PVF 5.4 in both samples;
AMVF 10.4 → 6.9.

**Interpretation (not verified).** PV-Finder's separation scales with
detector vertex precision; AMVF improved by more, suggesting its Run 2
configuration had an additional separation limit (e.g. a different
new-vertex significance or merging requirement). The Run 2 MC AMVF
configuration is not recorded in the ntuple and was not checked. At HL-LHC
low mu both algorithms produce more close reco pairs than truth pairs, so the
sigmoid σ is not a separation measure there (merge length negative).

**Other findings.** The Run 2 pkl maps h5 events to ROOT entries through
`configs/qibin_test_main_indices_v2.p`, not identity. The
`outputs/08_04_2026_output/eval_v6_heldout` peak lists correspond to integral
0.5 (not 0.2 as JOURNAL 2026-08-04 says).
