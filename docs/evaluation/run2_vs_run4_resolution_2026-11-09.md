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

## 2026-11-11: ideal target readout

Question (Mattia's professor): is the smaller Run 2 → HL-LHC gain of the
AMVF/PV-Finder ratio a sign that PV-Finder does not use the better HL-LHC
tracks fully, so that finer output bins are needed?

Test: the training target is what a perfect network would output. It goes
through the same peak finder and the same metrics as PV-Finder and AMVF, on
the same events. Run 2: the stored `target_y` of the 2,550 test events. HL-LHC:
targets rebuilt from the r16443 truth for the 1,920 PU200 events with the v6
converter rule (A, B, C = 0.14318, 0.362, -0.01797 from the training h5); the
rebuild equals the stored v6 training targets exactly (max |diff| 0 in
float16). A 0.02 mm grid version keeps the same physical peak-finder windows.
Code: `src/pv_finder/diagnostics/run2_vs_run4/target_readout.py`,
`scripts/target_readout.py`, `tests/test_target_readout.py`; outputs
`outputs/11_11_2026_output/target_readout/results.json`.

The production setting needs 3 bins above threshold. HL-LHC HS targets are
1-2-bin spikes (sigma 3-10 um), so it drops 4% of HS targets (23% on the
0.02 mm grid). The `w1` rows use a 1-bin minimum and find every HS.

| Sample, peak setting | Arm | sigma_vtx-vtx [mm] | core p68 [um] | HS, ITk [um] | eff @0.1694 / fake per evt |
|---|---|---|---|---|---|
| Run 2, Run 2 setting | target | 0.203 +- 0.003 | 5.1 | 0.09 | 99.6% / 0.00 |
| | PV-Finder | 0.410 +- 0.005 | 69.5 | 28.9 | 80.9% / 2.0 |
| | AMVF | 0.789 +- 0.008 | 66.8 | 18.7 | 75.7% / 2.5 |
| Run 2, note setting, 1-bin min | target | 0.180 +- 0.003 | 5.5 | 0.09 | 99.2% / 0.00 |
| HL PU200, Run 2 setting | target | 0.145 +- 0.001 | 2.1 | 0.77 (HS found 94.6%) | 99.9% / 0.00 |
| | PV-Finder | 0.241 +- 0.001 | 61.1 | 13.5 | 84.0% / 24.3 |
| HL PU200, note setting | target | 0.145 +- 0.001 | 3.6 | 0.77 (HS found 95.9%) | 99.6% / 0.00 |
| | target, 1-bin min | 0.137 +- 0.001 | 3.6 | 0.82 | 99.7% / 0.00 |
| | target, 1-bin min, 0.02 mm grid | 0.124 +- 0.001 | 1.9 | 0.01 | 99.9% / 0.00 |
| | PV-Finder | 0.218 +- 0.002 | 55.1 | 12.4 | 83.2% / 18.7 |
| | AMVF | 0.305 +- 0.002 | 56.5 | 10.2 | 79.8% / 19.7 |

Track-based ideal precision (median, from the study above): pile-up vertices
75.6 um (Run 2) and 44.4 um (HL), HS 16.4 and 6.0 um.

**Answer.** The headroom is in the network, not in the grid.
- PV-Finder's sigma_vtx-vtx is 2.0 times its own target's in Run 2 and 1.5-1.7
  times at HL-LHC. So PV-Finder is *closer* to its ideal output at HL-LHC than
  in Run 2.
- From Run 2 to HL-LHC, PV-Finder improves 1.70 times (0.410 to 0.241 mm,
  same setting), like the detector (1.70). The target itself improves only
  1.3-1.4 times, because its sigma_vtx-vtx is set by the target widths and by
  how close two peaks can be and still be found separately.
- A 0.02 mm grid improves the ideal by 10% (0.137 to 0.124 mm), far from the
  0.218 mm PV-Finder reaches. The 0.04 mm grid already carries the HS
  position to 0.8 um and all vertices to 3.6 um. PV-Finder reaches 12.4 and
  55 um. The precision is lost in the network output.
- There is headroom: a network that reproduced its target would reach about
  0.14 mm instead of 0.22 mm. Getting there means changing the network and its
  loss, not the binning. The target positions come from truth, so the
  achievable position precision is the track-based limit (HS median 6 um),
  not the target's 0.8 um.
