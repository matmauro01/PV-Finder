# Evaluation — Vertex Finding

Evaluation of PV-Finder vertex finding on MC test data.

> **What every number on this page means:**
> [metric_definitions.md](metric_definitions.md) is the authoritative reference
> — exact formulas, which side each denominator is on, the code path that
> computes it, and a worked example. The categories here are the **positional**
> taxonomy (`compare_res_reco`); the vertex-association page uses the same four
> words for a **track-purity** taxonomy that disagrees by a factor of two on the
> same vertices. Read §1 and §2 there before comparing anything across the two
> pages.

## Script

`src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py`
Plotting helpers: `src/pv_finder/evaluation/vertex_finding/plots_pvf.py`

Two pipeline modes (mutually exclusive):

| Flag | Pipeline |
|------|----------|
| _(default)_ | Analytical KDE_A_z (h5) → K2H (UNet_1000) |
| `--e2e-model` | Raw tracks → trackstoHists_UNet_1000 (no KDE stage) |

## CLI Options

All options have sensible defaults — only the model checkpoint is required.

| Flag | Default | Description |
|------|---------|-------------|
| `--e2e-model` | — | E2E checkpoint (mutually exclusive with `--k2h-model`) |
| `--k2h-model` | — | K2H checkpoint (mutually exclusive with `--e2e-model`) |
| `--h5` | standard data path | HDF5 file |
| `--root-truth` | standard ROOT path | ROOT truth file; also loads qibin automatically |
| `--indices` | `configs/test_main_indices_2550evt.p` | Test event indices |
| `--output-dir` | `outputs/eval_pvf` | Output directory |
| `--device` | `0` | CUDA device (-1 = CPU) |

## How to Run

```bash
source venv/bin/activate

# Canonical — E2E v1 Run 2 MC model, all defaults:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --e2e-model model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_300_fullstate.pth \
    --e2e-type v1 \
    --output-dir outputs/eval_mc_T2HIST_400ep_ep300 \
    --title "Run 2 MC — T2HIST 400ep ep300" --device 1

# K2H stage-2, custom output dir:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --output-dir outputs/eval_k2h

# Without ROOT truth (faster, no nTracks filter):
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf.py \
    --e2e-model model_weights/... --root-truth "" --output-dir outputs/eval_no_root
```

## Test Set

- **File:** `configs/test_main_indices_2550evt.p`
- **Events:** h5 indices 48450–50999 (2550 events = last 5% of 51000)
- **Subevents:** 581400–611999 (30600 subevents, split [0.7, 0.25, 0.05])
- Matches `mattia_finder` test set exactly.

## Key Design Decisions

### Peak-finding thresholds

| Parameter | Value (default) | Used for |
|-----------|-------|---------|
| `threshold` | `1e-2` | Min bin height to start a peak |
| `INTEGRAL_THRESHOLD` | `0.5` | Min peak area — performance counts (clean/merged/split/fake, efficiency, FP, all peak-count plots) |
| `INTEGRAL_THRESHOLD_RES` | `0.5` | Min peak area — σ_vtx_vtx pairwise Δz fit |
| `min_width` | `3` bins | Min peak width |
| `min_height` | `0.03` (run3 eval) | Min peak amplitude — drops lowest-amplitude fakes |

**Height floor (2026-06-09):** `pv_locations_updated_res` gained a `min_height`
arg (default `0.0` = off; `run_eval_pvf_run3.py --min-height` defaults to `0.03`).
The integral cut gates on *area*, so a wide low shoulder (~0.06 tall) clears it;
a height floor removes those. Operating-point scan (`diagnostics/peak_operating_point.py`,
v4b, 300 evt): floor `0.03` removes ~1 fake/evt for ~0.3 pp efficiency (near-free
junk removal). Pushing higher is a real efficiency trade — fake p90 (0.16) overlaps
real-peak p10 (0.14), so amplitude alone cannot separate fakes from low-ntrks PVs
beyond this point. See JOURNAL 2026-06-09.

Note on defaults: `run_eval_pvf_run3.py --min-height` defaults to `0.03`, but the
**headline Run 2 / Run 3 and HL-LHC baseline numbers are quoted at `--min-height 0.0`**
(reproduction parity); the `0.03` floor is reported as a separate HL-LHC
operating point, not the baseline.

### HL-LHC operating point (2026-08-04) — current recommendation

Chosen on half of the held-out r16443 events and reported on the other half,
then confirmed by a full eval on both held-out files. Launch it with
`scripts/eval_v6_operating_point.sh`, which records the invocation and git SHA
into the output directory.

```
--centroid-halfwidth 3     local centroid about the region maximum, clipped
--integral-threshold 0.30  (the previous production value was 0.20)
--min-height 0.03
--peak-threshold 0.01      unchanged      min_width 3   unchanged
```

v6 model, μ ∈ [185, 215], 1920 / 1888 events. Each row is r16443 / r16638. Rows
are kept rather than overwritten so the size of each correction is visible; only
the last one is current.

| | efficiency | fake/evt | σ_vtx-vtx | reco/evt |
|---|---|---|---|---|
| previous production | 0.8820 / 0.8825 | 21.52 / 21.32 | 0.2328 / 0.2325 | 106.8 |
| this operating point as published 08-04 | 0.8643 / 0.8656 | 16.64 / 16.36 | 0.2190 ± 0.0050 / 0.2200 ± 0.0047 | 101.0 |
| + commensurate pairwise binning | 0.8648 / 0.8662 | 16.60 / 16.32 | 0.2200 ± 0.0031 / 0.2213 ± 0.0030 | 101.0 / 100.3 |
| **+ σ fitted on the μ window — current** | **0.8691 / 0.8695** | **16.31 / 16.10** | **0.2290 ± 0.0031 / 0.2278 ± 0.0032** | **101.0 / 100.3** |
| AMVF, same events, PV-Finder's window | — | 17.78 / 17.55 | 0.3048 | 97.9 / 97.4 |

> **The first three rows quote a σ that was fitted on the wrong events, and an
> efficiency that inherited the error.** Until 2026-08-05 the eval accumulated
> pairwise Δz over *every event it read* and applied the μ window only to the
> summary. On these flat-μ files that is 25 000 events at **⟨μ⟩ = 99.6** against
> the 1920 / 1888 events at ⟨μ⟩ = 192.5 the efficiency is quoted on — so σ
> described a much lower-density population than the number printed beside it,
> and because σ is fed back as the matching window the error propagated into the
> efficiency too. Fixed in `2e55f79`; the eval now prints both, the μ-window σ as
> the headline and the all-events σ as a secondary line marked *mixed-μ; NOT the
> headline*. On r16443 those are **0.2290 ± 0.0031** and **0.2200 ± 0.0031**.
>
> Net effect of the correction, r16443 / r16638: σ **+4.1 % / +2.9 %**,
> efficiency **+0.43 / +0.33 points**, fake rate **−0.29 / −0.21 per event**.
> None of it is a change in the algorithm — the peak lists are identical and only
> the matching window moved. Confirmed independently by replaying the stored peak
> lists at a fixed 0.2283 mm window, which predicts 0.8688 and 16.331 against the
> re-run's measured 0.8691 and 16.308.

**Which convention these efficiencies use.** `eff = (tc + tm) / n_truth`, where
one reconstructed vertex may claim every unmatched truth vertex inside its
window — the merged-credit convention. That is close to the ATLAS "merged"
definition and it is a large inflation of the *absolute* number: **+11.5 points**
over strict one-to-one (0.7501 → 0.8648 on the pre-correction numbers). Do not
compare these figures against anything counted one-to-one. Exact formulas,
denominators and the code path are in
[metric_definitions.md](metric_definitions.md) §3; the PV-Finder vs AMVF
comparison at a common window, including the AMVF efficiency that had never been
computed, is in [amvf_fairness_audit.md](amvf_fairness_audit.md). Both supersede
anything restated here.

> **Reproducibility.** The artefacts under
> `outputs/08_05_2026_output/eval_v6_operating_point/` were produced at commit
> `53d1967`, before the σ fix, and are **superseded, not merely out of date** —
> they cannot be reproduced from current HEAD. The current artefacts are
> `outputs/08_05_2026_output/eval_v6_mu_window/`, produced at `2e55f79`
> (r16443) and `934930d` (r16638); those two differ only in a warning message on
> a branch neither run reaches, so both match HEAD's behaviour. `INVOCATION.txt`
> in each directory records the SHA.

> **⚠ Do not quote the AMVF column of this table as a comparison.** Every number
> in it was measured with a matching window equal to *PV-Finder's* fitted
> σ_vtx-vtx, which is 28 % tighter than AMVF's own. The row is kept because it is
> what the eval prints, not because it is a fair benchmark. The corrected
> comparison is
> [**PV-Finder vs AMVF at a common window**](#pv-finder-vs-amvf-at-a-common-matching-window-2026-08-05),
> and it replaces the claim that used to stand here — that PV-Finder reconstructs
> "more vertices than AMVF with fewer fakes (101.0 vs 97.9 reco/evt, 16.6 vs 17.8
> fake/evt)". That claim is not wrong so much as unearned: at a common window the
> efficiency lead is **3.5× smaller** than the same table implies, and the fake-rate
> lead does not survive being counted in a way that is immune to fake/split
> relabelling.

#### About a third of the quoted fake rate is real vertices — but AMVF is penalised too

Read the fake rate with this attached. Measured 2026-08-05 at this operating
point on both held-out files, against the **unfiltered** `TruthVertex` list and
with a displaced-peak control for accidental coincidences:

| | per event | % of surplus |
|---|---|---|
| surplus peaks (fake + split) | 17.22 / 17.12 | 100 % |
| within the matching window of a real truth vertex with nTrk < 2 | 5.08 / 5.10 | 29.5 / 29.8 % |
| …of which accidental (the same peaks displaced by 3–12 mm) | 0.91 | 5.3 % |
| **excess over accidental — real vertices counted as fakes** | **4.17** | **24.2 %** |
| genuinely spurious | ~12.0 | ~70 % |

Every one of them has nTrk = 1; none have 0. The accidental control is flat at
5.1–5.5 % for displacements of 1 to 12 mm, so this is not a vertex-density
effect. `run3_io.py` applies the standard nTrk ≥ 2 cut to truth as well as to
AMVF, so these vertices are not in the truth list and the matcher cannot see
them — **roughly 4.2 of the ~16.6 fake/event are real interaction vertices, not
algorithm failures.**

> **Correction, 2026-08-05.** The paragraph above is right about PV-Finder and
> was wrong to leave the impression that this is a PV-Finder-only handicap. It is
> not. Measured at a common 0.5 mm window on the same events, **AMVF also puts
> 2.36 ± 0.05 vertices/event on real nTrk = 1 interactions** (20.3 % of its
> fakes) against PV-Finder's 3.33 ± 0.05 (34.7 %). The direction is as expected —
> AMVF fits vertices and needs ≥ 2 tracks, so it is structurally less able to sit
> on a single-track interaction — but the size is not. **The net correction in
> PV-Finder's favour is 0.97/event, not 3.3.** Anyone applying this decomposition
> to a comparison rather than to PV-Finder alone must subtract AMVF's share.
> See [the common-window section](#pv-finder-vs-amvf-at-a-common-matching-window-2026-08-05).
>
> The two PV-Finder figures on this page are consistent, not competing: **4.17/evt**
> is measured at the production matching window (0.2328 mm) with the accidental
> level subtracted via a displaced-peak control, and **3.33/evt** at the audit's
> common 0.5 mm window, where AMVF can be measured the same way. Use 3.33 vs 2.36
> whenever the point is a *comparison*, and 4.17 only when the point is what
> PV-Finder's own quoted fake rate contains.

The convention should stay as it is: nTrk ≥ 2 is what AMVF is counted with, and
changing it would break that comparison. But any quote of the fake rate should
carry this decomposition. Numbers:
`outputs/08_05_2026_output/ripple_study/fake_decomposition.json`; method in
`docs/research/pairwise_dz_bump.md` §3.6.

The binning change moves efficiency by +0.05 points and the fake rate by
−0.04/evt, entirely through the matching window: the peak list is bit-identical
and σ shifted by +0.5 %. What it buys is the **fit error, down 38 %**, and a
plateau you can read. AMVF's numbers move a little for the same reason — it is
classified against truth with the same window.

Reading the efficiency drop against the previous production point. The numbers
below sit on the 08-04 row, before the σ correction moved the headline up by
0.43 points; the decomposition itself is unchanged, because it is about the
position estimator and the thresholds, not about which events the fit ran on.
Only about 1.06 of the 1.77 points is a real loss;
the rest is the matching window, which this eval derives from the fitted
σ_vtx-vtx and therefore tightens by the same 6% the estimator improved. Of the
1.06, 0.33 belongs to the position estimator and bought resolution rather than
fake suppression, and it is **entirely merge credit**: truth-clean vertices move
84.563 → 84.550 per event, so nothing that was resolved stopped being resolved.
The threshold move alone costs 0.151 efficiency points per fake/event removed.
The next step out (integral 0.5, height 0.05 → 11.84 fake/evt) costs 0.257 and
was rejected.

Note `min_height` is the right knob for **isolated** low-amplitude fakes and the
wrong one for the satellite peaks behind the pairwise-Δz bump, which it makes
relatively worse. See `docs/research/pairwise_dz_bump.md`.

**This operating point is the optimum of the stated trade, and post-hoc levers
do not improve on it (2026-08-05).** A joint scan of Gaussian pre-smoothing,
anti-lattice notch filters, NMS, minimum-separation and prominence gates on the
split, each crossed with the integral and height thresholds — **817
configurations**, selected on half the held-out events and reported on the
other — puts every one of them on or below the plain integral/height frontier.
**0 of 817 beat the deployed point, on either half**, and the two halves agree
to within 0.159 efficiency points and 0.126 fake/event across the whole grid.

Cheapest fake removal each family can manage, in efficiency points per
fake/event removed, against a budget of 0.2: **thresholds 0.216**, Gaussian
pre-smoothing 0.236, anti-lattice notch 0.246, prominence gate 0.313, relative
prominence gate 0.322, peakiness (height/integral) 0.422, NMS 0.708, minimum
separation 1.581. Even the cheapest is already over budget — that is the
quantitative statement that this operating point is at the knee. Details and the
ranked table: `docs/research/resolution_plot_ripple.md` and
`outputs/08_05_2026_output/ripple_study/`.

### PV-Finder vs AMVF at a common matching window (2026-08-05)

**This section supersedes every earlier AMVF comparison on this page.** The eval
as it stands does not compare the two algorithms symmetrically, in two
independent respects that run in *opposite* directions. Both are measured below.

- Code: `src/pv_finder/diagnostics/amvf_fair_comparison/`
- Test: `tests/test_amvf_fair_matching.py`
- Numbers, figure, invocation: `outputs/08_05_2026_output/amvf_fair_comparison/`

Same 1920 held-out r16443 events, μ ∈ [185, 215], ⟨μ⟩ = 192.5, v6 at the deployed
operating point. Nothing was re-inferred — PV-Finder's peak list comes from the
existing `eval_v6_operating_point` pkl, AMVF vertices and the unfiltered truth
list from the source ROOT, with per-event alignment between the two proved on
every event. All errors are bootstrap over events; every difference is a
**paired** bootstrap. `r16638` reproduces every number below to within 0.1
efficiency points and 0.2 fake/event, but it shares `e8481_s4494` with r16443 and
is the same generated events reconstructed twice — a consistency check, never an
independent sample.

#### The two asymmetries

**1. The matching window is ours, applied to them.** `run_eval_pvf_run3.py`
sets `sig_bins = sigma / BIN_WIDTH` from PV-Finder's fitted σ_vtx-vtx (line 492)
and uses it to classify AMVF as well (lines 518, 551). Fitted with the identical
procedure on the identical events, σ_PVF = 0.2182 ± 0.0035 mm and
σ_AMVF = 0.3047 ± 0.0057 mm, so **AMVF is judged with a window 28.4 % tighter
than its own resolution**. Judged at its own σ instead, AMVF's fake rate is
15.50/evt rather than 17.83 and its efficiency is 4.16 points higher.

This coupling is visible in our own published outputs. Same file, same events,
same AMVF: on 08-04 our σ was 0.2328 and AMVF's fake rate came out 17.37/evt;
on 08-05 our σ improved to 0.2200 and AMVF's fake rate rose to 17.78. **AMVF did
not change.** Improving our resolution tightened AMVF's window and cost it
0.41 fake/event, mechanically.

**2. nTrk = 1 truth vertices count against us.** `run3_io.py:46-58,155-161`
filters truth to nTracks ≥ 2. On these events there are 111.32 truth
vertices/event with nTrk ≥ 2 and **23.13 with nTrk = 1** (none with 0). A
reconstructed vertex landing on one of those real interactions is counted as a
fake.

#### The window scan — the headline result

The figure is
`outputs/08_05_2026_output/amvf_fair_comparison/window_scan_r16443.png` (that
directory is gitignored, so it is the only copy).

Efficiency and fake rate for both algorithms against a **common** window swept
from 0.1 to 1.0 mm. No single window choice can be accused of favouring either
algorithm, and the crossings are visible. Over the whole range:

- PV-Finder is above AMVF on efficiency at **every** window, on both the
  production and the strict one-to-one convention — no crossing on either. On the
  production convention the gap shrinks monotonically with window, from +3.12
  points at 0.1 mm to +0.86 at 0.5 mm to +0.45 at 1.0 mm; that shrinkage *is*
  asymmetry 1 plus the merge credit AMVF collects at wide windows.
- PV-Finder is below AMVF on fake rate at every window, standard or corrected
  truth — no crossing.
- **On surplus (fake + split) the two cross at 0.596 mm.** Below it PV-Finder is
  ahead by at most 0.3/event; above it AMVF is, by +0.63/event at 1.0 mm.

#### Headline table — pre-registered common window, 0.5 mm

0.5 mm was fixed before looking at any result, on three grounds, none of which is
either algorithm's own fit: it is a round physical value; it is ≥ 1.6× the fitted
σ of **both** algorithms, so neither has its own core cut into; and it is about
half the mean inter-vertex spacing at peak PU200 density (~1.0 vertex/mm), so
accidental matches stay subdominant. The same window is applied to both.

Truth definitions: **standard** = the ATLAS convention, truth is nTrk ≥ 2, and it
stays primary. **Corrected** = same efficiency denominator, but a reco vertex is
only a fake if it matches no truth interaction with nTrk ≥ 1.

| | efficiency [%] | efficiency, strict 1-to-1 [%] | fake/evt (standard, nTrk ≥ 2) | fake/evt (corrected, nTrk ≥ 1) | surplus/evt |
|---|---|---|---|---|---|
| **PV-Finder** | 93.32 ± 0.06 | 78.09 ± 0.08 | 9.59 ± 0.07 | 5.25 ± 0.06 | 14.08 ± 0.09 |
| **AMVF** | 92.46 ± 0.06 | 75.09 ± 0.08 | 11.65 ± 0.08 | 8.02 ± 0.06 | 14.28 ± 0.09 |
| **PVF − AMVF** (paired) | **+0.86 ± 0.05** | **+3.01 ± 0.06** | **−2.06 ± 0.07** | **−2.77 ± 0.06** | **−0.20 ± 0.08** |

**AMVF's truth-side efficiency is not printed by the eval at all** — only its
reco-side categories. That gap is why the AMVF efficiency column has never
appeared on this page before. It is 92.46 % here, and 83.49 % at the window the
eval actually uses.

**The efficiency convention has to be stated with the number.** `compare_res_reco`
credits one reco vertex with *every* truth vertex inside its window, so a single
reco standing between two truth vertices scores both. That is close to the ATLAS
"merged" convention and is a legitimate choice — but it inflates the absolute
efficiency, and by more as the window widens: at 0.5 mm it is worth **+15.2
points** to PV-Finder (93.32 vs 78.09 strict). Crucially it does **not** cancel in
the comparison. AMVF has fewer reco vertices, so each of its merges absorbs more
truth, and it collects proportionally more merge credit — which means the merge
convention *shrinks* our measured lead. Strict one-to-one gives **+3.01 ± 0.06
points** where the production convention gives +0.86 ± 0.05. Both are in the table;
neither is "the" answer, and quoting either without naming the convention is how
this comparison went wrong in the first place. (The independent adversarial audit
in `docs/evaluation/amvf_fairness_audit.md` ranks this the largest single effect
and reaches the same conclusion by a different matcher.)

#### What each asymmetry was worth

| | effect on the PVF − AMVF gap |
|---|---|
| Asymmetry 1 (window), on efficiency | published **+3.00** pts → common window **+0.86** pts: it flattered us by **2.14 points** |
| Asymmetry 1 (window), on fake rate | published **−1.16**/evt → common window **−2.06**/evt: it *penalised* us by **0.90 fake/evt** |
| Asymmetry 2 (nTrk = 1 truth) | worth **0.97 ± 0.07 fake/evt** in our favour — not 3.33 |

Asymmetry 1 does not point one way. It inflated our efficiency lead because a
tight window costs AMVF more truth matches than it costs us; it *understated* our
fake-rate lead because at a tight window AMVF's genuinely-found vertices fall out
of the window and are miscounted as fakes on both sides.

For asymmetry 2 the naive correction is wrong. **AMVF is penalised by the
nTrk ≥ 2 convention too**, and by more than the structural argument suggests:

| | on real nTrk = 1 truth /evt | accidental floor | genuine | % of its fakes |
|---|---|---|---|---|
| PV-Finder | 4.34 | 1.01 | **3.33 ± 0.05** | 34.7 % |
| AMVF | 3.63 | 1.27 | **2.36 ± 0.05** | 20.3 % |

The accidental floor is measured, not assumed: the same fakes are re-matched
against another event's nTrk = 1 list (same z profile, same multiplicity, no
association). A second, independent control that displaces this event's own list
by 10 mm agrees to within 0.04/event. The direction is as expected — AMVF needs
≥ 2 tracks and sits on single-track interactions less often — but the net
correction to the *comparison* is **0.97/event, not 3.33**.

**Net:** the previously quoted comparison flattered us by 2.14 efficiency points
and penalised us by 0.90 fake/event. Correcting both, PV-Finder's efficiency lead
is real but 3.5× smaller than the operating-point table implies, and its
fake-rate lead is real and slightly larger.

#### The caveat that most narrows our margin

The −2.06/evt fake-rate lead is **largely a relabelling**. Widening the window
converts a surplus vertex from "fake" to "split", and PV-Finder's surplus is
disproportionately made of near-neighbour satellites while AMVF's is
disproportionately isolated: at 0.5 mm PV-Finder has 4.49 split/evt against
AMVF's 2.63. On **surplus = fake + split**, which is immune to that relabelling,
the two are a statistical tie at 0.5 mm (−0.20 ± 0.08/evt) and AMVF is *ahead*
beyond 0.596 mm. Any claim about fake rates should be quoted next to the surplus
number.

There is a second reason not to read the headline row as a clean win: PV-Finder
emits 101.0 candidates/event against AMVF's 97.9. Raising the height floor from
0.03 to 0.0444 equalises the two candidate multiplicities — exactly equivalent to
re-running the peak finder at that `--min-height`, since the threshold only
decides whether a region is recorded and never moves a position:

| at equal candidate multiplicity, 0.5 mm | efficiency [%] | fake/evt | surplus/evt |
|---|---|---|---|
| PV-Finder, re-cut to 97.87 cand/evt | 92.58 | 7.76 | 11.91 |
| AMVF | 92.46 | 11.65 | 14.28 |
| **difference** (paired) | **+0.12 ± 0.05** | **−3.89 ± 0.07** | **−2.37 ± 0.08** |

So the honest statement is not "PV-Finder is more efficient than AMVF" but
**"PV-Finder sits on a better efficiency/purity frontier than AMVF, and the
deployed operating point spends that advantage on efficiency"**. At matched
candidate multiplicity the efficiency advantage is +0.12 ± 0.05 points — a tie in
any practical sense — while the surplus advantage becomes 2.37/event. Both
readings come from the same frontier; only the frontier claim is safe to publish
without naming the operating point.

#### Secondary: each algorithm judged by its own σ

This is the convention the eval uses, made symmetric. It is circular — a change
that improves an algorithm's resolution tightens its own window and moves its own
efficiency — and is recorded only because these numbers have been seen:

| window convention | PVF eff | PVF fake/evt | AMVF eff | AMVF fake/evt |
|---|---|---|---|---|
| both at σ_PVF = 0.2182 mm (**what the eval publishes**) | 0.8638 | 16.66 | 0.8339 | 17.83 |
| each at its own σ (0.2182 / 0.3047 mm) | 0.8638 | 16.66 | 0.8755 | 15.50 |
| **both at 0.5 mm (fair)** | **0.9332** | **9.59** | **0.9246** | **11.65** |

The first row is 0.8638 / 16.66 where the operating-point table above reports
0.8648 / 16.60, because that table's run fitted σ = 0.2200 over all 25 000 events
read using the resolution peak list (integral ≥ 0.50), while this study fits
σ = 0.2182 on the μ-window events using the efficiency peak list — the one
actually being matched. The peak lists are identical; only the window differs, by
0.0018 mm. That sensitivity is itself the point of this section.

Note the middle row: judged self-consistently, AMVF's efficiency is *higher* than
PV-Finder's. That is not a statement about the algorithms — it is what happens
when you let each one pick its own acceptance criterion, and it is the clearest
possible demonstration of why the self-consistent convention cannot settle a
comparison. Use the common window.

### Pairwise-Δz binning (2026-08-05) — `--pairwise-bins` now defaults to 300

The plateau of `resolution_plot.png` used to alternate between two levels by
3.9 %, four times its own error bars. That was not noise and not satellites: it
is a beat between the 0.04 mm quantisation of reco positions and the 0.05 mm
bins the 240-bin default gave. `lcm(0.04, 0.05) = 0.20 mm`, hence a 4-bin
sawtooth.

> **If you are comparing a resolution plot from before 2026-08-04 with one from
> after, read this first.** The sawtooth **arrived with the local-centroid
> position estimator** on 08-04. That change was a clean win on the physics
> (−3.85 % core residual width, −13 % σ_vtx-vtx) and it stands — but the local
> centroid quantises positions onto the 0.04 mm grid where the old full-region
> weighted mean did not, and that is what began beating against the plot
> binning. Same 240 bins, same events: **39.4 % |Δz| comb for the local
> centroid against 0.9 % for the previous estimator, 1.3 % for AMVF and 0.8 %
> for truth.** So an 08-04 plot looks dramatically worse than an 08-03 one for a
> purely presentational reason, and an 08-05 plot looks clean again without any
> of the physics having moved. Do not read the difference between those three
> plots as a change in resolution.

`--pairwise-bins` now defaults to `DEFAULT_PAIRWISE_BINS = 300` (0.04 mm bins,
derived as `2 × 6 mm / BIN_WIDTH_MM`), and the eval warns if it is given an
incommensurate value. Effect, both held-out files: comb amplitude 3.8–4.0 % →
0.04–0.15 %, plateau pull RMS 4.2–4.6 → 1.6–2.0, σ_vtx-vtx **fit error nearly
halved**. σ itself moves +0.6 to +0.9 %, inside the old error bar — this is a
presentation fix, not a resolution improvement, and no peak moves.
See `docs/research/resolution_plot_ripple.md`.

**Unified-threshold (2026-04-16):** Both performance and resolution use `0.5`
by default on Run 2 / Run 3 / MC. This is consistent and filters small sidelobe
peaks out of both metrics. Rationale:

- Dual-threshold (`0.2` perf + `0.5` res) was misleading: sidelobes counted as
  fakes in performance but were hidden from the resolution plot. Using `0.5`
  for both ensures consistent accounting.
- The `clean_run3` reference eval uses a single threshold (`0.4`) — same spirit.
- History: original sidelobe investigation thought E2E training had fixed the
  problem, but it was actually the stricter resolution threshold hiding them.

**HL-LHC PU200 override:** PU200 peaks are shallower (track density spread across
more vertices). Pass `--integral-threshold 0.2 --integral-threshold-res 0.2` to
avoid losing real vertices. Full scan results in `outputs/04_15_2026_output/thr_scan_hllhc/`.

### ROOT truth vs h5 truth

Without `--root`: truth PVs from h5 `pv` field — **no nTracks filter**. All truth PVs included, which inflates `reco_merged` count.

With `--root` + `--qibin`: truth PVs from `ATLAS_PVFinderData_TruthMatched.root`, filtered to **nTracks ≥ 2**. Matches `mattia_finder` exactly.

### h5 ↔ ROOT event index mapping

The h5 file uses reindexed ("pubindices") event ordering — h5 event `i` ≠ ROOT event `i`. The correct ROOT index for the `k`-th sequential test event is `qibin[k]`, stored in `configs/qibin_test_main_indices_v2.p` (copied from `mattia_finder/config/`).

### Pileup filter for summary

Summary statistics (clean/merged/split/fake averages) are computed only over events with **55 ≤ ActualNumOfInt ≤ 65** (from ROOT). This matches `mattia_finder`'s `plot_tracks2hist.py` convention. Overall efficiency across all events is also printed.

### Pairwise Δz for σ_vtx_vtx

`pv_locations_updated_res` returns PVs sorted ascending in z, so all pairwise differences `pvs[i]-pvs[j]` for `i<j` are negative. Both `+dz` and `-dz` are added to make the distribution symmetric before fitting the sigmoid.

> **Binning caveat (2026-07-20):** the default 60-bin fit over ±6 mm (0.2 mm/bin) under-samples the *PVF* dip, whose walls are near-vertical, biasing σ_vtx_vtx high (0.29 mm at PU200) — the fit does not even converge at 120 bins. Refining to ≥240 bins gives a stable PVF σ ≈ **0.22 mm** (AMVF's rounded dip is binning-independent at 0.28 mm). Quoted PU200 resolutions of ~0.28–0.29 mm from coarse-binned fits are artifacts; PVF is ~20% finer than AMVF. See JOURNAL 2026-07-20. *(2026-08-05: fineness is necessary but not sufficient — the bin width must also be an integer multiple of the 0.04 mm model grid. 240 bins is 0.05 mm and is not; the default is now 300. See the section above and `docs/research/resolution_plot_ripple.md`.)*

## Outputs

| File | Contents |
|------|----------|
| `resolution_plot.png` | Pairwise Δz histogram + sigmoid fit → σ_vtx_vtx |
| `performance_plot.png` | Clean/merged/split/fake fractions and efficiency vs pileup |
| `stats_histogram.png` | **Total reconstructed PVs/event vs pileup — PV-Finder (Σ clean+merged+split+fake) vs AMVF** (nTracks≥2). Two curves, SEM error bars, overall-mean annotation box. AMVF source: `n_amvf` (MC eval via `RecoVertex_nTracks`) or `n_truth` (real-data eval, where truth already is AMVF). |
| `reco_vs_mu.png` | Same idea as `stats_histogram` but also overlays the MC truth reference (dashed gray). MC eval only (requires `--root-truth`). |
| `category_counts_hist.png` | **5-bar summary** of per-event reco counts in the high-pileup window `μ ∈ [55, 65]`: **Total, Clean, Merged, Split, Fake**. Bars labeled with mean value on top, SEM error bars, pileup window + n_events + checkpoint metadata in the corner box. |
| `eval_results.pkl` | All per-event results, pred/truth PV positions, fit params |

## Model Checkpoints

| Model | File | Notes |
|-------|------|-------|
| **Run 2 MC — canonical (Model B, ep300)** | `model_weights/03_24_2026/reproduction_T2HIST_400ep_T2KDE100_K2H150_epoch_300_fullstate.pth` | E2E v1, 400-epoch Qi Bin reproduction, initialized from T2KDE ep100 + K2H ep150. `trackstoHists_UNet_1000` with default width (64 UNet ch, [100]×5 MLP). **Default for all Run 2 MC / Run 2 data / Run 3 data evals.** Previously used ep150 — moved to ep300 on 2026-04-15 (later in the 400-epoch schedule, more converged). |
| **HLLHC PU200 — canonical (v4b)** | `model_weights/hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3_fullstate.pth` | `TracksToHist_v2`, 280 UNet channels, 4 latent channels, `[128]×5` MLP (~3.55M params). **Default for HL-LHC PU200 evals**, run via `run_eval_pvf_run3.py` (the only script with the architecture flags): `--e2e-type v2 --e2e-unet-channels 280 --e2e-latent-channels 4 --e2e-hidden 128 128 128 128 128`, with `--integral-threshold 0.2 --integral-threshold-res 0.2`. See [training](../training/vertex_finding.md). |
| HLLHC PU200 — v2 wide (earlier) | `model_weights/hllhc_pu200_mlp50_e2e400_v2_phase2_epoch_100_fullstate.pth` | E2E v1 **wide** variant (`n_UNetChannels=96`, `l_HiddenNodes=[128]×5`, 680K params). Load via `--e2e-wide`. Phase 2 epoch 100, which is **150 effective epochs** counting the 50-epoch MLP warmup in Phase 1. LR-stable recipe: 1e-4 + 5-ep warmup + cosine decay + grad-clip. Superseded by v4b. |
| E2E v1 ep130 (Strategy B, older) | `model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth` | 50-ep MLP warmup + 400-ep E2E (`train_mlp_hist_then_e2e.py`). The "old Run 2 model" reference used in the 2026-04-09 HLLHC-vs-Run2 comparison. Default-width v1. |
| E2E v1 ep191 (tracks→hist) | `model_weights/tracks2hist_1channel_200epochs_epoch_191_fullstate.pth` | Manually extracted from a mattia_finder `.pyt` artifact (see Outstanding Issues). |
| E2E v2 ep90 (TracksToHist_v2) | `model_weights/T2HIST_v2_100epochs_epoch_90_fullstate.pth` | |
| K2H v1 ep190 | `model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth` | |
| K2H v2 ep190 | `model_weights/K2H_v2_interp_200epochs_epoch_190_fullstate.pth` | |
| T2KDE ep130 | `model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth` | |

The E2E checkpoint was extracted from the mattia_finder MLflow artifact (`.pyt` full model → state dict) using the `pvfinder` conda env, since the `.pyt` format embeds the `model` module path.

## Differences vs mattia_finder evaluate_model.py

This table is the ground truth for what is and isn't matched:

| Aspect | mattia_finder | Our script | Status |
|--------|--------------|-----------|--------|
| Truth source | ROOT `TruthVertex_z`, nTracks≥2 | Same (via `--root-truth`) | ✅ Matched |
| h5↔ROOT index mapping | `qibin_test_main_indices_v2.p` | Same | ✅ Matched |
| Peak finder thresholds (performance) | threshold=0.01, int=0.2, width=3 | threshold=0.01, int=**0.5**, width=3 | ⚠️ Intentional — unified to 0.5 (2026-04-16) |
| Peak finder thresholds (σ) | threshold=0.01, int=0.5, width=3 | Same | ✅ Matched |
| Pileup variable | `ActualNumOfInt` (float, rounded) | Same when ROOT available | ✅ Matched |
| Summary pileup filter | μ∈[55,65] | Same | ✅ Matched |
| NaN filter on predicted PVs | Called but disabled (dead code) | Not applied | ✅ Equivalent |
| Matching window units | Bins | Bins (when ROOT) | ✅ Matched |
| **Pairwise Δz for σ** | **One sign only (negative)** | **Both ±dz** | ⚠️ Intentional difference |
| **Sigmoid fit range** | All bins | All bins | ✅ Matched (clean_run3 excludes ±0.3mm centre — we don't) |

The **one intentional difference**: we add both `+dz` and `-dz` for each pair, giving a symmetric distribution. mattia_finder only stores the negative direction (PVs are sorted ascending so `pvs[i]-pvs[j]` is always negative). Our approach is more correct; the fitted σ may differ slightly in value.

## Moving Parts — Things to Be Aware Of

### qibin mapping
`configs/qibin_test_main_indices_v2.p` maps sequential test event position → ROOT event index. It has **exactly 2550 entries**, one per test event (h5 indices 48450–50999). If you change the test set or h5 file, this mapping is **invalid** and needs to be regenerated from mattia_finder.

### ActualNumOfInt
- A **float** from ROOT (e.g. 58.3), **rounded** to the nearest integer for pileup binning.
- Used for: (1) summary table filter μ∈[55,65], (2) x-axis of performance and stats histogram plots.
- Not available without ROOT — falls back to N truth PVs.
- Distinct from `NumRecoVtx` (number of reconstructed AMVF vertices, also in ROOT) and from N truth PVs (from h5/ROOT after nTracks≥2 filter).

### Vertex matching algorithm (compare_res_reco)

`compare_res_reco` classifies each reco vertex as clean/merged/split/fake and
each truth vertex as clean/merged/missed. Uses **greedy closest-first matching**:

1. Build all (reco, truth) pairs within the matching window (±σ_vtx_vtx bins)
2. Sort by distance, greedily assign 1-to-1 (closest pairs first)
3. Classify: assigned reco with no unmatched truth in window = **clean**;
   assigned reco with unmatched truth in window = **merged**; unassigned reco
   with truth in window (claimed by closer reco) = **split**; unassigned reco
   with no truth in window = **fake**

This replaced an older per-reco-independent algorithm (2026-04-23) that inflated
the merged count: if reco R saw truth T1 and T2, it was always "merged" even if
another reco R2 was a better match for T2. The greedy algorithm correctly assigns
R→T1 and R2→T2 as two clean matches.

**Truth-side merged/clean fix (2026-06-08/09).** On the truth side, a truth vertex
that wins its own dedicated reco in pass 1 (`primary_truth`) is labeled **clean**
even when that reco later absorbs a weaker neighbour; only the *absorbed* truth
(claimed in pass 2, with no dedicated reco of its own) is labeled **merged**. This
re-labels truth vertices *within* the non-missed set, so the efficiency
`(N_clean + N_merged) / N_truth` is unchanged — but the per-event merged count is
roughly halved (the absorbed-neighbour fraction) and the clean count rises
correspondingly. A merged reconstructed vertex therefore still accounts for two (or
more) truth vertices: the clean primary plus at least one absorbed neighbour. See
JOURNAL 2026-06-08/09 and `compare_res_reco` in
`efficiency_res_optimized_atlas.py`.

### σ_vtx_vtx is both output and input
σ_vtx_vtx is computed from the pairwise Δz distribution, then **used as the matching window** in `compare_res_reco`. This creates a mild circular dependency: a very different model will give a different σ, which changes how clean/merged/fake are counted. Keep this in mind when comparing numbers across very different models.

### Integral threshold — 0.5 for both (2026-04-16)

Unified threshold: both performance and resolution use `0.5` by default
(`INTEGRAL_THRESHOLD = 0.5`, `INTEGRAL_THRESHOLD_RES = 0.5`). This is
consistent: peaks counted as fakes in performance also appear in the
resolution plot's pairwise Δz. Small sidelobe peaks (integral < 0.5)
are filtered out of both. Overridable via `--integral-threshold` and
`--integral-threshold-res`.

**HL-LHC PU200 override:** peaks are shallower due to PU200 track density
spread. Pass `--integral-threshold 0.2 --integral-threshold-res 0.2`
explicitly for HL-LHC evals to avoid losing real vertices.

For reference: `clean_run3` uses `0.4` for both. Our 0.5 is slightly
stricter but same single-threshold property.

### Pileup filter scope
μ∈[55,65] applies **only to the printed summary table**. The performance plot and stats histogram use **all events**. Both bounds are overridable via `--mu-min` / `--mu-max` (e.g. `--mu-min 195 --mu-max 205` for HLLHC PU200).

### Adaptive sigmoid fit initial guess
The pairwise-Δz sigmoid fit uses initial parameters computed from the actual histogram
(`a = baseline − min(counts)`, `c = median(counts)`, `b = 10`, `rcc = 0.5`). This
adapts to widely different count scales — Run 2 (~1000/bin), HLLHC PU200 (~10000/bin) —
without hand-tuning. The older fixed `FIT_P0 = [1000, 10, 30, 0.8]` failed on HLLHC.

### E2E checkpoint format
mattia_finder saves full model objects (`.pyt`) embedding the `model` module path. These cannot be loaded directly from PV-Finder's venv. Extraction procedure: load with `conda run -n pvfinder python -c "... ckpt.state_dict() ..."` then `torch.save({"model_state": sd, "epoch": N}, "...fullstate.pth")`.

---

## Real Data Evaluation (Run 2 / Run 3)

`src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py`
Data loading: `src/pv_finder/data/run3_io.py`

Evaluates PV-Finder on real collision data (Run 2 or Run 3), using AMVF reconstructed vertices (nTracks >= 2) as the reference baseline. There is no MC truth on real data. The same script handles both Run 2 and Run 3 — the ROOT format is identical.

### Modes

| Flags | Pipeline |
|-------|----------|
| `--t2kde-model` + `--k2h-model` | Tracks → T2KDE (MaskedDNN) → K2H (UNet_1000) |
| `--e2e-model` + `--e2e-type v1` | Tracks → trackstoHists_UNet_1000 end-to-end |
| `--e2e-model` + `--e2e-type v1 --e2e-wide` | Same class, but wider (96 UNet ch, [128]×5 MLP) — for the HLLHC v2 checkpoint |
| `--e2e-model` + `--e2e-type v2` (or `v3`) | Tracks → TracksToHist_v2 end-to-end |

`--e2e-type` accepts `v1`, `v2`, `v3`; `v2` and `v3` build the **same**
`TracksToHist_v2` class (the labels are historical). For the canonical v4b
checkpoint, pass the architecture explicitly:
`--e2e-type v2 --e2e-unet-channels 280 --e2e-latent-channels 4 --e2e-hidden 128 128 128 128 128`
(defaults are 64 channels / 1 latent / `[100]×5`).

The same script also runs on **HLLHC PU200** ROOT files (Run 4) — the tree layout
is identical. Pass `--mu-min`/`--mu-max` to move the summary window from the Run 2/3
default of `[55, 65]` to something like `[195, 205]` for PU200.

### Data Sources (mutually exclusive)

| Flag | Source |
|------|--------|
| `--root` | ROOT file directly via uproot (supports `--entry-start`/`--entry-stop`) |
| `--npz` | Pre-extracted NPZ cache (faster, 2000 events) |

### How to Run

```bash
source venv/bin/activate

# Full pipeline (T2KDE + K2H), from ROOT file, 500 events:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run3/file_3.root \
    --t2kde-model model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth \
    --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --max-events 500 --entry-stop 10000 --output-dir outputs/eval_run3_pipeline

# E2E model, from NPZ cache:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --npz data/run3/cache_file3_2000ev_seed42.npz \
    --e2e-model model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth \
    --max-events 300 --output-dir outputs/eval_run3_e2e

# Run 2 real data (same script — identical ROOT format):
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run2/Run2_Data/.../user.rgarg.49035490.EXT0._000002.ATLAS_PVFinderData_Run3Data.root \
    --t2kde-model model_weights/reproduction_KDE_A_z_matmauro_run1_200_epoch_130_fullstate.pth \
    --k2h-model model_weights/reproduction_KDE2HIST_matmauro_200epochs_epoch_190_fullstate.pth \
    --max-events 2500 --output-dir outputs/eval_pvf_run2

# HLLHC PU200 — canonical v4b model, custom pileup window + threshold override:
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run4/PU200_withTiming/ATLAS_PVFinderData_601229_e8481_s4494_r16438_PU200.root \
    --e2e-model model_weights/hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3_fullstate.pth \
    --e2e-type v2 --e2e-unet-channels 280 --e2e-latent-channels 4 \
    --e2e-hidden 128 128 128 128 128 \
    --mu-min 185 --mu-max 215 \
    --integral-threshold 0.2 --integral-threshold-res 0.2 --min-height 0.0 \
    --max-events 2500 --output-dir outputs/eval_hllhc_v4b_ep3 \
    --dataset-name "HL-LHC PU200"
```

### Real Data vs MC Differences

| Aspect | MC (`run_eval_pvf.py`) | Real data (`run_eval_pvf_run3.py`) |
|--------|----------------------|-------------------------------|
| Data format | Flat HDF5 with pre-split subevents | ROOT or NPZ with variable-length track arrays |
| Pre-computed KDEs | Available (Stage 2 shortcut) | Not available — must run full pipeline |
| Ground truth | MC truth PVs | AMVF vertices (nTracks >= 2); **or MC truth (TruthVertex_z) when available (HL-LHC MC)** — AMVF then evaluated as a separate reco algorithm |
| Beam correction | Not needed (MC beam at origin) | **Applied by default** (subtracts BeamPosZ from AMVF z) |
| Pileup (μ) | ActualNumOfInt from ROOT | ActualNumOfInt from ROOT; unavailable from NPZ |
| Subevent building | Pre-split in HDF5 | Built on-the-fly from track z0 positions |
| Run 2 specifics | — | μ ≈ 25–30 peak, BeamPosZ ≈ -2.5 mm |
| Run 3 specifics | — | μ ≈ 60 peak, BeamPosZ varies |

### MC Truth Auto-Detection (HL-LHC)

When the ROOT file contains `TruthVertex_z` and `TruthVertex_nTracks` branches
(e.g. HL-LHC MC), the script automatically uses MC truth as ground truth and
evaluates AMVF as a separate reco algorithm — producing AMVF category bars in
the `category_counts_hist.png` plot alongside PV-Finder. When truth branches
are absent (real data), AMVF remains the ground truth reference.

### Beam Correction

Beam correction is **on by default** (`--no-correct-beam` to disable) for real
data where AMVF serves as truth. When MC truth is detected, no beam correction
is applied to truth or AMVF — both are in the detector frame, matching the MC
eval behavior.

### Outputs

Same as MC eval: `resolution_plot.png`, `performance_plot.png`, `stats_histogram.png`, `eval_results.pkl`.

### AMVF-Only Run 3 Diagnostic

`src/pv_finder/diagnostics/amvf_run3_performance_plots.py` produces AMVF-only
plots from the Run 3 ROOT file without loading a PV-Finder checkpoint. It uses
`RecoVertex_*` vertices with `nTracks >= 2`, computes the pairwise AMVF
`Δz` resolution curve, and, when `TruthVertex_*` branches are present, classifies
AMVF reconstructed vertices as matched, merged, split, or fake using the same
`compare_res_reco` matching logic as the evaluation scripts.

```bash
source venv/bin/activate
python -u src/pv_finder/diagnostics/amvf_run3_performance_plots.py \
    --root data/run3/file_3.root \
    --max-events 2500 \
    --out-dir outputs/MM_DD_YYYY_output/amvf_run3_2500
```

Outputs are `amvf_resolution_delta_z.{png,pdf}`,
`amvf_vertex_categories_vs_mu.{png,pdf}`, `summary.json`, and `amvf_arrays.npz`.

The category plot **requires MC truth** (`TruthVertex_*`, nTracks ≥ 2). Real Run 3
collision data (`data/run3/file_3.root`) has `NumTruthVtx = 0`, so on real data
only the resolution plot is produced (the script detects this and skips the
category plot with a message). For the category figure, use the MC ttbar sample
`data/monte_carlo/ATLAS_PVFinderData_TruthMatched.root` (13 TeV, μ = 1–80), which
is the same truth-matched sample used by `run_eval_pvf.py`. Category definitions
(Matched = `reco_clean`, Merged, Split, Fake) are identical to the eval because
the script calls the same `compare_res_reco`, with the matching window set to the
fitted `sigma_vtx_vtx` (the AMVF reco–reco analogue of the eval's window).

### Post-Processing: Smoothing + NMS

> **Off by default, not used in canonical evals.** Both steps default to off
> (`--smooth-sigma 0`, `--nms-min-sep 0`), and the headline Run 2 / Run 3 / HL-LHC
> numbers do **not** use them — surplus peaks are simply counted as fakes. They
> are experimental Run 3 tools and remain off at PU200: NMS removes genuine close
> pairs at a similar rate to surplus ones because it keys on a height ratio,
> which does not separate the two populations.
>
> The PU200 surplus population is **not** a deconvolution sidelobe: its amplitude
> is uncorrelated with its parent's (r = −0.004). What it actually is, and what
> does suppress it, is in
> [pairwise_dz_bump](../research/pairwise_dz_bump.md).

Two optional post-processing steps reduce fake sidelobe peaks on Run 3 data.
The E2E model produces UNet deconvolution sidelobes — small spurious peaks
0.5–0.85 mm from real vertex peaks, caused by over-resolving broad KDE features
in high track-density regions. These inflate the fake rate and contaminate the
resolution plot.

| Flag | Default | Description |
|------|---------|-------------|
| `--smooth-sigma` | `0` (off) | Gaussian sigma (bins) applied to histogram before peak finding only |
| `--nms-min-sep` | `0` (off) | Remove shorter peak if pair closer than this (mm) |
| `--nms-max-ratio` | `0.3` | Only suppress if short/tall height ratio < this |

**How it works:**

1. **Gaussian pre-smoothing** blurs the predicted histogram before peak finding
   (original preserved for all other purposes). Narrow sidelobe fluctuations get
   absorbed into their parent peak. Kills ~0.75 fake/evt, ~0.05 real/evt.

2. **NMS** (`suppress_neighbor_peaks()` in `efficiency_res_optimized_atlas.py`)
   scans pairs of peaks within `min_sep` mm. If the shorter peak's height is
   < `max_ratio` × the taller, the shorter is suppressed (tallest-first ordering).
   Preserves genuine close vertex pairs (similar heights) while killing sidelobe
   fakes (3–5× shorter than parent). Kills ~1.4–2.1 fake/evt depending on ratio.

**Two operating points:**

| Config | Eff | Fake/evt | Real lost/evt | Fake:real | sigma |
|--------|:---:|:---:|:---:|:---:|:---:|
| No PP (baseline) | 97.6% | 4.80 | — | — | 0.347 mm |
| s=2 + NMS(0.85, **0.3**) | 97.4% | 2.70 | 0.14 | **10:1** | 0.487 mm |
| s=2 + NMS(0.85, **0.5**) | 96.9% | 2.00 | 0.50 | **3.7:1** | 0.592 mm |

- **NMS 0.3** (conservative): 90.9% of removed peaks are fake; 98% purity in the
  0.55–0.65 mm sidelobe core. Barely touches real vertices.
- **NMS 0.5** (aggressive): halves the fake rate but removes 0.50 real/evt.

Sigma increases with PP because fake close pairs that artificially pulled it
down get removed — the post-PP sigma is a more honest measure.

**Example (conservative):**

```bash
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run3/file_3.root \
    --e2e-model model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth \
    --smooth-sigma 2.0 --nms-min-sep 0.85 --nms-max-ratio 0.3 \
    --output-dir outputs/eval_run3_s2_nms03
```

### Histogram-only GBT fake gate (HL-LHC, 2026-06-09)

At PU200 the sidelobe tools above do not apply (the surplus peaks are not
sidelobes — see [pairwise_dz_bump](../research/pairwise_dz_bump.md)). Instead, a
post-hoc gradient-boosted-tree classifier on **histogram-only** peak features can
gate fakes without retraining the model.

| Flag | Default | Description |
|------|---------|-------------|
| `--gbt-filter-model` | — | Path to a `peak_classifier_results.pkl` (reads the `gbt_hist_model` entry) |
| `--gbt-threshold` | `0.7` | Keep peaks with predicted P(real) ≥ threshold |

The gate uses the 8 histogram-shape features (`_hist_features`) that match
`peak_classifier_v2.py` features 15–22 exactly (peak height, local integral,
skewness, FWHM, curvature, relative height, nearest-peak Δz and height ratio). It is
applied right after peak finding, so filtered peaks are excluded from **both** the
category counts and the σ_vtx_vtx pairwise-Δz fit. Trained on `r16438` and validated
on the independent file `r16633`, the v4b gate at `--gbt-threshold 0.3` gives
Eff ≈ 0.927, ~11.3 fake/evt, σ ≈ 0.282 mm (vs ~14 fake/evt with no gate). See the
[peak_classification_study](../research/peak_classification_study.md) and JOURNAL
2026-06-09.

```bash
python src/pv_finder/evaluation/vertex_finding/run_eval_pvf_run3.py \
    --root data/run4/.../ATLAS_PVFinderData_..._r16633_PU200.root \
    --e2e-model model_weights/hllhc_pu200_e2e_v4b_3ep_280ch_4lat_stepwarmup_phase2_epoch_3_fullstate.pth \
    --e2e-type v2 --e2e-unet-channels 280 --e2e-latent-channels 4 \
    --e2e-hidden 128 128 128 128 128 \
    --integral-threshold 0.2 --integral-threshold-res 0.2 --min-height 0.0 \
    --gbt-filter-model outputs/.../peak_classifier_results.pkl --gbt-threshold 0.3 \
    --mu-min 185 --mu-max 215 --output-dir outputs/MM_DD_YYYY_output/eval_v4b_gbt
```

### NMS Diagnostic Script

`src/pv_finder/evaluation/vertex_finding/nms_diagnostic.py` — re-runs inference
on a subset of events, identifies which peaks NMS removes, classifies them as
real (truth-matched) or fake, and generates per-vertex zoom plots.

```bash
python src/pv_finder/evaluation/vertex_finding/nms_diagnostic.py \
    --root data/run3/file_3.root \
    --e2e-model model_weights/e2e_mlpHist50_e2e400_1latent_mse_phase2_epoch_130_fullstate.pth \
    --entry-start 300000 --entry-stop 300200 \
    --output-dir outputs/nms_diagnostic --device 0
```

Outputs: `removal_stats.png` (4-panel summary), `zoom_plots/` (~40 per-vertex
3-panel plots with analytical KDE + track scatter), `removed_peaks_summary.pkl`.

---

## Outstanding Issues

1. **E2E checkpoint extraction** — `tracks2hist_1channel_200epochs_epoch_191_fullstate.pth` was manually extracted. Other epoch checkpoints have not been extracted. Automate if needed.

2. **σ_vtx_vtx fit differences vs clean_run3** — clean_run3 excludes central |x|≤0.3 mm bins from the sigmoid fit and tries a Gaussian notch fit first; PV-Finder fits all bins with a sigmoid only. clean_run3 uses different peak-finding thresholds (threshold=0.02, integral=0.4, width=2 vs our 0.01/0.5). Note: the earlier dual-threshold design (0.2 for counts, 0.5 for resolution) was **superseded on 2026-04-16** by a unified `0.5` for both counts and resolution, so the two metrics now account for the same peak set (see "Integral threshold — 0.5 for both"). HL-LHC PU200 overrides both to `0.2` because PU200 peaks are shallower.

3. **No nTracks in h5** — the flat h5 `pv` field has only z positions. The nTracks≥2 filter requires ROOT. Running without `--root-truth` gives unfiltered truth (more merged, lower clean counts).
