# The pairwise-Δz bump: satellite peaks

**Status: this page supersedes everything else in the repo on this subject.**
[resolution_bump_analysis](resolution_bump_analysis.md) is a stub recording what
it got wrong. Earlier versions of *this* page carried two wrong mechanisms and
one wrong headline number; see [What changed and why](#what-changed-and-why).

Four questions, answered in order:

1. [Why the bump exists](#1-why-the-bump-exists)
2. [Why it got bigger: model or data](#2-why-it-got-bigger-model-or-data)
3. [How to reduce it](#3-how-to-reduce-it)
4. [Why AMVF shows less of it, and why not zero](#4-why-amvf-shows-less-of-it-and-why-not-zero)

**One-line answer.** PV-Finder emits ~20.8 surplus peaks per event, about half
of them within 0.8 mm of a real vertex. Those satellites, not any property of
the physics or of the resolution, are the bump. They come from the peak finder
promoting sub-noise ripple on the flank of a real peak into a vertex, and the
ripple is there because the network is asked to draw sub-bin spikes on a
lattice-quantised output. The bump grew from v4b to v6 mostly because the data
changed, not because the model got worse: the extended-|η| re-production
roughly doubles the satellite rate per peak for *both* models.

---

## 0. The observable, and the two numbers to quote

The pairwise-Δz distribution of reconstructed vertices is flat at large
separation (a combinatorial plateau) and suppressed inside the algorithm's
resolution. A satellite population shows up as an excess just outside the dip.

Measured on **1920 μ-matched held-out events** of `r16443` (⟨μ⟩ = 192.5, PU200,
v6, `--integral-threshold 0.2 --min-height 0.0`, historical position estimator),
240-bin binning, bootstrap-over-events errors:

| | plateau | dip depth | peak excess | at \|Δz\| | band 0.3–0.7 mm |
|---|---|---|---|---|---|
| **Truth** (nTrk≥2) | 6726 | +0.5 % | 3.4 ± 0.9 % | 0.57 mm | **+0.57 ± 0.50 %** |
| **AMVF** (nTrk≥2) | 4989 | −99.8 % | 9.0 ± 1.0 % | 0.9–1.1 mm | **−14.52 ± 0.44 %** |
| **PV-Finder** | 5764 | −99.8 % | **17.5 ± 1.1 %** | 0.57 mm | **+12.48 ± 0.48 %** |

The second held-out file `r16638` gives +12.5 % independently.

Two numbers, and you need both:

- **Band excess** (0.3–0.7 mm, relative to plateau). The headline. It is *not*
  multiplicity invariant: satellite pairs grow like *n*, total pairs like *n*²,
  so a configuration that simply finds more peaks reports a *smaller* excess for
  the same per-peak pathology. Never compare it across samples of different
  vertex density on its own.
- **Satellites per peak**: plateau-subtracted surplus companions in the
  0.25–2.0 mm shell, divided by the number of peaks. Multiplicity invariant and
  truth free, so it survives a change of sample or of truth definition. For v6
  on the current sample it is **0.058 ± 0.003**.

> **The old +7.8 % band / +13.7 % peak figures are withdrawn.** They came from
> an evaluation pkl that was overwritten the same afternoon; the tool that
> produced them, re-run today on both surviving held-out files, gives +12.5 %.
> Nothing on disk reproduces +7.8 %. See
> [What changed and why](#what-changed-and-why).

### What it is not

- **Not the sample.** Truth is flat in the band on both productions
  (+0.22 ± 0.57 % on |η| < 2.5, +0.57 ± 0.50 % on all-|η|). The Δz structure is
  entirely made by reconstruction.
- **Not genuine close pairs.** Decomposing by whether both members of a pair
  match distinct truth vertices, the genuine-pair band excess is *negative* at
  every matching window tried (−2.6 % / −5.0 % / −12.5 % at 0.3 / 0.5 / 0.8 mm)
  against the all-pairs value. Remove the surplus peaks and the bump is gone.
- **Not an outward position bias on marginally-resolved pairs.** That bias is
  real (⟨|Δz|_reco − s⟩ = +0.120 mm at true separation s = 0.1–0.2 mm, falling
  to zero by 0.7 mm) but redistribution conserves pairs and cannot lift the
  integral above the plateau, and its predicted shape (monotonically falling
  from +27 % at 0.3 mm) is the opposite of what is observed.
- **Not all fakes.** About 22 % of surplus peaks sit on *real* truth vertices
  with nTrk < 2, which `src/pv_finder/data/run3_io.py` (`_filter_amvf`, applied
  to truth as well as to AMVF) removes from the truth list, so they can never be
  matched. The reported fake rate is overstated by roughly a fifth. *(Carried
  over from the 2026-08-04 verification pass.)*

---

## 1. Why the bump exists

Five facts. Unless flagged otherwise they are measured on the same 1920 events
by `src/pv_finder/diagnostics/satellite_mechanism.py` →
`outputs/08_04_2026_output/satellite_mechanism/`.

### 1.1 The peak finder is not the cause on its own

Running the production peak finder on the **target** histograms yields
61 435 peaks and **zero** surplus against 67 051 truth vertices. The rule set is
not intrinsically satellite-producing. It becomes one only when fed the
network's output. *(600-event control from the 2026-08-04 verification pass, not
re-derived here.)*

### 1.2 Almost every band satellite is emitted by the conjoined-split rule

`src/pv_finder/utils/peak_finding.py` (`conjoined_split`) closes an above-threshold region
the moment the histogram rises again after having fallen, with **zero noise
tolerance**: one bin of non-monotonicity is enough. **92.5 %** of band satellites
share their above-threshold run with another peak, i.e. they exist only because
the split fired, against 51.9 % of truth-matched peaks. (A stricter flank-based
definition puts it at 99.5 %.)

The rule cannot simply be removed: with the split disabled, PV-Finder finds
71.9 peaks/event instead of 107.1 and efficiency collapses from 0.883 to 0.616.
It is doing real work on genuinely conjoined pairs. It has no notion of *how
much* of a dip is needed.

### 1.3 The surplus peaks are a near-neighbour population

Distance from each surplus peak to the nearest truth-matched peak, 20.78 surplus
peaks/event:

| nearest real peak at | fraction | per event |
|---|---|---|
| < 0.3 mm | 5.9 % | 1.23 |
| 0.3–0.8 mm | 46.5 % | 9.67 |
| 0.8–1.5 mm | 23.8 % | 4.94 |
| 1.5–3.0 mm | 16.6 % | 3.44 |
| > 3.0 mm (isolated) | **7.2 %** | 1.50 |

Median 0.76 mm. Only 7 % of surplus peaks are genuinely isolated. The claim
still on some slides ("fakes are not sidelobes: none within 0.3 mm of a real
peak") is true as stated and misleading as read: the surplus population *is*
clustered around real vertices, just outside the resolution rather than inside
it.

### 1.4 What it is splitting on is sub-noise ripple

Topographic prominence (height minus the higher flanking saddle):

| peak class | median prominence | fraction < 0.005 |
|---|---|---|
| truth-matched | 0.751 | 2.0 % |
| surplus | 0.040 | 16.2 % |
| band satellite | 0.029 | 22.3 % |

A band satellite is a factor ~26 less prominent than a real vertex. Their
amplitude is uncorrelated with their parent's (Pearson r = −0.004, p = 0.80,
median height ratio 0.126), which **rules out a linear deconvolution sidelobe**,
since a sidelobe scales with its parent. The language of "UNet deconvolution
sidelobes" elsewhere in `docs/` does not describe this population.

The ripple is not pure noise either. At a satellite's position the latent (KDE)
signal is 3.8× and the track count 1.7× what a mirrored control position on the
other side of the parent carries, and 95 % of satellites have more latent signal
than their mirror. Satellites sit on **real but sub-threshold input structure**,
most often a low-multiplicity neighbour: 53 % of band satellites have some truth
vertex within 0.5 mm, median distance to the nearest truth vertex 0.48 mm.
*(Amplitude correlation and mirror control are from the 2026-08-04 verification
pass, not re-derived here; the prominence table is.)*

### 1.5 Two structural reasons the network emits ripple there

**The upsampling lattice.** `UNet_1000_v2` pools twice and upsamples twice with
`F.interpolate(..., mode="nearest")` (`src/pv_finder/models/unet_v2.py:35`), so
the decoder writes on a stride-4 lattice. Peak argmax bins mod 4 are then not
uniform, and the effect is concentrated exactly where it should not be:

| peak class | modulation (max−min)/mean | χ² (3 dof) |
|---|---|---|
| truth-matched | 3.9 % | 58 |
| **surplus** | **35.1 %** | **854** |
| band satellite | 19.2 % | 93 |

Real peaks are placed by the tracks and are nearly phase-free; surplus peaks are
strongly phase-locked to the decoder's own grid. This is a genuine architectural
artefact, and it is a retrain-level fix.

**Targets the network cannot draw.** `root_to_h5._build_truth_histogram` gives
each truth vertex a Gaussian of width σ(nTrk) and then multiplies it by
`max(1, 0.15/σ)`. Under the `hllhc_alleta` preset, on this sample, **31.6 %** of
nTrk≥2 truth vertices get σ below one 0.04 mm bin (down to 0.0067 mm), with an
amplitude scale up to **22.4×** (median 2.84×, 73 % above 2×). The network is
being asked for spikes the 0.04 mm grid cannot represent. What it produces
instead is a taller, structured blob, and structure on a blob is what the split
rule converts into vertices.

That this matters is visible in the model comparison below: **v5, whose only
difference from v4b is the target-width preset (narrower and taller exactly at
low multiplicity), has the highest satellite rate of the three models on both
productions.**

---

## 2. Why it got bigger: model or data

### 2.1 The premise needs correcting first

"The model changed" is misleading. v4b, v5 and v6 have **identical architecture
and identical loss**: `TracksToHist_v2`, 280 UNet channels, 4 latent channels,
[128]×5 MLP, plain MSE (`configs/vertex_finding/config_hllhc_pu200_e2e_v4b_stepwarmup.yml`,
`..._v5_corrected.yml`, `config_hllhc_alleta_v6_mse.yml`). What differs between
them is **what they were trained on**: the pool (2.74 M |η| < 2.5 events → 5.17 M
all-|η| events), the target-width preset (`hllhc` → `hllhc_corrected` →
`hllhc_alleta`), and the epoch budget (3+3 → 2+2). So even the "model" axis is
mostly a data axis.

### 2.2 The measurement

The two ROOT productions contain the **same events**: `ActualNumOfInt` and
`RecoVertex_z` are identical entry by entry (verified). Only the track
collection differs (|η| < 2.5, 480 tracks/event → |η| < 4 with uniform ITk cuts,
707 tracks/event, +47 %) and, as a consequence, the truth-vertex bookkeeping
(95.2 → 111.3 nTrk≥2 truth vertices/event).

That makes a clean 3 × 2 grid: each checkpoint run on each production over the
same 1920 μ-matched events, one peak-finder configuration, historical position
estimator. `src/pv_finder/diagnostics/bump_model_vs_data.py` →
`outputs/08_04_2026_output/bump_model_vs_data/` (all ten curves overlaid in
`bump_model_vs_data.png`; the per-event pair histograms are cached under
`cells/`, so the table can be re-derived without re-running inference).

**Band excess in 0.3–0.7 mm [%]** (the headline observable):

| | \|η\| < 2.5 production | all-\|η\| production | Δ (data), paired |
|---|---|---|---|
| **v4b** | 7.99 ± 0.54 | 11.41 ± 0.51 | **+3.42 ± 0.40** |
| **v5** | 9.62 ± 0.49 | 10.68 ± 0.47 | +1.06 ± 0.38 |
| **v6** | 10.86 ± 0.51 | 12.48 ± 0.48 | **+1.62 ± 0.37** |
| Δ (model, v4b→v6) | **+2.87 ± 0.45** | **+1.07 ± 0.43** | total **+4.49 ± 0.46** |
| *AMVF* (control) | −8.37 ± 0.49 | −14.52 ± 0.44 | −6.16 ± 0.23 |
| *Truth* (control) | +0.22 ± 0.57 | +0.57 ± 0.50 | +0.36 ± 0.39 |

**Satellites per peak** (multiplicity invariant, the one to trust when the
vertex density changes):

| | \|η\| < 2.5 | all-\|η\| | Δ (data), paired |
|---|---|---|---|
| **v4b** | 0.0277 ± 0.0029 | 0.0609 ± 0.0029 | **+0.0332 ± 0.0026** |
| **v5** | 0.0460 ± 0.0029 | 0.0717 ± 0.0029 | +0.0257 ± 0.0027 |
| **v6** | 0.0339 ± 0.0028 | 0.0579 ± 0.0028 | **+0.0240 ± 0.0025** |
| Δ (model, v4b→v6) | **+0.0062 ± 0.0028** | **−0.0030 ± 0.0028** | total **+0.0302 ± 0.0028** |

Peaks per event: v4b 92.9 → 100.9, v5 101.3 → 110.2, v6 99.3 → 106.8; truth
(nTrk≥2) 95.2 → 111.3; AMVF 91.7 → 97.9.

Errors are **paired** bootstrap over events (2000 replicas): the same event
resampling is applied to both cells of a difference, which is correct because
every cell is measured on the same events in the same order.

### 2.3 Reading it

**It is the data.** On the multiplicity-invariant observable the answer is not
close. Moving from the |η| < 2.5 to the all-|η| production **roughly doubles the
satellite rate per peak for both models** (v4b ×2.2, v6 ×1.7); the two
data-axis contrasts account for **79 % to 110 %** of the total v4b-old → v6-new
change of +0.0302. The model axis accounts for **−10 % to +21 %** and is
consistent with zero on the production that matters (v6 is in fact *better* than
v4b on all-|η|, −0.0030 ± 0.0028).

**And it is not domain shift.** The obvious objection to the data axis is that
each model is being run outside its training domain in one of the two cells.
That objection fails on the v6 row: v6 moves from the production it was **not**
trained on to the one it **was**, and gets *worse* (0.0339 → 0.0579). Both
models degrade in the same direction on the new data regardless of which cell is
in-domain, so the extra tracks, not the mismatch, are doing the work.

**On the raw band excess the split looks more even, and that is an artefact of
the observable.** The two decomposition paths from v4b-old to v6-new both sum
to +4.49, but they apportion it differently:

| path | first step | second step |
|---|---|---|
| change data, then model | data **+3.42 ± 0.40** | model +1.07 ± 0.43 |
| change model, then data | model **+2.87 ± 0.45** | data +1.62 ± 0.37 |

so the data main effect brackets [+1.62, +3.42] (36–76 % of the total) and the
model main effect [+1.07, +2.87] (24–64 %), with a large negative interaction
(−1.80). The band excess mixes in the peak multiplicity, which rises by 6–8
peaks/event along both axes; the multiplicity-invariant number above is the
cleaner statement. **Neither number supports "the model got worse".**

**Within the model axis, the target-width preset is the lever.** At fixed data
and fixed training recipe, v5 differs from v4b only in the resolution preset used
to build the target histograms, and its satellite rate is **+0.0183 ± 0.0028**
higher (6.5σ), three times the whole v4b to v6 model effect.

Target σ [mm] and, in brackets, the amplitude scale `max(1, 0.15/σ)`:

| preset | nTrk = 2 | 5 | 10 | 50 |
|---|---|---|---|---|
| `hllhc` (v4b) | 0.108 (1.4×) | 0.056 (2.7×) | 0.034 (4.5×) | 0.010 (14×) |
| `hllhc_corrected` (v5) | **0.083 (1.8×)** | **0.052 (2.9×)** | 0.036 (4.2×) | 0.013 (11×) |
| `hllhc_alleta` (v6) | 0.093 (1.6×) | 0.062 (2.4×) | 0.044 (3.4×) | 0.017 (8.9×) |

v5's targets are narrower and taller than v4b's exactly at **low multiplicity**
(20–30 % narrower for nTrk = 2–5, which is where the marginal vertices are);
above nTrk ≈ 7 they are wider. So the satellite rate tracks how spiky the target
is *for weak vertices*, and making it spikier there, which is what v5 was for
and which did buy a small Pareto gain in efficiency, costs satellites. v6's
`hllhc_alleta` is wider than v5's everywhere, which is why v6 sits between v4b
and v5.

**The controls behave.** Truth moves by +0.36 ± 0.39 % in the band despite
gaining 16.2 vertices/event, so the observable does not manufacture an excess
out of a density change. AMVF's *vertex collection is bit-identical* between the
two productions (it ran at AOD level, before the ntuple track selection), so it
is a control on the metric and on the sample, not on how a vertex finder
responds to more tracks. Its band excess moves by −6.16 % purely because the
nTrk≥2 recount admits 6.2 more of its own vertices per event, diluting the
fraction.

**Why the extra tracks cost so much.** The forward tracks buy almost no
resolution (Fisher-bound gain 0.1–0.2 % at every multiplicity, since σ(z0) runs
from 65 μm at η = 0 to ~2.8 mm at |η| = 3.7) but they push **19.2 truth
vertices/event, 17.3 % of the nTrk≥2 denominator, over the threshold using
fewer than two central tracks**. Those vertices are real, poorly localised
(median achievable σ 257 μm, only 46 % better than PV-Finder's own vertex-vertex
resolution), and the network must now try to represent them. What it emits is
exactly the low, broad, marginal structure next to a stronger vertex that
[section 1](#1-why-the-bump-exists) shows the split rule converts into
satellites. This is also why ~22 % of surplus peaks sit on real nTrk<2 truth
vertices.

---

## 3. How to reduce it

Ranked by benefit per unit of cost. Everything below is measured on the same
1920 held-out events with `satellite_mechanism.py`; the reference is the
production configuration, band excess **+12.48 %**, 106.83 peaks/event,
86.05 truth-matched and 20.78 surplus peaks/event.

### 3.1 Local-centroid position estimator: free, already approved

The production estimator averaged over the **whole** above-threshold region. When
the conjoined split cuts a region in two, each half's centroid is dragged towards
its own outer flank, so the pair is pushed apart and lands in the band. Replacing
it with a centroid over a window around the region maximum:

| position estimator | band excess | peaks/event |
|---|---|---|
| full-region weighted mean (historical) | +12.48 % | 106.83 |
| local centroid ±2 bins (0.08 mm) | **+3.82 %** | 106.83 |
| local centroid ±3 bins (0.12 mm) | **+5.11 %** | 106.83 |
| local centroid ±5 bins (0.20 mm) | +6.72 % | 106.83 |

**Cost: no peaks are gained or lost.** The peak *list* is identical; only the
positions move. Matching is position-dependent, so the categories do shift a
little: the companion page measures **−0.34 efficiency points** at a fixed
matching window, against **−13 % on σ_vtx-vtx**. At the ±3 bin half-width now
landing in `peak_finding.py` this removes **59 %** of the band excess; ±2 would
remove 69 %. This is by far the best lever available and it is
already approved. The estimator change itself is written up in
[peak_position_estimator](peak_position_estimator.md), which also shows it buys
13 % on sigma_vtx-vtx and costs 0.34 efficiency points at a fixed matching
window.

> **Open discrepancy, flagged rather than papered over.** That page quotes the
> same band as an *absolute* count and reports +1.76 → +0.01 pairs/event for the
> same estimator change, i.e. essentially complete removal, where this page
> measures +12.48 % → +5.11 % of the plateau, i.e. 59 %. Converted to the same
> units with the plateau measured here (5764.5 pairs per 0.05 mm bin, 8 bins,
> 1920 events) this page's numbers are **+3.00 → +1.23 pairs/event**. The two
> disagree on both the starting value and the size of the reduction, so the
> baseline conventions differ (this page subtracts the median count for
> \|Δz\| > 3 mm). The direction and the choice of half-width are unaffected.
> Reconcile before either number goes into the note.

Caveat: this fixes the *observable*, not the underlying peak list. The surplus
peaks are still there and still counted as fakes; they simply stop being
mis-placed into the band. Items 3.3 and 3.4 are what remove them.

### 3.2 Prominence gate on the split: declined, recorded for completeness

Requiring a minimum topographic prominence before the split rule may emit a
second peak:

| prominence gate | band excess | peaks/evt lost | of which surplus | of which band satellite |
|---|---|---|---|---|
| 0.000 (production) | +12.48 % | 0 | 0 | 0 |
| 0.005 | +3.07 % | 5.07 | 3.37 | 2.32 |
| 0.010 | +0.30 % | 6.65 | 4.58 | 3.25 |
| 0.020 | −1.80 % | 9.61 | 6.85 | 4.44 |

It works (0.005 removes 75 % of the band excess) but it costs 1.70
truth-matched peaks/event, a surplus-to-real removal ratio of only 2.0:1.
**Declined.** Recorded here so the trade-off does not have to be re-derived.

### 3.3 Height floor: already in use, and it does not do what was assumed

| floor | 0.00 | 0.03 | 0.05 | 0.08 | 0.12 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|
| band excess | +12.5 % | **+13.9 %** | +10.8 % | +7.3 % | +4.2 % | −1.4 % | −4.3 % |
| peaks/evt | 106.8 | 102.5 | 97.1 | 91.7 | 85.9 | 77.1 | 69.1 |

The band excess **rises** from floor 0 to the production floor of 0.03 before
falling. The floor is a good fake-remover (4.9:1 surplus-to-real at 0.03, better
than the prominence gate) but it targets the wrong population: at 0.03 it drops
3.58 surplus peaks/event of which only 1.37 are band satellites, so it thins the
combinatorial baseline faster than the bump. Keep the floor for the fake rate;
do not expect it to fix the bump below ~0.12, which is far too expensive.

### 3.4 Retrain-level candidates, in order of expected effect

- **Replace the nearest-neighbour upsample** (`src/pv_finder/models/unet_v2.py:35`).
  Evidence in [1.5](#15-two-structural-reasons-the-network-emits-ripple-there):
  surplus peaks carry a 35 % mod-4 modulation (χ² = 854 on 3 dof) that
  truth-matched peaks do not. Cheap to try (linear interpolation, or a learned
  sub-pixel upsample), but needs a full retrain to evaluate.
- **Rebuild the target so it is drawable.** 31.6 % of nTrk≥2 truth vertices get a
  sub-bin σ with an amplitude scale up to 22×. Either cap σ at the bin width
  and drop the `0.15/σ` amplitude compensation, or move to a finer output grid
  for the peak region. The v4b/v5/v6 comparison shows the target-width preset is
  the largest model-side lever on the satellite rate, and it points the other
  way from what was assumed: **narrower targets make it worse.**
- **Objectness head or fake-aware loss.** Already on the fake-suppression
  roadmap. This is the only item that attacks the surplus peaks at the source
  rather than downstream of them.

### 3.5 Not recommended

- **Pre-smoothing before peak finding** (`--smooth-sigma`). A 0.08 mm Gaussian
  removes 31 % of band satellites but also 3.4 % of truth-matched peaks, and it
  widens the dip, which inflates σ_vtx-vtx. Off by default; leave it off.
- **NMS.** Keys on a height ratio, which does not separate satellites from
  genuine close pairs at PU200.
- **Disabling the conjoined split.** Efficiency 0.883 → 0.616. Not an option.

---

## 4. Why AMVF shows less of it, and why not zero

AMVF is not a histogram peak finder. It is an iterative annealing fit: vertex
candidates are seeded from unused tracks, every track is assigned a *soft*
weight to every candidate, and the temperature is lowered in steps so that the
fit sees a progressively finer scale. Two properties of that procedure act
directly against satellites, and PV-Finder's peak finder has neither.

**Competition.** A track's weight is shared between candidates, so a candidate
0.4 mm from a strong vertex must *take* weight away from it. A structure that is
merely a ripple on a stronger neighbour's flank loses that competition and its
candidate dissolves. PV-Finder's decision is local and unnormalised: a one-bin
non-monotonicity in a smooth field promotes a vertex regardless of whether the
tracks under it already belong to a stronger neighbour. This is the single
concrete thing PV-Finder's finder lacks.

**Scale-space annealing.** At high temperature the weight function is broad and
nearby candidates simply merge; structure finer than the current temperature
does not exist for the fit yet. The annealing schedule is a built-in noise
tolerance, calibrated in units of track χ². The split rule's tolerance is zero,
in units of nothing.

What we can show from this repo, rather than assert:

- **AMVF gains no new vertices from the forward tracks.** Between the two
  productions AMVF's vertex list is bit-identical; its nTrk≥2 count moves only
  because the recount pushes 6.2 vertices/event over the threshold
  (91.7 → 97.9), while mean AMVF-vertex nTracks rises 52 %. It attaches the new
  tracks to existing vertices instead of seeding new ones. PV-Finder, on the same
  tracks, adds 7.5 peaks/event.
- **Its satellite rate is lower but the same order**, not zero: 0.106 surplus
  vertices per real vertex in the 0.25–2.0 mm shell against 0.149 for
  PV-Finder, a factor 1.4 and not a factor 10 *(truth-anchored cross-correlation,
  carried over from the verification pass)*. It reconstructs 17.4 fake
  vertices/event against PV-Finder's 21.5 on the same events.
- **Its excess is displaced, not absent.** AMVF's band excess at 0.3–0.7 mm is
  −14.5 % simply because its dip is still recovering there: it reaches −2 % of
  the plateau only at 0.62 mm, against 0.33 mm for PV-Finder. It has its own
  excess as a shoulder at **0.9–1.1 mm, +9.0 ± 1.0 %**, that is, at *its*
  resolution scale, which is where a finite-resolution algorithm puts its
  satellites. Its peak position is not well determined (bootstrap 68 % interval
  [0.925, 1.075] mm); quote it as a shoulder, not a peak.

> **Do not use "satellites per peak" to compare the two algorithms.** That
> observable subtracts the plateau over a 0.25–2.0 mm shell, and AMVF's dip
> reaches into it, so AMVF scores −0.011 where PV-Finder scores +0.058. That is
> an artefact of the dip, not a factor-5 difference in satellite rate. Moving the
> shell outside both dips (0.35–2.0 mm) gives +0.013 (AMVF) against +0.065
> (PV-Finder). Across the PV-Finder cells in section 2, where every dip edge is
> the same 0.33 mm, the observable is clean and the choice of shell does not
> change any conclusion.

So the honest statement is: AMVF's annealing and weight competition give it a
principled significance test per vertex that PV-Finder's peak finder does not
have, which buys it roughly a factor 1.4 in satellite rate; but any algorithm
with finite resolution and a splitting criterion will place surplus vertices
just outside its own resolution, and AMVF does.

*Attribution:* the mechanism of AMVF (annealing schedule, soft track weights,
seeding, vertex dissolution) is general knowledge about the ATLAS Adaptive
Multi-Vertex Fitter, not something measured here. Everything in the bullet list
above is measured in this repo.

---

## Downstream claims that still need revisiting

Not fixed here, because they are slide decks rather than the wiki, but they now
contradict this page:

- `presentations/mattia/07_24_2026/slides.tex:766`: "a quality cut suppresses
  the residual PV-Finder **sidelobes**". They are not sidelobes; see
  [1.4](#14-what-it-is-splitting-on-is-sub-noise-ripple).
- `presentations/mattia/07_24_2026/slides.tex:806` and
  `presentations/mattia/04_16_2026/slides.tex:1315`: "Fakes are **not
  sidelobes**: none within 0.3 mm of a real peak". True as stated (5.9 % are
  within 0.3 mm) but it reads as "fakes are isolated", and only 7.2 % of surplus
  peaks are further than 3 mm from a real one. See
  [1.3](#13-the-surplus-peaks-are-a-near-neighbour-population).
- Any deck quoting σ_vtx-vtx = 0.22/0.28 mm is on the v4b/`r16438` sample, not
  this one.

## Provenance and reproducing

Measured today, from committed code, on 1920 μ-matched held-out events:
everything in sections 0, 1.2–1.5, 2 and 3, and the AMVF bullet list in 4.
Carried over from the 2026-08-04 verification pass and **not** re-derived here:
the genuine-pair decomposition (−2.6/−5.0/−12.5 %), the outward-bias profile,
the 0.149/0.106 surplus-per-real-vertex cross-correlation, the μ-independence of
the satellite rate, the tile-stitching bound, and the amplitude-correlation and
mirror-control numbers in 1.4. Those are flagged where they appear.

```bash
# 3x2 model x production grid, paired-bootstrap contrasts
PYTHONPATH=src python -u src/pv_finder/diagnostics/bump_model_vs_data.py \
    --cell v4b_old=<v4b.pth>@old --cell v4b_new=<v4b.pth>@new \
    --cell v5_old=<v5.pth>@old  --cell v5_new=<v5.pth>@new \
    --cell v6_old=<v6.pth>@old  --cell v6_new=<v6.pth>@new \
    --contrast v6_new-v4b_old --contrast v4b_new-v4b_old \
    --root-old data/run4/PU200_withTiming/..._r16443_PU200.root \
    --root-new data/run4_all_etas/.../..._r16443_PU200.root \
    --n-events 1920 --entry-stop 25000 --centroid-halfwidth 0 --device 2 \
    --output-dir outputs/08_04_2026_output/bump_model_vs_data

# mechanism: split origin, lattice, prominence, estimator and gate scans
PYTHONPATH=src python -u src/pv_finder/diagnostics/satellite_mechanism.py \
    --root data/run4_all_etas/.../..._r16443_PU200.root --ckpt <v6.pth> \
    --preset hllhc_alleta --n-events 1920 --entry-stop 25000 --device 3 \
    --output-dir outputs/08_04_2026_output/satellite_mechanism

# three-way PVF / AMVF / truth control from an existing eval pkl
PYTHONPATH=src python -u src/pv_finder/diagnostics/pairwise_dz_comparison.py \
    --pkl outputs/<date>/eval_.../eval_results.pkl --root <matching root> \
    --mu-min 185 --mu-max 215 --n-events 1920 --output-dir outputs/<date>/pairwise_dz
```

### Traps

- **`eval_results.pkl` stores every event the eval read**, not the μ-window
  subset it summarises. Select on the stored per-event `mu`. A 20.9 % figure
  once came from getting this wrong.
- **The σ_vtx-vtx sigmoid fit does not converge at 60 bins** (±10⁶ mm). Use 240.
  A fit performed at one binning must never be drawn over a histogram at
  another.
- **State the position estimator with every number.** It changed on 2026-08-04
  from the full-region weighted mean to a local centroid, and it moves the band
  excess by more than a factor of two. Every number on this page uses the
  historical full-region mean unless it says otherwise.
- **σ_vtx-vtx values are not interchangeable.** 0.2328 mm is the current v6
  held-out eval; 0.224 and 0.2465 belong to other samples and other pkls. Do not
  mix them into one table.
- `outputs/08_01_2026_output/bump_study/` is five PNGs with **no surviving
  code**. Three figures quoted from it were never reproduced. Do not cite it.
- `r16443`/`r16638` are **flat-μ** files (mean truth PV/evt 69.9), which is why
  they are held out; `r16438`/`r16633`/`601237` are the PU200 training files.
  `r16438` is inside the training pool and was historically also used as the
  eval file (measured bias: efficiency 87.5 % → 87.0 %).

## What changed and why

Corrections to earlier versions of this page, newest first.

- **2026-08-04 (this rewrite).** The headline band excess **+7.8 % → +12.5 %**
  and peak excess **+13.7 % → +17.5 %**. The published values could not be
  reproduced by the tool that produced them: the eval pkl they were computed
  from (15:35) was overwritten by a re-run (15:58), and both surviving held-out
  files now give +12.5 %. The peak-finder setting that produced the earlier pkl
  is unrecoverable. The truth and AMVF rows were unaffected and are confirmed.
  Also added: the model-vs-data decomposition, the lattice and prominence
  measurements, and the AMVF mechanism section.
- **2026-08-04, commit `203654f`:** attributed the excess to the outward bias on
  marginally-resolved pairs. The bias is real but conserves pairs and has the
  wrong shape.
- **2026-08-04, commit `b4a508e`:** claimed the bump was "PV-Finder resolving
  pairs AMVF merges, not a pathology". Refuted: the excess is carried by surplus
  peaks and the genuine-pair excess is negative.
- The claim "the excess scales as 1/n" is **overstated**. The measured log-log
  slope is −1.55 and the absolute excess saturates at ~2.9 pairs/event above
  n ≈ 40. What is solid is that the excess *fraction* is not n-independent,
  which is what rules out genuine two-vertex physics, together with the fact
  that the satellite rate per isolated real vertex is μ-independent (17.95 % at
  μ = 1–50, 20.31 % at μ = 150–215).
- Ruled out with measurements, and still ruled out: the peak-finder `min_width`
  floor (0.12 mm is reached only for σ ≈ 0.05 mm peaks; the dip edge is set by
  predicted peak *width*), a truth-convolved-with-resolution toy (no setting
  reproduces the observed band with a 1–3 mm tail), and sub-event tile stitching
  (real but ≲1 %: 5.97 % of excess-pair midpoints lie within 1 mm of a boundary
  against 4.99 % for plateau pairs; AMVF, which has no tiles, shows none).
