# The pairwise-Δz bump — a surplus-peak (satellite) artefact

**Status: independently verified 2026-08-04.** The excess is real
(16.7σ in the band), it is not the sample, and it is carried almost entirely
by *surplus* reconstructed vertices — peaks unmatched to any truth PV.
Removing them removes the bump at every matching window tried.

Two earlier conclusions on this page were wrong and are recorded in
[Corrections](#corrections).

Tools and evidence:
- `src/pv_finder/diagnostics/pairwise_dz_comparison.py` → `outputs/08_04_2026_output/pairwise_dz/`
- `outputs/08_01_2026_output/bump_study/` — five figures, **no surviving code**
  (see [Reproducibility gap](#reproducibility-gap))

## The observation

At the 240-bin binning used for the σ_vtx-vtx fit, the pairwise-Δz distribution
of reconstructed PVs shows a **+13.7 % excess at |Δz| = 0.57 mm**, just outside
the resolution dip, with a tail out to ~3 mm. At the old 60-bin binning it was
smeared away.

## What it is not

Three-way control on the same 1920 μ-matched held-out events (⟨μ⟩ = 192.5).
Uncertainties are bootstrap-over-events (4000 replicas):

| | plateau | dip depth | peak excess | at \|Δz\| | band 0.3–0.7 mm |
|---|---|---|---|---|---|
| **Truth** (nTrk≥2) | 6726 | +0.5 % | 3.4 % | 0.57 mm | **+0.57 ± 0.50 %** |
| **AMVF** (nTrk≥2) | 4989 | −99.8 % | 9.0 ± 1.5 % | 0.9–1.1 mm | **−14.5 %** |
| **PV-Finder** | 5816 | −99.6 % | **13.7 ± 1.4 %** | 0.57 mm | **+7.8 ± 0.5 %** |

- **Not the sample.** Truth is flat to within its 0.5 % statistical precision
  in the band (+0.57 ± 0.50 %, 1.1σ); in coarse 0.5 mm bands it is flat to
  <0.6 % over the full 0–6 mm range.
- **Not unique to PV-Finder.** AMVF shows the same dip-then-excess. Its peak
  location is *not* well defined — the bootstrap 68 % interval is
  [0.925, 1.075] mm — so quote it as a **shoulder at 0.9–1.1 mm**.

> "AMVF does it too" makes the effect **generic**, not benign. AMVF has the
> same satellite pathology: 0.106 surplus vertices per real vertex in the
> 0.25–2.0 mm shell, against 0.149 for PV-Finder.

## What it is: surplus peaks

Decomposing the excess by whether both pair members match **distinct** truth
vertices (greedy 1-to-1), integrated over 0.35–3.0 mm:

| matching window | total excess | genuine pairs | non-genuine |
|---|---|---|---|
| 0.3 mm | +11 626 | **−1 168 (−10 %)** | +12 052 (104 %) |
| 0.5 mm | +11 626 | **−726 (−6 %)** | +11 980 (103 %) |
| 0.8 mm | +11 626 | **−2 684 (−23 %)** | +13 832 (119 %) |

Genuine-pair excess in the band is *negative* at every window
(−2.6 % / −5.0 % / −12.5 %) against +7.8 % for all pairs. **Remove the surplus
peaks and the bump is gone.**

Four matching-free confirmations:

1. **Cross-correlation.** Around truth vertices isolated by ±5 mm,
   background-subtracted surplus density decays monotonically from 0.036 to
   0.005 per 0.1 mm between 0.25 and 1.5 mm; integrated **0.149 surplus peaks
   per real vertex** (AMVF 0.106).
2. **Not low-multiplicity real vertices.** Isolating against *all* truth
   vertices rather than nTrk≥2 moves it only 0.149 → 0.154.
3. **Supernumerary, not displaced.** Of PV-Finder peaks unmatched within
   0.3 mm, **76 %** have their nearest truth already claimed by another peak.
4. **Low amplitude.** A height floor eventually removes them — see
   [Operating point](#operating-point).

### Per-peak, not per-pair

The satellite rate per isolated real vertex is **μ-independent**: 17.95 % at
μ = 1–50, 20.31 % at μ = 150–215. Each real peak spawns satellites at a roughly
fixed rate, so satellite pairs grow like *n* while total pairs grow like *n*²,
and the excess *fraction* falls with pile-up.

> The specific claim "the excess scales as 1/n" is **not** what the data show.
> The measured log-log slope is **−1.55** (−1.7 for n ≥ 27), and the absolute
> excess saturates at ~2.9 pairs/event above n ≈ 40 rather than growing like n.
> What is solidly established is that the excess fraction is **not**
> n-independent, which is what rules out genuine two-vertex physics. Why it
> saturates is unexplained; increased merging at high density is a plausible
> but untested cause.

## What the outward bias does and does not explain

Close pairs really are pushed apart. For truth pairs where both vertices are
found, ⟨|Δz|_reco − s⟩ is **+0.120 mm at s = 0.1–0.2 mm**, falling to +0.011 at
0.4–0.5 mm and to zero by 0.7 mm.

It cannot produce the bump, for three reasons:

1. **Wrong shape.** A smooth outward shift on a flat distribution gives a
   relative excess −db/ds: +27 % at 0.3 mm falling monotonically to ~0 by
   0.7 mm. Observed is the opposite — a *deficit* of −5.9 % at 0.325 mm, a
   maximum at 0.57 mm, and a +4.2 % tail over 1–2 mm the mechanism cannot make.
2. **Redistribution conserves pairs.** It cannot lift the integral above the
   plateau; the observed integrated excess is +11 626 counts.
3. **The proposed micro-mechanism is not what the code does.** In
   `src/pv_finder/utils/peak_finding.py:109-112` the conjoined split is
   evaluated *after* bin `i` is accumulated, so the local-minimum bin and the
   first rising bin fall into the **left** region, pulling it inward. On
   synthetic double Gaussians the peak outward bias is only +0.045 mm, and for
   narrow peaks (σ = 0.10 mm) it turns **negative**.

## Ruled out

| candidate | test | verdict |
|---|---|---|
| peak-finder `min_width` floor | `min_width=3` × 0.04 mm = 0.12 mm is reached only for σ ≈ 0.05 mm peaks; for σ = 0.10/0.15 mm the minimum resolvable separation is 0.21/0.305 mm | **not the cause** — the dip edge is set by predicted peak *width*, not by the rule |
| truth ⊗ resolution (trivially expected) | toy: truth → 88 % efficiency → σ = 0.047 mm smear → hard merge at d; band ranges −12.2 % to +1.6 % over d = 0.20–0.35 mm, 1–2 mm flat | **not the cause** — no setting reproduces +7.8 % with a 1–3 mm tail |
| sub-event tile stitching (12 × 40 mm) | excess-pair midpoints within 1 mm of a boundary: 5.97 % vs 4.99 % for plateau pairs; AMVF (no tiles) shows nothing | **≲1 % of the excess** — real but minor, worth one line in the note |

## Operating point

Band 0.3–0.7 mm excess versus minimum-height floor, on held-out data:

| floor | 0.00 | 0.03 | 0.05 | 0.08 | 0.12 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|
| band excess | +7.8 % | **+11.7 %** | +9.9 % | +6.6 % | +2.9 % | −1.2 % | −5.5 % |

The excess **rises** from floor 0 to 0.03 before falling: the floor removes
baseline pairs faster than satellite pairs at first. Satellites are only
meaningfully suppressed at floors ≳0.12, far above the 0.03–0.05 currently
used. Whether that is affordable is an efficiency question, not a bump
question — see the operating-point scan.

## Reproducibility gap

The 2026-08-01 study in `outputs/08_01_2026_output/bump_study/` is **five PNG
files with no surviving code**: no script in git history, no JSON, no error
bars. Its qualitative conclusions were independently reproduced (this page),
but three specific figures quoted from it — the 1.06 / 1.56 pair-class ratios,
the 1.54× cross-correlation density, and an "excess/baseline integral
5.23 → 4.66 → 3.39 mm" — **could not be reproduced** and have been removed from
this page. Do not put them in a note without regenerating them from committed
code.

## Corrections

- **2026-08-04, commit `b4a508e`:** claimed the bump was "PV-Finder resolving
  pairs AMVF merges, not a pathology". Refuted — the excess is carried by
  surplus peaks; genuine-pair excess is negative.
- **2026-08-04, commit `203654f`:** attributed the excess to the outward bias
  on marginally-resolved pairs, and quoted the unreproducible 08-01 numbers.
  The bias is real but conserves pairs and has the wrong shape.

What survives from both: truth is flat, so this is not sample physics; and
PV-Finder resolves genuine close pairs better than AMVF — genuine-pair band
excess is −2.6 % (PVF) vs −15.9 % (AMVF). On this sample the σ_vtx-vtx fits are
**0.2465 ± 0.0025 mm (PVF)** and **0.3048 ± 0.0056 mm (AMVF)**; the
0.224 / 0.284 quoted elsewhere are v4b-on-r16438 numbers and should not be
mixed with this table.

## Reproducing

```bash
python -u src/pv_finder/diagnostics/pairwise_dz_comparison.py \
    --pkl outputs/<date>/eval_.../eval_results.pkl \
    --root data/run4_all_etas/Run4_MC21_ITk_LatestJuly2026/ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root \
    --mu-min 185 --mu-max 215 --n-events 1920 \
    --output-dir outputs/<date>/pairwise_dz
```

> **Pitfall.** `eval_results.pkl` stores every event the eval *read*, not the
> μ-window subset it summarises. The tool selects on the stored per-event μ and
> hard-caps the ROOT read at the pkl's event count — without that cap the μ
> selection walks past the pkl range and silently compares different events.
