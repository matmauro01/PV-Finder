# Resolution Bump Analysis — HL-LHC PU200 (ep100 wide v1)

Investigation date: 2026-04-23. Data: 2500 events, ep100, integral_threshold=0.3.

## Key finding: PV-Finder's resolution is cleaner than AMVF

Comparing pairwise Δz for PV-Finder, AMVF, and truth (all flat with no
structure), the bump above baseline is quantitatively very different:

| Metric | PV-Finder | AMVF |
|--------|----------|------|
| Baseline (|dz|>3mm) | 9362 | 12321 |
| Dip depth | 99.4% below baseline | 95.9% below baseline |
| Bump max above baseline | 5.4% | 9.6% |
| Bump excess (total pairs) | 1,030 | 6,440 |
| Significant bins (>2σ) | 2 bins (marginal) | 8 bins (7-9σ) |

**AMVF has a 6x larger bump than PV-Finder.** The AMVF bump at 0.7-1.9 mm
is highly significant (7-9σ above baseline) and likely comes from AMVF vertex
splitting — one truth vertex occasionally reconstructed as two nearby AMVF
vertices.

**PV-Finder's bump is mostly noise.** Only 2 bins are marginally above 2σ.
The visual appearance of a bump comes from the transition region (±0.3-0.5 mm)
where the distribution is recovering from the very deep dip, not from excess
above the flat baseline.

## Quantitative results

### Close predicted-peak pair classification (|dz| < 2mm, all 2500 events)

| Pair type | Count | Fraction | Median |dz| |
|-----------|------:|------:|---------:|
| Genuine close (both matched to different truth) | 137,043 | 84.6% | 1.15 mm |
| Sidelobe (one matched, one fake) | 23,787 | 14.7% | 1.10 mm |
| Both fake | 748 | 0.5% | 1.25 mm |
| Split (both matched to same truth) | 484 | 0.3% | 0.30 mm |

**The bump is 85% genuine physics, 15% sidelobes.** This is inverted from
Run 2/3 (where sidelobes were 60%).

### Truth vertex spacing at PU200

| Spacing range | Count | Fraction |
|--------------|------:|------:|
| 0.0 – 0.1 mm | 13,026 | 5.3% |
| 0.1 – 0.2 mm | 12,481 | 5.1% |
| 0.2 – 0.3 mm | 11,660 | 4.8% |
| 0.3 – 0.5 mm | 21,438 | 8.8% |
| 0.5 – 1.0 mm | 42,951 | 17.5% |
| 1.0 – 2.0 mm | 55,626 | 22.7% |
| > 2.0 mm | 78,148 | 31.9% |

Median truth spacing: **1.31 mm**. 15.2% of adjacent truth vertices are
within 0.3 mm of each other (completely unresolvable). The bump region
(0.3–1.5 mm) contains 49% of all adjacent truth pairs — this is the
dominant physics.

### Pairwise Δz decomposition

In the bump region (0.3–1.5 mm), pairs where both peaks are truth-matched
contribute the majority. The excess from unmatched peaks is small (~2000
pairs) compared to the matched baseline (~94000 pairs). The bump shape is
driven by the truth vertex spacing distribution, not by sidelobes.

### Fake peak characteristics

| Property | Fake peaks | Matched peaks | Ratio |
|----------|--------:|----------:|------:|
| Height (median) | 0.053 | 0.280 | 0.19x |
| FWHM (median) | 0.40 mm | 0.28 mm | 1.43x |
| Integral ±1mm (median) | 0.065 | 0.112 | 0.58x |

Fakes are shorter and broader than real peaks. 72.8% of fake peaks are
sidelobe-like (within 1.5 mm of a matched peak). But at PU200, this is
hard to distinguish from a missed genuine vertex that happens to be near
a reconstructed one.

### Why NMS fails at PU200

Within the NMS window (0.85 mm), genuine close pairs outnumber sidelobe
pairs **5.4:1** (8943 genuine vs 1654 sidelobes in 500 events).

32.6% of genuine close pairs have height ratio < 0.3 (i.e., one vertex
is much weaker than its neighbor). This is expected — at PU200, many
vertices have few tracks and produce weak peaks.

NMS(0.85, 0.3) removes **72.1% genuine peaks** and only 27.9% sidelobes.
Fake:real ratio is 0.4:1 — it kills 2.5 real peaks for every fake removed.
This explains the efficiency drop observed with NMS.

## Implications

1. **The bump is NOT from sidelobes.** It appears identically in AMVF
   (which has no UNet). It's the natural consequence of merging
   unresolvable close vertices — shared by any reconstruction algorithm.

2. **Truth distribution is flat** — no dip, no bump. The Δz structure
   comes entirely from the reconstruction, not from the physics.

3. **NMS is harmful at PU200** because close genuine vertex pairs have
   similar height ratios to what NMS targets. Fake:real removal ratio
   is 0.4:1 — kills 2.5 real peaks per fake removed.

4. **The sigmoid fit** breaks at PU200 because of the bump. AMVF's
   fit works (σ=0.285 ± 0.011 mm) because its bump is smaller relative
   to the dip depth. PV-Finder's dip is deeper but the bump is
   proportionally larger, confusing the fit. Options:
   - Fit only |Δz| > 2 mm (flat region) to extract baseline, then
     measure dip width directly
   - Use AMVF σ as reference and compare PV-Finder's dip shape to it
   - Report dip half-width at half-depth instead of sigmoid σ

5. **PV-Finder has better resolution than AMVF** — the dip goes to
   nearly zero (99.4% depth vs 95.9%) and the bump is 6x smaller.
   PV-Finder produces a cleaner sigmoid shape overall.

## Improvement paths

### Post-processing (no retraining)

- **Don't use NMS at PU200** — more harm than good
- **Use higher integral_threshold** (0.3–0.5) to filter weak sidelobe
  peaks without losing genuine vertices (sidelobes have lower integral)
- **Alternative σ measurement**: fit only |Δz| > 2mm for baseline, then
  measure dip half-width at half-depth

### Architecture/training (retraining required)

- **The model reconstructs 83/99 vertices/event** (84% occupancy) —
  recovering the 16 missed weak vertices per event is the biggest win
- **Sidelobes exist** (15% of close pairs are one-matched-one-fake) but
  are secondary to the merging artifact shared with AMVF
- **PV-Finder's deeper dip** (vs AMVF) suggests better z-resolution;
  the bigger bump is the tradeoff of attempting to resolve more close
  pairs
