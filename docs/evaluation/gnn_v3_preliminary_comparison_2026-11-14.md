# GNN v3: preliminary comparison for Mattia (14 November 2026)

Every network on the same 7,500 r16443 validation events (568 at PU200), all with
s8 peaks at the 13 Nov cuts, >= 2-track vertices, and AMVF on the same events.
The test set (r16443 entries 0-24,999) has not been touched. Details, methods and
checks: `docs/evaluation/gnn_v3_campaign_2026-11-13.md` (sections 5, 7, 8, 9).

## The networks

| name | network | trained on | params | CPU per PU200 event |
|---|---|---|---:|---:|
| v2 e527 | GAT, hidden 32, 2 layers (frozen) | truth vertices, 150k events | 39,521 | 82 ms (1.0x) |
| a | same as v2 | s8 peaks, same 150k events | 39,521 | 82 ms (1.0x) |
| b | interaction network, hidden 64, 3 layers, per-edge loss | s8 peaks, 150k | 256,479 | 217 ms (2.7x) |
| c | as b, per-track softmax loss with a "no vertex" option | s8 peaks, 150k | 260,704 | 218 ms (2.7x) |
| d | as c, hidden 128, 4 layers | s8 peaks, 150k | 1,327,530 | 713 ms (9.0x) |
| e | exactly c | truth vertices, 150k (control) | 260,704 | 218 ms (2.7x) |
| p2a | exactly c | s8 peaks, 349k (+ PU200 files) | 260,704 | 218 ms (2.7x) |
| p2c | as c, hidden 32 | s8 peaks, 349k | 66,880 | 108 ms (1.3x) |
| p2b | as d | s8 peaks, 349k | 1,327,530 | 713 ms (9.0x) |

CPU: GNN forward only, 1 thread, 60 PU200 events, timed interleaved and quoted as a
ratio to v2. e and p2a share c's architecture and p2c an untrained probe's, so
their times are those architectures'.

## One score cut (best t of each network)

"Failures" counts the (metric, mu bin) pairs where the network is worse than
AMVF: 13 metrics in 22 bins (inclusive, PU200, mu 0-10, ..., 190-200), 286 pairs.
The second number counts only failures beyond one paired bootstrap standard
deviation. PU200 values in %, HS resolution in um, merged per event.

| network | best t | failures | trk eff | F1 | HS sel | Clean HS | HS purity | HS res | Clean/truth | merged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMVF | - | - | 68.4 | 68.5 | 94.5 | 93.8 | 88.9 | 9.21 | 30.9 | 41.7 |
| v2 e527 | 0.98 | 62 / 56 | 51.6 | 65.2 | 94.7 | 96.5 | 91.0 | 9.00 | 54.2 | 10.1 |
| a | 0.75 | 50 / 42 | 68.8 | 69.5 | 93.5 | 71.5 | 75.7 | 8.41 | 38.2 | 33.9 |
| b | 0.96 | 31 / 28 | 58.4 | 69.7 | 96.8 | 97.9 | 91.4 | 8.49 | 54.6 | 11.5 |
| c | 0.55 | 30 / 26 | 57.4 | 69.3 | 96.8 | 98.2 | 92.2 | 8.58 | 55.0 | 10.6 |
| d | 0.55 | 30 / 26 | 57.7 | 69.4 | 96.8 | 98.1 | 92.2 | 8.56 | 55.2 | 10.6 |
| e (epoch 2) | 0.75 | 50 / 45 | 51.7 | 65.3 | 97.4 | 97.2 | 91.6 | 8.31 | 55.3 | 10.5 |
| e (last epoch) | 0.5 | 88 / 79 | 39.1 | 54.5 | 93.1 | 97.2 | 95.1 | 8.72 | 52.1 | 11.8 |
| p2a | 0.55 | 28 / 26 | 57.7 | 69.4 | 96.8 | 98.2 | 92.2 | 8.54 | 55.2 | 10.6 |
| p2c | 0.55 | 31 / 26 | 57.3 | 69.2 | 96.8 | 98.1 | 92.0 | 8.57 | 55.0 | 10.7 |

No network beats AMVF everywhere with one cut. A cut high enough to keep the HS
vertex pure throws away too many tracks overall: track efficiency ends 9-11
points below AMVF.

## Two score cuts (best t, t_HS of each network)

1. Assign each track to its best candidate if its score passes t.
2. Select the HS vertex with the Athena rule on that result.
3. Remove the HS vertex's tracks whose score is below t_HS. They are either left
   unassigned (unassign) or moved to their second-best candidate if that passes t
   (fallback).

No truth is used. The grid covers t from 0 to 0.5 and t_HS from 0.5 to 0.95 (up
to 0.99 for the per-edge-loss networks, whose scores sit near 1).

| network | best (t, t_HS) | failures | trk eff | F1 | HS sel | Clean HS | HS purity | HS res | Clean/truth | merged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMVF | - | - | 68.4 | 68.5 | 94.5 | 93.8 | 88.9 | 9.21 | 30.9 | 41.7 |
| v2 e527 | 0, 0.95 unassign | 67 / 54 | 68.7 | 69.3 | 94.9 | 88.6 | 82.6 | 8.52 | 39.7 | 34.6 |
| a | 0.4, 0.97 unassign | 28 / 14 | 68.5 | 69.3 | 95.4 | 94.9 | 89.2 | 8.39 | 37.5 | 34.7 |
| b | 0.5, 0.97 unassign | 7 / 3 | 69.2 | 69.9 | 96.1 | 96.7 | 92.0 | 8.51 | 40.2 | 32.5 |
| c | 0.1, 0.7 fallback | 3 / 1 | 69.4 | 70.0 | 96.3 | 98.1 | 94.0 | 8.60 | 40.1 | 32.5 |
| d | 0.2, 0.65 fallback | 4 / 0 | 68.6 | 71.0 | 96.3 | 97.9 | 93.5 | 8.61 | 43.5 | 28.6 |
| e (epoch 2) | 0.4, 0.75 unassign | 35 / 30 | 63.5 | 70.0 | 96.8 | 96.8 | 91.3 | 8.45 | 48.1 | 23.9 |
| p2a | 0.2, 0.65 unassign | 3 / 1 | 68.5 | 71.0 | 96.8 | 98.4 | 93.6 | 8.43 | 44.0 | 28.1 |
| p2c | 0.1, 0.65 fallback | 4 / 1 | 69.3 | 69.9 | 95.6 | 97.5 | 93.3 | 8.42 | 40.1 | 32.6 |

For v2 the grid does not reach the right region: its per-edge scores are inflated
(a cut at 0.94 means about 45% probability), and with t <= 0.5 its fakes and
low-mu purity fail. Its best stays the one-cut 62. a improves to 28 but still
fails HS purity in 11 bins.

The remaining failures of c, d, p2a and p2c are at the noise level:
- fakes in one mu bin (for c, 7 fakes against AMVF's 5 in 350 events);
- the HS resolution in 2-3 isolated bins of about 360 events;
- for p2a and p2c, HS purity at mu 0-10 by about 0.1 point (98.0-98.1 against
  98.15).

The price of the low global t: Clean/truth at PU200 drops from about 55% to
40-44% (AMVF 31%), and merged vertices per event rise from about 11 to 28-33
(AMVF 42).

## Differences between our networks (paired, same events, same cuts)

At the same (t, t_HS), for example (0.1, 0.7, fallback), with paired bootstrap
errors over 200 event replicas. PU200, model minus c, in points:

| | track eff | HS purity | Clean HS | Clean/truth |
|---|---:|---:|---:|---:|
| d | +0.12 ± 0.02 | +0.02 ± 0.25 | -0.5 ± 0.5 | +0.05 ± 0.08 |
| p2a | +0.13 ± 0.02 | +0.18 ± 0.16 | +0.2 ± 0.5 | +0.24 ± 0.09 |
| p2c | -0.10 ± 0.02 | +0.06 ± 0.15 | -0.4 ± 0.4 | -0.06 ± 0.08 |

Inclusive and mu 0-10 behave the same way: no difference exceeds 0.3 points.

## What we learned

1. **Training on the finder's peaks is what matters.** At the same network,
   training on truth vertices (e) loses about 3 points of track efficiency and
   3 points of HS purity at PU200. It also gets worse the longer it trains,
   because truth candidates look different from peaks: narrower, taller and never
   missing. At its best checkpoint it is only level with v2.
2. **The new network design is the second step** (a to b/c).
3. **Size and more data no longer help.** d (5x the parameters), p2a (2.3x the
   events) and p2c (4x fewer parameters) all sit within 0.3 points of c. The
   remaining gap to the candidate bound (88% track efficiency at PU200 if every
   track went to its own vertex's candidate) is not closed by capacity or data.
4. **Two cuts are needed to beat AMVF on every metric.** With one cut, track
   efficiency and HS purity cannot both pass.

## Recommendation

**p2c (hidden 32, 66,880 parameters, 1.3x v2's CPU time) with two cuts.** It is
the fastest network within a few tenths of a point of the best (p2a, c) on every
headline number. At (t = 0.1, t_HS = 0.65, fallback) it fails 4 of 286 pairs,
1 beyond one sigma: HS purity at mu 0-10, 98.0 against 98.15. At (0.1, 0.7,
unassign) it fails 5, none beyond one sigma. If that last bin
matters, c or p2a give 3 of 286 at 2.7x the CPU time. The exact (t, t_HS) is
for discussion: the grid shows broad minima (t 0.1-0.2, t_HS 0.6-0.7).

**Can p2b still change this?** Unlikely. p2b (hidden 128, the phase-2 data, 9x
v2's CPU) finishes about 21:30. At epoch 807 its monitor loss (0.6931) is already
below p2a's final value (0.6950). Across our runs, a gain of that size has meant
+0.1 to +0.2 points in the chain. p2b may end a few tenths above p2a, at 7x
p2c's CPU time. It will be evaluated the same way, and is worth choosing only if
a few tenths matter more than speed.

Not yet tried, and cheaper still: hidden 32 with 2 layers, as fast as v2 (1.0x)
in the timing probe.

Before any number goes into the note, the chosen network and cuts get one
evaluation on the test set.

## Update 14 Nov 22:30: p2b, and one cut (Mattia's decision 14 Nov 13:00)

p2b finished (1,260 epochs, best monitor epoch 1232) and was evaluated the same way
(`outputs/11_13_2026_output/gnn_v3_campaign/eval/phase2_p2b`, `twothr_p2b`). It is
within 0.3 points of p2a, c and p2c everywhere, as expected. Size does not pay.

Mattia chose one score cut, as before. PU200 (185 <= mu <= 215, 568 events), >= 2-track
vertices, 7,500 validation events:

| model | t | fails /286 | trk eff | trk pur | trk F1 | Clean/truth | Merged/evt | Split/evt | HS sel | Clean HS | HS purity | HS trk on HS vtx |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMVF | | | 68.4 | 68.7 | 68.5 | 30.9 | 41.7 | 22.1 | 94.5 | 93.8 | 88.9 | 74.3 |
| v2 e527 | 0.94 | 76 | 62.9 | 79.5 | 70.2 | 50.5 | 19.5 | 3.8 | 95.4 | 87.9 | 81.9 | 88.2 |
| p2c | 0.5 | 33 | 59.5 | 85.3 | 70.1 | 54.4 | 12.5 | 1.1 | 96.7 | 97.2 | 90.7 | 86.4 |
| p2c | 0.55 | 31 | 57.3 | 87.2 | 69.2 | 55.0 | 10.7 | 0.9 | 96.8 | 98.1 | 92.0 | 84.8 |
| p2c | 0.6 | 45 | 55.2 | 88.8 | 68.0 | 55.3 | 9.3 | 0.6 | 96.5 | 98.4 | 93.1 | 83.3 |
| p2a | 0.5 | 34 | 59.8 | 85.4 | 70.4 | 54.5 | 12.4 | 1.1 | 96.7 | 97.4 | 91.1 | 86.4 |
| p2b | 0.5 | 34 | 60.0 | 85.6 | 70.5 | 54.7 | 12.2 | 1.1 | 96.5 | 98.2 | 91.5 | 86.5 |
| p2b | 0.55 | 30 | 57.8 | 87.4 | 69.6 | 55.3 | 10.5 | 0.8 | 96.7 | 98.2 | 92.7 | 85.0 |

mu 0-10 (375 events), p2c at t = 0.5: track efficiency 95.9 (AMVF 95.6, v2 84.8),
F1 97.2 (96.7), Clean/truth 87.2 (83.8, v2 81.1), Clean HS 99.7 (99.7), HS purity 97.6 (98.15).

With one cut, track efficiency fails in every bin except mu 0-10, at every t that
keeps HS purity near AMVF. At t = 0.5, HS purity is below AMVF by 0.5-1.2 points
for mu below 90 and above it from mu 110 up (+1.8 at PU200).

**Proposal: p2c at t = 0.5.**
- The network gives each track a probability over its candidate vertices plus "no
  vertex". t = 0.5 assigns a track only when one vertex holds more than half of
  it: a natural cut, not a tuned one.
- Against t = 0.55 (fewest failures, 31 vs 33): +2.2 track efficiency and +0.9 F1
  at PU200, for -0.9 Clean HS and -1.3 HS purity, both still above AMVF there.
- Against v2 at 0.94, PU200: Clean HS +9.3, HS purity +8.8, Clean/truth +3.9,
  merged and split per event a third of v2's, F1 equal, track efficiency -3.4.
  At low mu it is better on everything (track efficiency +11).
- p2c is the fastest of the four (1.3x v2's CPU); p2a and p2b gain 0.1-0.4 points
  at 2.7x and 9x.

Next: Mattia approves (network, t), then one test-set evaluation.
