# GNN v3 campaign (13 November 2026)

Goal (Mattia, 13 Nov): every chain metric better than AMVF at one score cut t,
Clean HS and HS purity included; then, among the best models, the fastest.
t is chosen with him afterwards, so everything is reported against t.

Baseline to beat: s8 peaks + GNN v2 e527 (39,521 params), >= 2-track vertices,
t = 0.94. Sections 1-3 used the 12 Nov cuts (0.01 / 3 / 0.184 / 0.0184, centroid
±3). From section 5 on, every evaluation uses the 13 Nov cuts in `working_points.py`
(0.01 / 3 / integral 0.186 / no height cut / centroid ±3). The training graphs were
built with the old cuts. That moves about 0.17% of the peaks, so they were not rebuilt.
Outputs: `outputs/11_13_2026_output/gnn_v3_campaign/`.

Selection set: the 7,500 r16443 GNN validation entries (25,006-99,995; 568
PU200 events, 750 at mu < 20, 375 at mu < 10). The test set (r16443 entries
0-24,999) is not touched until the final, confirmed evaluation.

## 1. Diagnosis: where the chain loses on the HS vertex

Same 7,500 validation events, GNN v2 e527, three candidate lists:

- **s8**: the production chain (`gnn_v2_freeze/infer/val_s8`).
- **truth**: the truth vertex list, with the exact truth-training graphs
  (`scripts/gnn_v3_truthcand_inference.py`). This is what v2 was trained on.
- **AMVF**: its own vertices and track links.

Composition of the HS-matched vertex: each non-HS track on it is put in one
class (`scripts/gnn_v3_hs_diagnosis.py`):

- *own_cand*: its truth vertex owns a candidate of its own, so the GNN could
  have sent it there;
- *merged_hs*: its truth vertex has no candidate, and the nearest one is the
  HS vertex (a finder merge);
- *orphan*: its truth vertex has no candidate, and the nearest one is elsewhere;
- *unlab*: the track has no truth vertex.

"Owns a candidate" uses a one-to-one matching: truth vertices with >= 2
tracks, largest first, each take the nearest free candidate within 0.17 mm.

PU200 (568 events), values in %, fractions of the tracks on the HS vertex:

| arm | Clean HS | HS purity | HS tracks on HS vtx | own_cand | merged_hs | orphan | unlab | trk eff | trk pur | Clean/truth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s8, t = 0.94 | 87.9 | 81.9 | 88.2 | 11.5 | 3.4 | 1.9 | 1.3 | 62.9 | 79.5 | 50.5 |
| s8, t = 0.97 | 94.5 | 88.1 | 81.9 | 6.5 | 3.4 | 1.0 | 1.0 | 55.9 | 85.6 | 53.8 |
| truth, t = 0.94 | 95.8 | 86.3 | 86.9 | 12.2 | 0 | 0.3 | 1.2 | 62.4 | 84.8 | 59.6 |
| truth, t = 0.97 | 98.8 | 92.0 | 79.7 | 6.9 | 0 | 0.1 | 1.0 | 53.5 | 91.2 | 60.2 |
| AMVF | 93.8 | 88.9 | 74.3 | 5.5 | 3.4 | 1.1 | 1.1 | 68.4 | 68.7 | 30.9 |

Bootstrap errors (200 event replicas, s8 t = 0.94): Clean HS ±1.4 (paired
difference to AMVF ±1.3), HS purity ±0.5 (paired ±0.35), own_cand ±0.4,
Clean/truth ±0.2. Full table, all mu bins:
`diag/hs_diag/hs_diag.md`.

What it shows:

1. **Candidate mismatch costs 8 points of Clean HS.** The same network on truth
   candidates has 95.8% Clean HS at t = 0.94, against 87.9% on s8 peaks
   (AMVF 93.8). It also costs 4.4 points of HS purity, 9 points of
   Clean/truth at PU200 (59.6 against 50.5) and 7.7 points at mu < 10
   (88.8 against 81.1; AMVF 83.8).
2. **The largest excess on the HS vertex is not a finder problem.** It is
   own_cand: pile-up tracks whose vertex has its own candidate, sent to the HS
   by the GNN. This is 11.5% of the tracks, against 5.5% for AMVF. It is
   the same on truth candidates (12.2%), so it is the network itself.
   Merged-HS is the same as AMVF (3.4%). Orphans are 0.9 points above AMVF;
   they exist only because v2 never saw a missed vertex in training.
3. **AMVF buys its HS purity with HS efficiency.** It keeps only 74.3% of the
   HS tracks on its HS vertex (25.5% go to other vertices). v2 already
   dominates AMVF on all three HS numbers at t = 0.98 (Clean HS 96.3, purity
   91.0, HS tracks 76.8%). What fails is one t for everything: at 0.98, track
   efficiency, F1 and low-mu Clean/truth fall below AMVF.
4. **Low mu is a confidence problem.** At mu < 10 on s8, Clean/truth is 87.7%
   at t = 0.5 and 81.1% at t = 0.94 (AMVF 83.8): the cut removes correct
   tracks. On truth candidates it is 88.8% at t = 0.94, so the network is less
   confident on peaks than on truth vertices.
5. **The post-GNN fit-weight cut does not help (negative result).** Each vertex
   is refitted with the 3D adaptive beam-spot fit used for the HS resolution,
   and tracks with chi2 > c are unassigned. c = 9 is the fit's own cut
   (weight 0.5 at T = 1); c = 18.42 is Athena's maxVertexChi2. At PU200,
   t = 0.94:
   - HS purity changes by only +0.2 / -0.4 points;
   - Clean HS is unchanged / -1.2 points;
   - the fraction of HS tracks on the HS vertex falls from 88.2% to 71.9 / 75.9%.

   The wrongly assigned pile-up tracks are chi2-compatible with the HS. The
   cut mostly removes real HS tracks: 26% / 21% of them end up unassigned,
   against 8% before. Not pursued.

**Consequence for the design.** Train on the deployment candidates. That gives
the Clean HS, the orphans and the low-mu loss. The own_cand excess needs a
better network: better discrimination between the HS and a nearby pile-up
candidate, multiplicity awareness and a calibrated "leave it unassigned"
decision.

## 2. Training data: s8-candidate graphs with truth labels

`gnn.data.chain_training_graphs`. The training graph is the deployment graph,
the same code path as `chain_inference`:

- s8 histogram;
- peaks with the cuts read from `working_points.py` at run time (or `--cuts`);
- kNN k = 20 graph.

Labels come from the one-to-one matching above: an edge is positive iff its
candidate is the one owned by the track's truth vertex. Tracks of missed or
merged-away vertices, of 1-track vertices and unlabelled tracks have no
positive edge, so the network learns to leave them unassigned. Ordering by
track count stops a nearby small pile-up vertex from taking the HS peak.

Built on exactly the events of the v2 truth shards (60,000 r16443 +
89,820 r16638, the same shard order) plus the 1,500-event validation monitor,
so candidates are the only difference from v2. s8 never saw either file.
Build: 3 GPUs, about 25 min, 105 GB. Output:
`data/run4_all_etas/ttva_graphs/chain_s8_v3_20261113/` (one `.json` of stats
per shard).

Checks:

- **Oracle.** The labels taken as the assignment reach Clean/truth 73.0% at
  PU200 (finder cap 73.4%), 88.3% at mu < 10 and 77.1% elsewhere, with
  Clean HS and HS purity 100% (monitor). v2 reaches 50.5%.
- **kNN coverage.** Owned candidate among the 5 / 10 / 15 / 20 nearest peaks:
  96.6 / 98.9 / 99.5 / 99.7% of labelled tracks (monitor).
- **Inference reproduction.** `gnn.evaluation.chain_inference_v3`, the
  evaluation code for v3 networks, reproduces the v2 reference on 1,000
  validation events:
  - peaks are identical;
  - top-1 targets agree in 99.9997% of tracks (one tie);
  - scores agree within 4e-5 (GPU nondeterminism);
  - the same holds for the truth arm.

Chain-like augmentation of truth graphs (the July PU200 recipe) is not run.
Its job was to imitate the deployment candidates statistically, and here
they are used directly.

## 3. Phase-1 runs

All four runs share these settings:

- the same 149,820 events and the same monitor;
- 540 shard epochs (18 passes, v2's schedule);
- batch 32, grad clip 1.0, cosine schedule to 1e-5, shuffle within each shard;
- one tmux session each. Code: `gnn.models.ttva_v3`, `gnn.training.train_ttva_v3`;
  runner `scripts/run_gnn_v3_train.sh`.

Each pair of runs differs on one axis only.

| run | tmux | GPU | network | params | loss | axis |
|---|---|---|---|---:|---|---|
| a | gnnv3_a | 0 | v2 network unchanged (GAT, hidden 32, 2 layers) | 39,521 | BCE, batch pos_weight | candidates only (vs v2 e527) |
| b | gnnv3_b | 1 | interaction network, hidden 64, 3 layers | 256,479 | BCE | architecture (vs a) |
| c | gnnv3_c | 2 | same as b (+ no-vertex head) | 260,704 | per-track softmax + no-vertex | loss (vs b) |
| d | gnnv3_d | 0 | interaction network, hidden 128, 4 layers | 1,327,530 | per-track softmax + no-vertex | capacity (vs c) |

Run a shares GPU 0 with d because GPU 3 is still in use by the peak-finding
agent. Run a is small and limited by data loading.

Why these choices:

- **Interaction network rather than a wider GAT.** Each layer updates the edge
  state from both end nodes, then aggregates into the vertex with four
  aggregators: attention, gated mean, max and the log of a gated soft count.
  The count tells a vertex how many tracks it has, which v2's
  attention-softmax and max aggregation cannot. Further vertex inputs:
  - log(1 + n) of the tracks within 1 and 3 sigma;
  - the peak width.

  All inputs pass through asinh(x / s) with fixed physical scales, followed by
  LayerNorm. In a one-pass benchmark on the monitor, a hidden-32 version
  already reached BCE 0.268 against 0.769 for the v2 network. On CPU (1
  thread, one PU200 event) it is faster than v2: about 77 against 118 ms.
- **Softmax + no-vertex loss.** The chain decides per track: take the top-1
  edge and keep it if its score >= t. Per-edge BCE scores each edge alone,
  and the batch pos_weight (about 19 at k = 20) inflates the scores: a cut at
  0.94 corresponds to a true probability of about 0.45. The per-track softmax
  with an explicit "no vertex" option models that decision directly. Its score
  is a calibrated probability that the top-1 edge is right. This should help
  where one t has to serve all mu, and on ambiguous HS / pile-up tracks.
- **Data volume and PU200 thinning are left for phase 2.** All four runs use
  the v2 events, so the comparison is clean. The builder already supports
  pile-up thinning (`--thin-seed`): it keeps the HS and a random subset of the
  pile-up vertices with their tracks, so the pile-up count is U[0, mu], then
  runs s8 on the thinned event. This turns PU200 events (r16438, r16633,
  601237) into extra flat-mu events. Caveat: s8 was trained on PU200 events
  from these files.

## 4. Evaluation protocol (all runs identical)

- `gnn.evaluation.chain_inference_v3` on the 7,500 validation entries with s8
  peaks, all runs in one pass (identical candidates). Then `chain_t_metrics`
  with `--hs-min-tracks 2`, t from 0.5 to 0.995.
- mu bins: inclusive, 0-10, ..., 190-200, and PU200 (185-215).
- Metrics:
  - Clean/truth, Clean/reco, Merged, Split and Fake per event;
  - track efficiency, purity and F1;
  - HS selection, Clean HS, HS purity, HS tracks on the HS vertex, HS
    resolution;
  - AMVF on the same events. 200-replica paired event bootstrap.
- CPU: GNN forward per PU200 event, 1 thread, interleaved round-robin over
  models, median paired ratio against v2. Parameter count.

## 5. Phase-1 results (13 Nov, 13 Nov cuts)

Checkpoints: lowest monitor loss, a e516, b e526, c e528 (each within 3e-5 of
the last epoch). Same 7,500 validation events for every model; AMVF on the same
events. Summary: `eval/phase1_abc/summary/summary.md` (every t, every mu bin;
failure lists, matched points, trade-off figures). Run d is evaluated
automatically when it finishes (tmux `gnnv3_eval_d`, output `eval/phase1_d` and the
combined `eval/phase1_all`).

**Single-t test.** For each run and each t of the grid (0 to 0.995), a (metric, bin)
pair fails if the run is worse than AMVF. There are 13 metrics in 22 bins:
inclusive, PU200 and 20 bins of width 10 in mu.

| run | t with fewest failures | strict | > 1 sigma | what fails there |
|---|---:|---:|---:|---|
| v2 e527 | 0.98 | 62 | 56 | track eff and F1 in all 22 bins, low-mu Clean/truth, Clean HS, HS sel |
| a | 0.75 | 50 | 42 | Clean HS (19 bins), HS purity (22), HS sel (5) |
| b | 0.96 | 31 | 28 | track eff (21 bins), HS purity at mu < 80 (8) |
| c | 0.55 | 30 | 26 | track eff (22 bins), HS purity at mu < 50 (5, > 1 sigma) and 70-80 |

No run has a t without failures. The conflict is the same for every run:
- **Global track efficiency against HS purity.** For b and c, track efficiency
  passes only at t <= 0.85 (b) or t <= 0.2 (c). At those cuts HS purity is 5-6
  points below AMVF (PU200: c 82.6 against 88.9).
- At the t where HS purity passes (c at t >= 0.55), track efficiency is 9-11
  points low in every bin (PU200: 57.4 against 68.4).
- In the track efficiency/purity plane, AMVF sits at the t = 0 end of our curve.
  At AMVF's own track purity, c reaches 69.7 against 68.4 at PU200 and 75.6 against
  74.3 inclusive. The ceiling at t = 0 comes from the network's assignments, not
  from the candidates: the candidate oracle reaches about 90% (section 8).
- In the HS planes every candidate-trained run lies far above AMVF. At AMVF's
  fraction of HS tracks on the HS vertex (74.3% at PU200), c reaches HS purity 96.6
  against 88.9 and Clean HS 99.1 against 93.8.

Values (model / AMVF marks: x = worse, **x** = worse by > 1 sigma, paired
bootstrap). Clean/truth, Clean/reco, track and HS columns in %, HS res in um, the
last three per event.

PU200 (185-215), 568 events:

| model | t | C/truth | C/reco | trk eff | trk pur | F1 | HS sel | Clean HS | HS pur | HS on HS | HS res | Fake | Merged | Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMVF | - | 30.9 | 35.0 | 68.4 | 68.7 | 68.5 | 94.5 | 93.8 | 88.9 | 74.3 | 9.21 | 0.060 | 41.7 | 22.1 |
| v2 e527 | 0.94 | 50.5 | 70.7 | 62.9 **x** | 79.5 | 70.2 | 95.4 | 87.9 **x** | 81.9 **x** | 88.2 | 8.42 | 0.044 | 19.5 | 3.83 |
| b | 0.96 | 54.6 | 83.0 | 58.4 **x** | 86.3 | 69.7 | 96.8 | 97.9 | 91.4 | 85.5 | 8.49 | 0.016 | 11.5 | 0.97 |
| c | 0.2 | 43.7 | 61.2 | 68.8 | 73.3 | 71.0 | 95.4 | 90.8 **x** | 82.6 **x** | 92.0 | 8.54 | 0.035 | 28.3 | 2.57 |
| c | 0.55 | 55.0 | 84.2 | 57.4 **x** | 87.3 | 69.3 | 96.8 | 98.2 | 92.2 | 84.5 | 8.58 | 0.016 | 10.6 | 0.85 |
| c | 0.7 | 54.9 | 89.2 | 50.4 **x** | 91.6 | 65.1 **x** | 96.7 | 98.8 | 95.0 | 79.5 | 8.70 | 0.009 | 7.06 | 0.33 |

Inclusive (7,500 events):

| model | t | C/truth | C/reco | trk eff | trk pur | F1 | HS sel | Clean HS | HS pur | HS on HS | HS res | Fake | Merged | Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMVF | - | 40.8 | 43.7 | 74.3 | 74.9 | 74.6 | 95.2 | 97.0 | 92.7 | 77.6 | 9.87 | 0.035 | 18.9 | 11.6 |
| v2 e527 | 0.94 | 58.0 | 75.9 | 69.4 **x** | 83.5 | 75.8 | 96.9 | 94.6 **x** | 87.7 **x** | 91.4 | 9.39 | 0.024 | 8.61 | 2.03 |
| b | 0.96 | 60.6 | 86.5 | 66.1 **x** | 88.8 | 75.8 | 97.6 | 98.5 | 93.4 | 91.0 | 9.38 | 0.009 | 5.06 | 0.43 |
| c | 0.2 | 52.5 | 69.4 | 75.0 | 78.4 | 76.6 | 96.9 | 95.7 **x** | 88.1 **x** | 95.0 | 9.37 | 0.020 | 12.3 | 1.10 |
| c | 0.55 | 60.9 | 87.5 | 65.3 **x** | 89.5 | 75.5 | 97.7 | 98.8 | 93.9 | 90.4 | 9.35 | 0.009 | 4.65 | 0.37 |

mu 0-10 (375 events):

| model | t | C/truth | C/reco | trk eff | trk pur | F1 | HS sel | Clean HS | HS pur | HS on HS | HS res | Fake | Merged | Split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AMVF | - | 83.8 | 67.7 | 95.6 | 97.8 | 96.7 | 98.7 | 99.7 | 98.1 | 83.8 | 9.13 | 0.003 | 0.21 | 1.25 |
| v2 e527 | 0.94 | 81.1 **x** | 94.1 | 84.8 **x** | 98.6 | 91.2 **x** | 99.7 | 99.5 x | 97.6 **x** | 90.2 | 8.89 | 0 | 0.05 | 0.13 |
| b | 0.96 | 86.9 | 98.1 | 95.6 | 98.6 | 97.1 | 99.5 | 100.0 | 97.8 **x** | 99.2 | 8.95 | 0 | 0.05 | 0.01 |
| c | 0.55 | 87.4 | 98.2 | 95.4 x | 98.7 | 97.0 | 99.7 | 100.0 | 97.9 **x** | 99.3 | 8.95 | 0 | 0.05 | 0.01 |
| c | 0.7 | 84.3 | 98.7 | 93.9 **x** | 99.2 | 96.4 **x** | 99.7 | 100.0 | 98.3 | 98.6 | 8.95 | 0 | 0.04 | 0.00 |

What phase 1 established:
1. **Training on s8 candidates fixes low mu.** At mu 0-10, Clean/truth is
   87.4 against v2's 81.1 and AMVF's 83.8, with track efficiency at AMVF's level.
   Clean HS at PU200 is 98.2 against 87.9 for v2.
2. **Architecture (a to b) is the second step.** a (the v2 network on candidates)
   reaches PU200 Clean HS 97.7 / HS purity 92.9 at t = 0.98, but with track
   efficiency 49.1. At the PU200 and inclusive matched points, b and c are above a:
   - track efficiency at 85% purity: PU200 60.1 against 57.4;
   - HS purity at AMVF's HS-track fraction: 96.6 against 93.8.
3. **Loss (b to c): equal performance, different scale.** At matched points
   (PU200, track purity 85%), track efficiency is 59.9 for b and 60.1 for c, F1
   70.3 against 70.4. c's score is a calibrated probability, so its useful t is
   0.5-0.7 instead of 0.95-0.99. Phase 2 uses c's loss (one failure fewer, scale
   easier to discuss).
4. **HS resolution, fakes, merged and split are not limiting.** All runs beat AMVF
   in nearly every bin. The few HS-resolution failures (2-3 isolated bins of 375
   events) are within 1 sigma.

**Run d (hidden 128, 4 layers; evaluated 13 Nov 23:50, epoch 466, `eval/phase1_all/summary`).**
Capacity buys nothing measurable on the phase-1 data:
- d's best single t is also 0.55, with 30 failures (26 beyond 1 sigma), the same
  as c.
- At every matched point d is within 0.1-0.3 points of c. At PU200 and 85% track
  purity, track efficiency is 60.3 against 60.1. At AMVF's HS-track fraction, HS
  purity is 96.7 against 96.6 and Clean HS 99.15 against 99.1. At mu 0-10 and
  AMVF's track purity, Clean/truth is 88.8 against 88.9.
- PU200 at t = 0.55, d / c: track efficiency 57.7 / 57.4, Clean HS 98.1 / 98.2,
  HS purity 92.2 / 92.2.
- No paired d-c bootstrap was run. The differences are below the paired errors of
  either run against AMVF (0.3-1 point).
- d costs 3.3x c on CPU (713 against 218 ms per PU200 event; 9.0x v2). Whether
  capacity pays with more data is phase 2's p2b.

CPU time of the GNN forward per PU200 event (1 thread, 60 r16438 events, mean
1,367 tracks and 27,330 edges, 3 passes interleaved round-robin; median of the
per-event paired ratio to v2, 16-84% band). The h* rows are untrained networks of
c's design, timed only for architecture cost. Load average 89 at start, 148 at
the end.

| model | params | median ms | ratio to v2 (16-84%) |
|---|---:|---:|---:|
| v2 e527 / a | 39,521 | 81.5 / 82.0 | 1.00 / 1.02 (0.95-1.08) |
| h32 l2 | 47,606 | 78.4 | 0.98 (0.89-1.10) |
| h32 l3 (= p2c) | 66,880 | 107.9 | 1.33 (1.22-1.51) |
| h48 l3 | 147,920 | 157.2 | 1.99 |
| h64 l2 | 185,302 | 157.9 | 2.00 |
| b / c (h64 l3, = p2a) | 256,479 / 260,704 | 216.6 / 218.1 | 2.74 (2.5-3.0) |
| h96 l3 | 581,504 | 380.3 | 4.80 |
| h96 l4 | 749,898 | 475.1 | 6.09 |
| d (h128 l4, = p2b) | 1,327,530 | 713.4 | 8.95 (8.2-9.9) |

`cpu_timing/timing_phase1_and_probes.json`.

The conflict is between a global quantity (all tracks) and an HS-vertex
quantity. A two-threshold working point (section 8) separates the two.

## 6. Phase-2 graph check: are the training candidates realistic?

s8 was trained on PU200 events of r16438 and r16633. The phase-2 training graphs
use events from both files, so their candidates could be better than at
deployment. `scripts/gnn_v3_peak_quality.py --residual` compares them with the
r16443 validation entries (never seen by s8 or any GNN): ref = 7,500 validation
entries; thin = r16438 thinned shards 00-02 (15,000 events); pu200 = r16633 shards
00-01 (10,000 events). The residual is candidate z minus truth z of the owning
truth vertex, as the half-width of the central 68% interval. Truth z comes from
ROOT; for the thinned shards the thinning RNG is replayed, checked against every
graph's track count.

| set | mu bin | events | one-to-one eff [%] | candidates / truth | unmatched / event | res68 all [um] | res68 HS [um] |
|---|---|---:|---:|---:|---:|---:|---:|
| ref | 0-20 | 750 | 85.3 ± 0.5 | 1.191 | 2.15 ± 0.06 | 36.4 ± 0.6 | 11.7 ± 0.6 |
| thin | 0-20 | 1,499 | 85.9 ± 0.4 | 1.222 | 2.41 ± 0.05 | 37.1 ± 0.6 | 12.3 ± 0.4 |
| ref | 50-100 | 1,930 | 79.4 ± 0.1 | 1.009 | 9.45 ± 0.08 | 42.6 ± 0.2 | 12.7 ± 0.3 |
| thin | 50-100 | 3,715 | 79.1 ± 0.1 | 1.021 | 10.17 ± 0.06 | 42.6 ± 0.2 | 12.8 ± 0.2 |
| ref | PU200 | 568 | 73.2 ± 0.2 | 0.908 | 19.6 ± 0.2 | 43.6 ± 0.2 | 12.6 ± 0.6 |
| thin | PU200 | 1,119 | 73.2 ± 0.1 | 0.906 | 19.5 ± 0.1 | 44.4 ± 0.2 | 13.6 ± 0.5 |
| pu200 | PU200 | 10,000 | 73.0 ± 0.04 | 0.900 | 19.7 ± 0.04 | 44.1 ± 0.1 | 12.9 ± 0.1 |

All bins: `peak_quality/peak_quality.md`. s8 is not optimistic on its own
training files: efficiency and residual agree with validation within 1 point and
1 um. The thinned events have 0.3-0.7 more unmatched candidates per event below
mu = 100. These come from unowned tracks kept at the pile-up fraction. They are
slightly harder than real low-mu events, which is acceptable for training. The
check reads the old-cut graphs; the new cuts move 0.17% of the peaks.

## 7. Phase 2 (launched 13 Nov, 17:44)

All three runs use c's loss (softmax + no-vertex) and the same data. That is 70
shards, 349,420 events, interleaved:
- the 30 phase-1 shards (149,820 events);
- 20 r16438 shards thinned to flat mu (99,800 events);
- 20 r16633 PU200 shards (99,800 events).

Each run has 1,260 shard epochs (18 passes, 2.3x phase-1's steps), the same
monitor, batch 32, cosine from 1e-3 and a 128 GiB virtual-memory limit. The
worst-case PU200 batch (the 32 largest events) peaks at 11.3 GiB for h64 l3 and
27.6 GiB for h128 l4 (`memprobe`).

| run | tmux | GPU | network | params | CPU vs v2 | axis | s / shard epoch (base / PU200) | ETA (local) |
|---|---|---|---|---:|---:|---|---|---|
| p2a | gnnv3_p2a | 1 | h64 l3 (c) | 260,704 | 2.7x | data volume (vs c) | 40 / 61 | 14 Nov about 10:00 |
| p2b | gnnv3_p2b | 2 | h128 l4 (d) | 1,327,530 | 9.0x | capacity (vs p2a) | 66 / about 110 | 14 Nov about 21:30 |
| p2c | gnnv3_p2c | 3 | h32 l3 | 66,880 | 1.3x | speed (vs p2a) | 31 / 45 | 14 Nov about 06:00 |

p2a's epoch 0 reproduces c's epoch 0 exactly (same first shard and seed).
Evaluation: `TAG=phase2_<run> GPU=<g> bash scripts/run_gnn_v3_eval.sh <run>`, then
`gnn_v3_eval_summary.py` with all runs. If p2a is not clearly above c, more data is
not the lever. 601237 thinning is then not worth building, and the two-threshold
point (section 5) is the next step.

## 8. Two-threshold working point (analysis for Mattia, 13 Nov)

Definition, deployable without truth:
1. Assign every track to its top-1 candidate if its score passes t.
2. On that assignment, select the HS vertex with the Athena rule (vertices with
   >= 2 tracks, positions from the 3D adaptive fit).
3. Tracks on the selected vertex with score < t_HS are then either:
   - unassigned (`unassign`); or
   - moved to their second-best candidate if that edge passes t, else
     unassigned (`fallback`).

All metrics are then computed on the final assignment with the single-t code. The
HS-selection metric re-runs the rule on the final vertex list, which is the
conservative choice. Grid: t = 0, 0.1, ..., 0.5; t_HS = 0.5, 0.55, ..., 0.95 (for
b also 0.96-0.99, its BCE scores sit near 1). Same 7,500 events, same 286
(metric, bin) pairs against AMVF.

Code:
- `chain_t_metrics --two-threshold`: arms t, (t, t_HS, mode) and oracle;
- `chain_inference_v3 --top2` stores the second-best edge;
- summary: `scripts/gnn_v3_two_threshold_summary.py`;
- runner: `scripts/run_gnn_v3_two_threshold.sh`.

Output: `eval/twothr_bc/summary_{b,c}/summary.md`.

Checks:
- the default single-t path reproduces the phase-1 metrics bit for bit (one chunk:
  counts, selection, residuals);
- the top-2 inference has the same peaks and top-1 targets as phase 1 (0 of 5.3M
  tracks differ, scores within 1.5e-6);
- (t = 0.5, t_HS = 0.5) is bit-identical to t = 0.5 in both modes;
- `top2_edges` matches a brute-force sort on random cases, ties included.

**Result, c: two-threshold point reaches 3 failures of 286 (single t: 30 at best).**

| choice | point | strict | > 1 sigma | remaining failures |
|---|---|---:|---:|---|
| single t (phase-1 grid) | t = 0.55 | 30 | 26 | track eff (22 bins), HS purity at mu < 50 and 70-80, HS res 2 bins |
| two-t, unassign | t = 0.1, t_HS = 0.7 | 3 | 2 | fakes 50-60, HS res 130-140, 140-150 |
| two-t, fallback | t = 0.1, t_HS = 0.7 | 3 | 1 | fakes 50-60, HS res 130-140, 140-150 |

b (BCE): single t = 0.96 has 31 failures. The best two-threshold point,
(t = 0.5, t_HS = 0.97, unassign), has 7 (3 beyond 1 sigma), all of them HS purity
at mu < 30 (0-10: 97.9 against 98.15), fakes in 2 bins and HS res in 2 bins.
Fallback gives 12. With BCE, t <= 0.5 applies almost no cut, and the HS purity at
low mu needs t_HS near 0.97. c's calibrated score is better here, which supports
its loss for phase 2.

d (`eval/twothr_bcd/summary`): the best two-threshold points have 4 strict
failures. With unassign, (t = 0.2, t_HS = 0.6) leaves 1 beyond 1 sigma (HS purity
0-10). With fallback, (0.2, 0.65) leaves 0 beyond 1 sigma; the remaining failures
are HS res in 3 isolated bins and fakes in 50-60. c has 3 strict and 1 beyond
1 sigma.

d has 7 grid points with no failure beyond 1 sigma; c has none. d reaches the AMVF
track efficiency at a higher global t (0.2 against 0.1), so it keeps more
Clean/truth (PU200 43.5 against 40.1, AMVF 30.9), but its track-efficiency margin
is thinner (PU200 68.6 against 68.4 for AMVF; c 69.4). Rows at d's fallback point
(model / AMVF):
- PU200: trk eff 68.6 / 68.4, HS purity 93.5 / 88.9, Clean HS 97.9 / 93.8;
- inclusive: trk eff 74.7 / 74.3, HS purity 95.1 / 92.75;
- mu 0-10: HS purity 98.16 / 98.15.

The first d attempt crashed on GPU 0 when run e started (cuDNN engine error, GPU
memory full). The coordinator reran it on GPU 2; the crash log is kept as
`eval_twothr_d_crash_gpu0_oom.log`.

The fake failure is 7 fakes in 350 events against AMVF's 5 (inclusive 0.022 against
0.035 per event). The HS-resolution failures sit in two isolated bins (380 and 350
events), with no trend in mu. The minimum is broad, not a lucky grid point: for
t = 0.1-0.2 and t_HS = 0.6-0.7, every point has 3 or 4 failures in both modes.
The point was chosen on the same validation events it is quoted on; the test set
will say whether the 3 survive. Fallback recovers a few tracks (track efficiency
+0.3) at no cost elsewhere.

Values (model / AMVF marks as in section 5):

| bin | point | C/truth | C/reco | trk eff | trk pur | F1 | HS sel | Clean HS | HS pur | HS on HS | HS res | Fake | Merged | Split |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PU200 | AMVF | 30.9 | 35.0 | 68.4 | 68.7 | 68.5 | 94.5 | 93.8 | 88.9 | 74.3 | 9.21 | 0.060 | 41.7 | 22.1 |
| PU200 | c, t = 0.55 | 55.0 | 84.2 | 57.4 **x** | 87.3 | 69.3 | 96.8 | 98.2 | 92.2 | 84.5 | 8.58 | 0.016 | 10.6 | 0.85 |
| PU200 | c, (0.1, 0.7) unassign | 40.8 | 56.8 | 69.1 | 71.1 | 70.1 | 96.5 | 98.1 | 94.0 | 80.3 | 8.60 | 0.039 | 31.9 | 2.71 |
| PU200 | c, (0.1, 0.7) fallback | 40.1 | 55.7 | 69.4 | 70.6 | 70.0 | 96.3 | 98.1 | 94.0 | 80.3 | 8.60 | 0.040 | 32.5 | 2.95 |
| inclusive | AMVF | 40.8 | 43.7 | 74.3 | 74.9 | 74.6 | 95.2 | 97.0 | 92.7 | 77.6 | 9.87 | 0.035 | 18.9 | 11.6 |
| inclusive | c, t = 0.55 | 60.9 | 87.5 | 65.3 **x** | 89.5 | 75.5 | 97.7 | 98.8 | 93.9 | 90.4 | 9.35 | 0.009 | 4.65 | 0.37 |
| inclusive | c, (0.1, 0.7) unassign | 50.6 | 66.5 | 74.8 | 77.1 | 75.9 | 97.6 | 99.0 | 95.6 | 86.9 | 9.38 | 0.021 | 13.6 | 1.15 |
| inclusive | c, (0.1, 0.7) fallback | 49.8 | 65.3 | 75.2 | 76.5 | 75.8 | 97.1 | 99.0 | 95.5 | 86.9 | 9.38 | 0.022 | 14.0 | 1.29 |
| 0-10 | AMVF | 83.8 | 67.7 | 95.6 | 97.8 | 96.7 | 98.7 | 99.7 | 98.1 | 83.8 | 9.13 | 0.003 | 0.21 | 1.25 |
| 0-10 | c, t = 0.55 | 87.4 | 98.2 | 95.4 x | 98.7 | 97.0 | 99.7 | 100.0 | 97.9 **x** | 99.3 | 8.95 | 0 | 0.05 | 0.01 |
| 0-10 | c, (0.1, 0.7) unassign | 89.1 | 96.8 | 96.2 | 98.2 | 97.2 | 99.7 | 100.0 | 98.3 | 98.6 | 8.95 | 0 | 0.10 | 0.01 |
| 0-10 | c, (0.1, 0.7) fallback | 88.8 | 96.3 | 96.4 | 98.0 | 97.2 | 99.7 | 100.0 | 98.3 | 98.6 | 8.95 | 0 | 0.12 | 0.01 |

The price is in the quantities with the largest margin: at PU200, Clean/truth falls
from 55.0 to 40.8 (AMVF 30.9), Clean/reco from 84.2 to 56.8 (AMVF 35.0), and merged
per event rises from 10.6 to 31.9 (AMVF 41.7). This is what a low global t costs.

**Is track efficiency reachable with these candidates? Yes; the network limits it,
not the finder.** Per mu bin, as % of the labelled tracks (the denominator of
track efficiency):
- "no s8 candidate": tracks of >= 2-track truth vertices that own no s8 candidate
  under the one-to-one 0.17 mm label matching;
- "1-trk": tracks of 1-track truth vertices, which no >= 2-track rule can count;
- the oracle (each labelled track on the candidate its truth vertex owns, same
  metric code) equals 100 minus both, exactly;
- the same matching against AMVF's own vertices is in the last column.

| bin | AMVF | c, t = 0.5 (gap) | c, two-t fallback (gap) | oracle | 1-trk | no s8 candidate | no AMVF vertex |
|---|---:|---:|---:|---:|---:|---:|---:|
| inclusive | 74.3 | 67.3 (-7.0) | 75.2 (+0.9) | 90.1 | 1.7 | 8.2 | 9.6 |
| PU200 | 68.4 | 59.6 (-8.8) | 69.4 (+1.0) | 88.0 | 1.8 | 10.2 | 11.8 |
| 0-10 | 95.6 | 95.9 (+0.3) | 96.4 (+0.8) | 97.6 | 0.6 | 1.8 | 2.4 |
| 50-60 | 82.7 | 78.7 (-4.0) | 83.6 (+0.8) | 92.4 | 1.7 | 5.9 | 7.0 |
| 100-110 | 76.2 | 69.7 (-6.5) | 77.2 (+0.9) | 90.8 | 1.7 | 7.5 | 8.7 |
| 150-160 | 71.2 | 63.1 (-8.0) | 72.0 (+0.9) | 89.1 | 1.7 | 9.2 | 10.7 |

The plain-t column is the best plain t of this run's grid (t <= 0.5); t = 0.55 is
2 points lower still. All 22 bins: `summary_c/summary.md`.

What this shows:
- s8 leaves fewer tracks without a candidate than AMVF does (10.2% against 11.8% at
  PU200). The candidate bound, 88%, is 20 points above AMVF's 68.4.
- At the best single t, the 7-9 point deficit at PU200 is the score cut, not the
  candidates. At t = 0, c reaches 69.7, so the network's top-1 choice loses about
  18 points against the oracle at PU200. These are tracks whose truth vertex
  owns a candidate but which end on another candidate, or on one their vertex
  does not dominate.
- Track efficiency against AMVF is therefore reachable. The two-threshold point
  already passes it in every mu bin by +0.5 to +1.1 points. The headroom above
  that is network discrimination, the target of phase 2.

## 9. Control run e: truth-candidate training at c's network (queued 13 Nov)

Mattia suspects truth-candidate training may be as good or better. Phase 1 does
not answer this: a vs v2 changes the candidates but compares two networks at
different operating points, and c differs from v2 in both network and candidates.
Run e isolates the candidates. It is exactly c (network, loss, 540 shard epochs,
batch 32, cosine 1e-3 to 1e-5, seed, and the same s8 validation monitor for
checkpoint selection), trained on v2's truth-vertex graphs instead. Those cover
the same 149,820 events in the same shard order (v2's 30 shards; c's shards were
built on them one to one).

Vertex features for truth candidates, as v2 had them. None needs dropping, so c
needs no reduced-feature twin:
- z: truth z;
- height: the recipe height stored in the v2 truth graphs (nonzero, after the July
  height-bug fix);
- sigma (c's `pv_sigma` input; at deployment the peak width): the preset
  resolution sigma(nTracks) of `hllhc_alleta`, with nTracks counted from
  `track.truth_pv`. This is the sigma `chain_inference_v3.truth_graph` gives truth
  candidates in the truth-candidate diagnostic (section 1). On 40 events the shard
  graphs equal `truth_graph` (pv.x, edges, labels) and the derived sigma matches
  bit for bit. It is filled at load time (`pv_sigma_preset` in
  `train_ttva_v3`).
- the multiplicity counts within 1 and 3 sigma are computed by the network from
  sigma, as for c.

Training labels follow v2: every truth vertex is a candidate, including 1-track
vertices, and each labelled track has its own vertex as the positive edge.
Unlabelled tracks have none.

Launch: GPU 0 after d finishes (expected about 23:30). GPUs 1-3 have no safe room
next to the phase-2 jobs: p2a has cached 39 GB, p2c 24 GB, and p2b's worst batch
is 27.6 GiB. Pipeline (tmux `gnnv3_e`, `scripts/run_gnn_v3_e_pipeline.sh`):
1. train;
2. single-t evaluation, as for c (`eval/phase1_e`);
3. matched points of c and e (`eval/phase1_ce/summary`);
4. top-2 inference and the two-threshold grid (`eval/twothr_e`);
5. two-threshold summary of c and e (`eval/twothr_ce/summary`).

Expected done: 14 Nov, about 08:00. Config: `configs/gnn/config_gnn_ttva_v3_e.yml`.

**e_aug not run (not cheap).** The chain-like augmentation
(`graph_augmentation`) is applied at graph-build time from statistics that
`chain_gap_decomposition` measured for v6 peaks at PU200: miss probability vs
nTracks, z jitter, sigma, height ratio and junk rate. Reusing them for s8 at flat
mu would over-drop vertices at low mu. A faithful e_aug needs those statistics
remeasured on s8 at flat mu, 30 augmented shards rebuilt (about 105 GB) and a
second GPU slot. It is worth doing only if e comes close to c.

## Reproduction

```bash
# diagnosis (tmux gnnv3_diag)
PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_truthcand_inference.py --like outputs/11_12_2026_output/gnn_v2_freeze/infer/val_s8 \
    --gnn v2e527=model_weights/ttva_gnn_hllhc_fullmu_v2_20261111/ttva_gat_alleta_k20_fullmu_v2_150k_epoch_527.pyt \
    --out outputs/11_13_2026_output/gnn_v3_campaign/diag/infer/val_truth -d 0
PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_hs_diagnosis.py --arm s8=outputs/11_12_2026_output/gnn_v2_freeze/infer/val_s8 \
    truth=outputs/11_13_2026_output/gnn_v3_campaign/diag/infer/val_truth --gnn v2e527 --out outputs/11_13_2026_output/gnn_v3_campaign/diag/hs_diag
# training graphs (tmux gnnv3_build{0,1,2})
GPU=0 PART=0 NPART=3 bash scripts/run_gnn_v3_build_chain.sh   # and PART=1,2 on GPUs 1,2
# training (tmux gnnv3_<run>)
RUN=b bash scripts/run_gnn_v3_train.sh        # add --resume to continue
# phase-1 evaluation (tmux gnnv3_eval_p1), 13 Nov cuts read from working_points.py at run time
TAG=phase1_abc GPU=1 bash scripts/run_gnn_v3_eval.sh a b c
bash scripts/run_gnn_v3_eval_d_when_done.sh   # tmux gnnv3_eval_d: waits for d, evaluates, combined summary
# CPU timing (graphs: --build once; probes = untrained h*/l* networks with a model_config.json)
CUDA_VISIBLE_DEVICES="" PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_cpu_timing.py \
    --graphs outputs/11_13_2026_output/gnn_v3_campaign/cpu_timing/r16438_pu200_60evt_s8_graphs.pt \
    --gnn v2e527=... a=... b=... c=... d=... h32l3=... --out .../cpu_timing/timing_phase1_and_probes.json
# phase-2 graph check
PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_peak_quality.py --residual --set ref=.../phase2/r16443_validation_all.pt \
    --set thin=.../phase2/r16438_thin_shard_0{0,1,2}.pt --set pu200=.../phase2/r16633_pu200_shard_0{0,1}.pt \
    --out outputs/11_13_2026_output/gnn_v3_campaign/peak_quality
# phase 2 (tmux gnnv3_p2{a,b,c}); configs written by scripts/gnn_v3_make_config.py
VMEM_KB=134217728 RUN=p2a bash scripts/run_gnn_v3_train.sh   # add --resume to continue
# two-threshold analysis (section 8): top-2 inference, metrics per model, summary
PYTHONPATH=src venv/bin/python -u -m gnn.evaluation.chain_inference_v3 --root <r16443> --entries <validation_entries.npy> \
    --candidates s8 --top2 --gnn b=... c=... --out outputs/11_13_2026_output/gnn_v3_campaign/eval/twothr_bc/infer -d 3
INFER=.../twothr_bc/infer OUT=.../twothr_bc/metrics THS="0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95" \
    bash scripts/run_gnn_v3_two_threshold.sh c
PYTHONPATH=src venv/bin/python -u scripts/gnn_v3_two_threshold_summary.py --metrics c=.../twothr_bc/metrics/c.npz --out .../summary_c
bash scripts/run_gnn_v3_two_threshold_d_when_done.sh   # tmux gnnv3_twothr_d: d after its standard eval, then b+c+d summary
bash scripts/run_gnn_v3_e_pipeline.sh                  # tmux gnnv3_e: control run e (section 9), train + full evaluation
```
