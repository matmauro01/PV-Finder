# GNN v3 campaign (13 November 2026)

Goal (Mattia, 13 Nov): every chain metric better than AMVF at one score cut t,
Clean HS and HS purity included; then, among the best models, the fastest.
t is chosen with him afterwards, so everything is reported against t.

Baseline to beat: s8 peaks (cuts 0.01 / 3 / 0.184 / 0.0184, centroid ±3) +
GNN v2 e527 (39,521 params), >= 2-track vertices, t = 0.94.
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

## 5. Status

Phase 1 launched 13 Nov, 11:28 local time. Measured shard-epoch times and
ETAs for 540 epochs (local time):

| run | s / shard epoch | ETA |
|---|---:|---|
| a | about 30 | 13 Nov, about 16:00 |
| b | about 38 | 13 Nov, about 17:15 |
| c | about 38 | 13 Nov, about 17:15 |
| d | about 90 | 14 Nov, about 01:00 |

GPU 0 holds a (14 GB) and d (24.5 GB), close to the 40 GB limit. If one of
them stops with an out-of-memory error, restart it with
`RUN=<run> bash scripts/run_gnn_v3_train.sh --resume`.

The softmax runs' scores are calibrated probabilities, so their useful t is
lower. After 5 epochs on the monitor: c has F1 0.79 at t = 0.5, while b has
F1 0.79 at t = 0.9-0.94. Runs are therefore compared at matched operating
points, not at the same t.

Results, phase 2 and the speed selection: to follow.

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
```
