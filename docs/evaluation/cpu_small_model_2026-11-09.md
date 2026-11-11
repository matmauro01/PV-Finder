# Smaller PV-Finder for CPU inference, November 2026

Goal: a PV-Finder model that runs about 10 times faster on CPU than the
production v6 model, with no measurable loss of physics performance. The
acceptance bar is an efficiency loss well below 0.5 points, ideally below
0.2 points, with no increase in fakes and no worse vertex-vertex resolution,
on the held-out PU200 files.

## Where the CPU time goes

The measurements below use ONNX Runtime 1 thread, one event = 12 sub-events,
real PU200 events from r16438, on sneezy. sneezy is shared and heavily
loaded, so absolute times move by about 20% between invocations; compare
models measured in the same invocation
(`outputs/11_09_2026_output/cpu_bench_*`, produced by
`scripts/cpu_inference_benchmark.py`).

Arithmetic per sub-event (multiply-accumulates, all 695 slots filled):

| Model | U-Net ch | Latent | Params | MLP MAC | U-Net MAC | Total |
|---|---:|---:|---:|---:|---:|---:|
| v6 production | 280 | 4 | 3.55M | 402M | 2153M | 2555M |
| s5 | 112 | 1 | 0.67M | 135M | 344M | 479M |
| s8 | 80 | 1 | 0.44M | 135M | 176M | 311M |
| 64 ch, 64 hidden | 64 | 1 | 0.24M | 56M | 113M | 169M |

Time does not follow arithmetic alone. Operator profiling shows the
**softplus on the per-track MLP output** takes 30% of v6 time and 50% of s8
time. It runs over `n_slots x 1000 x n_latent` values per sub-event,
including padded slots, which are multiplied by zero afterwards. At PU200,
cropping each event to its widest sub-event leaves a median of 460 slots,
while a sub-event holds about 114 real tracks.

## Packed-track inference (exact, no retraining)

`src/pv_finder/models/packed_inference.py` (`PackedTracksToHist`,
`pack_event`) runs the MLP only on real tracks, packed over the 12
sub-events, and scatter-adds each track into its sub-event. It uses the
trained weights unchanged and differs from the padded model only by
summation order: max |difference| 3.8e-6 on v6 outputs, 6e-8 on the small
models, over 10 real events. Tests: `tests/test_packed_inference.py`.
The Athena tensor builder can provide the same inputs (a track list plus a
sub-event index), so this is a free speed-up for any model.

| Model | Padded (ms/event) | Packed (ms/event) | Speed-up vs v6 padded |
|---|---:|---:|---:|
| v6 | 1601 | 1159 | 1.4 |
| s5 | 358 | 250 | 6.4 |
| s8 | 279 | 178 | 9.0 |
| 64 ch, 64 hidden | 249 | 112 | 14 |

## Quantization

ONNX Runtime INT8, padded inputs, same invocation (v6 fp32 1291 ms):

- dynamic, MatMul/Gemm only: v6 1521 ms (slower), s5 321 ms vs 380 ms fp32.
  Dynamic Conv quantization produces `ConvInteger`, which the CPU execution
  provider does not implement.
- static QDQ, per-channel, 16 calibration events: v6 1037 ms (1.25x),
  s5 302 ms, s8 255 ms vs 257 ms fp32.

The gain is small because the remaining cost is the softplus (not
quantizable) and memory traffic; INT8 also needs its own physics validation.
Not pursued further until the packed small models are validated.

## Training runs

Same data, split (seed 42), loss and optimizer as v6
(`config_hllhc_alleta_v6_mse.yml`); width and schedule change only.
Schedule 1+1 epochs instead of 2+2: in v6, mini-validation efficiency was
flat from Phase-2 step 20k, while validation loss (0.0196 at 20k, 0.0175 at
100k, 0.0158 final) and fake rate (1.70 to 1.52 per sub-event) kept
improving slowly. The held-out evaluation decides whether the shorter
schedule costs anything.

| Run | Config | GPU | Started |
|---|---|---|---|
| s5 | `config_hllhc_alleta_v6_small_s5.yml` | 1 | 2026-11-09 21:19 |
| s8 | `config_hllhc_alleta_v6_small_s8.yml` | 3 | 2026-11-09 21:19 (Phase 2 resumed 2026-11-10 after a cuDNN init crash) |
| s5, 2+2 | `config_hllhc_alleta_v6_small_s5_2p2ep.yml` | 1 | 2026-11-10, about 16 h |

s5 (1+1) full validation loss 0.016352; v6 had 0.016638 after its first
Phase-2 epoch and 0.015755 at the end.

## Evaluation

`scripts/eval_small_models_heldout.sh` evaluates any checkpoint on r16443 and
r16638 with the v6 methodology (integral 0.30, min height 0.03, centroid 3,
300 pairwise bins, 185 <= mu <= 215, first 25,000 entries). v6 is re-evaluated
with the same script and code version, and `scripts/compare_heldout_evals.py`
reports paired differences with event-bootstrap errors.

## Results

Held-out PU200, 1,920 events per file, fixed 0.1694 mm window. Paired
differences vs v6, event-bootstrap errors
(`outputs/11_10_2026_output/small_model_eval/compare_*.json`):

| File | Model | Eff | dEff (pt) | FP/event | dFP | sigma vtx-vtx (mm) |
|---|---|---:|---:|---:|---:|---:|
| r16443 | v6 | 0.8316 | | 18.73 | | 0.2290 +- 0.0032 |
| r16443 | s5 (1+1) | 0.8283 | -0.33 +- 0.04 | 17.88 | -0.85 +- 0.07 | 0.2344 +- 0.0033 |
| r16638 | v6 | 0.8325 | | 18.50 | | 0.2278 +- 0.0031 |
| r16638 | s5 (1+1) | 0.8299 | -0.26 +- 0.04 | 17.71 | -0.79 +- 0.07 | 0.2329 +- 0.0031 |
| r16443 | s8 (1+1, step 450k) | 0.8236 | -0.80 +- 0.04 | 15.61 | -3.12 +- 0.07 | 0.2288 +- 0.0030 |
| r16638 | s8 (1+1, step 450k) | 0.8255 | -0.70 +- 0.04 | 15.47 | -3.03 +- 0.07 | 0.2291 +- 0.0031 |

s8 was evaluated at Phase-2 step 450,000 of 465,624 (97% of the epoch; the
run hit a host-memory error after the last step, before the epoch
checkpoint was written). It sits at a stricter operating point: 17% fewer
fakes, efficiency -0.7 to -0.8 points, sigma unchanged. The comparison at
matched fake rate decides whether this is a loss or an operating-point shift.

s5 with 1+1 epochs is inside the 0.5-point maximum but misses the 0.2-point
target. It has 0.8 fewer fakes per event, so part of the loss is a shift
along the efficiency-fake curve, and sigma is 5 um worse (about 1 sigma).
Its validation loss (0.01635) sits between v6 after one Phase-2 epoch and v6
at the end, so the shorter schedule is the first suspect. The s5 2+2 run
tests this; s8 (1+1) is evaluated when its Phase 2 finishes. Comparing at
matched fake rate (lowering s5's integral threshold) is the cheap next check.
