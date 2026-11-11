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
| s8 | `config_hllhc_alleta_v6_small_s8.yml` | 3 | 2026-11-09 21:19 |

## Evaluation

`scripts/eval_small_models_heldout.sh` evaluates any checkpoint on r16443 and
r16638 with the v6 methodology (integral 0.30, min height 0.03, centroid 3,
300 pairwise bins, 185 <= mu <= 215, first 25,000 entries). v6 is re-evaluated
with the same script and code version, and `scripts/compare_heldout_evals.py`
reports paired differences with event-bootstrap errors.

Results: pending.
