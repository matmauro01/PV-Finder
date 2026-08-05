# Metric definitions — the authoritative reference

Every performance number this project produces is defined here, with its exact
formula, the code path that computes it, and the population it is measured on.
Where this page and any other page disagree, **this page and the code win**;
discrepancies found while writing it are listed in §10.

Written for a physicist who has not read the codebase. Read §1 and §2 before
reading any number from this project.

Line numbers are as of commit `934930d`. `run_eval_pvf.py` carried uncommitted
local work when this was written, so it is cited by symbol rather than by line.

---

## 1. Read this first: there are two taxonomies, not one

The words **clean**, **merged**, **split** and **fake** are used by two
different classifiers that partition the *same* reconstructed vertices in
*different* ways. Both are correct. They ask different questions.

| | **Positional** | **Track-purity** |
|---|---|---|
| Code | `src/pv_finder/evaluation/vertex_finding/efficiency_res_optimized_atlas.py:224` (`compare_res_reco`) | `src/gnn/evaluation/classification.py:248` (`classify_assignments`) |
| Driven from | `run_eval_pvf_run3.py` (HL-LHC / Run 3), `run_eval_pvf.py` (Run 2 MC) | `gnn/evaluation/chain_scan.py`, `threshold_scan.py`, `evaluate_ttva*.py`, `gnn/data/pu200_chain_graphs.py` |
| Question asked | does this vertex sit within a matching window of a truth vertex? | do ≥ 70 % of the tracks assigned to this vertex come from one truth vertex? |
| Inspects | z only — never tracks | track lists only — never z |
| Needs | a vertex position and a fitted σ | a track-to-vertex assignment |
| "fake" means | no truth vertex inside its window | the plurality of its tracks belong to no truth vertex |

On the **identical 1920 events** of the r16443 held-out slice, AMVF's own
vertices score:

| AMVF, same 187.9 k vertices, same 213,738 truth PVs | clean | merged | split | fake | clean/truth |
|---|---|---|---|---|---|
| Positional | 60.9 % | 10.7 % | 0.3 % | 16.0 % | **0.6090** |
| Track-purity | 35.2 % | 42.6 % | 22.1 % | 0.07 % | **0.3094** |

A factor 1.97 on the headline. §7 explains the mechanism and gives the bridge.

**Whenever you quote a clean / merged / split / fake number, say which
taxonomy it came from.** They are not comparable and never have been.

---

## 2. The denominator rule

The single most common error is mixing truth-side and reco-side quantities.

- **Truth-side** quantities divide by the number of **truth** vertices. They
  answer *"how much of the physics did we find?"* → **efficiency**.
- **Reco-side** quantities divide by the number of **reconstructed** vertices,
  or by the number of **events**. They answer *"how much of what we produced is
  junk?"* → **fake rate**, **clean_rate**.

Efficiency and fake rate **do not sum to anything**. There is no
"efficiency + fake = 1" identity, and none of the four categories on one side
constrains the other side. The only closure identities that hold are:

```
reco-side:   clean + merged + split + fake  =  n_reco        (both taxonomies)
truth-side:  truth_clean + truth_merged + truth_missed  =  n_truth   (positional only)
```

The track-purity taxonomy has **no truth-side partition at all** — it never
decides whether a given truth vertex was found. Its only truth-side number is
`clean_per_truth`, a reco-side count over a truth-side denominator (§4).

---

## 3. The positional taxonomy (the finder evaluation)

### 3.1 How a vertex is classified

`compare_res_reco(target_PVs_loc, pred_PVs_loc, reco_res, debug)`, all
positions in **bin units** (1 bin = 0.04 mm).

1. **Candidate pairs** — for every reco *i*, every truth *j* with
   `|z_j − z_i| <= reco_res[i]` (`efficiency_res_optimized_atlas.py:258`).
   Note `<=`, and note the window is per-reco (in practice the same value for
   all of them, §5.1).
2. **Greedy 1-to-1 assignment** — all pairs sorted by distance, closest first,
   assigned greedily (`:263-269`). Ties are broken by insertion order, i.e. by
   (reco index, truth index), i.e. by z.
3. **Primaries** — the truth vertices that won a dedicated reco in step 2
   (`:274`). This set is what makes a truth vertex *clean*.
4. **Reco labels** (`:281-298`):
   - unassigned reco **with** truth in its window → **split** (`:287`)
   - unassigned reco **without** truth in its window → **fake** (`:289`)
   - assigned reco with an *unclaimed* truth also in its window → **merged**,
     and it claims that truth (`:294-296`)
   - assigned reco with nothing left over → **clean** (`:298`)
5. **Truth labels** (`:305-311`): primary → **clean**; claimed in step 4 →
   **merged**; otherwise → **missed**.

### 3.2 Metric table — positional

`N_ev` is the number of summarised events; `N_truth`, `N_reco` the totals over
those events.

| Metric | Formula (numerator / denominator) | Side | Units | Code |
|---|---|---|---|---|
| **truth PVs/evt** | Σ n_truth / N_ev | truth | per event | `run_eval_pvf_run3.py:580` |
| **reco PVs/evt** ("total reconstructed") | Σ (clean+merged+split+fake) / N_ev | reco | per event | `:588` |
| **clean** (reco) | # reco with a dedicated truth and nothing unclaimed in its window | reco | count, or per event | `:329-332` |
| **merged** (reco) | # reco that absorbed ≥ 1 extra truth | reco | count / per event | `:329-332` |
| **split** (reco) | # reco with truth in its window that a closer reco already took | reco | count / per event | `:329-332` |
| **fake** (reco) | # reco with **no** truth in its window | reco | count / per event | `:329-332` |
| **tc / tm / missed** | # truth labelled clean / merged / missed | truth | count / per event | `run_eval_pvf_run3.py:529-531` |
| **efficiency** (`Eff`) | (Σ tc + Σ tm) / Σ n_truth, over the μ-window events | **truth** | fraction | `:583` |
| **overall efficiency** | (Σ tc + Σ tm) / Σ n_truth, over **every event read** | **truth** | fraction | `:565` |
| **FP rate** (`FP`, fake/evt) | Σ fake / N_ev, over the μ-window events | **reco** | per event | `:584` |
| **clean/truth** | Σ clean(reco) / Σ n_truth | mixed | fraction | printed as the `%` column, `:588-591` |
| **σ_vtx-vtx** | sigmoid-fit half-width of the pairwise-Δz dip | — | mm | `:433-444` |

Per-event efficiency `eff` (`:532`) is `(ntc+ntm)/nt` for that one event; the
summary quantity is **not** the mean of it — it is the ratio of sums, which
weights high-μ events more. Both exist in `eval_results.pkl`; the printed
`Eff` is the ratio of sums.

### 3.3 The two efficiencies, and why they differ

The eval prints both on one line (`run_eval_pvf_run3.py:592-594`):

```
Eff=0.8648 (184837/213738)  FP=16.5974/evt  sigma=0.2200 mm  (overall eff=0.8633 1252634/1450979)
```

- `Eff` = **0.8648** is measured on the **1920 events inside the μ window**
  (`in_summary_window`, `pv_finder/utils/pairwise_dz.py:85`), i.e. rounded
  `ActualNumOfInt ∈ [185, 215]` and ≥ 1 truth vertex.
- `overall eff` = **0.8633** is measured on **all 25,000 events read**, whose
  mean μ is 99.6, not 192.5.

They are close here by coincidence: efficiency falls with pileup, but low-μ
events contribute few truth vertices each, so the sum-weighted average over a
flat-μ file lands near the high-μ value. **On a flat-μ file, `overall eff` is
not the efficiency at any pileup.** Quote `Eff` with its μ window; never quote
`overall eff` without saying it is mixed-μ.

> **Trap.** `eval_results.pkl` stores the **all-events** versions under
> `overall_efficiency` and `fp_rate_per_evt` (`:620-642`). For r16443,
> `fp_rate_per_evt = 9.638` while the printed, published FP is **16.597**.
> A script that reads the pkl top-level scalars gets the mixed-μ numbers.
> `total_clean`, `total_merged`, `total_split`, `total_fake`,
> `total_truth_*` and `n_events` are all all-events too. Only `per_event`
> lets you rebuild the μ-window summary — select on the stored per-event `mu`.

### 3.4 σ_vtx-vtx

Peaks are re-found with `--integral-threshold-res` (default 0.5, independent of
the performance threshold), all within-event pairwise differences `z_i − z_j`
are histogrammed symmetrically over ±6 mm, and a sigmoid
`a/(1+exp(b(rcc−|x|))) + c` (`run_eval_pvf_run3.py:136-138`) is fitted; σ is
`|popt[3]|`, the dip half-width in **mm**.

- It measures the **two-vertex separation** the finder can resolve, not the
  single-vertex position residual. It is not the same quantity as the residual
  RMS against truth, and it is systematically larger.
- It is sensitive to the histogram binning. Use `--pairwise-bins 300`
  (0.04 mm, commensurate with the model grid); 240 bins beats against the
  position quantisation and 60 bins biases σ high. See
  `docs/research/resolution_plot_ripple.md`.
- **It is fed back as the matching window** (§5.1). σ is not only a reported
  number; it is an input to efficiency and the fake rate.

### 3.5 AMVF under the positional taxonomy

The finder eval classifies AMVF's vertices against the same truth with the same
window (`run_eval_pvf_run3.py:550-552`) and prints an AMVF block
(`:595-603`). **It never prints AMVF's truth-side efficiency.** The block has
only reco-side counts, expressed as percentages of the *truth* count — so the
"clean %" line there is `clean/truth`, not `clean_rate`. AMVF's
`tc/tm/missed` are not computed at all, and there is no AMVF efficiency
anywhere in the finder output. If you need one, it has to be measured
separately.

**Do not use this block as a benchmark.** It scores AMVF with PV-Finder's
window (§5.1). The corrected, common-window comparison lives in
[vertex_finding.md](vertex_finding.md).

---

## 4. The track-purity taxonomy (the TTVA / chain evaluation)

### 4.1 How a vertex is classified

`classify_assignments(matched_tracks_per_pv, pt_event, truth_track_indices,
truth_pv_indices, truth_pvs_count)`.

For each reco vertex, count how many of its assigned tracks come from each
truth vertex; tracks with no truth vertex go into a bucket literally keyed
`"Fake"` (`classification.py:285-299`). Let `w_total` be the number of
assigned tracks and `max_key` the **plurality** bucket (`:314`).

- `max_key == "Fake"` → **Fake** (`:319-320`)
- `max_value / w_total >= PURITY_THRESHOLD` (0.70,
  `pv_finder/utils/constants.py`) → **Clean** (`:321-322`)
- otherwise → **Merged** (`:323-324`)
- then: among all reco vertices sharing the same plurality truth vertex, keep
  the one with the largest Σ pT² and demote **every other one** — Clean or
  Merged alike — to **Split** (`:328-349`)

A vertex with **zero** assigned tracks takes `max_key = "Fake"` (`:311-312`)
and is booked as Fake. See the drop-empty convention, §8.6.

### 4.2 Metric table — track-purity

| Metric | Formula | Side | Units | Code |
|---|---|---|---|---|
| **n_reco** | number of reco vertices classified | reco | count | `classification.py:369` |
| **n_truth** | number of truth PVs with **nTrk ≥ 2** | truth | count | `:370`, set by the caller |
| **clean_rate** | clean / n_reco | reco | fraction | `chain_scan.py:109` |
| **merged_rate** | merged / n_reco | reco | fraction | `:110` |
| **split_rate** | split / n_reco | reco | fraction | `:111` |
| **fake_rate** | fake / n_reco | **reco** | fraction | `:112` |
| **clean/truth** (`clean_per_truth`) | clean / n_truth | **mixed** | fraction | `:113` |
| **trkEff** | n_correct / n_truth_tracks | truth (track) | fraction | `chain_scan.py:116` |
| **trkPur** | n_correct / n_assigned_truth | reco (track) | fraction | `:117` |
| **trkF1** | 2·trkEff·trkPur / (trkEff + trkPur) | — | fraction | `:126` |
| **assigned_fraction** | n_assigned / n_tracks_total | reco (track) | fraction | `:127` |
| **HS-ID** | # events whose max-Σ pT² vertex has the true HS vertex as its plurality truth / # events with exactly one HS truth vertex | event | fraction | `hs_id_pu200.py:42,52,144` |

Track-level counts (`threshold_scan.py:149-164`):

- `n_assigned` — tracks the method put on some vertex,
- `n_assigned_truth` — of those, the ones that have a truth vertex,
- `n_correct` — of those, the ones whose **own** truth vertex equals the
  **plurality truth vertex of the vertex they were assigned to** (`:163`).

So `trkEff` is truth-side (denominator = all truth-associated tracks in the
event) and `trkPur` is reco-side (denominator = truth-associated tracks the
method chose to assign). **`trkEff` falls and `trkPur` rises as the score
threshold rises**, because a higher threshold leaves more tracks unassigned:
measured on the v4 chain, t = 0.90 → 0.999 moves trkEff 0.672 → 0.124 and
trkPur 0.752 → 0.960. Quoting either alone is meaningless; quote the pair or
`trkF1`, and always with the threshold.

**`clean/truth` is the headline of the chain evaluation.** It is the number of
reco vertices that are pure enough *and* won their truth vertex, over the
number of truth vertices — so it charges both missing a vertex and splitting
one. It cannot exceed 1 but it is not an efficiency: nothing guarantees each
truth vertex is counted at most once by the *positional* meaning of "found".

### 4.3 Bounds

Three ceilings are quoted alongside `clean/truth`, all on the same slice.

| Quantity | Definition | Code | r16443 (1920 evt) |
|---|---|---|---|
| **finder cap** | greedy 1-1 |Δz| < 0.5 mm match of peaks to truth, matched / n_truth. The fraction of truth vertices that have a peak at all. Association-free. | `chain_gap_decomposition.py:46,132-134` | **0.7809** |
| **oracle** | assign every truth-associated track to the peak nearest **its own true vertex** (< 1 mm), then classify with `classify_assignments`. What a *perfect* associator would get on the peaks we actually found. | `chain_gap_decomposition.py:153-214` | **0.7280** |
| **truth-graph ceiling** | run the GNN on graphs whose PV nodes are the **truth** vertices, not peaks. Isolates the associator from the finder. | `evaluate_ttva_graphs.py` | **0.7691** (t = 0.95) |
| **fraction-of-oracle** | (chain clean/truth) / (oracle clean/truth) | derived, not in code | 0.6843 / 0.7280 = **0.940** |

The finder cap is a **0.5 mm** window, four times looser than the eval's ~0.22 mm
matching window, so it is *not* comparable to the positional efficiency
(0.8648). It is the bound on the chain, not on the finder eval.

The oracle is below the finder cap (0.7280 < 0.7809) because a peak can be
found and still be unusable: 20,653 oracle-Merged vertices are peaks that sit
between two truth vertices and collect tracks from both, no matter who does
the assigning.

---

## 5. The matching window, in detail

### 5.1 The window is derived from our own resolution

```
sigma, serr, popt = fit_dz(dz_arr)          # run_eval_pvf_run3.py:468
sig_bins = sigma / BIN_WIDTH                # :492
res, tc_arr, _ = compare_res_reco(t_bins, p_bins, sig_bins * np.ones(np_), 0)   # :526-527
```

and the same `sig_bins` is handed to AMVF (`:518`, `:551`).

**Consequences, all of them uncomfortable:**

1. The window is **±1 σ_vtx-vtx**, not a fixed physical distance. Improving
   the model's resolution *tightens* its own acceptance criterion.
2. AMVF is judged with **PV-Finder's** window even though AMVF's own
   σ_vtx-vtx is 0.3048 mm — the window applied to it is 28 % tighter than its
   own resolution (equivalently, AMVF's σ is 39 % wider than ours). AMVF is
   being asked to place vertices to our precision. **This makes the AMVF block
   of the finder eval unusable as a benchmark**; see the common-window
   comparison in [vertex_finding.md](vertex_finding.md), which is the corrected
   version and supersedes any "more vertices than AMVF with fewer fakes"
   claim taken from the eval's own output.
3. Any change that moves σ moves efficiency and the fake rate even when the
   peak list is bit-identical.

Measured on the stored r16443 peak list (1920 events, 193,945 peaks, **the
peak list never changes** — only the window):

| σ used as window (mm) | efficiency | fake/evt | clean | merged | split | fake |
|---|---|---|---|---|---|---|
| 0.2190 | 0.8643 | 16.637 | 138,005 | 22,263 | 1,735 | 31,942 |
| **0.2200** (published) | **0.8648** | **16.597** | 138,012 | 22,312 | 1,754 | 31,867 |
| 0.2283 | 0.8688 | 16.331 | 137,988 | 22,709 | 1,892 | 31,356 |
| 0.2328 | 0.8709 | 16.187 | 137,945 | 22,948 | 1,973 | 31,079 |
| 0.5000 | 0.9332 | 9.589 | 138,402 | 28,515 | 8,618 | 18,410 |

The 0.2200 and 0.2283 rows keep the positions in the stored `float32` (the
0.2200 row then reproduces the published run bit-for-bit); the others promote to
`float64`. The two differ by at most one vertex between split and fake — §5.2.

A 6.3 % change in σ (0.2190 → 0.2328) is worth **+0.66 efficiency points and
−0.45 fake/event**, with no change whatever to what the model produced. The
bootstrap standard error over events at σ ≈ 0.22 is ±0.0007 on efficiency and
±0.097 on fake/evt (2000 replicas), so this is far outside the statistical
noise — and because it is the same events and the same peaks, the *paired*
uncertainty on the difference is smaller still.

Note also that reco-side **clean** is nearly flat across the whole range
(138,005 → 138,402) while merged, split and fake move a lot: widening the
window converts fakes into splits and cleans into merged. Efficiency rises
almost entirely through **merged**, i.e. through crediting close pairs that
share one peak.

Reproduce the table from the repo root:

```bash
python -u outputs/08_05_2026_output/metric_taxonomy/code/window_probe.py
python -u outputs/08_05_2026_output/metric_taxonomy/code/exact_probe.py   # §5.2
```

Output committed at `outputs/08_05_2026_output/metric_taxonomy/`
(`window_probe.json`, `window_probe.log`). Both scripts read only the stored
peak and truth lists — no model, no GPU, ~30 s.

### 5.2 The window is a knife edge at float precision

Positions are stored as `float32` (`pv_locations_updated_res` returns
`np.float32`). Re-running the published r16443 configuration reproduces
`clean 138012 / merged 22312 / split 1754 / fake 31867` **exactly** when the
stored `float32` positions are used, and gives `split 1753 / fake 31868` when
they are promoted to `float64` before `mm_to_bins`. One vertex in 193,945 sits
within 10⁻⁷ of the `<=` in `efficiency_res_optimized_atlas.py:258`. Harmless
at this scale; worth knowing before chasing a one-count difference.

---

## 6. Worked example

`src/pv_finder/diagnostics/metric_worked_example.py` builds one synthetic event
and runs it through **both production classifiers** (not a reimplementation:
the positional labels are parsed out of `compare_res_reco`'s own `debug=1`
trace and cross-checked against its returned totals).
`tests/test_metric_worked_example.py` pins every number below.

```bash
python -m pv_finder.diagnostics.metric_worked_example
pytest tests/test_metric_worked_example.py
```

### The event

Six truth vertices, six reconstructed ones, 65 tracks. Matching window
0.22 mm — the fitted σ of the v6 operating point.

| truth | z (mm) | nTrk | role |
|---|---|---|---|
| T0 | −5.00 | 15 | reconstructed by P_b |
| T1 | −4.70 | 10 | no peak of its own; P_b absorbs 6 of its tracks |
| T2 | −0.08 | 15 | close pair with T3, 0.20 mm apart |
| T3 | +0.12 | 12 | close pair with T2 |
| T4 | +8.10 | 9 | isolated; P_e and P_f share its tracks |
| T5 | +3.00 | **1** | a real interaction, **dropped by the nTrk ≥ 2 cut** |

| reco | z (mm) | tracks assigned |
|---|---|---|
| P_a | −5.14 | 6 of T0 — split-off, **inside** the window of T0 |
| P_b | −5.02 | 9 of T0 + 6 of T1 — right z, contaminated track list |
| P_c | 0.00 | 15 of T2 + 8 of T3 — one peak over a close pair |
| P_d | +3.00 | T5's single track + 3 truthless tracks — junk peak on T5 |
| P_e | +8.10 | 5 of T4 |
| P_f | +8.42 | 4 of T4 — split-off, **outside** the window of T4 |

### Positional pass, step by step

Truth is filtered to nTrk ≥ 2, so **T5 is not in the truth list**: 5 truth
vertices. Pairs within 0.22 mm, sorted closest-first:

```
(P_e, T4, 0.00)  (P_b, T0, 0.02)  (P_c, T2, 0.08)  (P_c, T3, 0.12)  (P_a, T0, 0.14)
```

Greedy: `P_e→T4`, `P_b→T0`, `P_c→T2`. `(P_c,T3)` is refused (P_c already
assigned), `(P_a,T0)` is refused (T0 already taken). Primaries = {T0, T2, T4}.

| reco | why | label |
|---|---|---|
| P_a | unassigned, T0 in its window | **split** |
| P_b | assigned, nothing unclaimed in its window | **clean** |
| P_c | assigned to T2, **T3 unclaimed in its window** → absorbs it | **merged** |
| P_d | unassigned, no truth in its window (T5 was filtered out) | **fake** |
| P_e | assigned, nothing left over | **clean** |
| P_f | unassigned, T4 is 0.32 mm away — **outside** the window | **fake** |

| truth | label |
|---|---|
| T0, T2, T4 | **clean** (primaries) |
| T3 | **merged** (claimed by P_c in pass 2) |
| T1 | **missed** |

```
reco  : clean 2   merged 1   split 1   fake 2      n_reco  = 6
truth : clean 3   merged 1   missed 1              n_truth = 5
efficiency  = (3 + 1) / 5 = 0.8000
fake/evt    = 2.0
clean/truth = 2 / 5 = 0.4000
```

### Track-purity pass, step by step

`truth_pvs_count = 5` (T5 has nTrk = 1). The truth **adjacency**, however,
still contains T5 — see §8.7.

| reco | contributions | purity | pre-demotion | final |
|---|---|---|---|---|
| P_a | {T0: 6} | 6/6 = 1.000 | Clean | **Split** (T0 also claimed by P_b, Σ pT² 15 > 6) |
| P_b | {T0: 9, T1: 6} | 9/15 = **0.600** < 0.70 | **Merged** | Merged |
| P_c | {T2: 15, T3: 8} | 15/23 = **0.652** < 0.70 | **Merged** | Merged |
| P_d | {T5: 1, **Fake: 3**} | plurality is the truthless bucket | **Fake** | Fake |
| P_e | {T4: 5} | 5/5 = 1.000 | Clean | Clean |
| P_f | {T4: 4} | 4/4 = 1.000 | Clean | **Split** (T4 also claimed by P_e, Σ pT² 5 > 4) |

```
reco  : clean 1   merged 2   split 2   fake 1      n_reco  = 6
n_truth (nTrk >= 2) = 5
clean/truth = 1 / 5 = 0.2000     clean_rate = 1/6 = 0.1667     fake_rate = 1/6 = 0.1667
```

### Side by side

| vertex | positional | track-purity | |
|---|---|---|---|
| P_a | split | Split | agree |
| P_b | **clean** | **Merged** | **disagree** — right z, 60 % pure track list |
| P_c | merged | Merged | agree |
| P_d | fake | Fake | agree |
| P_e | clean | Clean | agree |
| P_f | **fake** | **Split** | **disagree** — pure tracks, 0.32 mm outside the window |

**clean/truth: positional 0.400, track-purity 0.200 — a factor of exactly 2 on
six vertices.** The two disagreements point in opposite directions, so this is
not one convention being uniformly stricter: P_b is demoted by the purity cut,
P_f is demoted by the position window. On real data the P_b mechanism
dominates, which is why positional > track-purity there too (0.609 vs 0.309).

### Variants — one convention changed at a time

| variant | change | efficiency | fake/evt | reco C/M/S/F | clean/truth |
|---|---|---|---|---|---|
| base | — | 0.8000 | 2.0 | 2 / 1 / 1 / 2 | 0.400 |
| **(W)** | window 0.22 → 0.35 mm | **1.0000** | 1.0 | 1 / 2 / 2 / 1 | 0.200 |
| **(N)** | truth **not** filtered to nTrk ≥ 2 | **0.8333** | **1.0** | 3 / 1 / 1 / 1 | 0.500 |

(W): widening the window by 0.13 mm takes efficiency from 0.80 to 1.00 —
nothing about the reconstruction changed. P_f stops being a fake and becomes a
split; P_b stops being clean and becomes merged.

(N): keeping the nTrk = 1 vertex T5 in the truth list turns P_d from a **fake**
into a **clean match**, drops fake/evt from 2 to 1, and changes both the
efficiency numerator and its denominator (5/6 = 0.8333 instead of 4/5 = 0.8000).
The vertex P_d is a real interaction either way. Only the convention moved.

### (O) Reco-side labels are not mirror-symmetric in z

Four truth vertices at −0.15, 0.00, 0.15, 0.30 and two reco at 0.00 and 0.30,
window 0.22 mm. Both reco vertices have an unclaimed truth vertex in their
window; `compare_res_reco` walks the reco list in index order and lets the
first one it reaches absorb the shared neighbour.

| | clean | merged | split | fake | efficiency |
|---|---|---|---|---|---|
| nominal | 1 | 1 | 0 | 0 | 1.0000 |
| reflected in z | **0** | **2** | 0 | 0 | 1.0000 |

**Truth-side efficiency is invariant** (the set of truth vertices claimed by
*some* assigned reco does not depend on the walk order) **but the reco-side
clean/merged split is not.** Measured on real data: reflecting the whole
r16443 μ-window slice moves 23 vertices out of 193,945 from clean to merged
(0.012 %) and leaves split, fake, `tc`, `tm` and efficiency bit-identical.
Small, but it means reco-side `clean` carries a z-direction bias, because the
peak finder always returns peaks in ascending z.

---

## 7. The bridge between the two taxonomies

### 7.1 The mechanism

`gnn/diagnostics/amvf_convention_check.py` measures the dominant-truth purity
of every AMVF vertex on the r16443 slice
(`outputs/08_05_2026_output/gnn_ttva_v4/amvf_convention/amvf_convention.json`):

| | value |
|---|---|
| vertices with ≥ 1 track | 187,960 |
| median tracks/vertex | 11 |
| **median dominant-truth purity** | **0.6316** |
| quantiles (10/25/50/75/90) | 0.333 / 0.462 / 0.632 / 0.812 / 0.938 |

The distribution **straddles the 0.70 cut**. Scanning the cut:

| purity cut | 0.3 | 0.4 | 0.5 | 0.6 | **0.7** | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| fraction above | 0.947 | 0.853 | 0.737 | 0.561 | **0.407** | 0.280 | 0.143 | 0.072 |

At extended \|η\| AMVF puts vertices at approximately the right z — so the
positional taxonomy calls them clean — while their track lists absorb forward
tracks from neighbours, so the median vertex is 63 % pure and lands on the
wrong side of a 70 % cut.

### 7.2 The exact decomposition

The 0.407 above is **not** `clean_rate` (0.352), because the purity scan counts
vertices *before* the split demotion. The arithmetic closes exactly
(verified, closure error 0):

```
187,960 vertices
├─ purity >= 0.70 :  76,541  ->  Clean  66,127  +  Split 10,414
└─ purity <  0.70 : 111,419  ->  Merged 80,104  +  Fake   135  +  Split 31,180
                                                          Split total 41,594  ✓
```

So the factor of two on `clean/truth` has two additive causes, both worth about
half:
- **the purity cut**: 59 % of vertices are < 70 % pure and become Merged;
- **the split demotion**: a further 10,414 vertices are pure enough but lose
  the Σ pT² contest for their truth vertex.

The positional taxonomy has no analogue of the second — its split rate is
0.3 % for AMVF, because two AMVF vertices are rarely inside one σ of each
other, while two AMVF vertices *very often* share a dominant truth vertex.

### 7.3 Expectation table: given one number, what should the other be?

Rules of thumb on PU200 extended-\|η\| data, from the measurements above.
They are *not* transferable to other samples (§9).

| you have | you should expect | why |
|---|---|---|
| positional clean/truth ≈ 0.61 (AMVF) | track-purity clean/truth ≈ 0.31 | ÷ 1.97; purity cut + split demotion |
| positional fake/evt ≈ 17.8 (AMVF) | track-purity fake_rate ≈ 0.0007 | almost every vertex has ≥ 1 truth track; positional fakes are peaks off-window, not trackless vertices |
| track-purity fake_rate ≈ 0.002 (chain, drop-empty) | positional fake/evt ≈ 16.6 | same reason, from the other side |
| track-purity clean_rate high at high threshold | says nothing about position | raising the GNN threshold discards ambiguous tracks; purity rises trivially |
| positional efficiency 0.865 | **nothing** about track-purity clean/truth | the positional taxonomy has a truth-side partition; the track one does not |

The last row is the important one. **There is no positional analogue of
`clean/truth`'s split penalty, and no track-purity analogue of efficiency.**
The nearest honest comparison of the two taxonomies is
`positional clean/truth` (0.6457 for PV-Finder, 0.6090 for AMVF) against
`track-purity clean/truth` (0.6843 for the chain, 0.3094 for AMVF) — and even
that compares different objects, because the chain's reco list is the peak
list with empty vertices dropped.

---

## 8. Every trap

### 8.1 The matching window is our own σ, applied to AMVF too

§5.1. The window is `sigma_vtx_vtx / 0.04` bins
(`run_eval_pvf_run3.py:492`) and is passed to `compare_res_reco` for
PV-Finder (`:527`), for AMVF (`:551`), and for AMVF in zero-peak events
(`:518`). It is not a fixed physical quantity; it moves whenever our
resolution moves, whenever the pairwise-Δz binning changes, and whenever the
event selection the fit runs on changes.

### 8.2 Truth is filtered to nTrk ≥ 2, and the discarded vertices are real

`run3_io.py:46-55` (`_filter_amvf`) is applied to `TruthVertex` at `:157-163`
and to AMVF at `:149-152`.

Measured directly on r16443, μ ∈ [185, 215], 1920 events:

| per event | count |
|---|---|
| truth vertices with nTrk ≥ 2 (the denominator) | **111.32** |
| truth vertices with nTrk = 1 (**discarded**) | **23.13** |
| truth vertices with nTrk = 0 | 0.00 |
| all `TruthVertex` entries | 134.45 |
| ⟨μ⟩ (ActualNumOfInt) | 192.5 |

A peak landing on one of the 23.13 discarded interactions **counts as our
fake**, because the matcher cannot see it. Quantified independently in
`outputs/08_05_2026_output/ripple_study/fake_decomposition.json`: 29.5 % of
surplus peaks sit within the window of a real nTrk = 1 vertex, against a 5.3 %
accidental level, so **≈ 4.2 of the ~16.6 fake/event are real interaction
vertices**. Variant (N) of the worked example shows the mechanism on one event.

**This is not a PV-Finder-only handicap.** At a common 0.5 mm window on the same
events, AMVF also books 2.36 ± 0.05 vertices/event on real nTrk = 1
interactions against PV-Finder's 3.33 ± 0.05, so the net correction in
PV-Finder's favour is ≈ 0.97/event, not 3.3. Anyone applying this decomposition
to a *comparison* rather than to PV-Finder alone must subtract AMVF's share —
see [vertex_finding.md](vertex_finding.md).

Note also that only 134.45 of ⟨μ⟩ = 192.5 interactions appear in
`TruthVertex` at all. Even the truth list is a reconstruction-level object.

Keep the convention — it is what AMVF is counted with — but never quote the
fake rate without the decomposition.

### 8.3 AMVF's vertex list is groomed; ours is not

`_filter_amvf` removes AMVF vertices with nTracks < 2 before they are scored
(`run3_io.py:149-152`). **No equivalent cut is applied to our peak list** —
we cannot apply one, because a peak has no track count until the associator
runs. Measured on the same 1920 events: the finder eval sees 187,908 AMVF
vertices (97.87/evt) while the chain, which reads `RecoVertex_assocTracks`
without the cut (`pu200_chain_graphs.py:118-124`), sees 187,960 (97.90/evt).
Only 52 vertices here, so the grooming is nearly free at PU200 — but the
asymmetry is structural, and it always runs in AMVF's favour.

### 8.4 σ was fitted on a different population than the summary — fixed, but not in the published outputs

**Fixed on 2026-08-05** by commits `2e55f79` and `934930d`. Current behaviour
(`run_eval_pvf_run3.py:446-481`): pairwise Δz is kept per event, and when a μ
window is in force and selects a strict non-empty subset, σ is fitted on
**exactly the events the summary is quoted on**, via the shared predicate
`in_summary_window` (`pv_finder/utils/pairwise_dz.py:85`). The all-events σ is
still printed, labelled *"mixed-μ; NOT the headline"*. An empty window warns; an
all-inclusive window says so.

**The published r16443 / r16638 outputs predate the fix.** Their
`INVOCATION.txt` records the working tree committed as `53d1967`, which is
before `2e55f79`. So the σ = 0.2200 mm in
`outputs/08_05_2026_output/eval_v6_operating_point/r16443/` was fitted on all
25,000 events read (⟨μ⟩ = 99.6) and then used as the matching window for a
summary quoted at ⟨μ⟩ = 192.5.

Size of the effect: the μ-window σ at 300 bins is **0.2283 ± 0.0022 mm**
(`outputs/08_05_2026_output/ripple_study/binning.json`, measured on
μ ∈ [185, 215]) against the 0.2200 mm that was used — +3.8 %. Re-running the
matcher on the stored peak list with 0.2283 mm gives **efficiency 0.8688 and
16.331 fake/event** instead of 0.8648 and 16.597. So when these evals are
re-run on the fixed code, expect the headline to move by about
**+0.4 efficiency points and −0.27 fake/event**, with the model unchanged.

### 8.5 `eval_results.pkl` top-level scalars are all-events

§3.3. `overall_efficiency`, `fp_rate_per_evt`, `total_*` and `n_events` cover
every event read (25,000), not the μ-window subset (1,920) the printed summary
describes. Select on the per-event `mu` in `per_event`. Pkl files written
before `2e55f79` also lack `sigma_vtx_vtx_err_mm`, `in_mu_window`,
`sigma_fit_selection` and the all-events σ fields.

### 8.6 The drop-empty convention

`chain_scan.py:91-94` classifies every operating point twice:

- `all_peaks` — every PV node is a reconstructed vertex. A vertex to which the
  associator assigned **no track** is booked as **Fake** (`classification.py:311-312`).
- `drop_empty` — trackless vertices are removed from the reco list first.

`clean` and `n_truth` are identical between the two by construction (an empty
vertex is never Clean), so **`clean/truth` is the same in both**. Everything
with `n_reco` in its denominator differs, sometimes enormously. At t = 0.98 on
the v4 chain: fake 18,757 / n_reco 193,945 = **9.67 %** under `all_peaks`
against 303 / 175,491 = **0.173 %** under `drop_empty` — a factor 56 from a
bookkeeping choice. At t = 0.999 it is 66.4 % against 0.098 %.

`drop_empty` is the deployment-faithful convention (no vertexing chain emits a
trackless vertex) and is what the v4 tables quote. **Any fake rate from the
chain evaluation is meaningless without its convention.** The number of dropped
vertices is reported as `n_empty_vertices`.

### 8.7 The track taxonomy's numerator and denominator use different truth lists

`classify_assignments`'s truth **adjacency** is built over *every* truth vertex
(`build_truth_adjacency` loops over all of `pv_loc_z_event`,
`classification.py:238`; `truth_arrays` uses
`np.repeat(np.arange(len(pv_ntracks)), pv_ntracks)`,
`pu200_chain_graphs.py:109`), while `truth_pvs_count` — the denominator —
counts only nTrk ≥ 2 (`:114`).

So a reco vertex whose plurality truth vertex is an **nTrk = 1** truth vertex
can be classified Clean and enter the `clean/truth` numerator, while that truth
vertex is absent from the denominator. It needs a 1-track reco vertex (1/1 =
1.0 ≥ 0.70), so it is rare — but it is an unbounded-above ratio by
construction. The same adjacency also lets an nTrk = 1 vertex act as the key in
the split-demotion grouping.

### 8.8 Split/merged demotion rules and order dependence

**Positional.** Truth-side labels are order-independent. Reco-side `clean` vs
`merged` is **not**: `compare_res_reco` walks reco vertices in index order and
lets the first one to reach a shared unclaimed truth vertex absorb it
(`:281-296`). Since peaks arrive sorted ascending in z, the lower-z vertex is
systematically the one charged with the merge. Measured: 23 of 193,945 vertices
(0.012 %) move between clean and merged under z → −z; efficiency, split and
fake are invariant. Distance ties in the greedy pass (`:263`) are broken by
insertion order, i.e. also by z.

**Track-purity.** The demotion is a strict Σ pT² ranking within each plurality
truth vertex (`classification.py:339-349`), so it is order-independent up to
exact Σ pT² ties. It overrides **Clean as well as Merged** — a perfectly pure
vertex becomes Split if a larger sibling claims the same truth vertex — but
never overrides Fake (`:348`). Note the asymmetry: a Split is charged to
`n_reco` but its `clean` credit is gone, so splitting a vertex costs twice
in `clean_rate` and once in `clean/truth`.

### 8.9 `r16638` is not an independent sample

`r16443` and `r16638` share the generation and simulation tags
`e8481_s4494` and differ only in the reconstruction tag. Agreement between
them (efficiency 0.8648 vs 0.8662, σ 0.2200 vs 0.2213) is a **reconstruction
consistency check**, not a second measurement. Do not average them and do not
present the pair as reproducibility across samples.

### 8.10 The eval's own peak list is produced twice

`run_eval_pvf_run3.py:344-359` runs the peak finder twice per event: once with
`--integral-threshold` for the performance counts and once with
`--integral-threshold-res` for the σ fit. They default to 0.30 and 0.50 in the
current HL-LHC operating point, so **σ_vtx-vtx is measured on a different (and
sparser) peak list than the one efficiency is measured on.** This is deliberate
history (see `docs/evaluation/vertex_finding.md`, "Unified-threshold"), but it
means σ is not the resolution of the peak list being scored.

### 8.11 Errors

Bootstrap over **events**, never √N over vertices. Vertices within an event are
correlated through μ, through the shared histogram and through the matcher's
greedy assignment. The measured event-bootstrap SE on the r16443 μ-window slice
(2000 replicas) is ±0.0007 on efficiency and ±0.097 on fake/evt; a naive
√(N_truth) would give ±0.0007 on efficiency by coincidence and badly
mis-state anything reco-side. For A/B comparisons on the same events use a
**paired** bootstrap — the differences in §5.1 are far more significant than
the standalone errors suggest.

---

## 9. How to read our numbers

### 9.1 What is comparable

| quantity | across peak-finder settings, same data | across samples | across truth definitions | across campaigns |
|---|---|---|---|---|
| positional efficiency | ⚠ only if σ is held fixed | ✗ (μ-dependent) | ✗ | ✗ |
| positional fake/evt | ⚠ only if σ is held fixed | ✗ | ✗ | ✗ |
| reco/evt, truth/evt | ✓ | ✗ | ✗ (truth/evt is *defined* by the cut) | ✓ |
| σ_vtx-vtx | ✓ if the binning is held fixed | ⚠ | ✓ | ⚠ |
| track-purity clean/truth | ✓ | ✗ | ✗ | ✗ |
| **fraction-of-oracle** | ✓ | ⚠ | **✓** | **✓** |
| trkEff / trkPur | only at the same threshold | ✗ | ✓ | ⚠ |
| HS-ID | ✓ | ⚠ | ✓ | ✓ |

### 9.2 The rules, and where they come from

**Rule 1 — a ratio whose denominator is a convention is not portable across
that convention.** `clean/truth` divides by the nTrk ≥ 2 truth count. Change
the truth definition and the number moves for definitional reasons. AMVF's
track-purity clean/truth went 0.573 → 0.3094 between the v3 and v4
re-productions; truth density only rose 12.6 % (98.8 → 111.32 PVs/event), so
the denominator explains almost none of a 46 % relative drop. The purity
migration explains the rest.

**Rule 2 — normalise to a bound measured on the same data.**
*fraction-of-oracle* = (chain clean/truth) / (oracle clean/truth) divides the
definitional change out, because the oracle is recomputed on the same peaks and
the same truth. v3 at 95.7 % of its oracle and v4 at 94.0 % of its oracle is a
comparison a reader can trust; 0.716 against 0.6843 is not. **Quote
fraction-of-oracle as the headline and absolute clean/truth as secondary.**

**Rule 3 — a rate whose category definition is threshold-shaped reverses when
the population changes.** "PV-Finder makes 18× fewer fakes than AMVF" was true
on v3 data and is **false** on extended-\|η\| data — not because anything got
worse, but because a track-convention Fake needs a vertex whose tracks have no
truth association *in the plurality*, and at extended \|η\| an AMVF vertex
essentially always picks up truth-associated tracks. AMVF's track-convention
fake rate collapsed 0.91 % → 0.072 % for definitional reasons, and the chain
now makes *more* of them (0.173 %). **Do not repeat "18× fewer fakes" for any
extended-\|η\| result.**

**Rule 4 — never compare a number across a change in σ without saying so.**
Efficiency and fake rate are functions of the window, which is a function of σ
(§5.1). Two evals with different σ are two different measurements even on
identical peak lists. Always quote σ next to efficiency, in the same table.

**Rule 5 — a threshold-scan number is a curve, not a value.** trkEff, trkPur,
fake_rate and clean/truth all move strongly with the GNN threshold, in
different directions. Quote the threshold, or quote trkF1 and clean/truth
together with the operating point.

**Rule 6 — the same word, different taxonomies.** Anything containing "clean",
"merged", "split" or "fake" must carry its taxonomy. Anything containing
"efficiency" must carry its side (truth or reco) and its μ window.

### 9.3 What to quote for the current HL-LHC results

- **Finder:** efficiency, fake/evt, reco/evt and σ, all four together, with the
  μ window and the operating point. Plus the nTrk = 1 decomposition of the
  fake rate (§8.2).
- **Chain:** clean/truth **and** fraction-of-oracle, the convention
  (`drop_empty`), the threshold, and the AMVF row on the identical events.
- **Never:** a fake rate without its taxonomy and convention; an efficiency
  without its μ window; `clean/truth` compared across truth definitions.

---

## 10. Bugs and code/documentation discrepancies found

### 10.1 BUG (documentation): "Fake" is a plurality rule, not an all-or-nothing rule

Three places state that a track-purity Fake means *no* track has a truth
association:

- `src/gnn/evaluation/classification.py:147` — *"**Fake** if no truth PV matches
  any of its tracks."*
- `src/gnn/diagnostics/amvf_convention_check.py:12-13` — *"Fake only when none
  of its tracks has any truth vertex at all."*
- `docs/evaluation/vertex_association.md`, classification table — *"No matched
  tracks have truth associations"*.

**The code does not do this.** `classification.py:314` takes the plurality over
a dict in which truthless tracks occupy a bucket keyed `"Fake"`, and `:319`
classifies Fake when that bucket wins. A vertex with 3 truthless tracks and 1
truth-associated track is **Fake**. P_d in the worked example is exactly this
case. The distinction matters for any argument about what the fake rate
measures. The docstrings should be corrected; the behaviour is defensible and
should not change without a decision.

### 10.2 BUG (latent crash): a track shared by two truth vertices raises

`classification.py:287-293`:

```python
truth_index = np.nonzero(truth_track_indices == j)[0]
truth_pv_num = truth_pv_indices[truth_index]
...
pv_dict_name = f"Truth_PV_{int(truth_pv_num)}"
```

`int()` on a size-2 array raises `TypeError: only size-1 arrays can be
converted to Python scalars` (numpy 1.24.4, the pinned version). So if a track
index ever appears in two `TruthVertex_assocTracks` lists — or twice in one —
the whole evaluation dies mid-run rather than degrading.

Checked on 4,000 r16443 events: **zero** such tracks, zero negative indices,
zero out-of-range indices. So this is latent, not active. It is an unchecked
assumption at a data boundary; the project rule is to validate at boundaries.
Not fixed here because `classification.py` is shared by every TTVA eval and
touching it invalidates the regression guard.

### 10.3 Discrepancy: `amvf_convention_check` computes a different purity than it explains

`vertex_purities` (`amvf_convention_check.py:52-73`) takes the maximum over
**real truth vertices only** (`real = truths[truths >= 0]`), while
`classify_assignments` takes the maximum over a dict that **includes** the
truthless bucket. For a vertex where truthless tracks are the plurality, the
diagnostic reports a nonzero real-truth purity while the classifier calls it
Fake. Bounded by the AMVF fake count, 135 of 187,960 (0.07 %), so it does not
affect any published conclusion — but the two functions are not measuring the
same thing, and the script's stated purpose is to explain the classifier.

### 10.4 Discrepancy: the purity-cut scan is not `clean_rate`

`clean_rate_vs_purity_cut["0.7"] = 0.4072` and `clean_rate = 0.3518` in
`amvf_convention.json`, and nothing in the file or in
`docs/evaluation/vertex_association.md` says why. The scan counts vertices
above the cut **before** the split demotion; `clean_rate` counts them after.
The difference is 10,414 vertices (§7.2), and the arithmetic closes exactly.
A reader comparing 0.4072 to 0.3518 will otherwise conclude one of them is
wrong.

### 10.5 Asymmetry, not a bug: reco-side clean/merged is z-biased

§8.8. Deterministic, small (0.012 % of vertices), and does not touch
efficiency. Recorded so it is not rediscovered as a bug.

### 10.6 Stale: `run_eval_pvf.py` still hardcodes the μ window and 60 pairwise bins

`run_eval_pvf.py` fixes `MU_MIN, MU_MAX = 55, 65` in `main()` with no CLI
override, and its pairwise fit is hardcoded to `np.linspace(-6.0, 6.0, 61)`,
i.e. 60 bins of 0.2 mm — commensurate with the 0.04 mm grid, so no comb, but 60
bins is the binning documented as biasing σ high
(`docs/evaluation/vertex_finding.md`). Already flagged in `JOURNAL.md`,
deliberately not fixed. It also has not received the `2e55f79` σ-population
fix, so on any flat-μ input it has the §8.4 problem.

---

## 11. Provenance

Every number on this page, and how to regenerate it.

| numbers | source |
|---|---|
| positional totals, efficiency, fake/evt, AMVF block (r16443, r16638) | `outputs/08_05_2026_output/eval_v6_operating_point/{r16443,r16638}/eval.log` + `eval_results.pkl`; exact invocation and git SHA in `INVOCATION.txt` |
| window-sensitivity table, mirror test, bootstrap errors, float32/float64 knife edge | `outputs/08_05_2026_output/metric_taxonomy/window_probe.json`, recomputed from the stored peak list with `compare_res_reco` |
| truth-vertex nTrk census (23.13 nTrk = 1 per event) | read directly from `ATLAS_PVFinderData_601229_e8481_s4494_r16443_PU200.root`, `TruthVertex_nTracks` over the first 25,000 entries, μ-window selection |
| AMVF track-purity totals, purity distribution, cut scan | `outputs/08_05_2026_output/gnn_ttva_v4/amvf_convention/amvf_convention.json` (`gnn.diagnostics.amvf_convention_check`) |
| chain threshold scan, drop-empty, trkEff/trkPur/trkF1 | `outputs/08_05_2026_output/gnn_ttva_v4/chain_scan_v4noaug/chain_scan.json` |
| finder cap, oracle | `outputs/08_05_2026_output/gnn_ttva_v4/ttva_gap_v6_test/gap_decomposition.json` |
| truth-graph ceiling | `outputs/08_05_2026_output/gnn_ttva_v4/truth_eval_v4noaug_t095/summary.json` |
| HS-ID | `outputs/08_05_2026_output/gnn_ttva_v4/hs_id_v4noaug/hs_id.json` |
| μ-window σ at 300 bins (0.2283 mm) | `outputs/08_05_2026_output/ripple_study/binning.json` |
| nTrk = 1 fake decomposition | `outputs/08_05_2026_output/ripple_study/fake_decomposition.json` |
| worked example | `src/pv_finder/diagnostics/metric_worked_example.py`, pinned by `tests/test_metric_worked_example.py` |

Cross-checks performed while writing this page:

- The published r16443 summary block is reproduced **bit-for-bit** from the
  stored peak and truth lists (§5.2).
- The finder eval and the chain builder agree on the peak count on the same
  1920 events through independent code paths: 193,945 peaks (101.013/evt) in
  both.
- The truth count agrees exactly across the two taxonomies: 213,738.
- The split-demotion decomposition of the AMVF purity scan closes with zero
  residual (§7.2).

---

## See also

- [Evaluation — vertex finding](vertex_finding.md) — the finder eval, its
  operating point and its outputs.
- [Evaluation — vertex association](vertex_association.md) — the TTVA and chain
  evals, working points and results.
- [`docs/research/resolution_plot_ripple.md`](../research/resolution_plot_ripple.md)
  — pairwise-Δz binning and why σ moved.
- [`docs/research/pairwise_dz_bump.md`](../research/pairwise_dz_bump.md) — the
  satellite-peak mechanism behind part of the fake rate.
