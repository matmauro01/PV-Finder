# Additional studies: open items

Running list from the November 11, 2026 walkthrough with Mattia. Not on the slides.

## 1. Matching window
- Rerun the two §7 association-figure bars (GNN finder cap, oracle) at 0.1694 mm.
- Settle the oracle window: the note says 0.5 mm, the code defaults to 1.0 mm.
- Apply the note edits from matching_window_audit_2026-11-09.md (via lxplus, after review).

## 2. Split vertices
- Confirm the reviewer meant splits as "found"; if they meant a penalty, quote fake+split per event (19.3 vs 19.9).

## 3. HS studies vs pile-up
- Decide which GNN the vs-mu plots show (PU200-only v4 e107 vs full-mu v1 e213) and state the checkpoint in the note.

## 4. AMVF HS selection
- Fairness check: select the GNN primary with the same pT-capped sum(pT^2) rule (cap 15 GeV) and compare to AMVF re-ranked with the cap (97.8%). Replay only, no inference.

## Definitions
- If asked: recompute AMVF Clean with fit weights from a fresh Athena ntuple (ours are unweighted track counts).
