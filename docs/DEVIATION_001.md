# DEVIATION_001: sketch replaced after G1 failed at feasibility (2026-10-07)

Recorded before any verdict-bearing shard was run. The only data seen are those of the feasibility shard named
below, which is excluded from the verdict by construction (feasibility outputs, not published; PREREGISTRATION.md §9).

## What was observed

F0 feasibility shard `EleutherAI__pythia-410m/step143000/PG19_c512_d0-2` (64 chunks, two PG-19 books, exact
inner products against chunk 0 for 63 pairs):

| G1 component (PREREG §7) | Threshold | F0 | |
|---|---|---|---|
| mean(cos_sketch − cos_exact) | ≤ 0.003 in abs. | −0.00005 | pass |
| sd(cos_sketch − cos_exact) | ≤ 0.03 | 0.0294 | pass, at the limit |
| mean sketched / exact squared norm | 1 ± 0.03 | 1.051 | **fail** |

The preregistered sketch (R1ᵀ G R2 / 32 per weight matrix) is unbiased only on average over sketch draws. Its
error depends on the rank of G, and per-chunk gradients are dominated by low-rank matrices. With one fixed
sketch shared by all chunks the error is systematic: every chunk's norm came out ~5% high. A shared error of
that kind also biases the same-document numerator of ρ relative to the cross-document baseline. So the
instrument failed its own gate, and per the PREREG it may not produce a verdict.

The exact inner products themselves (not a verdict quantity) were in [−0.08, 0.13] in cosine. That puts the
per-pair sketch noise (0.03) at the same order as the signal.

## What changes

1. **Sketch:** a count sketch with 2^17 buckets per parameter group (attn, mlp, embed, other), with hashes and
   signs drawn from a generator seeded per parameter name (`code/probe.py::Sketcher`). Its inner-product error
   is about √(2/2^17) ≈ 0.004 of ‖a‖‖b‖ per group regardless of matrix rank; the four groups together give
   ~2^19 buckets.
2. **Diagonal:** `analyze.py` replaces the Gram diagonal by the exact squared norms the probe already records.
   σ² and the block terms of R(W) then carry no sketch error on the diagonal.
3. **G1 is re-checked** on the new sketch with the same thresholds, in a second feasibility shard (F0b), before
   the grid runs. The norm-ratio component still compares sketched with exact norms, so it still tests the sketch.

Nothing else changes: corpora, spans, chunking, models, checkpoints, statistics, kill criteria and other gates
are as preregistered.

## Disclosure: what else was seen before this record

The F0 job also ran `analyze.py --no-boot` over its feasibility outputs (as its instructions asked). On the old sketch,
pooling the two overlapping PG-19 shards (doc 0 appears twice, so cross pairs include a document with itself), it
printed ρ_long = +0.0141 and R(16K) = 1.66 for Pythia-410M final. These numbers are not a verdict quantity: wrong
instrument, two documents, a duplicated document. They sit near the kill thresholds (0.01, 1.5) and so could in
principle tempt a choice of instrument. The change above was committed before the F0 report containing these
lines was read, and it was motivated by the G1 failure alone. One thing was seen earlier: the analyst's own
G1 check on the F0 shard also printed raw, uncentred sketch cosines for book 0 by lag (0.014–0.046 at lags 1–31)
and the mean raw cosine between the two books (0.0016). Those are not ρ (no centring or σ² normalisation), but
they are disclosed here. The F0b job instructions forbid computing any statistic other than G1 and G4.

## Also from F0 (no change needed)

- The cloud job fixed an out-of-memory error in the exact-reference path (per-parameter float64 dot against
  a float32 reference copy; same quantity). Recorded in the feasibility fix log.
- Pythia-410M takes 6.2 s per 512-token chunk on the 4-vCPU cloud machine, including the exact reference.
