# DEVIATION_002: faster hash source for the count sketch; grid sizing (2026-10-07)

Recorded before any verdict-bearing shard was run.

**Observation (F0b feasibility run).** The DEVIATION_001 count sketch passed G1 with a wide
margin: mean(cos_sketch − cos_exact) = −0.00004, sd = 0.0024, mean norm ratio = 0.9999 (63 pairs). G4 was
bit-identical. But it was slow. Drawing hashes with `torch.randint` from a CPU generator took most of the time:
14.2 s per chunk on Pythia-410M (4.2 s forward/backward) and 60 s on OLMo-2-1B, about 160 CPU-hours for
the grid.

**Change.** `Sketcher` draws (bucket, sign) from numpy's PCG64 generator, seeded by the same per-parameter
seeds, and accumulates with `np.bincount`. It is still a count sketch with 2^17 buckets per group and
independent uniform hashes and signs; only the pseudo-random stream differs. It is about 5× faster on the
hashing step (local benchmark: 16M draws + accumulate ≈ 0.09 s). A numpy-only unit check on random vectors
recovered an exact cosine of 0.0991 as 0.0992 ± 0.0029 over 20 seeds; the theory predicts sd 0.0028.

**Effect on gates.** The F0b G1 numbers were measured with the old stream, so they do not certify this one. G1 is
therefore read from the grid itself. All 16 Pythia-410M shards carry the exact reference (511 pairs each), and
the thresholds are unchanged. A G1 failure there makes the verdict INCONCLUSIVE as preregistered.

**Grid sizing (PREREG §9).** Seconds per chunk assumed for sizing: 410M 8.5 (with exact reference),
1.4B 24, OLMo-2-1B 27 (F0 forward/backward plus the new hashing cost). A cell above ~2.1 h is split into two
8-document shards. Jobs stay under ~2.3 h. Nothing else changes.

**Memory.** OLMo-2-1B peaked at 13.9 GB with the old sketch, against a ~14 GB cgroup limit. The numpy path
allocates less per slice (uint32 hashes rather than int64 plus a float64 sign vector). An out-of-memory kill is
handled as a blocker for that shard, and the report says so if it happens.

**Addendum (same day, before any 1B-class shard ran).** Sketch rows are written to disk-backed memmaps and the
Gram is accumulated in float64 over column blocks, so peak RAM is model plus gradient only. This is the same
quantity up to float64 summation order. The Pythia-410M shards were launched before this change and
keep their rows in RAM; nothing else differs.

**Resharding (same day).** The F0c smoke run showed OLMo-2-1B at 55 s per chunk on the cloud
box, not the 27 s assumed, because hashing is slower there than in the local benchmark. Peak memory stayed below
the limit. The 1B-class cells are therefore split into four 4-document shards each (~2 h per shard). The documents, cells
and everything measured are unchanged.
