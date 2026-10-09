# ADDENDUM_001: control-relative criterion (b′) and gate G2′ (2026-10-07)

Approved by the owner on 2026-10-07, together with completing the 1B-class final checkpoints.
This file was committed and pushed **before** any shard of the confirmatory cells below landed. It does **not**
change the Phase 0 verdict in README.md, which stays INCONCLUSIVE under PREREGISTRATION.md.

## What was seen before writing this

All of Pythia-410M, and the partial 1B-class cells listed in README.md. That includes half of the Pythia-1.4B
final WEB cell, at R(16K) = 1.59 on 8 docs. The 410M numbers motivated the change. So 410M is **exploratory**
under this addendum and is not used to decide it. On 410M the restated rule would read R_rel = 3.9 for REPOS,
2.0 for ARXIV and 1.25 for PG19.

## Confirmatory cells

There are 8 confirmatory cells, all at c = 512 with all 16 docs each:
- Corpora: PG19, REPOS, ARXIV and WEB.
- Checkpoints: Pythia-1.4B step143000 and OLMo-2-0425-1B stage1-step1907359-tokens4001B.

Everything else is the same as before. The spans, sketch, probe, 1B-class final-checkpoint shards, the
bootstrap, and criteria (a) and (c) are unchanged.

## Changes

- **R_rel(16K)** = R_corpus(16K) / R_WEB(16K), taken at the same model and checkpoint. Its 95% CI is the
  percentile of the ratio of paired bootstrap draws. Corpora are resampled independently.
- **(b′)** replaces (b). It fires if and only if R_rel(16K) < 1.5 for every long corpus × confirmatory checkpoint.
  "Robust" means every CI upper bound is also < 1.5.
- **G2′** replaces G2. WEB ρ_long must be < 0.01 at both confirmatory checkpoints. The bar of R_WEB(16K) < 1.2 is
  dropped.
  - Rationale: in packed web data, documents of a few thousand tokens straddle chunk boundaries. That gives real
    short-lag within-span correlation (410M: ρ(512) = 0.042, decaying to about 0.01 by 2K), so R_WEB > 1 is
    expected and is not an artifact. The long-lag part of G2 still screens for position, format and centering
    artifacts.
  - **What this makes easier:** G2′ is strictly easier to pass than G2. In exchange, (b′) is strictly harder to
    survive than (b) whenever R_WEB > 1, because dividing by the control removes the part a packed-web baseline
    shares.
- **Narrowing (secondary, not verdict-bearing):** a long corpus "carries" the effect if its R_rel CI lower bound
  is ≥ 1.5 at both confirmatory checkpoints. Expected from 410M: REPOS and ARXIV carry; PG19 does not.

## Verdict rule

1. If any confirmatory cell is missing or has fewer than 16 docs: INCOMPLETE.
2. Else, if G1 or G2′ fails: INCONCLUSIVE.
3. Else, if (a), (b′) or (c) fires: KILL, or "KILL (not robust)" by the same robustness rule as before.
4. Else: SURVIVES, which leads to a funding decision on Phase 1.

Implementation: `addendum_001()` in `code/analyze.py`. It was smoke-tested on the pre-addendum results, where
it returns INCOMPLETE.

## Compute

There are 30 four-document shards, about 2 h each on a 4-vCPU CPU machine (OLMo-2 at about 55 s per
512-token chunk × 128 chunks).
