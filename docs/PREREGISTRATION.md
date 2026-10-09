# DOC-BATCH Phase 0 preregistration: within-document gradient correlation on long documents

Frozen 2026-10-07, before any gradient was computed. (Public copy: internal tracker links and budget
figures are redacted; technical run details are kept; the analysis plan is unchanged.)
Scope: **Phase 0 only** (inference; no training run).

## 1. Question and what each answer decides

When a training sequence is one long document, tokens per step fixed means documents per step fall as 1/W. If
gradients of chunks of one document are correlated, a W-token document is worth fewer than W/c independent
chunks: deff(W) = 1 + (2/m) Σ_{k<l} ρ(|k−l|c), m = W/c. Phase 0 measures ρ(Δ) and deff directly on public
checkpoints. It cannot show that composition *explains* any training penalty (IAP 2608.12218 already shows a
window penalty at matched composition); it can only show whether the composition channel is big enough to be
worth a training study.

| Outcome | Decision |
|---|---|
| Any kill criterion (§6) fires | Close Phase 0 negative; recommend retiring the topic (or reshaping per §6 note) |
| None fires, gates pass | Report numbers; a separate decision on whether to fund Phase 1 (no Phase 1 work starts) |
| An instrument gate fails | INCONCLUSIVE; report which gate and why; no verdict |

## 2. Models and checkpoints (fraction of training)

| Model | Checkpoints |
|---|---|
| `EleutherAI/pythia-410m` | step1000 (0.7%), step7000 (4.9%), step36000 (25.2%), step143000 (100%) |
| `EleutherAI/pythia-1.4b` | step7000 (4.9%), step143000 (100%) |
| `allenai/OLMo-2-0425-1B` | stage1-step90000-tokens189B (4.7% of stage 1), stage1-step1907359-tokens4001B (end of stage 1) |

Pythia was trained on the Pile, which contains PG-19, arXiv and GitHub: some probe documents may be training
data. This is recorded as a limitation, not corrected.

## 3. Corpora (frozen inputs, `data/`)

`code/data_prep.py` (seed 20261007) selected 16 documents per corpus; each span is tokens [2048, 18432) of the
document under each tokenizer (Pythia GPT-NeoX; OLMo-2). `data/manifest.json` records every source pointer,
token count and span SHA-256; `data/spans_{pythia,olmo2}.npz` hold the token ids.

| Corpus | Source | Role |
|---|---|---|
| PG19 | `emozilla/pg19` @ c021754c8e, train shard 0, seeded row permutation | long documents (books) |
| REPOS | `princeton-nlp/prolong-data-64K` `thestackv1_concat_by_repo-65536` (one repository per sequence) | long documents (code), ProLong's own long data |
| ARXIV | same dataset, `arxiv`, single documents cut out with `indices` | long documents (papers, LaTeX) |
| WEB | same dataset, `fineweb-2023-50`: consecutive packed web documents joined with EOS | **control**: packed short documents; deff ≈ 1 expected |

ProLong sequences are Llama-3 ids, decoded with `NousResearch/Meta-Llama-3-8B` and re-tokenized. The ARXIV
fallback (`ccdv/arxiv-summarization`) was coded but not needed (16 of 32 scanned qualified).

## 4. Instrument

- Chunk c = 512 tokens (primary); 32 chunks per document span. Each chunk runs as a standalone sequence;
  loss = mean next-token CE over its 511 targets; fp32, eval mode, CPU.
- Full-gradient sketch: every 2-D weight G → R1ᵀ G R2 / 32 with fixed Gaussian R1, R2 (32 columns each; seeded by
  parameter name). 1-D parameters kept exactly. Unbiased for inner products. Sketches grouped as attn / mlp /
  embed / other; only Gram matrices, exact squared norms and losses are stored (`results/`, ≈ 2 MB per shard).
- **Sketch fidelity** (Pythia-410M shards): the first chunk's exact gradient is kept and exact inner products
  with every later chunk are recorded.

## 5. Statistics (per model × checkpoint × corpus; `code/analyze.py`)

From the Gram matrix K of all 512 chunk gradients:

- S_cross = mean K over chunk pairs from **different** documents (estimates |G|², G = corpus mean gradient).
- σ² = mean K_ii − S_cross.
- ρ(Δ) = (mean over same-document pairs at lag Δ of K − S_cross) / σ², Δ = 512 … 15872.
- **ρ_long** = mean of ρ(Δ) over Δ ∈ [2048, 16384).
- **R(W)** = (mean over documents and aligned W-blocks of ‖block-mean gradient‖² − S_cross) / (σ²/m): the noise
  variance of one document-contiguous W-token unit over that of m independent chunks. R(W) is the B_simple ratio
  of contiguous to shuffled batches at equal tokens, and R(16K) = deff(16K), estimated without a parametric ρ.
- B_simple (shuffled, tokens) = c·σ²/S_cross; contiguous = R(W) × that.
- 95% percentile bootstrap over documents (2000 reps), with duplicated documents never counted as
  cross-document pairs. Validated on synthetic Grams with planted ρ = 0.05 (recovered 0.0507, CI covers) and
  ρ = 0 (CI covers 0).

Note: criteria (a) and (b) are related but not redundant. (a) concerns lags ≥ 2K only; (b) integrates all lags,
so short-range correlation alone can make R(16K) large.

## 6. Kill criteria (owner's (a)–(c), operationalized)

"Long corpora" = PG19, REPOS, ARXIV. All on c = 512.

- **(a)** fires iff ρ_long < 0.01 for every long corpus × model × checkpoint.
- **(b)** fires iff R(16K) < 1.5 for every long corpus × model at its final checkpoint.
- **(c)** fires iff R(16K) ≥ 1.5 for some long corpus at some checkpoint < 5% of training, and R(16K) < 1.5 for
  every long corpus × model at every checkpoint ≥ 25% of training.

Point estimates decide firing. A firing criterion is **robust** if every cell's 95% CI upper bound is also below
its threshold; otherwise the report says "KILL (not robust)". Verdict: KILL if any criterion fires; otherwise
SURVIVES → funding decision on Phase 1.

## 7. Instrument gates (checked before the verdict is read)

- **G1 sketch fidelity:** over all exact-checked pairs, |mean(cos_sketch − cos_exact)| ≤ 0.003,
  sd ≤ 0.03, and mean sketched/exact squared-norm ratio within 1 ± 0.03.
- **G2 control:** WEB at every model's final checkpoint has R(16K) < 1.2 and ρ_long < 0.01. A failure means the
  instrument sees "correlation" where documents are independent (position, formatting or centering artifact).
- **G3 well-posedness:** S_cross's bootstrap CI lower bound > 0 at final checkpoints. Failure affects only the
  B_simple numbers, not (a)–(c).
- **G4 determinism:** one document re-run in the feasibility job reproduces its Gram entries to 1e-5 relative.

G1 or G2 failure → INCONCLUSIVE (a failed gate kills the instrument, not the question).

## 8. Secondary (reported, not verdict-bearing)

ρ and R by parameter group; c = 2048 chunks on Pythia-410M final for the three long corpora; per-document spread
of ρ_long; per-chunk losses.

## 9. Execution

CPU only (4-vCPU machines; no GPU). A feasibility run first measures per-chunk time, G1 on one shard and G4;
the grid is then split into shards sized from it. Deviations are recorded before results are read
(`DEVIATION_001.md`, `DEVIATION_002.md`).

## 10. Known limitations, stated in advance

- Standalone 512-token chunks omit the left context a chunk has inside a long training window; the c = 2048
  secondary is the only check on this.
- Gradients at a checkpoint describe the noise at that point of a short-window pretraining run, not during
  long-context extension (ProLong's regime, LR 1e-5 on an 8B model).
- 16 documents per corpus; ρ varies across documents, and the CI is over documents.
- deff measures correlation of gradient *noise*. It says nothing about whether the correlated signal helps
  (SDO 2607.27273 and Fort 2105.13343 report coherent batches helping in other settings).
