# doc-batch-gradients

**How much gradient noise do whole-document long-context batches add?** This is a preregistered measurement on
open LM checkpoints.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23268861.svg)](https://doi.org/10.5281/zenodo.23268861)

Long-context training usually fills each sequence with a single long document, such as a book, a code
repository or a paper. Chunks of one document are not independent samples. If their gradients are correlated,
a batch of document-contiguous sequences carries fewer independent samples than its token count suggests: it
has a *design effect* in the survey-sampling sense.

This repo measures that effect directly, on real gradients:
- **Models and checkpoints:** Pythia-410M (4 checkpoints), Pythia-1.4B (2) and OLMo-2-0425-1B (2).
- **Corpora:** 16 documents of 16K tokens each from four corpora:
  - PG-19 books;
  - code repositories concatenated by repo;
  - arXiv papers;
  - packed FineWeb as a control.

## Result

R(16K) is the gradient-noise variance of one document-contiguous 16K-token block, divided by that of 32
independent 512-token chunks with the same token count. It is the Kish design effect, and equals the ratio of
McCandlish's B_simple for contiguous versus shuffled batches. R_rel divides by the packed-web control at the
same model and checkpoint. Brackets are 95% document-bootstrap intervals.

| Corpus (final checkpoints) | Pythia-1.4B R_rel | OLMo-2-1B R_rel | Raw R(16K), 1.4B / OLMo |
|---|---|---|---|
| Code repositories | 4.91 [2.61, 7.78] | 3.51 [2.37, 5.23] | 6.91 / 4.73 |
| arXiv | 2.36 [1.91, 2.96] | 2.23 [1.78, 2.92] | 3.32 / 3.01 |
| PG-19 books | 1.56 [1.26, 1.98] | 1.30 [1.07, 1.67] | 2.20 / 1.75 |
| FineWeb, packed (control) | 1 | 1 | 1.41 / 1.35 |

**What this means in practice.** At equal tokens, a batch of 16K-token code-repository blocks (32 consecutive
512-token chunks per document) has roughly 3.5–5× fewer effective independent samples than packed web text. For arXiv the factor is
about 2×; books are close to the control. This compares effective sample counts. Each R is measured against
shuffled chunks of its own corpus, so it does not compare the absolute gradient noise of code and web text; that
would also need the ratio of their per-chunk noise variances. Each chunk was run as a standalone 512-token
sequence, so these numbers are a proxy for 16K-context training: real long-context gradients, where each chunk
attends to the ones before it, were not measured.

- **The effect persists through training.** Pythia-410M shows the same ordering at 0.7%, 5%, 25% and 100% of
  training.
- **The code result is skewed.** A few of the 16 repositories dominate it; the per-document numbers are in
  `report.json`, under `secondary`.

## What this does *not* show

- **No causal claim.** It does not show that this design effect explains why long-sequence training hurts
  short-context quality. Testing that needs training runs that hold documents per step fixed while varying the
  window, and those were not run.
- **The preregistered verdict is INCONCLUSIVE.** The packed-web control was expected to have R(16K) < 1.2, but
  at the final checkpoints it measured 1.26–1.41. Short web documents straddle chunk boundaries, so the control has real short-lag
  correlation. The control-relative criterion was added as [docs/ADDENDUM_001.md](docs/ADDENDUM_001.md) after all
  Pythia-410M results and part of the 1B-class data had been seen, including half of the Pythia-1.4B final WEB
  cell, so it is not fully blind. Under it, none of the stopping criteria fire (the result **survives**). Some early 1B-class checkpoints are
  incomplete.
- **Known limitations:**
  - Cross-document pairs are formed only within 4-document shards, so the estimate is noisier than planned (the
    restriction itself does not bias it).
  - G3 fails for Pythia-1.4B PG-19: the S_cross interval crosses 0. That affects only that cell's B_simple.

## Related work

The idea that tokens in one sequence give correlated gradients is not new; as far as I can tell, measuring it
as a function of distance, with the design effect it implies, is.

- **Everett & Qiu (arXiv 2609.04577, App. I).** They note that tokens within a sequence "share context and are
  likely more correlated". They propose comparing (batch, length) splits at matched tokens as future work.
- **McCandlish et al. (1812.06162).** They define the gradient noise scale B_simple used here.
- **Thomas (arXiv 2607.05872).** A shuffled-document control on real LM gradients, used to show that
  within-document correlation explains only a small part of a gradient-spectrum effect. No distance profile or
  design effect.
- **Critical-batch-size work (e.g. 2410.21676).** It reports little sensitivity to context length, but on packed
  data.
- **Dataset Decomposition (2405.13226) and ProLong (2410.02660).** They study sequence-length curricula and
  long-document data mixes; ProLong's long data is drawn from code repositories and books. They do not measure
  gradient correlation.

I found no prior measurement of within-document gradient correlation against token distance, or of the design
effect it implies, on real LM gradients. Corrections are welcome.

## Method

- **Per-chunk gradients.** Each 512-token chunk is a standalone sequence, and the loss is mean next-token
  cross-entropy. The full parameter gradient is count-sketched into 2^17 buckets per parameter group (attention,
  MLP, embedding, other). Exact squared norms replace the Gram diagonal.
- **Sketch fidelity.** Checked against exact inner products: 8,176 pairs, mean cosine error −0.00007, sd 0.002.
- **Statistics.** These come from Gram matrices of chunk gradients:
  - S_cross is the mean over cross-document pairs and estimates the squared norm of the mean gradient;
  - σ² is the per-chunk noise variance;
  - ρ(Δ) is the same-document correlation at lag Δ;
  - R(W) is the block-mean noise variance relative to σ²/m.
- **Everything ran on CPU.** That is 4-vCPU cloud machines with no GPU.

The full plan is in [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md). Changes made before the verdict data are
in [docs/DEVIATION_001.md](docs/DEVIATION_001.md) and [docs/DEVIATION_002.md](docs/DEVIATION_002.md).

## Reproduce

```bash
pip install -r requirements.txt
python code/analyze.py --out report.json          # re-derive every number above from results/ (~20 min, CPU)
```

- `results/`: per-shard Gram matrices, exact squared norms and per-chunk losses (`.npz`), plus run metadata
  (`.json`).
- `report.json`: the full analysis output, including gates, criteria, per-document diagnostics and bootstrap
  intervals.
- **Checkpoint safety:** `probe.py` loads only the commits pinned in `data/checkpoint_revisions.json` unless you
  pass `--allow-unpinned`. Some Pythia revisions ship `pytorch_model.bin`; the pinned torch loads those with
  `weights_only=True`.
- **Re-running the probe:** first rebuild the input spans with `python code/data_prep.py`. It downloads public
  datasets and tokenizers; the token spans are not redistributed here. Then run, for example:
  `python code/probe.py --model EleutherAI/pythia-410m --revision step143000 --tok pythia --corpus REPOS --out results/...`.
- **Input spans:** `data/manifest.json` records document indices and span hashes, and `code/data_prep.py` and `data/checkpoint_revisions.json` pin every Hub dataset, tokenizer and checkpoint revision, so
  rebuilt spans can be checked against the ones used here.

## Citation

The illustrated report and this repository (without the Gram matrices, which it pins by SHA-256) are archived
on Zenodo: [doi:10.5281/zenodo.23268861](https://doi.org/10.5281/zenodo.23268861), which resolves to the latest version. An interactive version of the report is at
[severinvisionary.github.io/doc-batch-gradients](https://severinvisionary.github.io/doc-batch-gradients/).

```bibtex
@techreport{yang2026longdocbatch,
  author = {Yang, Hanyu},
  title  = {Long documents make your batch smaller than it looks: measuring within-document gradient correlation},
  year   = {2026},
  institution = {Zenodo},
  doi    = {10.5281/zenodo.23268861},
  url    = {https://doi.org/10.5281/zenodo.23268861}
}
```

## License

- **Code:** MIT ([LICENSE](LICENSE)).
- **Results and metadata** (`results/`, `report.json`, `data/*.json`): CC BY 4.0 ([results/LICENSE](results/LICENSE)).
- **Not included:** the source corpora and the tokenizers fetched by `code/data_prep.py` keep their own licenses.
  This includes the Llama 3 tokenizer, which is used only to decode ProLong token ids and is covered by the Llama 3
  Community License.

Contact: Hanyu Yang (hanyu.yang.92@gmail.com).
