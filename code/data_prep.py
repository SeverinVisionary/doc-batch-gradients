#!/usr/bin/env python3
"""Build the frozen DOC-BATCH Phase 0 inputs: 16 long documents per corpus, as token spans.

Network only (no model weights). Writes data/spans_<tok>.npz and data/manifest.json.

Corpora (docs/PREREGISTRATION.md §3):
  PG19   emozilla/pg19, train parquet shard 0, rows in a seeded permutation
  REPOS  princeton-nlp/prolong-data-64K thestackv1_concat_by_repo-65536 (one repo per 64K sequence)
  ARXIV  princeton-nlp/prolong-data-64K arxiv (single documents cut out via `indices`)
  WEB    princeton-nlp/prolong-data-64K fineweb-2023-50 (control: consecutive packed web documents,
         joined with EOS into one pseudo-document)
ProLong sources are Llama-3 token ids; they are decoded to text with an ungated Llama-3 tokenizer copy and
re-tokenized for each model family. A span is tokens [OFFSET, OFFSET + SPAN) of the model tokenizer's ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct

import numpy as np

SEED = 20261007
N_DOCS = 16
OFFSET = 2048
SPAN = 16384
NEED = OFFSET + SPAN
TOKENIZERS = {"pythia": "EleutherAI/pythia-410m", "olmo2": "allenai/OLMo-2-0425-1B"}
LLAMA3_TOK = "NousResearch/Meta-Llama-3-8B"
PROLONG = "princeton-nlp/prolong-data-64K"
# Hub revisions. The 2026-10-07 run used each repo's default branch; these are those branches' commits (none of
# the repos changed between 2024-04 and the run), pinned afterwards so a rebuild fetches identical files.
REVISIONS = {
    PROLONG: "1447128c751f0fbc7d2082b2a113a938426a9bba",
    "EleutherAI/pythia-410m": "9879c9b5f8bea9051dcb0e68dff21493d67e9d4f",
    "allenai/OLMo-2-0425-1B": "a1847dff35000b4271fa70afc5db10fd29fedbdf",
    LLAMA3_TOK: "315b20096dc791d381d514deb5f8bd9c8d6d3061",
    "ccdv/arxiv-summarization": "240aaf1a969b3f8cd0ade6986bfad0cd730ee288",  # fallback only; unused in this run
}
PG19 = ("emozilla/pg19", "c021754c8e")


def _hf(repo, filename, repo_type="dataset", revision=None):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename, repo_type=repo_type, revision=revision)


def _mds_ndarray(buf: bytes, itemsize: int = 4) -> np.ndarray:
    """Decode a streaming NDArray column whose dtype is fixed (uint32) but shape is not.

    Header = one code byte, then ndim dims of a common width (1/2/4/8 bytes). Observed in this dataset:
    input_ids = [6][uint32 65536], indices = [8][uint8 n][uint8 2]. We accept the unique (ndim, width) whose
    dims multiply to the payload size, and fail loudly otherwise.
    """
    hits = []
    for ndim in (1, 2, 3):
        for fmt, w in (("B", 1), ("H", 2), ("I", 4), ("Q", 8)):
            h = 1 + ndim * w
            if len(buf) <= h or (len(buf) - h) % itemsize:
                continue
            dims = struct.unpack("<" + fmt * ndim, buf[1:h])
            if all(dims) and int(np.prod(dims)) == (len(buf) - h) // itemsize:
                hits.append((h, dims))
    if len(hits) != 1:
        raise ValueError(f"ambiguous or unparseable ndarray header ({len(buf)} bytes): {hits}")
    h, dims = hits[0]
    return np.frombuffer(buf[h:], dtype="<u4").reshape(dims)


def mds_samples(path: str, column_names, column_sizes):
    data = pathlib.Path(path).read_bytes()
    (n,) = struct.unpack_from("<I", data, 0)
    offs = struct.unpack_from(f"<{n + 1}I", data, 4)
    var = [i for i, s in enumerate(column_sizes) if s is None]
    for i in range(n):
        s = data[offs[i]:offs[i + 1]]
        sizes = list(struct.unpack_from(f"<{len(var)}I", s, 0))
        pos = 4 * len(var)
        row = {}
        for ci, name in enumerate(column_names):
            size = column_sizes[ci] if column_sizes[ci] is not None else sizes.pop(0)
            row[name] = s[pos:pos + size]
            pos += size
        yield i, row


def prolong_docs(subset: str, mode: str, llama_tok, max_shards: int = 4):
    idx = json.loads(pathlib.Path(_hf(PROLONG, f"{subset}/proc00-64/index.json", revision=REVISIONS[PROLONG])).read_text())
    for meta in idx["shards"][:max_shards]:
        shard = f"proc00-64/{meta['raw_data']['basename']}"
        path = _hf(PROLONG, f"{subset}/{shard}", revision=REVISIONS[PROLONG])
        rows = list(mds_samples(path, meta["column_names"], meta["column_sizes"]))
        order = np.random.default_rng(SEED).permutation(len(rows))
        for r in order:
            i, row = rows[int(r)]
            ids = _mds_ndarray(row["input_ids"])
            ind = _mds_ndarray(row["indices"]).reshape(-1, 2)
            src = {"repo": PROLONG, "file": f"{subset}/{shard}", "sample": int(i)}
            if mode == "single":  # every document span in the sequence is a candidate
                for j, (a, b) in enumerate(ind):
                    if b - a < NEED // 2:  # cannot reach NEED model tokens; skip the decode
                        continue
                    yield {**src, "doc_in_sample": j}, llama_tok.decode(ids[a:b].tolist())
            else:  # WEB control: the whole packed sequence, documents joined by EOS in stored order
                yield {**src, "n_packed_docs": int(len(ind))}, [llama_tok.decode(ids[a:b].tolist()) for a, b in ind]


def arxiv_fallback():
    """PREREGISTRATION §3 fallback when ProLong's arxiv subset has too few documents of NEED tokens."""
    import pyarrow.parquet as pq

    repo = "ccdv/arxiv-summarization"
    from huggingface_hub import list_repo_files

    rev = REVISIONS[repo]
    f = sorted(x for x in list_repo_files(repo, repo_type="dataset", revision=rev) if x.startswith("document/train-00000"))[0]
    texts = pq.read_table(_hf(repo, f, revision=rev)).column("article").to_pylist()
    for r in np.random.default_rng(SEED).permutation(len(texts)):
        yield {"repo": repo, "file": f, "row": int(r), "fallback": True}, texts[r]


def pg19_docs():
    import pyarrow.parquet as pq

    repo, rev = PG19
    from huggingface_hub import list_repo_files

    f = sorted(x for x in list_repo_files(repo, repo_type="dataset", revision=rev) if "/train-00000-" in x)[0]
    t = pq.read_table(_hf(repo, f, revision=rev))
    texts = t.column("text").to_pylist()
    ids = t.column("short_book_title").to_pylist() if "short_book_title" in t.column_names else [None] * len(texts)
    for r in np.random.default_rng(SEED).permutation(len(texts)):
        yield {"repo": repo, "revision": rev, "file": f, "row": int(r), "title": ids[r]}, texts[r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "data"))
    args = ap.parse_args()
    from transformers import AutoTokenizer

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    toks = {k: AutoTokenizer.from_pretrained(v, revision=REVISIONS[v]) for k, v in TOKENIZERS.items()}
    llama = AutoTokenizer.from_pretrained(LLAMA3_TOK, revision=REVISIONS[LLAMA3_TOK])
    gens = {
        "PG19": pg19_docs(),
        "REPOS": prolong_docs("thestackv1_concat_by_repo-65536", "single", llama),
        "ARXIV": prolong_docs("arxiv", "single", llama),
        "WEB": prolong_docs("fineweb-2023-50", "packed", llama),
    }
    manifest = {"seed": SEED, "n_docs": N_DOCS, "offset": OFFSET, "span": SPAN, "tokenizers": TOKENIZERS,
                "llama3_tokenizer": LLAMA3_TOK, "corpora": {}}
    spans = {k: {} for k in toks}
    for corpus, gen in gens.items():
        picked, scanned = [], 0
        for src, text in gen:
            scanned += 1
            if isinstance(text, list):  # WEB: join packed docs with each tokenizer's EOS
                enc = {k: sum(([*t(x, add_special_tokens=False)["input_ids"], t.eos_token_id] for x in text), [])
                       for k, t in toks.items()}
            else:
                enc = {k: t(text, add_special_tokens=False)["input_ids"] for k, t in toks.items()}
            if min(len(v) for v in enc.values()) < NEED:
                continue
            rec = dict(src)
            for k, v in enc.items():
                s = np.asarray(v[OFFSET:NEED], dtype=np.int32)
                spans[k].setdefault(corpus, []).append(s)
                rec[f"sha256_{k}"] = hashlib.sha256(s.tobytes()).hexdigest()
                rec[f"ntok_{k}"] = len(v)
            picked.append(rec)
            if len(picked) == N_DOCS:
                break
        if len(picked) < N_DOCS and corpus == "ARXIV":
            print("ARXIV: ProLong arxiv short of docs; using the PREREG fallback", flush=True)
            picked, gen = [], arxiv_fallback()
            for k in toks:
                spans[k].pop(corpus, None)
            for src, text in gen:
                scanned += 1
                enc = {k: t(text, add_special_tokens=False)["input_ids"] for k, t in toks.items()}
                if min(len(v) for v in enc.values()) < NEED:
                    continue
                rec = dict(src)
                for k, v in enc.items():
                    sp = np.asarray(v[OFFSET:NEED], dtype=np.int32)
                    spans[k].setdefault(corpus, []).append(sp)
                    rec[f"sha256_{k}"] = hashlib.sha256(sp.tobytes()).hexdigest()
                    rec[f"ntok_{k}"] = len(v)
                picked.append(rec)
                if len(picked) == N_DOCS:
                    break
        if len(picked) < N_DOCS:
            raise SystemExit(f"{corpus}: only {len(picked)} qualifying docs in {scanned} scanned (PREREGISTRATION §3 fallback)")
        manifest["corpora"][corpus] = {"docs": picked, "scanned": scanned}
        print(corpus, "picked", len(picked), "of", scanned, flush=True)
    for k in toks:
        np.savez_compressed(out / f"spans_{k}.npz", **{c: np.stack(v) for c, v in spans[k].items()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
