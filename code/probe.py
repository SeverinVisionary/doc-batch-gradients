#!/usr/bin/env python3
"""DOC-BATCH Phase 0 probe: per-chunk gradient sketches -> Gram matrices (docs/PREREGISTRATION.md §4).

For each document span (data/spans_<tok>.npz) and each chunk of C tokens, the chunk is run as a
standalone sequence, loss = mean next-token CE over its C-1 targets, and the full parameter gradient is
sketched with a count sketch of 2^17 buckets per parameter group (attn, mlp, embed, other); see Sketcher and
DEVIATION_001.md (the preregistered 32x32 Kronecker sketch failed G1 at feasibility). Only the groups' Gram
matrices are written, with exact squared norms and per-chunk losses.

G1 (sketch fidelity): when --exact-ref is set, the full gradient of the first chunk is kept and exact inner
products with every later chunk are recorded next to the sketched ones.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import platform
import shutil
import tempfile
import time

import numpy as np
import torch

K = 2 ** 17  # count-sketch buckets per parameter group (DEVIATION_001; was a 32x32 Kronecker sketch)
SLICE = 2 ** 24  # hash generation is sliced so large embeddings never need a full-size index tensor
SEED = 20261007
GROUPS = ("attn", "mlp", "embed", "other")


def group_of(name: str) -> str:
    n = name.lower()
    if "embed" in n or "lm_head" in n:
        return "embed"
    if "attn" in n or "attention" in n:
        return "attn"
    if "mlp" in n:
        return "mlp"
    return "other"


class Sketcher:
    """Count sketch per group: coordinate i of a parameter goes to bucket h(i) with sign s(i).

    (h, s) come from a numpy PCG64 generator seeded by (SEED, parameter name, shape) and drawn in fixed slices, so the
    sketch is the same for every chunk and every run. Inner products are unbiased with relative error about
    sqrt(2/K) for gradients without dominant single coordinates.
    """

    def __init__(self, model):
        self.seeds = {name: int(hashlib.sha256(f"{SEED}:{name}:{tuple(p.shape)}".encode()).hexdigest()[:15], 16)
                      for name, p in model.named_parameters()}

    @torch.no_grad()
    def __call__(self, model):
        out = {k: np.zeros(K) for k in GROUPS}
        sq = {k: 0.0 for k in GROUPS}
        used = set()
        for name, p in model.named_parameters():
            gr = p.grad.reshape(-1)
            k = group_of(name)
            used.add(k)
            rng = np.random.default_rng(self.seeds[name])  # PCG64 (DEVIATION_002; was torch.randint)
            for i in range(0, gr.numel(), SLICE):
                x = gr[i:i + SLICE].double().numpy()
                sq[k] += float(x @ x)
                h = rng.integers(0, 2 * K, size=x.size, dtype=np.uint32)
                out[k] += np.bincount(h >> 1, weights=x * (1.0 - 2.0 * (h & 1)), minlength=K)
        return {k: out[k].astype(np.float32) for k in GROUPS if k in used}, sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tok", required=True, choices=["pythia", "olmo2"])
    ap.add_argument("--corpus", required=True, choices=["PG19", "REPOS", "ARXIV", "WEB"])
    ap.add_argument("--docs", default="0:16", help="python slice over the 16 docs, e.g. 0:8")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--exact-ref", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="load a model/revision not listed in data/checkpoint_revisions.json (mutable Hub ref)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    here = pathlib.Path(__file__).resolve().parent.parent
    spans = np.load(here / "data" / f"spans_{args.tok}.npz")[args.corpus]
    a, b = (int(x) for x in args.docs.split(":"))
    doc_ids = list(range(16))[a:b]
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    pins = json.loads((here / "data" / "checkpoint_revisions.json").read_text())["models"]
    revision = pins.get(args.model, {}).get(args.revision)  # branch name -> pinned commit
    if revision is None:
        if not args.allow_unpinned:
            raise SystemExit(f"{args.model}@{args.revision} is not pinned in data/checkpoint_revisions.json; "
                             "add its commit there or pass --allow-unpinned")
        revision = args.revision
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, torch_dtype=torch.float32)
    model.eval()  # no dropout; gradients still flow
    sk = Sketcher(model)
    t_load = time.time() - t0
    C = args.chunk
    m = spans.shape[1] // C
    # sketch rows go to disk-backed memmaps, so RAM stays at model + gradient (1B-class models sit ~0.5 GB under
    # the cloud box's limit); the Gram is then accumulated in float64 over column blocks.
    tmp = tempfile.mkdtemp(prefix="docbatch_")
    n_rows = len(doc_ids) * m
    rows = {k: np.lib.format.open_memmap(f"{tmp}/{k}.npy", mode="w+", dtype=np.float32, shape=(n_rows, K))
            for k in GROUPS}
    present = set()
    norms = {k: [] for k in GROUPS}
    losses, index, exact = [], [], []
    ref = None
    t1 = time.time()
    for d in doc_ids:
        for j in range(m):
            ids = torch.from_numpy(spans[d, j * C:(j + 1) * C].astype(np.int64))[None]
            model.zero_grad(set_to_none=True)
            out = model(input_ids=ids, labels=ids)
            out.loss.backward()
            vecs, sq = sk(model)
            for k in GROUPS:
                if k in vecs:
                    rows[k][len(index)] = vecs[k]
                    norms[k].append(sq[k])
                    present.add(k)
            losses.append(float(out.loss))
            index.append((d, j))
            if args.exact_ref:
                # per-parameter float64 dot against a float32 copy of chunk 0's gradient (same values as the
                # former single concatenated float64 vector; avoids two full-size float64 copies -> OOM at 410M)
                if ref is None:
                    ref = [p.grad.detach().flatten().clone() for p in model.parameters()]
                exact.append(float(sum(torch.dot(r.double(), p.grad.flatten().double())
                                       for r, p in zip(ref, model.parameters()))))
            print(f"doc {d} chunk {j} loss {losses[-1]:.4f} t {time.time() - t1:.0f}s", flush=True)
    del model, ref
    gc.collect()
    grams = {}
    for k in GROUPS:
        if k in present:
            G = np.zeros((n_rows, n_rows))
            for c0 in range(0, K, 2 ** 14):
                Xb = np.asarray(rows[k][:, c0:c0 + 2 ** 14], dtype=np.float64)
                G += Xb @ Xb.T
            grams[f"gram_{k}"] = G
    shutil.rmtree(tmp, ignore_errors=True)
    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outp,
        index=np.array(index, dtype=np.int32),
        loss=np.array(losses),
        **grams,
        **{f"sqnorm_{k}": np.array(v) for k, v in norms.items() if v},
        exact_ref_dot=np.array(exact),
    )
    meta = {
        "args": vars(args), "chunks": len(index), "chunks_per_doc": m, "sketch": f"count sketch, {K} buckets per group", "seed": SEED,
        "seconds_load": t_load, "seconds_chunks": time.time() - t1,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "numpy": np.__version__, "python": platform.python_version(), "threads": torch.get_num_threads(),
        "machine": platform.machine(), "spans_sha256": hashlib.sha256(spans[doc_ids].tobytes()).hexdigest(),
    }
    outp.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
