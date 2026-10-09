#!/usr/bin/env python3
"""DOC-BATCH Phase 0 analysis: Gram matrices -> rho(Delta), R(W), B_simple, gates and verdict (docs/PREREGISTRATION.md §5-6).

Input: results/<model>/<revision>/<corpus>[_c<chunk>][_<part>].npz from probe.py. Shards of one cell (same
model, revision, corpus, chunk) are pooled; cross-document pairs exist only within a shard.

Definitions (per cell, Gram K over chunks g_{d,k}):
  S_cross  = mean K_ij over pairs from different documents        (estimates |G|^2, G = corpus mean gradient)
  sigma2   = mean K_ii - S_cross                                    (per-chunk noise variance)
  rho(dl)  = (mean_{d,k} K_{(d,k),(d,k+dl)} - S_cross) / sigma2     (same-document noise correlation at lag dl)
  rho_long = mean of rho(dl) over lags dl*C in [2048, 16384)
  R(m)     = (mean over docs and aligned blocks of m chunks of ||block mean||^2 - S_cross) / (sigma2 / m)
             = noise variance of a document-contiguous batch unit over that of m independent chunks
             (B_simple ratio, contiguous / shuffled, at equal tokens). R(16K) is deff(16K).
  B_simple = C * sigma2 / S_cross tokens (shuffled chunks); contiguous = R(W) * that.
CIs: 95% percentile bootstrap over documents (2000 reps): documents are drawn with replacement from the pooled
cell and the draws are regrouped by source shard (see boot()).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
from collections import defaultdict

import numpy as np

B = 2000
SEED = 20261007
LONG = ("PG19", "REPOS", "ARXIV")
# fraction of training at each checkpoint (PREREGISTRATION §2)
FRAC = {
    ("EleutherAI/pythia-410m", "step1000"): 1000 / 143000, ("EleutherAI/pythia-410m", "step7000"): 7000 / 143000,
    ("EleutherAI/pythia-410m", "step36000"): 36000 / 143000, ("EleutherAI/pythia-410m", "step143000"): 1.0,
    ("EleutherAI/pythia-1.4b", "step7000"): 7000 / 143000, ("EleutherAI/pythia-1.4b", "step143000"): 1.0,
    ("allenai/OLMo-2-0425-1B", "stage1-step90000-tokens189B"): 189 / 4001,
    ("allenai/OLMo-2-0425-1B", "stage1-step1907359-tokens4001B"): 1.0,
}


def cell_stats(shards, C, group="all"):
    """shards: list of (K, doc_of_row, chunk_of_row[, source_doc_of_row]). Returns dict of statistics.

    Cross pairs use the source document, so a document drawn twice by the bootstrap never pairs with its
    own copy as if the two were independent.
    """
    same = defaultdict(list)  # lag -> list of K values
    diag, cross = [], []
    blocks = defaultdict(list)  # m -> list of ||block sum||^2 / m^2
    for K, doc, ch, *src in shards:
        src = src[0] if src else doc
        diag.append(np.diag(K))
        dd = src[:, None] != src[None, :]
        cross.append(K[dd])
        n_chunks = ch.max() + 1
        for d in np.unique(doc):
            idx = np.where(doc == d)[0][np.argsort(ch[doc == d])]
            Kd = K[np.ix_(idx, idx)]
            for lag in range(1, n_chunks):
                same[lag].append(np.diagonal(Kd, offset=lag))
            m = 1
            while m <= n_chunks:
                for s in range(0, n_chunks - m + 1, m):
                    blocks[m].append(Kd[s:s + m, s:s + m].sum() / m ** 2)
                m *= 2
    s_cross = np.concatenate(cross).mean()
    sig2 = np.concatenate(diag).mean() - s_cross
    rho = {lag: (np.concatenate(v).mean() - s_cross) / sig2 for lag, v in same.items()}
    long_lags = [lag for lag in rho if 2048 <= lag * C < 16384]
    R = {m * C: (np.mean(v) - s_cross) / (sig2 / m) for m, v in blocks.items()}
    return {
        "S_cross": s_cross, "sigma2": sig2,
        "rho": {int(lag * C): float(r) for lag, r in sorted(rho.items())},
        "rho_long": float(np.mean([rho[lag] for lag in long_lags])) if long_lags else float("nan"),
        "R": {int(w): float(r) for w, r in sorted(R.items())},
        "B_simple_shuf_tokens": float(C * sig2 / s_cross) if s_cross > 0 else float("nan"),
    }


def secondary(files, shards, st, C):
    """PREREGISTRATION §8 diagnostics: per-document rho_long (using the cell's S_cross and sigma2) and per-chunk losses."""
    per_doc = []
    for K, doc, ch in shards:
        for d in np.unique(doc):
            idx = np.where(doc == d)[0][np.argsort(ch[doc == d])]
            Kd = K[np.ix_(idx, idx)]
            v = [np.diagonal(Kd, offset=lag).mean() for lag in range(1, len(idx)) if 2048 <= lag * C < 16384]
            per_doc.append(float((np.mean(v) - st["S_cross"]) / st["sigma2"]) if v else float("nan"))
    losses = [np.load(f)["loss"] for f in files]
    idx = [np.load(f)["index"] for f in files]
    m = max(int(i[:, 1].max()) + 1 for i in idx)
    by_pos = [float(np.mean(np.concatenate([l[i[:, 1] == j] for l, i in zip(losses, idx)]))) for j in range(m)]
    return {"rho_long_per_doc": per_doc,
            "rho_long_per_doc_quartiles": [float(q) for q in np.nanpercentile(per_doc, [0, 25, 50, 75, 100])],
            "loss_mean": float(np.mean(np.concatenate(losses))), "loss_by_chunk_position": by_pos}


def load_cell(files, group):
    shards = []
    for f in files:
        z = np.load(f)
        if group == "all":
            K = sum(z[k] for k in z.files if k.startswith("gram_"))
            sq = sum(z[k] for k in z.files if k.startswith("sqnorm_"))
        else:
            K, sq = z[f"gram_{group}"].copy(), z[f"sqnorm_{group}"]
        K[np.diag_indices_from(K)] = sq  # exact squared norms on the diagonal (DEVIATION_001)
        idx = z["index"]
        shards.append((K, idx[:, 0], idx[:, 1]))
    return shards


def boot(shards, C, rng):
    """Document bootstrap over the pooled cell: draw n_docs documents with replacement from the union of all
    shards, then regroup the draws by source shard (cross-document pairs exist only within a shard)."""
    pool = [(si, d) for si, (K, doc, ch) in enumerate(shards) for d in np.unique(doc)]
    out = []
    for _ in range(B):
        pick = [pool[i] for i in rng.integers(0, len(pool), size=len(pool))]
        bs = []
        for si, (K, doc, ch) in enumerate(shards):
            mine = [d for s_, d in pick if s_ == si]
            if not mine:
                continue
            rows, newdoc, newch = [], [], []
            for i, d in enumerate(mine):  # relabel so a doc drawn twice counts as two docs
                r = np.where(doc == d)[0]
                rows.append(r)
                newdoc.append(np.full(len(r), i))
                newch.append(ch[r])
            rows = np.concatenate(rows)
            bs.append((K[np.ix_(rows, rows)], np.concatenate(newdoc), np.concatenate(newch), doc[rows]))
        st = cell_stats(bs, C)
        out.append((st["rho_long"], st["R"], st["B_simple_shuf_tokens"], st["S_cross"]))
    return out


def ci(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)])
    return [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))] if len(xs) else [None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(pathlib.Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-boot", action="store_true")
    args = ap.parse_args()
    root = pathlib.Path(args.results)
    cells = defaultdict(list)
    for f in sorted(root.rglob("*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text())["args"]
        cells[(meta["model"], meta["revision"], meta["corpus"], meta["chunk"])].append(f)
    rng = np.random.default_rng(SEED)
    report = {"cells": [], "gates": {}, "criteria": {}}
    rboot = {}  # (model, rev, corpus, C) -> bootstrap draws of R(16K), for ADDENDUM_001 ratios
    for (model, rev, corpus, C), files in sorted(cells.items()):
        shards = load_cell(files, "all")
        per = [set(np.unique(s[1]).tolist()) for s in shards]
        ids = set().union(*per)
        overlap = sorted(d for d in ids if sum(d in p for p in per) > 1)
        if overlap:
            raise SystemExit(f"overlapping shards in {model} {rev} {corpus} c={C}: docs {overlap}")
        st = cell_stats(shards, C)
        row = {"model": model, "revision": rev, "corpus": corpus, "chunk": C,
               "frac_training": FRAC.get((model, rev)), "n_docs": len(ids), "doc_ids": sorted(ids),
               "overlapping_docs": overlap,
               **{k: (float(v) if np.isscalar(v) else v) for k, v in st.items()}}
        if not args.no_boot:
            bt = boot(shards, C, rng)
            row["rho_long_ci"] = ci([b[0] for b in bt])
            row["R_16K_ci"] = ci([b[1].get(16384, np.nan) for b in bt])
            rboot[(model, rev, corpus, C)] = np.array([b[1].get(16384, np.nan) for b in bt])
            row["B_simple_shuf_ci"] = ci([b[2] for b in bt])
            row["S_cross_ci"] = ci([b[3] for b in bt])
            row["boot_nonfinite_frac"] = float(np.mean([not np.isfinite(b[1].get(16384, np.nan)) for b in bt]))
        row["secondary"] = secondary(files, shards, st, C)
        row["groups"] = {}
        for g in ("attn", "mlp", "embed", "other"):  # absent groups raise KeyError
            try:
                gs = cell_stats(load_cell(files, g), C)
                row["groups"][g] = {"rho_long": gs["rho_long"], "R_16K": gs["R"].get(16384)}
            except KeyError:
                pass
        report["cells"].append(row)
        print(f"{model:28s} {rev:32s} {corpus:6s} c={C:5d} rho_long={st['rho_long']:+.4f} "
              f"R16K={st['R'].get(16384, float('nan')):.2f} Bsimple={st['B_simple_shuf_tokens']:.3g} "
              f"ci={row.get('rho_long_ci')} {row.get('R_16K_ci')}", flush=True)

    # G1 sketch fidelity (pythia-410m shards run with --exact-ref)
    diffs, nrat = [], []
    for f in sorted(root.rglob("*.npz")):
        z = np.load(f)
        ex = z["exact_ref_dot"]
        if len(ex) == 0:
            continue
        K = sum(z[k] for k in z.files if k.startswith("gram_"))
        sq = sum(z[k] for k in z.files if k.startswith("sqnorm_"))
        n0 = np.sqrt(sq[0] * sq[1:])
        diffs += list(K[0, 1:] / n0 - ex[1:] / n0)
        nrat += list(np.diag(K) / sq)
    if diffs:
        report["gates"]["G1"] = {"mean_cos_diff": float(np.mean(diffs)), "sd_cos_diff": float(np.std(diffs)),
                                 "mean_norm_ratio": float(np.mean(nrat)), "n_pairs": len(diffs),
                                 "pass": bool(abs(np.mean(diffs)) <= 0.003 and np.std(diffs) <= 0.03
                                              and abs(np.mean(nrat) - 1) <= 0.03)}
    finals = [c for c in report["cells"] if c["frac_training"] == 1.0 and c["chunk"] == 512]
    web = [c for c in finals if c["corpus"] == "WEB"]
    if web:
        report["gates"]["G2"] = {"cells": [(c["model"], c["R"].get(16384), c["rho_long"]) for c in web],
                                 "pass": all(c["R"].get(16384, 9) < 1.2 and c["rho_long"] < 0.01 for c in web)}
    report["gates"]["G3"] = {"cells": [(c["model"], c["corpus"], c.get("S_cross_ci")) for c in finals],
                             "pass": bool(finals) and all((c.get("S_cross_ci") or [None])[0] is not None
                                                          and c["S_cross_ci"][0] > 0 for c in finals)}

    prim = [c for c in report["cells"] if c["chunk"] == 512 and c["corpus"] in LONG]
    fin = [c for c in prim if c["frac_training"] == 1.0]
    late = [c for c in prim if (c["frac_training"] or 0) >= 0.25]
    early = [c for c in prim if (c["frac_training"] or 1) < 0.05]

    def robust(cs, key, thr):
        return all(c.get(key) and c[key][1] is not None and c[key][1] < thr for c in cs)

    a = bool(prim) and all(c["rho_long"] < 0.01 for c in prim)
    b = bool(fin) and all(c["R"].get(16384, 0) < 1.5 for c in fin)
    c_ = (bool(early) and any(c["R"].get(16384, 0) >= 1.5 for c in early)
          and all(c["R"].get(16384, 0) < 1.5 for c in late))
    report["criteria"] = {
        "a": {"fires": a, "robust": a and robust(prim, "rho_long_ci", 0.01)},
        "b": {"fires": b, "robust": b and robust(fin, "R_16K_ci", 1.5)},
        "c": {"fires": c_, "robust": c_ and robust(late, "R_16K_ci", 1.5)},
    }
    # every preregistered cell (PREREGISTRATION §2-3: each model x checkpoint in FRAC x 4 corpora, c=512) with all 16 docs
    have = {(c["model"], c["revision"], c["corpus"]): c["n_docs"] for c in report["cells"] if c["chunk"] == 512}
    incomplete = [[m, r, k, have.get((m, r, k), 0)] for (m, r) in FRAC for k in LONG + ("WEB",)
                  if have.get((m, r, k), 0) < 16]
    # n_docs is the size of the union of document ids (0-15); overlapping shards abort above
    report["incomplete_cells"] = incomplete
    gates_ok = all(report["gates"].get(k, {}).get("pass", False) for k in ("G1", "G2"))  # absent gate = not passed
    if incomplete:
        verdict = "INCOMPLETE (preregistered cells missing or short of 16 docs)"
        if not gates_ok:
            verdict += "; instrument gate failed"
    elif not gates_ok:
        verdict = "INCONCLUSIVE (instrument gate failed)"
    elif a or b or c_:
        verdict = "KILL" if all(report["criteria"][k]["robust"] for k in "abc" if report["criteria"][k]["fires"]) \
            else "KILL (not robust: a firing criterion's CI crosses its threshold)"
    else:
        verdict = "SURVIVES"
    report["verdict"] = verdict
    report["addendum_001"] = addendum_001(report, rboot)
    print(json.dumps({"gates": report["gates"], "criteria": report["criteria"], "verdict": verdict}, indent=1))
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(report, indent=1))


CONFIRM = (("EleutherAI/pythia-1.4b", "step143000"), ("allenai/OLMo-2-0425-1B", "stage1-step1907359-tokens4001B"))


def addendum_001(report, rboot):
    """ADDENDUM_001.md: control-relative R_rel = R_long(16K) / R_WEB(16K) on the 1B-class final checkpoints.

    Corpora are resampled independently, so the i-th draws of the two cells pair into one ratio draw.
    """
    cells = {(c["model"], c["revision"], c["corpus"]): c for c in report["cells"] if c["chunk"] == 512}
    rel, missing = [], []
    for model, rev in CONFIRM:
        web = cells.get((model, rev, "WEB"))
        for corpus in LONG + ("WEB",):
            c = cells.get((model, rev, corpus))
            if c is None or c["n_docs"] < 16:
                missing.append([model, rev, corpus, 0 if c is None else c["n_docs"]])
        if web is None:
            continue
        for corpus in LONG:
            c = cells.get((model, rev, corpus))
            if c is None:
                continue
            r = c["R"].get(16384, np.nan) / web["R"].get(16384, np.nan)
            bw, bc = rboot.get((model, rev, "WEB", 512)), rboot.get((model, rev, corpus, 512))
            rci = ci(bc / bw) if bw is not None and bc is not None else [None, None]
            rel.append({"model": model, "revision": rev, "corpus": corpus, "R_rel": float(r), "R_rel_ci": rci,
                        "n_docs": c["n_docs"], "n_docs_web": web["n_docs"]})
    webs = [cells[(m, r, "WEB")] for m, r in CONFIRM if (m, r, "WEB") in cells]
    g2 = {"cells": [(c["model"], c["rho_long"], c.get("rho_long_ci")) for c in webs],
          "pass": len(webs) == len(CONFIRM) and all(c["rho_long"] < 0.01 for c in webs)}
    b = len(rel) == len(CONFIRM) * len(LONG) and all(x["R_rel"] < 1.5 for x in rel)
    b_rob = b and all(x["R_rel_ci"][1] is not None and x["R_rel_ci"][1] < 1.5 for x in rel)
    carries = {k: all(x["R_rel_ci"][0] is not None and x["R_rel_ci"][0] >= 1.5
                      for x in rel if x["corpus"] == k) and sum(x["corpus"] == k for x in rel) == len(CONFIRM)
               for k in LONG}
    crit = report["criteria"]
    if missing:
        verdict = "INCOMPLETE (confirmatory cells missing)"
    elif not (report["gates"].get("G1", {}).get("pass") and g2["pass"]):
        verdict = "INCONCLUSIVE (instrument gate failed)"
    elif crit["a"]["fires"] or b or crit["c"]["fires"]:
        rob = [crit[k]["robust"] for k in "ac" if crit[k]["fires"]] + ([b_rob] if b else [])
        verdict = "KILL" if all(rob) else "KILL (not robust)"
    else:
        verdict = "SURVIVES"
    out = {"G2_prime": g2, "R_rel": rel, "b_prime": {"fires": b, "robust": b_rob},
           "carries": carries, "missing": missing, "verdict": verdict}
    print(json.dumps({"addendum_001": out}, indent=1))
    return out


if __name__ == "__main__":
    main()
