"""
Run the 100-question dataset through the REAL retrieval pipeline and score it.

    PYTHONPATH=. python experiments/prod_readiness/run_retrieval.py --tag before

Writes:
    experiments/prod_readiness/results_<tag>.jsonl   one line per query (raw, resumable)
    experiments/prod_readiness/results_<tag>.json    compiled results + metrics

METRICS
  * paraphrase group consistency : every member of a group should return the SAME final
    doc_key set. We report exact-set-match rate and mean pairwise Jaccard per group.
  * boilerplate zero-rate        : stock phrases must return 0 results.
  * out-of-domain zero-rate      : unknown topics must return 0 results.
  * direct hit-rate              : real corpus topics should return >= 1 result.
  * latency                      : p50/p95 total ms.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from dataset import Q  # noqa: E402

from nhpc_qa.config import Settings, load_dotenv  # noqa: E402
from nhpc_qa.retrieval.graph.run import query_engine  # noqa: E402


def run_all(tag: str, out_dir: str):
    jsonl = os.path.join(out_dir, f"results_{tag}.jsonl")
    done = {}
    if os.path.exists(jsonl):                       # resumable: skip already-run ids
        with open(jsonl, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except json.JSONDecodeError:
                    pass
        print(f"resuming: {len(done)} already done")

    load_dotenv()
    cfg = Settings()
    fh = open(jsonl, "a", encoding="utf-8")
    with query_engine(cfg) as (run, deps):
        for i, q in enumerate(Q, 1):
            if q["id"] in done:
                continue
            t0 = time.time()
            try:
                out = run(q["text"])
                results = out.get("results") or []
                rec = {
                    **q,
                    "ok": True,
                    "result_count": len(results),
                    "doc_keys": [r["doc_key"] for r in results],
                    "entities": out.get("entities") or [],
                    "query_canon": out.get("query_canon"),
                    "verification_unavailable": bool(out.get("verification_unavailable")),
                    "verify_checked": (out.get("verify_meta") or {}).get("checked"),
                    "verify_kept": (out.get("verify_meta") or {}).get("kept"),
                    "widened": bool(out.get("widened")),
                    "top": [{"doc_key": r["doc_key"],
                             "relevance": r.get("relevance"),
                             "q": (r.get("question_text") or "")[:90]}
                            for r in results[:5]],
                    "ms": int((time.time() - t0) * 1000),
                }
            except Exception as e:      # noqa: BLE001 -- record the failure, keep going
                rec = {**q, "ok": False, "error": f"{type(e).__name__}: {e}",
                       "ms": int((time.time() - t0) * 1000)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[q["id"]] = rec
            n = rec.get("result_count", "ERR")
            print(f"  [{i:3}/100] {q['kind']:13} {q['text'][:56]:58} -> {n}", flush=True)
    fh.close()
    return [done[q["id"]] for q in Q]


def jaccard(a, b):
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))


def metrics(rows):
    by_group = {}
    for r in rows:
        if r.get("group"):
            by_group.setdefault(r["group"], []).append(r)

    groups = {}
    for g, members in by_group.items():
        sets = [tuple(sorted(m.get("doc_keys") or [])) for m in members]
        exact = len(set(sets)) == 1
        pairs = list(itertools.combinations(range(len(members)), 2))
        jac = [jaccard(members[i].get("doc_keys") or [], members[j].get("doc_keys") or [])
               for i, j in pairs]
        groups[g] = {
            "n": len(members),
            "exact_set_match": exact,
            "mean_jaccard": round(statistics.mean(jac), 3) if jac else 1.0,
            "counts": [m.get("result_count") for m in members],
            "queries": [m["text"][:60] for m in members],
        }

    def _rows(kind):
        return [r for r in rows if r["kind"] == kind and r.get("ok")]

    boiler = _rows("boilerplate")
    ood = _rows("out_of_domain")
    direct = _rows("direct")
    lat = [r["ms"] for r in rows if r.get("ok")]

    n_exact = sum(1 for g in groups.values() if g["exact_set_match"])
    m = {
        "paraphrase_groups_total": len(groups),
        "paraphrase_groups_exact_match": n_exact,
        "paraphrase_exact_rate": round(n_exact / max(1, len(groups)), 3),
        "paraphrase_mean_jaccard": round(statistics.mean(
            [g["mean_jaccard"] for g in groups.values()]), 3) if groups else None,
        "boilerplate_zero_rate": round(sum(1 for r in boiler if r["result_count"] == 0)
                                       / max(1, len(boiler)), 3),
        "out_of_domain_zero_rate": round(sum(1 for r in ood if r["result_count"] == 0)
                                         / max(1, len(ood)), 3),
        "direct_hit_rate": round(sum(1 for r in direct if r["result_count"] > 0)
                                 / max(1, len(direct)), 3),
        "direct_zero_queries": [r["text"][:70] for r in direct if r["result_count"] == 0],
        "errors": [r["id"] for r in rows if not r.get("ok")],
        "latency_ms_p50": int(statistics.median(lat)) if lat else None,
        "latency_ms_p95": int(sorted(lat)[int(0.95 * len(lat)) - 1]) if lat else None,
        "verification_unavailable_count": sum(
            1 for r in rows if r.get("verification_unavailable")),
    }
    return m, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    rows = run_all(args.tag, out_dir)
    m, groups = metrics(rows)

    out = {"tag": args.tag, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "metrics": m, "groups": groups, "rows": rows}
    path = os.path.join(out_dir, f"results_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"METRICS ({args.tag})")
    print("=" * 72)
    for k, v in m.items():
        print(f"  {k:34} {v}")
    print("\nGROUPS NOT MATCHING EXACTLY:")
    for g, info in groups.items():
        if not info["exact_set_match"]:
            print(f"  {g}: counts={info['counts']} jaccard={info['mean_jaccard']}")
            for qt in info["queries"]:
                print(f"      - {qt}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
