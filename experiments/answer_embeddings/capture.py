"""
STEP 0 baseline capture — runs the FIXED query set through the CURRENT system
(question-only retrieval, exactly as-is) and saves the full results.

This is the BEFORE half of the answer-embedding experiment. It must be run and saved
BEFORE any code change. Reused verbatim later for the AFTER pass (same query set, flag ON)
so the two are strictly comparable.

Captures, per query: ranked final results (doc_key, sub_question_id, question_text,
rerank_logit, relevance/sigmoid, verify_verdict), the final verified count, entities,
retrievers fired/eligible, and fuse top/gap. Written to a JSON keyed by query.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.retrieval.graph.run import query_engine

# The EXACT query set from the spec. Grouped so the report can treat each class correctly.
QUERY_SET = [
    ("direct",      "electricity dues in Jammu and Kashmir"),
    ("direct",      "current status of Subansiri Lower hydroelectric project"),
    ("direct",      "seismic monitoring and earthquakes near NHPC dams"),
    ("direct",      "NHPC bonds raised on behalf of Ministry of Power"),
    ("answer",      "Subansiri commissioning schedule"),
    ("answer",      "hydro projects in the North East region"),
    ("answer",      "Teesta projects in Sikkim"),
    ("boilerplate", "dam safety and seismic design"),
    ("boilerplate", "matters to be replied by Ministry of Power"),
    ("boilerplate", "information given at annexure"),
    ("paraphrase",  "projects in HP"),
    ("paraphrase",  "projects in Himachal Pradesh"),
]


def _slim_result(r: dict) -> dict:
    """The comparable fields of one displayed result. Order-preserving."""
    s = r.get("signals") or {}
    return {
        "rank": r.get("rank"),
        "relevance_rank": r.get("relevance_rank"),
        "doc_key": r.get("doc_key"),
        "sub_question_id": r.get("sub_question_id"),
        "part_label": r.get("part_label"),
        "question_text": (r.get("question_text") or "")[:160],
        "relevance": r.get("relevance"),               # sigmoid(logit)
        "rerank_logit": s.get("rerank_logit"),
        "verify_verdict": r.get("verify_verdict"),
        "reply_date": r.get("reply_date"),
        "retrievers": s.get("retrievers"),
    }


def capture(tag: str, out_path: str):
    load_dotenv()
    cfg = Settings()
    # Record the flag's value so the report can prove which regime produced this file.
    use_answer = getattr(cfg, "use_answer_embeddings", None)

    records = {}
    with query_engine(cfg) as (run, deps):
        for kind, q in QUERY_SET:
            t0 = time.time()
            out = run(q)
            vm = out.get("verify_meta") or {}
            fs = out.get("fuse_stats") or {}
            results = out.get("results") or []
            records[q] = {
                "kind": kind,
                "query": q,
                "language": out.get("language"),
                "entities": out.get("entities") or [],
                "query_canon": out.get("query_canon"),
                "result_count": len(results),
                "verification_unavailable": bool(out.get("verification_unavailable")),
                "verify_checked": vm.get("checked"),
                "verify_kept": vm.get("kept"),
                "fuse_candidates": fs.get("n_candidates"),
                "fuse_top": fs.get("top_score"),
                "fuse_gap": fs.get("score_gap"),
                "retrievers_eligible": fs.get("eligible"),
                "retrievers_fired": fs.get("fired"),
                "elapsed_ms": int((time.time() - t0) * 1000),
                # ORDERED final set the officer sees + the doc_key set for quick diffing
                "results": [_slim_result(r) for r in results],
                "doc_keys": [r.get("doc_key") for r in results],
            }
            print(f"  [{kind:11}] {q[:52]:54} -> {len(results):2} results "
                  f"(verify {vm.get('kept')}/{vm.get('checked')})", flush=True)

    payload = {
        "tag": tag,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "use_answer_embeddings": use_answer,   # None before the feature exists; bool after
        "config": {
            "similarity_threshold": cfg.similarity_threshold,
            "dense_top_n": cfg.dense_top_n,
            "entity_top_n": cfg.entity_top_n,
            "rerank_candidate_pool": cfg.final_top_k,
            "rerank_model": cfg.rerank_model,
            "rrf_k": cfg.rrf_k,
            "llm_verify_enabled": cfg.llm_verify_enabled,
            "answer_embed_weight": getattr(cfg, "answer_embed_weight", None),
        },
        "queries": records,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved {tag} -> {out_path}")
    print(f"  queries captured : {len(records)}")
    print(f"  USE_ANSWER_EMBEDDINGS at capture : {use_answer}")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    capture(args.tag, args.out)
