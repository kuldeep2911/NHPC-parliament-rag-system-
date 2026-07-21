"""
BEFORE/AFTER comparison for the answer-embedding experiment.

Reads before_baseline.json (question-only) and after_flag_on.json (answer-enabled), both
produced by capture.py over the SAME fixed query set, and prints a per-query diff plus a
class-by-class summary. Presents evidence only — it does NOT recommend enabling the flag.
"""
from __future__ import annotations

import json
import sys


def load(p):
    return json.load(open(p, encoding="utf-8"))


def by_key(results):
    return {r["doc_key"]: r for r in results}


def main(before_path, after_path):
    B = load(before_path)
    A = load(after_path)
    bq, aq = B["queries"], A["queries"]

    print("=" * 92)
    print("ANSWER-EMBEDDING EXPERIMENT — BEFORE vs AFTER")
    print("=" * 92)
    print(f"BEFORE: use_answer_embeddings={B['use_answer_embeddings']}  "
          f"threshold={B['config']['similarity_threshold']}  weight={B['config']['answer_embed_weight']}")
    print(f"AFTER : use_answer_embeddings={A['use_answer_embeddings']}  "
          f"threshold={A['config']['similarity_threshold']}  weight={A['config']['answer_embed_weight']}")
    print()

    # per-class tallies
    tally = {}
    para = {}

    for q, brec in bq.items():
        arec = aq[q]
        kind = brec["kind"]
        bkeys = brec["doc_keys"]
        akeys = arec["doc_keys"]
        bset, aset = set(bkeys), set(akeys)
        new = [k for k in akeys if k not in bset]      # appeared with answers on
        dropped = [k for k in bkeys if k not in aset]  # lost when answers on
        # ranking change among the docs present in BOTH
        common = [k for k in akeys if k in bset]
        reordered = any(
            [k for k in bkeys if k in aset].index(k) != common.index(k) for k in common
        ) if common else False

        t = tally.setdefault(kind, {"new": 0, "dropped": 0, "queries": 0,
                                    "before": 0, "after": 0})
        t["queries"] += 1
        t["new"] += len(new)
        t["dropped"] += len(dropped)
        t["before"] += len(bkeys)
        t["after"] += len(akeys)
        if kind == "paraphrase":
            para[q] = akeys

        flag = ""
        if kind == "boilerplate" and new:
            flag = "  ⚠ FALSE POSITIVE RISK (new boilerplate matches)"
        elif kind == "answer" and new:
            flag = "  ✓ recall gain candidate"

        print("-" * 92)
        print(f"[{kind}] {q!r}")
        print(f"    count      : {len(bkeys)} -> {len(akeys)}"
              f"   (verify kept {brec['verify_kept']}/{brec['verify_checked']} "
              f"-> {arec['verify_kept']}/{arec['verify_checked']}){flag}")
        print(f"    entities   : {brec['entities']}")
        if new:
            print(f"    NEW ({len(new)}):")
            akmap = by_key(arec["results"])
            for k in new:
                r = akmap[k]
                print(f"        + {k}  sig={r['relevance']} logit={r['rerank_logit']} "
                      f"verdict={r['verify_verdict']}")
                print(f"          via retrievers={r['retrievers']}  Q={r['question_text'][:70]!r}")
        if dropped:
            print(f"    DROPPED ({len(dropped)}):")
            for k in dropped:
                print(f"        - {k}")
        if not new and not dropped:
            print(f"    unchanged set{' (reordered)' if reordered else ''}")

    # ---- paraphrase stability ----
    print("\n" + "=" * 92)
    print("PARAPHRASE STABILITY  (HP  ==  Himachal Pradesh)")
    print("=" * 92)
    pkeys = list(para.values())
    if len(pkeys) == 2:
        same_set = set(pkeys[0]) == set(pkeys[1])
        same_order = pkeys[0] == pkeys[1]
        print(f"  same doc_key SET  : {same_set}")
        print(f"  same ORDER        : {same_order}")
        print("  VERDICT: paraphrase pair still returns the same results"
              if same_set else "  ⚠ VERDICT: paraphrase pair DIVERGED with answers on")

    # ---- class summary ----
    print("\n" + "=" * 92)
    print("SUMMARY BY QUERY CLASS")
    print("=" * 92)
    print(f"  {'class':12} {'queries':>7} {'results before':>15} {'results after':>14} "
          f"{'new':>5} {'dropped':>8}")
    for kind, t in tally.items():
        print(f"  {kind:12} {t['queries']:>7} {t['before']:>15} {t['after']:>14} "
              f"{t['new']:>5} {t['dropped']:>8}")

    print("\n" + "=" * 92)
    print("READING (evidence, not a recommendation)")
    print("=" * 92)
    bt = tally.get("boilerplate", {})
    at = tally.get("answer", {})
    print(f"  RECALL GAIN  : answer-content queries added {at.get('new',0)} new result(s) "
          f"across {at.get('queries',0)} queries.")
    print(f"  FALSE POSITIVES: boilerplate-bait queries added {bt.get('new',0)} new result(s) "
          f"(want 0 — these survive the LLM verify only if genuinely similar).")
    print(f"  DROPPED       : {sum(t['dropped'] for t in tally.values())} previously-shown "
          f"result(s) lost anywhere (want 0 — answers should ADD, not remove).")
    print("  The flag remains default FALSE. Decision is yours from this evidence.")


if __name__ == "__main__":
    b = sys.argv[1] if len(sys.argv) > 1 else "experiments/answer_embeddings/before_baseline.json"
    a = sys.argv[2] if len(sys.argv) > 2 else "experiments/answer_embeddings/after_flag_on.json"
    main(b, a)
