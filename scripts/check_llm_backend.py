"""
Smoke-test the configured LLM backend end to end, without touching the corpus.

    python -X utf8 scripts/check_llm_backend.py                # uses NHPC_LLM_BACKEND
    python -X utf8 scripts/check_llm_backend.py --backend gemini
    python -X utf8 scripts/check_llm_backend.py --backend ollama

Checks, in order:
  1. config validates (key present, base URL set)
  2. the provider answers a trivial JSON request
  3. the provider returns usable LINE SPANS on a miniature reply, and
     spans.verify_and_repair accepts them

Prints the model name and never prints the API key.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.config import Config, load_dotenv           # noqa: E402
from phase2.providers import get_llm, BackendError       # noqa: E402
from phase2 import spans as _spans                        # noqa: E402
from phase2.extract import _SPAN_SYSTEM, _is_real_model   # noqa: E402

MINI_REPLY = "\n".join([
    "Subject: Information for framing the reply of Lok Sabha question no. 1234",
    "a) whether NHPC has commissioned any new project;",
    "Comment: Yes, NHPC commissioned one project in FY 2023-24.",
    "b) the installed capacity thereof; and",
    "c) the details thereof?",
    "Comment: The capacity is 800 MW. Details are placed at Annexure-I.",
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None,
                    choices=[None, "gemini", "groq", "ollama", "deterministic"])
    args = ap.parse_args()

    load_dotenv()
    cfg = Config()
    if args.backend:
        cfg.llm_backend = args.backend

    pb, lb = cfg.resolve_backends()
    print(f"parser_backend = {pb}")
    print(f"llm_backend    = {lb}")

    errs = cfg.validate()
    if errs:
        print("\nCONFIG ERRORS:")
        for e in errs:
            print("  -", e)
        return 1
    print("config         = OK")

    llm = get_llm(cfg)
    print(f"provider       = {type(llm).__name__}  ({llm.name})")
    print(f"real model     = {_is_real_model(llm)}")

    # 1. trivial JSON round-trip
    print("\n[1/2] trivial JSON call ...", flush=True)
    try:
        obj = llm.complete_json(
            'Return STRICT JSON only. Reply with exactly {"ok":true}.',
            "Say ok.", schema_hint=None)
        print("      ->", json.dumps(obj)[:120])
    except BackendError as e:
        print("      FAILED:", e)
        return 1

    # 2. real span extraction on a miniature reply
    print("\n[2/2] line-span extraction ...", flush=True)
    numbered = _spans.number_lines(MINI_REPLY)
    user = json.dumps({"document": numbered,
                       "n_lines": len(MINI_REPLY.split("\n")),
                       "metadata_question_id": "1234",
                       "n_tables": 0}, ensure_ascii=False)
    try:
        out = llm.complete_json(_SPAN_SYSTEM, user, schema_hint="sub_questions")
    except BackendError as e:
        print("      FAILED:", e)
        return 1

    sqs, verrs, reps = _spans.verify_and_repair(out, MINI_REPLY)
    print("      raw:", json.dumps(out)[:220])
    if reps:
        print("      repairs:", reps)
    if verrs:
        print("      VERIFY ERRORS:", verrs)
        return 1

    for sq in sqs:
        print(f"      ({sq['part_label']}) q={sq['question_lines']} a={sq['answer_lines']}")
        print(f"          Q: {sq['question_text'][:64]}")
        print(f"          A: {sq['answer_text'][:64]}")

    groups = _spans.group_answers(sqs)
    print(f"\n      {len(sqs)} sub-questions -> {len(groups)} answer group(s)")
    for g in groups:
        print(f"        parts={g['parts']} lines={g['answer_lines']}")

    shared = [g for g in groups if len(g["parts"]) > 1]
    print("\n      shared-answer detection:",
          "yes " + str([g["parts"] for g in shared]) if shared else "none found")
    print("\nBACKEND OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
