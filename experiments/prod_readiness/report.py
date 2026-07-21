"""
Final BEFORE/AFTER report for the production-readiness pass.

    PYTHONPATH=. python experiments/prod_readiness/report.py

Reads results_before.json / results_after.json (retrieval, 100 questions) and
results_drafts_before.json / results_drafts_after.json (drafts, 20 scenarios), prints the
comparison and writes REPORT.md next to them.
"""
from __future__ import annotations

import json
import os

D = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(D, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    rb, ra = load("results_before.json"), load("results_after.json")
    db, da = load("results_drafts_before.json"), load("results_drafts_after.json")
    lines = []

    def w(s=""):
        lines.append(s)
        print(s)

    w("# NHPC production-readiness pass — before/after")
    w()
    w("Dataset: 100 questions (dataset_v1.json) — 16 paraphrase groups (45 q), 40 direct")
    w("(1-3 lines), 5 Hindi, 5 boilerplate-bait, 5 out-of-domain. Draft suite: 20 scenarios")
    w("against the two live supporting documents (financial digest, UC-projects progress).")
    w()

    if rb and ra:
        mb, ma = rb["metrics"], ra["metrics"]
        w("## Retrieval (100 questions)")
        w()
        w("| metric | before | after |")
        w("|---|---|---|")
        rows = [
            ("paraphrase groups exact-set match",
             f"{mb['paraphrase_groups_exact_match']}/{mb['paraphrase_groups_total']} ({mb['paraphrase_exact_rate']:.0%})",
             f"{ma['paraphrase_groups_exact_match']}/{ma['paraphrase_groups_total']} ({ma['paraphrase_exact_rate']:.0%})"),
            ("paraphrase mean Jaccard", mb["paraphrase_mean_jaccard"], ma["paraphrase_mean_jaccard"]),
            ("boilerplate zero-rate (want 100%)", f"{mb['boilerplate_zero_rate']:.0%}", f"{ma['boilerplate_zero_rate']:.0%}"),
            ("out-of-domain zero-rate (want 100%)", f"{mb['out_of_domain_zero_rate']:.0%}", f"{ma['out_of_domain_zero_rate']:.0%}"),
            ("direct-topic hit rate", f"{mb['direct_hit_rate']:.0%}", f"{ma['direct_hit_rate']:.0%}"),
            ("latency p50 / p95 (ms)", f"{mb['latency_ms_p50']} / {mb['latency_ms_p95']}",
             f"{ma['latency_ms_p50']} / {ma['latency_ms_p95']}"),
            ("errors", len(mb["errors"]), len(ma["errors"])),
        ]
        for name, b, a in rows:
            w(f"| {name} | {b} | {a} |")
        w()
        w("### Groups still not exact after")
        for g, info in ra["groups"].items():
            if not info["exact_set_match"]:
                w(f"- `{g}` counts={info['counts']} jaccard={info['mean_jaccard']}")
        w()
        w("### Direct queries still returning zero (after)")
        for q in ma["direct_zero_queries"]:
            w(f"- {q}")
        w()

    if db and da:
        sb, sa = db["summary"], da["summary"]
        w("## Drafts (20 scenarios)")
        w()
        w(f"| | before | after |")
        w(f"|---|---|---|")
        w(f"| drafts generated | {sb['drafts_ok']}/{sb['scenarios']} | {sa['drafts_ok']}/{sa['scenarios']} |")
        w(f"| checks passed | {sb['checks_passed']}/{sb['checks_total']} | {sa['checks_passed']}/{sa['checks_total']} |")
        w()
        if sa["failed_checks"]:
            w("### Checks still failing (after)")
            for f in sa["failed_checks"]:
                w(f"- [{f['id']}] {f['query']}")
                for c in f["fails"]:
                    w(f"    - {c['name']} — {c['note']}")
        else:
            w("All draft checks pass.")
        w()

    w("## What was changed (universal fixes, not point patches)")
    for item in [
        "Query normalization pipeline (entities -> canonicalise -> synonyms(protected) -> "
        "filler-strip -> entity re-match): 'all/list of/the/number of' wrappers and "
        "abbreviations no longer split result sets",
        "canonicalise_text joiner now matches '&' ('J&K' == 'Jammu and Kashmir')",
        "9 new concept-synonym groups (hydel/hydropower family, rehabilitation==resettlement, "
        "expenditure==spending, status qualifiers, vacancies, DPR, DISCOM, recruitment, "
        "'the country'==India)",
        "Entity dictionary consolidation (entities/dedupe.py): 77 duplicate entities merged "
        "(6 'Subansiri Lower' variants -> 1); years kept as distinguishing tokens",
        "PSU/PSUs/J&K/JK aliases added",
        "SIMILARITY_THRESHOLD 0.1 -> 0.02 (sigmoid is a cost bound, not a relevance gate; "
        "0.1 was silently zeroing real topics)",
        "RETRIEVE_DENSE_TOP_N 30 -> 50 (tail-topic recall)",
        "LLM-verify prompt: generic-fragment rule (boilerplate matched to boilerplate = no "
        "match) + short-topic-query rule; verify cache invalidated on prompt change",
        "Plateau guard implemented but default OFF (measured: real frequent topics saturate "
        "like fragments on this corpus; the LLM rule discriminates, the shape does not)",
        "Weak-match fallback: empty result sets now carry the top-3 nearest candidates, "
        "explicitly labelled, so an officer is never left with a bare empty page",
        "Draft rule E: using a relevant attached document is MANDATORY; unused docs are "
        "flagged in code (docs_unused + auto-gap) — never silently ignored",
        "DRAFT_CONTEXT_K 5 -> 10 (the draft now sees what the officer sees)",
        "Table ingestion: transposed-layout detection (title-spill + numbered-attribute "
        "signals), '(col N: label missing)' placeholders for broken scanned headers",
        "Supporting docs re-ingested: financial digest now carries 'FY 2020-21 to 2024-25', "
        "UC-progress 'as on 30.06.2026' + transposed orientation",
    ]:
        w(f"- {item}")
    w()
    w("## Artifacts")
    for f in ["dataset_v1.json", "results_before.json", "results_after.json",
              "results_drafts_before.json", "results_drafts_after.json",
              "run_retrieval.py", "run_drafts.py", "dataset.py"]:
        w(f"- experiments/prod_readiness/{f}")

    with open(os.path.join(D, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwritten -> {os.path.join(D, 'REPORT.md')}")


if __name__ == "__main__":
    main()
