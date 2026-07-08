"""
Structural audit of parsed.json outputs — catches likely mis-parses across the
whole corpus so you don't have to eyeball thousands of files.

For each parsed.json it re-reads the source reply text and checks the parse against
cheap structural expectations, emitting a per-file verdict + reasons. It does NOT
re-parse; it validates what was written.

    python -m phase2.audit [--organized organized] [--only SUBPATH]

Writes organized/_reports/audit.csv and prints a summary of suspicious files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")

_COMMENT = re.compile(r"\b(comments?|answer|reply|उत्तर)\b\s*[:.\-]", re.I)


def _source_signals(qdir):
    """Cheap signals from the source reply text (via pdfplumber/docx), no full parse."""
    import glob
    al = glob.glob(os.path.join(qdir, "answer_latest.*"))
    if not al:
        return None
    path = al[0]
    ext = os.path.splitext(path)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                text = "\n".join((pg.extract_text() or "") for pg in pdf.pages[:12])
        elif ext == ".docx":
            import docx
            text = "\n".join(p.text for p in docx.Document(path).paragraphs)
        else:
            return {"n_comments": None, "has_table": None}
    except Exception:
        return {"n_comments": None, "has_table": None}
    return {"n_comments": len(_COMMENT.findall(text)), "text_len": len(text)}


def audit_one(qdir):
    pj = os.path.join(qdir, "parsed.json")
    with open(pj, encoding="utf-8") as fh:
        d = json.load(fh)
    reasons = []
    sqs = d.get("sub_questions", [])
    grps = d.get("answer_groups", [])
    gids = {g["answer_group_id"] for g in grps}

    # 1) link integrity
    for sq in sqs:
        if sq.get("answer_group_id") not in gids:
            reasons.append("dangling_group_pointer")
            break
    # 2) empty answers in groups
    empty = [g["answer_group_id"] for g in grps
             if not (g.get("answer_text") or "").strip() and not g.get("answer_is_table")]
    if empty:
        reasons.append(f"empty_answer_group({len(empty)})")
    # 3) parts covered == sub_questions count
    covered = set()
    for g in grps:
        covered |= set(g.get("answers_parts", []))
    if sqs and len(covered) != len(sqs):
        reasons.append("parts_coverage_mismatch")
    # 4) source Comment: count vs groups (heuristic: distinct answers <= comments)
    sig = _source_signals(qdir)
    if sig and sig.get("n_comments"):
        nc = sig["n_comments"]
        # distinct non-empty answers should not EXCEED the number of Comment: blocks
        distinct_ans = len({(g.get("answer_text") or "").strip()[:80] for g in grps
                            if (g.get("answer_text") or "").strip()})
        if distinct_ans > nc + 0:
            reasons.append(f"more_answers_than_comments({distinct_ans}>{nc})")
        if nc >= 2 and len(grps) == 1 and len(sqs) == 1:
            reasons.append("single_part_but_multi_comment_source")
    # 5) existing review flags worth surfacing
    for fl in ("group_link_broken", "qa_count_mismatch", "table_group_uncertain",
               "no_diary_number_found"):
        if fl in d.get("extraction_flags", []):
            reasons.append(fl)

    return {
        "path": os.path.relpath(qdir),
        "diary": ";".join(d.get("diary_numbers", [])),
        "n_sub_questions": len(sqs),
        "n_answer_groups": len(grps),
        "reply_format": d.get("reply_format"),
        "source_comments": (sig or {}).get("n_comments"),
        "suspicious": bool(reasons),
        "reasons": ";".join(reasons),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--organized", default="organized")
    ap.add_argument("--only", default=None)
    args = ap.parse_args(argv)
    root = os.path.abspath(args.organized)

    rows = []
    for dp, _dn, fn in os.walk(root):
        if "_reports" in dp.split(os.sep):
            continue
        if "parsed.json" not in fn:
            continue
        if args.only and args.only.replace("/", os.sep) not in dp:
            continue
        try:
            rows.append(audit_one(dp))
        except Exception as e:
            rows.append({"path": os.path.relpath(dp), "suspicious": True,
                         "reasons": f"audit_error:{type(e).__name__}"})

    rep = os.path.join(root, "_reports")
    os.makedirs(rep, exist_ok=True)
    out = os.path.join(rep, "audit.csv")
    if rows:
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    susp = [r for r in rows if r.get("suspicious")]
    from collections import Counter
    reason_ct = Counter()
    for r in susp:
        for x in (r.get("reasons") or "").split(";"):
            if x:
                reason_ct[re.sub(r"\(.*\)", "", x)] += 1
    print(f"audited {len(rows)} files | suspicious: {len(susp)}")
    print("top reasons:")
    for k, v in reason_ct.most_common():
        print(f"  {v:4}  {k}")
    print(f"full report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
