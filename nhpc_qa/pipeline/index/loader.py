"""
Phase-3 loader: every parsed.json -> Postgres.

    python -m nhpc_qa.pipeline.index.loader --dry-run           # validate + report, ZERO writes
    python -m nhpc_qa.pipeline.index.loader --limit 5           # load the first 5
    python -m nhpc_qa.pipeline.index.loader --only 8773         # load specific question folders
    python -m nhpc_qa.pipeline.index.loader                     # load everything
    python -m nhpc_qa.pipeline.index.loader --force             # re-load (and mark for re-embed)

POLICY — EVERYTHING LOADS. needs_review and extraction_flags are DEVELOPER WARNINGS:
they are stored, queryable, and reported, but they NEVER exclude a record from loading,
embedding, or retrieval. There is no quarantine. A document is only 'failed' when the
database physically refuses it (a hard constraint/IO error); anything else -- a dangling
link, a missing optional field -- is logged as a WARNING and the document still loads.

IDEMPOTENT: every row is UPSERTed on its deterministic primary key (8773_a, 8773_g3,
8773_g3_t1, 8773_g3_t1_r1), so re-running UPDATES in place and never duplicates.

TRANSACTIONAL PER DOCUMENT: one document = one transaction. A failure rolls back THAT
document only; the run continues. That makes the loader resumable -- just run it again.

READ-ONLY on parsed.json. File paths stay relative to the organized/ root.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.config.index import session_year
from nhpc_qa.core.db.session import connect

VALID_ANSWER_TYPES = {"substantive", "deferred_to_ministry", "nil", "not_applicable"}


# ---------------------------------------------------------------------------
# validation — produces WARNINGS, never exclusions
# ---------------------------------------------------------------------------

def validate(doc: dict) -> list:
    """Structural checks. Returns warnings; NONE of them prevent loading."""
    w = []
    if not doc.get("question_id"):
        w.append("missing question_id")
    if not doc.get("house"):
        w.append("missing house")
    if doc.get("house") not in (None, "lok_sabha", "rajya_sabha", "vidhan_sabha"):
        w.append(f"unexpected house {doc.get('house')!r}")

    gids = {g.get("answer_group_id") for g in doc.get("answer_groups") or []}
    for sq in doc.get("sub_questions") or []:
        if not sq.get("sub_question_id"):
            w.append("sub_question with no sub_question_id")
        if not (sq.get("question_text") or "").strip():
            w.append(f"empty question_text on {sq.get('sub_question_id')}")
        agid = sq.get("answer_group_id")
        if agid not in gids:
            # DANGLING LINK: reported, but the row still loads (FK would reject it, so
            # the loader drops just that link -- see _load_sub_questions).
            w.append(f"dangling answer_group_id {agid!r} on {sq.get('sub_question_id')}")

    for g in doc.get("answer_groups") or []:
        at = g.get("answer_type")
        if at is not None and at not in VALID_ANSWER_TYPES:
            w.append(f"unknown answer_type {at!r} on {g.get('answer_group_id')}")
    return w


# ---------------------------------------------------------------------------
# row builders (parsed.json -> table rows)
# ---------------------------------------------------------------------------

def _ts(val):
    """ISO-8601 -> datetime, tolerant of 'Z' and None."""
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except ValueError:
        return None


def doc_key(doc, rel_path):
    """
    The GLOBALLY UNIQUE key of a document: '<session>/<house>/<question_id>'.

    question_id alone is NOT unique -- it is the parliament diary number, and the same
    number is reused in later sessions for a completely different question (diary 1894
    exists in both 2023-jan-apr/lok_sabha and 2025-monsoon/rajya_sabha). Keying on it
    made the second document UPSERT over the first and silently destroy it: 9 diaries,
    25 sub_questions and 15 answer_groups were lost on the full corpus.

    This equals the folder path under organized/, so a row traces straight back to its
    file. Falls back to the on-disk relative path if a field is missing.
    """
    s, h, q = doc.get("session"), doc.get("house"), doc.get("question_id")
    if s and h and q:
        return f"{s}/{h}/{q}"
    return rel_path


def _ns(key, local_id):
    """Namespace a Phase-2 id into a globally unique one: '<doc_key>#<local id>'.
    The Phase-2 id is kept alongside in a *_local column, so nothing is lost."""
    return f"{key}#{local_id}"


def _diary_row(doc, rel_path):
    fm = doc.get("file_metadata") or {}
    return {
        "doc_key": doc_key(doc, rel_path),
        "question_id": doc.get("question_id"),
        "diary_numbers": doc.get("diary_numbers") or [],
        "house": doc.get("house"),
        "session": doc.get("session"),
        "session_year": session_year(doc.get("session")),
        "state": doc.get("state"),
        "subject": doc.get("subject"),
        # The reply date, extracted by the LLM during the span-extraction call and
        # validated (parse/dates.py). Nullable on purpose: an undated document still loads
        # and stays fully retrievable -- the date only orders the DISPLAY.
        "reply_date": doc.get("reply_date"),
        "starred": doc.get("starred"),
        "reply_format": doc.get("reply_format"),
        "is_nhpc_relevant": doc.get("is_nhpc_relevant"),
        "document_language": doc.get("document_language"),
        "layout_structure": doc.get("layout_structure"),
        "layout_case_detected": doc.get("layout_case_detected"),
        "qa_table": doc.get("qa_table"),
        "answer_file_path": doc.get("answer_file_path"),
        "source_answer_file": doc.get("source_answer_file"),
        "original_filename": fm.get("original_filename"),
        "phase1_source_path": fm.get("phase1_source_path"),
        "answer_file_selection_reason": fm.get("answer_file_selection_reason"),
        "file_extension": fm.get("extension"),
        "file_sha256": fm.get("sha256"),
        "file_size_bytes": fm.get("size_bytes"),
        "page_count": fm.get("page_count"),
        "file_last_modified": _ts(fm.get("last_modified")),
        "parsed_schema_version": doc.get("parsed_schema_version"),
        "parser_used": doc.get("parser_used"),
        "run_id": doc.get("run_id"),
        "backend": doc.get("backend"),
        "models_used": json.dumps(doc.get("models_used") or {}, ensure_ascii=False),
        "page_routing": json.dumps(doc.get("page_routing") or [], ensure_ascii=False),
        "embedding_unit": doc.get("embedding_unit"),
        "parsed_at": _ts(doc.get("parsed_at")),
        # WARNINGS, stored not gated
        "needs_review": bool(doc.get("needs_review")),
        "extraction_flags": doc.get("extraction_flags") or [],
        "annexures_referenced": doc.get("annexures_referenced") or [],
        "annexure_content_present": doc.get("annexure_content_present"),
        "tables_index": doc.get("tables_index") or [],
        "raw_json": json.dumps(doc, ensure_ascii=False),
    }


_DIARY_COLS = list(_diary_row({"file_metadata": {}}, "").keys())


def _upsert(cur, table, pk, rows, cols):
    """Batch UPSERT on the deterministic primary key. Re-runs UPDATE, never duplicate."""
    if not rows:
        return 0
    collist = ", ".join(cols)
    ph = ", ".join(["%s"] * len(cols))
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != pk)
    sql = (f"INSERT INTO {table} ({collist}) VALUES ({ph}) "
           f"ON CONFLICT ({pk}) DO UPDATE SET {updates}")
    cur.executemany(sql, [[r[c] for c in cols] for r in rows])
    return len(rows)


def load_document(conn, doc, rel_path, force=False):
    """
    Load ONE parsed.json inside ONE transaction. Returns (counts, warnings).
    Raises on a hard DB error, which the caller records as 'failed' -- that document
    rolls back alone and the run continues.
    """
    warnings = validate(doc)
    qid = doc.get("question_id")
    key = doc_key(doc, rel_path)     # '<session>/<house>/<question_id>' — globally unique
    counts = {"sub_questions": 0, "answer_groups": 0, "answer_tables": 0,
              "answer_table_rows": 0, "annexures": 0, "diary_level_tables": 0}

    groups = doc.get("answer_groups") or []
    gids = {g.get("answer_group_id") for g in groups}

    with conn.transaction():
        with conn.cursor() as cur:
            # 1. diary
            _upsert(cur, "diaries", "doc_key",
                    [_diary_row(doc, rel_path)], _DIARY_COLS)

            # 2. answer_groups (before sub_questions: FK target)
            g_rows = [{
                "answer_group_id": _ns(key, g["answer_group_id"]),
                "answer_group_local": g["answer_group_id"],
                "doc_key": key,
                "question_id": qid,
                "answers_parts": g.get("answers_parts") or [],
                "answer_text": g.get("answer_text"),
                "answer_type": (g.get("answer_type")
                                if g.get("answer_type") in VALID_ANSWER_TYPES else None),
                "answer_language": g.get("answer_language"),
                "answer_is_table": g.get("answer_is_table"),
                "answer_blocks": json.dumps(g.get("answer_blocks") or [], ensure_ascii=False),
                "annexure_refs": g.get("annexure_refs") or [],
                "confidence": g.get("confidence"),
            } for g in groups if g.get("answer_group_id")]
            counts["answer_groups"] = _upsert(
                cur, "answer_groups", "answer_group_id", g_rows,
                list(g_rows[0]) if g_rows else [])

            # 3. sub_questions. A dangling answer_group_id cannot satisfy the FK. Policy
            #    is that EVERYTHING loads, so rather than drop the sub-question we attach
            #    it to a synthesised placeholder group; the warning says to fix Phase 2.
            sq_rows = []
            for sq in doc.get("sub_questions") or []:
                if not sq.get("sub_question_id"):
                    continue
                local_g = sq.get("answer_group_id")
                if local_g not in gids:
                    local_g = f"{qid}_gorphan"
                    _upsert(cur, "answer_groups", "answer_group_id", [{
                        "answer_group_id": _ns(key, local_g),
                        "answer_group_local": local_g,
                        "doc_key": key, "question_id": qid,
                        "answers_parts": [], "answer_text": None, "answer_type": None,
                        "answer_language": None, "answer_is_table": False,
                        "answer_blocks": "[]", "annexure_refs": [], "confidence": "low",
                    }], ["answer_group_id", "answer_group_local", "doc_key", "question_id",
                         "answers_parts", "answer_text", "answer_type", "answer_language",
                         "answer_is_table", "answer_blocks", "annexure_refs", "confidence"])
                    gids.add(local_g)
                sq_rows.append({
                    "sub_question_id": _ns(key, sq["sub_question_id"]),
                    "sub_question_local": sq["sub_question_id"],
                    "doc_key": key,
                    "question_id": qid,
                    "answer_group_id": _ns(key, local_g),
                    "part_label": sq.get("part_label"),
                    "question_text": sq.get("question_text") or "",
                    "question_language": sq.get("question_language"),
                    "annexure_refs": sq.get("annexure_refs") or [],
                })
            counts["sub_questions"] = _upsert(
                cur, "sub_questions", "sub_question_id", sq_rows,
                list(sq_rows[0]) if sq_rows else [])

            # --force: invalidate this document's vectors so the embedder redoes them
            if force:
                cur.execute(
                    "UPDATE sub_questions SET embedding = NULL, embedding_model = NULL, "
                    "embedding_created_at = NULL WHERE doc_key = %s", (key,))

            # 4. tables (nested INSIDE their answer group) + rows
            t_rows, r_rows = [], []
            for g in groups:
                for t in (g.get("tables") or []):
                    t_rows.append({
                        "table_id": _ns(key, t["table_id"]),
                        "table_local": t["table_id"],
                        "answer_group_id": _ns(key, g["answer_group_id"]),
                        "doc_key": key,
                        "question_id": qid,
                        "caption": t.get("caption"),
                        "table_role": t.get("table_role"),
                        "answer_is_table": t.get("answer_is_table"),
                        "columns": json.dumps(t.get("columns") or [], ensure_ascii=False),
                        "stitched_across_pages": t.get("stitched_across_pages"),
                        "extraction_confidence": t.get("extraction_confidence"),
                    })
                    for i, r in enumerate(t.get("rows") or [], start=1):
                        r_rows.append({
                            "row_id": _ns(key, r["row_id"]),
                            "row_local": r["row_id"],
                            "table_id": _ns(key, t["table_id"]),
                            "row_index": i,
                            "cells": json.dumps(r.get("cells") or {}, ensure_ascii=False),
                            "row_language": r.get("row_language"),
                            "nl_rendering": r.get("nl_rendering"),
                            "entities": r.get("entities") or [],
                        })
            counts["answer_tables"] = _upsert(
                cur, "answer_tables", "table_id", t_rows, list(t_rows[0]) if t_rows else [])
            counts["answer_table_rows"] = _upsert(
                cur, "answer_table_rows", "row_id", r_rows, list(r_rows[0]) if r_rows else [])

            # 5. diary-level tables (rare)
            dt_rows, dr_rows = [], []
            for t in (doc.get("diary_level_tables") or []):
                dt_rows.append({
                    "table_id": _ns(key, t["table_id"]), "table_local": t["table_id"],
                    "doc_key": key, "question_id": qid,
                    "caption": t.get("caption"), "table_role": t.get("table_role"),
                    "answer_is_table": t.get("answer_is_table"),
                    "columns": json.dumps(t.get("columns") or [], ensure_ascii=False),
                    "stitched_across_pages": t.get("stitched_across_pages"),
                    "extraction_confidence": t.get("extraction_confidence"),
                })
                for i, r in enumerate(t.get("rows") or [], start=1):
                    dr_rows.append({
                        "row_id": _ns(key, r["row_id"]), "row_local": r["row_id"],
                        "table_id": _ns(key, t["table_id"]), "row_index": i,
                        "cells": json.dumps(r.get("cells") or {}, ensure_ascii=False),
                        "row_language": r.get("row_language"),
                        "nl_rendering": r.get("nl_rendering"),
                        "entities": r.get("entities") or [],
                    })
            counts["diary_level_tables"] = _upsert(
                cur, "diary_level_tables", "table_id", dt_rows,
                list(dt_rows[0]) if dt_rows else [])
            _upsert(cur, "diary_level_table_rows", "row_id", dr_rows,
                    list(dr_rows[0]) if dr_rows else [])

            # 6. annexures — PK synthesised deterministically (the JSON has no id)
            a_rows = []
            for a in (doc.get("annexures") or []):
                label = a.get("ref_label")
                if not label:
                    continue
                a_rows.append({
                    "annexure_id": _ns(key, label),
                    "doc_key": key,
                    "question_id": qid,
                    "ref_label": label,
                    "referenced_in_parts": a.get("referenced_in_parts") or [],
                    "file_path": a.get("file_path"),
                    "file_present": a.get("file_present"),
                    "match_confidence": a.get("match_confidence"),
                })
            counts["annexures"] = _upsert(
                cur, "annexures", "annexure_id", a_rows, list(a_rows[0]) if a_rows else [])

    return counts, warnings


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _report(cfg, rows, summary):
    os.makedirs(cfg.reports_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jpath = os.path.join(cfg.reports_dir, f"load_report_{stamp}.json")
    cpath = os.path.join(cfg.reports_dir, f"load_report_{stamp}.csv")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "config": cfg.describe(), "documents": rows},
                  fh, ensure_ascii=False, indent=2)
    with open(cpath, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "question_path", "question_id", "status", "needs_review",
            "sub_questions", "answer_groups", "answer_tables", "answer_table_rows",
            "annexures", "warnings", "error"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (";".join(r[k]) if isinstance(r.get(k), list) else r.get(k))
                        for k in w.fieldnames})
    return jpath, cpath


def main(argv=None):
    ap = argparse.ArgumentParser(description="Load Phase-2 parsed.json into Postgres")
    ap.add_argument("--dry-run", action="store_true", help="validate + report, no writes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", action="append", default=None,
                    help="substring of the question path (repeatable), e.g. --only 8773")
    ap.add_argument("--force", action="store_true",
                    help="re-load and clear embeddings so they are regenerated")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Settings()
    errs = cfg.validate(need_db=not args.dry_run, need_embed=False)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    files = sorted(glob.glob(os.path.join(cfg.organized_root, "*", "*", "*", "parsed.json")))
    if args.only:
        files = [f for f in files if any(o in f.replace(os.sep, "/") for o in args.only)]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print("no parsed.json matched")
        return 1

    print(f"Phase 3 loader | {len(files)} document(s) | "
          f"{'DRY RUN (no writes)' if args.dry_run else 'db=' + cfg.describe()['db_dsn']}"
          f"{' | FORCE' if args.force else ''}")

    rows, totals = [], {"sub_questions": 0, "answer_groups": 0, "answer_tables": 0,
                        "answer_table_rows": 0, "annexures": 0, "diary_level_tables": 0}
    loaded = failed = n_review = 0
    t0 = time.time()

    conn_ctx = connect(cfg) if not args.dry_run else _NullConn()
    with conn_ctx as conn:
        for f in files:
            rel = os.path.dirname(f).replace(cfg.organized_root + os.sep, "") \
                                    .replace(os.sep, "/")
            rec = {"question_path": rel, "question_id": None, "status": "ok",
                   "needs_review": False, "warnings": [], "error": None,
                   "sub_questions": 0, "answer_groups": 0, "answer_tables": 0,
                   "answer_table_rows": 0, "annexures": 0}
            try:
                with open(f, encoding="utf-8") as fh:
                    doc = json.load(fh)
            except Exception as e:      # unreadable JSON = a genuinely hard failure
                rec["status"] = "failed"
                rec["error"] = f"unreadable json: {type(e).__name__}: {e}"
                failed += 1
                rows.append(rec)
                continue

            rec["question_id"] = doc.get("question_id")
            rec["needs_review"] = bool(doc.get("needs_review"))
            if rec["needs_review"]:
                n_review += 1

            if args.dry_run:
                rec["warnings"] = validate(doc)
                rec["sub_questions"] = len(doc.get("sub_questions") or [])
                rec["answer_groups"] = len(doc.get("answer_groups") or [])
                tabs = [t for g in (doc.get("answer_groups") or [])
                        for t in (g.get("tables") or [])]
                rec["answer_tables"] = len(tabs)
                rec["answer_table_rows"] = sum(len(t.get("rows") or []) for t in tabs)
                rec["annexures"] = len(doc.get("annexures") or [])
                for k in totals:
                    totals[k] += rec.get(k, 0)
                loaded += 1
                rows.append(rec)
                continue

            try:
                counts, warns = load_document(conn, doc, rel, force=args.force)
                rec.update(counts)
                rec["warnings"] = warns
                for k, v in counts.items():
                    totals[k] += v
                loaded += 1
            except Exception as e:      # HARD db error -> this doc only
                rec["status"] = "failed"
                rec["error"] = f"{type(e).__name__}: {e}"
                failed += 1
            rows.append(rec)

    dt = time.time() - t0
    n_warn = sum(1 for r in rows if r["warnings"])
    summary = {
        "documents_seen": len(files),
        "loaded": loaded,
        "failed": failed,
        "documents_with_warnings": n_warn,
        "needs_review_count": n_review,   # INFORMATIONAL — these all loaded
        "dry_run": args.dry_run,
        "elapsed_s": round(dt, 1),
        **totals,
    }
    jpath, cpath = _report(cfg, rows, summary)

    print("\n" + "=" * 62)
    print("LOAD SUMMARY" + ("  (DRY RUN — nothing written)" if args.dry_run else ""))
    print("=" * 62)
    for k in ("documents_seen", "loaded", "failed", "documents_with_warnings"):
        print(f"  {k:26} {summary[k]}")
    print(f"  {'needs_review (informational)':26} {n_review}   <- all of these LOADED")
    print("  " + "-" * 44)
    for k in ("sub_questions", "answer_groups", "answer_tables",
              "answer_table_rows", "annexures", "diary_level_tables"):
        print(f"  {k:26} {summary[k]}")
    if failed:
        print("\n  FAILED (hard errors only):")
        for r in rows:
            if r["status"] == "failed":
                print(f"    {r['question_path']}: {r['error']}")
    if n_warn:
        print(f"\n  warnings in {n_warn} document(s) — see the report (they still loaded)")
    print(f"\n  report: {jpath}")
    print(f"          {cpath}")
    return 0 if failed == 0 else 2


class _NullConn:
    """Stand-in connection for --dry-run so no DB is needed to validate."""
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
