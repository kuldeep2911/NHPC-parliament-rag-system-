"""
Phase-2 orchestrator + CLI.

Walks organized/ question folders, parses each answer file into parsed.json, and
writes a run report. Idempotent (skips folders already having parsed.json unless
--force), resumable (a crash on one document is caught and logged; the run
continues), read-only on all Phase-1 inputs.

    python -m phase2.pipeline [--organized organized] [--limit N] [--force]
                              [--dry-run] [--only SUBPATH] [--backend NAME]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import traceback
from collections import Counter

from .config import load_config, load_dotenv
from .ir import detect_language
from .layout import analyze_layout
from .reader import read_document, HAS_DOCLING
from .extract import extract_qa, validate_pairs, build_document_fields
from .providers import get_parser, get_llm
from .trace import RunTracer, new_run_id


SCHEMA_VERSION = "2.1"   # bump when parsed.json schema changes (for DB migrations)


def _utf8_stdout():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _now_iso():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _file_metadata(path, root, meta):
    """
    File + provenance metadata for DB storage. Captures the reply file's on-disk
    facts (mtime, size, sha256, ext, page count) plus the Phase-1 source path and
    selection provenance. All paths RELATIVE to organized/ root. None if no file.
    """
    import datetime
    md = {
        "relative_path": None,
        "original_filename": meta.get("answer_file_selected"),
        "phase1_source_path": meta.get("source_path"),
        "answer_file_selection_reason": meta.get("answer_file_selection_reason"),
        "extension": None,
        "size_bytes": None,
        "last_modified": None,
        "sha256": None,
        "page_count": None,
    }
    if not path or not os.path.isfile(path):
        return md
    try:
        st = os.stat(path)
        md["relative_path"] = os.path.relpath(path, root).replace(os.sep, "/")
        md["extension"] = os.path.splitext(path)[1].lower().lstrip(".")
        md["size_bytes"] = st.st_size
        md["last_modified"] = datetime.datetime.fromtimestamp(
            st.st_mtime, datetime.timezone.utc).isoformat()
    except OSError:
        return md
    # sha256 (stream, so large files don't blow memory)
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        md["sha256"] = h.hexdigest()
    except OSError:
        pass
    # page count for PDFs (cheap, best-effort)
    if md["extension"] == "pdf":
        try:
            import pdfplumber
            import warnings
            warnings.filterwarnings("ignore")
            with pdfplumber.open(path) as pdf:
                md["page_count"] = len(pdf.pages)
        except Exception:
            pass
    return md


def find_question_dirs(root):
    for dirpath, dirnames, filenames in os.walk(root):
        if "_reports" in dirpath.split(os.sep):
            continue
        if "metadata.json" in filenames:
            yield dirpath
            dirnames[:] = [d for d in dirnames
                           if d not in ("answer_all_versions",
                                        "answer_latest_candidates",
                                        "question_other_versions")]


def load_metadata(qdir):
    with open(os.path.join(qdir, "metadata.json"), encoding="utf-8") as fh:
        return json.load(fh)


def pick_answer_file(qdir, meta):
    """Return (path, flags). Prefer answer_latest.*; fall back per metadata."""
    flags = []
    for f in sorted(os.listdir(qdir)):
        if f.lower().startswith("answer_latest.") and os.path.isfile(os.path.join(qdir, f)):
            return os.path.join(qdir, f), flags
    # fallback: answer_latest_candidates/ (ambiguous)
    cand_dir = os.path.join(qdir, "answer_latest_candidates")
    if os.path.isdir(cand_dir):
        cands = [c for c in sorted(os.listdir(cand_dir)) if not c.startswith("~$")]
        if cands:
            flags.append("ambiguous_latest_reply")
            return os.path.join(cand_dir, cands[0]), flags
    # fallback: answer_all_versions/
    allv = os.path.join(qdir, "answer_all_versions")
    if os.path.isdir(allv):
        for r, _d, fs in os.walk(allv):
            for f in sorted(fs):
                if f.startswith("~$"):
                    continue
                if os.path.splitext(f)[1].lower() in (".pdf", ".docx", ".doc", ".xlsx", ".txt", ".rtf"):
                    flags.append("answer_from_all_versions_fallback")
                    return os.path.join(r, f), flags
    flags.append("no_answer_file")
    return None, flags


def parse_one(qdir, root, cfg, parser, llm, run_tracer=None):
    """Parse a single question folder. Returns (record_dict, parsed_obj_or_None).

    `parser` = parse/OCR/table provider (Nemotron NIMs or None for Docling).
    `llm`    = LLM provider for the extraction pass (Ollama / deterministic).
    """
    meta = load_metadata(qdir)
    rel = os.path.relpath(qdir, root)
    flags = []

    # per-document tracer (carries run_id + doc_run_id join keys)
    backend_label = f"parser={getattr(parser,'name','docling')},llm={llm.name}"
    dtracer = run_tracer.for_doc(rel, meta.get("question_id"), backend_label) \
        if run_tracer else None
    run_id = run_tracer.run_id if run_tracer else None

    # carry forward Phase-1 review context
    if meta.get("status") == "needs_review":
        flags.append("phase1_flagged")

    ans_path, af_flags = pick_answer_file(qdir, meta)
    flags += af_flags

    parsed = {
        "question_id": meta.get("question_id"),
        "session": meta.get("session"),
        "house": meta.get("house"),
        "state": meta.get("state"),
        "source_answer_file": os.path.basename(ans_path) if ans_path else None,
        # File + processing metadata for DB storage (mtime, size, hash, ext, etc.).
        "file_metadata": _file_metadata(ans_path, root, meta),
        "parsed_at": _now_iso(),
        "parsed_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "backend": backend_label,
        "models_used": {},
        "parser_used": None,
        "document_language": None,
        "layout_structure": None,
        "layout_case_detected": None,
        "qa_table": False,
        "page_routing": [],
        "embedding_unit": "sub_question.question_text",
        "answer_file_path": None,
        "diary_numbers": [],
        "starred": None,
        "subject": None,
        "reply_format": "unknown",
        "is_nhpc_relevant": None,
        "sub_questions": [],
        "answer_groups": [],
        "diary_level_tables": [],
        "annexures_referenced": [],
        "annexures": [],
        "annexure_content_present": False,
        "tables_index": [],
        "extraction_flags": [],
        "needs_review": False,
    }

    if ans_path is None:
        parsed["extraction_flags"] = _dedup(flags)
        parsed["needs_review"] = True
        return _record(rel, meta, parsed, status="review", reason="no_answer_file"), parsed

    doc = read_document(ans_path, cfg, provider=parser, tracer=dtracer)
    flags += doc.flags
    parsed["parser_used"] = doc.parser_used
    parsed["document_language"] = doc.language()
    parsed["page_routing"] = doc.page_routing
    parsed["models_used"] = dict(doc.models_used)

    if doc.parser_used == "error" or (not doc.blocks and not doc.tables):
        flags.append("empty_or_unreadable")
        parsed["extraction_flags"] = _dedup(flags)
        parsed["needs_review"] = True
        return _record(rel, meta, parsed, status="review",
                       reason="empty_or_unreadable"), parsed

    layout = analyze_layout(doc)
    parsed["layout_structure"] = layout["structure"]
    parsed["layout_case_detected"] = layout["case"]
    parsed["qa_table"] = layout["structure"] == "qa_table"

    pairs, tindex, xflags = extract_qa(doc, layout, meta, cfg, llm, tracer=dtracer)
    flags += xflags

    # record the LLM model actually used for extraction (if a real model ran)
    if "model_extracted" in xflags and hasattr(llm, "model_for"):
        parsed["models_used"]["llm"] = llm.model_for("llm")

    parsed["tables_index"] = tindex

    # diary-as-container document fields (sub_questions, diary_numbers, reply_format,
    # annexures, answer_type, identity separation). `sub_questions` is the schema;
    # the legacy `pairs` array is NOT written (it duplicated sub_questions and
    # repeated tables). `pairs` is still used in-memory for validation + to build
    # sub_questions. answer_file_path + annexure paths are RELATIVE to organized/.
    ans_rel = os.path.relpath(ans_path, root).replace(os.sep, "/")
    doc_fields, flags = build_document_fields(
        doc, layout, meta, pairs, flags,
        qdir=qdir, organized_root=root, answer_file_path=ans_rel)
    parsed.update(doc_fields)

    # validation (against the in-memory pairs; pairs are not written to output)
    verrs = validate_pairs({"pairs": [p.to_dict() for p in pairs]})
    if verrs:
        flags.append("schema_validation_failed")

    # review routing — answer_groups confidence
    sqs = parsed.get("sub_questions", [])
    grps = parsed.get("answer_groups", [])
    low_conf = any(g.get("confidence") == "low" for g in grps)
    if low_conf:
        flags.append("low_confidence_pair")
    if not sqs:
        flags.append("no_pairs_extracted")

    review_triggers = {
        "no_answer_file", "empty_or_unreadable", "schema_validation_failed",
        "low_confidence_pair", "no_pairs_extracted", "format_converted",
        "ocr_used", "phase1_flagged", "qa_table_columns_inferred",
        "table_alignment_uncertain",
        "ambiguous_latest_reply", "answer_from_all_versions_fallback",
        "pdfplumber_fallback", "ocr_unavailable", "format_conversion_unavailable",
        "read_error", "email_wrapper_reply_may_be_attachment",
        "qa_table_row_stitched", "table_wrapped_rows_stitched",
        "empty_answer_after_recovery",
        # new (Change 4/5): annexure-only content, covering letters, no diary found
        "answer_in_annexure_not_present", "covering_letter_format",
        "no_diary_number_found", "annexure_match_ambiguous",
        # answer_groups safeguards
        "table_group_uncertain", "qa_count_mismatch", "group_link_broken",
        "llm_crosscheck_disagree", "annexure_ref_unresolved",
    }
    flags = _dedup(flags)
    needs_review = bool(set(flags) & review_triggers) or bool(verrs)
    parsed["extraction_flags"] = flags
    parsed["needs_review"] = needs_review

    status = "review" if needs_review else "ok"
    reason = ";".join(f for f in flags if f in review_triggers) or ("" if status == "ok" else "review")
    return _record(rel, meta, parsed, status=status, reason=reason), parsed


def _record(rel, meta, parsed, status, reason):
    return {
        "path": rel,
        "question_id": meta.get("question_id"),
        "session": meta.get("session"),
        "house": meta.get("house"),
        "parser_used": parsed.get("parser_used"),
        "document_language": parsed.get("document_language"),
        "layout_structure": parsed.get("layout_structure"),
        "layout_case": parsed.get("layout_case_detected"),
        "n_pairs": len(parsed.get("sub_questions", [])),
        "n_tables": len(parsed.get("tables_index", [])),
        "status": status,
        "reason": reason,
        "flags": ";".join(parsed.get("extraction_flags", [])),
    }


def _dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def run(cfg, only=None):
    _utf8_stdout()
    root = os.path.abspath(cfg.organized_root)

    # Fail fast on misconfiguration (missing NVIDIA key, no Ollama URL, ...).
    cfg_errs = cfg.validate()
    if cfg_errs:
        for e in cfg_errs:
            print(f"[CONFIG ERROR] {e}", file=sys.stderr)
        raise SystemExit(2)

    parser = get_parser(cfg)      # Nemotron NIMs (or None for Docling)
    llm = get_llm(cfg)            # Ollama llama3.2:3b (or deterministic)
    pb, lb = cfg.resolve_backends()
    backend_label = f"parser={pb},llm={lb}"

    # trace layer (Postgres if DSN set, else JSONL; disabled if trace_enabled=False)
    run_id = new_run_id()
    tracer = RunTracer(cfg, run_id, root)
    if not cfg.dry_run:
        tracer.start(backend_label)

    print(f"Phase 2 parsing | parser={pb} llm={llm.name} | docling={'yes' if HAS_DOCLING else 'no'}"
          f" | run_id={run_id} | trace={tracer.status}"
          f"{' | DRY RUN' if cfg.dry_run else ''}")

    records = []
    errors = []
    processed = skipped = 0

    qdirs = list(find_question_dirs(root))
    if only:
        qdirs = [q for q in qdirs if only.replace("/", os.sep) in q]
    qdirs.sort()

    for qdir in qdirs:
        if cfg.limit and processed >= cfg.limit:
            break
        parsed_path = os.path.join(qdir, cfg.parsed_filename)
        if os.path.exists(parsed_path) and not cfg.force:
            skipped += 1
            continue
        rel = os.path.relpath(qdir, root)
        try:
            rec, parsed = parse_one(qdir, root, cfg, parser, llm,
                                    run_tracer=tracer if not cfg.dry_run else None)
            if not cfg.dry_run:
                _atomic_write_json(parsed_path, parsed)
            records.append(rec)
            processed += 1
            tag = rec["status"].upper()
            print(f"[{tag:6}] {rel}  ({rec['parser_used']}, {rec['layout_structure']}, "
                  f"{rec['n_pairs']} pairs)" + (f"  <{rec['reason']}>" if rec['reason'] else ""))
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            errors.append({"path": rel, "error": f"{type(e).__name__}: {e}",
                           "trace": tb})
            print(f"[ERROR ] {rel}  {type(e).__name__}: {e}")
            processed += 1  # counted as processed-but-errored; run continues

    _write_reports(cfg, root, records, errors, skipped)
    if not cfg.dry_run:
        tracer.finish({"processed": len(records), "errored": len(errors),
                       "skipped": skipped})
    _print_summary(records, errors, skipped, root, cfg, run_id, tracer.status)


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _write_reports(cfg, root, records, errors, skipped):
    if cfg.dry_run:
        return
    rep_dir = os.path.join(root, cfg.reports_subdir)
    os.makedirs(rep_dir, exist_ok=True)

    by_status = Counter(r["status"] for r in records)
    by_parser = Counter(r["parser_used"] for r in records)
    by_lang = Counter(r["document_language"] for r in records)
    by_struct = Counter(r["layout_structure"] for r in records)
    by_case = Counter(r["layout_case"] for r in records)
    reason_counter = Counter()
    for r in records:
        for fl in (r["flags"] or "").split(";"):
            if fl:
                reason_counter[fl] += 1

    summary = {
        "total_processed": len(records),
        "skipped_already_parsed": skipped,
        "ok": by_status.get("ok", 0),
        "review": by_status.get("review", 0),
        "errored": len(errors),
        "by_parser": dict(by_parser),
        "by_language": dict(by_lang),
        "by_layout_structure": dict(by_struct),
        "by_layout_case": dict(by_case),
        "flag_counts": dict(reason_counter.most_common()),
        "qa_table_replies": by_struct.get("qa_table", 0),
        "table_is_answer_replies": reason_counter.get("answer_is_table", 0),
        "ocr_docs": reason_counter.get("ocr_used", 0),
        "visual_docs": reason_counter.get("visual_used", 0),
        "mixed_page_type_docs": reason_counter.get("mixed_page_types", 0),
        "format_conversions": reason_counter.get("format_converted", 0),
        "backend": cfg.backend,
    }

    with open(os.path.join(rep_dir, "parse_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "records": records, "errors": errors},
                  fh, ensure_ascii=False, indent=2)

    if records:
        with open(os.path.join(rep_dir, "parse_report.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    review_rows = [r for r in records if r["status"] == "review"]
    with open(os.path.join(rep_dir, "review_queue.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "question_id", "reason", "flags"])
        for r in review_rows:
            w.writerow([r["path"], r["question_id"], r["reason"], r["flags"]])
        for e in errors:
            w.writerow([e["path"], "", "error", e["error"]])


def _print_summary(records, errors, skipped, root, cfg, run_id=None, trace_status=None):
    line = "=" * 62
    by_status = Counter(r["status"] for r in records)
    print(line)
    print("Phase 2 — parsing & Q&A extraction" + ("  [DRY RUN]" if cfg.dry_run else ""))
    print(line)
    print(f"Backend            : {cfg.backend}")
    if run_id:
        print(f"Run id             : {run_id}")
    if trace_status:
        print(f"Trace sink         : {trace_status}")
    print(f"Processed this run : {len(records)}")
    print(f"  ok               : {by_status.get('ok', 0)}")
    print(f"  needs review     : {by_status.get('review', 0)}")
    print(f"  errored          : {len(errors)}")
    print(f"Skipped (parsed)   : {skipped}")
    struct = Counter(r["layout_structure"] for r in records)
    print(f"Layout: {dict(struct)}")
    parser = Counter(r["parser_used"] for r in records)
    print(f"Parsers: {dict(parser)}")
    if not cfg.dry_run:
        rep = os.path.join(root, cfg.reports_subdir)
        print(f"Report: {os.path.join(rep, 'parse_report.json')}")
        print(f"        {os.path.join(rep, 'parse_report.csv')}")
        print(f"        {os.path.join(rep, 'review_queue.csv')}")
    print(line)


def main(argv=None):
    load_dotenv()  # pick up NVIDIA_API_KEY etc. from a project .env (never logged)
    ap = argparse.ArgumentParser(description="NHPC Phase-2 parsing pipeline")
    ap.add_argument("--organized", default="organized")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="only process folders under this subpath")
    ap.add_argument("--backend", default=None, choices=[None, "local", "nvidia"],
                    help="legacy single switch (overrides NHPC_BACKEND env)")
    ap.add_argument("--parser-backend", default=None, choices=[None, "nemotron", "docling"],
                    help="document parser backend (overrides NHPC_PARSER_BACKEND)")
    ap.add_argument("--llm-backend", default=None,
                    choices=[None, "ollama", "groq", "deterministic"],
                    help="LLM extraction backend (overrides NHPC_LLM_BACKEND)")
    ap.add_argument("--no-docling", action="store_true", help="disable Docling (use fallbacks)")
    ap.add_argument("--no-trace", action="store_true", help="disable the trace layer")
    ap.add_argument("--llm-crosscheck", action="store_true",
                    help="run the LLM as a second-opinion check on every prose file")
    ap.add_argument("--llm-grouping", action="store_true",
                    help="LLM decides question<->answer grouping (best with a 70B model)")
    args = ap.parse_args(argv)

    cfg = load_config(
        organized_root=args.organized, limit=args.limit, force=args.force,
        dry_run=args.dry_run, backend=args.backend,
        parser_backend=args.parser_backend, llm_backend=args.llm_backend,
        prefer_docling=False if args.no_docling else None,
        trace_enabled=False if args.no_trace else None,
        llm_crosscheck=True if args.llm_crosscheck else None,
        llm_grouping=True if args.llm_grouping else None,
    )
    run(cfg, only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
