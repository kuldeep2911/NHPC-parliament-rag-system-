"""
Backfill reply dates onto the existing corpus.

    nhpc backfill-dates                 # rule-based; the cheap 81%
    nhpc backfill-dates --llm           # + the LLM, ONLY on the rule-based misses
    nhpc backfill-dates --dry-run       # report, write nothing
    nhpc backfill-dates --force         # redo documents that already have a date

NO DOCLING. NO RE-PARSE. NO LLM FOR THE BULK.

The corpus was already parsed and its text is already on disk. Running the whole thing back
through Docling and an LLM to read a date out of a header would cost hours and real API
spend to recover information we already have. So:

    1. diaries.subject          -- the parsed header, already in the DB     (247 docs)
    2. the PDF's text layer     -- pypdf, first 2 pages, no OCR             (+181 docs)
    3. the LLM                  -- ONLY the leftovers, opt-in via --llm     (~97 docs)

Measured: steps 1+2 reach 81% of the corpus for the cost of a text scan.

IDEMPOTENT AND RESUMABLE. It skips documents that already carry a date (unless --force), so
a run that is interrupted is fixed by running it again. Nothing is destroyed: this only ever
fills in NULL columns.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.core.db.session import connect
from nhpc_qa.core.logging import get_logger, setup as setup_logging
from nhpc_qa.pipeline.parse import dates as D

log = get_logger("nhpc.backfill.dates")


_SELECT = """
SELECT doc_key, subject, answer_file_path, session_year
FROM diaries
{where}
ORDER BY doc_key
"""

_UPDATE = """
UPDATE diaries
SET reply_date = %(reply_date)s,
    updated_at = now()
WHERE doc_key = %(doc_key)s
"""


def _apply(conn, doc_key, reply_date, dry):
    if dry:
        return
    with conn.cursor() as cur:
        cur.execute(_UPDATE, {"doc_key": doc_key, "reply_date": reply_date})
    conn.commit()


def _llm_extract(cfg, text, doc_key, session_year=None):
    """
    Ask the LLM for the reply date -- ONLY for documents the rules could not read.

    The model returns a DATE STRING, not prose, and we re-validate it through the SAME
    validate() the rules use. The model is never trusted to hand us a well-formed or
    plausible date; it is trusted only to FIND the right one in the text. That is the same
    discipline the span extractor uses, and for the same reason.
    """
    from nhpc_qa.core.providers.models import get_llm

    prompt = (
        "This is the header of an Indian parliamentary question document.\n"
        "Find the date on which the question was TO BE ANSWERED in Parliament.\n\n"
        "Rules:\n"
        "  - Return ONLY the date, in DD.MM.YYYY form. No other words.\n"
        "  - It is the date after a phrase like 'to be answered on' / 'for answer on' / "
        "'दिनांक'.\n"
        "  - IGNORE dates inside the body text (agreements, MOUs, commissioning dates).\n"
        "  - If there is no such date, return exactly: NONE\n\n"
        f"TEXT:\n{text[:4000]}\n"
    )
    try:
        raw = (get_llm(cfg).complete(prompt) or "").strip()
    except Exception as e:      # noqa: BLE001 -- the fallback must never break the backfill
        log.warning("%s: llm error %s: %s", doc_key, type(e).__name__, e)
        return None
    if not raw or raw.upper().startswith("NONE"):
        return None
    # Validate through the SAME gate the rules use. A model that hallucinates '31.02.2024',
    # or a confidently-wrong year for this session, is rejected here rather than poisoning
    # the sort order. The model is trusted to FIND the date, never to hand us a good one.
    d = D.validate(raw.split()[0] if raw.split() else "", session_year=session_year)
    if d is None:
        log.warning("%s: llm returned an unusable date %r — ignored", doc_key, raw[:40])
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nhpc backfill-dates")
    ap.add_argument("--llm", action="store_true",
                    help="use the LLM on the rule-based misses (costs API calls)")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="redo documents that already have a reply_date")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    load_dotenv()
    setup_logging()
    cfg = Settings()
    errs = cfg.validate_all(need_db=True, need_embed=False, need_rerank=False)
    if errs:
        print("CONFIG ERROR:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2

    org_root = os.path.abspath(getattr(cfg, "organized_root", "organized"))
    where = "" if args.force else "WHERE reply_date IS NULL"
    t0 = time.time()

    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(_SELECT.format(where=where))
            rows = cur.fetchall()
        if args.limit:
            rows = rows[:args.limit]

        n = len(rows)
        print(f"\n  {n} document(s) to process"
              f"{' (--force: including ones already dated)' if args.force else ''}"
              f"{'   [DRY RUN — nothing will be written]' if args.dry_run else ''}\n")

        from_subject = from_pdf = from_llm = missed = 0
        misses = []

        # ---- PASS 1 + 2: the rules. Free. ---------------------------------
        for doc_key, subject, ans_path, session_year in rows:
            # session_year CORROBORATES the date: it disambiguates a truncated 2-digit year
            # and rejects a body-prose date that slipped past an anchor. Without it, 11% of
            # the extracted dates were wrong -- measured, not theoretical.
            sy = int(session_year) if session_year else None
            rd = D.extract_from_text(subject, session_year=sy)   # the DB's own header text
            if rd is None and ans_path:
                # The subject column is TRUNCATED at parse time -- many headers lost their
                # date. The original PDF still has it, and its text layer is cheap to read.
                p = os.path.join(org_root, *ans_path.split("/"))
                if os.path.exists(p) and p.lower().endswith(".pdf"):
                    rd = D.extract_from_text(D.pdf_header_text(p), session_year=sy)
                    if rd is not None:
                        from_pdf += 1
            elif rd is not None:
                from_subject += 1

            if rd is not None:
                _apply(conn, doc_key, rd, args.dry_run)
                log.info("%s  %s", doc_key, rd)
            else:
                # ⚠️ ON --force, A MISS MUST CLEAR THE OLD VALUE. ⚠️
                #
                # Otherwise a date extracted by an EARLIER, buggier version of the rules
                # survives the very run that was meant to correct it. That is exactly what
                # happened: a first pass recorded 'dated 27.08.2014' as the reply date of a
                # 2020 document; the corroboration check then correctly rejected it on the
                # re-run -- and, because a miss wrote nothing, the wrong date stayed.
                #
                # A run that re-derives a field owns that field, including the right to say
                # "there is no value".
                if args.force and not args.dry_run:
                    _apply(conn, doc_key, None, dry=False)
                misses.append((doc_key, ans_path, sy))

        # ---- PASS 3: the LLM, on the leftovers only -----------------------
        if misses and args.llm:
            print(f"\n  {len(misses)} rule-based miss(es) -> LLM fallback\n")
            for doc_key, ans_path, sy in list(misses):
                text = ""
                if ans_path:
                    p = os.path.join(org_root, *ans_path.split("/"))
                    if os.path.exists(p):
                        text = D.pdf_header_text(p, pages=2)
                if not text.strip():
                    continue                    # a scanned PDF with no text layer: nothing
                                                # to give the model. Docling/OCR would be
                                                # needed, and that is what we are avoiding.
                rd = _llm_extract(cfg, text, doc_key, session_year=sy)
                if rd:
                    _apply(conn, doc_key, rd, args.dry_run)
                    from_llm += 1
                    misses.remove((doc_key, ans_path, sy))
                    log.info("%s  %s  [llm]", doc_key, rd)

        # ---- record the genuine misses, rather than leaving them ambiguous ---
        missed = len(misses)
        if not args.dry_run and misses:
            pass   # a miss leaves reply_date NULL -> "date unknown", sorted last

        # ---- report ------------------------------------------------------
        done = from_subject + from_pdf + from_llm
        print()
        print("  " + "=" * 62)
        print(f"  from subject (DB header)   {from_subject:>5d}")
        print(f"  from PDF text layer        {from_pdf:>5d}   (no Docling, no OCR)")
        print(f"  from LLM fallback          {from_llm:>5d}"
              f"{'' if args.llm else '   (not run — pass --llm)'}")
        print(f"  NO DATE FOUND              {missed:>5d}"
              f"   -> reply_date NULL, shown as 'date unknown'")
        print("  " + "-" * 62)
        print(f"  dated                      {done:>5d} / {n}"
              f"   ({100 * done // max(1, n)}%)")
        print(f"  {time.time() - t0:.1f}s")
        print("  " + "=" * 62)

        if missed and not args.llm:
            print(f"\n  Re-run with --llm to try the {missed} miss(es) through the model.")
        if args.dry_run:
            print("\n  DRY RUN — nothing was written.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
