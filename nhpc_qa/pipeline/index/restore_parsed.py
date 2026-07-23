"""
Restore parsed.json into each question folder from what the database already holds.

    python -m nhpc_qa.pipeline.index.restore_parsed --dry-run   # report only
    python -m nhpc_qa.pipeline.index.restore_parsed             # write missing files
    python -m nhpc_qa.pipeline.index.restore_parsed --force     # rewrite existing ones too

Why this exists. The parse stage writes parsed.json next to the answer file, and the loader
stores the same document in diaries.raw_json. Those files can go missing (cleaned up, moved,
restored from a partial backup) while the database still has every record — leaving the
corpus indexed but with no on-disk JSON to inspect, diff or re-load.

Rather than re-run the LLM over the whole corpus (slow and costly, and it would produce
DIFFERENT output because extraction is not bit-reproducible), this rewrites the exact
document the database was loaded from. The file and the DB therefore agree by construction.

Idempotent: a folder that already has parsed.json is skipped unless --force. Writes are
atomic (temp file + replace), so an interrupted run never leaves a truncated file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.core.db.session import connect


def _atomic_write_json(path: str, obj: dict) -> None:
    """Write JSON via temp file + os.replace so readers never see a partial file."""
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def question_dir_for(organized_root: str, doc_key: str) -> str:
    """
    doc_key is '<session>/<house>/<question_id>', which is exactly the folder layout the
    crawler writes, so the on-disk path is the doc_key under organized/.
    """
    return os.path.join(organized_root, *doc_key.split("/"))


def restore(cfg, *, dry_run: bool = False, force: bool = False, limit: int = 0) -> dict:
    """Write parsed.json for every active document whose folder is missing one."""
    organized_root = getattr(cfg, "organized_root", "organized")
    parsed_name = getattr(cfg, "parsed_filename", "parsed.json")

    written = skipped = missing_dir = no_raw = 0
    problems = []

    with connect(cfg) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT doc_key, raw_json
            FROM diaries
            WHERE active
            ORDER BY doc_key
        """)
        rows = cur.fetchall()

    for doc_key, raw in rows:
        if limit and written >= limit:
            break
        if not raw:
            no_raw += 1
            problems.append((doc_key, "no raw_json in the database"))
            continue

        qdir = question_dir_for(organized_root, doc_key)
        if not os.path.isdir(qdir):
            missing_dir += 1
            problems.append((doc_key, f"folder not found: {qdir}"))
            continue

        target = os.path.join(qdir, parsed_name)
        if os.path.exists(target) and not force:
            skipped += 1
            continue

        doc = raw if isinstance(raw, dict) else json.loads(raw)
        if not dry_run:
            _atomic_write_json(target, doc)
        written += 1

    return {"documents": len(rows), "written": written, "skipped_existing": skipped,
            "folder_missing": missing_dir, "no_raw_json": no_raw, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Restore parsed.json into question folders from diaries.raw_json")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true", help="rewrite parsed.json even if present")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Settings()
    errs = cfg.validate(need_db=True, need_embed=False)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    r = restore(cfg, dry_run=args.dry_run, force=args.force, limit=args.limit)

    print("\n" + "=" * 56)
    print("RESTORE parsed.json" + ("  (DRY RUN — nothing written)" if args.dry_run else ""))
    print("=" * 56)
    print(f"  documents in db     : {r['documents']}")
    print(f"  written             : {r['written']}")
    print(f"  skipped (existing)  : {r['skipped_existing']}")
    print(f"  folder missing      : {r['folder_missing']}")
    print(f"  no raw_json         : {r['no_raw_json']}")
    if r["problems"]:
        print("\n  problems (first 10):")
        for dk, why in r["problems"][:10]:
            print(f"    {dk}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
