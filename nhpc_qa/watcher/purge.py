"""
`nhpc purge` — permanently remove records that have been soft-deleted long enough.

THIS IS THE ONLY PLACE IN THE SYSTEM THAT DESTROYS DATA, and it is never reached from a
filesystem event. A file disappearing soft-deletes (flag, don't act); a human running this
command, with an explicit grace period and a confirmation, is what actually deletes.

That separation is the whole point:

    filesystem event  ->  soft delete  ->  excluded from retrieval, fully recoverable
    human decision    ->  purge        ->  gone

A purge cascades: removing a diaries row takes its sub_questions (and their 2048-dim
vectors), answer_groups, tables, rows and annexures with it, via ON DELETE CASCADE. There
is no undo. Hence: a long default grace period (30 days), a dry-run, and a confirmation
prompt unless --yes.

    nhpc purge --dry-run                 # what WOULD go
    nhpc purge --older-than 30d          # asks for confirmation
    nhpc purge --older-than 90d --yes    # unattended (e.g. a cron)
"""

from __future__ import annotations

import re
import sys

from nhpc_qa.core.logging import get_logger, setup as setup_logging
from nhpc_qa.core import queue as q

log = get_logger("nhpc.purge")


def parse_age(s: str) -> int:
    """'30d' -> 30 days in seconds. Accepts d/h/m."""
    m = re.fullmatch(r"\s*(\d+)\s*([dhm])\s*", (s or "").lower())
    if not m:
        raise ValueError(f"bad --older-than {s!r} (use e.g. 30d, 12h, 90m)")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 86400, "h": 3600, "m": 60}[unit]


_CANDIDATES_SQL = """
SELECT d.doc_key, d.session, d.house, d.question_id, d.deleted_at, d.deleted_reason,
       count(sq.sub_question_id) AS n_sub_questions
FROM diaries d
LEFT JOIN sub_questions sq ON sq.doc_key = d.doc_key
WHERE NOT d.active
  AND d.deleted_at IS NOT NULL
  AND d.deleted_at < now() - (%(secs)s * interval '1 second')
GROUP BY d.doc_key, d.session, d.house, d.question_id, d.deleted_at, d.deleted_reason
ORDER BY d.deleted_at
"""


def main(args):
    from nhpc_qa.config import Settings, load_dotenv
    from nhpc_qa.core.db.session import connect

    setup_logging()
    load_dotenv()
    cfg = Settings()
    errs = cfg.validate_all(need_embed=False, need_rerank=False)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    try:
        secs = parse_age(getattr(args, "older_than", None) or f"{cfg.purge_grace_days}d")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(_CANDIDATES_SQL, {"secs": secs})
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]

        print("=" * 74)
        print(f"PURGE — soft-deleted for longer than {args.older_than}")
        print("=" * 74)
        if not rows:
            print("\n  nothing is eligible. (Soft-deleted records younger than the grace "
                  "period are kept, and can still be reactivated.)")
            return 0

        total_sq = sum(r["n_sub_questions"] for r in rows)
        for r in rows:
            print(f"  {r['doc_key']:<44} deleted {r['deleted_at']:%Y-%m-%d} "
                  f"({r['n_sub_questions']} sub-questions)")
        print(f"\n  {len(rows)} document(s), {total_sq} sub-question(s) and their vectors.")

        if getattr(args, "dry_run", False):
            print("\n  DRY RUN — nothing removed.")
            return 0

        print("\n  THIS IS PERMANENT. Rows, answers, tables and embeddings are destroyed;")
        print("  there is no undo. Re-ingesting means re-parsing and re-embedding.")
        if not getattr(args, "yes", False):
            try:
                ans = input("\n  Type 'purge' to confirm: ").strip()
            except EOFError:
                ans = ""
            if ans != "purge":
                print("  aborted.")
                return 1

        keys = [r["doc_key"] for r in rows]
        with conn.transaction():
            with conn.cursor() as cur:
                # ON DELETE CASCADE removes sub_questions (+vectors), answer_groups,
                # tables, rows and annexures.
                cur.execute("DELETE FROM diaries WHERE doc_key = ANY(%s)", (keys,))
        for r in rows:
            log.warning("PURGED %s (%d sub-questions) — permanent",
                        r["doc_key"], r["n_sub_questions"])
            q.log_action(conn, "purged", doc_key=r["doc_key"],
                         detail=f"purged after {args.older_than} soft-deleted",
                         n_sub_questions=r["n_sub_questions"])

        print(f"\n  purged {len(keys)} document(s).")
    return 0
