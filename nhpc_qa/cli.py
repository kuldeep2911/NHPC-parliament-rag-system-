"""
nhpc — the single entry point for the whole application.

    nhpc run                        full pipeline: crawl -> parse -> index (+ embeddings)
    nhpc run --stages index         only DB load + embedding generation
    nhpc run --stages parse,index   a subset, always in canonical order
    nhpc run --from parse           from a stage to the end
    nhpc run --only 8773 --force    a single question folder, re-processed

    nhpc serve                      start the API + officer UI
    nhpc watch                      start the source-directory watcher (incremental sync)
    nhpc query "electricity dues"   one query from the terminal
    nhpc migrate                    apply DB migrations
    nhpc inspect --doc 8773         browse the database
    nhpc purge --older-than 30d     permanently remove long-soft-deleted records (explicit)

Global flags on `run`: --dry-run, --limit N, --force, --only SUBPATH.

Every stage stays INDEPENDENTLY runnable and IDEMPOTENT (deterministic ids, upsert by
primary key), so partial runs compose and a crash is fixed by re-running the same command.

Install as `nhpc` via pyproject; until then:  python -m nhpc_qa.cli <command>
"""

from __future__ import annotations

import argparse
import sys

from nhpc_qa.core.logging import get_logger, setup as setup_logging

log = get_logger("nhpc.cli")


def _load_cfg(need_db=True, need_embed=True, need_rerank=None):
    """Build the ONE config object and fail fast, reporting every problem at once."""
    from nhpc_qa.config import Settings, load_dotenv

    load_dotenv()
    cfg = Settings()
    errs = cfg.validate_all(need_db=need_db, need_embed=need_embed,
                            need_rerank=need_rerank)
    if errs:
        print("CONFIG ERROR:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)
    return cfg


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_run(args):
    from nhpc_qa.pipeline.orchestrator import StageError, run_stages

    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    # crawl/parse do not need the DB or a reranker; index does. Only demand what the
    # selected stages actually use, so `nhpc run --stages crawl` works with no DB up.
    plan_needs_db = (not stages and not args.from_stage) or \
                    (stages and "index" in stages) or \
                    (args.from_stage in ("crawl", "parse", "index"))
    cfg = _load_cfg(need_db=plan_needs_db, need_embed=plan_needs_db, need_rerank=False)
    try:
        results = run_stages(cfg, stages=stages, from_stage=args.from_stage,
                             source=args.source, dry_run=args.dry_run,
                             limit=args.limit, force=args.force, only=args.only)
    except StageError as e:
        print(f"\nPIPELINE FAILED: {e}", file=sys.stderr)
        return 2

    print("\n" + "=" * 58)
    print("PIPELINE SUMMARY" + ("  (DRY RUN — nothing written)" if args.dry_run else ""))
    print("=" * 58)
    for stage, r in results.items():
        print(f"  {stage:8} {'ok' if r['ok'] else 'FAILED':8} {r['seconds']:>7.1f}s")
    return 0


def cmd_serve(args):
    from nhpc_qa.api.app import main as api_main
    return api_main()


def cmd_watch(args):
    from nhpc_qa.watcher.runner import main as watch_main
    return watch_main(args)


def cmd_query(args):
    from nhpc_qa.retrieval.graph.run import main as query_main

    argv = [args.text]
    if args.json:
        argv.append("--json")
    return query_main(argv)


def cmd_migrate(args):
    from nhpc_qa.core.db.migrate import main as migrate_main
    return migrate_main(["--status"] if args.status else [])


def cmd_inspect(args):
    from nhpc_qa.scripts.inspect_db import main as inspect_main

    argv = []
    if args.doc:
        argv += ["--doc", args.doc]
    if args.search:
        argv += ["--search", args.search]
    if args.similar:
        argv += ["--similar", args.similar]
    if args.sql:
        argv += ["--sql", args.sql]
    if args.schema:
        argv.append("--schema")
    return inspect_main(argv)


def cmd_purge(args):
    from nhpc_qa.watcher.purge import main as purge_main
    return purge_main(args)


def cmd_backfill_dates(args):
    from nhpc_qa.pipeline.index.backfill_dates import main as bf
    argv = []
    if args.llm:     argv.append("--llm")
    if args.dry_run: argv.append("--dry-run")
    if args.force:   argv.append("--force")
    if args.limit:   argv += ["--limit", str(args.limit)]
    return bf(argv)


def cmd_create_admin(args):
    from nhpc_qa.api.security.bootstrap import create_admin
    return create_admin(email=args.email)


def cmd_reset_password(args):
    from nhpc_qa.api.security.bootstrap import reset_password
    return reset_password(email=args.email)


def cmd_deactivate_user(args):
    from nhpc_qa.api.security.bootstrap import deactivate
    return deactivate(email=args.email)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="nhpc",
        description="NHPC parliamentary Q&A — pipeline, retrieval API, and incremental sync")
    ap.add_argument("--log-level", default=None,
                    help="DEBUG|INFO|WARNING (or NHPC_LOG_LEVEL)")
    sub = ap.add_subparsers(dest="command", required=True)

    # run
    p = sub.add_parser("run", help="run the pipeline (crawl -> parse -> index)")
    p.add_argument("--stages", default=None,
                   help="comma-separated subset: crawl,parse,index (always run in order)")
    p.add_argument("--from", dest="from_stage", default=None,
                   help="run from this stage to the end")
    p.add_argument("--source", default=None, help="original source dir (crawl input)")
    p.add_argument("--only", default=None,
                   help="restrict to question folders matching this substring")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--force", action="store_true", help="re-process work already done")
    p.add_argument("--dry-run", action="store_true", help="validate + report, no writes")
    p.set_defaults(func=cmd_run)

    # serve
    p = sub.add_parser("serve", help="start the API + officer UI")
    p.set_defaults(func=cmd_serve)

    # watch
    p = sub.add_parser("watch", help="watch the source dir and sync incrementally")
    p.add_argument("--source", default=None)
    p.add_argument("--once", action="store_true",
                   help="drain the queue and exit (does not keep watching)")
    p.set_defaults(func=cmd_watch)

    # query
    p = sub.add_parser("query", help="run one retrieval query")
    p.add_argument("text")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_query)

    # migrate
    p = sub.add_parser("migrate", help="apply database migrations")
    p.add_argument("--status", action="store_true", help="show state, apply nothing")
    p.set_defaults(func=cmd_migrate)

    # inspect
    p = sub.add_parser("inspect", help="browse the database")
    p.add_argument("--doc", help="expand one document by doc_key or diary number")
    p.add_argument("--search", help="full-text search")
    p.add_argument("--similar", help="vector-similarity search from a sub_question id")
    p.add_argument("--sql", help="read-only SELECT")
    p.add_argument("--schema", action="store_true")
    p.set_defaults(func=cmd_inspect)

    # purge — DELIBERATE, never triggered by a filesystem event
    p = sub.add_parser(
        "purge",
        help="PERMANENTLY remove records soft-deleted longer than a grace period")
    p.add_argument("--older-than", default="30d",
                   help="e.g. 30d, 12h — how long a record must have been soft-deleted")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.add_argument("--dry-run", action="store_true", help="show what would be purged")
    p.set_defaults(func=cmd_purge)

    # backfill-dates — rule-based over EXISTING data. No Docling, no re-parse; the LLM
    # only on the leftovers, and only if asked.
    p = sub.add_parser(
        "backfill-dates",
        help="extract reply dates for existing documents (no re-parse; --llm for misses)")
    p.add_argument("--llm", action="store_true",
                   help="use the LLM on rule-based misses (costs API calls)")
    p.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    p.add_argument("--force", action="store_true", help="redo already-dated documents")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_backfill_dates)

    # create-admin — one-time first-run bootstrap. Prints a generated password ONCE.
    p = sub.add_parser(
        "create-admin",
        help="one-time: create the administrator (prints a generated password once)")
    p.add_argument("--email", default=None,
                   help="the admin's email (or set AUTH_ADMIN_EMAIL)")
    p.set_defaults(func=cmd_create_admin)

    # reset-password — BREAK GLASS. An admin who is locked out cannot use the admin UI to
    # fix it, and hand-written SQL against the users table is how people get this wrong.
    p = sub.add_parser(
        "reset-password",
        help="break-glass: issue a new password for a user (prints it once)")
    p.add_argument("--email", required=True)
    p.set_defaults(func=cmd_reset_password)

    p = sub.add_parser(
        "deactivate-user",
        help="disable an account and revoke its sessions (the account is retained)")
    p.add_argument("--email", required=True)
    p.set_defaults(func=cmd_deactivate_user)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    setup_logging(level=args.log_level)
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
