"""
Versioned SQL migrations for Phase 3.

Applies phase3/migrations/*.sql in filename order, recording each in a
`schema_migrations` table with its sha256 so a re-run is a no-op and an EDITED
already-applied migration is caught rather than silently ignored.

    python -m phase3.migrate            # apply pending
    python -m phase3.migrate --status   # show applied/pending, no writes

Reproducible on the on-prem server: the same files, same order, same result.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

from .config import Phase3Config, load_dotenv
from .db import connect

MIGRATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")

_TRACKING_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    sha256      text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


def _files():
    return sorted(glob.glob(os.path.join(MIGRATIONS_DIR, "*.sql")))


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Apply Phase-3 SQL migrations")
    ap.add_argument("--status", action="store_true", help="show state, apply nothing")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Phase3Config()
    errs = cfg.validate(need_db=True, need_embed=False)
    if errs:
        for e in errs:
            print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 1

    files = _files()
    if not files:
        print(f"no migrations found in {MIGRATIONS_DIR}")
        return 1

    with connect(cfg) as conn:
        with conn.cursor() as cur:
            cur.execute(_TRACKING_DDL)
            cur.execute("SELECT version, sha256 FROM schema_migrations")
            applied = {v: s for v, s in cur.fetchall()}

        pending, drifted = [], []
        for path in files:
            version = os.path.basename(path)
            sha = _sha(path)
            if version not in applied:
                pending.append((version, path, sha))
            elif applied[version] != sha:
                drifted.append(version)

        if drifted:
            print("ERROR: these migrations were EDITED after being applied:",
                  file=sys.stderr)
            for v in drifted:
                print(f"  {v}", file=sys.stderr)
            print("Add a NEW migration instead of editing an applied one.",
                  file=sys.stderr)
            return 1

        if args.status:
            print(f"applied : {len(applied)}")
            for v in sorted(applied):
                print(f"   [x] {v}")
            print(f"pending : {len(pending)}")
            for v, _p, _s in pending:
                print(f"   [ ] {v}")
            return 0

        if not pending:
            print(f"up to date ({len(applied)} migration(s) applied)")
            return 0

        for version, path, sha in pending:
            with open(path, encoding="utf-8") as fh:
                sql = fh.read()
            print(f"applying {version} ...", flush=True)
            # One transaction per migration: it applies fully or not at all.
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cur.execute(
                        "INSERT INTO schema_migrations (version, sha256) VALUES (%s, %s)",
                        (version, sha))
            print(f"  ok  {version}")

    print(f"done — {len(pending)} migration(s) applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
