"""
Database connection for Phase 3 (psycopg v3 + pgvector).

One place that opens a connection, so the DSN, statement timeout and the pgvector
type adapter are configured identically everywhere. The DSN (which carries the
password) comes from env and is never logged.
"""

from __future__ import annotations

import contextlib


@contextlib.contextmanager
def connect(cfg, autocommit=True):
    """
    Open a connection with the pgvector adapter registered.

    autocommit=True by default: DDL and the migration runner manage their own
    transactions explicitly (`with conn.transaction():`), and the loader wraps ONE
    DOCUMENT per transaction so a bad document rolls back alone.
    """
    import psycopg

    conn = psycopg.connect(cfg.db_dsn, autocommit=autocommit)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {int(cfg.db_statement_timeout_ms)}")
        _register_vector(conn)
        yield conn
    finally:
        conn.close()


def _register_vector(conn):
    """
    Teach psycopg the pgvector types, so a python list round-trips as vector(N).

    Safe to call before the extension exists (during the first migration): pgvector's
    register_vector looks the type up and raises if it isn't there yet, which is fine
    to ignore -- migration 001 creates the extension, and every later connection
    registers successfully.
    """
    try:
        from pgvector.psycopg import register_vector
        register_vector(conn)
    except Exception:
        pass


def fetch_scalar(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params or ())
        row = cur.fetchone()
        return row[0] if row else None
