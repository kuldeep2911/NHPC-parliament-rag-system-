"""
Audit log — every query and every file open, ALLOWED OR DENIED.

Government data: the record of who looked at what, and when, is not optional. Denials are
audited too -- a denied access attempt is the most interesting event in the log.

Writes are best-effort at the call site (an audit failure must not deny an officer their
results), but they are ordinary DB rows, not a text log, so they are queryable and
joinable to the query trace.
"""

from __future__ import annotations

import logging

log = logging.getLogger("nhpc.phase4.audit")


def log_query(conn, run_id, query_text, user_id, user_role, allowed,
              denial_reason=None, n_results=None, doc_keys=None):
    """Record a query attempt. `doc_keys` = which documents were surfaced."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_audit (run_id, query_text, user_id, user_role,
                                         allowed, denial_reason, n_results, doc_keys_shown)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (run_id, query_text, user_id, user_role, allowed, denial_reason,
                  n_results, list(doc_keys or [])))
    except Exception as e:                      # noqa: BLE001
        # An audit write failing must not break the officer's query -- but it MUST be
        # loud, because a silently unaudited system is a compliance failure.
        log.error("AUDIT WRITE FAILED (query): %s: %s", type(e).__name__, e)


def log_file_access(conn, run_id, doc_key, file_kind, ref_label, resolved_path,
                    user_id, user_role, allowed, denial_reason=None):
    """Record a file open (or a blocked attempt)."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO file_access_audit (run_id, doc_key, file_kind, ref_label,
                                               resolved_path, user_id, user_role,
                                               allowed, denial_reason)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (run_id, doc_key, file_kind, ref_label, resolved_path,
                  user_id, user_role, allowed, denial_reason))
    except Exception as e:                      # noqa: BLE001
        log.error("AUDIT WRITE FAILED (file): %s: %s", type(e).__name__, e)
