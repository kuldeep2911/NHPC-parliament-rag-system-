-- 009 — let auth_audit record the admin's tree deletions.
--
-- A HARD delete is the most destructive thing an admin can do in this application: it
-- removes parliamentary records, their answers and their 2048-dim vectors, permanently.
-- If it is not in the audit log, "who removed diary 8779, and when?" has no answer.

ALTER TABLE auth_audit DROP CONSTRAINT IF EXISTS auth_audit_event_check;

ALTER TABLE auth_audit ADD CONSTRAINT auth_audit_event_check CHECK (event IN (
    'login_success', 'login_failed', 'login_locked', 'logout',
    'user_created', 'password_changed', 'password_reset',
    'user_deactivated', 'user_reactivated', 'admin_bootstrapped',
    'access_denied',
    'upload_accepted', 'upload_rejected', 'upload_conflict',
    -- 009: irreversible removal of source files + indexed documents by an admin
    'tree_deleted'
));


-- sync_log must also accept 'hard_deleted'. Its CHECK list did not include it, and
-- queue.log_action() swallows its own exceptions (logging must never break the pipeline)
-- -- so a hard delete would have been silently absent from the sync history. The one
-- record of an irreversible act, quietly dropped.
ALTER TABLE sync_log DROP CONSTRAINT IF EXISTS sync_log_action_check;

ALTER TABLE sync_log ADD CONSTRAINT sync_log_action_check CHECK (action IN (
    'added', 'updated', 'soft_deleted', 'reactivated', 'purged',
    'failed', 'skipped',
    'hard_deleted'          -- 009: deliberate, audited removal by an admin
));
