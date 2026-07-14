-- 007 — let auth_audit record upload events.
--
-- 005 pinned the `event` column to a CHECK list. Upload adds three events to it. Extending
-- the constraint rather than dropping it: a free-text event column would let a typo
-- ('login_faild') silently create a category nobody ever greps for, and the whole point of
-- the audit table is that a security question has one answer.

ALTER TABLE auth_audit DROP CONSTRAINT IF EXISTS auth_audit_event_check;

ALTER TABLE auth_audit ADD CONSTRAINT auth_audit_event_check CHECK (event IN (
    -- authentication (005)
    'login_success', 'login_failed', 'login_locked', 'logout',
    'user_created', 'password_changed', 'password_reset',
    'user_deactivated', 'user_reactivated', 'admin_bootstrapped',
    'access_denied',
    -- upload (006). A refusal is as much an audit event as an acceptance: "why is that
    -- file not in the system?" must always be answerable.
    'upload_accepted', 'upload_rejected', 'upload_conflict'
));
