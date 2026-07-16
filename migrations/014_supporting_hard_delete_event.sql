-- 014 — allow the supporting hard-delete audit event.
--
-- Deleting a reference document from the UI removes the FILE and the ROW (not just a
-- soft-delete). That deliberate, irreversible act is audited as 'supporting_deleted', which
-- the auth_audit CHECK must permit -- otherwise the delete succeeds but its audit row is
-- rejected (the same class of bug migration 013 fixed for the upload events).

ALTER TABLE auth_audit DROP CONSTRAINT IF EXISTS auth_audit_event_check;

ALTER TABLE auth_audit ADD CONSTRAINT auth_audit_event_check CHECK (event IN (
    'login_success', 'login_failed', 'login_locked', 'logout',
    'user_created', 'password_changed', 'password_reset',
    'user_deactivated', 'user_reactivated', 'admin_bootstrapped',
    'access_denied',
    'upload_accepted', 'upload_rejected', 'upload_conflict',
    'tree_deleted',
    'supporting_uploaded', 'supporting_upload_rejected',
    'supporting_deactivated', 'supporting_file_opened',
    -- 014: hard delete (file + row) of a reference document
    'supporting_deleted'
));
