-- 013 — let auth_audit record the supporting-document events.
--
-- The supporting-documents feature audits its uploads, deactivations and file opens (a
-- government reply must be traceable to the exact inputs it was built from). auth_audit has
-- a CHECK on `event`, so a new event type is rejected until it is listed here -- which is
-- exactly what a 500 on the first supporting upload revealed. This migration re-states the
-- full allowed set with the four new events appended.

ALTER TABLE auth_audit DROP CONSTRAINT IF EXISTS auth_audit_event_check;

ALTER TABLE auth_audit ADD CONSTRAINT auth_audit_event_check CHECK (event IN (
    'login_success', 'login_failed', 'login_locked', 'logout',
    'user_created', 'password_changed', 'password_reset',
    'user_deactivated', 'user_reactivated', 'admin_bootstrapped',
    'access_denied',
    'upload_accepted', 'upload_rejected', 'upload_conflict',
    'tree_deleted',
    -- 013: supporting reference documents
    'supporting_uploaded', 'supporting_upload_rejected',
    'supporting_deactivated', 'supporting_file_opened'
));
