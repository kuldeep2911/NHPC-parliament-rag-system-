-- 016 — allow the supporting_category_created audit event.
ALTER TABLE auth_audit DROP CONSTRAINT IF EXISTS auth_audit_event_check;
ALTER TABLE auth_audit ADD CONSTRAINT auth_audit_event_check CHECK (event IN (
    'login_success', 'login_failed', 'login_locked', 'logout',
    'user_created', 'password_changed', 'password_reset',
    'user_deactivated', 'user_reactivated', 'admin_bootstrapped',
    'access_denied',
    'upload_accepted', 'upload_rejected', 'upload_conflict',
    'tree_deleted',
    'supporting_uploaded', 'supporting_upload_rejected',
    'supporting_deactivated', 'supporting_file_opened', 'supporting_deleted',
    'supporting_category_created'
));
