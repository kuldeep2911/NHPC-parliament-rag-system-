"""
The user + session store. Every DB write that auth performs lives here.

This is the ONLY module that reads or writes `users`, `sessions` and `auth_audit`. The
endpoints call these functions; they never write auth SQL themselves.

Everything commits explicitly. An audit row that is rolled back is not an audit row.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone

from nhpc_qa.api.security import passwords

log = logging.getLogger("nhpc.auth")


def _now():
    return datetime.now(timezone.utc)


class AuthError(Exception):
    """Anything the client is allowed to be told. The message is deliberately generic."""


# ---------------------------------------------------------------------------
# AUDIT — write first, ask questions later
# ---------------------------------------------------------------------------
def audit(conn, event, *, success, actor=None, target=None, reason=None,
          ip=None, user_agent=None):
    """
    Record an auth event. FAILURES ARE AUDITED TOO -- a login_failed row is the whole
    point of having this table.

    `reason` is a short machine-ish string ('bad_password', 'locked', 'inactive'). It
    NEVER contains a password, a token, or a hash.
    """
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO auth_audit (event, actor_user_id, actor_email, target_user_id,
                                    target_email, success, reason, ip, user_agent)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (event,
              (actor or {}).get("user_id"), (actor or {}).get("email"),
              (target or {}).get("user_id"), (target or {}).get("email"),
              success, reason, ip, user_agent))
    conn.commit()


# ---------------------------------------------------------------------------
# USERS
# ---------------------------------------------------------------------------
_USER_COLS = ("user_id, email, password_hash, role, is_active, must_change_password, "
              "failed_login_count, locked_until, created_by, created_at, last_login_at")


def _row_to_user(cols, row):
    return dict(zip(cols, row)) if row else None


def get_by_email(conn, email):
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_USER_COLS} FROM users WHERE email = %s", (email,))
        cols = [c.name for c in cur.description]
        return _row_to_user(cols, cur.fetchone())


def get_by_id(conn, user_id):
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_USER_COLS} FROM users WHERE user_id = %s", (user_id,))
        cols = [c.name for c in cur.description]
        return _row_to_user(cols, cur.fetchone())


def list_users(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT u.user_id, u.email, u.role, u.is_active, u.must_change_password,
                   u.failed_login_count, u.locked_until, u.created_at, u.last_login_at,
                   c.email AS created_by_email
            FROM users u
            LEFT JOIN users c ON c.user_id = u.created_by
            ORDER BY u.created_at
        """)
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def admin_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM users WHERE role='admin' AND is_active LIMIT 1")
        return cur.fetchone() is not None


def any_user_exists(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM users LIMIT 1")
        return cur.fetchone() is not None


def create_user(conn, cfg, email, password, role, *, created_by=None,
                must_change=True, actor=None, ip=None, user_agent=None):
    """
    Create a user. The password is policy-checked SERVER-side and stored only as an
    Argon2id hash -- the plaintext is never persisted, never logged.
    """
    email = (email or "").strip().lower()
    if "@" not in email or len(email) < 5:
        raise AuthError("a valid email address is required")
    if role not in cfg.roles():
        raise AuthError(f"unknown role (allowed: {sorted(cfg.roles())})")

    problems = passwords.check_policy(cfg, password, email=email)
    if problems:
        raise AuthError("password " + "; ".join(problems))

    pw_hash = passwords.hash_password(cfg, password)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, password_hash, role, created_by,
                                   must_change_password)
                VALUES (%s,%s,%s,%s,%s)
                RETURNING user_id
            """, (email, pw_hash, role, created_by, must_change))
            uid = cur.fetchone()[0]
        conn.commit()
    except Exception as e:                      # noqa: BLE001
        conn.rollback()
        # citext UNIQUE means 'A@x.com' and 'a@x.com' collide -- as they should.
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise AuthError("a user with that email already exists")
        raise

    audit(conn, "user_created", success=True, actor=actor,
          target={"user_id": uid, "email": email}, reason=f"role={role}",
          ip=ip, user_agent=user_agent)
    log.info("user created: %s (role=%s) by %s", email, role,
             (actor or {}).get("email", "bootstrap"))
    return uid


def set_password(conn, cfg, user_id, new_password, *, must_change,
                 revoke_sessions_except=None):
    """
    Set a user's password and (by default) REVOKE THEIR OTHER SESSIONS.

    That revocation is the point: if a password is being changed because it may have been
    compromised, leaving the attacker's existing session alive defeats the change entirely.
    The caller's own session is preserved (revoke_sessions_except) so a user who changes
    their own password is not immediately logged out.
    """
    pw_hash = passwords.hash_password(cfg, new_password)
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE users SET password_hash=%s, must_change_password=%s,
                             failed_login_count=0, locked_until=NULL, updated_at=now()
            WHERE user_id=%s
        """, (pw_hash, must_change, user_id))
        cur.execute("""
            UPDATE sessions SET revoked_at = now()
            WHERE user_id = %s AND revoked_at IS NULL
              AND (%s::bigint IS NULL OR session_id <> %s::bigint)
        """, (user_id, revoke_sessions_except, revoke_sessions_except))
    conn.commit()


def set_active(conn, user_id, active: bool):
    """Deactivate/reactivate. Deactivating KILLS EVERY LIVE SESSION IMMEDIATELY -- which
    is precisely what a stateless JWT could not have done."""
    with conn.cursor() as cur:
        cur.execute("UPDATE users SET is_active=%s, updated_at=now() WHERE user_id=%s",
                    (active, user_id))
        if not active:
            cur.execute("""UPDATE sessions SET revoked_at = now()
                           WHERE user_id = %s AND revoked_at IS NULL""", (user_id,))
    conn.commit()


# ---------------------------------------------------------------------------
# LOGIN — the security-critical path
# ---------------------------------------------------------------------------
# Every failure below returns the SAME message. An attacker must not learn from the
# response whether the account exists, is locked, or is deactivated -- only that the
# attempt failed.
GENERIC_LOGIN_ERROR = "invalid email or password"


def authenticate(conn, cfg, email, password, *, ip=None, user_agent=None):
    """
    Verify credentials. Returns the user dict, or raises AuthError(GENERIC_LOGIN_ERROR).

    The real reason is written to auth_audit and to the server log -- never to the client.
    """
    email = (email or "").strip().lower()
    user = get_by_email(conn, email)

    if user is None:
        # Burn the same Argon2 work we would have spent on a real user, so "no such
        # account" does not answer measurably faster. See passwords.burn_dummy_verify.
        passwords.burn_dummy_verify(cfg)
        audit(conn, "login_failed", success=False, target={"email": email},
              reason="no_such_user", ip=ip, user_agent=user_agent)
        raise AuthError(GENERIC_LOGIN_ERROR)

    target = {"user_id": user["user_id"], "email": user["email"]}

    if not user["is_active"]:
        passwords.burn_dummy_verify(cfg)
        audit(conn, "login_failed", success=False, target=target, reason="inactive",
              ip=ip, user_agent=user_agent)
        raise AuthError(GENERIC_LOGIN_ERROR)

    locked_until = user["locked_until"]
    if locked_until and locked_until > _now():
        passwords.burn_dummy_verify(cfg)
        audit(conn, "login_locked", success=False, target=target,
              reason=f"locked_until={locked_until.isoformat()}", ip=ip, user_agent=user_agent)
        # Same generic message. Telling them "locked for 15 minutes" would confirm the
        # account exists AND hand them a scheduling hint.
        raise AuthError(GENERIC_LOGIN_ERROR)

    ok, needs_rehash = passwords.verify_password(cfg, password, user["password_hash"])

    if not ok:
        fails = user["failed_login_count"] + 1
        lock_to = None
        if fails >= cfg.max_failed_logins:
            lock_to = _now() + timedelta(minutes=cfg.lockout_minutes)
        with conn.cursor() as cur:
            cur.execute("""UPDATE users SET failed_login_count=%s, locked_until=%s,
                                            updated_at=now()
                           WHERE user_id=%s""", (fails, lock_to, user["user_id"]))
        conn.commit()
        audit(conn, "login_failed", success=False, target=target,
              reason=f"bad_password (attempt {fails}/{cfg.max_failed_logins})"
                     + (" -> LOCKED" if lock_to else ""),
              ip=ip, user_agent=user_agent)
        raise AuthError(GENERIC_LOGIN_ERROR)

    # success -- reset the counters
    with conn.cursor() as cur:
        cur.execute("""UPDATE users SET failed_login_count=0, locked_until=NULL,
                                        last_login_at=now(), updated_at=now()
                       WHERE user_id=%s""", (user["user_id"],))
        if needs_rehash:
            # The stored hash used weaker Argon2 params than we now require. We hold the
            # plaintext exactly once, right here -- upgrade it.
            cur.execute("UPDATE users SET password_hash=%s WHERE user_id=%s",
                        (passwords.hash_password(cfg, password), user["user_id"]))
            log.info("rehashed password for %s (argon2 params raised)", user["email"])
    conn.commit()

    audit(conn, "login_success", success=True, actor=target, target=target,
          ip=ip, user_agent=user_agent)
    return get_by_id(conn, user["user_id"])


# ---------------------------------------------------------------------------
# SESSIONS
# ---------------------------------------------------------------------------
def _hash_token(token: str) -> str:
    """SHA-256, not Argon2 -- and that is correct HERE.

    The input is already 256 bits of CSPRNG output, so there is nothing to brute-force:
    a slow hash would add no security and would be re-run on EVERY authenticated request,
    which is a self-inflicted DoS. Argon2's slowness exists to defend LOW-ENTROPY human
    passwords. Different problem, different tool."""
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(conn, cfg, user_id, *, ip=None, user_agent=None) -> str:
    """Mint a session. Returns the RAW token (the only time it exists); the DB gets only
    its hash."""
    token = secrets.token_urlsafe(32)               # 256 bits
    expires = _now() + timedelta(hours=cfg.session_hours)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO sessions (token_hash, user_id, expires_at, ip, user_agent)
                       VALUES (%s,%s,%s,%s,%s)""",
                    (_hash_token(token), user_id, expires, ip, user_agent))
    conn.commit()
    return token


def resolve_session(conn, token: str):
    """
    Token -> the live user, or None.

    Checked on EVERY request, which is what makes revocation instant: a logout, a
    deactivation or a password change takes effect on the attacker's very next call.
    Joins users so a deactivated account cannot ride an old session.
    """
    if not token:
        return None
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.session_id, u.user_id, u.email, u.role, u.is_active,
                   u.must_change_password
            FROM sessions s
            JOIN users u ON u.user_id = s.user_id
            WHERE s.token_hash = %s
              AND s.revoked_at IS NULL
              AND s.expires_at > now()
              AND u.is_active
        """, (_hash_token(token),))
        cols = [c.name for c in cur.description]
        row = cur.fetchone()
        if not row:
            return None
        sess = dict(zip(cols, row))
        cur.execute("UPDATE sessions SET last_seen_at = now() WHERE session_id = %s",
                    (sess["session_id"],))
    conn.commit()
    return sess


def revoke_session(conn, token: str):
    with conn.cursor() as cur:
        cur.execute("""UPDATE sessions SET revoked_at = now()
                       WHERE token_hash = %s AND revoked_at IS NULL""", (_hash_token(token),))
    conn.commit()
