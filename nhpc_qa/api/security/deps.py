"""
THE IDENTITY SEAM.

Everything in this application that cares who you are consumes exactly one shape:

    {"user_id": <str>, "user_role": <str>}

rbac.check(), audit.log_query(), audit.log_file_access() and all three app endpoints were
already written against it. This module changes only WHERE that shape comes from:

    BEFORE   X-User-Id / X-User-Role headers        <- client-supplied, self-asserted
    NOW      session cookie -> sessions -> users    <- server-derived, revocable

Nothing downstream changed. That is the point, and it is also the SSO seam: an OIDC/LDAP
provider later populates the same dict from a token, and no endpoint authorization is
rewritten.

⚠️ THE ROLE IS NEVER READ FROM THE CLIENT. It is read from the users table, every request.
A client can send whatever it likes; it is ignored.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from nhpc_qa.api.security import users

log = logging.getLogger("nhpc.auth")

# Endpoints a user with must_change_password=true may still reach. Everything else is
# refused until they have set a new password -- enforced HERE, in the dependency, not by
# hiding a screen in the UI.
_PASSWORD_CHANGE_ALLOWED = {"/auth/change-password", "/auth/me", "/auth/logout"}


# ---------------------------------------------------------------------------
# PER-IP RATE LIMIT
# ---------------------------------------------------------------------------
# In-memory sliding window. Deliberately NOT in the DB: it is checked before we know who
# is calling, it is high-churn, and losing it on restart is acceptable (the per-user
# lockout, which is NOT lost on restart, is the durable half of the defence).
#
# Per-user counters alone are not enough: an attacker trying 5 passwords each against 1000
# accounts trips no per-user lock at all. This catches the spray.
_ip_hits: dict[str, deque] = defaultdict(deque)


def client_ip(request: Request) -> str:
    # X-Forwarded-For is only meaningful because a reverse proxy sets it. We take the
    # LEFTMOST entry (the original client); everything after it is proxy chain.
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_ip_rate_limit(cfg, ip: str) -> None:
    """Raises 429 when an IP has burned its attempt budget."""
    window = cfg.ip_window_minutes * 60
    now = time.time()
    hits = _ip_hits[ip]
    while hits and now - hits[0] > window:
        hits.popleft()
    if len(hits) >= cfg.ip_max_attempts:
        log.warning("rate limit: ip=%s exceeded %d attempts in %d min",
                    ip, cfg.ip_max_attempts, cfg.ip_window_minutes)
        raise HTTPException(429, "too many attempts — try again later")


def record_ip_attempt(ip: str) -> None:
    _ip_hits[ip].append(time.time())


def clear_ip_attempts(ip: str) -> None:
    """A successful login clears the budget, so a legitimate user who fumbled their
    password a few times is not punished afterwards."""
    _ip_hits.pop(ip, None)


# ---------------------------------------------------------------------------
# THE DEPENDENCIES
# ---------------------------------------------------------------------------
def _state(request: Request):
    return request.app.state.nhpc          # {"cfg":..., "conn":..., ...}


def current_user(request: Request):
    """
    The authenticated user, or None. Does not raise -- for endpoints that merely want to
    know (e.g. the UI shell).
    """
    st = _state(request)
    cfg, conn = st["cfg"], st["conn"]

    if not cfg.auth_enabled:
        # Auth switched off (dev / pre-bootstrap). The legacy fixed identity, so the app
        # still runs exactly as it did before this feature landed.
        return {"user_id": "officer1", "email": "officer1", "user_role": "officer",
                "must_change_password": False, "session_id": None}

    token = request.cookies.get(cfg.cookie_name)
    sess = users.resolve_session(conn, token) if token else None
    if not sess:
        return None
    # user_id is the EMAIL, because that is what the existing audit tables store as text
    # and what an operator actually wants to read in a log.
    return {"user_id": sess["email"], "email": sess["email"],
            "user_role": sess["role"], "must_change_password": sess["must_change_password"],
            "db_user_id": sess["user_id"], "session_id": sess["session_id"]}


def require_user(request: Request):
    """
    Authenticated, active, and NOT owing a password change. This is what the app
    endpoints depend on.

    401 = not logged in.  403 password_change_required = logged in but must reset first.
    """
    who = current_user(request)
    if who is None:
        raise HTTPException(401, "not authenticated")

    if who.get("must_change_password") and request.url.path not in _PASSWORD_CHANGE_ALLOWED:
        # Server-side, so it holds even if the frontend is bypassed entirely. A hidden
        # screen is not a control.
        raise HTTPException(403, "password_change_required")
    return who


def require_admin(request: Request):
    """
    Admin only. The role is read from the DB via the session -- NEVER from the request.

    A denial is audited: an officer probing /admin/users is exactly the event a security
    log exists to capture.
    """
    who = require_user(request)
    if who["user_role"] != "admin":
        st = _state(request)
        users.audit(st["conn"], "access_denied", success=False,
                    actor={"user_id": who.get("db_user_id"), "email": who["email"]},
                    reason=f"role={who['user_role']} attempted admin action "
                           f"{request.method} {request.url.path}",
                    ip=client_ip(request),
                    user_agent=request.headers.get("user-agent"))
        log.warning("DENIED: %s (role=%s) attempted %s %s",
                    who["email"], who["user_role"], request.method, request.url.path)
        raise HTTPException(403, "administrator access required")
    return who
