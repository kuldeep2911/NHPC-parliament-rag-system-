"""
Auth + user-management endpoints.

    POST /auth/login             email + password  -> sets the session cookie
    POST /auth/logout            revokes the session
    GET  /auth/me                who am I (drives the UI shell)
    POST /auth/change-password   own password (current + new). Clears must_change_password.

    POST /admin/users                        create a user            ADMIN
    GET  /admin/users                        list users               ADMIN
    POST /admin/users/{id}/deactivate        disable + kill sessions  ADMIN
    POST /admin/users/{id}/reactivate        re-enable                ADMIN
    POST /admin/users/{id}/reset-password    temp password + re-arm   ADMIN

EVERY admin route depends on require_admin, which reads the role FROM THE DATABASE via the
session. Nothing here trusts a role sent by the client.

No response in this module ever contains a password hash, a session token, or a hint about
whether an account exists.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Response

from nhpc_qa.api.security import deps, passwords, users

log = logging.getLogger("nhpc.auth")

router = APIRouter()


def _st(request: Request):
    return request.app.state.nhpc


def _set_cookie(response: Response, cfg, token: str):
    response.set_cookie(
        key=cfg.cookie_name,
        value=token,
        max_age=cfg.session_hours * 3600,
        # HttpOnly: JavaScript CANNOT read this cookie. An XSS bug on the page therefore
        # cannot steal the session -- which is exactly why we are not putting a token in
        # localStorage.
        httponly=True,
        # Secure: the browser will only send it over HTTPS. Config-validated; may be
        # false ONLY on loopback (see config/auth.py).
        secure=cfg.cookie_secure,
        # Lax: not sent on cross-site POSTs -> CSRF protection for the state-changing
        # endpoints, while a normal link into the app still works.
        samesite=cfg.cookie_samesite,
        path="/",
    )


# ---------------------------------------------------------------------------
# LOGIN
# ---------------------------------------------------------------------------
@router.post("/auth/login")
def login(request: Request, response: Response, payload: dict = Body(...)):
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    ip = deps.client_ip(request)
    ua = request.headers.get("user-agent")

    # Per-IP budget FIRST -- before we touch the DB or spend Argon2 time. This is the half
    # of the defence that catches one attacker spraying many accounts.
    deps.check_ip_rate_limit(cfg, ip)
    deps.record_ip_attempt(ip)

    email = (payload.get("email") or "").strip()
    password = payload.get("password") or ""
    if not email or not password:
        raise HTTPException(400, users.GENERIC_LOGIN_ERROR)

    try:
        user = users.authenticate(conn, cfg, email, password, ip=ip, user_agent=ua)
    except users.AuthError as e:
        # ONE generic message for: no such user, wrong password, locked, deactivated.
        # The real reason went to auth_audit and the server log, never to the client.
        raise HTTPException(401, str(e))

    deps.clear_ip_attempts(ip)
    token = users.create_session(conn, cfg, user["user_id"], ip=ip, user_agent=ua)
    _set_cookie(response, cfg, token)

    return {"email": user["email"], "role": user["role"],
            "must_change_password": user["must_change_password"]}


@router.post("/auth/logout")
def logout(request: Request, response: Response):
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    token = request.cookies.get(cfg.cookie_name)
    who = deps.current_user(request)
    if token:
        users.revoke_session(conn, token)      # server-side revocation, not just a cookie wipe
    if who and who.get("db_user_id"):
        users.audit(conn, "logout", success=True,
                    actor={"user_id": who["db_user_id"], "email": who["email"]},
                    ip=deps.client_ip(request),
                    user_agent=request.headers.get("user-agent"))
    response.delete_cookie(cfg.cookie_name, path="/")
    return {"ok": True}


@router.get("/auth/me")
def me(request: Request):
    who = deps.current_user(request)
    if who is None:
        raise HTTPException(401, "not authenticated")
    return {"email": who["email"], "role": who["user_role"],
            "must_change_password": bool(who.get("must_change_password")),
            "auth_enabled": _st(request)["cfg"].auth_enabled}


# ---------------------------------------------------------------------------
# CHANGE OWN PASSWORD
# ---------------------------------------------------------------------------
@router.post("/auth/change-password")
def change_password(request: Request, payload: dict = Body(...),
                    who=Depends(deps.current_user)):
    """
    Requires the CURRENT password even though the caller is already authenticated --
    otherwise a stolen session alone is enough to lock the real owner out of their account.
    """
    if who is None:
        raise HTTPException(401, "not authenticated")
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]

    current = payload.get("current_password") or ""
    new = payload.get("new_password") or ""

    user = users.get_by_email(conn, who["email"])
    if user is None:
        raise HTTPException(401, "not authenticated")

    ok, _ = passwords.verify_password(cfg, current, user["password_hash"])
    if not ok:
        users.audit(conn, "password_changed", success=False,
                    actor={"user_id": user["user_id"], "email": user["email"]},
                    target={"user_id": user["user_id"], "email": user["email"]},
                    reason="wrong current password", ip=deps.client_ip(request),
                    user_agent=request.headers.get("user-agent"))
        raise HTTPException(400, "current password is incorrect")

    if new == current:
        raise HTTPException(400, "the new password must be different from the current one")

    problems = passwords.check_policy(cfg, new, email=user["email"])
    if problems:
        # Policy failures ARE returned in detail -- the user needs to know what to fix,
        # and this reveals nothing an attacker does not already know (the policy is public).
        raise HTTPException(400, "password " + "; ".join(problems))

    users.set_password(conn, cfg, user["user_id"], new, must_change=False,
                       revoke_sessions_except=who.get("session_id"))
    users.audit(conn, "password_changed", success=True,
                actor={"user_id": user["user_id"], "email": user["email"]},
                target={"user_id": user["user_id"], "email": user["email"]},
                reason="self-service", ip=deps.client_ip(request),
                user_agent=request.headers.get("user-agent"))
    # Their OTHER sessions were revoked; this one survives, so they are not logged out
    # mid-change.
    return {"ok": True, "must_change_password": False}


# ---------------------------------------------------------------------------
# ADMIN — user management
# ---------------------------------------------------------------------------
@router.get("/admin/users")
def admin_list_users(request: Request, admin=Depends(deps.require_admin)):
    rows = users.list_users(_st(request)["conn"])
    # No password_hash. Ever. Not even to an admin.
    return {"users": [{
        "user_id": r["user_id"], "email": r["email"], "role": r["role"],
        "is_active": r["is_active"], "must_change_password": r["must_change_password"],
        "failed_login_count": r["failed_login_count"],
        "locked": bool(r["locked_until"]),
        "created_at": r["created_at"].isoformat() if r["created_at"] else None,
        "last_login_at": r["last_login_at"].isoformat() if r["last_login_at"] else None,
        "created_by": r["created_by_email"],
    } for r in rows]}


@router.post("/admin/users")
def admin_create_user(request: Request, payload: dict = Body(...),
                      admin=Depends(deps.require_admin)):
    """
    Create a user with an initial password the admin hands over OUT OF BAND.

    must_change_password is forced true: the admin knows this password, so it is not the
    user's password until the user has replaced it.
    """
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    email = payload.get("email") or ""
    role = payload.get("role") or "officer"
    password = payload.get("password") or ""

    # Convenience: let the admin ask the SERVER to generate a strong one rather than
    # inventing 'Welcome@123'.
    generated = False
    if not password:
        password = passwords.generate_password()
        generated = True

    try:
        uid = users.create_user(
            conn, cfg, email, password, role,
            created_by=admin.get("db_user_id"), must_change=True,
            actor={"user_id": admin.get("db_user_id"), "email": admin["email"]},
            ip=deps.client_ip(request), user_agent=request.headers.get("user-agent"))
    except users.AuthError as e:
        raise HTTPException(400, str(e))

    # The plaintext is returned EXACTLY ONCE, to the admin who just set it, over the
    # already-authenticated TLS channel -- so they can hand it to the user. It is not
    # stored and not logged. If the admin supplied it, we do not echo it back.
    return {"ok": True, "user_id": uid, "email": email.strip().lower(), "role": role,
            "initial_password": password if generated else None,
            "must_change_password": True}


@router.post("/admin/users/{user_id}/deactivate")
def admin_deactivate(request: Request, user_id: int, admin=Depends(deps.require_admin)):
    conn = _st(request)["conn"]
    target = users.get_by_id(conn, user_id)
    if not target:
        raise HTTPException(404, "no such user")

    # An admin who deactivates themselves locks everyone out of user management, with no
    # way back in short of SQL. Refuse.
    if target["user_id"] == admin.get("db_user_id"):
        raise HTTPException(400, "you cannot deactivate your own account")
    if target["role"] == "admin" and _count_active_admins(conn) <= 1:
        raise HTTPException(400, "cannot deactivate the last active administrator")

    users.set_active(conn, user_id, False)      # also revokes every live session
    users.audit(conn, "user_deactivated", success=True,
                actor={"user_id": admin.get("db_user_id"), "email": admin["email"]},
                target={"user_id": user_id, "email": target["email"]},
                reason="sessions revoked", ip=deps.client_ip(request),
                user_agent=request.headers.get("user-agent"))
    return {"ok": True, "email": target["email"], "is_active": False}


@router.post("/admin/users/{user_id}/reactivate")
def admin_reactivate(request: Request, user_id: int, admin=Depends(deps.require_admin)):
    conn = _st(request)["conn"]
    target = users.get_by_id(conn, user_id)
    if not target:
        raise HTTPException(404, "no such user")
    users.set_active(conn, user_id, True)
    users.audit(conn, "user_reactivated", success=True,
                actor={"user_id": admin.get("db_user_id"), "email": admin["email"]},
                target={"user_id": user_id, "email": target["email"]},
                ip=deps.client_ip(request),
                user_agent=request.headers.get("user-agent"))
    return {"ok": True, "email": target["email"], "is_active": True}


@router.post("/admin/users/{user_id}/reset-password")
def admin_reset_password(request: Request, user_id: int, payload: dict = Body(default={}),
                         admin=Depends(deps.require_admin)):
    """
    Issue a new temporary password and RE-ARM must_change_password.

    Every live session of that user is revoked (users.set_password), because a reset is
    usually a response to a suspected compromise -- and leaving the attacker's session
    alive would make the reset pointless.
    """
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    target = users.get_by_id(conn, user_id)
    if not target:
        raise HTTPException(404, "no such user")

    password = (payload or {}).get("password") or ""
    generated = False
    if not password:
        password = passwords.generate_password()
        generated = True
    else:
        problems = passwords.check_policy(cfg, password, email=target["email"])
        if problems:
            raise HTTPException(400, "password " + "; ".join(problems))

    users.set_password(conn, cfg, user_id, password, must_change=True,
                       revoke_sessions_except=None)   # kill ALL of the target's sessions
    users.audit(conn, "password_reset", success=True,
                actor={"user_id": admin.get("db_user_id"), "email": admin["email"]},
                target={"user_id": user_id, "email": target["email"]},
                reason="temporary password issued; all sessions revoked",
                ip=deps.client_ip(request),
                user_agent=request.headers.get("user-agent"))
    return {"ok": True, "email": target["email"],
            "temporary_password": password if generated else None,
            "must_change_password": True}


def _count_active_admins(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM users WHERE role='admin' AND is_active")
        return cur.fetchone()[0]
