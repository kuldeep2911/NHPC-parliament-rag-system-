"""
Role-based access control.

Deliberately minimal and explicit: a role check before /query and before /file. Roles
come from config (PHASE4_ROLES_QUERY / PHASE4_ROLES_FILE), never hardcoded.

IDENTITY SOURCE: this single-server on-prem deployment sits behind the organisation's
own authentication (reverse proxy / SSO), which is what actually authenticates the user.
This module TRUSTS the identity headers that proxy sets and enforces AUTHORISATION on
top of them. That trust boundary is stated explicitly here so it cannot be assumed away:

    X-User-Id    who
    X-User-Role  what they may do

⚠️ If the API is ever exposed WITHOUT such a proxy, these headers are self-asserted and
the RBAC is worthless. Bind to 127.0.0.1 (the default) and put a real authenticator in
front of it before exposing the port. This is documented rather than silently assumed.
"""

from __future__ import annotations


class AccessDenied(Exception):
    """Raised when a role is not permitted an action. Always audited."""


def check(cfg, action: str, user_role: str | None):
    """
    Authorise `action` ('query' | 'file') for `user_role`. Raises AccessDenied.

    Fails CLOSED: a missing or unknown role is denied, never defaulted to something
    permissive.
    """
    allowed = cfg.roles_for(action)
    role = (user_role or "").strip()
    if not role:
        raise AccessDenied(f"no role supplied for '{action}' (allowed: {sorted(allowed)})")
    if role not in allowed:
        raise AccessDenied(
            f"role '{role}' may not '{action}' (allowed: {sorted(allowed)})")
    return True
