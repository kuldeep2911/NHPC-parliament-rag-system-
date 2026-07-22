"""
Authentication config. Every knob from the environment; nothing hardcoded; validated at
startup so a misconfiguration fails LOUDLY at boot rather than quietly at 3am.

There is exactly one secret here (AUTH_SECRET_KEY) and it has NO DEFAULT. A default
secret is worse than no secret: it looks configured, and every deployment that forgets to
set it shares the same one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(k, d=""):   return os.getenv(k, d).strip()
def _env_int(k, d):
    try:    return int(os.getenv(k, str(d)))
    except ValueError: return d
def _env_bool(k, d):     return os.getenv(k, str(d)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class AuthConfig:
    # --- the seam -----------------------------------------------------------
    # Auth is OFF by default so this ships dark and is switched on deliberately, after
    # the operator has run `nhpc create-admin`. Turning it on with no admin in the DB
    # would lock everyone out, so serve() checks for one and refuses to start.
    auth_enabled: bool = field(default_factory=lambda: _env_bool("AUTH_ENABLED", False))

    # Extendable: add a role here and to the users.role CHECK in migration 005.
    auth_roles: str = field(default_factory=lambda: _env_str("AUTH_ROLES", "admin,officer"))

    # --- session cookie -----------------------------------------------------
    # NO DEFAULT. Used to sign the cookie; validate() refuses to run without it.
    secret_key: str = field(default_factory=lambda: _env_str("AUTH_SECRET_KEY"))

    cookie_name: str = field(default_factory=lambda: _env_str("AUTH_COOKIE_NAME", "nhpc_session"))

    # Secure=true means the browser will only ever send the cookie over HTTPS.
    # DEFAULTS TO TRUE. It may be turned off only on loopback (see validate()), because a
    # non-Secure cookie on a real interface is a session token in cleartext.
    cookie_secure: bool = field(default_factory=lambda: _env_bool("AUTH_COOKIE_SECURE", True))

    # Lax, not Strict: Strict would drop the cookie when an officer follows a link into
    # the app from an email or an intranet page, which reads as "randomly logged out".
    # The API is same-origin, so Lax gives CSRF protection without that breakage.
    cookie_samesite: str = field(default_factory=lambda: _env_str("AUTH_COOKIE_SAMESITE", "lax"))

    session_hours: int = field(default_factory=lambda: _env_int("AUTH_SESSION_HOURS", 12))

    # --- Argon2id -----------------------------------------------------------
    # OWASP's current recommendation. Memory-hard, so a GPU farm buys far less than it
    # would against bcrypt. The encoded hash carries its own params: raise these later and
    # existing hashes still verify.
    argon2_time_cost:   int = field(default_factory=lambda: _env_int("ARGON2_TIME_COST", 3))
    argon2_memory_cost: int = field(default_factory=lambda: _env_int("ARGON2_MEMORY_COST", 65536))  # 64 MiB
    argon2_parallelism: int = field(default_factory=lambda: _env_int("ARGON2_PARALLELISM", 4))

    # --- password policy (enforced SERVER-side; the browser is not a control) -
    password_min_length: int = field(default_factory=lambda: _env_int("AUTH_PASSWORD_MIN_LENGTH", 8))

    # --- brute-force defence ------------------------------------------------
    # Per-user: persisted in the DB, so restarting the process does NOT reset an
    # attacker's budget.
    max_failed_logins: int = field(default_factory=lambda: _env_int("AUTH_MAX_FAILED", 5))
    lockout_minutes:   int = field(default_factory=lambda: _env_int("AUTH_LOCKOUT_MINUTES", 15))

    # Per-IP: catches one attacker spraying MANY accounts, which per-user counters alone
    # would never see (5 tries each against 1000 accounts trips nothing).
    ip_max_attempts:   int = field(default_factory=lambda: _env_int("AUTH_IP_MAX_ATTEMPTS", 20))
    ip_window_minutes: int = field(default_factory=lambda: _env_int("AUTH_IP_WINDOW_MINUTES", 15))

    # --- bootstrap ----------------------------------------------------------
    admin_email: str = field(default_factory=lambda: _env_str("AUTH_ADMIN_EMAIL"))

    def roles(self) -> set[str]:
        return {r.strip() for r in self.auth_roles.split(",") if r.strip()}

    def validate_auth(self, host: str | None = None):
        """Errors, not warnings. The API refuses to start on any of these."""
        errs = []
        if not self.auth_enabled:
            return errs                     # nothing to validate when the feature is off

        if not self.secret_key:
            errs.append("AUTH_SECRET_KEY is required when AUTH_ENABLED=true "
                        "(generate one: python -c \"import secrets;print(secrets.token_urlsafe(48))\")")
        elif len(self.secret_key) < 32:
            errs.append("AUTH_SECRET_KEY is too short (>= 32 chars)")

        # A non-Secure cookie is a session token travelling in cleartext. Allowed ONLY on
        # loopback, where there is no wire to sniff -- so a developer can run without TLS
        # but a real deployment CANNOT ship that way by accident.
        if not self.cookie_secure:
            loopback = (host or "") in ("127.0.0.1", "localhost", "::1")
            if not loopback:
                errs.append(f"AUTH_COOKIE_SECURE=false is only permitted on loopback "
                            f"(host is {host!r}). Terminate TLS in front and leave it true.")

        if self.cookie_samesite not in ("lax", "strict", "none"):
            errs.append("AUTH_COOKIE_SAMESITE must be lax | strict | none")
        if self.cookie_samesite == "none" and not self.cookie_secure:
            errs.append("SameSite=None requires Secure (the browser will reject it otherwise)")

        if self.password_min_length < 8:
            errs.append("AUTH_PASSWORD_MIN_LENGTH must be >= 8")
        if self.argon2_memory_cost < 19456:      # OWASP floor: 19 MiB
            errs.append("ARGON2_MEMORY_COST is below the OWASP minimum (19456 KiB)")
        if "admin" not in self.roles():
            errs.append("AUTH_ROLES must include 'admin'")
        return errs
