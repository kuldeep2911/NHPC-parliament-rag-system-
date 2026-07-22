"""
`nhpc create-admin` — the one-time first-run bootstrap.

    nhpc create-admin --email ops@nhpc.in

Creates the single administrator, GENERATES a cryptographically strong password, and
PRINTS IT ONCE. Only the Argon2id hash is stored.

THE PASSWORD IS NEVER:
  * hardcoded anywhere,
  * written to a migration, a config file, or the repo,
  * passed to log.info() / the structured logger,
  * stored in the database in any recoverable form.

It goes to STDOUT, once, for a human to read and put in a password manager. If they lose
it, there is no recovery path but another admin or a direct SQL reset -- which is the
correct property for a credential: unrecoverable by design.

IDEMPOTENT: if an active admin already exists this refuses and changes nothing, so
re-running it in a deploy script cannot silently mint a second administrator.
"""

from __future__ import annotations

import logging
import os
import sys

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.core.db.session import connect
from nhpc_qa.api.security import passwords, users

log = logging.getLogger("nhpc.auth")

_RULE = "─" * 68


def create_admin(email: str | None = None, password: str | None = None) -> int:
    load_dotenv()
    cfg = Settings()

    email = (email or cfg.admin_email or "").strip().lower()
    if not email or "@" not in email:
        print("ERROR: an admin email is required.\n"
              "       nhpc create-admin --email you@nhpc.in\n"
              "       (or set AUTH_ADMIN_EMAIL in .env)", file=sys.stderr)
        return 2

    errs = cfg.validate_all(need_db=True, need_embed=False, need_rerank=False)
    # AUTH_SECRET_KEY is only *required* once auth is switched on -- but bootstrapping an
    # admin before setting it is the normal order, so only DB errors are fatal here.
    fatal = [e for e in errs if "DSN" in e or "DB" in e.upper()]
    if fatal:
        print("CONFIG ERROR:\n  " + "\n  ".join(fatal), file=sys.stderr)
        return 2

    with connect(cfg) as conn:
        # Idempotent. Refuse rather than create a second admin.
        if users.admin_exists(conn):
            print("An active administrator already exists. Nothing to do.\n"
                  "To issue a new password for them, use the admin UI, or (if locked out)\n"
                  "reset it directly in the database.", file=sys.stderr)
            return 1

        # A password may be supplied explicitly, OR via AUTH_ADMIN_PASSWORD in .env, so an
        # automated deploy creates a login-ready admin without an operator fishing a
        # generated one out of the logs. If neither is given, we generate and print one.
        if password is None:
            password = (os.environ.get("AUTH_ADMIN_PASSWORD") or "").strip() or None
        generated = password is None
        if generated:
            password = passwords.generate_password()
        else:
            problems = passwords.check_policy(cfg, password, email=email)
            if problems:
                print("ERROR: admin password " + "; ".join(problems), file=sys.stderr)
                return 2

        # Force a first-login change ONLY when we generated the password (it was printed to
        # a terminal, which gets scrolled back / screenshotted / shoulder-surfed). If the
        # operator supplied their own via AUTH_ADMIN_PASSWORD, it was never displayed and
        # they chose it deliberately, so let them log in with it directly.
        uid = users.create_user(
            conn, cfg, email, password, "admin",
            created_by=None,          # by definition, nobody created the first admin
            must_change=generated,
            actor=None,
        )
        users.audit(conn, "admin_bootstrapped", success=True,
                    target={"user_id": uid, "email": email},
                    reason="first-run bootstrap")

    # ---- the ONE time this value is ever displayed --------------------------
    # print(), not the logger: a log line ends up in journald, in a file, and quite
    # possibly in a log aggregator. This must go to the operator's terminal and nowhere
    # else.
    print()
    print(_RULE)
    print("  ADMINISTRATOR CREATED")
    print(_RULE)
    print(f"  email     {email}")
    if generated:
        print(f"  password  {password}")
        print()
        print("  ⚠  This password is shown ONCE and is NOT recoverable.")
        print("     Save it in a password manager NOW.")
        print("     Only its Argon2id hash is stored in the database.")
    else:
        print("  password  (the one you supplied — not echoed)")
    print()
    if generated:
        print("  You will be required to change it on first login.")
    else:
        print("  Log in with this password. (Change it anytime from the account menu.)")
    print(_RULE)
    print()
    print("  Next:  set AUTH_ENABLED=true and AUTH_SECRET_KEY in .env, then `nhpc serve`")
    print()
    return 0


def reset_password(email: str, password: str | None = None) -> int:
    """
    `nhpc reset-password --email X` — the BREAK-GLASS path.

    An admin who has forgotten their password, or been locked out, cannot use the admin UI
    to fix it: they cannot get in. Without this the only recovery is hand-written SQL
    against the users table, which is exactly the kind of thing people get wrong under
    pressure (and which nobody audits).

    Requires shell access on the server, which is the authorisation: if you are already
    root on the box, you can read the database anyway.

    Generates a new password, prints it ONCE, re-arms must_change_password, clears any
    lockout, and REVOKES EVERY LIVE SESSION for that user -- because the usual reason to
    run this is a suspected compromise, and leaving the attacker's session alive would
    make the reset pointless. Audited as password_reset.
    """
    load_dotenv()
    cfg = Settings()
    email = (email or "").strip().lower()

    with connect(cfg) as conn:
        user = users.get_by_email(conn, email)
        if user is None:
            print(f"ERROR: no user with email {email!r}", file=sys.stderr)
            return 1

        generated = password is None
        if generated:
            password = passwords.generate_password()
        else:
            problems = passwords.check_policy(cfg, password, email=email)
            if problems:
                print("ERROR: password " + "; ".join(problems), file=sys.stderr)
                return 2

        users.set_password(conn, cfg, user["user_id"], password,
                           must_change=True, revoke_sessions_except=None)
        users.audit(conn, "password_reset", success=True,
                    target={"user_id": user["user_id"], "email": email},
                    reason="CLI break-glass reset; all sessions revoked")

    print()
    print(_RULE)
    print("  PASSWORD RESET")
    print(_RULE)
    print(f"  email     {email}")
    print(f"  role      {user['role']}")
    if generated:
        print(f"  password  {password}")
        print()
        print("  ⚠  Shown ONCE. Not recoverable. Save it now.")
    print()
    print("  All of this user's sessions were signed out.")
    print("  They must change this password on their next sign-in.")
    print(_RULE)
    print()
    return 0


def deactivate(email: str) -> int:
    """`nhpc deactivate-user --email X` — disable an account and kill its sessions.

    The same action the admin UI performs, available from the shell. Refuses to disable
    the last active admin, which would leave nobody able to manage users."""
    load_dotenv()
    cfg = Settings()
    email = (email or "").strip().lower()

    with connect(cfg) as conn:
        user = users.get_by_email(conn, email)
        if user is None:
            print(f"ERROR: no user with email {email!r}", file=sys.stderr)
            return 1
        if user["role"] == "admin":
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM users WHERE role='admin' AND is_active")
                if cur.fetchone()[0] <= 1 and user["is_active"]:
                    print("ERROR: that is the last active administrator. Create another "
                          "admin first, or you will lock yourself out of user management.",
                          file=sys.stderr)
                    return 1

        users.set_active(conn, user["user_id"], False)      # revokes live sessions too
        users.audit(conn, "user_deactivated", success=True,
                    target={"user_id": user["user_id"], "email": email},
                    reason="CLI deactivation; sessions revoked")

    print(f"deactivated {email} — signed out of every session. The account and its audit "
          f"history are retained.")
    return 0
