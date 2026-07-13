-- 005 — authentication: users, sessions, auth audit.
--
-- PURELY ADDITIVE. Touches nothing in phases 1-4: no diaries, no sub_questions, no
-- embeddings, no index. Auth wraps the API; it does not reach into the pipeline.
--
-- The identity seam the whole app already consumes is {user_id, user_role}. Before this
-- migration those came from client-supplied headers (self-asserted, therefore worthless
-- without a proxy in front). After it they come from a session cookie -> this table.
-- Endpoint authorization (rbac.check) is UNCHANGED -- only the source of identity moves.

-- Case-insensitive email, enforced by the type rather than by remembering to .lower()
-- at every call site. 'Ops@NHPC.in' and 'ops@nhpc.in' are the same user, and the UNIQUE
-- constraint knows it.
CREATE EXTENSION IF NOT EXISTS citext;


-- ---------------------------------------------------------------------------
-- USERS
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    user_id              bigserial PRIMARY KEY,
    email                citext      NOT NULL UNIQUE,

    -- Argon2id (OWASP's current recommendation; memory-hard). The encoded hash carries
    -- its OWN parameters, so the cost can be raised later and existing hashes still
    -- verify. NEVER a plaintext password, NEVER a fast hash (md5/sha1/sha256).
    password_hash        text        NOT NULL,

    -- Extendable by design: add a role here and to AUTH_ROLES. The app authorizes on
    -- role, so a new role needs no code change.
    role                 text        NOT NULL CHECK (role IN ('admin', 'officer')),

    is_active            boolean     NOT NULL DEFAULT true,

    -- Set on create and on admin reset. Enforced in the AUTH DEPENDENCY, not the UI:
    -- a user in this state can reach ONLY change-password / me / logout.
    must_change_password boolean     NOT NULL DEFAULT true,

    -- Brute-force defence. In the DB, not in memory, so it SURVIVES A RESTART --
    -- otherwise bouncing the process resets an attacker's budget.
    failed_login_count   int         NOT NULL DEFAULT 0,
    locked_until         timestamptz,

    -- NULL only for the bootstrap admin, who by definition has no creator.
    created_by           bigint      REFERENCES users(user_id) ON DELETE SET NULL,

    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    last_login_at        timestamptz
);

-- Partial: there is normally exactly one active admin, and this is the hot lookup for
-- create-admin's "does an admin already exist?" check.
CREATE INDEX IF NOT EXISTS idx_users_active_admin
    ON users (role) WHERE is_active AND role = 'admin';


-- ---------------------------------------------------------------------------
-- SESSIONS
-- ---------------------------------------------------------------------------
-- Server-side sessions, deliberately NOT stateless JWTs.
--
-- The requirements include "deactivate a user", "force a password change" and "logout
-- invalidates". A JWT cannot do any of those: it stays valid until it expires, so a
-- deactivated admin would keep their access. Revoking a JWT means a server-side denylist
-- -- which is a session table with extra steps. This is a single on-prem server, so
-- JWT's one real advantage (no shared state between nodes) buys nothing here.
CREATE TABLE IF NOT EXISTS sessions (
    session_id   bigserial   PRIMARY KEY,

    -- ⚠️ THE SHA-256 OF THE COOKIE VALUE, NEVER THE COOKIE VALUE ITSELF.
    -- The cookie holds 256 bits of secrets.token_urlsafe() entropy. If this table ever
    -- leaks, the attacker holds hashes, not live sessions -- exactly as with passwords.
    -- SHA-256 (not Argon2) is right HERE and only here: the input is already
    -- high-entropy random, so there is nothing to brute-force, and this is verified on
    -- every single request -- a slow hash would be a self-inflicted DoS.
    token_hash   text        NOT NULL UNIQUE,

    user_id      bigint      NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,

    created_at   timestamptz NOT NULL DEFAULT now(),
    expires_at   timestamptz NOT NULL,
    last_seen_at timestamptz NOT NULL DEFAULT now(),

    -- Set on logout, on a password change (all other sessions), and on deactivation.
    -- Rows are KEPT rather than deleted, so "when was this session killed, and why?"
    -- stays answerable.
    revoked_at   timestamptz,

    ip           text,
    user_agent   text
);

CREATE INDEX IF NOT EXISTS idx_sessions_user  ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_alive ON sessions (expires_at)
    WHERE revoked_at IS NULL;


-- ---------------------------------------------------------------------------
-- AUTH AUDIT
-- ---------------------------------------------------------------------------
-- Separate from query_audit / file_access_audit on purpose. Different shape, different
-- retention, different reader: "who logged in and who created whom" is a security
-- question; "what did an officer search for" is a usage question. Cramming both into one
-- table makes both harder to query.
--
-- The EMAIL is stored alongside the id because a purged user must not turn their own
-- audit trail into a row of NULLs.
CREATE TABLE IF NOT EXISTS auth_audit (
    id             bigserial   PRIMARY KEY,
    event          text        NOT NULL CHECK (event IN (
                       'login_success', 'login_failed', 'login_locked', 'logout',
                       'user_created', 'password_changed', 'password_reset',
                       'user_deactivated', 'user_reactivated', 'admin_bootstrapped',
                       'access_denied')),

    actor_user_id  bigint      REFERENCES users(user_id) ON DELETE SET NULL,
    actor_email    text,                       -- kept even if the user is later removed
    target_user_id bigint      REFERENCES users(user_id) ON DELETE SET NULL,
    target_email   text,

    success        boolean     NOT NULL,       -- FAILURES ARE AUDITED TOO
    reason         text,                       -- never a password, never a token
    ip             text,
    user_agent     text,
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_auth_audit_created ON auth_audit (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_auth_audit_actor   ON auth_audit (actor_email);
CREATE INDEX IF NOT EXISTS idx_auth_audit_event   ON auth_audit (event, created_at DESC);
