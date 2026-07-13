"""
Password hashing and policy. Argon2id, via argon2-cffi.

WHY ARGON2id AND NOT sha256/md5:
A fast hash is the wrong tool by construction. The whole point of a password hash is to be
SLOW and MEMORY-HARD, so that an attacker holding the dump cannot test billions of
candidates per second on a GPU. SHA-256 is designed to be fast -- which is exactly the
property you do not want here. Argon2id is OWASP's current recommendation and is
memory-hard, so a GPU farm buys far less advantage than it would against bcrypt.

The encoded hash carries its own parameters:

    $argon2id$v=19$m=65536,t=3,p=4$<salt>$<digest>

so the cost can be RAISED later and every existing hash still verifies. verify() reports
when a hash is stale and the caller re-hashes on the next successful login.

Salting is not a decision we make: argon2-cffi generates a fresh random salt per hash.
Two users with the same password get different hashes.
"""

from __future__ import annotations

import re
import secrets
import string

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError


def _hasher(cfg) -> PasswordHasher:
    return PasswordHasher(
        time_cost=cfg.argon2_time_cost,
        memory_cost=cfg.argon2_memory_cost,
        parallelism=cfg.argon2_parallelism,
    )


def hash_password(cfg, password: str) -> str:
    """Argon2id hash. The salt is generated internally, per call."""
    return _hasher(cfg).hash(password)


def verify_password(cfg, password: str, encoded: str) -> tuple[bool, bool]:
    """
    Returns (ok, needs_rehash).

    Never raises on a bad password -- a wrong password is an expected event, not an
    exception. A malformed/legacy hash also returns False rather than exploding: the
    caller must treat it as a failed login, not a 500.
    """
    ph = _hasher(cfg)
    try:
        ph.verify(encoded, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False, False
    # The stored hash used weaker params than we now require -> upgrade it silently on
    # this successful login, while we have the plaintext in hand.
    try:
        return True, ph.check_needs_rehash(encoded)
    except InvalidHashError:
        return True, False


# ---------------------------------------------------------------------------
# A CONSTANT-TIME-ISH DUMMY
# ---------------------------------------------------------------------------
# When the email does not exist we still burn a real Argon2 verify against this hash.
#
# Without it, login for an unknown user returns in ~0ms while login for a known user takes
# ~50ms -- and that timing difference is a working USER-ENUMERATION ORACLE. An attacker
# does not need the error message to differ; the clock tells them. Returning a generic
# "invalid email or password" while replying three times faster for unknown accounts would
# be security theatre.
_DUMMY_CACHE: dict = {}


def burn_dummy_verify(cfg) -> None:
    """Spend the same work verifying a throwaway hash, so a missing account is not
    detectable by how fast we say no."""
    key = (cfg.argon2_time_cost, cfg.argon2_memory_cost, cfg.argon2_parallelism)
    encoded = _DUMMY_CACHE.get(key)
    if encoded is None:
        encoded = _DUMMY_CACHE[key] = hash_password(cfg, "dummy-password-not-a-secret")
    verify_password(cfg, "definitely-wrong", encoded)


# ---------------------------------------------------------------------------
# POLICY — enforced SERVER-side. The browser is a convenience, never a control.
# ---------------------------------------------------------------------------
_COMMON = {
    "password", "password1", "passw0rd", "qwerty", "letmein", "welcome", "admin",
    "administrator", "changeme", "abc123", "iloveyou", "monkey", "dragon",
    "123456", "12345678", "123456789", "1234567890", "111111", "000000",
    "nhpc", "nhpc123", "nhpc@123", "parliament", "officer", "secret",
}


def check_policy(cfg, password: str, email: str | None = None) -> list[str]:
    """
    Returns a list of reasons the password is unacceptable ([] = acceptable).

    Length is weighted far above symbol-juggling, because it genuinely matters more: a
    long passphrase beats 'P@ssw0rd!' by an enormous margin despite looking less
    'complex'. We ask for 12+ chars and three of the four character classes, reject known
    common passwords, and reject a password that simply echoes the email.
    """
    errs = []
    pw = password or ""

    if len(pw) < cfg.password_min_length:
        errs.append(f"must be at least {cfg.password_min_length} characters")
    if len(pw) > 200:
        errs.append("must be at most 200 characters")   # bound the Argon2 work per request

    classes = sum(bool(re.search(p, pw)) for p in
                  (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    if classes < 3:
        errs.append("must mix at least three of: lowercase, uppercase, digits, symbols")

    low = pw.lower()
    if low in _COMMON:
        errs.append("is a commonly used password")
    for c in _COMMON:
        if len(c) >= 6 and c in low:
            errs.append(f"contains the common word '{c}'")
            break

    if email:
        local = email.split("@")[0].lower()
        if local and len(local) >= 3 and local in low:
            errs.append("must not contain your email address")

    if pw and len(set(pw)) <= 4:
        errs.append("has too little variety (too few distinct characters)")

    return errs


# ---------------------------------------------------------------------------
# GENERATED PASSWORDS — for `nhpc create-admin` and admin-issued temporaries
# ---------------------------------------------------------------------------
# `secrets`, never `random`. random is a Mersenne Twister seeded from the clock: observe
# enough output and you can predict the rest. secrets is the CSPRNG.
_ALPHABET = string.ascii_lowercase + string.ascii_uppercase + string.digits + "!@#$%^&*-_=+?"


def generate_password(length: int = 20) -> str:
    """A cryptographically strong random password that satisfies check_policy()."""
    while True:
        pw = "".join(secrets.choice(_ALPHABET) for _ in range(length))
        # Loop until it happens to satisfy the policy -- rather than FORCING one char from
        # each class at a fixed position, which would leak structure to an attacker who
        # knows we generated it.
        if (re.search(r"[a-z]", pw) and re.search(r"[A-Z]", pw)
                and re.search(r"\d", pw) and re.search(r"[^A-Za-z0-9]", pw)
                and len(set(pw)) > 8):
            return pw
