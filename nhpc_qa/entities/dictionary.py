"""
The entity dictionary: normalisation, deterministic ids, store, and the deterministic
matcher used at BOTH index time and query time.

⚠️ THE MATCHER IS DETERMINISTIC AND SHARED. ⚠️
match_entities() is the ONE function that turns text into canonical entity ids, and it is
called identically for a record's question/answer at index time and for the officer's query
at query time. That is what makes "HP" and "Himachal Pradesh" resolve to the same
entity_id, which is the whole fix for the abbreviation instability.
"""

from __future__ import annotations

import re
import unicodedata

from nhpc_qa.core.logging import get_logger

log = get_logger("nhpc.entities.dict")

# aliases shorter than this are only matched with word boundaries AND uppercase intent --
# a 2-letter alias like 'ap' must not fire on the 'ap' inside 'apply'. Handled in the
# matcher by requiring word boundaries; length is a secondary guard for very short ones.
_MIN_BARE = 2


def normalise(s: str) -> str:
    """Lowercase, strip accents, collapse whitespace/punctuation-runs. The canonical form
    for storing and comparing an alias. Devanagari passes through (casefold is a no-op)."""
    s = unicodedata.normalize("NFKC", str(s or "")).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[.–—_/]+", " ", s)      # dots, dashes, slashes -> space
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def canonical_id(name: str) -> str:
    """
    DETERMINISTIC id from a canonical name. 'Himachal Pradesh' -> 'himachal_pradesh',
    'Teesta-VI' -> 'teesta_vi'. Deterministic so a re-run of the build upserts the same row
    instead of creating a duplicate.
    """
    s = normalise(name)
    s = re.sub(r"[^a-z0-9ऀ-ॿ]+", "_", s)   # keep alnum + Devanagari
    return s.strip("_") or "entity"


# ---------------------------------------------------------------------------
# STORE — upsert entities + aliases
# ---------------------------------------------------------------------------
def upsert_entity(conn, *, canonical, entity_type, aliases, source="manual",
                  needs_review=False, confidence="high"):
    """
    Add or update one canonical entity and its aliases. Idempotent on the deterministic id.

    An alias already owned by a DIFFERENT entity is LEFT ALONE (not stolen): the first
    entity to claim a surface form keeps it, because an ambiguous alias mapping to two
    entities is worse than a missing one. Returns (entity_id, n_new_aliases).
    """
    eid = canonical_id(canonical)
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO entities (entity_id, canonical, entity_type, source, needs_review,
                                  confidence)
            VALUES (%s,%s,%s,%s,%s,%s)
            ON CONFLICT (entity_id) DO UPDATE SET
                canonical    = EXCLUDED.canonical,
                entity_type  = EXCLUDED.entity_type,
                -- once reviewed/high, do not silently downgrade on a later low-conf re-add
                needs_review = entities.needs_review AND EXCLUDED.needs_review,
                confidence   = CASE WHEN entities.confidence='high' THEN 'high'
                                    ELSE EXCLUDED.confidence END,
                updated_at   = now()
        """, (eid, canonical.strip(), entity_type, source, needs_review, confidence))

        # the canonical name is itself an alias
        all_aliases = {normalise(canonical)} | {normalise(a) for a in (aliases or [])}
        all_aliases = {a for a in all_aliases if a}

        n_new = 0
        for anorm in all_aliases:
            cur.execute("""
                INSERT INTO entity_aliases (alias_norm, alias, entity_id, source)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT (alias_norm) DO NOTHING
            """, (anorm, anorm, eid, source))
            n_new += cur.rowcount
    return eid, n_new


# ---------------------------------------------------------------------------
# LOAD — the alias -> entity map, for the matcher
# ---------------------------------------------------------------------------
def load_alias_map(conn) -> dict:
    """
    {alias_norm: (entity_id, entity_type, canonical)}. Loaded once and reused. This is the
    entire dictionary the matcher needs.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.alias_norm, a.entity_id, e.entity_type, e.canonical
            FROM entity_aliases a JOIN entities e ON e.entity_id = a.entity_id
        """)
        return {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}


# ---------------------------------------------------------------------------
# MATCH — the deterministic extractor (index time AND query time)
# ---------------------------------------------------------------------------
def match_entities(text: str, alias_map: dict) -> list:
    """
    The canonical entities named in `text`. Deterministic, case-insensitive, word-boundary.

    LONGEST-ALIAS-FIRST so "himachal pradesh" wins over "himachal", and a matched span is
    consumed so one mention is not double-counted. Returns a list of
    {entity_id, entity_type, canonical, surface} in first-appearance order, de-duplicated by
    entity_id.

    This is the SAME function for a record and for a query -- that identity is the fix.
    """
    if not (text or "").strip() or not alias_map:
        return []

    norm = normalise(text)
    # match on the normalised text with spaces around it, so word boundaries are simple.
    padded = f" {norm} "

    # longest aliases first -> specific beats generic; each is a phrase match on token
    # boundaries (spaces), which normalise() has made uniform.
    hits = []
    taken = [False] * len(padded)
    for alias in sorted(alias_map, key=len, reverse=True):
        if len(alias) < _MIN_BARE:
            continue
        needle = f" {alias} "
        start = 0
        while True:
            i = padded.find(needle, start)
            if i < 0:
                break
            # the span of the alias itself (inside the padding spaces)
            s, e = i + 1, i + len(needle) - 1
            if not any(taken[s:e]):
                for k in range(s, e):
                    taken[k] = True
                eid, etype, canon = alias_map[alias]
                hits.append((i, {"entity_id": eid, "entity_type": etype,
                                 "canonical": canon, "surface": alias}))
            start = i + 1

    hits.sort(key=lambda h: h[0])
    out, seen = [], set()
    for _, h in hits:
        if h["entity_id"] not in seen:
            seen.add(h["entity_id"])
            out.append(h)
    return out


def load_synonym_map(conn) -> dict:
    """{phrase_norm: canonical_representative}. The concept-synonym rewrite table."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT phrase_norm, canonical FROM concept_synonyms")
            return dict(cur.fetchall())
    except Exception:      # noqa: BLE001 -- pre-migration is not fatal
        return {}


def apply_synonyms(text: str, synonym_map: dict) -> str:
    """
    Rewrite every context-synonym in `text` to its canonical representative, so "ongoing
    hydro projects" and "under construction hydro projects" become the SAME string and
    therefore embed identically.

    Longest phrase first (so "under construction stage" wins over "under construction"),
    word-boundary, case-insensitive. Deterministic; no LLM at query time.
    """
    if not text or not synonym_map:
        return text
    out = text
    for phrase in sorted(synonym_map, key=len, reverse=True):
        canon = synonym_map[phrase]
        if normalise(phrase) == normalise(canon):
            continue                       # the representative maps to itself; skip
        parts = [re.escape(tok) for tok in phrase.split(" ") if tok]
        if not parts:
            continue
        pat = r"\b" + r"[\s.\-]*".join(parts) + r"\b"
        out = re.sub(pat, canon, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def canonicalise_text(text: str, matched: list) -> str:
    """
    Rewrite each matched entity's SURFACE form in `text` to its canonical name, so
    "projects in HP" -> "projects in Himachal Pradesh". Used to canonicalise the QUERY before
    dense embedding + reranking, so an abbreviation and the full name embed identically.

    `matched` is the output of match_entities(text). We replace on a normalised copy is not
    enough (we must preserve the original text's non-entity words), so we do a
    case-insensitive, word-boundary replace of each surface alias with the canonical name.
    Longest surface first, so a longer alias is not partially overwritten by a shorter one.
    """
    if not text or not matched:
        return text
    out = text
    for m in sorted(matched, key=lambda x: len(x.get("surface") or ""), reverse=True):
        surface = m.get("surface") or ""
        canon = m.get("canonical") or ""
        if not surface or not canon:
            continue
        # the surface is normalised (spaces for dots/dashes); build a flexible pattern that
        # matches the original punctuation ("H.P." / "H P" / "HP" for surface "h p").
        parts = [re.escape(tok) for tok in surface.split(" ") if tok]
        if not parts:
            continue
        pat = r"\b" + r"[\s.\-]*".join(parts) + r"\b"
        out = re.sub(pat, canon, out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def dictionary_stats(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT entity_type, count(*) FROM entities GROUP BY 1 ORDER BY 2 DESC")
        by_type = dict(cur.fetchall())
        cur.execute("SELECT count(*) FROM entity_aliases")
        n_alias = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM entities WHERE needs_review")
        n_review = cur.fetchone()[0]
    return {"by_type": by_type, "aliases": n_alias, "needs_review": n_review,
            "entities": sum(by_type.values())}
