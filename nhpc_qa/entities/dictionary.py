"""
Entity dictionary: normalisation, deterministic ids, storage, and the matcher that runs at
both index time and query time.

The matcher is deterministic and shared. match_entities() is the single function that turns
text into canonical entity ids, called identically for a record at index time and for the
officer's query at query time. That shared path is what makes "HP" and "Himachal Pradesh"
resolve to the same entity_id — the fix for abbreviation instability.
"""

from __future__ import annotations

import re
import unicodedata

from nhpc_qa.core.logging import get_logger

log = get_logger("nhpc.entities.dict")

# Aliases shorter than this match only on word boundaries, so a 2-letter alias like 'ap'
# never fires inside 'apply'. Length is a secondary guard for very short aliases.
_MIN_BARE = 2


def normalise(s: str) -> str:
    """
    Canonical form of an alias: lowercase, accents stripped, whitespace/punctuation collapsed.
    Used for both storing and comparing. Devanagari passes through unchanged.
    """
    s = unicodedata.normalize("NFKC", str(s or "")).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[.–—_/]+", " ", s)      # dots/dashes/slashes -> space
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def canonical_id(name: str) -> str:
    """
    Deterministic id from a canonical name: 'Himachal Pradesh' -> 'himachal_pradesh',
    'Teesta-VI' -> 'teesta_vi'. Deterministic so a rebuild upserts the same row instead of
    creating a duplicate.
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

    An alias already owned by a different entity is left alone: the first entity to claim a
    surface form keeps it, since an ambiguous alias is worse than a missing one.
    Returns (entity_id, n_new_aliases).
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
                -- once high-confidence, a later low-confidence re-add must not downgrade it
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
    Canonical entities named in `text`. Deterministic, case-insensitive, word-boundary.

    Longest alias first, so "himachal pradesh" wins over "himachal", and a matched span is
    consumed so one mention is not double-counted. Returns {entity_id, entity_type,
    canonical, surface} in first-appearance order, de-duplicated by entity_id.

    Same function for a record and for a query — that shared path is the fix.
    """
    if not (text or "").strip() or not alias_map:
        return []

    norm = normalise(text)
    # Pad with spaces so word boundaries are just space-delimited substring matches.
    padded = f" {norm} "

    # Longest aliases first so specific beats generic; each is a phrase match on the
    # space-delimited boundaries that normalise() produced.
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


def apply_synonyms(text: str, synonym_map: dict, protected: list | None = None) -> str:
    """
    Rewrite every context-synonym in `text` to its canonical representative, so "ongoing
    hydro projects" and "under construction hydro projects" become the same string and embed
    identically. Longest phrase first, word-boundary, case-insensitive; no LLM at query time.

    `protected` holds canonical entity names already placed in the text; they are shielded so
    a synonym like "hydroelectric"->"hydro" cannot rewrite the inside of "Subansiri Lower
    Hydroelectric Project". Protected spans become placeholders during the rewrite, restored
    after.
    """
    if not text or not synonym_map:
        return text
    out = text

    # Shield the canonical entity names from the synonym regexes.
    placeholders = {}
    for i, phrase in enumerate(sorted(set(protected or []), key=len, reverse=True)):
        if not (phrase or "").strip():
            continue
        token = f"⟦E{i}⟧"          # ⟦E0⟧ — never occurs in real queries
        pat = re.compile(re.escape(phrase), re.IGNORECASE)
        if pat.search(out):
            out = pat.sub(token, out)
            placeholders[token] = phrase

    for phrase in sorted(synonym_map, key=len, reverse=True):
        canon = synonym_map[phrase]
        if normalise(phrase) == normalise(canon):
            continue                       # the representative maps to itself; skip
        parts = [re.escape(tok) for tok in phrase.split(" ") if tok]
        if not parts:
            continue
        pat = r"\b" + r"[\s.\-]*".join(parts) + r"\b"
        out = re.sub(pat, canon, out, flags=re.IGNORECASE)

    for token, phrase in placeholders.items():
        out = out.replace(token, phrase)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# FILLER STRIP — low-information wrappers that change the embedding but not the ask
# ---------------------------------------------------------------------------
# Measured on the 100-question harness: "all hydro projects in HP" returned 1 result while
# "hydro projects in HP" returned 3; "list of ongoing..." vs "ongoing..." similarly split.
# These wrappers carry no retrieval intent — an officer asking "all X" wants X. Stripping
# them deterministically makes every wrapped phrasing embed EXACTLY like the bare one.
#
# ⚠️ CONSERVATIVE BY DESIGN. Only phrases that are pure request-wrappers are here. Words
# that narrow meaning ("new", "pending", "current") are NOT stripped — they change the ask.
_FILLER_LEADING = [
    # request wrappers an officer types before the real question
    "please provide", "kindly provide", "provide", "give me", "give",
    "list of all", "a list of", "list of", "list",
    "details of all", "the details of", "details of", "details regarding",
    "information about", "information on", "information regarding",
    "what is the", "what are the", "what is", "what are",
    "tell me about", "tell me",
    # "number of X" asks for X's count — the count lives in the SAME answers X does, so
    # for retrieval the wrapper only shifts the embedding ("number of vacant posts in
    # NHPC" split from "vacant posts in NHPC" on the harness)
    "the total number of", "total number of", "the number of", "number of",
]
# "all projects in J&K" == "projects in J&K"; articles never carry retrieval intent and
# their presence measurably split paraphrase sets ("status of the X" vs "status of X").
_FILLER_ANYWHERE = ["all", "the", "an", "a"]

_LEAD_RE = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in
                       sorted(_FILLER_LEADING, key=len, reverse=True)) + r")\s+",
    re.IGNORECASE)
_ANY_RES = [re.compile(r"\b" + re.escape(w) + r"\b\s*", re.IGNORECASE)
            for w in _FILLER_ANYWHERE]


def strip_filler(text: str) -> str:
    """
    Remove request-wrapper filler so paraphrases collapse to one canonical ask.
    "list of all ongoing projects in J&K" -> "ongoing projects in J&K".

    Applied AFTER entity canonicalisation and synonyms (so it never eats part of an entity
    name) and ONLY for retrieval-side processing — the officer's original text is preserved
    everywhere it is displayed. Never empties the query: if stripping would leave nothing,
    the original text is returned.
    """
    if not (text or "").strip():
        return text
    out = text.strip()
    # peel leading wrappers repeatedly ("please provide the details of ...")
    for _ in range(4):
        new = _LEAD_RE.sub("", out).strip()
        if new == out:
            break
        out = new
    for pat in _ANY_RES:
        out = pat.sub("", out)
    out = re.sub(r"\s+", " ", out).strip(" ,.:;-")
    return out if out else text


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
        # the surface is normalised (spaces for dots/dashes, '&' -> ' and '); build a
        # flexible pattern that matches the original punctuation: "H.P." / "H P" / "HP" for
        # surface "h p", and "J&K" / "J and K" / "JK" for surface "j and k". The token
        # "and" is matched as (?:and|&)? because normalise() created it FROM an ampersand —
        # the original text may hold "and", "&", or nothing at all.
        toks = [t for t in surface.split(" ") if t]
        if not toks:
            continue
        parts = [r"(?:and|&)?" if t == "and" else re.escape(t) for t in toks]
        pat = r"\b" + r"[\s.\-&]*".join(parts) + r"\b"
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
