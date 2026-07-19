"""
Build / update the entity dictionary, then extract per-record entities.

    nhpc build-entities                 # seeds + "Full (ABBR)" mining + extract (no LLM)
    nhpc build-entities --llm           # + offline LLM discovery over every document
    nhpc build-entities --only 8773     # one document (used by the watcher on upload)
    nhpc build-entities --extract-only  # re-extract records against the current dictionary

ORDER MATTERS (and is the same on a full build and on an upload):
    1. update the DICTIONARY  (seeds -> mining -> LLM)   so new entities exist
    2. EXTRACT per record     (deterministic match)      so records link to them
    3. records are now retrievable by the entity retriever

Idempotent: deterministic ids mean a re-run upserts, never duplicates. Every source is
additive -- a second run only ever ADDS entities/aliases and refreshes links.

THE LLM RUNS OFFLINE, HERE, NEVER IN THE QUERY PATH. It only DISCOVERS candidate entities;
they are validated and written to the dictionary as data. Live retrieval reads only the
deterministic dictionary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.core.db.session import connect
from nhpc_qa.core.logging import get_logger, setup as setup_logging
from nhpc_qa.entities import dictionary as D
from nhpc_qa.entities import seeds

log = get_logger("nhpc.entities.build")


# "Full Name (ABBR)" — government docs define an abbreviation on first use. The ABBR is
# 2-6 upper-case letters (optionally dotted); the Full Name is the Title-Case run before it.
_ABBR = re.compile(r"([A-Z][A-Za-z&.,'\- ]{3,60}?)\s*\(\s*([A-Z][A-Z.&\-]{1,7})\s*\)")


def seed_states(conn) -> int:
    n = 0
    for canon, aliases in seeds.STATES.items():
        _, na = D.upsert_entity(conn, canonical=canon, entity_type="state",
                                aliases=aliases, source="seed_states")
        n += 1
    for canon, aliases in seeds.ORGANIZATIONS.items():
        D.upsert_entity(conn, canonical=canon, entity_type="organization",
                        aliases=aliases, source="seed_orgs")
        n += 1
    conn.commit()
    log.info("seeded %d states + organizations", n)
    return n


def seed_projects(conn) -> int:
    """
    NHPC project names already in the DB -- from the supporting UC-projects table and from
    the diaries' own subjects. Authoritative, no guessing.
    """
    names = set()
    with conn.cursor() as cur:
        # project names captured in the supporting UC-projects tables (transposed: names are
        # in the header cells). We pull any cell that looks like "<Name> (<n>x<n> = <n> MW)".
        cur.execute("""SELECT nl_rendering FROM supporting_document_tables t
                       JOIN supporting_documents d ON d.id=t.supporting_doc_id
                       WHERE d.category='projects_progress' AND d.is_active""")
        for (nl,) in cur.fetchall():
            for m in re.finditer(r"([A-Z][A-Za-z\-]+(?:\s[IVX]+)?)\s*\(\s*\d+\s*[x×]",
                                 nl or ""):
                names.add(m.group(1).strip())
    n = 0
    for name in sorted(names):
        if len(name) >= 4:
            D.upsert_entity(conn, canonical=name, entity_type="project",
                            aliases=[name.replace("-", " "), name.replace("-", "")],
                            source="seed_projects")
            n += 1
    conn.commit()
    log.info("seeded %d project names from the DB", n)
    return n


def seed_synonyms(conn) -> int:
    """
    Load the curated context-synonym groups. Each member (except the representative) is
    rewritten to the representative at query time. Idempotent on the normalised phrase.
    """
    from nhpc_qa.entities import synonyms
    n = 0
    with conn.cursor() as cur:
        for group in synonyms.SYNONYM_GROUPS:
            if not group:
                continue
            canon = group[0]
            cid = D.canonical_id(canon)
            for phrase in group:
                cur.execute("""
                    INSERT INTO concept_synonyms (phrase_norm, canonical, concept_id, source)
                    VALUES (%s,%s,%s,'seed')
                    ON CONFLICT (phrase_norm) DO UPDATE SET
                        canonical = EXCLUDED.canonical, concept_id = EXCLUDED.concept_id
                """, (D.normalise(phrase), canon, cid))
                n += cur.rowcount
    conn.commit()
    log.info("seeded %d synonym phrase(s)", n)
    return n


def mine_abbreviations(conn) -> int:
    """
    Mine "Full Name (ABBR)" from every question + answer. Adds the ABBR as an alias of the
    Full Name -- exactly how "Himachal Pradesh (HP)" teaches the dictionary that hp -> the
    state. Deterministic; re-running adds nothing new.
    """
    added = 0
    with conn.cursor() as cur:
        cur.execute("""
            SELECT string_agg(coalesce(sq.question_text,''),' ') || ' ' ||
                   string_agg(coalesce(ag.answer_text,''),' ')
            FROM sub_questions sq
            LEFT JOIN answer_groups ag ON ag.answer_group_id = sq.answer_group_id
        """)
        text = cur.fetchone()[0] or ""

    seen = {}
    for m in _ABBR.finditer(text):
        full = re.sub(r"\s+", " ", m.group(1)).strip(" ,.-")
        abbr = m.group(2).strip()
        if len(full) < 4 or len(abbr) < 2:
            continue
        # the abbreviation's letters should plausibly come from the full name (first letters)
        initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", full)).upper()
        letters = re.sub(r"[^A-Z]", "", abbr.upper())
        if letters and letters not in initials and initials not in letters:
            continue                       # "(2024)" style false positives
        seen.setdefault(full, set()).add(abbr)

    for full, abbrs in seen.items():
        etype = ("organization" if any(w in full.lower() for w in
                 ("ministry", "commission", "authority", "corporation", "tribunal",
                  "limited", "board", "department")) else "other")
        D.upsert_entity(conn, canonical=full, entity_type=etype,
                        aliases=list(abbrs), source="abbr_mining", confidence="high")
        added += len(abbrs)
    conn.commit()
    log.info("abbr mining: %d full names, %d aliases", len(seen), added)
    return added


# ---------------------------------------------------------------------------
# LLM discovery — offline, discovers candidate entities per document
# ---------------------------------------------------------------------------
_LLM_SYSTEM = """You extract named entities from Indian parliamentary Q&A about NHPC (a
hydropower company). From the text, list the specific NAMED entities: hydro PROJECTS/power
stations (e.g. "Subansiri Lower", "Teesta-VI", "Chamera-III"), STATES/UTs, government
SCHEMES, and ORGANIZATIONS.

Rules:
- Only PROPER NOUNS actually named in the text. Not generic phrases ("hydro projects",
  "under construction", "power station") -- those are not entities.
- Keep a project's number/suffix exact: "Teesta-VI" is not "Teesta-V".
- Return STRICT JSON: [{"name": "<canonical name>", "type": "project|state|scheme|organization"}]
- No duplicates, no prose."""


def llm_discover(conn, cfg, llm, only=None) -> int:
    """Run the LLM over each document's question+answer text; add NEW entities it finds.
    Flagged needs_review + low confidence -- usable, but marked for optional QA."""
    with conn.cursor() as cur:
        if only:
            cur.execute("""SELECT d.doc_key,
                                  string_agg(coalesce(sq.question_text,''),' '),
                                  string_agg(coalesce(ag.answer_text,''),' ')
                           FROM diaries d
                           JOIN sub_questions sq ON sq.doc_key=d.doc_key
                           LEFT JOIN answer_groups ag ON ag.answer_group_id=sq.answer_group_id
                           WHERE d.question_id=%s GROUP BY d.doc_key""", (only,))
        else:
            cur.execute("""SELECT d.doc_key,
                                  string_agg(coalesce(sq.question_text,''),' '),
                                  string_agg(coalesce(ag.answer_text,''),' ')
                           FROM diaries d
                           JOIN sub_questions sq ON sq.doc_key=d.doc_key
                           LEFT JOIN answer_groups ag ON ag.answer_group_id=sq.answer_group_id
                           WHERE d.active GROUP BY d.doc_key""")
        docs = cur.fetchall()

    added = 0
    for i, (dk, q, a) in enumerate(docs, 1):
        text = f"{q or ''}\n{a or ''}".strip()
        if not text:
            continue
        try:
            raw = llm.complete_text(_LLM_SYSTEM, text[:6000], max_tokens=600, temperature=0.0)
        except Exception as e:      # noqa: BLE001 -- discovery must never break the build
            log.warning("llm_discover %s: %s", dk, type(e).__name__)
            continue
        for ent in _parse_entities(raw):
            name, etype = ent
            _, na = D.upsert_entity(conn, canonical=name, entity_type=etype,
                                    aliases=[], source="llm",
                                    needs_review=True, confidence="low")
            added += na
        conn.commit()
        if i % 50 == 0:
            log.info("llm_discover: %d/%d docs", i, len(docs))
    log.info("llm discovery: %d new alias(es) added", added)
    return added


def _parse_entities(raw):
    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    m = re.search(r"\[.*\]", s, re.S)
    try:
        arr = json.loads(m.group(0) if m else s)
    except (json.JSONDecodeError, AttributeError):
        return []
    out = []
    _VALID = {"project", "state", "scheme", "organization"}
    _BAD = {"hydro projects", "power station", "under construction", "power project",
            "hydro project", "hydroelectric project", "projects"}
    for it in arr if isinstance(arr, list) else []:
        if not isinstance(it, dict):
            continue
        name = (it.get("name") or "").strip()
        etype = (it.get("type") or "other").strip().lower()
        if len(name) < 4 or name.lower() in _BAD or etype not in _VALID:
            continue
        out.append((name, etype))
    return out


_SYN_SYSTEM = """You build a domain synonym dictionary for Indian parliamentary Q&A about
NHPC hydropower. Given text, propose groups of phrases that are INTERCHANGEABLE in this
domain -- a reply using one could use another with no change of meaning.

Examples of valid groups:
  ["under construction", "ongoing", "under execution"]
  ["commissioned", "completed", "operational"]
  ["sanctioned", "approved"]

Rules:
- ONLY genuinely interchangeable phrases. If two phrases could change the answer, do NOT
  group them (e.g. "under construction" and "commissioned" are NOT synonyms -- different
  status). Be conservative; a wrong synonym silently corrupts results.
- The FIRST phrase in each group is the most standard/common wording.
- Lowercase. Return STRICT JSON: [["phrase a","phrase b",...], ...]. No prose."""


def llm_discover_synonyms(conn, cfg, llm) -> int:
    """
    Ask the LLM for domain synonym groups over the corpus vocabulary. Added flagged
    (needs_review, source=llm) -- USABLE but marked, because a wrong synonym is higher risk
    than a wrong entity alias. Never groups a phrase already owned by a seed group.
    """
    with conn.cursor() as cur:
        cur.execute("""SELECT string_agg(DISTINCT left(question_text,300), ' | ')
                       FROM sub_questions
                       TABLESAMPLE SYSTEM (30)""")   # a sample is enough for vocabulary
        sample = cur.fetchone()[0] or ""
        cur.execute("SELECT phrase_norm FROM concept_synonyms")
        owned = {r[0] for r in cur.fetchall()}

    if not sample.strip():
        return 0
    try:
        raw = llm.complete_text(_SYN_SYSTEM, sample[:8000], max_tokens=800, temperature=0.0)
    except Exception as e:      # noqa: BLE001
        log.warning("llm_discover_synonyms: %s", type(e).__name__)
        return 0

    s = (raw or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    m = re.search(r"\[.*\]", s, re.S)
    try:
        groups = json.loads(m.group(0) if m else s)
    except (json.JSONDecodeError, AttributeError):
        return 0

    added = 0
    with conn.cursor() as cur:
        for group in groups if isinstance(groups, list) else []:
            if not isinstance(group, list) or len(group) < 2:
                continue
            canon = str(group[0]).strip().lower()
            if not canon:
                continue
            cid = D.canonical_id(canon)
            for phrase in group:
                pn = D.normalise(phrase)
                if not pn or pn in owned:      # never override a seed group
                    continue
                cur.execute("""INSERT INTO concept_synonyms
                                 (phrase_norm, canonical, concept_id, source, needs_review)
                               VALUES (%s,%s,%s,'llm',true)
                               ON CONFLICT (phrase_norm) DO NOTHING""", (pn, canon, cid))
                added += cur.rowcount
    conn.commit()
    log.info("llm synonym discovery: +%d phrase(s) (flagged for review)", added)
    return added


# ---------------------------------------------------------------------------
# EXTRACT — deterministic per-record entities against the current dictionary
# ---------------------------------------------------------------------------
def extract_records(conn, only=None) -> int:
    """
    For every sub-question, match its QUESTION and its ANSWER against the dictionary and
    store the union in sub_question_entities, tagged found_in (question|answer|both).

    Deterministic -- the SAME matcher used at query time. Re-running replaces a record's
    links cleanly (delete + insert), so it is idempotent and reflects the current dictionary.
    """
    alias_map = D.load_alias_map(conn)
    with conn.cursor() as cur:
        if only:
            cur.execute("""SELECT sq.sub_question_id, sq.doc_key, sq.question_text,
                                  ag.answer_text
                           FROM sub_questions sq
                           JOIN diaries d ON d.doc_key=sq.doc_key
                           LEFT JOIN answer_groups ag ON ag.answer_group_id=sq.answer_group_id
                           WHERE d.question_id=%s""", (only,))
        else:
            cur.execute("""SELECT sq.sub_question_id, sq.doc_key, sq.question_text,
                                  ag.answer_text
                           FROM sub_questions sq
                           LEFT JOIN answer_groups ag ON ag.answer_group_id=sq.answer_group_id""")
        rows = cur.fetchall()

    n_links = 0
    with conn.cursor() as cur:
        for sqid, dk, qt, at in rows:
            q_ents = {e["entity_id"] for e in D.match_entities(qt or "", alias_map)}
            a_ents = {e["entity_id"] for e in D.match_entities(at or "", alias_map)}
            alle = q_ents | a_ents
            # rewrite this record's links (idempotent)
            cur.execute("DELETE FROM sub_question_entities WHERE sub_question_id=%s", (sqid,))
            for eid in alle:
                found = "both" if (eid in q_ents and eid in a_ents) else (
                    "question" if eid in q_ents else "answer")
                cur.execute("""INSERT INTO sub_question_entities
                                 (sub_question_id, entity_id, doc_key, found_in)
                               VALUES (%s,%s,%s,%s)
                               ON CONFLICT (sub_question_id, entity_id) DO UPDATE
                                 SET found_in=EXCLUDED.found_in""",
                            (sqid, eid, dk, found))
                n_links += 1
    conn.commit()
    log.info("extracted entities for %d record(s): %d link(s)", len(rows), n_links)
    return n_links


def build(cfg, conn, *, use_llm=False, only=None, extract_only=False):
    """The full build, in the order that matters. Returns a summary dict."""
    before = D.dictionary_stats(conn)

    if not extract_only:
        if not only:                          # seeds are corpus-wide, not per-file
            seed_states(conn)
            seed_projects(conn)
            seed_synonyms(conn)               # context-synonym groups (query expansion)
            mine_abbreviations(conn)
        else:
            # an upload only mines its own new text for abbreviations; the curated synonym
            # seed is corpus-wide and already loaded.
            mine_abbreviations(conn)
        if use_llm:
            from nhpc_qa.core.providers import get_llm
            llm = get_llm(cfg)
            llm_discover(conn, cfg, llm, only=only)
            if not only:
                llm_discover_synonyms(conn, cfg, llm)

    # ALWAYS extract (dictionary first, then extract -> records retrievable)
    links = extract_records(conn, only=only)

    after = D.dictionary_stats(conn)
    return {
        "entities_before": before["entities"], "entities_after": after["entities"],
        "aliases_before": before["aliases"], "aliases_after": after["aliases"],
        "links": links, "by_type": after["by_type"], "needs_review": after["needs_review"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nhpc build-entities")
    ap.add_argument("--llm", action="store_true", help="offline LLM discovery (API cost)")
    ap.add_argument("--only", default=None, help="one question_id (used by the watcher)")
    ap.add_argument("--extract-only", action="store_true",
                    help="re-extract records against the current dictionary")
    args = ap.parse_args(argv)

    load_dotenv()
    setup_logging()
    cfg = Settings()
    errs = cfg.validate_all(need_db=True, need_embed=False, need_rerank=False)
    if errs:
        print("CONFIG ERROR:\n  " + "\n  ".join(errs), file=sys.stderr)
        return 2

    with connect(cfg) as conn:
        s = build(cfg, conn, use_llm=args.llm, only=args.only,
                  extract_only=args.extract_only)

    print("\n  " + "=" * 60)
    print(f"  entities : {s['entities_before']} -> {s['entities_after']}"
          f"   (+{s['entities_after'] - s['entities_before']})")
    print(f"  aliases  : {s['aliases_before']} -> {s['aliases_after']}"
          f"   (+{s['aliases_after'] - s['aliases_before']})")
    print(f"  by type  : {s['by_type']}")
    print(f"  record links written: {s['links']}")
    print(f"  flagged for review (LLM low-conf): {s['needs_review']}")
    print("  " + "=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
