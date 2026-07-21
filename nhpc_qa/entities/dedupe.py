"""
Entity-dictionary consolidation — merge near-duplicate entities deterministically.

WHY. LLM discovery + abbreviation mining created MANY entities for one real thing:
"Subansiri Lower HE Project", "Subansiri Lower Project", "Subansiri Lower Hydroelectric
Project", "Lower Subansiri Hydroelectric Project", "Subansiri Lower HE Poject" (OCR typo)
— six rows for ONE dam. That fragments retrieval (records linked to different variants do
not join on entity_id) and canonicalisation (two phrasings of one query canonicalise to
DIFFERENT strings, splitting the result sets — measured on the 100-question harness).

HOW (deterministic, no LLM):
  1. MERGE KEY = the entity name, lowercased, minus GENERIC suffix tokens (project, hep,
     he, hydroelectric, power, station, scheme, mpp, mw + bare numbers), with known OCR
     typos repaired, as an ORDER-INSENSITIVE token set. "Lower Subansiri Hydroelectric
     Project" and "Subansiri Lower HE Poject" both key to {lower, subansiri}.
  2. Distinguishing tokens SURVIVE: "Subansiri Lower" {lower,subansiri} never merges with
     "Subansiri Middle" {middle,subansiri} or bare "Subansiri" {subansiri}.
  3. SURVIVOR = the entity with the most sub_question_entities links (the corpus's
     dominant form); ties break to the longest canonical name (most explicit).
  4. Losers' aliases and links are REPOINTED to the survivor, then the losers are deleted.
     Idempotent: a second run finds nothing to merge.

Only entities of the SAME entity_type are merged — a project never merges with a state.

    python -m nhpc_qa.entities.dedupe --dry-run     # show what would merge
    python -m nhpc_qa.entities.dedupe               # do it
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

from nhpc_qa.core.logging import get_logger
from nhpc_qa.entities.dictionary import normalise

log = get_logger("nhpc.entities.dedupe")

# Generic tokens that describe WHAT KIND of thing it is, not WHICH one. Dropped from the
# merge key. Includes plural forms and the OCR typos seen in the live dictionary.
_GENERIC = {
    "project", "projects", "poject", "pojects",          # + the observed OCR typo
    "hydroelectric", "hydro", "electric", "hydel", "hydropower",
    "he", "hep", "heps", "mpp", "psp",
    "power", "station", "stations", "plant", "plants", "scheme", "schemes",
    "stage", "limited", "ltd", "mw",
}
_NUM = re.compile(r"^[\d,.]+$")
_YEAR = re.compile(r"^(19|20)\d\d$")     # years DISTINGUISH: "EIA Notification 2006" and
#                                          "... 2025" are DIFFERENT documents; a capacity
#                                          figure ("2000 MW") pays the small price of
#                                          looking like a year and not merging — correct
#                                          beats complete here.
_ROMAN = re.compile(r"^(i{1,3}|iv|v|vi{0,3}|ix|x)$")   # stage numbers DISTINGUISH — keep


def merge_key(canonical: str) -> frozenset:
    """Order-insensitive distinguishing-token set. Empty set = never merged."""
    toks = normalise(canonical).replace("(", " ").replace(")", " ").split()
    kept = []
    for t in toks:
        if t in _GENERIC:
            continue
        if _NUM.match(t) and not _YEAR.match(t):
            continue
        kept.append(t)
    return frozenset(kept)


def plan(conn):
    """Compute the merge plan: {survivor_id: [loser_ids]}. Read-only."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.entity_id, e.canonical, e.entity_type,
                   (SELECT count(*) FROM sub_question_entities s
                     WHERE s.entity_id = e.entity_id) AS links
            FROM entities e
        """)
        rows = cur.fetchall()

    buckets = defaultdict(list)
    for eid, canonical, etype, links in rows:
        key = merge_key(canonical)
        if not key:
            continue                       # nothing distinguishing left — never merge
        buckets[(etype, key)].append({"id": eid, "canonical": canonical, "links": links})

    merges = {}
    for (_etype, _key), members in buckets.items():
        if len(members) < 2:
            continue
        members.sort(key=lambda m: (-m["links"], -len(m["canonical"])))
        survivor, losers = members[0], members[1:]
        merges[survivor["id"]] = {
            "survivor": survivor,
            "losers": losers,
        }
    return merges


def apply(conn, merges) -> int:
    """Repoint aliases + links to the survivor, delete the losers. Returns rows merged."""
    n = 0
    with conn.cursor() as cur:
        for sid, m in merges.items():
            for loser in m["losers"]:
                lid = loser["id"]
                # aliases: repoint, except where the survivor already owns that surface
                cur.execute("""
                    UPDATE entity_aliases a SET entity_id = %s
                    WHERE a.entity_id = %s
                      AND NOT EXISTS (SELECT 1 FROM entity_aliases b
                                      WHERE b.alias_norm = a.alias_norm
                                        AND b.entity_id = %s)
                """, (sid, lid, sid))
                cur.execute("DELETE FROM entity_aliases WHERE entity_id = %s", (lid,))
                # sub-question links: repoint, dedup on the (sub_question, entity) pair
                cur.execute("""
                    UPDATE sub_question_entities s SET entity_id = %s
                    WHERE s.entity_id = %s
                      AND NOT EXISTS (SELECT 1 FROM sub_question_entities t
                                      WHERE t.sub_question_id = s.sub_question_id
                                        AND t.entity_id = %s)
                """, (sid, lid, sid))
                cur.execute("DELETE FROM sub_question_entities WHERE entity_id = %s", (lid,))
                cur.execute("DELETE FROM entities WHERE entity_id = %s", (lid,))
                n += 1
    conn.commit()
    return n


def main(argv=None):
    from nhpc_qa.config import Settings, load_dotenv
    from nhpc_qa.core.db.session import connect

    ap = argparse.ArgumentParser(description="Merge near-duplicate dictionary entities")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv()
    cfg = Settings()
    with connect(cfg) as conn:
        merges = plan(conn)
        total_losers = sum(len(m["losers"]) for m in merges.values())
        print(f"merge groups: {len(merges)}   entities to fold in: {total_losers}")
        for sid, m in sorted(merges.items()):
            losers = ", ".join(f"{L['canonical']!r}({L['links']})" for L in m["losers"])
            print(f"  KEEP {m['survivor']['canonical']!r}({m['survivor']['links']})  <-  {losers}")
        if args.dry_run:
            print("\nDRY RUN — nothing changed")
            return 0
        n = apply(conn, merges)
        print(f"\nmerged {n} duplicate entities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
