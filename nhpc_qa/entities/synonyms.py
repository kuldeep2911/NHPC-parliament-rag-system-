"""
Context-synonym groups for the NHPC / parliamentary hydropower domain.

DIFFERENT FROM ENTITY ALIASES. An entity alias is a surface variant of the SAME proper noun
("HP" == "Himachal Pradesh"). A CONCEPT SYNONYM is a set of DIFFERENT words that mean the
same thing IN THIS DOMAIN ("ongoing" == "under construction" == "under execution", when
talking about a hydro project). The embedding model reads the full sentences as ~0.91
similar already, but not identical -- so a query using one word and a query using another can
fall on opposite sides of the relevance threshold. Canonicalising every member of a group to
one representative makes the two queries embed IDENTICALLY, so they score identically and
return the same results.

⚠️ SYNONYMS ARE HIGHER-RISK THAN ALIASES. ⚠️ A wrong synonym silently changes results. So the
seed list is CURATED and conservative -- only words that are genuinely interchangeable in a
parliamentary reply about NHPC. The representative (first member) is the phrase the corpus
uses most, so canonicalising toward it moves queries INTO the corpus vocabulary.

Each group: the FIRST member is the canonical representative; the rest are rewritten to it.
"""

from __future__ import annotations

# canonical_representative -> [synonyms rewritten to it]
# The representative is chosen as the dominant corpus phrasing.
SYNONYM_GROUPS = [
    # project execution status
    ["under construction", "ongoing", "under execution", "under implementation",
     "being constructed", "under construction stage", "presently under construction"],
    ["commissioned", "completed", "operational", "made operational", "put into operation",
     "brought into operation"],
    ["under survey and investigation", "under survey & investigation",
     "survey and investigation stage", "under investigation"],
    ["pre-construction stage", "pre construction stage", "preconstruction stage"],

    # approvals / sanction
    ["sanctioned", "approved", "accorded approval", "granted approval", "cleared"],
    ["allotted", "allocated", "awarded"],

    # clearances
    ["environmental clearance", "environment clearance", "green clearance", "ec"],
    ["forest clearance", "forestry clearance", "fc"],

    # money / dues
    ["outstanding dues", "pending dues", "unpaid dues", "arrears", "outstanding payments"],
    ["cost overrun", "cost escalation", "increase in cost"],
    ["time overrun", "delay", "time escalation", "schedule overrun"],

    # generic project vocabulary
    ["hydroelectric project", "hydro electric project", "hydel project", "hydro project",
     "hydropower project", "hep"],
    ["power station", "power plant", "generating station", "power house"],
    ["installed capacity", "generation capacity", "capacity"],
    ["pumped storage project", "pumped storage plant", "psp", "pumped storage scheme"],

    # people / rehabilitation
    ["rehabilitation and resettlement", "rehabilitation & resettlement", "r&r",
     "resettlement and rehabilitation"],
    ["project affected families", "project affected persons", "displaced persons",
     "affected people", "paf"],

    # ------------------------------------------------------------------
    # Groups added from the 100-question harness (each fixes a MEASURED
    # paraphrase divergence — the group id from results_before.json noted)
    # ------------------------------------------------------------------
    # g13: "hydropower potential" 17 results vs "hydel power potential" 11 vs
    # "hydro electric potential" 18. The project-suffixed group above only covers
    # "...project"; these are the BARE energy terms.
    # NOTE the potential-group members are LONGER than the bare-power phrases, so
    # longest-first matching rewrites "hydel power potential" as one unit -> one canonical,
    # instead of "hydel power"->"hydro power" leaving a half-rewritten string.
    ["hydro power potential", "hydropower potential", "hydel power potential",
     "hydro electric potential", "hydroelectric potential", "hydel potential",
     "hydro potential"],
    ["hydro power", "hydropower", "hydel power", "hydro electric power",
     "hydroelectric power", "hydel"],

    # g14: rehabilitation vs resettlement of displaced families (jaccard 0.0).
    # In this domain the two words are used interchangeably for the same programme.
    ["rehabilitation", "resettlement"],

    # g06: expenditure vs spending (jaccard 0.644)
    ["expenditure", "spending", "amount spent", "expenses incurred"],

    # g03/g05-style: "current status" vs "present status" vs "status" split result sets.
    # Longest-first matching rewrites the qualified forms down to the bare ask.
    ["status", "current status", "present status", "latest status", "current position",
     "present position"],

    # g05: vacant posts vs vacancies (jaccard 0.533)
    ["vacant posts", "vacancies", "vacant positions", "posts lying vacant",
     "unfilled posts"],

    # g10: DPR vs detailed project report (4 vs 11 results). Not a proper noun ->
    # concept synonym, not an entity alias.
    ["detailed project reports", "detailed project report", "dprs", "dpr"],

    # g12: DISCOM phrasings (jaccard 0.538)
    ["discoms", "discom", "distribution companies", "power distribution companies",
     "distribution licensees"],

    # g08: recruitment phrasings ("new recruitments" vs "recruitment", jaccard 0.189).
    ["recruitment", "recruitments", "new recruitments", "new recruitment",
     "fresh recruitments", "fresh recruitment", "hiring"],

    # g13 residual: "in the country" == "in India" in a Government-of-India corpus. Only
    # the full phrases — bare "country" is left alone ("country-wise" must not become
    # "india-wise").
    ["india", "the country", "the nation"],
]
