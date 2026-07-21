"""
The production-readiness QUESTION DATASET — 100 queries, grounded in the real corpus.

Design (why these questions):
  * PARAPHRASE GROUPS  — the same intent phrased with tiny differences (add "all", add
    "list of", abbreviation vs full name, synonym swap). Every member of a group SHOULD
    return the same final result set; divergence = the instability the user reported
    ("all projects in J&K" vs "projects in J&K" gave different results).
  * LENGTH MIX         — 1-line, 2-line and 3-line questions in real parliamentary phrasing,
    because officers paste long multi-part questions, not keywords.
  * HINDI              — cross-lingual retrieval is a core capability.
  * BOILERPLATE BAIT   — stock phrases shared by hundreds of answers; the system must
    return ZERO results for these, not confident noise.
  * OUT-OF-DOMAIN      — topics with no corpus precedent; the honest answer is zero.

Each entry: {id, kind, group, lines, text}
  kind  : direct | paraphrase | hindi | boilerplate | out_of_domain
  group : paraphrase-group name (members must return the SAME set) or None
"""

from __future__ import annotations

Q = []
_n = 0


def _add(kind, text, group=None, lines=1):
    global _n
    _n += 1
    Q.append({"id": f"q{_n:03d}", "kind": kind, "group": group,
              "lines": lines, "text": " ".join(text.split())})


# ═══════════════════════════════════════════════════════════════════════════
# PARAPHRASE GROUPS — 15 groups, 45 questions. Same intent, tiny surface changes.
# ═══════════════════════════════════════════════════════════════════════════
_add("paraphrase", "projects in J&K",                                  group="g01_jk")
_add("paraphrase", "all projects in J&K",                              group="g01_jk")
_add("paraphrase", "projects in Jammu and Kashmir",                    group="g01_jk")

_add("paraphrase", "electricity dues in Jammu and Kashmir",            group="g02_dues")
_add("paraphrase", "outstanding electricity dues in J&K",              group="g02_dues")
_add("paraphrase", "electricity dues owed in Jammu and Kashmir",       group="g02_dues")

_add("paraphrase", "status of Subansiri Lower project",                group="g03_subansiri")
_add("paraphrase", "current status of Subansiri Lower project",        group="g03_subansiri")
_add("paraphrase", "present status of the Subansiri Lower hydroelectric project",
     group="g03_subansiri")

_add("paraphrase", "under construction hydro projects",                group="g04_uc")
_add("paraphrase", "ongoing hydro projects",                           group="g04_uc")
_add("paraphrase", "all ongoing hydro projects",                       group="g04_uc")

_add("paraphrase", "vacant posts in NHPC",                             group="g05_vacancy")
_add("paraphrase", "vacancies in NHPC",                                group="g05_vacancy")
_add("paraphrase", "number of vacant posts in NHPC",                   group="g05_vacancy")

_add("paraphrase", "CSR expenditure of NHPC",                          group="g06_csr")
_add("paraphrase", "CSR spending of NHPC",                             group="g06_csr")
_add("paraphrase", "NHPC expenditure on corporate social responsibility", group="g06_csr")

_add("paraphrase", "hydro projects in Himachal Pradesh",               group="g07_hp")
_add("paraphrase", "hydro projects in HP",                             group="g07_hp")
_add("paraphrase", "all hydro projects in Himachal Pradesh",           group="g07_hp")

_add("paraphrase", "recruitment in PSUs",                              group="g08_recruit")
_add("paraphrase", "recruitment in public sector undertakings",        group="g08_recruit")
_add("paraphrase", "new recruitments in PSUs",                         group="g08_recruit")

_add("paraphrase", "Teesta projects in Sikkim",                        group="g09_teesta")
_add("paraphrase", "Teesta hydro projects in Sikkim",                  group="g09_teesta")
_add("paraphrase", "all Teesta projects in Sikkim",                    group="g09_teesta")

_add("paraphrase", "DPRs approved by CEA",                             group="g10_dpr")
_add("paraphrase", "detailed project reports approved by CEA",         group="g10_dpr")
_add("paraphrase", "DPRs approved by Central Electricity Authority",   group="g10_dpr")

_add("paraphrase", "power projects in West Bengal",                    group="g11_wb")
_add("paraphrase", "ongoing power projects in West Bengal",            group="g11_wb")
_add("paraphrase", "details of ongoing power projects in West Bengal", group="g11_wb")

_add("paraphrase", "dues of DISCOMs to power generating companies",    group="g12_discom")
_add("paraphrase", "outstanding dues of DISCOMs to generating companies", group="g12_discom")
_add("paraphrase", "DISCOM dues to power generation companies",        group="g12_discom")

_add("paraphrase", "hydropower potential in India",                    group="g13_potential")
_add("paraphrase", "hydel power potential in the country",             group="g13_potential")
_add("paraphrase", "hydro electric potential in India",                group="g13_potential")

_add("paraphrase", "rehabilitation of families displaced by Chamera project in Chamba",
     group="g14_chamera", lines=2)
_add("paraphrase", "resettlement of families displaced by the Chamera project in Chamba district",
     group="g14_chamera", lines=2)
_add("paraphrase", "rehabilitation of people displaced by NHPC Chamera project Chamba",
     group="g14_chamera", lines=2)

_add("paraphrase", "transmission lines added in the last five years",  group="g15_transmission")
_add("paraphrase", "circuit kilometres of transmission lines added in last five years",
     group="g15_transmission")
_add("paraphrase", "transmission line kilometres added in the past five years",
     group="g15_transmission")

# ═══════════════════════════════════════════════════════════════════════════
# DIRECT one-liners — 15 questions, real corpus topics
# ═══════════════════════════════════════════════════════════════════════════
_add("direct", "seismic monitoring near NHPC dams")
_add("direct", "NHPC bonds raised on behalf of Ministry of Power")
_add("direct", "joint ventures of NHPC")
_add("direct", "Kishanganga project status")
_add("direct", "power supplied to Delhi by NHPC")
_add("direct", "renovation and modernisation of hydro power stations in Maharashtra")
_add("direct", "flood control measures in Himalayan states")
_add("direct", "pumped storage projects under construction")
_add("direct", "battery energy storage capacity required by 2026-27")
_add("direct", "hydro projects in Arunachal Pradesh")
_add("direct", "NHPC profit and dividend paid to the government")
_add("direct", "contractual employees versus regular employees in PSUs")
_add("direct", "Parbati hydroelectric project commissioning")
_add("direct", "solar projects under implementation by NHPC")
_add("direct", "land acquisition for hydro projects in Sikkim")
_add("direct", "Salal hydroelectric project power generation")
_add("direct", "Uri power station performance")
_add("direct", "Dulhasti power station generation")
_add("direct", "NHPC acquisition of Lanco Teesta Hydro Power")
_add("direct", "tariff of hydro power compared to solar power")

# ═══════════════════════════════════════════════════════════════════════════
# DIRECT 2-line questions — 10, closer to how officers actually paste them
# ═══════════════════════════════════════════════════════════════════════════
_add("direct", """whether the Government has reviewed the implementation of hydro projects
     in the North East and if so the main reasons for the delays""", lines=2)
_add("direct", """the number of hydroelectric projects currently under construction in the
     country along with their expected completion timelines""", lines=2)
_add("direct", """whether any hydro power stations were covered under the renovation and
     modernisation programme and the additional capacity gained""", lines=2)
_add("direct", """the details of project proposals sent by the state of Manipur and the
     amount sanctioned in the last three years""", lines=2)
_add("direct", """the steps taken by the Government to mitigate the increase of natural
     disasters in sensitive areas of Himalayan states""", lines=2)
_add("direct", """whether expected quantum of electricity is not being generated through
     hydel power despite huge potential in the country""", lines=2)
_add("direct", """the details of new dams whose proposals are still pending in Himachal
     Pradesh along with their current status""", lines=2)
_add("direct", """the number of people belonging to landless families affected under the
     NHPC Chamera-III project in Chamba district""", lines=2)
_add("direct", """whether the power generating companies have moved to Supreme Court to
     recover their dues from DISCOMs""", lines=2)
_add("direct", """the details of the compensation paid to the families displaced by
     hydroelectric projects in Jammu and Kashmir""", lines=2)

# ═══════════════════════════════════════════════════════════════════════════
# DIRECT 3-line long parliamentary questions — 10
# ═══════════════════════════════════════════════════════════════════════════
_add("direct", """whether the Government proposes to set up new hydroelectric power projects
     in the North Eastern region during the next five years and if so the details thereof
     along with the estimated capacity and investment involved state-wise""", lines=3)
_add("direct", """the total installed capacity of hydro power in the country at present and
     the steps taken by the Government to increase the share of hydro power in the total
     energy mix over the coming years""", lines=3)
_add("direct", """whether NHPC has signed any memorandum of understanding with state
     governments for the development of hydroelectric projects and if so the details of the
     projects proposed under the said MoUs""", lines=3)
_add("direct", """the details of funds allocated and utilised for the rehabilitation and
     resettlement of families affected by hydroelectric projects during the last three years
     project-wise and state-wise""", lines=3)
_add("direct", """whether instances of landslides and floods have increased in the districts
     of Darjeeling and Kalimpong and if so whether the Ministry has taken note of the impact
     on hydro projects in the region""", lines=3)
_add("direct", """the number of Detailed Project Reports for hydro projects approved by the
     Central Electricity Authority in the last ten years along with the present status of
     each such project""", lines=3)
_add("direct", """whether the Government has any proposal to revive the stalled or stressed
     hydroelectric projects in the country and if so the details of the projects identified
     and the financial assistance proposed""", lines=3)
_add("direct", """the quantum of power generated by NHPC power stations during each of the
     last three years station-wise and the revenue earned from the sale of such power during
     the said period""", lines=3)
_add("direct", """whether the environmental clearances for hydroelectric projects in the
     Himalayan region are being reviewed in the light of recent natural disasters and if so
     the details of the projects affected""", lines=3)
_add("direct", """the steps taken or being taken by the Union Government to complete the
     ongoing hydroelectric projects in a time bound manner and the mechanism put in place to
     monitor the progress of these projects""", lines=3)

# ═══════════════════════════════════════════════════════════════════════════
# HINDI — 5 (cross-lingual; g16 is a Hindi paraphrase pair)
# ═══════════════════════════════════════════════════════════════════════════
_add("hindi", "सुबनसिरी परियोजना की स्थिति",                          group="g16_hi_subansiri")
_add("hindi", "सुबनसिरी लोअर परियोजना की वर्तमान स्थिति",              group="g16_hi_subansiri")
_add("hindi", "जल विद्युत परियोजनाओं से विस्थापित परिवारों का पुनर्वास")
_add("hindi", "हिमाचल प्रदेश में जल विद्युत परियोजनाएं")
_add("hindi", "एनएचपीसी की निर्माणाधीन परियोजनाएं")

# ═══════════════════════════════════════════════════════════════════════════
# BOILERPLATE BAIT — 5. Stock phrases from hundreds of answers. WANT: zero results.
# ═══════════════════════════════════════════════════════════════════════════
_add("boilerplate", "if so the details thereof")
_add("boilerplate", "steps taken by the government")
_add("boilerplate", "the reasons therefor")
_add("boilerplate", "details thereof and if not the reasons therefor")
_add("boilerplate", "may be replied by Ministry of Power")

# ═══════════════════════════════════════════════════════════════════════════
# OUT-OF-DOMAIN — 5. No corpus precedent. WANT: zero results (honest empty state).
# ═══════════════════════════════════════════════════════════════════════════
_add("out_of_domain", "railway station redevelopment scheme")
_add("out_of_domain", "income tax slab rates for salaried employees")
_add("out_of_domain", "minimum support price for wheat")
_add("out_of_domain", "Ayushman Bharat hospital empanelment")
_add("out_of_domain", "semiconductor fabrication plant incentives")

assert len(Q) == 100, f"dataset must be exactly 100 questions, got {len(Q)}"

if __name__ == "__main__":
    import json
    from collections import Counter
    print(json.dumps(Q[:3], ensure_ascii=False, indent=2))
    print("total:", len(Q))
    print("by kind:", dict(Counter(q["kind"] for q in Q)))
    print("groups:", len({q["group"] for q in Q if q["group"]}))
