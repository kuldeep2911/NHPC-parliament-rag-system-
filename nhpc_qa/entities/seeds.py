"""
Authoritative SEED lists — zero-guess entities the dictionary starts from.

Indian states and union territories with their standard abbreviations. These are a closed,
authoritative list, so they need no LLM and no mining. J&K, HP, UP, MP, AP etc. are exactly
the abbreviations that caused the query instability, so seeding them is the core of the fix.

NHPC project names are NOT hardcoded here -- they are read from the DB (the UC-projects
supporting doc's tables, and project mentions in the corpus), which keeps them current.
"""

from __future__ import annotations

# canonical name -> [aliases]. The canonical name also becomes an alias automatically.
STATES = {
    "Andhra Pradesh": ["ap", "a.p."],
    "Arunachal Pradesh": ["arunachal", "ar.p.", "arp"],
    "Assam": [],
    "Bihar": [],
    "Chhattisgarh": ["cg", "c.g."],
    "Goa": [],
    "Gujarat": ["guj"],
    "Haryana": [],
    "Himachal Pradesh": ["himachal", "hp", "h.p."],
    "Jharkhand": ["jh"],
    "Karnataka": ["ka"],
    "Kerala": [],
    "Madhya Pradesh": ["mp", "m.p."],
    "Maharashtra": ["mh"],
    "Manipur": [],
    "Meghalaya": [],
    "Mizoram": [],
    "Nagaland": [],
    "Odisha": ["orissa"],
    "Punjab": ["pb"],
    "Rajasthan": ["raj"],
    "Sikkim": [],
    "Tamil Nadu": ["tn", "t.n."],
    "Telangana": ["ts"],
    "Tripura": [],
    "Uttar Pradesh": ["up", "u.p."],
    "Uttarakhand": ["uttaranchal", "uk"],
    "West Bengal": ["wb", "w.b."],
    # union territories relevant to NHPC's footprint
    "Jammu and Kashmir": ["jammu & kashmir", "jammu kashmir", "j&k", "j & k", "jk", "j and k"],
    "Ladakh": [],
    "Delhi": ["new delhi"],
    "Puducherry": ["pondicherry"],
    "Andaman and Nicobar Islands": ["andaman & nicobar", "a&n islands", "andaman and nicobar"],
    "Dadra and Nagar Haveli and Daman and Diu": ["dnh", "daman and diu"],
    "Chandigarh": [],
    "Lakshadweep": [],
}


# Organizations/schemes that recur in NHPC parliamentary replies and are stable, closed
# names. Mining "Full (ABBR)" will also catch these on first use, but seeding the common
# ones means they work even in a document that does not define them.
ORGANIZATIONS = {
    "Ministry of Power": ["mop", "mo p"],
    "Central Electricity Authority": ["cea"],
    "Central Water Commission": ["cwc"],
    "National Green Tribunal": ["ngt"],
    "Ministry of Environment, Forest and Climate Change": ["moefcc", "moef&cc", "moef"],
    "Central Electricity Regulatory Commission": ["cerc"],
    "Power Grid Corporation of India Limited": ["powergrid", "pgcil"],
    "NTPC Limited": ["ntpc"],
    "Damodar Valley Corporation": ["dvc"],
    "North Eastern Electric Power Corporation": ["neepco"],
    "Public Sector Undertaking": ["psu"],
    "Chenab Valley Power Projects Private Limited": ["cvppl", "cvpppl"],
    "Ratle Hydroelectric Power Corporation Limited": ["rhpcl"],
}
