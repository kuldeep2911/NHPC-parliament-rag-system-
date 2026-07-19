"""
Entity dictionary: canonical entities + surface aliases, mined from the corpus.

The dictionary is the deterministic layer that fixes the abbreviation instability: it maps
every surface form of an entity ("HP", "H.P.", "Himachal Pradesh") to ONE canonical id, so a
query and a record that name the same thing differently resolve identically.

The LLM is used ONLY to DISCOVER candidate entities offline at build time. Its output is
reviewed into the dictionary as data; the non-deterministic model never runs in the live
query path -- only the resulting deterministic dictionary does.
"""
