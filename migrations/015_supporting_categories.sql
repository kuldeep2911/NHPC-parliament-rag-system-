-- 015 — admin-created supporting-document categories.
--
-- The base categories come from the SUPPORTING_CATEGORIES env registry (config, not
-- hardcoded). But an admin needs to add a category from the UI without editing .env and
-- restarting -- there are many document types beyond the three defaults. Those live here,
-- and are MERGED with the env registry at read time.
--
-- Additive: a new table, nothing else touched.

CREATE TABLE IF NOT EXISTS supporting_categories (
    slug        text PRIMARY KEY,           -- folder name + DB category; path-safe
    label       text NOT NULL,              -- what the officer sees in the dropdown
    created_by  text,
    created_at  timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE supporting_categories IS
    'Admin-created supporting-document categories, merged with the SUPPORTING_CATEGORIES '
    'env registry. Adding one is a UI action + a folder, not a config edit + restart.';
