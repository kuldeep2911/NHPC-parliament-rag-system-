# NHPC Parliament Data — Phase 1 (Organization)

`organize_parliament.py` crawls the messy source tree of parliamentary Q/A
records and produces a clean, canonical copy under `organized/`.

**This phase organizes files only. It does NOT read or parse document content.**
Text extraction / embeddings are a later phase.

## Run

```bash
python organize_parliament.py              # copies into ./organized
python organize_parliament.py --dry-run    # scan + report only, copies nothing
python organize_parliament.py --source . --out organized
```

- **Read-only on the source.** Nothing in the `PARLIAMENT *` folders is ever
  modified, moved, renamed, or deleted. All output is COPIED.
- **Idempotent.** Re-running rewrites only each affected question folder; it
  does not duplicate or corrupt existing output.
- Standard library only (no pandas/dateutil needed). UTF-8 / Devanagari safe.

## Output layout

```
organized/
  <session>/<house>[/<state>]/<question_id>/
      question_original.<ext>          # the question file, if identified
      answer_latest.<ext>              # selected latest reply
      answer_all_versions/…            # every reply-folder file, untouched
      question_other_versions/…        # extra question files when merged
      metadata.json
  _reports/
      report.json                      # full machine-readable report
      report.csv                       # one row per question folder
      orphans.csv                      # supporting/unmatched items for humans
```

## Normalization rules (agreed with stakeholder)

| Aspect | Rule |
|---|---|
| **Session** | Month-range slug `YYYY-mon[-mon]` (e.g. `NOV DEC 2025` → `2025-nov-dec`, `FEB MAR 24` → `2024-feb-mar`). `MONSOON 25` → `2025-monsoon`. |
| **Cross-year winter** (`NOV 23-JAN 24`) | Best-effort START year + flag `ambiguous_session_year`. Never dropped. |
| **House** | `lok_sabha` / `rajya_sabha` / `vidhan_sabha`. `Assembly/Assemble question` → `vidhan_sabha`. Unknown house-level folders (`likely issues`, `Parl questions`, …) → logged as orphans, not reorganized. |
| **Vidhan state** | Extra state level (`Himachal Pradesh`, `J&K`, `ASSAM`) preserved in the path and metadata. |
| **Question ID** | Original number preserved; chamber prefixes (`Dy`, `LS`) stripped. Starred/Unstarred markers (`S`/`U`) and suffix letters (`5240A`) kept in the merge-key so distinct questions don't merge. Vidhan date-style IDs (`14-8-128`) kept whole. Opaque names get a stable path-hash ID (`gen_…`). Original folder name always recorded. |
| **Duplicates** | Same question appearing under a category folder (`CSR/Dy 10824`) and flat, or `5880` + `5880 (Revised l)`, are MERGED into one folder; all source paths recorded and flagged `merged_from_multiple_sources` / `possible_revised_duplicate`. |
| **Latest reply** | Priority: version subfolder (`approved`>`final`>`revised`>`draft`) → version token in filename (`final`, `revised`, `reply2`, `v2`, `r1`) → filesystem mtime. Same-stem PDF preferred over DOCX. **Dates in filenames are treated as question DUE dates, not version dates.** Genuine ties → ALL candidates copied to `answer_latest_candidates/` + flag `ambiguous_latest_reply`. |
| **Excluded from candidates** | Word/Excel locks (`~$…`), `~WRL*.tmp`, `Thumbs.db`, `desktop.ini`, `*.lnk`, `*.download`, `*.db`. `Note*`/`Annexure*`/`Binder*` are copied to `answer_all_versions/` but never SELECTED as the answer. |
| **Archives** | `*.zip/.rar/.7z` copied and flagged `contains_archive` — never silently skipped, never extracted. |

## Review reasons (see `report.csv` / `metadata.json`)

- `ambiguous_session_year` – cross-year session, year picked best-effort.
- `no_reply_selected` – no answer file found in the folder.
- `no_question_file_found` – no question file identified at the folder top level.
- `contains_archive` – a zip is present; a human should extract/inspect it.
- `merged_from_multiple_sources` – folder pools 2+ source locations.
- `possible_revised_duplicate` – a `(Revised)` variant was merged in.
- `multi_question` – folder covers a range / multiple question numbers.
- `ambiguous_latest_reply` – latest-reply signal tied; all candidates kept.

Orphans (`orphans.csv`) list non-house folders, unmatched folders, empty
folders, and loose supporting files at session/house level for human triage.
