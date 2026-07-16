"""
Draft-assist endpoints.

    POST /draft        {run_id} -> the grounded, cited draft
    POST /draft/docx   {run_id} -> the same draft as a .docx download

NON-BLOCKING BY CONSTRUCTION. Drafting is NOT a graph node here. /query returns the
retrieved results at once, as it always has; the UI then calls /draft when the officer
clicks "Generate draft". Retrieval never waits for the LLM.

(The old graph node, generation/draft.py, still exists behind GENERATION_ENABLED for CLI
and batch use. It is off, and it stays off, precisely because it would put an LLM call on
the critical path of every search.)

THE RETRIEVED SET IS REBUILT FROM THE DATABASE, NOT FROM THE CLIENT.
The request carries a run_id, and we re-read query_results for it. The browser never gets
to say "draft from THESE results" -- otherwise a client could feed the model any text it
liked and get it back wearing NHPC's letterhead. It also means the draft is provably built
from the results the officer actually saw, which is the whole point of the audit trail.

FAILS SOFT. An LLM error returns 200 with draft=null and a reason. The officer keeps their
results; the UI shows "draft unavailable". Losing the draft beats losing the results.
"""

from __future__ import annotations

import io

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from nhpc_qa.core.logging import get_logger
from nhpc_qa.api.security import deps
from nhpc_qa.retrieval.generation import assist
from nhpc_qa.retrieval.generation.docx_export import build_docx, safe_filename

log = get_logger("nhpc.draft")

router = APIRouter()


def _st(request: Request):
    return request.app.state.nhpc


# The retrieved set, rebuilt from the run. Mirrors what assemble produced, because the
# draft must see exactly what the officer saw.
_RESULTS_SQL = """
SELECT qr.rank, qr.doc_key,
       sq.question_text, sq.part_label,
       d.session, d.house, d.question_id, d.reply_date, d.answer_file_path,
       ag.answer_text, ag.answer_type, ag.answers_parts
FROM query_results qr
JOIN sub_questions sq ON sq.sub_question_id = qr.sub_question_id
JOIN diaries d        ON d.doc_key = qr.doc_key
LEFT JOIN answer_groups ag ON ag.answer_group_id = sq.answer_group_id
WHERE qr.run_id = %(run_id)s
  AND d.active                 -- a document deleted since the search is not drafted from
ORDER BY qr.rank
"""


def _load_results(conn, run_id: str):
    with conn.cursor() as cur:
        cur.execute(_RESULTS_SQL, {"run_id": run_id})
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return [{
        "rank": r["rank"],
        "doc_key": r["doc_key"],
        "question_text": r["question_text"],
        "part_label": r["part_label"],
        "session": r["session"],
        "house": r["house"],
        "diary_number": r["question_id"],
        "reply_date": r["reply_date"].isoformat() if r["reply_date"] else None,
        "answer_text": r["answer_text"],
        "answer_type": r["answer_type"],
        "answer_covers_parts": r["answers_parts"] or [],
        "reply_file": {"available": bool(r["answer_file_path"])},
    } for r in rows]


def _query_text(conn, run_id: str):
    with conn.cursor() as cur:
        cur.execute("SELECT query_text FROM query_runs WHERE run_id = %s", (run_id,))
        row = cur.fetchone()
    return row[0] if row else None


def _audit(conn, event, who, run_id, query, detail=None):
    """Every draft and every download is recorded, joined to the run it came from.

    A government reply must be traceable: 'which retrieved set produced this draft, for
    whom, and when' has to be answerable months later."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_audit (run_id, query_text, user_id, user_role, allowed,
                                         denial_reason, n_results)
                VALUES (%s, %s, %s, %s, true, %s, NULL)
            """, (run_id, f"[{event}] {query or ''}"[:2000], who["user_id"],
                  who["user_role"], detail))
        conn.commit()
    except Exception as e:      # noqa: BLE001 -- audit must never break the feature
        log.error("draft audit failed: %s: %s", type(e).__name__, e)


def _build(request: Request, run_id: str, who, supporting_ids=None, officer_prompt=None):
    """Shared by /draft and /draft/docx. Returns (query, draft) or raises HTTPException."""
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]

    if not cfg.draft_enabled:
        raise HTTPException(503, "draft assistance is disabled (DRAFT_ENABLED=false)")
    if not run_id:
        raise HTTPException(400, "run_id is required")

    query = _query_text(conn, run_id)
    if query is None:
        raise HTTPException(404, "no such run_id — run the search again")

    results = _load_results(conn, run_id)

    # Selected supporting documents (optional). Empty -> the draft is byte-identical to
    # before this feature. Loaded from the DB by ID so the client can only reference real,
    # active documents -- never inject arbitrary text.
    supporting = _load_supporting(request, cfg, conn, supporting_ids)

    # No past Q&A AND no supporting doc -> nothing to ground on. But a draft grounded ONLY
    # in a supporting document is valid: a financial question may have no past parliamentary
    # precedent, yet the financial digest IS the answer. So we only reject when BOTH are
    # empty. build_draft handles the "supporting only" case (usable=[] but supporting!=[]).
    if not results and not supporting:
        raise HTTPException(404, "that search returned no results, and no supporting "
                                 "document was selected — nothing to draft from")

    # Built lazily, and only when drafting is actually used: nothing is constructed when
    # the feature is off, and a broken LLM config cannot stop the API from starting.
    llm = _st(request).get("draft_llm")
    if llm is None:
        from nhpc_qa.core.providers import get_llm
        try:
            llm = get_llm(cfg)
            _st(request)["draft_llm"] = llm
        except Exception as e:      # noqa: BLE001
            log.warning("draft: no LLM available (%s: %s)", type(e).__name__, e)
            return query, {"ok": False, "reason": "no drafting model is configured"}

    return query, assist.build_draft(cfg, llm, query, results, run_id=run_id,
                                     supporting=supporting, officer_prompt=officer_prompt)


def _load_supporting(request, cfg, conn, ids):
    """
    Load the selected supporting documents -- text + a flattened rendering of their tables --
    from the DB, by ID. Only ACTIVE documents; a stale/deleted id is silently skipped.

    Returns [] when nothing is selected or the feature is off, so the draft path is
    unchanged. The client sends only IDs, never content: it cannot feed the model arbitrary
    text dressed up as an NHPC document.
    """
    if not ids or not getattr(cfg, "supporting_enabled", False):
        return []
    ids = [int(i) for i in ids if str(i).strip().isdigit()][:8]   # bound the payload
    if not ids:
        return []
    labels = cfg.supporting_categories()
    out = []
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, category, display_name, period_label, as_of_date, page_count,
                   document_text
            FROM supporting_documents
            WHERE id = ANY(%s) AND is_active
            ORDER BY category, display_name
        """, (ids,))
        docs = [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]
        for d in docs:
            cur.execute("""
                SELECT t.orientation, t.nl_rendering
                FROM supporting_document_tables t
                WHERE t.supporting_doc_id = %s ORDER BY t.table_index
            """, (d["id"],))
            tbls = cur.fetchall()
            tables_text = "\n\n".join(
                f"TABLE ({orient}):\n{nl}" for orient, nl in tbls if nl)
            out.append({
                "id": d["id"], "category": d["category"],
                "category_label": labels.get(d["category"], d["category"]),
                "display_name": d["display_name"],
                "period_label": d["period_label"],
                "as_of_date": d["as_of_date"].isoformat() if d["as_of_date"] else None,
                "page_count": d["page_count"],
                "document_text": d["document_text"],
                "tables_text": tables_text,
            })
    return out


# ---------------------------------------------------------------------------
# POST /draft
# ---------------------------------------------------------------------------
@router.post("/draft")
def draft(request: Request, payload: dict = Body(...), who=Depends(deps.require_user)):
    conn = _st(request)["conn"]
    run_id = (payload.get("run_id") or "").strip()
    sup_ids = payload.get("supporting_ids") or []
    prompt = payload.get("prompt") or ""
    query, out = _build(request, run_id, who, supporting_ids=sup_ids, officer_prompt=prompt)

    if not out.get("ok"):
        # 200, not 5xx. The officer's results are fine; only the optional draft failed, and
        # the UI must render "draft unavailable" rather than treat this as a broken page.
        _audit(conn, "draft_failed", who, run_id, query, out.get("reason"))
        log.info("draft unavailable for %s: %s", run_id, out.get("reason"))
        return {"draft": None, "reason": out.get("reason", "draft unavailable")}

    d = out["draft"]
    # audit WHICH supporting docs fed the draft, joined to run_id -> full traceability
    sup_cited = ",".join(str(s["id"]) for s in d.get("supporting_sources") or []) or "none"
    _audit(conn, "draft_generated", who, run_id, query,
           f"{len(d['parts'])} parts, {len(d['key_points'])} key points, "
           f"{len(d['gaps'])} gaps, {len(d.get('contradictions') or [])} contradictions, "
           f"pattern={d['pattern']}, supporting_docs=[{sup_cited}], "
           f"{d['citations_dropped']} invented citations dropped")
    return {"draft": d}


# ---------------------------------------------------------------------------
# POST /draft/docx
# ---------------------------------------------------------------------------
@router.post("/draft/docx")
def draft_docx(request: Request, payload: dict = Body(...),
               who=Depends(deps.require_user)):
    conn = _st(request)["conn"]
    run_id = (payload.get("run_id") or "").strip()
    query, out = _build(request, run_id, who,
                        supporting_ids=payload.get("supporting_ids") or [],
                        officer_prompt=payload.get("prompt") or "")

    if not out.get("ok"):
        _audit(conn, "draft_docx_failed", who, run_id, query, out.get("reason"))
        raise HTTPException(503, out.get("reason", "draft unavailable"))

    d = out["draft"]
    blob = build_docx(query, d, user_email=who.get("email") or who["user_id"],
                      run_id=run_id)
    fname = safe_filename(query, run_id)      # officer-supplied text -> sanitised

    _audit(conn, "draft_downloaded", who, run_id, query,
           f"docx, {len(blob)} bytes, pattern={d['pattern']}")
    log.info("draft docx: %s downloaded %s (%d bytes)", who["user_id"], fname, len(blob))

    return StreamingResponse(
        io.BytesIO(blob),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'})
