"""
Supporting-document endpoints.

    GET  /supporting/categories                list the config category registry
    GET  /supporting                           list active docs (for the draft dropdown)
    POST /admin/supporting/upload              admin-only: stage -> validate -> as-of -> store
    POST /admin/supporting/{id}/deactivate     admin-only: soft-delete
    GET  /supporting/file                       stream a supporting doc (ID -> jail)

REUSES the existing upload security verbatim: sanitize_relpath -> check_extension -> jail ->
sniff -> stream -> atomic move. Nothing here reimplements a control the upload feature
already has right.

ISOLATION: nothing in this module touches the Q&A tables, the retrieval graph, or the
crawler. It writes to supporting_* and reads read_document(). The Q&A search is unaffected.
"""

from __future__ import annotations

import io
import os
import shutil
import uuid

from fastapi import (APIRouter, Body, Depends, File, Form, HTTPException, Query,
                     Request, UploadFile)
from fastapi.responses import StreamingResponse

from nhpc_qa.core.logging import get_logger
from nhpc_qa.api.security import deps, upload_guard as guard, users
from nhpc_qa.api.security.upload_guard import Rejected
from nhpc_qa.supporting import ingest

log = get_logger("nhpc.supporting")

router = APIRouter()

_CHUNK = 1024 * 1024


def _st(request: Request):
    return request.app.state.nhpc


def _enabled(cfg):
    if not cfg.supporting_enabled:
        raise HTTPException(503, "supporting documents are disabled (SUPPORTING_ENABLED=false)")


# ---------------------------------------------------------------------------
# read endpoints
# ---------------------------------------------------------------------------
@router.get("/supporting/categories")
def categories(request: Request, who=Depends(deps.require_user)):
    cfg = _st(request)["cfg"]
    _enabled(cfg)
    return {"categories": [{"slug": s, "label": l}
                           for s, l in cfg.supporting_categories().items()]}


@router.get("/supporting")
def list_docs(request: Request, who=Depends(deps.require_user)):
    """Active documents, grouped by category, with the period so the officer picks the right
    vintage. Available to any authenticated user -- officers select these into drafts."""
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    _enabled(cfg)
    labels = cfg.supporting_categories()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, category, display_name, period_label, as_of_date, page_count,
                   needs_review, original_filename
            FROM supporting_documents
            WHERE is_active
            ORDER BY category, display_name
        """)
        cols = [c.name for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"documents": [{
        "id": r["id"], "category": r["category"],
        "category_label": labels.get(r["category"], r["category"]),
        "display_name": r["display_name"],
        "period_label": r["period_label"],
        "as_of_date": r["as_of_date"].isoformat() if r["as_of_date"] else None,
        "page_count": r["page_count"],
        "needs_review": r["needs_review"],
    } for r in rows]}


# ---------------------------------------------------------------------------
# admin upload
# ---------------------------------------------------------------------------
@router.post("/admin/supporting/upload")
async def upload(request: Request,
                 file: UploadFile = File(...),
                 category: str = Form(...),
                 display_name: str = Form(default=""),
                 as_of_date: str = Form(default=""),
                 period_label: str = Form(default=""),
                 admin=Depends(deps.require_admin)):
    """
    ONE file per call (these are individual reference documents, not a folder tree).

    stage -> validate (ext + magic bytes + size + jail) -> parse (whole-doc, no chunking) ->
    propose as-of (LLM) which the admin's supplied value OVERRIDES -> atomic move -> store.
    """
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    _enabled(cfg)

    category = (category or "").strip().lower()
    if category not in cfg.supporting_categories():
        raise HTTPException(400, {"message": f"unknown category {category!r}",
                                  "detail": f"allowed: {list(cfg.supporting_categories())}"})

    ip = deps.client_ip(request)
    ua = request.headers.get("user-agent")
    actor = {"user_id": admin.get("db_user_id"), "email": admin["email"]}

    root = cfg.supporting_root_abs()
    cat_dir = os.path.join(root, category)
    os.makedirs(cat_dir, exist_ok=True)

    allowed = cfg.allowed_exts()          # REUSE upload's allow-list
    max_file = cfg.upload_max_file_mb * 1024 * 1024

    # ---- validate the filename through the SAME guards ---------------------
    orig = file.filename or "document"
    try:
        rel = guard.sanitize_relpath(f"{category}/{os.path.basename(orig)}")
        ext = guard.check_extension(rel, allowed)
        guard.jail(root, rel)             # prove it lands inside the supporting root
    except Rejected as e:
        users.audit(conn, "supporting_upload_rejected", success=False, actor=actor,
                    reason=str(e), ip=ip, user_agent=ua)
        raise HTTPException(400, str(e))

    # ---- stage to a temp file, sniff, THEN move ----------------------------
    stage = os.path.join(cat_dir, f".staging_{uuid.uuid4().hex}{ext}")
    size = 0
    try:
        with open(stage, "wb") as out:
            while True:
                chunk = await file.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_file:
                    raise Rejected(f"file exceeds {cfg.upload_max_file_mb} MB")
                out.write(chunk)
        guard.sniff(stage, ext)           # magic bytes AFTER on disk, BEFORE it counts

        sha = ingest.sha256_of(stage)
        final_name = f"{sha[:16]}{ext}"   # content-addressed -> re-upload overwrites itself
        final_abs = os.path.join(cat_dir, final_name)
        rel_path = os.path.relpath(final_abs, root).replace("\\", "/")

        # PARSE the staged file (whole-doc, existing parse layer, no chunking)
        parsed = ingest.parse_supporting_file(cfg, stage, provider=_st(request).get("parser"))

        # as-of: LLM proposes, the admin's typed value WINS.
        proposed = {"as_of_date": None, "period_label": None}
        if cfg.supporting_llm_asof:
            proposed = ingest.propose_as_of(cfg, _get_llm(request),
                                            parsed.get("document_text") or "")
        final_as_of = (as_of_date.strip() or proposed["as_of_date"]) or None
        final_period = (period_label.strip() or proposed["period_label"]) or None

        os.replace(stage, final_abs)      # atomic move into the live tree
    except Rejected as e:
        _rm(stage)
        raise HTTPException(400, str(e))
    except Exception as e:                # noqa: BLE001
        _rm(stage)
        log.exception("supporting upload failed")
        raise HTTPException(500, f"could not process the file: {e}")

    doc_id = ingest.store(
        conn, cfg, category=category,
        display_name=(display_name.strip() or os.path.splitext(orig)[0]),
        file_path=rel_path, original_filename=orig, sha256=sha, parsed=parsed,
        as_of_date=final_as_of, period_label=final_period, uploaded_by=admin["email"])

    users.audit(conn, "supporting_uploaded", success=True, actor=actor,
                reason=f"{category}/{orig} -> {rel_path} ({size} bytes, "
                       f"{len(parsed.get('tables') or [])} table(s), "
                       f"needs_review={parsed.get('needs_review')})",
                ip=ip, user_agent=ua)

    return {"ok": True, "id": doc_id, "category": category,
            "display_name": (display_name.strip() or os.path.splitext(orig)[0]),
            "as_of_date": final_as_of, "period_label": final_period,
            "as_of_proposed_by_llm": bool(cfg.supporting_llm_asof and not as_of_date.strip()),
            "page_count": parsed.get("page_count"),
            "tables": len(parsed.get("tables") or []),
            "needs_review": parsed.get("needs_review"),
            "parse_flags": parsed.get("parse_flags")}


@router.post("/admin/supporting/{doc_id}/deactivate")
def deactivate(request: Request, doc_id: int, admin=Depends(deps.require_admin)):
    """Soft-delete: drops out of the dropdown, row + tables retained (same discipline as the
    Q&A watcher). Re-uploading the same bytes reactivates it."""
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    _enabled(cfg)
    with conn.cursor() as cur:
        cur.execute("""UPDATE supporting_documents SET is_active=false, deleted_at=now()
                       WHERE id=%s AND is_active RETURNING display_name""", (doc_id,))
        row = cur.fetchone()
    conn.commit()
    if not row:
        raise HTTPException(404, "no such active document")
    users.audit(conn, "supporting_deactivated", success=True,
                actor={"user_id": admin.get("db_user_id"), "email": admin["email"]},
                reason=f"id={doc_id} ({row[0]})", ip=deps.client_ip(request),
                user_agent=request.headers.get("user-agent"))
    return {"ok": True, "id": doc_id, "is_active": False}


# ---------------------------------------------------------------------------
# file serving — ID -> path, realpath-jailed, same discipline as /file
# ---------------------------------------------------------------------------
@router.get("/supporting/file")
def supporting_file(request: Request, doc_id: int = Query(...),
                    who=Depends(deps.require_user)):
    cfg, conn = _st(request)["cfg"], _st(request)["conn"]
    _enabled(cfg)
    with conn.cursor() as cur:
        cur.execute("""SELECT file_path, original_filename FROM supporting_documents
                       WHERE id=%s AND is_active""", (doc_id,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(404, "no such document")
    rel, orig = row

    # Resolve + JAIL: the stored path is relative to the supporting root, and we re-prove it
    # is inside that root before opening -- never trust a stored path blindly.
    root = cfg.supporting_root_abs()
    try:
        abs_path = guard.jail(root, guard.sanitize_relpath(rel))
    except Rejected as e:
        raise HTTPException(400, f"bad path: {e}")
    if not os.path.isfile(abs_path):
        raise HTTPException(404, "file not on disk")

    users.audit(conn, "supporting_file_opened", success=True,
                actor={"user_id": who.get("db_user_id"), "email": who["user_id"]},
                reason=f"id={doc_id} {rel}", ip=deps.client_ip(request),
                user_agent=request.headers.get("user-agent"))

    def _stream():
        with open(abs_path, "rb") as fh:
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                yield chunk
    fname = orig or os.path.basename(abs_path)
    return StreamingResponse(_stream(), media_type="application/octet-stream",
                             headers={"Content-Disposition": f'attachment; filename="{_ascii(fname)}"'})


def _get_llm(request):
    st = _st(request)
    llm = st.get("draft_llm")
    if llm is None:
        from nhpc_qa.core.providers import get_llm
        try:
            llm = get_llm(st["cfg"]); st["draft_llm"] = llm
        except Exception:      # noqa: BLE001
            llm = None
    return llm


def _rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _ascii(name):
    return "".join(c if 32 <= ord(c) < 127 and c not in '"\\' else "_" for c in (name or "file"))
