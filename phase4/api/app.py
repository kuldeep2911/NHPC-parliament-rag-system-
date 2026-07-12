"""
Phase-4 API — query, file serving, feedback.

    python -m phase4.api.app          ->  http://127.0.0.1:8099

Port 8099, not 8080: on Windows 8080 often falls inside a reserved TCP exclusion range
(Hyper-V/WinNAT) and fails to bind with WinError 10013 even though nothing is listening.
Override with PHASE4_API_PORT.

ENDPOINTS
    GET  /                 the officer UI
    POST /query            run a query -> ranked past questions + answers + file buttons
    GET  /file             stream a reply or annexure BY ID (never by path)
    POST /feedback         capture 👍/👎 (updatable); never mutates ranking
    GET  /health

SECURITY
  * RBAC on /query and /file. Roles come from config.
  * Identity comes from X-User-Id / X-User-Role, set by the authenticating reverse proxy
    in front of this service. The API binds to 127.0.0.1 by default -- see
    phase4/security/rbac.py for the trust boundary, which is explicit, not assumed.
  * /file takes doc_key + file_kind (+ ref_label), NEVER a filesystem path. The path is
    looked up server-side and proven to be inside organized/ (see security/paths.py).
  * EVERY query and EVERY file open is audited -- including denials.

READ-ONLY on Phases 1-3 data. Nothing here deletes, shares, or modifies source data.
"""

from __future__ import annotations

import contextlib
import logging
import os

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse

from phase2.trace import new_run_id
from phase3.db import connect
from phase3.embeddings import get_embedder
from phase4.config import Phase4Config, load_dotenv
from phase4.feedback import store
from phase4.graph.build import build_graph
from phase4.graph.tracing import QueryTracer
from phase4.rerank.providers import get_reranker
from phase4.retrieval import entity
from phase4.security import audit, paths, rbac

log = logging.getLogger("nhpc.phase4.api")

# Built once at startup: providers, entity vocabulary, the compiled graph.
_STATE: dict = {}


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the providers once on startup; close the DB connection on shutdown."""
    load_dotenv()
    cfg = Phase4Config()
    errs = cfg.validate_phase4()
    if errs:
        # fail fast and loudly -- a half-configured retrieval service is worse than none
        raise RuntimeError("CONFIG ERROR:\n  " + "\n  ".join(errs))

    # Hold the context manager itself for the process lifetime. Calling
    # connect(cfg).__enter__() and dropping the manager lets the generator be
    # garbage-collected, which runs its finally: and CLOSES the connection --
    # producing "the connection is closed" on the first request.
    conn_ctx = connect(cfg)
    conn = conn_ctx.__enter__()
    deps = {
        "cfg": cfg,
        "conn": conn,
        "embedder": get_embedder(cfg),
        "reranker": get_reranker(cfg) if cfg.rerank_enabled else None,
        "entity_vocab": entity.load_vocabulary(conn),
        "llm": None,
        "tracer": QueryTracer(cfg),
    }
    _STATE["cfg"] = cfg
    _STATE["conn"] = conn
    _STATE["deps"] = deps
    _STATE["graph"] = build_graph(deps)
    log.info("phase4 api ready | rerank=%s generation=%s | %d entities",
             cfg.rerank_enabled, cfg.generation_enabled, len(deps["entity_vocab"]))

    yield

    try:
        conn_ctx.__exit__(None, None, None)
    except Exception:           # noqa: BLE001 -- shutdown must not raise
        pass


app = FastAPI(title="NHPC Parliamentary Q&A — Retrieval", version="1.0",
              lifespan=lifespan)


def identity(x_user_id: str = Header(default="anonymous"),
             x_user_role: str = Header(default="")):
    """The caller's identity, as asserted by the authenticating proxy in front of us."""
    return {"user_id": x_user_id, "user_role": x_user_role}


@app.get("/health")
def health():
    cfg = _STATE.get("cfg")
    return {"ok": bool(cfg),
            "rerank_enabled": bool(cfg and cfg.rerank_enabled),
            "generation_enabled": bool(cfg and cfg.generation_enabled)}


# ---------------------------------------------------------------------------
# QUERY
# ---------------------------------------------------------------------------
@app.post("/query")
def query(payload: dict = Body(...), who=Depends(identity)):
    cfg, conn = _STATE["cfg"], _STATE["conn"]
    q = (payload.get("query") or "").strip()
    run_id = new_run_id()

    # RBAC first -- and a denial is audited, not silently dropped
    try:
        rbac.check(cfg, "query", who["user_role"])
    except rbac.AccessDenied as e:
        audit.log_query(conn, run_id, q, who["user_id"], who["user_role"],
                        allowed=False, denial_reason=str(e))
        raise HTTPException(403, str(e))

    if not q:
        raise HTTPException(400, "query is required")

    state = {
        "run_id": run_id,
        "query": q,
        "user_id": who["user_id"],
        "user_role": who["user_role"],
        "house": payload.get("house"),
        "session": payload.get("session"),
        "nhpc_only": bool(payload.get("nhpc_only")),
        "widened": False,
        "timings_ms": {},
        "errors": [],
    }
    _STATE["deps"]["tracer"].start(state, cfg)
    out = _STATE["graph"].invoke(state)
    _STATE["deps"]["tracer"].finish()
    results = out.get("results") or []

    # durable trace (so a later 👎 is debuggable) + audit
    store.save_run(conn, cfg, out)
    audit.log_query(conn, run_id, q, who["user_id"], who["user_role"], allowed=True,
                    n_results=len(results),
                    doc_keys=[r["doc_key"] for r in results])

    stats = out.get("fuse_stats") or {}
    return {
        "run_id": run_id,
        "query": q,
        # PROCESSING ONLY -- language never filtered the candidate set
        "language": out.get("language"),
        "entities": out.get("entities") or [],
        "results": results,
        "draft": out.get("draft"),           # None unless generation is enabled
        "diagnostics": {
            "_note": "confidence signals are HEURISTICS for triage, not correctness",
            "top_score": stats.get("top_score"),
            "score_gap": stats.get("score_gap"),
            "retrievers_eligible": stats.get("eligible"),
            "retrievers_fired": stats.get("fired"),
            "n_candidates": stats.get("n_candidates"),
            "widened": bool(out.get("widened")),
            "widen_reason": out.get("widen_reason"),
            "rerank_failed": bool(out.get("rerank_failed")),
            "timings_ms": out.get("timings_ms"),
        },
    }


# ---------------------------------------------------------------------------
# FILE — by ID, never by path
# ---------------------------------------------------------------------------
@app.get("/file")
def file(doc_key: str = Query(...), file_kind: str = Query(...),
         ref_label: str | None = Query(default=None),
         run_id: str | None = Query(default=None),
         who=Depends(identity)):
    cfg, conn = _STATE["cfg"], _STATE["conn"]

    try:
        rbac.check(cfg, "file", who["user_role"])
    except rbac.AccessDenied as e:
        audit.log_file_access(conn, run_id, doc_key, file_kind if file_kind in
                              ("reply", "annexure") else "reply", ref_label, None,
                              who["user_id"], who["user_role"], allowed=False,
                              denial_reason=str(e))
        raise HTTPException(403, str(e))

    try:
        abs_path, rel_path = paths.resolve(conn, cfg, doc_key, file_kind, ref_label)
    except paths.FileNotAvailable as e:
        # honest: referenced but never found. Not a security event, but still audited.
        audit.log_file_access(conn, run_id, doc_key, file_kind, ref_label, None,
                              who["user_id"], who["user_role"], allowed=False,
                              denial_reason=f"unavailable: {e}")
        raise HTTPException(404, str(e))
    except paths.FileAccessDenied as e:
        audit.log_file_access(conn, run_id, doc_key, file_kind, ref_label, None,
                              who["user_id"], who["user_role"], allowed=False,
                              denial_reason=str(e))
        raise HTTPException(403, str(e))

    audit.log_file_access(conn, run_id, doc_key, file_kind, ref_label, rel_path,
                          who["user_id"], who["user_role"], allowed=True)
    return FileResponse(abs_path, media_type=paths.content_type(abs_path),
                        filename=os.path.basename(abs_path))


# ---------------------------------------------------------------------------
# FEEDBACK — capture only; never mutates ranking
# ---------------------------------------------------------------------------
@app.post("/feedback")
def feedback(payload: dict = Body(...), who=Depends(identity)):
    conn = _STATE["conn"]
    run_id = payload.get("run_id")
    verdict = payload.get("verdict")
    doc_key = payload.get("doc_key")        # None = feedback on the whole query
    reason = payload.get("reason")

    if not run_id or not verdict:
        raise HTTPException(400, "run_id and verdict are required")
    try:
        fid = store.record_feedback(conn, run_id, who["user_id"], verdict,
                                    doc_key=doc_key, reason=reason,
                                    user_role=who["user_role"])
    except ValueError as e:
        raise HTTPException(400, str(e))

    # A repeat vote UPDATES the previous one (Change 1) -- officers can change their mind.
    return {"ok": True, "feedback_id": fid, "run_id": run_id, "doc_key": doc_key,
            "verdict": verdict,
            "_note": "captured for audit/eval; it does NOT change live rankings"}


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def ui():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "static", "index.html"), encoding="utf-8") as fh:
        return fh.read()


def main():
    import socket
    import uvicorn

    load_dotenv()
    cfg = Phase4Config()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    # Fail EARLY with an actionable message. Uvicorn otherwise builds the whole app
    # (loading the embedder, the reranker and 1200+ entities) and only then discovers it
    # cannot bind -- which reads like the service started and then died for no reason.
    #
    # On Windows a port can be inside a reserved TCP exclusion range (Hyper-V/WinNAT) and
    # bind with WinError 10013 even though nothing is listening on it. 8080 commonly is.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((cfg.api_host, cfg.api_port))
    except OSError as e:
        host, port = cfg.api_host, cfg.api_port
        code = getattr(e, "winerror", None) or e.errno
        print(f"\nCannot bind {host}:{port} — {e}\n")

        # The two Windows failures need OPPOSITE responses, so name them apart rather
        # than printing one guess for both.
        if code in (10048, 98):        # WSAEADDRINUSE / EADDRINUSE
            print("  The port is ALREADY IN USE — most likely an earlier run of this")
            print("  server that is still alive. Find and stop it:\n")
            print(f"      PowerShell: Get-NetTCPConnection -LocalPort {port} -State Listen |")
            print("                    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }\n")
        elif code in (10013, 13):      # WSAEACCES — reserved, NOT in use
            print("  The port is RESERVED by Windows (Hyper-V/WinNAT) — nothing is")
            print("  listening on it, but the OS will not let you bind it. Check:\n")
            print("      netsh interface ipv4 show excludedportrange protocol=tcp\n")

        print("  Or just use a different port:")
        print("      PowerShell: $env:PHASE4_API_PORT=8123; python -X utf8 -m phase4.api.app")
        print("      or set PHASE4_API_PORT=8123 in your .env\n")
        return 1
    finally:
        probe.close()

    print(f"\n  Officer UI: http://{cfg.api_host}:{cfg.api_port}\n")
    uvicorn.run("phase4.api.app:app", host=cfg.api_host, port=cfg.api_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
