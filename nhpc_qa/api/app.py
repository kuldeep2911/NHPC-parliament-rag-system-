"""
Phase-4 API — query, file serving, feedback.

    python -m nhpc_qa.api.app          ->  http://127.0.0.1:8099

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

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse

from nhpc_qa.core.trace.tracer import new_run_id
from nhpc_qa.core.db.session import connect
from nhpc_qa.core.providers.embeddings import get_embedder
from nhpc_qa.config import Settings, load_dotenv
from nhpc_qa.retrieval.feedback import store
from nhpc_qa.retrieval.graph.build import build_graph
from nhpc_qa.core.trace.query_tracer import QueryTracer
from nhpc_qa.core.providers.rerank import get_reranker
from nhpc_qa.retrieval.search import entity
from nhpc_qa.api import auth_routes, tree_routes, upload_routes
from nhpc_qa.api.security import audit, deps, paths, rbac, users

log = logging.getLogger("nhpc.phase4.api")

# Built once at startup: providers, entity vocabulary, the compiled graph.
_STATE: dict = {}


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    """Build the providers once on startup; close the DB connection on shutdown."""
    load_dotenv()
    cfg = Settings()
    errs = cfg.validate_all()
    if errs:
        # fail fast and loudly -- a half-configured retrieval service is worse than none
        raise RuntimeError("CONFIG ERROR:\n  " + "\n  ".join(errs))

    # Hold the context manager itself for the process lifetime. Calling
    # connect(cfg).__enter__() and dropping the manager lets the generator be
    # garbage-collected, which runs its finally: and CLOSES the connection --
    # producing "the connection is closed" on the first request.
    conn_ctx = connect(cfg)
    conn = conn_ctx.__enter__()
    # Named graph_deps, not deps: `deps` is now the auth-dependency MODULE (imported
    # above), and shadowing it here would break require_user/require_admin.
    graph_deps = {
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
    _STATE["deps"] = graph_deps
    _STATE["graph"] = build_graph(graph_deps)

    # The auth dependencies read cfg/conn off app.state (they get a Request, not _STATE).
    _app.state.nhpc = {"cfg": cfg, "conn": conn}

    if cfg.auth_enabled:
        # Refuse to start with auth ON and no admin in the DB -- that configuration locks
        # every human out of the application, including the person who could fix it.
        if not users.admin_exists(conn):
            raise RuntimeError(
                "AUTH_ENABLED=true but there is no active admin user.\n"
                "  Run:  nhpc create-admin --email you@nhpc.in\n"
                "  (or set AUTH_ENABLED=false to run without authentication)")
        log.info("authentication ENABLED | cookie=%s secure=%s samesite=%s | "
                 "lockout after %d failures for %d min",
                 cfg.cookie_name, cfg.cookie_secure, cfg.cookie_samesite,
                 cfg.max_failed_logins, cfg.lockout_minutes)
    else:
        log.warning("authentication is DISABLED (AUTH_ENABLED=false) — every caller is "
                    "treated as officer1/officer. Do not run this way in production.")

    log.info("phase4 api ready | rerank=%s generation=%s | %d entities",
             cfg.rerank_enabled, cfg.generation_enabled, len(graph_deps["entity_vocab"]))

    yield

    try:
        conn_ctx.__exit__(None, None, None)
    except Exception:           # noqa: BLE001 -- shutdown must not raise
        pass


app = FastAPI(title="NHPC Parliamentary Q&A — Retrieval", version="1.0",
              lifespan=lifespan)

# /auth/* and /admin/*. Every /admin route is behind require_admin, which reads the role
# from the DATABASE via the session cookie -- never from the request.
app.include_router(auth_routes.router)
# /admin/upload — the intake path in FRONT of the existing pipeline. It writes files and
# calls queue.enqueue(); it does not parse, embed or index anything itself.
app.include_router(upload_routes.router)
# /admin/tree — browse the source tree, and HARD-delete from it. Deliberate and audited;
# distinct from the watcher's soft delete, which reacts to an AMBIGUOUS filesystem event.
app.include_router(tree_routes.router)


def identity(request: Request):
    """
    The caller's identity — DERIVED FROM THE SESSION, never from the request.

    This function is the whole auth change. It used to read X-User-Id / X-User-Role
    headers, which a client could set to anything ("X-User-Role: admin" was a valid
    request). It now resolves the session cookie against the sessions+users tables.

    The returned SHAPE is unchanged -- {"user_id", "user_role"} -- so rbac.check(),
    audit.log_query(), audit.log_file_access() and every endpoint below are untouched.
    That is also the SSO seam: swap this one function for an OIDC/LDAP token reader and
    nothing else moves.
    """
    return deps.require_user(request)


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
    # Depends(identity) -> require_user: this endpoint used to accept ANY identity,
    # including none at all (a Step-0 gap). It now requires an authenticated, active user
    # who does not owe a password change, like every other app endpoint.
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
    cfg = Settings()
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
        print("      PowerShell: $env:PHASE4_API_PORT=8123; python -X utf8 -m nhpc_qa.api.app")
        print("      or set PHASE4_API_PORT=8123 in your .env\n")
        return 1
    finally:
        probe.close()

    print(f"\n  Officer UI: http://{cfg.api_host}:{cfg.api_port}\n")
    uvicorn.run("nhpc_qa.api.app:app", host=cfg.api_host, port=cfg.api_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
