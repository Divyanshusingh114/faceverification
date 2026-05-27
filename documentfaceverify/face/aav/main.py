"""
aav.main
--------
FastAPI backend for Aadhaar face verification.

Production wiring lives here:
  - lifespan: warm the model, open Redis, register graceful shutdown
  - middleware: CORS allowlist, security headers, prometheus, rate limit
  - dependencies: API key auth
  - error mapping: all client-visible errors come from `errors.py` as
    stable `{code, message}` payloads
  - state: sessions live in Redis with atomic GETDEL on verify, so an
    embedding can never be reused across attempts

Flow:
  1. POST /api/document        -> upload a KYC document (aadhaar/pan/voter/
                                  driving_licence/passport), extract face +
                                  parse fields, return a session_id.
     POST /api/upload          -> legacy alias (doc_type defaults to aadhaar).
  2. GET  /api/challenge/{sid} -> server picks a random liveness challenge.
  3. POST /api/verify/{sid}    -> webcam frames -> tiered decision with
                                  score_pct, plus the parsed document report.
"""

import base64
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from aav.core import audit
from aav.storage import sessions
from aav.pipeline import documents, parsers
from aav.core.errors import (
    APIError,
    ERR_NEED_CHALLENGE,
    ERR_NO_FRAMES,
    ERR_NOT_READY,
    ERR_RATE_LIMITED,
    ERR_REFERENCE_REJECTED,
    ERR_SESSION_NOT_FOUND,
    ERR_UNSUPPORTED_DOC_TYPE,
)
from aav.pipeline.face_engine import detect_faces, largest_face, normed_embedding, warmup
from aav.pipeline.liveness import (
    FRONTAL_YAW,
    head_signals,
    random_challenge,
    screen_replay_score,
    verify_challenge,
)
from aav.core.logging import configure_logging, logger
from aav.core.metrics import (
    decisions_total,
    liveness_fail_total,
    upload_rejections_total,
    verify_latency_seconds,
)
from aav.core.security import (
    SecurityHeadersMiddleware,
    assert_frames_ok,
    assert_upload_ok,
    limiter,
    require_api_key,
)
from aav.settings import get_settings
from aav.pipeline.verify import average_embedding, cosine_similarity, decide_pct

configure_logging()
log = logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    audit.init_audit()
    log.info("startup_warming_model")
    warmup()
    log.info("startup_connecting_redis")
    await sessions.init_sessions()
    log.info("startup_ready")
    try:
        yield
    finally:
        log.info("shutdown_closing_redis")
        await sessions.close_sessions()


app = FastAPI(title="Aadhaar Face Verification", lifespan=lifespan)

settings = get_settings()

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.state.limiter = limiter

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip logging noise
    if request.url.path in ("/metrics", "/api/healthz", "/api/readyz", "/"):
        return await call_next(request)

    start_time = time.perf_counter()
    log.info("http_request_start", method=request.method, path=request.url.path)
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        log.info(
            "http_request_end",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=int(duration * 1000),
        )
        return response
    except Exception as e:
        duration = time.perf_counter() - start_time
        log.error(
            "http_request_failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration_ms=int(duration * 1000),
        )
        raise

if settings.metrics_enabled:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


# -------------------------------------------------- error handlers
@app.exception_handler(APIError)
async def _api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"ok": False, **exc.detail})


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(_: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"ok": False, "code": ERR_RATE_LIMITED, "message": str(exc.detail)},
    )


# -------------------------------------------------- helpers
def _decode_data_url(data_url: str) -> np.ndarray | None:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url, validate=False)
    except Exception:
        return None
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ============================================================
# 1. Upload an identity document (Aadhaar / PAN / Voter / DL / Passport)
# ============================================================
def _public_report(report: dict) -> dict:
    """Trim internal/diagnostic fields off the report before returning it."""
    return {
        "doc_type": report["doc_type"],
        "fields": report["fields"],
        "parse_confidence": report["parse_confidence"],
    }


async def _ingest_document(file: UploadFile, doc_type: str) -> JSONResponse:
    data = await file.read()
    assert_upload_ok(data)

    if doc_type not in parsers.SUPPORTED_DOC_TYPES:
        raise APIError(
            400,
            ERR_UNSUPPORTED_DOC_TYPE,
            f"doc_type must be one of: {', '.join(parsers.SUPPORTED_DOC_TYPES)}",
        )

    face, report = documents.extract(data, doc_type)
    if face is None:
        upload_rejections_total.labels(reason=report["reason"]).inc()
        log.info("upload_rejected", reason=report["reason"], doc_type=doc_type)
        raise APIError(422, ERR_REFERENCE_REJECTED, report["reason"])

    public_report = _public_report(report)
    sid = uuid.uuid4().hex
    await sessions.get_store().create(
        sid=sid,
        ref_embedding=normed_embedding(face),
        ref_age=float(getattr(face, "age", 0) or 0),
        doc_type=doc_type,
        doc_report=public_report,
    )
    log.info(
        "upload_accepted",
        sid=sid,
        doc_type=doc_type,
        parse_confidence=public_report["parse_confidence"],
        face_px=report["metrics"]["face_px"],
        det_score=report["metrics"]["det_score"],
    )
    return JSONResponse(
        content={
            "ok": True,
            "session_id": sid,
            "document": public_report,
            "reference_quality": report["metrics"],
            "note": "Reference face accepted. Proceed to live verification.",
        }
    )


@app.post("/api/document", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit_upload)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form("aadhaar"),
) -> JSONResponse:
    return await _ingest_document(file, doc_type)


@app.post("/api/upload", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit_upload)
async def upload_legacy(
    request: Request,
    file: UploadFile = File(...),
    doc_type: str = Form("aadhaar"),
) -> JSONResponse:
    """Legacy alias for /api/document. doc_type defaults to aadhaar."""
    return await _ingest_document(file, doc_type)


# ============================================================
# 2. Get a liveness challenge
# ============================================================
@app.get("/api/challenge/{sid}", dependencies=[Depends(require_api_key)])
async def get_challenge(sid: str) -> dict:
    cid, instruction = random_challenge()
    ok = await sessions.get_store().set_challenge(sid, cid)
    if not ok:
        raise APIError(404, ERR_SESSION_NOT_FOUND)
    return {"challenge_id": cid, "instruction": instruction}


# ============================================================
# 3. Verify live frames against the reference
# ============================================================
class VerifyRequest(BaseModel):
    frames: list[str] = Field(default_factory=list)
    mode: str = "live"


@app.post("/api/verify/{sid}", dependencies=[Depends(require_api_key)])
@limiter.limit(settings.rate_limit_verify)
async def verify(request: Request, sid: str, body: VerifyRequest) -> dict:
    started = time.perf_counter()
    if not body.frames:
        raise APIError(400, ERR_NO_FRAMES)
    assert_frames_ok(body.frames)

    sess = await sessions.get_store().consume(sid)
    if sess is None:
        raise APIError(404, ERR_SESSION_NOT_FOUND)
    if sess.challenge is None and body.mode != "upload":
        raise APIError(400, ERR_NEED_CHALLENGE)

    per_frame: list[dict | None] = []
    live_embeddings: list[np.ndarray] = []
    live_ages: list[float] = []
    best_frontal = None

    for data_url in body.frames:
        img = _decode_data_url(data_url)
        if img is None:
            per_frame.append(None)
            continue

        faces = detect_faces(img)
        if len(faces) == 0:
            per_frame.append(None)
            continue

        face = largest_face(faces)
        sig = head_signals(face)
        per_frame.append({"signals": sig, "n_faces": len(faces)})

        is_frontal = abs(sig["yaw"]) < FRONTAL_YAW if body.mode == "live" else True
        if is_frontal and len(faces) == 1:
            live_embeddings.append(normed_embedding(face))
            live_ages.append(float(getattr(face, "age", 0) or 0))
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            ay = abs(sig["yaw"])
            if best_frontal is None or ay < best_frontal[0]:
                best_frontal = (ay, crop)

    # liveness ---------------------------------------------------------
    if body.mode == "upload":
        liveness = {"passed": True, "reason": "bypassed_for_upload", "evidence": {}}
    else:
        liveness = verify_challenge(sess.challenge or "", per_frame)

    response: dict = {
        "challenge_id": sess.challenge or "upload",
        "liveness": liveness,
    }

    if not liveness["passed"]:
        liveness_fail_total.labels(reason=liveness["reason"]).inc()
        response["decision"] = {"tier": "REJECTED", "cause": "liveness_failed"}
        _finish(sid, response, started)
        return response

    if not live_embeddings:
        response["decision"] = {"tier": "MANUAL_REVIEW", "cause": "no_frontal_frame_for_match"}
        _finish(sid, response, started)
        return response

    live_emb = average_embedding(live_embeddings)
    similarity = cosine_similarity(sess.ref_embedding, live_emb)
    live_age = float(np.median(live_ages)) if live_ages else 0.0
    age_gap = abs(sess.ref_age - live_age) if (sess.ref_age and live_age) else 0.0

    decision = decide_pct(similarity, age_gap, doc_type=sess.doc_type)
    replay = screen_replay_score(best_frontal[1]) if best_frontal else 0.0
    decision["frontal_frames_used"] = len(live_embeddings)
    decision["passive_replay_score"] = replay
    decision["passive_replay_note"] = (
        "advisory heuristic only - high values may indicate a screen/print replay"
    )
    response["decision"] = decision
    response["document"] = sess.doc_report
    # Surface a top-level `report` mirror so callers / the SPA can render
    # the parsed user data in one consistent place.
    response["report"] = {
        "doc_type": sess.doc_type,
        "fields": (sess.doc_report or {}).get("fields", {}),
        "parse_confidence": (sess.doc_report or {}).get("parse_confidence", 0.0),
        "decision_tier": decision["tier"],
        "score_pct": decision["score_pct"],
    }
    _finish(sid, response, started)
    return response


def _finish(sid: str, response: dict, started: float) -> None:
    elapsed = time.perf_counter() - started
    verify_latency_seconds.observe(elapsed)
    tier = response["decision"]["tier"]
    decisions_total.labels(tier=tier).inc()
    log.info(
        "verify_complete",
        sid=sid,
        tier=tier,
        liveness_passed=response["liveness"]["passed"],
        liveness_reason=response["liveness"].get("reason"),
        elapsed_ms=int(elapsed * 1000),
    )
    audit.record(
        "verify",
        sid=sid,
        tier=tier,
        liveness=response["liveness"],
        decision=response["decision"],
        elapsed_ms=int(elapsed * 1000),
    )


# ============================================================
# health probes
# ============================================================
@app.get("/api/healthz")
async def healthz() -> dict:
    """Liveness probe — process is up."""
    return {"status": "ok"}


@app.get("/api/readyz")
async def readyz() -> JSONResponse:
    """Readiness probe — model loaded AND Redis reachable."""
    redis_ok = await sessions.ping()
    if not redis_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "code": ERR_NOT_READY, "redis": False},
        )
    return JSONResponse(content={"status": "ready", "redis": True})


# ============================================================
# frontend
# ============================================================
STATIC_DIR = Path(__file__).parent / "web" / "static"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
