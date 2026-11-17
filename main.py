# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: rebrand, header-only API key, MIME checks,
# system/report endpoints, job + upload cleanup, truthful model info (see DERIVATION.md).
"""DEEPGUARD AI API gateway - FastAPI.

Endpoints
---------
POST /api/v1/detect              upload a video, receive task_id
GET  /api/v1/status/{task_id}    poll job state / final REAL|FAKE verdict
POST /api/v1/detect/sample       enqueue the bundled sample clip
GET  /api/v1/health              redis / worker / model liveness
GET  /api/v1/model               model + checkpoint facts (truthful about demo mode)
GET  /api/v1/system              runtime, device and limits
GET  /api/v1/report/{task_id}    download the result as JSON or PDF (?format=json|pdf)
GET  /                           dashboard
"""

from __future__ import annotations

import logging
import platform
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import (
    ALLOWED_VIDEO_EXT,
    API_KEY,
    APP_DEVELOPER,
    APP_NAME,
    DATA_DIR,
    ENABLE_DOCS,
    ENVIRONMENT,
    IS_PRODUCTION,
    MAX_UPLOAD_MB,
    MAX_VIDEO_SECONDS,
    REDIS_URL,
    ROOT,
    SEQ_LEN,
    TRUSTED_HOSTS,
    UPLOAD_DIR,
    UPLOAD_TTL_SECONDS,
    VERSION,
    WEB_DIR,
)
from jobstore import store
from security import (
    assert_video_magic,
    check_detect_rate,
    check_sample_rate,
    check_status_rate,
    cleanup_stale_uploads,
    client_ip,
    cors_kwargs,
    extension_of,
    max_upload_bytes,
    mime_acceptable,
    new_request_id,
    public_error,
    public_job,
    redact_url,
    require_api_key,
    sanitize_filename,
    security_headers,
    valid_task_id,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deepguard.api")

_DOCS = "/api/docs" if ENABLE_DOCS else None
_REDOC = "/api/redoc" if ENABLE_DOCS else None
_OPENAPI = "/api/openapi.json" if ENABLE_DOCS else None

app = FastAPI(
    title="DEEPGUARD AI - Deepfake Detection API",
    description=(
        "Asynchronous spatiotemporal deepfake video detector "
        "(FastAPI + Redis/Celery + PyTorch ResNeXt-50 + LSTM). Derived from SENTINEL by Dev Parth (Apache-2.0)."
    ),
    version=VERSION,
    docs_url=_DOCS,
    redoc_url=_REDOC,
    openapi_url=_OPENAPI,
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0.html"},
    contact={"name": APP_DEVELOPER, "url": "/.well-known/security.txt"},
)

app.add_middleware(CORSMiddleware, **cors_kwargs())
if TRUSTED_HOSTS:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=[*TRUSTED_HOSTS, "127.0.0.1", "localhost"])


class _SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = new_request_id(request.headers.get("x-request-id"))
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except HTTPException:
            raise
        except Exception:
            logger.exception("unhandled error rid=%s", request_id)
            response = JSONResponse({"detail": "Internal server error"}, status_code=500)
        headers = security_headers()
        # Swagger UI / ReDoc pull inline scripts and CDN assets; do not break /api/docs.
        if request.url.path.startswith(("/api/docs", "/api/redoc")):
            headers.pop("Content-Security-Policy", None)
        for key, value in headers.items():
            response.headers.setdefault(key, value)
        response.headers["X-Request-ID"] = request_id
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Powered-By"] = APP_NAME
        return response


app.add_middleware(_SecurityHeaders)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepguard-job")
_celery_ok_cache = {"ok": False, "checked": 0.0}
_STARTED = time.time()


def _ip(request: Request) -> str:
    fallback = request.client.host if request.client else "unknown"
    return client_ip(dict(request.headers), fallback)


def _auth(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """API key is accepted in the X-API-Key header only (never in a URL)."""
    try:
        require_api_key(x_api_key)
    except PermissionError as exc:
        raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "ApiKey"}) from exc


def _celery_is_alive(ttl: float = 4.0) -> bool:
    now = time.time()
    if now - _celery_ok_cache["checked"] < ttl:
        return bool(_celery_ok_cache["ok"])
    ok = False
    try:
        from worker import celery_app

        inspect = celery_app.control.inspect(timeout=0.8)
        ping = inspect.ping() if inspect is not None else None
        ok = bool(ping)
    except Exception:
        ok = False
    _celery_ok_cache["ok"] = ok
    _celery_ok_cache["checked"] = now
    return ok


def _enqueue(video_path: Path, task_id: str, filename: str) -> dict:
    store.put(
        task_id,
        {
            "task_id": task_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Waiting for a worker",
            "filename": filename,
            "created_at": time.time(),
        },
    )

    dispatched = "thread"
    if store.redis_ok and _celery_is_alive():
        try:
            from worker import detect_deepfake

            detect_deepfake.apply_async(args=[str(video_path), task_id], task_id=task_id)
            dispatched = "celery"
            logger.info("enqueued celery task_id=%s", task_id)
        except Exception as exc:
            logger.warning("celery dispatch failed (%s) - using in-process worker", type(exc).__name__)
    if dispatched == "thread":

        def _run() -> None:
            try:
                from worker import run_pipeline

                run_pipeline(str(video_path), task_id)
            except Exception as exc:
                logger.exception("in-process job failed task=%s", task_id)
                store.merge(
                    task_id, status="failed", stage="failed", progress=1.0, message=public_error(exc)
                )
                from worker import maybe_delete_failed_upload

                maybe_delete_failed_upload(str(video_path))

        _executor.submit(_run)
        logger.info("enqueued thread task_id=%s", task_id)

    return {
        "task_id": task_id,
        "status": "queued",
        "dispatch": dispatched,
        "poll_url": f"/api/v1/status/{task_id}",
    }


def _save_upload(upload: UploadFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit = max_upload_bytes()
    written = 0
    with dest.open("wb") as fh:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds {limit // (1024 * 1024)} MB limit")
            fh.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty upload")
    try:
        assert_video_magic(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(415, str(exc)) from exc


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    if isinstance(exc, RequestValidationError):
        return JSONResponse({"detail": "Invalid request"}, status_code=422)
    logger.exception("unhandled error")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


@app.exception_handler(RequestValidationError)
async def _validation(request: Request, exc: RequestValidationError):
    return JSONResponse({"detail": "Invalid request"}, status_code=422)


@app.post("/api/v1/detect", dependencies=[Depends(_auth)], status_code=202)
async def detect(request: Request, file: UploadFile = File(...)):
    """Accept a video, persist it under a UUID name, queue analysis, return ``task_id`` immediately."""
    try:
        check_detect_rate(_ip(request))
    except TimeoutError:
        raise HTTPException(429, "Too many uploads. Try again shortly.", headers={"Retry-After": "30"})

    filename = sanitize_filename(file.filename)
    try:
        # Check the client's real extension: sanitize_filename() rewrites unknown ones to ".mp4".
        ext = extension_of(Path(file.filename or "upload.mp4").name)
    except ValueError as exc:
        raise HTTPException(415, f"{exc}. Allowed: {sorted(ALLOWED_VIDEO_EXT)}") from exc
    if not mime_acceptable(file.content_type):
        raise HTTPException(415, "Declared content type is not a video type")

    task_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{task_id}{ext}"
    _save_upload(file, dest)
    return JSONResponse(_enqueue(dest, task_id, filename), status_code=202)


@app.post("/api/v1/detect/sample", dependencies=[Depends(_auth)], status_code=202)
async def detect_sample(request: Request):
    """Enqueue the bundled synthetic sample clip (generated on first boot)."""
    try:
        check_sample_rate(_ip(request))
    except TimeoutError:
        raise HTTPException(429, "Too many sample runs. Try again shortly.", headers={"Retry-After": "30"})

    from samplegen import ensure_sample

    sample = ensure_sample()
    task_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{task_id}.mp4"
    shutil.copy2(sample, dest)
    try:
        assert_video_magic(dest)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(415, str(exc)) from exc
    return JSONResponse(_enqueue(dest, task_id, "deepguard_sample.mp4"), status_code=202)


@app.get("/api/v1/status/{task_id}", dependencies=[Depends(_auth)])
async def status(task_id: str, request: Request):
    try:
        check_status_rate(_ip(request))
    except TimeoutError:
        raise HTTPException(429, "Too many polls.", headers={"Retry-After": "5"})
    if not valid_task_id(task_id):
        raise HTTPException(400, "Malformed task_id")
    job = store.get(task_id)
    if job is None:
        raise HTTPException(404, "Unknown task_id")
    return public_job(job)


@app.get("/api/v1/report/{task_id}", dependencies=[Depends(_auth)])
async def report(task_id: str, request: Request, format: str = Query("json", pattern="^(json|pdf)$")):
    """Download a completed analysis as JSON or PDF."""
    try:
        check_status_rate(_ip(request))
    except TimeoutError:
        raise HTTPException(429, "Too many requests.", headers={"Retry-After": "5"})
    if not valid_task_id(task_id):
        raise HTTPException(400, "Malformed task_id")
    job = store.get(task_id)
    if job is None:
        raise HTTPException(404, "Unknown task_id")
    if job.get("status") != "completed":
        raise HTTPException(409, "Analysis is not completed yet")
    public = public_job(job)
    name = f"deepguard-report-{task_id[:8]}"
    if format == "pdf":
        from report import build_pdf

        pdf = build_pdf(public)
        return Response(
            pdf, media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'},
        )
    return JSONResponse(public, headers={"Content-Disposition": f'attachment; filename="{name}.json"'})


@app.get("/api/v1/health")
async def health():
    from inference import light_status

    redis_ok = store.redis_ok
    worker_ok = _celery_is_alive() if redis_ok else False
    model = light_status()
    body = {
        "ok": True,
        "service": "deepguard-ai",
        "version": VERSION,
        "environment": ENVIRONMENT,
        "redis": redis_ok,
        "celery_worker": worker_ok,
        "job_backend": store.backend,
        "dispatch": "celery" if (redis_ok and worker_ok) else "thread",
        "auth": "api_key" if API_KEY else "open",
        "inference_mode": model["inference_mode"],
        "is_demo": model["is_demo"],
        "uptime_seconds": int(time.time() - _STARTED),
    }
    if not IS_PRODUCTION:
        body["redis_url"] = redact_url(REDIS_URL)
    return body


@app.get("/api/v1/model")
async def model_info():
    from inference import light_status

    status_ = light_status()
    ckpt = status_.get("checkpoint") or {}
    return {
        "name": "DEEPGUARD AI hybrid detector",
        "architecture": "CNN-RNN hybrid",
        "spatial": f"{status_['backbone']} (head stripped)",
        "temporal": "2-layer LSTM, hidden=512, batch_first",
        "head": "MLP -> 1 raw logit (sigmoid applied at inference)",
        "loss": "Binary Focal Loss, gamma=2.0, alpha=0.75 (fakes = positive class)",
        "input": f"{SEQ_LEN} uniformly sampled faces, 224x224, ImageNet-normalised",
        "task": "video deepfake detection (face-swap / face-manipulation)",
        "modalities": {"video": True, "image": False, "audio": False},
        "inference_mode": status_["inference_mode"],
        "is_demo": status_["is_demo"],
        "model_id": status_["model_id"],
        "backbone_configured": status_["backbone"],
        "device": status_["device"],
        "device_note": status_.get("device_note"),
        "weights_present": status_["weights_present"],
        "weights_loaded": status_["weights_loaded"],
        "checkpoint": ckpt or None,
        "checkpoint_error": status_.get("checkpoint_error"),
        "parameters_millions": status_.get("parameters_millions"),
        "evaluation": _load_eval_report(),
        "note": (
            "A trained checkpoint is loaded; accuracy applies only to data resembling its training set."
            if not status_["is_demo"]
            else "No trained checkpoint is loaded: results come from a rule-based heuristic, not a trained model."
        ),
        "license": "Apache-2.0",
    }


def _load_eval_report() -> dict | None:
    """Return the evaluation report written by scripts/evaluate.py, if one exists."""
    import json

    from config import EVAL_REPORT_PATH

    try:
        if EVAL_REPORT_PATH.is_file() and EVAL_REPORT_PATH.stat().st_size < 1_000_000:
            data = json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        pass
    return None


@app.get("/api/v1/system", dependencies=[Depends(_auth)])
async def system_info():
    """Runtime facts for the System Health page. No paths, hostnames or secrets."""
    import cv2
    import torch

    from inference import light_status

    model = light_status()
    usage = shutil.disk_usage(DATA_DIR)
    gib = 1024 ** 3
    return {
        "version": VERSION,
        "python": platform.python_version(),
        "platform": platform.system(),
        "torch": torch.__version__.split("+")[0],
        "opencv": cv2.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": model["device"],
        "device_note": model.get("device_note"),
        "redis": store.redis_ok,
        "job_backend": store.backend,
        "dispatch": "celery" if (store.redis_ok and _celery_is_alive()) else "thread",
        "disk": {"free_gb": round(usage.free / gib, 1), "total_gb": round(usage.total / gib, 1)},
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_video_seconds": MAX_VIDEO_SECONDS,
            "allowed_extensions": sorted(ALLOWED_VIDEO_EXT),
            "upload_retention_seconds": UPLOAD_TTL_SECONDS,
        },
        "uptime_seconds": int(time.time() - _STARTED),
    }


_INDEX_NOCACHE = {"Cache-Control": "no-cache"}


@app.get("/")
async def index():
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(500, "Frontend missing")
    return FileResponse(page, headers=_INDEX_NOCACHE)


@app.get("/license", response_class=PlainTextResponse)
async def license_text():
    path = ROOT / "LICENSE"
    if not path.exists():
        raise HTTPException(404, "LICENSE missing")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.get("/notice", response_class=PlainTextResponse)
async def notice_text():
    path = ROOT / "NOTICE"
    if not path.exists():
        raise HTTPException(404, "NOTICE missing")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.get("/security", response_class=PlainTextResponse)
async def security_policy():
    path = ROOT / "SECURITY.md"
    if not path.exists():
        raise HTTPException(404, "SECURITY.md missing")
    return FileResponse(path, media_type="text/markdown; charset=utf-8")


@app.get("/.well-known/security.txt", response_class=PlainTextResponse)
async def security_txt():
    path = ROOT / ".well-known" / "security.txt"
    if not path.exists():
        raise HTTPException(404, "security.txt missing")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def _cleanup_loop(interval_s: float = 600.0) -> None:
    """Remove orphaned uploads (failed/abandoned jobs) after UPLOAD_TTL_SECONDS."""
    while True:
        time.sleep(interval_s)
        try:
            removed = cleanup_stale_uploads(UPLOAD_TTL_SECONDS)
            if removed:
                logger.info("removed %d stale upload(s)", removed)
        except Exception:
            logger.exception("upload cleanup failed")


def _warmup_sample() -> None:
    try:
        from samplegen import ensure_sample

        ensure_sample()
    except Exception as exc:
        logger.warning("sample warmup failed: %s", type(exc).__name__)


def _startup() -> None:
    try:
        cleanup_stale_uploads(UPLOAD_TTL_SECONDS)
    except Exception:
        logger.exception("startup cleanup failed")
    threading.Thread(target=_warmup_sample, daemon=True, name="deepguard-warmup").start()
    threading.Thread(target=_cleanup_loop, daemon=True, name="deepguard-cleanup").start()


_startup()
