# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: MIME check, stale-upload cleanup,
# extended public result allowlist (see DERIVATION.md).
"""Upload validation, path sandbox, rate limits, headers, public job filter."""

from __future__ import annotations

import hmac
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

from config import (
    ALLOWED_GENERIC_MIME,
    ALLOWED_MIME_PREFIXES,
    ALLOWED_VIDEO_EXT,
    API_KEY,
    CORS_ORIGINS,
    CSP_POLICY,
    DELETE_UPLOAD_AFTER,
    DETECT_RATE,
    DETECT_WINDOW_S,
    IS_PRODUCTION,
    MAX_DECODE_FRAMES,
    MAX_UPLOAD_MB,
    MAX_VIDEO_PIXELS,
    MAX_VIDEO_SECONDS,
    SAMPLE_DIR,
    SAMPLE_RATE,
    STATUS_RATE,
    STATUS_WINDOW_S,
    TRUST_PROXY,
    UPLOAD_DIR,
    WEIGHTS_DIR,
)

logger = logging.getLogger("deepguard.security")

TASK_ID_RE = re.compile(r"^[0-9a-f]{32}$")
FILENAME_RE = re.compile(r"[^\w.\- ]+", re.UNICODE)

# ISO BMFF (`ftyp`) covers mp4 / m4v / mov. EBML covers mkv / webm. RIFF+AVI covers avi.
_FTYP = b"ftyp"
_EBML = b"\x1aE\xdf\xa3"
_RIFF = b"RIFF"
_AVI = b"AVI "

PUBLIC_JOB_KEYS = {
    "task_id",
    "status",
    "stage",
    "progress",
    "message",
    "filename",
    "label",
    "confidence",
    "probability_fake",
    "logit",
    "hybrid_logit",
    "score_source",
    "mode",
    "backbone",
    "weights_loaded",
    "forensics",
    "cosine_series",
    "frames_sampled",
    "faces_detected",
    "detector",
    "video",
    "thumbs",
    "frame_trace",
    "processing_ms",
    "model",
    "created_at",
    "updated_at",
    "dispatch",
    "manipulation_probability",
    "inference_mode",
    "is_demo",
    "model_id",
    "device",
    "max_faces_in_frame",
    "duration_seconds",
    "frame_scores",
    "suspicious_frames",
    "persons",
    "warnings",
    "disclaimer",
    "app_version",
    "completed_at",
}

DEV_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'"
)
PROD_CSP = DEV_CSP + "; frame-ancestors 'none'; upgrade-insecure-requests"


class RateLimiter:
    """In-process sliding window. Fine for a single API replica."""

    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window_s: float) -> bool:
        now = time.monotonic()
        with self._lock:
            bucket = [t for t in self._hits[key] if now - t < window_s]
            if len(bucket) >= limit:
                self._hits[key] = bucket
                return False
            bucket.append(now)
            self._hits[key] = bucket
            return True


limiter = RateLimiter()


def client_ip(headers: dict[str, str], fallback: str = "unknown") -> str:
    if TRUST_PROXY:
        forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()[:80] or fallback
    return fallback


def require_api_key(offered: Optional[str]) -> None:
    """Raise PermissionError if an API key is configured and the request lacks it."""
    if not API_KEY:
        return
    incoming = (offered or "").encode("utf-8")
    expected = API_KEY.encode("utf-8")
    if len(incoming) != len(expected) or not hmac.compare_digest(incoming, expected):
        raise PermissionError("Invalid or missing API key")


def valid_task_id(task_id: str) -> bool:
    return bool(task_id) and bool(TASK_ID_RE.fullmatch(task_id))


def sanitize_filename(name: Optional[str]) -> str:
    raw = Path(name or "upload.mp4").name.replace("\x00", "")
    cleaned = FILENAME_RE.sub("_", raw).strip(" .")[:180]
    if not cleaned:
        cleaned = "upload.mp4"
    ext = Path(cleaned).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        cleaned = Path(cleaned).stem[:80] + ".mp4"
    return cleaned


def extension_of(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        raise ValueError(f"Unsupported extension '{ext}'")
    return ext


def looks_like_video(header: bytes) -> bool:
    if len(header) < 12:
        return False
    if header[4:8] == _FTYP:
        return True
    if header.startswith(_EBML):
        return True
    if header.startswith(_RIFF) and header[8:12] == _AVI:
        return True
    return False


def mime_acceptable(content_type: Optional[str]) -> bool:
    """Client-declared MIME is advisory only (magic bytes are authoritative), but a
    declared non-video type such as ``text/html`` or ``image/png`` is rejected early."""
    if not content_type:
        return True
    base = content_type.split(";", 1)[0].strip().lower()
    if not base:
        return True
    return base.startswith(ALLOWED_MIME_PREFIXES) or base in ALLOWED_GENERIC_MIME


def assert_video_magic(path: Path) -> None:
    with path.open("rb") as fh:
        header = fh.read(32)
    if not looks_like_video(header):
        raise ValueError("File is not a recognised video container")


def assert_inside(path: Path, *roots: Path) -> Path:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise PermissionError("Path is outside the allowed directories")


def assert_safe_video_path(path: Path | str) -> Path:
    return assert_inside(Path(path), UPLOAD_DIR, SAMPLE_DIR)


def assert_safe_weights_path(path: Path | str) -> Path:
    return assert_inside(Path(path), WEIGHTS_DIR)


def validate_video_meta(meta: dict[str, Any]) -> None:
    duration = float(meta.get("duration") or 0.0)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    frames = int(meta.get("frame_count") or 0)
    if width and height:
        if width < 32 or height < 32:
            raise ValueError("Video resolution is too small")
        if width * height > MAX_VIDEO_PIXELS:
            raise ValueError("Video resolution exceeds the server cap")
    if duration > MAX_VIDEO_SECONDS:
        raise ValueError(f"Video longer than {MAX_VIDEO_SECONDS}s is not accepted")
    if frames > MAX_DECODE_FRAMES:
        raise ValueError("Video has too many frames to process")


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Strip internals (tracebacks, absolute paths) before they hit the wire."""
    out = {k: job[k] for k in PUBLIC_JOB_KEYS if k in job}
    if "filename" in out:
        out["filename"] = sanitize_filename(str(out["filename"]))
    return out


def public_error(exc: BaseException) -> str:
    msg = str(exc)
    lowered = msg.lower()
    if isinstance(exc, PermissionError):
        return "The video could not be read."
    if isinstance(exc, FileNotFoundError):
        return "The video could not be found."
    if any(
        token in lowered
        for token in (
            "detectable face", "faces", "too long", "resolution", "too many frames", "too small",
            "recognised video", "unsupported", "could not be opened", "decodable frames",
        )
    ):
        return msg
    logger.info("sanitised worker error: %s", msg)
    return "Processing failed."


def maybe_delete_upload(path: Path) -> None:
    if not DELETE_UPLOAD_AFTER:
        return
    try:
        safe = assert_safe_video_path(path)
        if safe.is_file() and safe.parent.resolve() == UPLOAD_DIR.resolve():
            safe.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("upload cleanup skipped: %s", exc)


def cors_kwargs() -> dict[str, Any]:
    if CORS_ORIGINS == ["*"]:
        return {
            "allow_origins": ["*"],
            "allow_credentials": False,
            "allow_methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
        }
    return {
        "allow_origins": CORS_ORIGINS,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    }


def security_headers() -> dict[str, str]:
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
        "X-DNS-Prefetch-Control": "off",
        "Cross-Origin-Resource-Policy": "same-origin",
        "Content-Security-Policy": CSP_POLICY or (PROD_CSP if IS_PRODUCTION else DEV_CSP),
    }
    if IS_PRODUCTION:
        headers["X-Frame-Options"] = "DENY"
        headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"
    return headers


def new_request_id(incoming: Optional[str] = None) -> str:
    if incoming and TASK_ID_RE.fullmatch(incoming.replace("-", "")[:32]):
        return incoming[:64]
    return uuid.uuid4().hex


def redact_url(url: str) -> str:
    if "@" not in url:
        return url
    try:
        scheme, rest = url.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        return f"{scheme}://***@{host}"
    except ValueError:
        return "***"


def max_upload_bytes() -> int:
    return MAX_UPLOAD_MB * 1024 * 1024


def detect_rate_key(ip: str) -> str:
    return f"detect:{ip}"


def sample_rate_key(ip: str) -> str:
    return f"sample:{ip}"


def status_rate_key(ip: str) -> str:
    return f"status:{ip}"


def check_detect_rate(ip: str) -> None:
    if not limiter.allow(detect_rate_key(ip), DETECT_RATE, DETECT_WINDOW_S):
        raise TimeoutError("Rate limit exceeded")


def check_sample_rate(ip: str) -> None:
    if not limiter.allow(sample_rate_key(ip), SAMPLE_RATE, DETECT_WINDOW_S):
        raise TimeoutError("Rate limit exceeded")


def check_status_rate(ip: str) -> None:
    if not limiter.allow(status_rate_key(ip), STATUS_RATE, STATUS_WINDOW_S):
        raise TimeoutError("Rate limit exceeded")


def iter_allowed_ext() -> Iterable[str]:
    return sorted(ALLOWED_VIDEO_EXT)


def cleanup_stale_uploads(max_age_s: float, now: Optional[float] = None) -> int:
    """Delete orphaned files in the upload directory older than ``max_age_s``.

    Only regular files directly inside ``UPLOAD_DIR`` are touched, never directories
    or anything outside it. Returns the number of files removed.
    """
    now = time.time() if now is None else now
    removed = 0
    try:
        entries = list(UPLOAD_DIR.iterdir())
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.is_file() and not entry.is_symlink() and now - entry.stat().st_mtime > max_age_s:
                entry.unlink()
                removed += 1
        except OSError as exc:
            logger.warning("stale upload cleanup skipped one file: %s", type(exc).__name__)
    return removed
