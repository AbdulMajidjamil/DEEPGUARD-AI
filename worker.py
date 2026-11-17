# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: structured result payload, warnings,
# frame-level scores, per-person results (see DERIVATION.md).
"""Celery worker: OpenCV face extraction -> PyTorch hybrid inference.

Start with::

    celery -A worker.celery_app worker --loglevel=INFO --concurrency=1
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from celery import Celery
from celery.signals import worker_process_init

from config import REDIS_URL, SEQ_LEN, VERSION
from jobstore import store
from pipeline import extract_face_sequence, thumbs_to_data_uris
from security import (
    assert_safe_video_path,
    maybe_delete_upload,
    public_error,
    valid_task_id,
)

logger = logging.getLogger("deepguard.worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

celery_app = Celery("deepguard", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    broker_connection_retry_on_startup=True,
    task_reject_on_worker_lost=True,
    result_expires=3600,
    timezone="UTC",
    enable_utc=True,
)


def _progress(task_id: str):
    def inner(stage: str, pct: float, message: str) -> None:
        store.merge(task_id, status="processing", stage=stage, progress=round(float(pct), 3), message=message)

    return inner


def run_pipeline(video_path: str, task_id: str) -> dict[str, Any]:
    """Synchronous pipeline used by both Celery and the in-process executor."""
    t0 = time.perf_counter()
    if not valid_task_id(task_id):
        raise PermissionError("malformed task id")
    path = assert_safe_video_path(video_path)
    if not path.exists():
        raise FileNotFoundError("video missing")
    original_name = (store.get(task_id) or {}).get("filename") or path.name

    store.merge(
        task_id, status="processing", stage="probing_video", progress=0.04,
        message="Probing container", filename=original_name,
    )
    extraction = extract_face_sequence(path, n_frames=SEQ_LEN, progress=_progress(task_id))

    store.merge(
        task_id, status="processing", stage="running_model", progress=0.68,
        message="Running ResNeXt-50 + LSTM",
        frames_sampled=extraction.n_sampled, faces_detected=extraction.n_faces,
        detector=extraction.detector_name, video=_public_video(extraction.video),
    )

    from inference import get_engine

    result = get_engine().run(extraction)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    video = _public_video(extraction.video)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "status": "completed",
        "stage": "completed",
        "progress": 1.0,
        "message": "Classification complete",
        "filename": original_name,
        **{
            k: result[k]
            for k in (
                "label", "confidence", "probability_fake", "manipulation_probability", "logit",
                "hybrid_logit", "score_source", "mode", "inference_mode", "is_demo", "model_id",
                "backbone", "device", "weights_loaded", "forensics", "cosine_series",
                "frame_scores", "suspicious_frames", "persons", "warnings", "disclaimer",
            )
        },
        "frames_sampled": extraction.n_sampled,
        "faces_detected": extraction.n_faces,
        "max_faces_in_frame": extraction.max_faces_in_frame,
        "detector": extraction.detector_name,
        "video": video,
        "duration_seconds": video.get("duration"),
        "thumbs": thumbs_to_data_uris(extraction.frame_meta),
        "frame_trace": [
            {
                "index": f.index, "timestamp": f.timestamp, "has_face": f.has_face,
                "n_faces": f.n_faces, "score": f.score, "laplacian_var": f.laplacian_var, "box": f.box,
            }
            for f in extraction.frame_meta
        ],
        "processing_ms": elapsed_ms,
        "model": f"{result['backbone']} + LSTM",
        "app_version": VERSION,
        "completed_at": time.time(),
    }
    store.put(task_id, payload)
    maybe_delete_upload(path)
    logger.info(
        "task=%s label=%s p_fake=%.3f mode=%s faces=%d/%d ms=%d",
        task_id, payload["label"], payload["probability_fake"], payload["inference_mode"],
        extraction.n_faces, extraction.n_sampled, elapsed_ms,
    )
    return payload


def _public_video(meta: dict[str, Any]) -> dict[str, Any]:
    """Only container facts the UI shows; no codec ints, no paths."""
    return {
        "fps": round(float(meta.get("fps") or 0.0), 3),
        "fps_estimated": bool(meta.get("fps_estimated")),
        "frame_count": int(meta.get("frame_count") or 0),
        "width": int(meta.get("width") or 0),
        "height": int(meta.get("height") or 0),
        "duration": round(float(meta.get("duration") or 0.0), 3),
    }


@worker_process_init.connect
def _preload_engine(**_kwargs: Any) -> None:
    """Load the model once per worker process so the first video isn't cold."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    try:
        from inference import get_engine

        get_engine()
        logger.info("inference engine preloaded")
    except Exception:
        logger.exception("engine preload failed - will retry on first task")


@celery_app.task(bind=True, name="detect_deepfake", max_retries=0)
def detect_deepfake(self, video_path: str, task_id: str) -> dict[str, Any]:
    try:
        return run_pipeline(video_path, task_id)
    except Exception as exc:
        logger.exception("detect_deepfake failed task=%s", task_id)
        store.merge(task_id, status="failed", stage="failed", progress=1.0, message=public_error(exc))
        maybe_delete_failed_upload(video_path)
        # Do not re-raise the original: the result backend would store a traceback.
        return {"task_id": task_id, "status": "failed"}


def maybe_delete_failed_upload(video_path: str) -> None:
    try:
        maybe_delete_upload(Path(video_path))
    except Exception:  # pragma: no cover - cleanup is best effort
        pass


if __name__ == "__main__":
    celery_app.start()
