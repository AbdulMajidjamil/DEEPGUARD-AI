# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: renamed sample asset (see DERIVATION.md).
"""Build a short synthetic talking-head clip the pipeline can actually detect.

A still portrait is lightly warped, brightness-modulated and zoomed so the
face detector sees a consistent subject while the temporal head has motion
to inspect. Used by the "Run sample" button and health warmup.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config import SAMPLE_DIR

logger = logging.getLogger("deepguard.sample")

SAMPLE_PATH = SAMPLE_DIR / "deepguard_sample.mp4"
PORTRAIT_PATH = SAMPLE_DIR / "portrait.jpg"


def _synthetic_portrait(w: int = 640, h: int = 480) -> np.ndarray:
    """Paint a Haar/YuNet-friendly face if no generated portrait exists."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (28, 24, 22)
    cv2.rectangle(img, (0, int(h * 0.62)), (w, h), (42, 38, 36), -1)

    cx, cy = w // 2, int(h * 0.46)
    cv2.ellipse(img, (cx, cy + 40), (110, 150), 0, 0, 360, (78, 92, 168), -1)  # hair
    cv2.ellipse(img, (cx, cy), (86, 112), 0, 0, 360, (166, 184, 220), -1)  # face
    cv2.ellipse(img, (cx, cy + 18), (70, 90), 0, 0, 360, (154, 172, 210), -1)

    # brows
    cv2.ellipse(img, (cx - 32, cy - 22), (18, 6), 12, 0, 180, (48, 42, 40), 2)
    cv2.ellipse(img, (cx + 32, cy - 22), (18, 6), -12, 0, 180, (48, 42, 40), 2)
    # eyes
    cv2.ellipse(img, (cx - 30, cy - 8), (16, 10), 0, 0, 360, (245, 245, 245), -1)
    cv2.ellipse(img, (cx + 30, cy - 8), (16, 10), 0, 0, 360, (245, 245, 245), -1)
    cv2.circle(img, (cx - 30, cy - 7), 6, (40, 32, 28), -1)
    cv2.circle(img, (cx + 30, cy - 7), 6, (40, 32, 28), -1)
    cv2.circle(img, (cx - 28, cy - 9), 2, (255, 255, 255), -1)
    cv2.circle(img, (cx + 32, cy - 9), 2, (255, 255, 255), -1)
    # nose
    cv2.line(img, (cx, cy - 4), (cx - 4, cy + 22), (120, 130, 170), 2)
    cv2.line(img, (cx, cy - 4), (cx + 4, cy + 22), (120, 130, 170), 2)
    cv2.ellipse(img, (cx, cy + 24), (10, 6), 0, 0, 180, (120, 130, 170), 2)
    # mouth
    cv2.ellipse(img, (cx, cy + 52), (28, 12), 0, 0, 180, (80, 70, 140), 2)
    cv2.ellipse(img, (cx, cy + 50), (22, 8), 0, 0, 180, (90, 80, 150), -1)
    # ears
    cv2.ellipse(img, (cx - 86, cy + 4), (14, 24), 0, 0, 360, (154, 172, 210), -1)
    cv2.ellipse(img, (cx + 86, cy + 4), (14, 24), 0, 0, 360, (154, 172, 210), -1)
    return img


def _load_base() -> np.ndarray:
    if PORTRAIT_PATH.exists():
        img = cv2.imread(str(PORTRAIT_PATH))
        if img is not None:
            return cv2.resize(img, (640, 480))
    return _synthetic_portrait()


def render_sample(dest: Path = SAMPLE_PATH, seconds: float = 3.2, fps: int = 24) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    base = _load_base()
    h, w = base.shape[:2]
    n = int(seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        dest = dest.with_suffix(".avi")
        writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
    rng = np.random.default_rng(7)
    for i in range(n):
        t = i / max(n - 1, 1)
        # Slow Ken-Burns zoom + tiny affine jitter (simulates a handheld talking head).
        scale = 1.0 + 0.06 * t
        angle = 1.4 * np.sin(2 * np.pi * t)
        tx = 6 * np.sin(2 * np.pi * t * 1.3)
        ty = 4 * np.cos(2 * np.pi * t * 0.9)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        frame = cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Subtle illumination flicker (compression / recapture artefact analogue).
        gain = 1.0 + 0.045 * np.sin(2 * np.pi * t * 3.0)
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        if i % 11 == 0:
            noise = rng.normal(0, 3.5, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    logger.info("Wrote sample clip %s (%d frames)", dest, n)
    return dest


def ensure_sample() -> Path:
    if SAMPLE_PATH.exists() and SAMPLE_PATH.stat().st_size > 10_000:
        return SAMPLE_PATH
    avi = SAMPLE_DIR / "deepguard_sample.avi"
    if avi.exists() and avi.stat().st_size > 10_000:
        return avi
    return render_sample()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(render_sample())
