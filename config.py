# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI (see DERIVATION.md).
"""Runtime configuration for the DEEPGUARD AI deepfake-detection stack."""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "DEEPGUARD AI"
APP_FULL_NAME = "DEEPGUARD AI – AI-Powered Deepfake Detection"
APP_DEVELOPER = "Abdul Majid Jamil"
VERSION = "2.0.0"


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: str = "") -> list[str]:
    raw = os.getenv(name, default).strip()
    if raw in {"", "*"}:
        return ["*"]
    return [part.strip() for part in raw.split(",") if part.strip()]


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DEEPGUARD_DATA_DIR", str(ROOT / "data"))).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
THUMB_DIR = DATA_DIR / "thumbs"
WEIGHTS_DIR = ROOT / "weights"
WEB_DIR = ROOT / "web"
SAMPLE_DIR = ROOT / "sample"

for _p in (UPLOAD_DIR, RESULT_DIR, THUMB_DIR, WEIGHTS_DIR, SAMPLE_DIR, DATA_DIR):
    _p.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# 30–40 frames is the paper range; 32 is a clean power-of-two default.
SEQ_LEN = int(os.getenv("SEQ_LEN", "32"))
FACE_SIZE = 224
MIN_FACES = 4  # fewer aligned faces than this cannot be classified
SPATIAL_BATCH = int(os.getenv("SPATIAL_BATCH", "4"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "120"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))
UPLOAD_TTL_SECONDS = int(os.getenv("UPLOAD_TTL_SECONDS", "3600"))

BACKBONE = os.getenv("BACKBONE", "resnext50")  # resnext50 | efficientnet_b3 | resnet18
_WEIGHTS_RAW = Path(os.getenv("WEIGHTS_PATH", str(WEIGHTS_DIR / "deepfake_detector.pth")))
WEIGHTS_PATH = _WEIGHTS_RAW if _WEIGHTS_RAW.is_absolute() else (ROOT / _WEIGHTS_RAW)
YUNET_PATH = WEIGHTS_DIR / "face_detection_yunet_2023mar.onnx"
EVAL_REPORT_PATH = WEIGHTS_DIR / "eval_report.json"

DEVICE = os.getenv("DEVICE", "cpu").strip().lower()  # cpu | cuda | auto
TORCH_THREADS = int(os.getenv("TORCH_THREADS", "1"))
# Load ImageNet weights for the spatial extractor (downloads once from torchvision).
PRETRAINED_BACKBONE = _bool("PRETRAINED_BACKBONE", True)

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
ALLOWED_MIME_PREFIXES = ("video/",)
# Browsers/clients sometimes send these for video uploads.
ALLOWED_GENERIC_MIME = {"application/octet-stream", "binary/octet-stream", "application/mp4"}

# --- security ---------------------------------------------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().lower()
IS_PRODUCTION = ENVIRONMENT in {"prod", "production"}
API_KEY = os.getenv("API_KEY", "").strip()
CORS_ORIGINS = _csv("CORS_ORIGINS", "*")
TRUSTED_HOSTS = [h for h in _csv("TRUSTED_HOSTS", "") if h != "*"]
TRUST_PROXY = _bool("TRUST_PROXY", False)
ENABLE_DOCS = _bool("ENABLE_DOCS", not IS_PRODUCTION)
DELETE_UPLOAD_AFTER = _bool("DELETE_UPLOAD_AFTER", IS_PRODUCTION)
# Never enable for untrusted checkpoints (torch pickle can execute code).
# SENTINEL_ALLOW_PICKLE is the legacy name and is still honoured.
ALLOW_PICKLE_WEIGHTS = _bool("DEEPGUARD_ALLOW_PICKLE", _bool("SENTINEL_ALLOW_PICKLE", False))
CSP_POLICY = os.getenv("CSP_POLICY", "").strip()

MAX_VIDEO_SECONDS = int(os.getenv("MAX_VIDEO_SECONDS", "180"))
MAX_VIDEO_PIXELS = int(os.getenv("MAX_VIDEO_PIXELS", str(4096 * 4096)))
MAX_DECODE_FRAMES = int(os.getenv("MAX_DECODE_FRAMES", "14400"))

DETECT_RATE = int(os.getenv("DETECT_RATE", "12"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "6"))
DETECT_WINDOW_S = float(os.getenv("DETECT_WINDOW_S", "60"))
STATUS_RATE = int(os.getenv("STATUS_RATE", "120"))
STATUS_WINDOW_S = float(os.getenv("STATUS_WINDOW_S", "60"))

# Label convention used everywhere: 1 = FAKE (positive / minority), 0 = REAL.
FAKE_LABEL = 1
REAL_LABEL = 0
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DISCLAIMER = (
    "Confidence is the model's score for its own verdict, not a guarantee of factual accuracy. "
    "Automated deepfake detection produces false positives and false negatives; treat this "
    "output as one indicator in a forensic review, not as proof."
)
