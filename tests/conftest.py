# Copyright 2026 Abdul Majid Jamil
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures.

Environment is fixed *before* any application module is imported so the suite
is hermetic: a temp data directory, a tiny backbone, no ImageNet download, no
Redis and no GPU. Nothing here talks to the network.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="deepguard-tests-")
os.environ["DEEPGUARD_DATA_DIR"] = _TMP
os.environ["BACKBONE"] = "resnet18"
os.environ["PRETRAINED_BACKBONE"] = "0"
os.environ["DEVICE"] = "cpu"
os.environ["REDIS_URL"] = "redis://127.0.0.1:1/0"  # nothing listens here -> in-process fallback
os.environ["ENVIRONMENT"] = "development"
os.environ["API_KEY"] = ""
os.environ["SEQ_LEN"] = "8"
os.environ["DETECT_RATE"] = "1000"
os.environ["SAMPLE_RATE"] = "1000"
os.environ["STATUS_RATE"] = "100000"
os.environ.pop("WEIGHTS_PATH", None)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def tmp_media(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("media")


@pytest.fixture(scope="session")
def video_path(tmp_media) -> Path:
    """A short synthetic clip with a drawn face (see samplegen.py). No real person is depicted."""
    from samplegen import render_sample

    return render_sample(tmp_media / "clip.mp4", seconds=2.0, fps=12)


@pytest.fixture(scope="session")
def blank_video_path(tmp_media) -> Path:
    """A valid video that contains no face at all."""
    import cv2
    import numpy as np

    path = tmp_media / "blank.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), 40 + i, np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture(autouse=True)
def _reset_limiter():
    import security

    security.limiter._hits.clear()
    yield
    security.limiter._hits.clear()


@pytest.fixture()
def api_key(monkeypatch):
    import security

    monkeypatch.setattr(security, "API_KEY", "test-key-123")
    return "test-key-123"
