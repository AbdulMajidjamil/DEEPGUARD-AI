# Copyright 2026 Abdul Majid Jamil
# SPDX-License-Identifier: Apache-2.0
"""HTTP API tests: endpoints, upload validation, auth, rate limiting, error hygiene, reports."""

from __future__ import annotations

import time

import pytest


def _wait(client, task_id, timeout=240.0):
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        r = client.get(f"/api/v1/status/{task_id}")
        assert r.status_code == 200
        job = r.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.5)
    raise AssertionError(f"job did not finish: {job}")


# -- read-only endpoints ---------------------------------------------------------------------

def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["job_backend"] in ("memory", "redis")


def test_model_endpoint_is_honest_without_weights(client):
    body = client.get("/api/v1/model").json()
    assert body["inference_mode"] == "demo_heuristic"
    assert body["is_demo"] is True
    assert body["weights_loaded"] is False


def test_system_endpoint(client):
    r = client.get("/api/v1/system")
    assert r.status_code == 200
    assert r.json()["version"] and "torch" in r.json()


def test_docs_and_frontend_served(client):
    assert client.get("/").status_code == 200
    assert "DEEPGUARD" in client.get("/").text
    assert client.get("/api/docs").status_code == 200
    assert client.get("/license").status_code == 200
    assert "Dev Parth" in client.get("/notice").text


def test_security_headers_present(client):
    r = client.get("/api/v1/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "script-src 'self'" in client.get("/").headers["content-security-policy"]
    assert r.headers["x-request-id"]


# -- status / report error paths -------------------------------------------------------------

def test_status_malformed_and_unknown(client):
    assert client.get("/api/v1/status/not-a-real-id").status_code == 400
    assert client.get("/api/v1/status/" + "a" * 32).status_code == 404


def test_report_unknown_and_malformed(client):
    assert client.get("/api/v1/report/xyz").status_code == 400
    assert client.get("/api/v1/report/" + "b" * 32).status_code == 404


def test_report_not_ready_is_409(client):
    import jobstore

    tid = "c" * 32
    jobstore.store.put(tid, {"task_id": tid, "status": "processing", "progress": 0.3})
    assert client.get(f"/api/v1/report/{tid}").status_code == 409


# -- upload validation -----------------------------------------------------------------------

def test_reject_disallowed_extension(client):
    r = client.post("/api/v1/detect", files={"file": ("evil.exe", b"MZ\x00\x00", "video/mp4")})
    assert r.status_code == 415


def test_reject_non_video_mime(client):
    r = client.post("/api/v1/detect", files={"file": ("a.mp4", b"x" * 64, "text/html")})
    assert r.status_code == 415


def test_reject_bad_magic_bytes(client):
    r = client.post("/api/v1/detect", files={"file": ("fake.mp4", b"<html>not a video at all</html>" * 4, "video/mp4")})
    assert r.status_code == 415


def test_reject_empty_upload(client):
    r = client.post("/api/v1/detect", files={"file": ("empty.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_reject_oversize(client, monkeypatch):
    import main

    monkeypatch.setattr(main, "max_upload_bytes", lambda: 1024)
    payload = b"\x00\x00\x00\x18ftypmp42" + b"0" * 4096
    r = client.post("/api/v1/detect", files={"file": ("big.mp4", payload, "video/mp4")})
    assert r.status_code == 413


def test_missing_file_field_is_422_without_details(client):
    r = client.post("/api/v1/detect")
    assert r.status_code == 422
    assert r.json() == {"detail": "Invalid request"}


def test_rejected_upload_leaves_no_file(client):
    import config

    before = set(config.UPLOAD_DIR.iterdir())
    client.post("/api/v1/detect", files={"file": ("fake.mp4", b"hello world, definitely not mp4", "video/mp4")})
    assert set(config.UPLOAD_DIR.iterdir()) == before


# -- end-to-end analysis ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def completed_job(client, video_path):
    with video_path.open("rb") as fh:
        r = client.post("/api/v1/detect", files={"file": ("my clip.mp4", fh, "video/mp4")})
    assert r.status_code == 202, r.text
    body = r.json()
    assert len(body["task_id"]) == 32
    assert body["poll_url"].endswith(body["task_id"])
    return _wait(client, body["task_id"])


def test_detect_completes_with_structured_result(completed_job):
    job = completed_job
    assert job["status"] == "completed", job
    res = job
    assert res["label"] in ("REAL", "FAKE")
    assert 0.0 <= res["confidence"] <= 1.0
    assert 0.0 <= res["manipulation_probability"] <= 1.0
    assert res["inference_mode"] == "demo_heuristic"
    assert res["is_demo"] is True
    assert res["frames_sampled"] >= 4
    assert res["faces_detected"] >= 4
    assert res["model_id"]
    assert res["warnings"], "demo mode must always carry a warning"
    assert isinstance(res["suspicious_frames"], list)


def test_status_never_leaks_server_paths(completed_job):
    import config

    blob = str(completed_job)
    assert str(config.DATA_DIR) not in blob
    assert str(config.UPLOAD_DIR) not in blob
    assert "video_path" not in blob and "Traceback" not in blob


def test_report_json_and_pdf(client, completed_job):
    tid = completed_job["task_id"]
    j = client.get(f"/api/v1/report/{tid}?format=json")
    assert j.status_code == 200 and j.json()["status"] == "completed"
    assert "attachment" in j.headers["content-disposition"]
    p = client.get(f"/api/v1/report/{tid}?format=pdf")
    assert p.status_code == 200
    assert p.headers["content-type"] == "application/pdf"
    assert p.content.startswith(b"%PDF")
    assert client.get(f"/api/v1/report/{tid}?format=docx").status_code == 422


def test_detect_sample_completes(client):
    r = client.post("/api/v1/detect/sample")
    assert r.status_code == 202
    job = _wait(client, r.json()["task_id"])
    assert job["status"] == "completed", job
    assert job["is_demo"] is True


def test_no_face_video_fails_with_clear_safe_error(client, blank_video_path):
    with blank_video_path.open("rb") as fh:
        r = client.post("/api/v1/detect", files={"file": ("blank.mp4", fh, "video/mp4")})
    job = _wait(client, r.json()["task_id"])
    assert job["status"] == "failed"
    assert "face" in job["message"].lower()
    assert "Traceback" not in job["message"] and "\\" not in job["message"]


def test_corrupt_video_with_valid_magic_fails_safely(client):
    # Passes the magic-byte check (ftyp box) but is not decodable.
    junk = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\xde\xad\xbe\xef" * 2000
    r = client.post("/api/v1/detect", files={"file": ("junk.mp4", junk, "video/mp4")})
    assert r.status_code == 202
    job = _wait(client, r.json()["task_id"])
    assert job["status"] == "failed"
    assert "Traceback" not in job["message"]
    assert "tests" not in job["message"].lower()


# -- auth + rate limit -----------------------------------------------------------------------

def test_api_key_required_when_configured(client, api_key):
    assert client.get("/api/v1/status/" + "a" * 32).status_code == 401
    assert client.get("/api/v1/status/" + "a" * 32, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/status/" + "a" * 32, headers={"X-API-Key": api_key}).status_code == 404
    # The key is never accepted from the URL.
    assert client.get(f"/api/v1/status/{'a' * 32}?api_key={api_key}").status_code == 401
    # Health stays open for liveness probes.
    assert client.get("/api/v1/health").status_code == 200


def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    import security

    monkeypatch.setattr(security, "STATUS_RATE", 3, raising=False)
    monkeypatch.setattr(security.limiter, "allow", lambda *a, **k: False)
    r = client.get("/api/v1/status/" + "a" * 32)
    assert r.status_code == 429
    assert "retry-after" in r.headers


def test_unhandled_error_is_generic(client, monkeypatch):
    import main

    def boom(*a, **k):
        raise RuntimeError("secret /home/user/.env REDIS_PASSWORD=hunter2")

    monkeypatch.setattr(main.store, "get", boom)
    from fastapi.testclient import TestClient

    with TestClient(main.app, raise_server_exceptions=False) as c:
        r = c.get("/api/v1/status/" + "a" * 32)
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}
    assert "hunter2" not in r.text
