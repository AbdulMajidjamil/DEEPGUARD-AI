# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI (see DERIVATION.md).
"""Security unit tests — no network, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from security import (  # noqa: E402
    assert_inside,
    looks_like_video,
    public_error,
    public_job,
    redact_url,
    sanitize_filename,
    valid_task_id,
)


def test_task_id_format():
    assert valid_task_id("a" * 32)
    assert valid_task_id("0123456789abcdef" * 2)
    assert not valid_task_id("../etc/passwd")
    assert not valid_task_id("job:1")
    assert not valid_task_id("")
    assert not valid_task_id("A" * 32)  # uppercase rejected


def test_sanitize_filename():
    assert sanitize_filename("../../etc/passwd.mp4") == "passwd.mp4"
    assert sanitize_filename("clip.mp4.exe").endswith(".mp4")
    assert "\x00" not in sanitize_filename("hi\x00.mp4")
    assert sanitize_filename("my clip (1).mp4") == "my clip (1).mp4" or "clip" in sanitize_filename("my clip (1).mp4")


def test_magic_bytes():
    mp4 = b"\x00\x00\x00\x1cftypisom" + b"\x00" * 8
    avi = b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 8
    mkv = b"\x1aE\xdf\xa3" + b"\x00" * 12
    html = b"<!DOCTYPE html>...."
    assert looks_like_video(mp4)
    assert looks_like_video(avi)
    assert looks_like_video(mkv)
    assert not looks_like_video(html)
    assert not looks_like_video(b"PK\x03\x04" + b"\x00" * 20)


def test_path_sandbox(tmp_path: Path | None = None):
    root = ROOT / "data" / "uploads"
    inside = root / "abc.mp4"
    assert assert_inside(inside, root) == inside.resolve()
    try:
        assert_inside(ROOT / "config.py", root)
        raise AssertionError("escaped")
    except PermissionError:
        pass


def test_public_job_strips_traceback():
    job = {
        "task_id": "a" * 32,
        "status": "failed",
        "traceback": "secret stack",
        "error": "secret",
        "message": "Processing failed.",
        "filename": "clip.mp4",
    }
    out = public_job(job)
    assert "traceback" not in out
    assert "error" not in out
    assert out["message"] == "Processing failed."


def test_public_error_hides_paths():
    assert public_error(PermissionError("/etc/shadow")) == "The video could not be read."
    assert "faces" in public_error(RuntimeError("Need at least 4 detected faces")).lower()


def test_redact_url():
    assert "secret" not in redact_url("redis://user:secret@localhost:6379/0")
    assert "localhost" in redact_url("redis://user:secret@localhost:6379/0")
    assert redact_url("redis://127.0.0.1:6379/0") == "redis://127.0.0.1:6379/0"


if __name__ == "__main__":
    test_task_id_format()
    test_sanitize_filename()
    test_magic_bytes()
    test_path_sandbox()
    test_public_job_strips_traceback()
    test_public_error_hides_paths()
    test_redact_url()
    print("security tests passed")
