# Copyright 2026 Abdul Majid Jamil
# SPDX-License-Identifier: Apache-2.0
"""Engine tests: frame sampling, face detection, checkpoint loading, CPU inference, Celery task."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import torch

import config
from checkpoint import CheckpointError, load_checkpoint
from inference import InferenceEngine
from model import DeepfakeDetector
from pipeline import extract_face_sequence, faces_to_tensor, probe_video, sample_frames


# -- frame sampling / probing ----------------------------------------------------------------

def test_probe_video_reports_metadata(video_path):
    meta = probe_video(video_path)
    assert meta["width"] > 0 and meta["height"] > 0
    assert meta["fps"] == pytest.approx(12, abs=1)
    assert meta["duration"] == pytest.approx(2.0, abs=0.5)


def test_sample_frames_is_uniform_and_bounded(video_path):
    frames, idx, times, meta = sample_frames(video_path, n=8)
    assert 1 <= len(frames) <= 8
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert times == sorted(times)
    assert all(f.ndim == 3 for f in frames)


def test_probe_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x01" * 500)
    with pytest.raises(Exception):
        sample_frames(bad, n=8)


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(Exception):
        probe_video(tmp_path / "nope.mp4")


# -- face detection --------------------------------------------------------------------------

def test_face_extraction_finds_face_in_sample(video_path):
    ext = extract_face_sequence(video_path, n_frames=8)
    assert ext.n_faces >= 4
    assert ext.detector_name in ("yunet", "mtcnn", "haar")
    assert ext.tracks and ext.faces_bgr
    assert ext.faces_bgr[0].shape == (config.FACE_SIZE, config.FACE_SIZE, 3)
    tensor = faces_to_tensor(ext.faces_bgr)
    assert tensor.shape == (1, len(ext.faces_bgr), 3, config.FACE_SIZE, config.FACE_SIZE)
    assert torch.isfinite(tensor).all()


def test_face_extraction_reports_no_faces(blank_video_path):
    ext = extract_face_sequence(blank_video_path, n_frames=8)
    assert ext.n_faces == 0
    assert not ext.tracks or all(t.n_faces == 0 for t in ext.tracks)


# -- inference (CPU) -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    return InferenceEngine(backbone="resnet18", pretrained=False, weights=config.WEIGHTS_DIR / "does_not_exist.pth")


def test_missing_checkpoint_means_explicit_demo_mode(engine):
    assert engine.mode == "demo_heuristic"
    assert engine.weights_loaded is False
    desc = engine.describe()
    assert desc["is_demo"] is True and desc["weights_present"] is False
    assert desc["model_id"].startswith("heuristic-demo")


def test_cpu_inference_output_contract(engine, video_path):
    result = engine.run(extract_face_sequence(video_path, n_frames=8))
    assert result["device"] == "cpu"
    assert result["label"] in ("REAL", "FAKE")
    assert 0.0 <= result["probability_fake"] <= 1.0
    assert result["confidence"] == pytest.approx(
        result["probability_fake"] if result["label"] == "FAKE" else 1 - result["probability_fake"], abs=1e-3
    )
    assert result["is_demo"] is True and result["warnings"]
    assert "guarantee" in result["disclaimer"].lower() or "not" in result["disclaimer"].lower()
    for key in ("model_id", "inference_mode", "frame_scores", "suspicious_frames", "persons", "forensics"):
        assert key in result


def test_inference_is_deterministic(engine, video_path):
    ext = extract_face_sequence(video_path, n_frames=8)
    a = engine.run(ext)["probability_fake"]
    b = engine.run(ext)["probability_fake"]
    assert a == pytest.approx(b, abs=1e-6)


def test_too_few_faces_raises_clear_error(engine, video_path):
    ext = extract_face_sequence(video_path, n_frames=8)
    for t in ext.tracks:
        t.faces_bgr = t.faces_bgr[:1]
        t.frame_meta = t.frame_meta[:1]
    with pytest.raises(RuntimeError, match="detectable face"):
        engine.run(ext)


def test_weights_outside_weights_dir_are_ignored(tmp_path):
    rogue = tmp_path / "rogue.pth"
    torch.save({"state_dict": {"x": torch.zeros(1)}}, rogue)
    eng = InferenceEngine(backbone="resnet18", pretrained=False, weights=rogue)
    assert eng.mode == "demo_heuristic"
    assert eng.checkpoint_error and "outside" in eng.checkpoint_error
    assert str(tmp_path) not in eng.checkpoint_error


# -- checkpoint loading ----------------------------------------------------------------------

@pytest.fixture()
def ckpt_path():
    path = config.WEIGHTS_DIR / f"pytest_{uuid.uuid4().hex[:8]}.pth"
    yield path
    path.unlink(missing_ok=True)


def _fresh_model():
    return DeepfakeDetector(backbone="resnet18", lstm_hidden=512, lstm_layers=2, dropout=0.5, pretrained=False)


def test_valid_checkpoint_loads_in_trained_mode(ckpt_path, video_path):
    torch.save({"state_dict": _fresh_model().state_dict(), "backbone": "resnet18", "epoch": 3, "val": {"f1": 0.5}},
               ckpt_path)
    eng = InferenceEngine(backbone="resnet18", pretrained=False, weights=ckpt_path)
    assert eng.mode == "trained_model" and eng.weights_loaded
    assert eng.checkpoint["epoch"] == 3 and len(eng.checkpoint["sha256"]) == 12
    result = eng.run(extract_face_sequence(video_path, n_frames=8))
    assert result["inference_mode"] == "trained_model" and result["is_demo"] is False
    assert result["model_id"].startswith("deepguard-resnet18-lstm@")


def test_wrong_backbone_is_rejected_and_falls_back(ckpt_path):
    torch.save({"state_dict": _fresh_model().state_dict(), "backbone": "resnext50"}, ckpt_path)
    eng = InferenceEngine(backbone="resnet18", pretrained=False, weights=ckpt_path)
    assert eng.mode == "demo_heuristic"
    assert "resnext50" in eng.checkpoint_error


def test_shape_mismatch_rejected(ckpt_path):
    state = _fresh_model().state_dict()
    key = next(k for k in state if k.startswith("lstm"))
    state[key] = torch.zeros(3, 3)
    torch.save({"state_dict": state, "backbone": "resnet18"}, ckpt_path)
    with pytest.raises(CheckpointError):
        load_checkpoint(_fresh_model(), ckpt_path, backbone="resnet18")


def test_missing_keys_rejected(ckpt_path):
    state = dict(list(_fresh_model().state_dict().items())[:5])
    torch.save({"state_dict": state, "backbone": "resnet18"}, ckpt_path)
    with pytest.raises(CheckpointError, match="architecture"):
        load_checkpoint(_fresh_model(), ckpt_path, backbone="resnet18")


def test_nan_weights_rejected(ckpt_path):
    state = _fresh_model().state_dict()
    key = next(k for k, v in state.items() if v.is_floating_point())
    state[key] = state[key].clone()
    state[key].view(-1)[0] = float("nan")
    torch.save({"state_dict": state, "backbone": "resnet18"}, ckpt_path)
    with pytest.raises(CheckpointError, match="NaN"):
        load_checkpoint(_fresh_model(), ckpt_path, backbone="resnet18")


def test_corrupted_file_rejected(ckpt_path):
    ckpt_path.write_bytes(b"this is not a torch checkpoint" * 20)
    with pytest.raises(CheckpointError):
        load_checkpoint(_fresh_model(), ckpt_path, backbone="resnet18")


def test_pickle_payload_is_refused_by_default(ckpt_path):
    class Evil:
        def __reduce__(self):  # would execute on an unsafe torch.load
            return (print, ("pwned",))

    torch.save({"state_dict": Evil()}, ckpt_path, pickle_protocol=2)
    with pytest.raises(CheckpointError):
        load_checkpoint(_fresh_model(), ckpt_path, backbone="resnet18", allow_pickle=False)


def test_checkpoint_missing_file(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(_fresh_model(), tmp_path / "nope.pth", backbone="resnet18")


# -- Celery task (eager, no broker) ----------------------------------------------------------

def test_celery_task_runs_eagerly(video_path):
    import shutil

    import jobstore
    from worker import detect_deepfake

    tid = uuid.uuid4().hex
    dest = config.UPLOAD_DIR / f"{tid}.mp4"
    shutil.copy2(video_path, dest)
    jobstore.store.put(tid, {"task_id": tid, "status": "queued", "filename": "clip.mp4"})
    out = detect_deepfake.apply(args=[str(dest), tid]).get()
    assert out["status"] == "completed"
    assert jobstore.store.get(tid)["status"] == "completed"


def test_celery_task_failure_is_sanitised(tmp_path):
    import jobstore
    from worker import detect_deepfake

    tid = uuid.uuid4().hex
    jobstore.store.put(tid, {"task_id": tid, "status": "queued"})
    outside = Path(tmp_path / "x.mp4")
    outside.write_bytes(b"x")
    out = detect_deepfake.apply(args=[str(outside), tid]).get()
    assert out["status"] == "failed"
    job = jobstore.store.get(tid)
    assert job["status"] == "failed" and str(tmp_path) not in job["message"]
