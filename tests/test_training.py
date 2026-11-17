# Copyright 2026 Abdul Majid Jamil
# SPDX-License-Identifier: Apache-2.0
"""Job store / Redis fallback, metrics, dataset leakage checks and a training smoke test."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


# -- JobStore + Redis ------------------------------------------------------------------------

def test_jobstore_memory_fallback_when_redis_down():
    from jobstore import JobStore

    s = JobStore()  # conftest points REDIS_URL at a closed port
    assert s.backend == "memory" and not s.redis_ok
    tid = "d" * 32
    s.put(tid, {"status": "queued"})
    s.merge(tid, status="processing", progress=0.5)
    got = s.get(tid)
    assert got["status"] == "processing" and got["progress"] == 0.5


def test_jobstore_rejects_bad_ids():
    from jobstore import JobStore

    assert JobStore().get("../../etc/passwd") is None


def test_jobstore_with_redis_backend():
    fakeredis = pytest.importorskip("fakeredis")
    from jobstore import JobStore

    s = JobStore()
    s._redis = fakeredis.FakeRedis(decode_responses=True)
    s.backend = "redis"
    tid = "e" * 32
    s.put(tid, {"status": "queued"})
    assert s._redis.get(f"job:{tid}")  # really stored in Redis
    assert s._redis.ttl(f"job:{tid}") > 0  # with an expiry
    assert s.get(tid)["status"] == "queued"


def test_jobstore_degrades_when_redis_dies_mid_run():
    from jobstore import JobStore

    class Dead:
        def set(self, *a, **k):
            raise ConnectionError("redis://:secretpass@host:6379 went away")

        def get(self, *a, **k):
            raise ConnectionError("gone")

    s = JobStore()
    s._redis = Dead()
    s.backend = "redis"
    tid = "f" * 32
    s.put(tid, {"status": "queued"})  # must not raise
    assert s.backend == "memory"
    assert s.get(tid)["status"] == "queued"


def test_jobstore_evicts_expired_memory_jobs(monkeypatch):
    import jobstore
    from jobstore import JobStore

    s = JobStore()
    old = "1" * 32
    s.put(old, {"status": "completed"})
    s._mem[old]["updated_at"] = 0
    monkeypatch.setattr(jobstore, "JOB_TTL_SECONDS", 10)
    s.put("2" * 32, {"status": "queued"})
    assert old not in s._mem


def test_redact_url_hides_password():
    from security import redact_url

    assert "hunter2" not in redact_url("redis://user:hunter2@10.0.0.5:6379/0")


# -- metrics ---------------------------------------------------------------------------------

def test_metrics_match_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    from metrics import binary_metrics

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    m = binary_metrics(y, p)
    pred = (p >= 0.5).astype(int)
    assert m["accuracy"] == pytest.approx(sk.accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(sk.precision_score(y, pred))
    assert m["recall"] == pytest.approx(sk.recall_score(y, pred))
    assert m["f1"] == pytest.approx(sk.f1_score(y, pred))
    assert m["roc_auc"] == pytest.approx(sk.roc_auc_score(y, p))
    tn, fp, fn, tp = sk.confusion_matrix(y, pred).ravel()
    assert m["confusion_matrix"] == [[tn, fp], [fn, tp]]
    assert m["fpr"] == pytest.approx(fp / (fp + tn)) and m["fnr"] == pytest.approx(fn / (fn + tp))


def test_metrics_undefined_is_nan_not_zero():
    from metrics import binary_metrics

    m = binary_metrics([1, 1, 1], [0.9, 0.8, 0.7])  # one class only
    assert math.isnan(m["roc_auc"]) and math.isnan(m["fpr"])
    with pytest.raises(ValueError):
        binary_metrics([], [])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.5])


# -- dataset leakage -------------------------------------------------------------------------

def _touch(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_leakage_detects_duplicate_content_and_shared_identity(tmp_path):
    import dataprep

    _touch(tmp_path / "train/real/100.mp4", b"A" * 100)
    _touch(tmp_path / "train/fake/100_200.mp4", b"B" * 100)
    _touch(tmp_path / "val/real/dup.mp4", b"A" * 100)          # same bytes as a train clip
    _touch(tmp_path / "val/fake/300_200.mp4", b"C" * 100)      # shares identity 200 with train
    _touch(tmp_path / "val/real/999.mp4", b"D" * 100)          # clean
    train = dataprep.scan_split(tmp_path, "train")
    val = dataprep.scan_split(tmp_path, "val")
    out = dataprep.check_leakage(train, val)
    text = " ".join(out["errors"])
    assert "duplicate content" in text and "dup.mp4" in text
    assert "shared identity '200'" in text
    assert "999.mp4" not in text


def test_leakage_clean_split_has_no_errors(tmp_path):
    import dataprep

    _touch(tmp_path / "train/real/1.mp4", b"A" * 50)
    _touch(tmp_path / "val/real/2.mp4", b"B" * 50)
    out = dataprep.check_leakage(dataprep.scan_split(tmp_path, "train"), dataprep.scan_split(tmp_path, "val"))
    assert out["errors"] == []


def test_scan_split_ignores_other_extensions(tmp_path):
    import dataprep

    _touch(tmp_path / "train/real/a.txt", b"x")
    _touch(tmp_path / "train/real/b.mp4", b"x")
    assert [i.path.name for i in dataprep.scan_split(tmp_path, "train")] == ["b.mp4"]


def test_cache_key_distinguishes_same_stem(tmp_path):
    import dataprep

    a, b = tmp_path / "x/clip.mp4", tmp_path / "y/clip.mp4"
    _touch(a, b"1")
    _touch(b, b"1")
    assert dataprep.cache_key(a, 8) != dataprep.cache_key(b, 8)


# -- focal loss ------------------------------------------------------------------------------

def test_focal_loss_downweights_easy_examples():
    import torch

    from model import BinaryFocalLoss

    loss = BinaryFocalLoss(gamma=2.0, alpha=0.75)
    target = torch.tensor([1.0, 1.0])
    easy = loss(torch.tensor([6.0, 6.0]), target)
    hard = loss(torch.tensor([-2.0, -2.0]), target)
    assert hard > easy * 50 and torch.isfinite(easy) and torch.isfinite(hard)


# -- training smoke test (synthetic data: proves plumbing only, says nothing about accuracy) --

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from samplegen import render_sample

    root = tmp_path_factory.mktemp("ds")
    n = 0
    for split, count in (("train", 3), ("val", 2)):
        for cls in ("real", "fake"):
            for i in range(count):
                n += 1
                # vary duration/fps so the bytes differ (identical files are flagged as leakage)
                render_sample(root / split / cls / f"{cls}{n:03d}.mp4", seconds=1.0 + 0.05 * n, fps=10 + n % 5)
    return root


def _run(args, cwd=ROOT):
    env = {**__import__("os").environ, "BACKBONE": "resnet18", "PRETRAINED_BACKBONE": "0"}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=900)


def test_validate_dataset_script(tiny_dataset):
    r = _run(["scripts/validate_dataset.py", "--data", str(tiny_dataset)])
    assert r.returncode == 0, r.stdout + r.stderr


def test_validate_dataset_flags_leakage(tmp_path):
    _touch(tmp_path / "train/real/1.mp4", b"A" * 64)
    _touch(tmp_path / "train/fake/2.mp4", b"B" * 64)
    _touch(tmp_path / "val/real/3.mp4", b"A" * 64)
    _touch(tmp_path / "val/fake/4.mp4", b"C" * 64)
    r = _run(["scripts/validate_dataset.py", "--data", str(tmp_path)])
    assert r.returncode != 0
    assert "duplicate" in (r.stdout + r.stderr).lower()


def test_train_then_evaluate_then_refuse_overwrite(tiny_dataset, tmp_path):
    out = tmp_path / "model.pth"
    base = ["scripts/train.py", "--data", str(tiny_dataset), "--epochs", "1", "--freeze-epochs", "0",
            "--batch-size", "2", "--backbone", "resnet18", "--no-pretrained", "--seq-len", "6",
            "--workers", "0", "--seed", "1", "--out", str(out), "--runs", str(tmp_path / "runs"),
            "--cache", str(tmp_path / "cache")]
    r = _run(base)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    assert out.is_file()

    import torch

    blob = torch.load(out, weights_only=True)
    assert blob["backbone"] == "resnet18" and "state_dict" in blob and blob["seq_len"] == 6

    again = _run(base)
    assert again.returncode != 0, "must not overwrite an existing checkpoint without --overwrite"
    assert out.is_file()

    report = tmp_path / "eval.json"
    ev = _run(["scripts/evaluate.py", "--data", str(tiny_dataset), "--weights", str(out), "--split", "val",
               "--backbone", "resnet18", "--seq-len", "6", "--out", str(report), "--cache", str(tmp_path / "cache")])
    assert ev.returncode == 0, ev.stdout[-1500:] + ev.stderr[-1500:]
    data = json.loads(report.read_text())
    for key in ("accuracy", "precision", "recall", "f1", "confusion_matrix"):
        assert key in json.dumps(data)
