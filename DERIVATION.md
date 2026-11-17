# Derivation and attribution

**DEEPGUARD AI is a derivative work.** It was built by taking the open-source
project **SENTINEL** by **Dev Parth** and extending it. SENTINEL is licensed under
the Apache License 2.0, as is this project.

| | |
|---|---|
| Original project | SENTINEL (repository `Deepfake-detection1`) |
| Original author | Dev Parth |
| Original repository | <https://github.com/Devparth7-coder/Deepfake-detection1> |
| Original license | Apache-2.0 (`LICENSE`, kept unchanged) |
| Derived project | DEEPGUARD AI |
| Maintainer of the derivative | Abdul Majid Jamil |
| Baseline commit | `22e9931` (git tag `baseline-sentinel-22e9931` in the working repository) |

The original author's copyright and the `NOTICE` origin statement are kept.
Nothing here claims SENTINEL's original code as the work of the DEEPGUARD AI
maintainer. This file does not rewrite history: the upstream repository is the
authoritative record of what Dev Parth wrote.

## What came from SENTINEL

The architecture and the first implementation of these pieces are Dev Parth's:

- The overall design: FastAPI service, Celery + Redis queue, `/api/v1/detect`
  and `/api/v1/status/{task_id}`, PyTorch worker.
- The ResNeXt-50 + 2-layer LSTM detector and `BinaryFocalLoss` (`model.py`).
- The face-extraction pipeline idea and the training script skeleton.
- The Docker/Compose layout and the first security measures (UUID task ids,
  path sandbox, magic-byte check).

Files that started as SENTINEL files and were then modified carry a header reading
`Originally part of SENTINEL, Copyright 2026 Dev Parth` followed by
`Modified 2026 by Abdul Majid Jamil`: `config.py`, `inference.py`, `jobstore.py`, `main.py`,
`model.py`, `pipeline.py`, `samplegen.py`, `security.py`, `worker.py`, `scripts/train.py`,
`tests/test_model.py`, `tests/test_security.py`. Most were substantially rewritten; the header
stays because the origin does.

The dashboard (`web/*`), `Dockerfile` and `docker-compose.yml` were replaced with new
implementations. They carry Abdul Majid Jamil's copyright, and the files that can hold a
comment name SENTINEL as the project they derive from.

## What was written for DEEPGUARD AI

Files marked `Copyright 2026 Abdul Majid Jamil`:

- `checkpoint.py` - safe `weights_only` checkpoint loading with strict validation.
- `metrics.py`, `dataprep.py`, `scripts/evaluate.py`, `scripts/validate_dataset.py` -
  dataset validation, train/val leakage prevention, evaluation metrics.
- `report.py` - PDF / JSON forensic reports.
- `install.sh`, `start.sh`, `stop.sh`, `health_check.sh`, `requirements*.txt`,
  `.env.example`, `.dockerignore`, `.github/workflows/ci.yml`.
- `tests/conftest.py`, `tests/test_api.py`, `tests/test_engine.py`, `tests/test_training.py`.
- This documentation (`README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `DERIVATION.md`).

## Main changes relative to SENTINEL 1.x

See `CHANGELOG.md` for the full list. In short: rebrand; honest demo/trained
inference modes; multi-face tracking; frame-level suspicious-frame previews;
rewritten dashboard (history, PDF/JSON reports, model/system pages); input
validation, rate limiting, optional API key and sanitised errors; a safe
checkpoint loader; training with leakage checks and full metrics; Kali/Docker
scripts; tests and CI.

## Things this project does NOT own

- **Model weights.** No trained checkpoint is included. A checkpoint you train
  inherits the license/terms of the dataset it was trained on.
- **Datasets.** FaceForensics++, Celeb-DF, DFDC and others are distributed under
  their own agreements, usually research-only. Do not redistribute them.
- **Third-party libraries and the YuNet ONNX model** keep their own licenses
  (see `NOTICE`).
- **Pretrained backbones** (torchvision ImageNet weights) keep their upstream terms.

If you redistribute this project, keep `LICENSE`, `NOTICE` and this file.
