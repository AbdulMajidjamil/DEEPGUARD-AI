# Changelog

All notable changes to DEEPGUARD AI. Derived from SENTINEL 1.x by Dev Parth (see `DERIVATION.md`).

## 2.0.0 - 2026

First DEEPGUARD AI release.

### Added
- **Honest inference modes.** `trained_model` only when a checkpoint passes strict validation;
  otherwise `demo_heuristic`, a rule-based temporal-consistency heuristic that is always
  labelled as a demo, carries a warning and must not be treated as a detector.
- Safe checkpoint loader (`checkpoint.py`): `torch.load(weights_only=True)`, tensors only,
  finite values, backbone/key/shape match, SHA-256 prefix reported.
- Multi-face handling (greedy tracking, up to 3 people), per-person scores, no-face and
  too-few-face errors, corrupted-video and odd-fps handling.
- Frame-level analysis: per-frame anomaly trace and suspicious-frame thumbnails.
- Structured result: label, confidence, manipulation probability, frames, faces, duration,
  model id, inference mode, device, warnings and an explicit "not a guarantee" disclaimer.
- API: `GET /api/v1/model`, `GET /api/v1/system`, `GET /api/v1/report/{task_id}?format=json|pdf`.
- Dashboard (dark navy/cyan): Overview, Video Analysis (drag and drop, upload and processing
  progress, task id), results, Detection History (local to the browser), Forensic Reports,
  Model Information, System Health, Settings (API key, theme), About.
- PDF and JSON forensic reports.
- Security: size limit, extension + MIME + magic-byte validation, UUID filenames, path sandbox,
  rate limiting, optional `X-API-Key`, configurable CORS and trusted hosts, CSP and security
  headers, sanitised errors, stale-upload cleanup, redacted Redis URLs.
- Training/evaluation: dataset validation, train/val leakage detection, class-imbalance
  sampler, Binary Focal Loss, checkpointing, resume, early stopping, seeds, run logs;
  accuracy / precision / recall / F1 / ROC-AUC / confusion matrix / FPR / FNR.
- Kali/Debian scripts (`install.sh`, `start.sh`, `stop.sh`, `health_check.sh`), Dockerfile,
  Compose stack with Redis, API and worker.
- Tests (pytest) and GitHub Actions CI.

### Changed
- Rebranded SENTINEL to DEEPGUARD AI (origin references kept in `NOTICE`, `DERIVATION.md`,
  file headers and the About page).
- `JobStore` falls back to an in-process dictionary on any Redis error, and the API falls back
  to an in-process worker when Redis/Celery is unavailable (documented in the README).
- Environment variable prefix `SENTINEL_*` -> `DEEPGUARD_*` (`SENTINEL_ALLOW_PICKLE` still honoured).

- Upload validation now rejects an unsupported file extension (e.g. `.txt`) up front instead of
  silently renaming it to `.mp4`; content is still verified by magic bytes.

### Removed
- The npm-publish and pylint GitHub workflows (not applicable to this Python project).
- The unconditional random-weight classifier path: with no checkpoint the app now says so.

### Known limitations
See the README. No trained checkpoint is bundled; image and audio detection are not implemented.
