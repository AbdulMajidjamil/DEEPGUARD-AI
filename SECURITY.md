# Security Policy

DEEPGUARD AI is derived from SENTINEL by Dev Parth (Apache-2.0); see `DERIVATION.md`.

## Supported versions

| Version | Supported |
|---|---|
| 2.x (this repository) | Yes |
| SENTINEL 1.x (upstream) | See the upstream repository |

## Reporting a vulnerability

**Do not open a public issue for security bugs.**

1. Use GitHub's **private vulnerability reporting** ("Security" tab -> "Report a vulnerability") on this
   repository, or contact the maintainer listed in `.well-known/security.txt`.
2. Include the version/commit, deployment mode (`ENVIRONMENT`, Celery or in-process), steps to
   reproduce, and whether any uploaded video, face crop or job JSON left the host.
3. Please allow 90 days for a fix before public disclosure, unless the issue is already being exploited.
   This is a volunteer project; there is no guaranteed response time.

## Threat model

DEEPGUARD AI accepts **untrusted video** from the network, writes it to disk, and decodes it with
OpenCV/FFmpeg inside a worker. That is the primary attack surface.

| Asset | Risk | Control |
|---|---|---|
| Uploaded videos | Malicious containers, frame bombs, polyglots | Size cap, extension + MIME + magic-byte check, duration/resolution/frame caps, UUID storage names |
| Job store | IDOR / traceback leak | 128-bit random `task_id`, public-field allowlist, sanitised errors, no stack traces or server paths in responses |
| Redis | Unauthenticated broker | JSON-only Celery serializer, Redis not published by Compose, URL redaction in logs |
| Model weights | `torch.load` pickle code execution | Weights path confined to `weights/`; `torch.load(weights_only=True)`; tensors-only, finite-value, shape and backbone validation; pickle refused unless `DEEPGUARD_ALLOW_PICKLE=true` |
| Worker filesystem | Path traversal via task arguments | Video path must resolve under `data/uploads` or `sample/` |
| API | Abuse, XSS, key leakage | Sliding-window rate limits, strict CSP (`script-src 'self'`), `nosniff`, `no-store` on `/api/*`, API key accepted only in the `X-API-Key` header and compared in constant time |
| Face crops | Biometric residue | Uploads can be deleted after analysis (`DELETE_UPLOAD_AFTER=true`, default in production); results hold small JPEG previews only; history lives in the user's browser |

This software is **not** a substitute for a WAF, a TLS terminator or an authenticating reverse proxy.
Put it behind one before exposing it to the internet.

## Production hardening checklist

```bash
ENVIRONMENT=production
API_KEY=<long random token>          # python3 -c "import secrets; print(secrets.token_urlsafe(32))"
CORS_ORIGINS=https://your.frontend.example
TRUSTED_HOSTS=api.example.com
DELETE_UPLOAD_AFTER=true
ENABLE_DOCS=false
REDIS_URL=redis://:password@redis:6379/0
```

- Terminate TLS in front (HSTS is emitted only when `ENVIRONMENT=production`).
- Do not publish the Redis or Celery ports.
- Set `TRUST_PROXY=true` only behind a reverse proxy you control; otherwise `X-Forwarded-For` is ignored.
- Run the worker with `--pool=solo`, concurrency 1 (PyTorch + fork is unsafe).
- Never load checkpoints from untrusted senders.
- Rate limiting is per API process (in memory). Behind several replicas, also limit at the proxy.

## Known limits

- The in-process fallback worker (used when Redis/Celery is down) shares the API process; a crash in
  video decoding can take the API down with it. Use Redis + a worker for anything shared.
- A dashboard API key saved in the browser is stored in `localStorage`; use it only on devices you trust.
