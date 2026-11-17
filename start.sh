#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
# Start DEEPGUARD AI (API + optional Celery worker + optional private Redis).
# Never deletes data or weights. Records the PIDs it starts in data/run/ so ./stop.sh
# stops only those processes.      HOST=0.0.0.0 PORT=8000 ./start.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

say()  { printf '[start] %s\n' "$*"; }
fail() { printf '[start] ERROR: %s\n' "$*" >&2; exit 1; }

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
REDIS_PORT="${REDIS_PORT:-6379}"
RUN="$ROOT/data/run"
LOGS="$ROOT/data/logs"
mkdir -p "$RUN" "$LOGS" data/uploads data/results data/thumbs weights

[[ -x .venv/bin/python ]] || fail "no .venv found. Run ./install.sh first."
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import fastapi, uvicorn, torch, cv2" 2>/dev/null || fail "Python dependencies missing. Run ./install.sh"

port_busy() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$1\$"
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
  fi
}
alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }

if alive "$RUN/api.pid"; then
  say "already running (pid $(cat "$RUN/api.pid")). Use ./stop.sh first."
  exit 0
fi
if port_busy "$PORT"; then
  fail "port $PORT is already in use by another program. Choose another: PORT=8001 ./start.sh"
fi

# --- Redis: use an existing one, else start our own private one, else fall back ---
redis_ok() { command -v redis-cli >/dev/null 2>&1 && redis-cli -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; }
USE_CELERY=0
if redis_ok; then
  say "Redis already running on port $REDIS_PORT (left untouched)"
  USE_CELERY=1
elif command -v redis-server >/dev/null 2>&1; then
  if port_busy "$REDIS_PORT"; then
    fail "port $REDIS_PORT is busy but is not answering as Redis."
  fi
  say "starting private Redis on 127.0.0.1:$REDIS_PORT (no persistence)"
  redis-server --port "$REDIS_PORT" --bind 127.0.0.1 --save "" --appendonly no \
    --daemonize yes --dir "$RUN" --pidfile "$RUN/redis.pid" --logfile "$LOGS/redis.log"
  for _ in 1 2 3 4 5 6 7 8 9 10; do redis_ok && break; sleep 0.3; done
  if redis_ok; then
    touch "$RUN/redis.started_by_us"
    USE_CELERY=1
  else
    say "WARNING: Redis did not start; using the in-process job fallback."
  fi
else
  say "Redis not installed: using the documented in-process fallback (jobs are lost on restart)."
fi
if [[ $USE_CELERY -eq 1 ]]; then
  export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:$REDIS_PORT/0}"
else
  export REDIS_URL="redis://127.0.0.1:1/0"   # unreachable on purpose -> in-memory job store
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
export TORCH_THREADS="${TORCH_THREADS:-1}" SPATIAL_BATCH="${SPATIAL_BATCH:-4}"

if [[ $USE_CELERY -eq 1 ]]; then
  say "starting Celery worker"
  nohup celery -A worker.celery_app worker --loglevel=INFO --concurrency=1 --pool=solo \
    >"$LOGS/worker.log" 2>&1 &
  echo $! > "$RUN/worker.pid"
fi

say "starting API on $HOST:$PORT"
nohup uvicorn main:app --host "$HOST" --port "$PORT" >"$LOGS/api.log" 2>&1 &
echo $! > "$RUN/api.pid"

URL_HOST="$HOST"
[[ "$HOST" == "0.0.0.0" ]] && URL_HOST="127.0.0.1"
for _ in $(seq 1 90); do
  if ! alive "$RUN/api.pid"; then
    tail -n 15 "$LOGS/api.log" >&2
    fail "API exited during startup (see data/logs/api.log)"
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "http://$URL_HOST:$PORT/api/v1/health" >/dev/null 2>&1 && break
  else
    port_busy "$PORT" && break
  fi
  sleep 1
done
say "DEEPGUARD AI is up."
say "Dashboard:  http://$URL_HOST:$PORT/"
say "API docs:   http://$URL_HOST:$PORT/api/docs   (disabled when ENVIRONMENT=production)"
say "Logs: data/logs/    Stop: ./stop.sh    Check: ./health_check.sh"
