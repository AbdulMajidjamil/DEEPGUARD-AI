#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
# Read-only check of a running DEEPGUARD AI. Exit 0 = reachable, 1 = unreachable.
#   PORT=8000 ./health_check.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
HOST="${HOST:-127.0.0.1}"
[[ "$HOST" == "0.0.0.0" ]] && HOST=127.0.0.1
PORT="${PORT:-8000}"
URL="http://$HOST:$PORT/api/v1/health"
command -v curl >/dev/null 2>&1 || { echo "curl is required (sudo apt-get install curl)"; exit 1; }

body="$(curl -fsS --max-time 8 "$URL" 2>/dev/null)" || { echo "FAIL  API not reachable at $URL"; exit 1; }
echo "OK    API reachable at $URL"
if python3 -c "import json" >/dev/null 2>&1; then
  printf '%s' "$body" | python3 -c '
import json, sys
h = json.load(sys.stdin)
print("      ok            :", h.get("ok"), "| environment:", h.get("environment"))
print("      job backend   :", h.get("job_backend"))
print("      model         :", h.get("inference_mode"), "(DEMO - no trained checkpoint)" if h.get("is_demo") else "")
print("      redis         :", h.get("redis"))
print("      celery worker :", h.get("celery_worker"), "| dispatch:", h.get("dispatch"))
print("      version       :", h.get("version"))
'
else
  echo "      $body"
fi
for name in api worker; do
  pf="data/run/$name.pid"
  if [[ -f "$pf" ]] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    echo "OK    $name process running (pid $(cat "$pf"))"
  fi
done
exit 0
