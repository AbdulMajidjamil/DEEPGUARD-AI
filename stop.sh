#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Abdul Majid Jamil
# Stop only the processes ./start.sh started (PIDs in data/run/). A PID is signalled only
# if it is still alive AND its command line matches what we launched, so a recycled PID
# or an unrelated program is never touched. Data, uploads and weights are left alone.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RUN="data/run"
say() { printf '[stop] %s\n' "$*"; }

stop_pid() {  # name pidfile expected-substring-in-cmdline
  local name="$1" file="$2" pattern="$3" pid cmd
  if [[ ! -f "$file" ]]; then
    say "$name: not started by start.sh"
    return
  fi
  pid="$(cat "$file" 2>/dev/null)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    say "$name: not running"
    rm -f "$file"
    return
  fi
  cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || ps -o args= -p "$pid" 2>/dev/null)"
  if [[ "$cmd" != *"$pattern"* ]]; then
    say "$name: pid $pid is no longer ours; leaving it alone"
    rm -f "$file"
    return
  fi
  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    say "$name: pid $pid ignored SIGTERM; not forcing (stop it manually if needed)"
    return
  fi
  say "$name: stopped"
  rm -f "$file"
}

stop_pid "api"    "$RUN/api.pid"    "uvicorn"
stop_pid "worker" "$RUN/worker.pid" "celery"

if [[ -f "$RUN/redis.started_by_us" ]]; then
  stop_pid "redis (private)" "$RUN/redis.pid" "redis-server"
  rm -f "$RUN/redis.started_by_us"
else
  say "redis: not started by start.sh; left untouched"
fi
