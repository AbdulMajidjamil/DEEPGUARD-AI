# SPDX-License-Identifier: Apache-2.0
# Originally part of SENTINEL, Copyright 2026 Dev Parth.
# Modified 2026 by Abdul Majid Jamil for DEEPGUARD AI: in-memory TTL eviction, redis failover (see DERIVATION.md).
"""Redis-backed job dictionary keyed by task_id (with documented in-process fallback)."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

from config import JOB_TTL_SECONDS, REDIS_URL
from security import redact_url, valid_task_id

logger = logging.getLogger("deepguard.jobstore")


class JobStore:
    def __init__(self) -> None:
        self._redis = None
        self._mem: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.backend = "memory"
        try:
            import redis

            client = redis.Redis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=1.5)
            client.ping()
            self._redis = client
            self.backend = "redis"
            logger.info("JobStore connected to %s", redact_url(REDIS_URL))
        except Exception as exc:
            logger.warning("Redis unavailable (%s) - using in-process job dictionary", type(exc).__name__)

    @property
    def redis_ok(self) -> bool:
        return self._redis is not None

    def put(self, task_id: str, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("task_id", task_id)
        payload["updated_at"] = time.time()
        if self._redis is not None:
            try:
                self._redis.set(f"job:{task_id}", json.dumps(payload), ex=JOB_TTL_SECONDS)
                return
            except Exception as exc:  # Redis went away mid-run: degrade, never crash a job
                logger.warning("Redis write failed (%s) - switching to in-process job dictionary", type(exc).__name__)
                self._redis = None
                self.backend = "memory"
        with self._lock:
            self._mem[task_id] = payload
            self._evict_locked()

    def _evict_locked(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        stale = [k for k, v in self._mem.items() if v.get("updated_at", 0) < cutoff]
        for key in stale:
            self._mem.pop(key, None)

    def merge(self, task_id: str, **fields: Any) -> dict[str, Any]:
        current = self.get(task_id) or {"task_id": task_id}
        current.update(fields)
        self.put(task_id, current)
        return current

    def get(self, task_id: str) -> Optional[dict[str, Any]]:
        if not valid_task_id(task_id):
            return None
        if self._redis is not None:
            try:
                raw = self._redis.get(f"job:{task_id}")
                if not raw:
                    return None
                return json.loads(raw)
            except Exception as exc:
                logger.warning("Redis read failed (%s) - switching to in-process job dictionary", type(exc).__name__)
                self._redis = None
                self.backend = "memory"
        with self._lock:
            data = self._mem.get(task_id)
            return dict(data) if data else None


store = JobStore()
