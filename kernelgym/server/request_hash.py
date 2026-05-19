"""Canonical request hashing for server-side result cache validation."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


REQUEST_HASH_IGNORED_KEYS = {
    "assigned_device",
    "completed_at",
    "estimated_completion",
    "force_refresh",
    "line_index",
    "metadata",
    "model_id",
    "output_index",
    "progress",
    "queue_position",
    "run_id",
    "status",
    "submitted_at",
    "task_id",
    "turn_id",
}


def canonicalize_for_request_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: canonicalize_for_request_hash(item)
            for key, item in sorted(value.items())
            if key not in REQUEST_HASH_IGNORED_KEYS
        }
    if isinstance(value, list):
        return [canonicalize_for_request_hash(item) for item in value]
    if isinstance(value, tuple):
        return [canonicalize_for_request_hash(item) for item in value]
    return value


def request_hash(workflow_name: str, payload: Dict[str, Any]) -> str:
    canonical = {
        "workflow": workflow_name or "kernelbench",
        "payload": canonicalize_for_request_hash(payload),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()
