"""Reference runtime cache provider for KernelBench workflows."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import redis as redis_sync
except Exception:  # pragma: no cover - dependency is optional at import time
    redis_sync = None

logger = logging.getLogger("kernelgym.api")

_UUID_PATHS = (
    "uuid",
    "problem_id",
    "metadata.uuid",
    "metadata.problem_id",
    "extra_info.uuid",
    "extra_info.problem_id",
    "result.uuid",
    "result.problem_id",
    "result.metadata.uuid",
    "result.metadata.problem_id",
)
_RUNTIME_PATHS = (
    "reference_runtime",
    "runtime",
    "result.reference_runtime",
    "result.runtime",
)
_REFERENCE_CODE_PATHS = (
    "reference_code",
    "ground_truth",
    "reward_model.ground_truth",
    "result.reference_code",
)


def _code_hash(reference_code: str) -> Optional[str]:
    if not reference_code:
        return None
    return hashlib.sha256(reference_code.encode("utf-8")).hexdigest()


def _nested_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _extract_first(payload: dict[str, Any], paths: Iterable[str]) -> Any:
    for path in paths:
        value = _nested_get(payload, path)
        if value not in (None, ""):
            return value
    return None


class ReferenceRuntimeCache:
    """Shared reference-runtime cache with local memory and optional Redis backing."""

    def __init__(
        self,
        *,
        redis_url: str | None = None,
        redis_prefix: str = "kernelgym:reference_cache",
    ) -> None:
        self._memory: dict[str, dict[str, Any]] = {}
        self._redis_prefix = redis_prefix
        self._preloaded_entries = 0
        self._redis = None

        if redis_url and redis_sync is not None:
            try:
                self._redis = redis_sync.Redis.from_url(
                    redis_url,
                    decode_responses=True,
                )
                self._redis.ping()
            except Exception as exc:  # pragma: no cover - depends on runtime infra
                logger.warning("Reference cache Redis unavailable: %s", exc)
                self._redis = None

    def _entry_key(self, uuid: str, is_valid: bool) -> str:
        namespace = "val" if is_valid else "train"
        return f"{namespace}:{uuid}"

    def _redis_key(self, uuid: str, is_valid: bool) -> str:
        return f"{self._redis_prefix}:{self._entry_key(uuid, is_valid)}"

    def _is_compatible(
        self,
        entry: dict[str, Any],
        reference_code: str,
    ) -> bool:
        expected_hash = _code_hash(reference_code)
        cached_hash = entry.get("code_hash")
        if expected_hash and cached_hash and expected_hash != cached_hash:
            return False
        return True

    def get(
        self,
        uuid: Optional[str],
        reference_code: str,
        is_valid: bool,
    ) -> Optional[float]:
        if not uuid:
            return None

        uuid_str = str(uuid)
        entry_key = self._entry_key(uuid_str, is_valid)
        entry = self._memory.get(entry_key)

        if entry is None and self._redis is not None:
            try:
                payload = self._redis.get(self._redis_key(uuid_str, is_valid))
            except Exception as exc:  # pragma: no cover - depends on runtime infra
                logger.warning("Reference cache Redis get failed for %s: %s", uuid_str, exc)
                payload = None

            if payload:
                try:
                    entry = json.loads(payload)
                except json.JSONDecodeError:
                    entry = None
                if entry is not None:
                    self._memory[entry_key] = entry

        if not entry or not self._is_compatible(entry, reference_code):
            return None

        try:
            runtime = float(entry["reference_runtime"])
        except Exception:
            return None
        return runtime if runtime > 0 else None

    def put(
        self,
        uuid: Optional[str],
        reference_code: str,
        is_valid: bool,
        runtime: Optional[float],
    ) -> None:
        if not uuid or runtime is None:
            return

        try:
            runtime_value = float(runtime)
        except Exception:
            return

        if runtime_value <= 0:
            return

        uuid_str = str(uuid)
        entry = {
            "reference_runtime": runtime_value,
            "code_hash": _code_hash(reference_code),
        }
        entry_key = self._entry_key(uuid_str, is_valid)
        self._memory[entry_key] = entry

        if self._redis is not None:
            try:
                self._redis.set(self._redis_key(uuid_str, is_valid), json.dumps(entry))
            except Exception as exc:  # pragma: no cover - depends on runtime infra
                logger.warning("Reference cache Redis put failed for %s: %s", uuid_str, exc)

    def preload(self, dataset_path: str, *, is_valid: bool) -> int:
        path = Path(dataset_path)
        if not dataset_path or not path.exists():
            return 0

        loaded = 0
        try:
            for record in self._iter_records(path):
                if not isinstance(record, dict):
                    continue
                uuid = _extract_first(record, _UUID_PATHS)
                runtime = _extract_first(record, _RUNTIME_PATHS)
                reference_code = _extract_first(record, _REFERENCE_CODE_PATHS) or ""
                before = self.get(str(uuid) if uuid is not None else None, reference_code, is_valid)
                self.put(str(uuid) if uuid is not None else None, reference_code, is_valid, runtime)
                after = self.get(str(uuid) if uuid is not None else None, reference_code, is_valid)
                if before is None and after is not None:
                    loaded += 1
        except Exception as exc:
            logger.warning("Reference cache preload failed for %s: %s", path, exc)
            return 0

        self._preloaded_entries += loaded
        return loaded

    def describe(self) -> dict[str, Any]:
        return {
            "memory_entries": len(self._memory),
            "preloaded_entries": self._preloaded_entries,
            "redis_enabled": self._redis is not None,
        }

    def _iter_records(self, path: Path) -> Iterable[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
            return

        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                for record in payload:
                    if isinstance(record, dict):
                        yield record
            elif isinstance(payload, dict):
                records = payload.get("records")
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            yield record
                else:
                    yield payload
            return
        except Exception:
            pass

        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            for record in table.to_pylist():
                if isinstance(record, dict):
                    yield record
        except Exception as exc:
            raise RuntimeError(f"unsupported cache dataset format: {path}") from exc


def build_reference_runtime_cache(
    *,
    redis_url: str | None,
    redis_key_prefix: str,
    reference_cache_dataset_path: str = "",
    val_data_cache_dataset_path: str = "",
) -> ReferenceRuntimeCache:
    cache = ReferenceRuntimeCache(
        redis_url=redis_url,
        redis_prefix=f"{redis_key_prefix}:reference_runtime_cache",
    )
    if reference_cache_dataset_path:
        cache.preload(reference_cache_dataset_path, is_valid=False)
    if val_data_cache_dataset_path:
        cache.preload(val_data_cache_dataset_path, is_valid=True)
    return cache
