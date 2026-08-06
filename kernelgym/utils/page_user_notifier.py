"""Small, fail-safe client for the ``page_user`` MCP tool.

The worker only needs one outbound MCP operation when a physical GPU is
quarantined, so pulling a full MCP SDK into the runtime would add unnecessary
surface area.  This module implements that operation over Streamable HTTP and
keeps the credential in a mode-restricted JSON file.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

import aiohttp

from kernelgym.utils.gpu_quarantine import (
    GPUQuarantineNotificationClaim,
    UNLATCHED_NOTIFICATION_PROVENANCE,
    acquire_gpu_quarantine_notification_claim,
    finish_gpu_quarantine_notification_claim,
    release_gpu_quarantine_notification_claim,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH_ENV = "KERNELGYM_PAGE_USER_MCP_CONFIG"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / ".secrets" / "page_user_mcp.json"
MODERN_PROTOCOL_VERSION = "2026-07-28"
LEGACY_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 30.0
MAX_CONFIG_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 256 * 1024
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024
_CLIENT_INFO = {"name": "kernelgym", "version": "0.1.0"}
_PROTOCOL_VERSION_PATTERN = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")


@dataclass(frozen=True)
class PageUserNotificationOutcome:
    """Sanitized result suitable for logging or durable notification state."""

    success: bool
    protocol_version: Optional[str] = None
    error_kind: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class _PageUserConfig:
    url: str = field(repr=False)
    authorization: str = field(repr=False)
    timeout_seconds: float
    agent: str
    host: str
    session: str
    tag: str


@dataclass(frozen=True)
class _HttpReply:
    status: int
    headers: Mapping[str, str]
    body: str


class _NotificationError(RuntimeError):
    def __init__(self, kind: str, message: str, *, legacy_fallback: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.legacy_fallback = legacy_fallback


def _config_path(config_path: Optional[os.PathLike[str] | str]) -> Path:
    if config_path is not None:
        return Path(config_path)
    configured = os.environ.get(CONFIG_PATH_ENV)
    return Path(configured) if configured else DEFAULT_CONFIG_PATH


async def _run_in_thread_to_completion(function: Any, /, *args: Any, **kwargs: Any) -> tuple[Any, bool]:
    """Never abandon a thread that may own or mutate an advisory lock."""

    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancellation_requested = False
    while not operation.done():
        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            cancellation_requested = True
    try:
        return operation.result(), cancellation_requested
    except BaseException:
        if cancellation_requested:
            raise asyncio.CancelledError from None
        raise


async def _release_claim_to_completion(claim: GPUQuarantineNotificationClaim) -> None:
    _, cancellation_requested = await _run_in_thread_to_completion(
        release_gpu_quarantine_notification_claim,
        claim,
    )
    if cancellation_requested:
        raise asyncio.CancelledError


def _load_config(config_path: Optional[os.PathLike[str] | str]) -> _PageUserConfig:
    path = _config_path(config_path)
    parent_fd: Optional[int] = None
    config_fd: Optional[int] = None
    try:
        parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(path.parent, parent_flags)
        parent_stat = os.fstat(parent_fd)
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise _NotificationError("config_invalid", f"page-user MCP config parent is not a directory: {path}")
        if path.parent.name == ".secrets" and parent_stat.st_mode & 0o077:
            raise _NotificationError(
                "config_permissions",
                f"page-user MCP .secrets directory must not grant group/world permissions: {path.parent}",
            )
        config_flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        config_fd = os.open(path.name, config_flags, dir_fd=parent_fd)
    except FileNotFoundError as exc:
        raise _NotificationError("config_missing", f"page-user MCP config does not exist: {path}") from exc
    except _NotificationError:
        raise
    except OSError as exc:
        raise _NotificationError("config_invalid", f"unable to securely open page-user MCP config: {path}") from exc
    finally:
        if parent_fd is not None:
            os.close(parent_fd)

    try:
        file_stat = os.fstat(config_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise _NotificationError("config_invalid", f"page-user MCP config is not a regular file: {path}")
        if file_stat.st_nlink != 1:
            raise _NotificationError("config_invalid", f"page-user MCP config must have one hard link: {path}")
        if file_stat.st_mode & 0o077:
            raise _NotificationError(
                "config_permissions",
                f"page-user MCP config must not grant group/world permissions: {path}",
            )
        with os.fdopen(config_fd, "r", encoding="utf-8") as handle:
            config_fd = None
            raw_payload = handle.read(MAX_CONFIG_BYTES + 1)
        if len(raw_payload) > MAX_CONFIG_BYTES:
            raise _NotificationError("config_invalid", "page-user MCP config is too large")
        payload = json.loads(raw_payload)
    except _NotificationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise _NotificationError("config_invalid", f"unable to read page-user MCP config: {path}") from exc
    finally:
        if config_fd is not None:
            os.close(config_fd)
    if not isinstance(payload, dict):
        raise _NotificationError("config_invalid", "page-user MCP config must contain a JSON object")

    url = payload.get("url")
    authorization = payload.get("authorization") or payload.get("Authorization")
    if authorization is None and isinstance(payload.get("http_headers"), dict):
        authorization = payload["http_headers"].get("Authorization")
    try:
        parsed_url = urlsplit(url) if isinstance(url, str) else None
    except ValueError:
        parsed_url = None
    if parsed_url is None or parsed_url.scheme != "https" or not parsed_url.hostname:
        raise _NotificationError("config_invalid", "page-user MCP config requires an HTTPS url")
    if not isinstance(authorization, str) or not authorization.strip():
        raise _NotificationError("config_invalid", "page-user MCP config requires authorization")

    raw_timeout = payload.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(raw_timeout, bool):
        raise _NotificationError("config_invalid", "page-user MCP timeout_seconds must be numeric")
    try:
        timeout_seconds = float(raw_timeout)
    except (TypeError, ValueError) as exc:
        raise _NotificationError("config_invalid", "page-user MCP timeout_seconds must be numeric") from exc
    if not 0.1 <= timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise _NotificationError(
            "config_invalid",
            f"page-user MCP timeout_seconds must be between 0.1 and {MAX_TIMEOUT_SECONDS:g}",
        )

    return _PageUserConfig(
        url=url,
        authorization=authorization,
        timeout_seconds=timeout_seconds,
        agent=str(payload.get("agent") or "kernelgym"),
        host=str(payload.get("host") or ""),
        session=str(payload.get("session") or "kernelgym-gpu-quarantine"),
        tag=str(payload.get("tag") or "default"),
    )


def _redact(message: object, config: _PageUserConfig) -> str:
    text = str(message)
    parsed_url = urlsplit(config.url)
    secrets = {
        config.authorization,
        config.url,
        urlunsplit((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.query, "")),
        urlunsplit((parsed_url.scheme, parsed_url.netloc, parsed_url.path, "", "")),
        parsed_url.netloc,
        parsed_url.hostname or "",
        parsed_url.username or "",
        unquote(parsed_url.username or ""),
        parsed_url.password or "",
        unquote(parsed_url.password or ""),
        parsed_url.path if len(parsed_url.path) > 1 else "",
        parsed_url.query,
        parsed_url.fragment,
    }
    for _, value in parse_qsl(parsed_url.query, keep_blank_values=True):
        secrets.add(value)
        secrets.add(unquote(value))
    if " " in config.authorization:
        secrets.add(config.authorization.split(" ", 1)[1])
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    # A server response is untrusted input; keep failure state bounded as well.
    return text[:500]


def _base_headers(config: _PageUserConfig) -> dict[str, str]:
    headers = {
        "Authorization": config.authorization,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "X-Agent": config.agent,
    }
    if config.host:
        headers["X-Host"] = config.host
    return headers


async def _post_json(
    client: aiohttp.ClientSession,
    config: _PageUserConfig,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
) -> _HttpReply:
    async with client.post(
        config.url,
        json=payload,
        headers=headers,
        allow_redirects=False,
    ) as response:
        body_bytes = bytearray()
        while len(body_bytes) <= MAX_RESPONSE_BYTES:
            remaining = MAX_RESPONSE_BYTES + 1 - len(body_bytes)
            chunk = await response.content.read(min(_RESPONSE_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            body_bytes.extend(chunk)
        if len(body_bytes) > MAX_RESPONSE_BYTES:
            raise _NotificationError(
                "response_error",
                f"page-user MCP response exceeded the {MAX_RESPONSE_BYTES}-byte limit",
            )
        try:
            body = body_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _NotificationError("response_error", "page-user MCP returned invalid UTF-8") from exc
        return _HttpReply(
            status=response.status,
            headers={str(key).lower(): str(value) for key, value in response.headers.items()},
            body=body,
        )


def _decode_rpc_body(body: str, request_id: int) -> Mapping[str, Any]:
    stripped = body.strip()
    if not stripped:
        raise _NotificationError("response_error", "page-user MCP returned an empty response")

    candidates: list[Any] = []
    if stripped.startswith("data:") or "\ndata:" in stripped:
        for event in stripped.split("\n\n"):
            data_lines = [line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")]
            if not data_lines:
                continue
            try:
                candidates.append(json.loads("\n".join(data_lines)))
            except json.JSONDecodeError:
                continue
    else:
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise _NotificationError("response_error", "page-user MCP returned malformed JSON") from exc
        candidates.extend(decoded if isinstance(decoded, list) else [decoded])

    for candidate in reversed(candidates):
        if isinstance(candidate, dict) and candidate.get("id") == request_id:
            return candidate
    raise _NotificationError("response_error", "page-user MCP response did not match the request")


def _rpc_result(
    reply: _HttpReply,
    request_id: int,
    config: _PageUserConfig,
    *,
    modern_attempt: bool = False,
) -> Mapping[str, Any]:
    fallback_status = modern_attempt and reply.status in {400, 404, 405, 406, 415}
    try:
        response = _decode_rpc_body(reply.body, request_id)
    except _NotificationError:
        if not 200 <= reply.status < 300:
            raise _NotificationError(
                "http_error",
                f"page-user MCP returned HTTP {reply.status}",
                legacy_fallback=fallback_status,
            )
        raise

    if not 200 <= reply.status < 300:
        server_error = response.get("error")
        detail = server_error.get("message") if isinstance(server_error, dict) else f"HTTP {reply.status}"
        raise _NotificationError(
            "http_error",
            f"page-user MCP returned HTTP {reply.status}: {_redact(detail, config)}",
            legacy_fallback=fallback_status,
        )

    rpc_error = response.get("error")
    if isinstance(rpc_error, dict):
        code = rpc_error.get("code")
        message = _redact(rpc_error.get("message", "JSON-RPC error"), config)
        normalized = message.lower()
        fallback_error = modern_attempt and (
            code in {-32022, -32002}
            or "unsupported protocol" in normalized
            or "not initialized" in normalized
            or "initialize first" in normalized
            or "session required" in normalized
            or "missing session" in normalized
        )
        raise _NotificationError(
            "rpc_error",
            f"page-user MCP JSON-RPC error {code}: {message}",
            legacy_fallback=fallback_error,
        )

    result = response.get("result")
    if not isinstance(result, dict):
        raise _NotificationError("response_error", "page-user MCP response has no result object")
    return result


def _check_tool_result(result: Mapping[str, Any], config: _PageUserConfig) -> None:
    if not result.get("isError"):
        return
    detail = "page-user tool reported an error"
    content = result.get("content")
    if isinstance(content, list):
        text_parts = [item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        if text_parts:
            detail = " ".join(str(part) for part in text_parts)
    raise _NotificationError("tool_error", _redact(detail, config))


async def _send_modern(
    client: aiohttp.ClientSession,
    config: _PageUserConfig,
    arguments: Mapping[str, Any],
) -> None:
    request_id = 1
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {
            "name": "page_user",
            "arguments": dict(arguments),
            "_meta": {"io.modelcontextprotocol/clientInfo": _CLIENT_INFO},
        },
    }
    headers = {
        **_base_headers(config),
        "MCP-Protocol-Version": MODERN_PROTOCOL_VERSION,
        "Mcp-Method": "tools/call",
        "Mcp-Name": "page_user",
    }
    result = _rpc_result(
        await _post_json(client, config, payload, headers),
        request_id,
        config,
        modern_attempt=True,
    )
    _check_tool_result(result, config)


async def _send_legacy(
    client: aiohttp.ClientSession,
    config: _PageUserConfig,
    arguments: Mapping[str, Any],
) -> str:
    initialize_id = 10
    initialize_payload = {
        "jsonrpc": "2.0",
        "id": initialize_id,
        "method": "initialize",
        "params": {
            "protocolVersion": LEGACY_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": _CLIENT_INFO,
        },
    }
    initialize_reply = await _post_json(client, config, initialize_payload, _base_headers(config))
    initialize_result = _rpc_result(initialize_reply, initialize_id, config)
    negotiated_version = str(initialize_result.get("protocolVersion") or LEGACY_PROTOCOL_VERSION)
    if not _PROTOCOL_VERSION_PATTERN.fullmatch(negotiated_version):
        raise _NotificationError("response_error", "page-user MCP returned an invalid protocol version")
    session_id = initialize_reply.headers.get("mcp-session-id")
    session_headers = {
        **_base_headers(config),
        "MCP-Protocol-Version": negotiated_version,
    }
    if session_id:
        session_headers["Mcp-Session-Id"] = session_id

    initialized_payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    initialized_reply = await _post_json(client, config, initialized_payload, session_headers)
    if not 200 <= initialized_reply.status < 300:
        raise _NotificationError(
            "http_error",
            f"page-user MCP initialized notification returned HTTP {initialized_reply.status}",
        )

    call_id = 11
    call_payload = {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {"name": "page_user", "arguments": dict(arguments)},
    }
    result = _rpc_result(await _post_json(client, config, call_payload, session_headers), call_id, config)
    _check_tool_result(result, config)
    return negotiated_version


async def send_page_user_notification(
    message: str,
    *,
    title: str = "",
    agent: Optional[str] = None,
    host: Optional[str] = None,
    session: Optional[str] = None,
    tag: Optional[str] = None,
    config_path: Optional[os.PathLike[str] | str] = None,
) -> PageUserNotificationOutcome:
    """Send one phone notification without ever raising into worker recovery."""

    if not isinstance(message, str) or not message.strip():
        return PageUserNotificationOutcome(False, error_kind="input_error", error="notification message is empty")
    try:
        config = await asyncio.to_thread(_load_config, config_path)
    except _NotificationError as exc:
        return PageUserNotificationOutcome(False, error_kind=exc.kind, error=str(exc))
    except Exception as exc:
        return PageUserNotificationOutcome(
            False,
            error_kind="config_invalid",
            error=f"unexpected config loader error: {type(exc).__name__}",
        )

    arguments = {
        "message": message,
        "agent": agent or config.agent,
        "host": host or config.host or socket.gethostname(),
        "session": session or config.session,
        "tag": tag or config.tag,
    }
    if title:
        arguments["title"] = title

    try:
        timeout = aiohttp.ClientTimeout(total=config.timeout_seconds)
        async with asyncio.timeout(config.timeout_seconds):
            async with aiohttp.ClientSession(timeout=timeout) as client:
                try:
                    await _send_modern(client, config, arguments)
                    return PageUserNotificationOutcome(True, protocol_version=MODERN_PROTOCOL_VERSION)
                except _NotificationError as modern_error:
                    if not modern_error.legacy_fallback:
                        raise
                negotiated_version = await _send_legacy(client, config, arguments)
                return PageUserNotificationOutcome(True, protocol_version=negotiated_version)
    except TimeoutError:
        return PageUserNotificationOutcome(False, error_kind="timeout", error="page-user MCP request timed out")
    except _NotificationError as exc:
        return PageUserNotificationOutcome(False, error_kind=exc.kind, error=_redact(exc, config))
    except (aiohttp.ClientError, OSError) as exc:
        # Exception types are useful operationally; their free-form text may
        # contain request material, so only expose a sanitized, bounded form.
        return PageUserNotificationOutcome(
            False,
            error_kind="transport_error",
            error=_redact(f"{type(exc).__name__}: {exc}", config),
        )
    except Exception as exc:
        return PageUserNotificationOutcome(
            False,
            error_kind="unexpected_error",
            error=_redact(f"{type(exc).__name__}: {exc}", config),
        )


async def _send_claimed_gpu_notification(
    record: Mapping[str, Any],
    *,
    expected_scope: str,
    content_builder: Callable[[Mapping[str, Any]], tuple[str, str, str]],
    config_path: Optional[os.PathLike[str] | str] = None,
) -> PageUserNotificationOutcome:
    """Deliver one durable, cross-process-deduplicated GPU safety page."""

    claim: Optional[GPUQuarantineNotificationClaim] = None
    scope = record.get("scope")
    if scope != expected_scope:
        return PageUserNotificationOutcome(
            False,
            error_kind="input_error",
            error=f"GPU notification requires scope={expected_scope}",
        )
    unlatched_best_effort = record.get("notification_provenance") == UNLATCHED_NOTIFICATION_PROVENANCE
    if not unlatched_best_effort:
        try:
            claim, cancellation_requested = await _run_in_thread_to_completion(
                acquire_gpu_quarantine_notification_claim,
                record,
            )
            if cancellation_requested:
                await _release_claim_to_completion(claim)
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Only records explicitly tagged by the persistence-failure path
            # may bypass durable dedupe. Treat every other claim failure as a
            # retryable notification failure instead of risking duplicates.
            return PageUserNotificationOutcome(
                False,
                error_kind="claim_error",
                error=f"unable to acquire durable notification claim: {type(exc).__name__}",
            )

    if claim is not None and not claim.should_send:
        protocol_version = "superseded" if claim.superseded else "deduplicated"
        await _release_claim_to_completion(claim)
        return PageUserNotificationOutcome(True, protocol_version=protocol_version)

    notification_record: Mapping[str, Any] = claim.record if claim is not None else record
    message, title, hostname = content_builder(notification_record)
    try:
        outcome = await send_page_user_notification(
            message,
            title=title,
            host=hostname,
            config_path=config_path,
        )
    except BaseException:
        if claim is not None:
            try:
                _, cancellation_requested = await _run_in_thread_to_completion(
                    finish_gpu_quarantine_notification_claim,
                    claim,
                    state="failed",
                    error="delivery interrupted before completion",
                )
                if cancellation_requested:
                    raise asyncio.CancelledError
            except Exception:
                pass
        raise
    else:
        if claim is not None:
            error = "" if outcome.success else f"{outcome.error_kind or 'unknown'}: {outcome.error or ''}"
            try:
                _, cancellation_requested = await _run_in_thread_to_completion(
                    finish_gpu_quarantine_notification_claim,
                    claim,
                    state="sent" if outcome.success else "failed",
                    error=error,
                )
                if cancellation_requested:
                    raise asyncio.CancelledError
            except Exception:
                # The outer worker/monitor records the same outcome in Redis
                # and durable state. Never turn a confirmed delivery into a
                # retry merely because this early durable update failed.
                pass
        return outcome
    finally:
        if claim is not None:
            await _release_claim_to_completion(claim)


def _physical_notification_content(record: Mapping[str, Any]) -> tuple[str, str, str]:
    hostname = str(record.get("hostname") or socket.gethostname())
    device = str(record.get("device") or "unknown-device")
    worker_id = str(record.get("worker_id") or "unknown-worker")
    fault_class = str(record.get("fault_class") or "unknown")
    message = (
        f"{hostname} {device} removed from KernelGYM scheduling\n"
        f"- worker: {worker_id}\n"
        f"- fault: {fault_class}\n"
        "- manual clear required"
    )
    return message, "KernelGYM GPU quarantined", hostname


def _worker_notification_content(record: Mapping[str, Any]) -> tuple[str, str, str]:
    hostname = str(record.get("hostname") or socket.gethostname())
    device = str(record.get("device") or "unknown-device")
    worker_id = str(record.get("worker_id") or "unknown-worker")
    fault_class = str(record.get("fault_class") or "unknown")
    if fault_class == "restart_limit":
        headline = f"{hostname} {device} GPU worker removed after restart limit"
    else:
        headline = f"{hostname} {device} GPU worker removed from KernelGYM scheduling"
    message = (
        f"{headline}\n"
        f"- worker: {worker_id}\n"
        f"- fault: {fault_class}\n"
        "- physical GPU fault: not yet proven\n"
        "- scheduling: disabled for this worker until manual clear"
    )
    return message, "KernelGYM GPU worker excluded", hostname


async def send_gpu_quarantine_page(
    record: Mapping[str, Any],
    *,
    config_path: Optional[os.PathLike[str] | str] = None,
) -> PageUserNotificationOutcome:
    """Send one cross-process-deduplicated physical-GPU quarantine page."""

    return await _send_claimed_gpu_notification(
        record,
        expected_scope="physical_gpu",
        content_builder=_physical_notification_content,
        config_path=config_path,
    )


async def send_gpu_worker_exclusion_page(
    record: Mapping[str, Any],
    *,
    config_path: Optional[os.PathLike[str] | str] = None,
) -> PageUserNotificationOutcome:
    """Notify that a CUDA worker, but not a proven-bad physical GPU, was excluded."""

    return await _send_claimed_gpu_notification(
        record,
        expected_scope="worker_process",
        content_builder=_worker_notification_content,
        config_path=config_path,
    )


__all__ = [
    "CONFIG_PATH_ENV",
    "DEFAULT_CONFIG_PATH",
    "PageUserNotificationOutcome",
    "send_gpu_quarantine_page",
    "send_gpu_worker_exclusion_page",
    "send_page_user_notification",
]
