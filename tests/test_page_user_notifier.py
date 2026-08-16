from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import stat
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from kernelgym.utils import gpu_quarantine
from kernelgym.utils import page_user_notifier as notifier


AUTHORIZATION = "Bearer test-secret-that-must-not-leak"


class FakeContent:
    def __init__(self, body: str | bytes) -> None:
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._body):
            return b""
        end = len(self._body) if size < 0 else min(len(self._body), self._offset + size)
        chunk = self._body[self._offset : end]
        self._offset = end
        return chunk


class FakeResponse:
    def __init__(self, status: int, body: str | bytes = "", *, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.content = FakeContent(body)
        self.headers = headers or {}

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None


class FakeSession:
    def __init__(self, responses: list[FakeResponse | BaseException]) -> None:
        self.responses = deque(responses)
        self.requests: list[dict[str, Any]] = []
        self.timeout = None
        self.trust_env = None

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001
        return None

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
        allow_redirects: bool,
        proxy: str | None,
    ) -> FakeResponse:
        self.requests.append(
            {
                "url": url,
                "json": json,
                "headers": dict(headers),
                "allow_redirects": allow_redirects,
                "proxy": proxy,
            }
        )
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response


class TimeoutResponse(FakeResponse):
    def __init__(self, status: int) -> None:
        super().__init__(status)

        async def timeout_read(size: int = -1) -> bytes:  # noqa: ARG001
            raise TimeoutError

        self.content.read = timeout_read


@pytest.fixture(autouse=True)
def _forbid_real_page(monkeypatch, tmp_path):  # noqa: ANN001
    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("test attempted a real page-user request")

    monkeypatch.setenv("KERNELGYM_SAFETY_LATCH_DIR", str(tmp_path / "safety_latches"))
    monkeypatch.setattr(notifier.aiohttp, "ClientSession", fail_if_called)


def _write_config(path: Path, **overrides: Any) -> Path:
    payload = {
        "url": "https://page-user.invalid/mcp",
        "authorization": AUTHORIZATION,
        "timeout_seconds": 2,
        "agent": "kernelgym-test",
        "host": "configured-host",
        "session": "test-session",
        "tag": "test-tag",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _install_fake_session(monkeypatch, responses: list[FakeResponse | BaseException]) -> FakeSession:  # noqa: ANN001
    session = FakeSession(responses)

    def factory(*, timeout, trust_env):  # noqa: ANN001
        session.timeout = timeout
        session.trust_env = trust_env
        return session

    monkeypatch.setattr(notifier.aiohttp, "ClientSession", factory)
    return session


def _rpc_result(request_id: int, *, is_error: bool = False, text: str = "sent") -> str:
    return json.dumps(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": is_error,
            },
        }
    )


def test_missing_default_config_fails_without_opening_network(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.delenv(notifier.CONFIG_PATH_ENV, raising=False)
    monkeypatch.setattr(notifier, "DEFAULT_CONFIG_PATH", tmp_path / ".secrets" / "page_user_mcp.json")

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined"))

    assert outcome.success is False
    assert outcome.error_kind == "config_missing"


def test_environment_override_and_mode_restriction(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "override.json")
    config_path.chmod(0o640)
    monkeypatch.setenv(notifier.CONFIG_PATH_ENV, str(config_path))

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined"))

    assert outcome.success is False
    assert outcome.error_kind == "config_permissions"
    assert AUTHORIZATION not in repr(outcome)


def test_default_secrets_directory_must_be_private(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    secrets_dir = tmp_path / ".secrets"
    secrets_dir.mkdir(mode=0o755)
    secrets_dir.chmod(0o755)
    config_path = _write_config(secrets_dir / "page_user_mcp.json")

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "config_permissions"


def test_symlinked_config_is_rejected_before_network(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    target = _write_config(tmp_path / "actual.json")
    link = tmp_path / "page-user.json"
    link.symlink_to(target)

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=link))

    assert outcome.success is False
    assert outcome.error_kind == "config_invalid"


def test_http_config_is_rejected_before_network(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json", url="http://page-user.invalid/mcp")

    # The autouse fixture makes ClientSession raise, so reaching the network
    # would fail this test rather than merely returning a mocked response.
    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "config_invalid"
    assert outcome.error is not None and "HTTPS" in outcome.error
    assert AUTHORIZATION not in repr(outcome)


def test_config_is_read_from_verified_fd_during_path_replacement(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    replacement = _write_config(
        tmp_path / "replacement.json",
        url="https://attacker.invalid/mcp",
        authorization="Bearer attacker-value",
    )
    real_fstat = notifier.os.fstat
    replaced = False

    def replace_after_fstat(fd):  # noqa: ANN001
        nonlocal replaced
        file_stat = real_fstat(fd)
        if stat.S_ISREG(file_stat.st_mode) and not replaced:
            os.replace(replacement, config_path)
            replaced = True
        return file_stat

    monkeypatch.setattr(notifier.os, "fstat", replace_after_fstat)

    config = notifier._load_config(config_path)

    assert config.authorization == AUTHORIZATION
    assert config.url == "https://page-user.invalid/mcp"
    assert AUTHORIZATION not in repr(config)
    assert config.url not in repr(config)


def test_modern_stateless_tool_call_succeeds_with_expected_envelope(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    proxy = "http://proxy.invalid:8080"
    config_path = _write_config(tmp_path / "page-user.json", proxy=proxy)
    body = "event: message\ndata: " + _rpc_result(1) + "\n\n"
    fake = _install_fake_session(monkeypatch, [FakeResponse(200, body)])

    outcome = asyncio.run(
        notifier.send_page_user_notification(
            "node21 cuda:0 removed",
            title="KernelGYM GPU quarantined",
            host="node21",
            config_path=config_path,
        )
    )

    assert outcome == notifier.PageUserNotificationOutcome(
        success=True,
        protocol_version=notifier.MODERN_PROTOCOL_VERSION,
    )
    assert len(fake.requests) == 1
    request = fake.requests[0]
    assert request["url"] == "https://page-user.invalid/mcp"
    assert request["allow_redirects"] is False
    assert request["proxy"] == proxy
    assert request["headers"]["Authorization"] == AUTHORIZATION
    assert request["headers"]["MCP-Protocol-Version"] == "2026-07-28"
    assert request["headers"]["Mcp-Method"] == "tools/call"
    assert request["headers"]["Mcp-Name"] == "page_user"
    assert request["headers"]["X-Agent"] == "kernelgym-test"
    assert request["headers"]["X-Host"] == "configured-host"
    assert request["json"]["method"] == "tools/call"
    assert request["json"]["params"]["name"] == "page_user"
    assert request["json"]["params"]["arguments"] == {
        "message": "node21 cuda:0 removed",
        "agent": "kernelgym-test",
        "host": "node21",
        "session": "test-session",
        "tag": "test-tag",
        "title": "KernelGYM GPU quarantined",
    }
    assert request["json"]["params"]["_meta"]["io.modelcontextprotocol/clientInfo"]["name"] == "kernelgym"
    assert isinstance(fake.timeout, aiohttp.ClientTimeout)
    assert fake.timeout.total == 2
    assert fake.trust_env is True
    assert AUTHORIZATION not in repr(outcome)


def test_redirect_response_is_not_followed(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    redirect_target = "https://redirect.invalid/credential-capture"
    fake = _install_fake_session(
        monkeypatch,
        [FakeResponse(307, headers={"Location": redirect_target})],
    )

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "http_error"
    assert redirect_target not in repr(outcome)
    assert len(fake.requests) == 1
    assert fake.requests[0]["allow_redirects"] is False


def test_oversized_response_is_bounded_and_not_exposed(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    untrusted_marker = "oversized-server-secret"
    oversized_body = untrusted_marker.encode() + b"x" * notifier.MAX_RESPONSE_BYTES
    fake = _install_fake_session(monkeypatch, [FakeResponse(200, oversized_body)])

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "response_error"
    assert outcome.error == f"page-user MCP response exceeded the {notifier.MAX_RESPONSE_BYTES}-byte limit"
    assert untrusted_marker not in repr(outcome)
    assert AUTHORIZATION not in repr(outcome)
    assert len(fake.requests) == 1


def test_unsupported_modern_protocol_falls_back_to_legacy_session(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    modern_error = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32022, "message": "Unsupported protocol version"},
        }
    )
    initialize = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock", "version": "1"},
            },
        }
    )
    fake = _install_fake_session(
        monkeypatch,
        [
            FakeResponse(200, modern_error),
            FakeResponse(200, initialize, headers={"Mcp-Session-Id": "mock-session-id"}),
            FakeResponse(202),
            FakeResponse(200, _rpc_result(11)),
        ],
    )

    outcome = asyncio.run(
        notifier.send_page_user_notification("GPU quarantined", config_path=config_path, host="node21")
    )

    assert outcome.success is True
    assert outcome.protocol_version == "2025-06-18"
    assert [request["json"]["method"] for request in fake.requests] == [
        "tools/call",
        "initialize",
        "notifications/initialized",
        "tools/call",
    ]
    initialize_request = fake.requests[1]
    assert initialize_request["json"]["params"]["protocolVersion"] == "2025-06-18"
    initialized_request = fake.requests[2]
    tool_request = fake.requests[3]
    assert initialized_request["headers"]["Mcp-Session-Id"] == "mock-session-id"
    assert tool_request["headers"]["Mcp-Session-Id"] == "mock-session-id"
    assert tool_request["headers"]["MCP-Protocol-Version"] == "2025-06-18"


def test_untrusted_protocol_version_cannot_reach_logs_or_outcome(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    modern_error = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32022, "message": "Unsupported protocol version"},
        }
    )
    initialize = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "result": {
                "protocolVersion": AUTHORIZATION,
                "capabilities": {"tools": {}},
            },
        }
    )
    fake = _install_fake_session(
        monkeypatch,
        [FakeResponse(200, modern_error), FakeResponse(200, initialize)],
    )

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "response_error"
    assert AUTHORIZATION not in repr(outcome)
    assert len(fake.requests) == 2


def test_tool_error_does_not_retry_and_redacts_authorization(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    fake = _install_fake_session(
        monkeypatch,
        [FakeResponse(200, _rpc_result(1, is_error=True, text=f"delivery rejected: {AUTHORIZATION}"))],
    )

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "tool_error"
    assert "[REDACTED]" in str(outcome.error)
    assert AUTHORIZATION not in repr(outcome)
    assert len(fake.requests) == 1


def test_transport_error_is_structured_and_secret_is_redacted(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    fake = _install_fake_session(
        monkeypatch,
        [aiohttp.ClientConnectionError(f"mock connection rejected {AUTHORIZATION}")],
    )

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "transport_error"
    assert "[REDACTED]" in str(outcome.error)
    assert AUTHORIZATION not in repr(outcome)
    assert len(fake.requests) == 1


def test_transport_error_redacts_url_credentials_query_and_fragment(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    url = "https://api-user:url-password@page-user.invalid/mcp-secret?token=query-secret#fragment-secret"
    config_path = _write_config(tmp_path / "page-user.json", url=url)
    fake = _install_fake_session(
        monkeypatch,
        [aiohttp.ClientConnectionError(f"failed {url} query-secret fragment-secret url-password")],
    )

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "transport_error"
    assert "[REDACTED]" in str(outcome.error)
    for secret in (url, "page-user.invalid", "query-secret", "fragment-secret", "url-password"):
        assert secret not in repr(outcome)
    assert len(fake.requests) == 1


def test_transport_error_redacts_proxy_credentials(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    proxy = "http://proxy-user:proxy-password@proxy.invalid:8080"
    config_path = _write_config(tmp_path / "page-user.json", proxy=proxy)
    fake = _install_fake_session(
        monkeypatch,
        [aiohttp.ClientConnectionError(f"failed to connect through {proxy}")],
    )

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "transport_error"
    assert "[REDACTED]" in str(outcome.error)
    for secret in (proxy, "proxy.invalid", "proxy-user", "proxy-password"):
        assert secret not in repr(outcome)
    assert len(fake.requests) == 1


def test_invalid_proxy_is_rejected_before_network(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json", proxy="socks5://proxy.invalid:1080")

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "config_invalid"


def test_timeout_is_structured_and_does_not_fall_back(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    fake = _install_fake_session(monkeypatch, [TimeoutResponse(200)])

    outcome = asyncio.run(notifier.send_page_user_notification("GPU quarantined", config_path=config_path))

    assert outcome.success is False
    assert outcome.error_kind == "timeout"
    assert len(fake.requests) == 1


def test_gpu_quarantine_wrapper_builds_glanceable_message(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    config_path = _write_config(tmp_path / "page-user.json")
    fake = _install_fake_session(monkeypatch, [FakeResponse(200, _rpc_result(1))])
    record = {
        "scope": "physical_gpu",
        "hostname": "node21",
        "device": "cuda:3",
        "worker_id": "node21_gpu_3",
        "fault_class": "device_fault",
        "notification_provenance": gpu_quarantine.UNLATCHED_NOTIFICATION_PROVENANCE,
    }

    outcome = asyncio.run(notifier.send_gpu_quarantine_page(record, config_path=config_path))

    assert outcome.success is True
    arguments = fake.requests[0]["json"]["params"]["arguments"]
    assert arguments["title"] == "KernelGYM GPU quarantined"
    assert arguments["host"] == "node21"
    assert arguments["message"] == (
        "node21 cuda:3 removed from KernelGYM scheduling\n"
        "- worker: node21_gpu_3\n"
        "- fault: device_fault\n"
        "- manual clear required"
    )


@pytest.mark.parametrize(
    ("sender", "scope"),
    [
        (notifier.send_gpu_quarantine_page, None),
        (notifier.send_gpu_quarantine_page, "worker_process"),
        (notifier.send_gpu_quarantine_page, "invalid"),
        (notifier.send_gpu_worker_exclusion_page, None),
        (notifier.send_gpu_worker_exclusion_page, "physical_gpu"),
        (notifier.send_gpu_worker_exclusion_page, "invalid"),
    ],
)
def test_gpu_notification_wrapper_rejects_missing_or_wrong_scope(monkeypatch, sender, scope) -> None:  # noqa: ANN001
    calls = 0
    record = {
        "hostname": "node21",
        "device": "cuda:3",
        "worker_id": "node21_gpu_3",
        "notification_provenance": gpu_quarantine.UNLATCHED_NOTIFICATION_PROVENANCE,
    }
    if scope is not None:
        record["scope"] = scope

    async def unexpected_send(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

    def unexpected_claim(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("invalid notification scope reached the durable claim")

    monkeypatch.setattr(notifier, "acquire_gpu_quarantine_notification_claim", unexpected_claim)
    monkeypatch.setattr(notifier, "send_page_user_notification", unexpected_send)

    outcome = asyncio.run(sender(record))

    assert outcome.success is False
    assert outcome.error_kind == "input_error"
    assert calls == 0


def _physical_quarantine_record() -> dict[str, str]:
    return {
        "state": "quarantined",
        "scope": "physical_gpu",
        "hostname": "node21",
        "device": "cuda:3",
        "worker_id": "node21_gpu_3",
        "event_id": "physical-test-event",
        "fault_class": "device_fault",
        "reason": "test fault",
        "page_user_state": "pending",
    }


def _worker_exclusion_record() -> dict[str, str]:
    return {
        "state": "quarantined",
        "scope": "worker_process",
        "hostname": "node21",
        "device": "cuda:3",
        "worker_id": "node21_gpu_3",
        "event_id": "worker-test-event",
        "fault_class": "restart_limit",
        "reason": "worker failed three restart attempts",
        "page_user_state": "not_applicable",
    }


def _persist_notification_record(record: dict[str, str]) -> dict[str, str]:
    persisted = dict(record)
    for path, alias in gpu_quarantine._records_for_durable_paths(persisted, persisted["worker_id"]):
        gpu_quarantine._write_json_atomic(path, alias)
    return persisted


def test_stale_notification_input_after_manual_clear_does_not_recreate_latch(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        calls = 0
        record = _persist_notification_record(_physical_quarantine_record())
        gpu_quarantine._clear_durable_latches(
            record["worker_id"],
            device=record["device"],
            hostname=record["hostname"],
        )

        async def unexpected_send(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", unexpected_send)
        outcome = await notifier.send_gpu_quarantine_page(record)

        assert outcome.protocol_version == "superseded"
        assert calls == 0
        assert (
            gpu_quarantine._read_json(gpu_quarantine._device_latch_path(record["hostname"], record["device"])) is None
        )
        assert gpu_quarantine._read_json(gpu_quarantine._worker_latch_path(record["worker_id"])) is None

    asyncio.run(scenario())


def test_stale_generation_cannot_send_or_replace_new_latch_message(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        messages: list[str] = []
        old_record = _persist_notification_record(_physical_quarantine_record())
        gpu_quarantine._clear_durable_latches(
            old_record["worker_id"],
            device=old_record["device"],
            hostname=old_record["hostname"],
        )
        new_record = dict(old_record)
        new_record.update(
            {
                "event_id": "replacement-physical-event",
                "worker_id": "replacement_gpu_3",
                "fault_class": "initialization_failure",
                "reason": "new fault after repair",
            }
        )
        new_record = _persist_notification_record(new_record)

        async def capture_send(message, **kwargs):  # noqa: ANN001, ARG001
            messages.append(message)
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", capture_send)
        stale_outcome = await notifier.send_gpu_quarantine_page(old_record)
        current_outcome = await notifier.send_gpu_quarantine_page(new_record)

        assert stale_outcome.protocol_version == "superseded"
        assert current_outcome.success is True
        assert len(messages) == 1
        assert "replacement_gpu_3" in messages[0]
        assert "initialization_failure" in messages[0]
        assert "node21_gpu_3" not in messages[0]

    asyncio.run(scenario())


def test_explicit_unlatched_persistence_failure_pages_best_effort(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        calls = 0
        record = _physical_quarantine_record()
        record["notification_provenance"] = gpu_quarantine.UNLATCHED_NOTIFICATION_PROVENANCE

        async def capture_send(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", capture_send)
        outcome = await notifier.send_gpu_quarantine_page(record)

        assert outcome.success is True
        assert calls == 1
        assert (
            gpu_quarantine._read_json(gpu_quarantine._device_latch_path(record["hostname"], record["device"])) is None
        )

    asyncio.run(scenario())


def test_latched_claim_failure_does_not_fall_back_to_uncoordinated_send(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        calls = 0

        def fail_claim(record):  # noqa: ANN001, ARG001
            raise OSError("durable claim unavailable")

        async def unexpected_send(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "acquire_gpu_quarantine_notification_claim", fail_claim)
        monkeypatch.setattr(notifier, "send_page_user_notification", unexpected_send)
        outcome = await notifier.send_gpu_quarantine_page(_physical_quarantine_record())

        assert outcome.success is False
        assert outcome.error_kind == "claim_error"
        assert calls == 0

    asyncio.run(scenario())


def test_notification_message_is_built_from_lock_held_current_record(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        messages: list[str] = []
        current = _physical_quarantine_record()
        current["fault_class"] = "current_device_fault"
        current["worker_id"] = "current_gpu_3"
        current = _persist_notification_record(current)
        stale_snapshot = dict(current)
        stale_snapshot["fault_class"] = "stale_fault_text"
        stale_snapshot["worker_id"] = "stale_gpu_3"

        async def capture_send(message, **kwargs):  # noqa: ANN001, ARG001
            messages.append(message)
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", capture_send)
        outcome = await notifier.send_gpu_quarantine_page(stale_snapshot)

        assert outcome.success is True
        assert len(messages) == 1
        assert "current_gpu_3" in messages[0]
        assert "current_device_fault" in messages[0]
        assert "stale_gpu_3" not in messages[0]
        assert "stale_fault_text" not in messages[0]

    asyncio.run(scenario())


def test_worker_exclusion_wrapper_is_explicit_and_deduplicated(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        requests = []

        async def capture_send(message, **kwargs):  # noqa: ANN001
            requests.append((message, kwargs))
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", capture_send)
        record = _persist_notification_record(_worker_exclusion_record())
        first = await notifier.send_gpu_worker_exclusion_page(record)
        second = await notifier.send_gpu_worker_exclusion_page(record)

        assert first.success is True
        assert second.protocol_version == "deduplicated"
        assert len(requests) == 1
        message, arguments = requests[0]
        assert "GPU worker removed after restart limit" in message
        assert "physical GPU fault: not yet proven" in message
        assert arguments["title"] == "KernelGYM GPU worker excluded"

        claim = notifier.acquire_gpu_quarantine_notification_claim(record)
        try:
            assert claim.should_send is False
            assert claim.record["page_user_state"] == "sent"
            assert claim.record["page_user_attempt_count"] == "1"
        finally:
            notifier.release_gpu_quarantine_notification_claim(claim)

    asyncio.run(scenario())


def test_non_restart_worker_exclusion_message_does_not_claim_physical_fault(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        requests = []

        async def capture_send(message, **kwargs):  # noqa: ANN001
            requests.append((message, kwargs))
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        record = _worker_exclusion_record()
        record["fault_class"] = "worker_bootstrap_failure"
        record["reason"] = "warm-spare bootstrap failed"
        record = _persist_notification_record(record)
        monkeypatch.setattr(notifier, "send_page_user_notification", capture_send)

        outcome = await notifier.send_gpu_worker_exclusion_page(record)

        assert outcome.success is True
        assert len(requests) == 1
        message, arguments = requests[0]
        assert "GPU worker removed from KernelGYM scheduling" in message
        assert "fault: worker_bootstrap_failure" in message
        assert "physical GPU fault: not yet proven" in message
        assert "restart limit" not in message
        assert arguments["title"] == "KernelGYM GPU worker excluded"

    asyncio.run(scenario())


def test_worker_page_is_superseded_when_physical_latch_wins_first(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        requests = []
        worker_record = _persist_notification_record(_worker_exclusion_record())

        # Persist the weaker worker event, then let a physical event acquire the
        # device lock and replace it before the worker notifier claims delivery.
        worker_claim = notifier.acquire_gpu_quarantine_notification_claim(worker_record)
        try:
            notifier.finish_gpu_quarantine_notification_claim(worker_claim, state="failed")
        finally:
            notifier.release_gpu_quarantine_notification_claim(worker_claim)

        physical_record = _persist_notification_record(_physical_quarantine_record())
        physical_claim = notifier.acquire_gpu_quarantine_notification_claim(physical_record)
        try:
            notifier.finish_gpu_quarantine_notification_claim(physical_claim, state="failed")
        finally:
            notifier.release_gpu_quarantine_notification_claim(physical_claim)

        async def capture_send(message, **kwargs):  # noqa: ANN001
            requests.append((message, kwargs))
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", capture_send)

        stale_worker_outcome = await notifier.send_gpu_worker_exclusion_page(worker_record)
        physical_outcome = await notifier.send_gpu_quarantine_page(physical_record)

        assert stale_worker_outcome == notifier.PageUserNotificationOutcome(
            True,
            protocol_version="superseded",
        )
        assert physical_outcome.success is True
        assert len(requests) == 1
        assert requests[0][1]["title"] == "KernelGYM GPU quarantined"

    asyncio.run(scenario())


def _claim_in_child_process(record: dict[str, str], result_queue) -> None:  # noqa: ANN001
    claim = notifier.acquire_gpu_quarantine_notification_claim(record)
    try:
        result_queue.put((claim.should_send, claim.record.get("page_user_state")))
    finally:
        notifier.release_gpu_quarantine_notification_claim(claim)


def test_physical_gpu_claim_uses_cross_process_file_lock() -> None:
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    record = _persist_notification_record(_physical_quarantine_record())
    claim = notifier.acquire_gpu_quarantine_notification_claim(record)
    process = context.Process(
        target=_claim_in_child_process,
        args=(record, result_queue),
    )
    try:
        process.start()
        time.sleep(0.1)
        assert process.is_alive()

        notifier.finish_gpu_quarantine_notification_claim(claim, state="sent")
        notifier.release_gpu_quarantine_notification_claim(claim)
        should_send, page_state = result_queue.get(timeout=2)
        process.join(timeout=2)

        assert process.exitcode == 0
        assert should_send is False
        assert page_state == "sent"
    finally:
        if claim.lock_fd is not None:
            notifier.release_gpu_quarantine_notification_claim(claim)
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)
        result_queue.close()


def test_physical_gpu_page_claim_deduplicates_concurrent_processes(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        calls = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_send(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", delayed_send)
        record = _persist_notification_record(_physical_quarantine_record())
        first = asyncio.create_task(notifier.send_gpu_quarantine_page(record))
        await started.wait()
        replacement_record = dict(record)
        replacement_record["worker_id"] = "replacement_gpu_3"
        second = asyncio.create_task(notifier.send_gpu_quarantine_page(replacement_record))
        await asyncio.sleep(0.05)

        assert calls == 1
        assert not second.done()
        release.set()
        first_outcome, second_outcome = await asyncio.gather(first, second)

        assert first_outcome.success is True
        assert second_outcome == notifier.PageUserNotificationOutcome(
            True,
            protocol_version="deduplicated",
        )
        assert calls == 1

    asyncio.run(scenario())


def test_failed_physical_gpu_page_claim_is_retried_by_later_process(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        outcomes = iter(
            (
                notifier.PageUserNotificationOutcome(
                    False,
                    error_kind="transport_error",
                    error="offline",
                ),
                notifier.PageUserNotificationOutcome(True, protocol_version="mock"),
            )
        )
        calls = 0

        async def staged_send(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            return next(outcomes)

        monkeypatch.setattr(notifier, "send_page_user_notification", staged_send)
        record = _persist_notification_record(_physical_quarantine_record())
        first = await notifier.send_gpu_quarantine_page(record)
        replacement_record = dict(record)
        replacement_record["worker_id"] = "replacement_gpu_3"
        second = await notifier.send_gpu_quarantine_page(replacement_record)
        third_record = dict(record)
        third_record["worker_id"] = "third_gpu_3"
        third = await notifier.send_gpu_quarantine_page(third_record)

        assert first.success is False
        assert second.success is True
        assert third == notifier.PageUserNotificationOutcome(
            True,
            protocol_version="deduplicated",
        )
        assert calls == 2

    asyncio.run(scenario())


def test_cancelled_claim_waiter_releases_late_acquired_lock(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        calls = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def delayed_send(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            if calls > 1:
                raise AssertionError("deduplicated waiters must not send")
            started.set()
            await release.wait()
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        monkeypatch.setattr(notifier, "send_page_user_notification", delayed_send)
        record = _persist_notification_record(_physical_quarantine_record())
        first = asyncio.create_task(notifier.send_gpu_quarantine_page(record))
        await started.wait()
        replacement = dict(record)
        replacement["worker_id"] = "replacement_gpu_3"
        cancelled_waiter = asyncio.create_task(notifier.send_gpu_quarantine_page(replacement))
        await asyncio.sleep(0.05)
        cancelled_waiter.cancel()
        release.set()

        assert (await first).success is True
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        third = dict(record)
        third["worker_id"] = "third_gpu_3"
        outcome = await asyncio.wait_for(notifier.send_gpu_quarantine_page(third), timeout=1)

        assert outcome.protocol_version == "deduplicated"
        assert calls == 1

    asyncio.run(scenario())


def test_cancelled_finish_waits_for_durable_marker_before_unlock(monkeypatch) -> None:  # noqa: ANN001
    async def scenario() -> None:
        calls = 0
        finish_started = threading.Event()
        allow_finish = threading.Event()
        original_finish = notifier.finish_gpu_quarantine_notification_claim

        async def immediate_send(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal calls
            calls += 1
            return notifier.PageUserNotificationOutcome(True, protocol_version="mock")

        def delayed_finish(*args, **kwargs):  # noqa: ANN002, ANN003
            finish_started.set()
            if not allow_finish.wait(timeout=2):
                raise AssertionError("test did not release durable finish")
            return original_finish(*args, **kwargs)

        monkeypatch.setattr(notifier, "send_page_user_notification", immediate_send)
        monkeypatch.setattr(notifier, "finish_gpu_quarantine_notification_claim", delayed_finish)
        record = _persist_notification_record(_physical_quarantine_record())
        first = asyncio.create_task(notifier.send_gpu_quarantine_page(record))
        while not finish_started.is_set():
            await asyncio.sleep(0.01)
        first.cancel()
        replacement = dict(record)
        replacement["worker_id"] = "replacement_gpu_3"
        waiter = asyncio.create_task(notifier.send_gpu_quarantine_page(replacement))
        await asyncio.sleep(0.05)

        assert not waiter.done()
        allow_finish.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        outcome = await asyncio.wait_for(waiter, timeout=1)

        assert outcome.protocol_version == "deduplicated"
        assert calls == 1

    asyncio.run(scenario())
