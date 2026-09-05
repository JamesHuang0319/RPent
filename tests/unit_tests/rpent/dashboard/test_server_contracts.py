# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpent.dashboard.server import DashboardServer
from rpent.planner import check as check_mod
from rpent.planner.check import (
    STATUS_AUTH_FAILED,
    STATUS_OK,
    LlmCheckRequest,
    LlmCheckResult,
)

DASHBOARD_SPEC: dict[str, Any] = {
    "task": {"usage": "task"},
    "frame_channels": [{"name": "camera", "label": "fixed camera"}],
    "runtime_components": [],
}


def _server(**kwargs: Any) -> DashboardServer:
    return DashboardServer(dashboard_spec=DASHBOARD_SPEC, **kwargs)


def _client(server: DashboardServer) -> TestClient:
    return TestClient(server._app)


def _stub_check(
    monkeypatch: pytest.MonkeyPatch, result: LlmCheckResult
) -> dict[str, Any]:
    """Patch the shared check at its source and capture the request."""
    captured: dict[str, Any] = {}

    def fake_check(request: LlmCheckRequest) -> LlmCheckResult:
        captured["request"] = request
        return result

    monkeypatch.setattr(check_mod, "check_llm", fake_check)
    return captured


def test_check_route_returns_200_and_the_full_result_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = LlmCheckResult(
        ok=True,
        status=STATUS_OK,
        planner="api",
        model="anthropic:m",
        reply="ok",
        latency_s=0.4,
    )
    _stub_check(monkeypatch, result)

    resp = _client(_server()).post(
        "/api/llm/check", json={"planner": "api", "model": "anthropic:m"}
    )

    assert resp.status_code == 200
    assert resp.json() == result.as_dict()


def test_a_failed_check_is_still_a_successful_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_check(
        monkeypatch,
        LlmCheckResult(
            ok=False,
            status=STATUS_AUTH_FAILED,
            planner="api",
            detail="ModelHTTPError: status_code: 401",
        ),
    )

    resp = _client(_server()).post("/api/llm/check", json={"model": "anthropic:m"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is False
    assert body["status"] == STATUS_AUTH_FAILED
    assert "401" in body["detail"]


def test_server_side_base_url_is_merged_into_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _stub_check(
        monkeypatch,
        LlmCheckResult(ok=True, status=STATUS_OK, planner="api"),
    )
    server = _server(
        llm_check_defaults={"planner": "codex", "base_url": "https://gw.example"}
    )

    _client(server).post("/api/llm/check", json={"planner": "api", "model": "a:b"})

    request = captured["request"]
    # The form wins on planner/model; the server supplies what it cannot know.
    assert request.planner == "api"
    assert request.model == "a:b"
    assert request.base_url == "https://gw.example"


def test_the_server_planner_default_applies_when_the_form_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _stub_check(
        monkeypatch,
        LlmCheckResult(ok=True, status=STATUS_OK, planner="codex"),
    )
    server = _server(llm_check_defaults={"planner": "codex", "base_url": None})

    _client(server).post("/api/llm/check", json={})

    assert captured["request"].planner == "codex"


def test_the_run_timeout_is_never_inherited_by_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _stub_check(
        monkeypatch,
        LlmCheckResult(ok=True, status=STATUS_OK, planner="api"),
    )
    server = _server(llm_check_defaults={"base_url": None, "timeout_s": 1200})

    _client(server).post("/api/llm/check", json={"model": "a:b"})

    # timeout_s stays None so the check applies its own 30s/90s defaults.
    assert captured["request"].timeout_s is None
    assert captured["request"].resolved_timeout_s() == 30


def test_a_concurrent_check_is_rejected_with_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_check(request: LlmCheckRequest) -> LlmCheckResult:
        entered.set()
        release.wait(timeout=5)
        return LlmCheckResult(ok=True, status=STATUS_OK, planner="api")

    monkeypatch.setattr(check_mod, "check_llm", blocking_check)
    client = _client(_server())
    responses: list[int] = []

    def first_call() -> None:
        responses.append(
            client.post("/api/llm/check", json={"model": "a:b"}).status_code
        )

    worker = threading.Thread(target=first_call, daemon=True)
    worker.start()
    assert entered.wait(timeout=5), "the first check never started"

    second = client.post("/api/llm/check", json={"model": "a:b"})
    assert second.status_code == 409
    assert "already running" in second.json()["error"]

    release.set()
    worker.join(timeout=5)
    assert responses == [200]


def test_the_lock_is_released_after_a_failing_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exploding_check(request: LlmCheckRequest) -> LlmCheckResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(check_mod, "check_llm", exploding_check)
    server = _server()
    client = _client(server)

    with pytest.raises(RuntimeError):
        client.post("/api/llm/check", json={"model": "a:b"})

    # A crashed check must not wedge the endpoint for the rest of the session.
    assert server._llm_check_lock.acquire(blocking=False)
    server._llm_check_lock.release()


def test_healthz_still_reports_transport_liveness_only() -> None:
    resp = _client(_server()).get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
