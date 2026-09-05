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

import asyncio
from typing import Any

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from rpent.planner import check as check_mod
from rpent.planner.check import (
    CHECK_STATUSES,
    DEFAULT_TIMEOUT_S,
    STATUS_AUTH_FAILED,
    STATUS_MISSING_API_KEY,
    STATUS_MISSING_CONFIG,
    STATUS_NETWORK_ERROR,
    STATUS_OK,
    STATUS_PROVIDER_ERROR,
    STATUS_SDK_ERROR,
    STATUS_UNSUPPORTED_PROVIDER,
    LlmCheckRequest,
    check_llm,
)

SENTINEL_KEY = "sk-sentinel-must-never-be-echoed"


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real credentials out of every test."""
    for name in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL",
        "CODEX_BASE_URL",
        "CODEX_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _reply_model(text: str = "ok") -> FunctionModel:
    """Return a model that answers the probe with ``text``."""

    def respond(messages: Any, info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(respond)


def _raising_model(exc: Exception) -> FunctionModel:
    """Return a model whose request raises ``exc``."""

    def respond(messages: Any, info: Any) -> ModelResponse:
        raise exc

    return FunctionModel(respond)


def _patch_model(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    """Make build_api_model return ``model`` without touching a provider."""
    monkeypatch.setattr(
        "rpent.planner.base.build_api_model",
        lambda model_id, base_url=None: model,
    )


# ---------------------------------------------------------------------------
# Contract surface
# ---------------------------------------------------------------------------


def test_every_status_is_declared_in_the_public_tuple() -> None:
    assert set(CHECK_STATUSES) == {
        STATUS_OK,
        STATUS_MISSING_CONFIG,
        STATUS_UNSUPPORTED_PROVIDER,
        STATUS_MISSING_API_KEY,
        STATUS_AUTH_FAILED,
        STATUS_NETWORK_ERROR,
        STATUS_PROVIDER_ERROR,
        STATUS_SDK_ERROR,
    }


def test_diagnostic_timeouts_never_inherit_the_run_default() -> None:
    assert DEFAULT_TIMEOUT_S == {"api": 30, "claude_code": 90, "codex": 90}
    for planner, expected in DEFAULT_TIMEOUT_S.items():
        assert LlmCheckRequest(planner=planner).resolved_timeout_s() == expected
    assert LlmCheckRequest(planner="api", timeout_s=5).resolved_timeout_s() == 5


def test_unknown_planner_is_rejected_without_touching_a_backend() -> None:
    result = check_llm(LlmCheckRequest(planner="telepathy", model="x:y"))

    assert result.ok is False
    assert result.status == STATUS_MISSING_CONFIG
    assert "telepathy" in result.detail


def test_check_refuses_to_run_inside_an_active_event_loop() -> None:
    async def call_from_loop() -> Any:
        return check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    result = asyncio.run(call_from_loop())

    assert result.ok is False
    assert result.status == STATUS_SDK_ERROR
    assert "event loop" in result.detail


def test_check_never_raises_even_when_the_backend_explodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(request: Any) -> Any:
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(check_mod, "_check_api", boom)
    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.ok is False
    assert result.status == STATUS_SDK_ERROR
    assert "backend exploded" in result.detail


# ---------------------------------------------------------------------------
# api planner: configuration failures, detected before any request
# ---------------------------------------------------------------------------


def test_api_without_a_model_reports_missing_config() -> None:
    result = check_llm(LlmCheckRequest(planner="api"))

    assert result.status == STATUS_MISSING_CONFIG
    assert "provider prefix" in result.detail


def test_api_model_without_a_provider_prefix_reports_missing_config() -> None:
    result = check_llm(LlmCheckRequest(planner="api", model="gpt-5.5"))

    assert result.status == STATUS_MISSING_CONFIG
    assert "gpt-5.5" in result.detail


def test_api_unknown_provider_prefix_reports_unsupported_provider() -> None:
    result = check_llm(LlmCheckRequest(planner="api", model="nosuchprovider:m"))

    assert result.status == STATUS_UNSUPPORTED_PROVIDER
    assert result.ok is False


@pytest.mark.parametrize(
    ("model", "env_var"),
    [
        ("anthropic:claude-opus-4-8", "ANTHROPIC_API_KEY"),
        ("openai:gpt-5.5", "OPENAI_API_KEY"),
        ("openai-chat:glm-5.2", "OPENAI_API_KEY"),
    ],
)
def test_api_missing_key_names_the_variable_for_each_documented_provider(
    model: str, env_var: str
) -> None:
    result = check_llm(LlmCheckRequest(planner="api", model=model))

    assert result.status == STATUS_MISSING_API_KEY
    assert result.credential_env == env_var
    assert result.credential_present is False
    assert env_var in result.detail


def test_api_empty_key_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.status == STATUS_MISSING_API_KEY


# ---------------------------------------------------------------------------
# api planner: request outcomes
# ---------------------------------------------------------------------------


def test_api_success_reports_reply_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(monkeypatch, _reply_model("ok"))

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.ok is True
    assert result.status == STATUS_OK
    assert result.reply == "ok"
    assert result.latency_s is not None and result.latency_s >= 0
    assert result.credential_present is True


def test_api_empty_reply_is_a_provider_error_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(monkeypatch, _reply_model("   "))

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.ok is False
    assert result.status == STATUS_PROVIDER_ERROR


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, STATUS_AUTH_FAILED),
        (403, STATUS_AUTH_FAILED),
        (404, STATUS_PROVIDER_ERROR),
        (429, STATUS_PROVIDER_ERROR),
        (500, STATUS_PROVIDER_ERROR),
        (503, STATUS_PROVIDER_ERROR),
    ],
)
def test_api_http_errors_split_auth_from_provider_failures(
    monkeypatch: pytest.MonkeyPatch, status_code: int, expected: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(
        monkeypatch,
        _raising_model(
            ModelHTTPError(status_code=status_code, model_name="m", body="denied")
        ),
    )

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.status == expected
    assert "denied" in result.detail


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.ProxyError("proxy refused"),
    ],
)
def test_api_transport_failures_report_network_error(
    monkeypatch: pytest.MonkeyPatch, exc: Exception
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(monkeypatch, _raising_model(exc))

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.status == STATUS_NETWORK_ERROR


def test_api_timeout_reports_network_error_and_names_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)

    async def never_finishes(*args: Any, **kwargs: Any) -> str:
        await asyncio.sleep(30)
        return "unreachable"

    _patch_model(monkeypatch, _reply_model("ok"))
    monkeypatch.setattr(check_mod, "_run_api_probe", never_finishes)

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m", timeout_s=1))

    assert result.status == STATUS_NETWORK_ERROR
    assert "1s" in result.detail


def test_api_user_error_during_resolution_is_classified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)

    def raise_user_error(model_id: str, base_url: str | None = None) -> Any:
        raise UserError("Unknown model: anthropic:m")

    monkeypatch.setattr("rpent.planner.base.build_api_model", raise_user_error)
    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.status == STATUS_UNSUPPORTED_PROVIDER


def test_api_base_url_override_is_forwarded_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    seen: dict[str, Any] = {}

    def capture(model_id: str, base_url: str | None = None) -> Any:
        seen["model"] = model_id
        seen["base_url"] = base_url
        return _reply_model("ok")

    monkeypatch.setattr("rpent.planner.base.build_api_model", capture)
    result = check_llm(
        LlmCheckRequest(
            planner="api", model="anthropic:m", base_url="https://gateway.example"
        )
    )

    assert seen == {"model": "anthropic:m", "base_url": "https://gateway.example"}
    assert result.base_url == "https://gateway.example"
    assert result.ok is True


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    ["success", "auth_failed", "network_error"],
)
def test_the_api_key_value_never_reaches_the_result(
    monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    models = {
        "success": _reply_model("ok"),
        "auth_failed": _raising_model(
            ModelHTTPError(status_code=401, model_name="m", body="bad key")
        ),
        "network_error": _raising_model(httpx.ConnectError("refused")),
    }
    _patch_model(monkeypatch, models[scenario])

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert SENTINEL_KEY not in repr(result)
    assert SENTINEL_KEY not in str(result.as_dict())
    assert result.credential_env == "ANTHROPIC_API_KEY"


def test_a_provider_error_body_echoing_the_key_is_not_carried_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(
        monkeypatch,
        _raising_model(
            ModelHTTPError(
                status_code=401,
                model_name="m",
                body=f"invalid key: {SENTINEL_KEY}",
            )
        ),
    )

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.status == STATUS_AUTH_FAILED
    assert SENTINEL_KEY not in result.detail
    assert SENTINEL_KEY not in str(result.as_dict())


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_as_dict_exposes_the_full_documented_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(monkeypatch, _reply_model("ok"))

    payload = check_llm(LlmCheckRequest(planner="api", model="anthropic:m")).as_dict()

    assert set(payload) == {
        "ok",
        "status",
        "planner",
        "model",
        "credential_env",
        "credential_present",
        "base_url",
        "base_url_env",
        "detail",
        "reply",
        "latency_s",
    }


# ---------------------------------------------------------------------------
# SDK backends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("planner", ["claude_code", "codex"])
def test_sdk_backends_report_sdk_error_when_the_package_is_absent(
    monkeypatch: pytest.MonkeyPatch, planner: str
) -> None:
    import builtins

    blocked = {"claude_agent_sdk", "rpent.planner.codex"}
    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in blocked:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = check_llm(LlmCheckRequest(planner=planner, model="m"))

    assert result.ok is False
    assert result.status == STATUS_SDK_ERROR


def test_claude_code_defaults_to_sonnet_and_reports_its_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def fake_probe(sdk: Any, model: str) -> str:
        seen["model"] = model
        return "ok"

    monkeypatch.setattr(check_mod, "_run_claude_probe", fake_probe)
    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert seen["model"] == "sonnet"
    assert result.ok is True
    assert result.credential_env == "ANTHROPIC_API_KEY"
    assert result.credential_present is False


def test_codex_probe_runs_without_an_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_config(base_url: str | None = None) -> Any:
        seen["base_url"] = base_url
        return object()

    def fake_probe(config: Any, model: str | None, timeout_s: int) -> str:
        seen["model"] = model
        seen["timeout_s"] = timeout_s
        return "ok"

    monkeypatch.setattr("rpent.planner.codex.build_probe_config", fake_config)
    monkeypatch.setattr(check_mod, "_run_codex_probe", fake_probe)
    result = check_llm(LlmCheckRequest(planner="codex", model="gpt-5.5"))

    assert result.ok is True
    assert seen["model"] == "gpt-5.5"
    assert seen["timeout_s"] == 90
    assert result.credential_env == "CODEX_API_KEY"


@pytest.mark.parametrize(
    ("message", "key_present", "expected"),
    [
        ("HTTP 401 Unauthorized", False, STATUS_AUTH_FAILED),
        ("authentication failed", True, STATUS_AUTH_FAILED),
        ("not logged in", False, STATUS_MISSING_API_KEY),
        ("please run login", True, STATUS_AUTH_FAILED),
        ("failed to resolve host", False, STATUS_NETWORK_ERROR),
        ("429 rate limit exceeded", False, STATUS_PROVIDER_ERROR),
        ("something entirely unexpected", False, STATUS_SDK_ERROR),
    ],
)
def test_sdk_error_messages_are_classified(
    message: str, key_present: bool, expected: str
) -> None:
    status = check_mod._classify_sdk_error(
        RuntimeError(message), key_present=key_present
    )

    assert status == expected


def test_sdk_backend_timeout_reports_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def never_finishes(sdk: Any, model: str) -> str:
        await asyncio.sleep(30)
        return "unreachable"

    monkeypatch.setattr(check_mod, "_run_claude_probe", never_finishes)
    result = check_llm(LlmCheckRequest(planner="claude_code", timeout_s=1))

    assert result.status == STATUS_NETWORK_ERROR
