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
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from rpent.planner import check as check_mod
from rpent.planner.check import (
    CHECK_STATUSES,
    DEFAULT_TIMEOUT_S,
    NO_CREDENTIAL_TIMEOUT_S,
    PROBE_PROMPT,
    STATUS_AUTH_FAILED,
    STATUS_IMAGE_REJECTED,
    STATUS_INVALID_MODEL,
    STATUS_MISSING_API_KEY,
    STATUS_MISSING_CONFIG,
    STATUS_NETWORK_ERROR,
    STATUS_OK,
    STATUS_PROVIDER_ERROR,
    STATUS_SDK_ERROR,
    STATUS_TOOL_CALLS_UNSUPPORTED,
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


@pytest.fixture(autouse=True)
def _no_cli_login(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real CLI login out of the no-credential path.

    ``_has_cli_login`` reads files under ``~``, so without this the SDK
    tests would pass or fail depending on whose machine they run on.
    """
    monkeypatch.setattr(check_mod, "_has_cli_login", lambda planner: False)


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
        STATUS_INVALID_MODEL,
        STATUS_IMAGE_REJECTED,
        STATUS_TOOL_CALLS_UNSUPPORTED,
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
        (400, STATUS_INVALID_MODEL),
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
    # Patched at the SDK boundary, so the probe's own parsing stays under test.
    import claude_agent_sdk
    from claude_agent_sdk import TextBlock

    seen: dict[str, Any] = {}

    async def fake_query(*, prompt: str, options: Any):
        seen["model"] = options.model
        yield _assistant([TextBlock(text="ok")])
        yield _result_message(result="ok")

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)
    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert seen["model"] == "sonnet"
    assert result.ok is True
    assert result.reply == "ok"
    assert result.credential_env == "ANTHROPIC_API_KEY"
    assert result.credential_present is False


def test_codex_probe_runs_without_an_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_config(base_url: str | None = None) -> Any:
        seen["base_url"] = base_url
        return object()

    def fake_probe(
        config: Any, *, prompt: str, model: str | None, timeout_s: int
    ) -> str:
        seen["prompt"] = prompt
        seen["model"] = model
        seen["timeout_s"] = timeout_s
        return "ok"

    # A credential is present, so the probe gets the full default budget.
    monkeypatch.setenv("CODEX_API_KEY", SENTINEL_KEY)
    monkeypatch.setattr("rpent.planner.codex.build_probe_config", fake_config)
    monkeypatch.setattr("rpent.planner.codex.run_probe_turn", fake_probe)
    result = check_llm(LlmCheckRequest(planner="codex", model="gpt-5.5"))

    assert result.ok is True
    assert seen["model"] == "gpt-5.5"
    assert seen["timeout_s"] == DEFAULT_TIMEOUT_S["codex"]
    assert seen["prompt"] == PROBE_PROMPT
    assert result.credential_env == "CODEX_API_KEY"


def test_codex_probe_failure_is_classified_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_probe(config: Any, **kwargs: Any) -> str:
        raise RuntimeError("HTTP 401 Unauthorized")

    monkeypatch.setattr(
        "rpent.planner.codex.build_probe_config", lambda base_url=None: object()
    )
    monkeypatch.setattr("rpent.planner.codex.run_probe_turn", failing_probe)
    result = check_llm(LlmCheckRequest(planner="codex"))

    assert result.ok is False
    assert result.status == STATUS_AUTH_FAILED


def test_codex_probe_timeout_reports_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timing_out_probe(config: Any, **kwargs: Any) -> str:
        raise TimeoutError("the Codex SDK did not finish within 90s.")

    monkeypatch.setenv("CODEX_API_KEY", SENTINEL_KEY)
    monkeypatch.setattr(
        "rpent.planner.codex.build_probe_config", lambda base_url=None: object()
    )
    monkeypatch.setattr("rpent.planner.codex.run_probe_turn", timing_out_probe)
    result = check_llm(LlmCheckRequest(planner="codex"))

    assert result.status == STATUS_NETWORK_ERROR


# ---------------------------------------------------------------------------
# The deep probe
#
# A planner run always carries images and tool schemas. A text-only probe can
# therefore pass against an endpoint the real run cannot use -- which is why
# --no-images exists at all.
# ---------------------------------------------------------------------------


def _tool_calling_model(call_tool: bool) -> FunctionModel:
    """Return a model that either calls the offered tool or just talks."""

    def respond(messages: Any, info: Any) -> ModelResponse:
        already_called = any(
            type(part).__name__ == "ToolCallPart"
            for message in messages
            for part in getattr(message, "parts", ())
        )
        if call_tool and not already_called:
            return ModelResponse(parts=[ToolCallPart("probe_ok", {"note": "ok"})])
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(respond)


def test_the_deep_probe_sends_an_image_and_a_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    seen: dict[str, Any] = {}

    def respond(messages: Any, info: Any) -> ModelResponse:
        seen.setdefault(
            "content_types",
            [
                type(item).__name__
                for message in messages
                for part in getattr(message, "parts", ())
                if isinstance(getattr(part, "content", None), list)
                for item in part.content
            ],
        )
        seen["tools"] = [t.name for t in (info.function_tools or [])]
        return ModelResponse(parts=[ToolCallPart("probe_ok", {"note": "ok"})])

    _patch_model(monkeypatch, FunctionModel(respond))
    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m", deep=True))

    assert result.status == STATUS_OK
    assert "BinaryContent" in seen["content_types"]
    assert "probe_ok" in seen["tools"]
    assert result.as_dict()["deep"] is True


def test_a_model_that_never_calls_the_tool_is_not_a_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(monkeypatch, _tool_calling_model(call_tool=False))

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m", deep=True))

    assert result.status == STATUS_TOOL_CALLS_UNSUPPORTED


def test_an_endpoint_refusing_images_is_named_as_such(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text-only OpenAI-compatible endpoints answer an image block with a 400.

    Reporting that as ``invalid_model`` would send people to change the model
    id when the fix is --no-images or a multimodal model.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(
        monkeypatch,
        _raising_model(
            ModelHTTPError(
                status_code=400,
                model_name="m",
                body="message type 'image_url' is not supported",
            )
        ),
    )

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m", deep=True))

    assert result.status == STATUS_IMAGE_REJECTED


def test_a_shallow_probe_never_reports_image_or_tool_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --deep the probe sends no image, so it cannot blame one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)
    _patch_model(monkeypatch, _tool_calling_model(call_tool=False))

    result = check_llm(LlmCheckRequest(planner="api", model="anthropic:m"))

    assert result.status == STATUS_OK
    assert "deep" not in result.as_dict()


@pytest.mark.parametrize("planner", ["claude_code", "codex"])
def test_the_sdk_backends_say_deep_does_not_apply(
    monkeypatch: pytest.MonkeyPatch, planner: str
) -> None:
    """Claiming coverage the probe cannot provide would be worse than none."""

    async def claude_boom(sdk: Any, model: str) -> Any:
        raise RuntimeError("stubbed")

    def codex_boom(config: Any, **kwargs: Any) -> str:
        raise RuntimeError("stubbed")

    monkeypatch.setattr(check_mod, "_run_claude_probe", claude_boom)
    monkeypatch.setattr(
        "rpent.planner.codex.build_probe_config", lambda base_url=None: object()
    )
    monkeypatch.setattr("rpent.planner.codex.run_probe_turn", codex_boom)

    result = check_llm(LlmCheckRequest(planner=planner, deep=True))

    assert result.as_dict()["deep"] == "not applicable to this backend"


# ---------------------------------------------------------------------------
# SDK backends: the no-credential budget
#
# Both accept an interactive CLI login instead of an env var, so a missing
# variable can never skip the probe outright. It only shortens the budget and
# renames the timeout, so a user with no key waits seconds, not a minute and
# a half, and is told which of the two it was.
# ---------------------------------------------------------------------------


def test_a_login_file_is_detected_as_a_credential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.undo()  # drop the autouse _has_cli_login stub
    login = tmp_path / "auth.json"
    login.write_text("{}")
    monkeypatch.setattr(check_mod, "_CLI_LOGIN_FILES", {"codex": (str(login),)})
    monkeypatch.setattr(check_mod, "_KEYCHAIN_SERVICES", {})

    assert check_mod._has_cli_login("codex") is True
    assert check_mod._has_cli_login("claude_code") is False


def test_the_macos_keychain_counts_as_a_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS stores the Claude credential outside the filesystem.

    Without this lookup a logged-in Mac user with no env var would be put on
    the no-credential path and a slow reply misreported as a missing key.
    """
    monkeypatch.undo()  # drop the autouse _has_cli_login stub
    monkeypatch.setattr(check_mod, "_CLI_LOGIN_FILES", {})
    monkeypatch.setattr(check_mod.sys, "platform", "darwin")
    seen: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(check_mod.subprocess, "run", fake_run)

    assert check_mod._has_cli_login("claude_code") is True
    # Existence only: -w would read the secret and prompt for the Keychain.
    assert "-w" not in seen["cmd"]


def test_the_keychain_is_not_consulted_off_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # drop the autouse _has_cli_login stub
    monkeypatch.setattr(check_mod, "_CLI_LOGIN_FILES", {})
    monkeypatch.setattr(check_mod.sys, "platform", "linux")

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the keychain must not be consulted off macOS")

    monkeypatch.setattr(check_mod.subprocess, "run", explode)

    assert check_mod._has_cli_login("claude_code") is False


def test_a_broken_keychain_lookup_is_not_a_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.undo()  # drop the autouse _has_cli_login stub
    monkeypatch.setattr(check_mod, "_CLI_LOGIN_FILES", {})
    monkeypatch.setattr(check_mod.sys, "platform", "darwin")

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise OSError("security binary is missing")

    monkeypatch.setattr(check_mod.subprocess, "run", explode)

    assert check_mod._has_cli_login("claude_code") is False


@pytest.mark.parametrize("planner", ["claude_code", "codex"])
def test_no_credential_shortens_the_probe_budget(planner: str) -> None:
    request = LlmCheckRequest(planner=planner)

    budget = check_mod._sdk_probe_budget(request, no_credential=True)

    assert budget == NO_CREDENTIAL_TIMEOUT_S
    assert NO_CREDENTIAL_TIMEOUT_S < DEFAULT_TIMEOUT_S[planner]


def test_the_no_credential_budget_clears_a_measured_cold_probe() -> None:
    """A slow success must not be mistaken for a missing credential.

    A cold Codex probe against a real gateway replied in 15.3s. A budget at
    or below that turns every such run into a false ``missing_api_key``
    whenever login detection misses, which it can: it reads known files and
    the macOS Keychain, and a backend may store its credential elsewhere.
    """
    assert NO_CREDENTIAL_TIMEOUT_S > 15.3


@pytest.mark.parametrize("planner", ["claude_code", "codex"])
def test_a_credential_keeps_the_full_probe_budget(planner: str) -> None:
    request = LlmCheckRequest(planner=planner)

    budget = check_mod._sdk_probe_budget(request, no_credential=False)

    assert budget == DEFAULT_TIMEOUT_S[planner]


def test_an_explicit_timeout_survives_the_no_credential_shortcut() -> None:
    request = LlmCheckRequest(planner="codex", timeout_s=45)

    assert check_mod._sdk_probe_budget(request, no_credential=True) == 45


def test_a_cli_login_counts_as_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A logged-in user with no env var must not be shortened or misreported."""
    monkeypatch.setattr(check_mod, "_has_cli_login", lambda planner: True)

    async def never_finishes(sdk: Any, model: str) -> str:
        await asyncio.sleep(30)
        return "unreachable"

    monkeypatch.setattr(check_mod, "_run_claude_probe", never_finishes)
    result = check_llm(LlmCheckRequest(planner="claude_code", timeout_s=1))

    assert result.status == STATUS_NETWORK_ERROR


def test_claude_code_without_any_credential_reports_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def never_finishes(sdk: Any, model: str) -> str:
        await asyncio.sleep(30)
        return "unreachable"

    monkeypatch.setattr(check_mod, "_run_claude_probe", never_finishes)
    result = check_llm(LlmCheckRequest(planner="claude_code", timeout_s=1))

    assert result.status == STATUS_MISSING_API_KEY
    assert "ANTHROPIC_API_KEY" in result.detail
    assert "CLI login" in result.detail


def test_codex_without_any_credential_reports_missing_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timing_out_probe(config: Any, **kwargs: Any) -> str:
        raise TimeoutError("the Codex SDK did not finish in time.")

    monkeypatch.setattr(
        "rpent.planner.codex.build_probe_config", lambda base_url=None: object()
    )
    monkeypatch.setattr("rpent.planner.codex.run_probe_turn", timing_out_probe)
    result = check_llm(LlmCheckRequest(planner="codex", timeout_s=1))

    assert result.status == STATUS_MISSING_API_KEY
    assert "CODEX_API_KEY" in result.detail


@pytest.mark.parametrize(
    ("message", "key_present", "expected"),
    [
        ("HTTP 401 Unauthorized", False, STATUS_AUTH_FAILED),
        ("authentication failed", True, STATUS_AUTH_FAILED),
        ("not logged in", False, STATUS_MISSING_API_KEY),
        ("please run login", True, STATUS_AUTH_FAILED),
        ("failed to resolve host", False, STATUS_NETWORK_ERROR),
        ("429 rate limit exceeded", False, STATUS_PROVIDER_ERROR),
        ("400 model does not exist", True, STATUS_INVALID_MODEL),
        ("unknown model 'gpt-9'", True, STATUS_INVALID_MODEL),
        ("HTTP 400 Bad Request", True, STATUS_INVALID_MODEL),
        # Verbatim from a Codex probe against a real gateway: the status is
        # 404, not 400, and the wording matches none of the phrases above.
        (
            'unexpected status 404 Not Found: Model "definitely-not-a-real-model" '
            "is not available for this group, url: https://gateway.example/responses, "
            "cf-ray: a37dbb0389e965c9-FRA",
            True,
            STATUS_INVALID_MODEL,
        ),
        # Verbatim from the Claude Code SDK.
        (
            '[claude-code:unrecognized_model] {"model":"nope"}',
            True,
            STATUS_INVALID_MODEL,
        ),
        # "400" inside a larger number must not trip the bare-status fallback.
        ("served 2400 tokens then crashed", False, STATUS_SDK_ERROR),
        # A bare 404 with no model wording is a bad endpoint, not a bad model.
        (
            "unexpected status 404 Not Found: no route for /v1/chat",
            False,
            STATUS_SDK_ERROR,
        ),
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
    # With a credential in hand a timeout really is a transport failure.
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL_KEY)

    async def never_finishes(sdk: Any, model: str) -> str:
        await asyncio.sleep(30)
        return "unreachable"

    monkeypatch.setattr(check_mod, "_run_claude_probe", never_finishes)
    result = check_llm(LlmCheckRequest(planner="claude_code", timeout_s=1))

    assert result.status == STATUS_NETWORK_ERROR


# ---------------------------------------------------------------------------
# claude_code probe: SDK message-parsing contract
#
# The probe's success criterion is "a non-empty assistant reply", so it must
# parse the SDK's message stream. These tests pin that parsing against the
# real SDK types instead of a hand-shaped double, the way
# test_codex_contracts.py pins TurnHandle.
# ---------------------------------------------------------------------------


def test_claude_sdk_message_types_carry_the_fields_the_probe_reads() -> None:
    """Pin the SDK surface the probe depends on.

    ``UserMessage.content`` is asserted deliberately: the stream echoes the
    prompt back, so a parser that reads ``content`` off every message would
    mistake our own prompt for the model's reply.
    """
    import claude_agent_sdk as sdk

    assert "content" in sdk.AssistantMessage.__annotations__
    assert "text" in sdk.TextBlock.__annotations__
    # The structured error channels the probe must classify from.
    assert "error" in sdk.AssistantMessage.__annotations__
    assert "api_error_status" in sdk.ResultMessage.__annotations__
    assert "is_error" in sdk.ResultMessage.__annotations__
    # The trap: a user message carries content too.
    assert "content" in sdk.UserMessage.__annotations__
    # Thinking text lives under a different field and must not be collected.
    assert "text" not in sdk.ThinkingBlock.__annotations__
    assert "thinking" in sdk.ThinkingBlock.__annotations__


def _assistant(content: Any, **kwargs: Any) -> Any:
    from claude_agent_sdk import AssistantMessage

    return AssistantMessage(content=content, model="claude-test", **kwargs)


def _result_message(**kwargs: Any) -> Any:
    from claude_agent_sdk import ResultMessage

    defaults: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 10,
        "duration_api_ms": 10,
        "is_error": False,
        "num_turns": 1,
        "session_id": "s1",
    }
    return ResultMessage(**{**defaults, **kwargs})


def test_probe_ignores_the_echoed_user_prompt() -> None:
    """The stream echoes our prompt; it is not the model's reply."""
    from claude_agent_sdk import UserMessage

    assert check_mod._assistant_text(UserMessage(content=PROBE_PROMPT)) == []


def test_probe_collects_only_assistant_text_blocks() -> None:
    from claude_agent_sdk import TextBlock, ThinkingBlock

    message = _assistant(
        [ThinkingBlock(thinking="pondering", signature="sig"), TextBlock(text="ok")]
    )

    assert check_mod._assistant_text(message) == ["ok"]


def test_probe_ignores_system_and_result_messages() -> None:
    from claude_agent_sdk import SystemMessage

    assert check_mod._assistant_text(SystemMessage(subtype="init", data={})) == []
    assert check_mod._assistant_text(_result_message(result="ok")) == []


def _patch_claude_query(monkeypatch: pytest.MonkeyPatch, messages: list[Any]) -> None:
    """Replace sdk.query at the SDK boundary, keeping our parsing under test."""
    import claude_agent_sdk

    async def fake_query(*, prompt: str, options: Any):
        for message in messages:
            yield message

    monkeypatch.setattr(claude_agent_sdk, "query", fake_query)


def test_claude_code_success_reads_the_assistant_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from claude_agent_sdk import TextBlock, UserMessage

    _patch_claude_query(
        monkeypatch,
        [
            UserMessage(content=PROBE_PROMPT),
            _assistant([TextBlock(text="ok")]),
            _result_message(result="ok"),
        ],
    )

    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert result.ok is True
    assert result.reply == "ok"


def test_claude_code_echo_only_stream_is_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model said nothing; only our own prompt came back."""
    from claude_agent_sdk import UserMessage

    _patch_claude_query(
        monkeypatch,
        [
            UserMessage(content=PROBE_PROMPT),
            _assistant([]),
            _result_message(is_error=True, subtype="error"),
        ],
    )

    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert result.ok is False, "an echoed prompt must not count as a reply"


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        ("authentication_failed", STATUS_AUTH_FAILED),
        ("billing_error", STATUS_PROVIDER_ERROR),
        ("rate_limit", STATUS_PROVIDER_ERROR),
        ("invalid_request", STATUS_PROVIDER_ERROR),
        ("server_error", STATUS_PROVIDER_ERROR),
    ],
)
def test_claude_code_structured_error_is_classified(
    monkeypatch: pytest.MonkeyPatch, sdk_error: str, expected: str
) -> None:
    """The SDK reports typed errors; the probe must not flatten them."""
    _patch_claude_query(
        monkeypatch,
        [
            _assistant([], error=sdk_error),
            _result_message(is_error=True, subtype="error"),
        ],
    )

    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert result.ok is False
    assert result.status == expected
    assert sdk_error in result.detail


def test_claude_code_api_error_status_401_is_auth_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_claude_query(
        monkeypatch,
        [
            _assistant([]),
            _result_message(is_error=True, subtype="error", api_error_status=401),
        ],
    )

    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert result.status == STATUS_AUTH_FAILED


def test_claude_code_api_error_status_400_is_invalid_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_claude_query(
        monkeypatch,
        [
            _assistant([]),
            _result_message(is_error=True, subtype="error", api_error_status=400),
        ],
    )

    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert result.status == STATUS_INVALID_MODEL


def test_claude_code_api_error_status_429_is_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_claude_query(
        monkeypatch,
        [
            _assistant([]),
            _result_message(is_error=True, subtype="error", api_error_status=429),
        ],
    )

    result = check_llm(LlmCheckRequest(planner="claude_code"))

    assert result.status == STATUS_PROVIDER_ERROR
