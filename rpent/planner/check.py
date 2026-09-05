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

"""Connectivity check for the configured planner backend.

This is the single implementation behind both the ``rpent-check-llm`` console
script and the Dashboard's ``POST /api/llm/check`` route. It sends the smallest
real request each backend supports — no tools, no images, no ``Toolkit``, no
robot runtime, no output directory — and classifies the outcome.

The layer is deliberately UI-agnostic: it reports a machine-readable ``status``
plus structured credential metadata and the provider's verbatim ``detail``.
Human-readable remediation text belongs to the CLI and the Dashboard, not here.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

from rpent.utils.logging import get_logger

logger = get_logger("check")

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

#: The backend replied. Everything else is a failure.
STATUS_OK = "ok"
#: No model id, or an ``api`` model id without a ``provider:`` prefix.
STATUS_MISSING_CONFIG = "missing_config"
#: The provider prefix is not one pydantic-ai can resolve.
STATUS_UNSUPPORTED_PROVIDER = "unsupported_provider"
#: The backend's credential env var is unset or empty.
STATUS_MISSING_API_KEY = "missing_api_key"
#: The provider rejected the credential (HTTP 401 / 403).
STATUS_AUTH_FAILED = "auth_failed"
#: DNS, connection, TLS failure, or a timeout.
STATUS_NETWORK_ERROR = "network_error"
#: The provider was reached and refused (404, 429, 5xx, ...).
STATUS_PROVIDER_ERROR = "provider_error"
#: SDK missing, child process failure, or anything unclassified.
STATUS_SDK_ERROR = "sdk_error"

#: Every status this module can return, in rough severity order.
CHECK_STATUSES = (
    STATUS_OK,
    STATUS_MISSING_CONFIG,
    STATUS_UNSUPPORTED_PROVIDER,
    STATUS_MISSING_API_KEY,
    STATUS_AUTH_FAILED,
    STATUS_NETWORK_ERROR,
    STATUS_PROVIDER_ERROR,
    STATUS_SDK_ERROR,
)

#: Planner backends this module can probe.
CHECK_PLANNERS = ("api", "claude_code", "codex")

#: Diagnostic timeouts, deliberately independent of the 1200s run default.
DEFAULT_TIMEOUT_S = {"api": 30, "claude_code": 90, "codex": 90}

#: The smallest prompt that still proves the model answered.
PROBE_PROMPT = "Reply with the single word: ok"

#: Output cap for the probe. Large enough for a word, small enough to be free.
PROBE_MAX_TOKENS = 16

#: Advisory only: names the env var in an error before pydantic-ai is asked to
#: resolve it. An unmapped prefix skips the pre-flight and falls through to
#: ``UserError`` classification, so this table can never reject a provider it
#: does not know about. Mirrors docs/source-en/.../usage/configure_planner.rst.
_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-chat": "OPENAI_API_KEY",
}
_API_BASE_URL_ENV = {
    "anthropic": "ANTHROPIC_BASE_URL",
    "openai": "OPENAI_BASE_URL",
    "openai-chat": "OPENAI_BASE_URL",
}

#: Credential env vars for the two child-process SDK backends. Both accept an
#: existing CLI login instead, so a missing var is reported but not fatal.
_CLAUDE_CODE_KEY_ENV = "ANTHROPIC_API_KEY"
_CODEX_KEY_ENV = "CODEX_API_KEY"
_CODEX_BASE_URL_ENV = "CODEX_BASE_URL"

#: Cap on how much provider error text is carried in ``detail``.
_DETAIL_LIMIT = 2000

#: Credential env vars whose *values* are redacted out of any reported text.
#: Provider error bodies and SDK stderr can echo the key that was sent, and a
#: check report is exactly the kind of output that gets pasted into an issue.
_SECRET_ENV_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "RPENT_CODEX_PROVIDER_KEY",
)

#: Values shorter than this are too generic to redact without mangling text.
_MIN_SECRET_LEN = 8


@dataclass(frozen=True, slots=True)
class LlmCheckRequest:
    """One connectivity check to perform.

    Attributes:
        planner: Backend to probe; one of :data:`CHECK_PLANNERS`.
        model: Model id. Required (and provider-prefixed) for ``api``;
            optional for ``claude_code`` and ``codex``.
        base_url: Base URL overriding the backend's own env var.
        timeout_s: Wall-clock cap. Defaults per backend when ``None``.
    """

    planner: str = "api"
    model: str | None = None
    base_url: str | None = None
    timeout_s: int | None = None

    def resolved_timeout_s(self) -> int:
        """Return the effective timeout, applying the per-backend default."""
        if self.timeout_s is not None:
            return int(self.timeout_s)
        return DEFAULT_TIMEOUT_S.get(self.planner, 30)


@dataclass(frozen=True, slots=True)
class LlmCheckResult:
    """The structured outcome of one check.

    Attributes:
        ok: True only when the backend replied with non-empty text.
        status: One of :data:`CHECK_STATUSES`.
        planner: The backend that was probed.
        model: The model id that was used, when known.
        credential_env: Name of the credential env var consulted, when known.
            The value is never captured.
        credential_present: Whether that env var was set and non-empty.
        base_url: The base URL override in effect, when given.
        base_url_env: Name of the env var that supplies the base URL otherwise.
        detail: Verbatim error text from the provider or SDK. Empty on success.
        reply: The model's reply text on success.
        latency_s: Seconds spent on the request itself, when one was made.
    """

    ok: bool
    status: str
    planner: str
    model: str | None = None
    credential_env: str | None = None
    credential_present: bool | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    detail: str = ""
    reply: str = ""
    latency_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for ``--json`` and the HTTP route."""
        payload: dict[str, Any] = {
            "ok": self.ok,
            "status": self.status,
            "planner": self.planner,
            "model": self.model,
            "credential_env": self.credential_env,
            "credential_present": self.credential_present,
            "base_url": self.base_url,
            "base_url_env": self.base_url_env,
            "detail": self.detail,
            "reply": self.reply,
            "latency_s": self.latency_s,
        }
        if self.extra:
            payload.update(self.extra)
        return payload


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def check_llm(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the configured planner backend and classify the outcome.

    Never raises: every failure is returned as a typed
    :class:`LlmCheckResult`, so callers do not each need their own error
    taxonomy.

    This function is synchronous and starts its own event loop, so it must be
    called from a thread with no running loop. FastAPI's threadpool for ``def``
    routes satisfies this; calling it from inside a coroutine does not.

    Args:
        request: The backend, model, base URL, and timeout to check.

    Returns:
        The structured outcome. ``result.ok`` is the single success signal.
    """
    planner = request.planner
    if planner not in CHECK_PLANNERS:
        return LlmCheckResult(
            ok=False,
            status=STATUS_MISSING_CONFIG,
            planner=planner,
            model=request.model,
            detail=(
                f"unknown planner {planner!r}; expected one of "
                f"{', '.join(CHECK_PLANNERS)}"
            ),
        )

    if _running_loop():
        return LlmCheckResult(
            ok=False,
            status=STATUS_SDK_ERROR,
            planner=planner,
            model=request.model,
            detail=(
                "check_llm() is synchronous and cannot run inside an active "
                "event loop; call it from a worker thread instead."
            ),
        )

    checker = {
        "api": _check_api,
        "claude_code": _check_claude_code,
        "codex": _check_codex,
    }[planner]

    logger.info(
        "checking planner %s (model=%s, timeout=%ds)",
        planner,
        request.model or "<backend default>",
        request.resolved_timeout_s(),
    )
    try:
        return checker(request)
    except Exception as exc:  # noqa: BLE001 - the contract is to never raise
        return LlmCheckResult(
            ok=False,
            status=STATUS_SDK_ERROR,
            planner=planner,
            model=request.model,
            detail=_describe(exc),
        )


def _running_loop() -> bool:
    """Return True when the calling thread already drives an event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def redact_secrets(text: str) -> str:
    """Replace any configured credential value with its variable name.

    Args:
        text: Text that may embed a credential, such as a provider error body.

    Returns:
        The text with every known key value replaced by ``<VAR_NAME>``.
    """
    for env_name in _SECRET_ENV_NAMES:
        value = os.environ.get(env_name)
        if value and len(value) >= _MIN_SECRET_LEN and value in text:
            text = text.replace(value, f"<{env_name}>")
    return text


def _describe(exc: BaseException) -> str:
    """Render an exception as ``TypeName: message``, redacted and capped."""
    text = redact_secrets(f"{type(exc).__name__}: {exc}")
    if len(text) > _DETAIL_LIMIT:
        return text[:_DETAIL_LIMIT] + " …[truncated]"
    return text


# ---------------------------------------------------------------------------
# api planner
# ---------------------------------------------------------------------------


def _check_api(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the pydantic-ai ``api`` planner with one minimal request."""
    model = (request.model or "").strip()
    provider_name = model.split(":", 1)[0] if ":" in model else ""
    key_env = _API_KEY_ENV.get(provider_name)
    base_url_env = _API_BASE_URL_ENV.get(provider_name)
    key_present = bool(os.environ.get(key_env)) if key_env else None

    def _result(status: str, **kwargs: Any) -> LlmCheckResult:
        return LlmCheckResult(
            ok=status == STATUS_OK,
            status=status,
            planner="api",
            model=model or None,
            credential_env=key_env,
            credential_present=key_present,
            base_url=request.base_url,
            base_url_env=base_url_env,
            **kwargs,
        )

    if not model:
        return _result(
            STATUS_MISSING_CONFIG,
            detail=(
                "the 'api' planner requires a model id; pass --model with a "
                "provider prefix (e.g. 'anthropic:claude-opus-4-8', "
                "'openai:gpt-5.5', 'openai-chat:glm-5.2')."
            ),
        )
    if not provider_name:
        return _result(
            STATUS_MISSING_CONFIG,
            detail=(
                f"model {model!r} has no provider prefix; expected "
                f"'<provider>:<model>', e.g. 'anthropic:{model}'."
            ),
        )
    if key_env is not None and not key_present:
        return _result(
            STATUS_MISSING_API_KEY,
            detail=f"{key_env} is not set for provider {provider_name!r}.",
        )

    try:
        from pydantic_ai import Agent, ModelSettings
        from pydantic_ai.exceptions import ModelHTTPError, UserError

        from rpent.planner.base import build_api_model
    except ImportError as exc:
        return _result(STATUS_SDK_ERROR, detail=_describe(exc))

    try:
        api_model = build_api_model(model, request.base_url)
    except UserError as exc:
        return _result(_classify_user_error(exc), detail=_describe(exc))
    except ValueError as exc:
        # ``model`` is non-empty by the checks above, so build_api_model's own
        # ValueError cannot fire here: this comes from provider resolution,
        # where infer_provider raises ValueError for an unknown prefix.
        return _result(STATUS_UNSUPPORTED_PROVIDER, detail=_describe(exc))

    timeout_s = request.resolved_timeout_s()
    started = time.monotonic()
    try:
        reply = asyncio.run(
            asyncio.wait_for(
                _run_api_probe(Agent, ModelSettings, api_model),
                timeout=timeout_s,
            )
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _result(
            STATUS_NETWORK_ERROR,
            detail=f"the request did not complete within {timeout_s}s.",
            latency_s=round(time.monotonic() - started, 3),
        )
    except ModelHTTPError as exc:
        status = (
            STATUS_AUTH_FAILED
            if exc.status_code in (401, 403)
            else STATUS_PROVIDER_ERROR
        )
        return _result(
            status,
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )
    except UserError as exc:
        return _result(_classify_user_error(exc), detail=_describe(exc))
    except Exception as exc:  # noqa: BLE001 - classified below
        return _result(
            _classify_transport_error(exc),
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )

    latency_s = round(time.monotonic() - started, 3)
    if not reply.strip():
        return _result(
            STATUS_PROVIDER_ERROR,
            detail="the provider returned an empty response.",
            latency_s=latency_s,
        )
    return _result(STATUS_OK, reply=reply.strip(), latency_s=latency_s)


async def _run_api_probe(agent_cls: Any, settings_cls: Any, api_model: Any) -> str:
    """Run the minimal tool-free pydantic-ai request and return its text."""
    agent = agent_cls(api_model)
    result = await agent.run(
        PROBE_PROMPT,
        model_settings=settings_cls(max_tokens=PROBE_MAX_TOKENS),
    )
    return str(result.output or "")


def _classify_user_error(exc: Exception) -> str:
    """Map a pydantic-ai ``UserError`` onto a check status.

    pydantic-ai reports an unresolvable model id as ``Unknown model: <id>``
    and a missing credential as ``Set the `<PROVIDER>_API_KEY` environment
    variable ...``.

    Args:
        exc: The ``UserError`` raised while resolving the model.

    Returns:
        The matching status constant.
    """
    text = str(exc)
    if text.startswith("Unknown model"):
        return STATUS_UNSUPPORTED_PROVIDER
    if "_API_KEY" in text or "environment variable" in text:
        return STATUS_MISSING_API_KEY
    return STATUS_SDK_ERROR


def _classify_transport_error(exc: Exception) -> str:
    """Map a transport-layer exception onto a network or SDK status.

    Args:
        exc: The exception raised while performing the request.

    Returns:
        :data:`STATUS_NETWORK_ERROR` for transport failures, else
        :data:`STATUS_SDK_ERROR`.
    """
    try:
        import httpx
    except ImportError:
        return STATUS_SDK_ERROR
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return STATUS_NETWORK_ERROR
    return STATUS_SDK_ERROR


# ---------------------------------------------------------------------------
# claude_code planner
# ---------------------------------------------------------------------------


def _check_claude_code(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the Claude Agent SDK with one tool-free, single-turn query."""
    model = (request.model or "").strip() or "sonnet"
    key_present = bool(os.environ.get(_CLAUDE_CODE_KEY_ENV))

    def _result(status: str, **kwargs: Any) -> LlmCheckResult:
        return LlmCheckResult(
            ok=status == STATUS_OK,
            status=status,
            planner="claude_code",
            model=model,
            credential_env=_CLAUDE_CODE_KEY_ENV,
            credential_present=key_present,
            **kwargs,
        )

    try:
        import claude_agent_sdk
    except ImportError as exc:
        return _result(STATUS_SDK_ERROR, detail=_describe(exc))

    timeout_s = request.resolved_timeout_s()
    started = time.monotonic()
    try:
        reply = asyncio.run(
            asyncio.wait_for(
                _run_claude_probe(claude_agent_sdk, model),
                timeout=timeout_s,
            )
        )
    except (asyncio.TimeoutError, TimeoutError):
        return _result(
            STATUS_NETWORK_ERROR,
            detail=f"the Claude Agent SDK did not respond within {timeout_s}s.",
            latency_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001 - classified below
        return _result(
            _classify_sdk_error(exc, key_present=key_present),
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )

    latency_s = round(time.monotonic() - started, 3)
    if not reply.strip():
        return _result(
            STATUS_PROVIDER_ERROR,
            detail="the Claude Agent SDK returned no assistant text.",
            latency_s=latency_s,
        )
    return _result(STATUS_OK, reply=reply.strip(), latency_s=latency_s)


async def _run_claude_probe(sdk: Any, model: str) -> str:
    """Consume one tool-free Claude Agent SDK turn and return its text."""
    options = sdk.ClaudeAgentOptions(
        model=model,
        max_turns=1,
        tools=None,
        allowed_tools=[],
        mcp_servers={},
    )
    chunks: list[str] = []
    async for message in sdk.query(prompt=PROBE_PROMPT, options=options):
        chunks.extend(_assistant_text(message))
    return "".join(chunks)


def _assistant_text(message: Any) -> list[str]:
    """Extract plain assistant text from one Claude Agent SDK message."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            out.append(text)
    return out


# ---------------------------------------------------------------------------
# codex planner
# ---------------------------------------------------------------------------


def _check_codex(request: LlmCheckRequest) -> LlmCheckResult:
    """Probe the Codex SDK with one turn and no MCP server attached."""
    model = (request.model or "").strip() or os.environ.get("CODEX_MODEL") or None
    key_present = bool(os.environ.get(_CODEX_KEY_ENV))
    base_url = request.base_url or os.environ.get(_CODEX_BASE_URL_ENV) or None

    def _result(status: str, **kwargs: Any) -> LlmCheckResult:
        return LlmCheckResult(
            ok=status == STATUS_OK,
            status=status,
            planner="codex",
            model=model,
            credential_env=_CODEX_KEY_ENV,
            credential_present=key_present,
            base_url=base_url,
            base_url_env=_CODEX_BASE_URL_ENV,
            **kwargs,
        )

    try:
        from rpent.planner.codex import build_probe_config
    except ImportError as exc:
        return _result(STATUS_SDK_ERROR, detail=_describe(exc))

    timeout_s = request.resolved_timeout_s()
    started = time.monotonic()
    try:
        reply = _run_codex_probe(build_probe_config(base_url), model, timeout_s)
    except (asyncio.TimeoutError, TimeoutError):
        return _result(
            STATUS_NETWORK_ERROR,
            detail=f"the Codex SDK did not respond within {timeout_s}s.",
            latency_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001 - classified below
        return _result(
            _classify_sdk_error(exc, key_present=key_present),
            detail=_describe(exc),
            latency_s=round(time.monotonic() - started, 3),
        )

    latency_s = round(time.monotonic() - started, 3)
    if not reply.strip():
        return _result(
            STATUS_PROVIDER_ERROR,
            detail="the Codex SDK returned no assistant text.",
            latency_s=latency_s,
        )
    return _result(STATUS_OK, reply=reply.strip(), latency_s=latency_s)


def _run_codex_probe(config: Any, model: str | None, timeout_s: int) -> str:
    """Run one Codex turn against ``config`` and return its assistant text."""
    import openai_codex

    options: dict[str, Any] = {
        "approval_mode": openai_codex.ApprovalMode.deny_all,
        "sandbox": openai_codex.Sandbox.read_only,
    }
    if model:
        options["model"] = model

    deadline = time.monotonic() + timeout_s
    chunks: list[str] = []
    with openai_codex.Codex(config=config) as codex:
        thread = codex.thread_start(**options)
        turn = thread.turn(PROBE_PROMPT, **options)
        for event in turn:
            chunks.extend(_codex_event_text(event))
            if time.monotonic() > deadline:
                raise TimeoutError(f"the Codex SDK did not finish within {timeout_s}s.")
    return "".join(chunks)


def _codex_event_text(event: Any) -> list[str]:
    """Extract assistant text from one Codex stream event."""
    item = getattr(event, "item", None) or event
    text = getattr(item, "text", None)
    if isinstance(text, str) and text:
        return [text]
    content = getattr(item, "content", None)
    if isinstance(content, str) and content:
        return [content]
    if isinstance(content, list):
        out: list[str] = []
        for block in content:
            block_text = getattr(block, "text", None)
            if isinstance(block_text, str) and block_text:
                out.append(block_text)
        return out
    return []


def _classify_sdk_error(exc: Exception, *, key_present: bool) -> str:
    """Classify a child-process SDK failure from its message.

    The Claude and Codex SDKs surface provider problems as SDK exceptions
    rather than typed HTTP errors, so the message is the only signal.

    Args:
        exc: The exception raised by the SDK.
        key_present: Whether the backend's credential env var was set.

    Returns:
        The matching status constant.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return STATUS_NETWORK_ERROR
    if any(
        token in text
        for token in ("401", "403", "unauthor", "authentication", "invalid api key")
    ):
        return STATUS_AUTH_FAILED
    if any(
        token in text
        for token in ("not logged in", "no credentials", "login", "api key")
    ):
        return STATUS_MISSING_API_KEY if not key_present else STATUS_AUTH_FAILED
    if any(
        token in text
        for token in ("connection", "dns", "resolve", "network", "unreachable", "tls")
    ):
        return STATUS_NETWORK_ERROR
    if any(token in text for token in ("429", "rate limit", "500", "502", "503")):
        return STATUS_PROVIDER_ERROR
    return STATUS_SDK_ERROR
