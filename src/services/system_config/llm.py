# -*- coding: utf-8 -*-
"""LLM 渠道发现、能力校验与诊断（自 system_config_service 拆出，v1.1.1 重构）。

公开入口 ``src.services.system_config_service.SystemConfigService`` 通过多继承
聚合各 Mixin，方法经 MRO 解析，行为与拆分前完全一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from src.config import (
    ANSPIRE_LLM_BASE_URL_DEFAULT,
    ANSPIRE_LLM_MODEL_DEFAULT,
    Config,
    SUPPORTED_LLM_CHANNEL_API_SURFACES,
    SUPPORTED_LLM_CHANNEL_PROTOCOLS,
    _get_litellm_provider,
    _uses_direct_env_provider,
    apply_litellm_api_surface,
    canonicalize_llm_channel_api_surface,
    canonicalize_llm_channel_protocol,
    channel_allows_empty_api_key,
    find_incompatible_llm_channel_models,
    find_llm_channel_surface_conflicts,
    get_configured_llm_models,
    is_supported_llm_channel_api_surface_value,
    normalize_agent_litellm_model,
    normalize_llm_channel_api_surface,
    normalize_llm_channel_model,
    parse_env_bool,
    resolve_llm_channel_protocol,
)
from src.llm.errors import call_litellm_with_param_recovery
from src.llm.generation_params import apply_litellm_generation_params
from src.llm.hermes import (
    HERMES_DEFAULT_BASE_URL,
    HERMES_DEFAULT_MODEL,
    HERMES_DEFAULT_PROTOCOL,
    build_hermes_redaction_values,
    canonicalize_hermes_base_url,
    canonicalize_hermes_model_ref,
    is_reserved_hermes_name,
    open_hermes_no_proxy_client,
    parse_hermes_channel,
)
from src.llm.response_content import strip_leading_think_wrapper
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import (
    urlparse,
    urlunparse,
)
import json
import re
import requests
import time

logger = logging.getLogger("src.services.system_config_service")


@dataclass(frozen=True)
class _LLMDiagnostic:
    """Internal structured diagnosis for LLM test and discovery failures."""

    error_code: str
    retryable: bool
    message: str
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class SystemConfigLLMMixin:
    """Mixin：见模块 docstring。"""

    def discover_llm_channel_models(
        self,
        *,
        name: str,
        protocol: str,
        base_url: str,
        api_key: str,
        models: Sequence[str] = (),
        timeout_seconds: float = 20.0,
        use_saved_secret: bool = False,
    ) -> Dict[str, Any]:
        """Discover available models from an OpenAI-compatible `/models` endpoint."""
        channel_name = name.strip() or "channel"
        resolved_secret, secret_error, redaction_values = self._resolve_hermes_saved_secret(
            channel_name=channel_name,
            protocol=protocol,
            base_url=base_url,
            submitted_api_key=api_key,
            use_saved_secret=use_saved_secret,
            stage="model_discovery",
        )
        if resolved_secret is None:
            return secret_error
        api_key = resolved_secret
        redaction_values.update(self._build_redaction_values(api_key))
        if is_reserved_hermes_name(channel_name):
            secret_error = self._validate_hermes_submitted_secret(
                api_key=api_key,
                use_saved_secret=use_saved_secret,
                stage="model_discovery",
                models=[],
                redaction_values=redaction_values,
            )
            if secret_error is not None:
                return secret_error
            try:
                base_url = canonicalize_hermes_base_url(base_url)
            except ValueError as exc:
                return self._build_llm_channel_result(
                    success=False,
                    message="Hermes Base URL is invalid",
                    error=str(exc),
                    stage="model_discovery",
                    error_code="invalid_config",
                    retryable=False,
                    details={
                        "issue_key": "discover_channel_BASE_URL",
                        "issue_code": "invalid_hermes_url",
                        "reason": "invalid_hermes_url",
                    },
                    resolved_protocol=None,
                    models=[],
                    latency_ms=None,
                    redaction_values=redaction_values,
                )
        existing_models = [str(m).strip() for m in models if str(m).strip()]
        validation_issues, resolved_protocol = self._validate_llm_channel_connection(
            channel_name=channel_name,
            protocol_value=protocol,
            base_url_value=base_url,
            api_key_value=api_key,
            model_values=existing_models,
            field_prefix="discover_channel",
            require_base_url=True,
        )
        if not resolved_protocol and existing_models:
            resolved_protocol = resolve_llm_channel_protocol(
                protocol,
                base_url=base_url,
                models=existing_models,
                channel_name=channel_name,
            )
        errors = [issue for issue in validation_issues if issue["severity"] == "error"]
        if errors:
            return self._build_llm_channel_result(
                success=False,
                message="LLM channel configuration is invalid",
                error=errors[0]["message"],
                stage="model_discovery",
                error_code="invalid_config",
                retryable=False,
                details={
                    "issue_key": errors[0]["key"],
                    "issue_code": errors[0]["code"],
                    "reason": errors[0]["code"],
                },
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=None,
                redaction_values=redaction_values,
            )

        if resolved_protocol not in {"openai", "deepseek"}:
            return self._build_llm_channel_result(
                success=False,
                message="Model discovery is not supported for this protocol",
                error=(
                    f"LLM channel '{channel_name}' protocol '{resolved_protocol}' "
                    "does not support /models discovery yet"
                ),
                stage="model_discovery",
                error_code="unsupported_protocol",
                retryable=False,
                details={"protocol": resolved_protocol or None},
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=None,
                redaction_values=redaction_values,
            )

        api_keys = [segment.strip() for segment in api_key.split(",") if segment.strip()]
        selected_api_key = api_keys[0] if api_keys else ""
        redaction_values.update(self._build_redaction_values(selected_api_key))
        request_headers = {"Accept": "application/json"}
        if selected_api_key:
            request_headers["Authorization"] = f"Bearer {selected_api_key}"

        try:
            models_url = self._build_llm_models_url(base_url)
        except ValueError as exc:
            return self._build_llm_channel_result(
                success=False,
                message="LLM channel configuration is invalid",
                error=str(exc),
                stage="model_discovery",
                error_code="invalid_config",
                retryable=False,
                details={
                    "issue_key": "discover_channel_BASE_URL",
                    "issue_code": "invalid_url",
                    "reason": "invalid_url",
                },
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=None,
                redaction_values=redaction_values,
            )

        try:
            started_at = time.perf_counter()
            if is_reserved_hermes_name(channel_name):
                session = requests.Session()
                session.trust_env = False
                try:
                    response = session.get(
                        models_url,
                        headers=request_headers,
                        timeout=max(5.0, float(timeout_seconds)),
                        allow_redirects=False,
                    )
                finally:
                    session.close()
            else:
                response = requests.get(
                    models_url,
                    headers=request_headers,
                    timeout=max(5.0, float(timeout_seconds)),
                    allow_redirects=False,
                )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
        except requests.RequestException as exc:
            logger.warning(
                "LLM channel model discovery failed for %s: %s",
                channel_name,
                self._sanitize_llm_error_text(exc, redaction_values=redaction_values),
            )
            diagnostic = self._classify_llm_exception(exc)
            return self._build_llm_channel_result(
                success=False,
                message=diagnostic.message,
                error=str(exc),
                stage="model_discovery",
                error_code=diagnostic.error_code,
                retryable=diagnostic.retryable,
                details=self._merge_llm_diagnostic_details({"endpoint": models_url}, diagnostic),
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=None,
                redaction_values=redaction_values,
            )

        if 300 <= response.status_code < 400:
            return self._build_llm_channel_result(
                success=False,
                message="Model discovery request was redirected",
                error="Redirect responses are not allowed for model discovery",
                stage="model_discovery",
                error_code="network_error",
                retryable=False,
                details={"endpoint": models_url, "http_status": response.status_code},
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=latency_ms,
                redaction_values=redaction_values,
            )

        if not response.ok:
            error_text = self._extract_llm_discovery_error(response)
            diagnostic = self._classify_llm_http_error(
                status_code=response.status_code,
                error_text=error_text,
            )
            return self._build_llm_channel_result(
                success=False,
                message=diagnostic.message,
                error=error_text,
                stage="model_discovery",
                error_code=diagnostic.error_code,
                retryable=diagnostic.retryable,
                details=self._merge_llm_diagnostic_details(
                    {"endpoint": models_url, "http_status": response.status_code},
                    diagnostic,
                ),
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=latency_ms,
                redaction_values=redaction_values,
            )

        try:
            payload = response.json()
        except ValueError:
            return self._build_llm_channel_result(
                success=False,
                message="Model discovery returned invalid JSON",
                error="The /models endpoint did not return valid JSON",
                stage="response_parse",
                error_code="format_error",
                retryable=False,
                details={"endpoint": models_url, "http_status": response.status_code, "reason": "non_json"},
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=latency_ms,
                redaction_values=redaction_values,
            )

        models = self._extract_discovered_llm_models(payload)
        if not models:
            return self._build_llm_channel_result(
                success=False,
                message="Model discovery returned no models",
                error="The /models endpoint did not return any model IDs",
                stage="response_parse",
                error_code="empty_response",
                retryable=False,
                details={"endpoint": models_url, "http_status": response.status_code, "reason": "empty_models"},
                resolved_protocol=resolved_protocol or None,
                models=[],
                latency_ms=latency_ms,
                redaction_values=redaction_values,
            )

        return self._build_llm_channel_result(
            success=True,
            message="LLM channel model discovery succeeded",
            error=None,
            stage="model_discovery",
            error_code=None,
            retryable=False,
            details={"endpoint": models_url, "model_count": len(models)},
            resolved_protocol=resolved_protocol or None,
            models=models,
            latency_ms=latency_ms,
            redaction_values=redaction_values,
        )


    def test_llm_channel(
        self,
        *,
        name: str,
        protocol: str,
        api_surface: str = "chat_completions",
        base_url: str,
        api_key: str,
        models: Sequence[str],
        enabled: bool = True,
        timeout_seconds: float = 20.0,
        capability_checks: Sequence[str] = (),
        use_saved_secret: bool = False,
    ) -> Dict[str, Any]:
        """Run a minimal completion call against one channel definition."""
        requested_capabilities = self._normalize_llm_capability_checks(capability_checks)
        raw_models = [str(model).strip() for model in models if str(model).strip()]
        channel_name = name.strip() or "channel"
        resolved_api_surface = normalize_llm_channel_api_surface(api_surface)
        generation_stage = "responses" if resolved_api_surface == "responses" else "chat_completion"
        resolved_secret, secret_error, redaction_values = self._resolve_hermes_saved_secret(
            channel_name=channel_name,
            protocol=protocol,
            base_url=base_url,
            submitted_api_key=api_key,
            use_saved_secret=use_saved_secret,
            stage=generation_stage,
        )
        if resolved_secret is None:
            result = secret_error
            if requested_capabilities and "capability_results" not in result:
                result["capability_results"] = self._build_skipped_capability_results(
                    requested_capabilities,
                    "base_test_failed",
                    "Skipped because the base channel test did not pass",
                    redaction_values=redaction_values,
                )
            return result
        api_key = resolved_secret
        redaction_values.update(self._build_redaction_values(api_key))
        if is_reserved_hermes_name(channel_name):
            secret_error = self._validate_hermes_submitted_secret(
                api_key=api_key,
                use_saved_secret=use_saved_secret,
                stage=generation_stage,
                capability_checks=requested_capabilities,
                redaction_values=redaction_values,
            )
            if secret_error is not None:
                return secret_error
            try:
                base_url = canonicalize_hermes_base_url(base_url)
            except ValueError as exc:
                return self._build_llm_channel_result(
                    success=False,
                    message="Hermes Base URL is invalid",
                    error=str(exc),
                    stage=generation_stage,
                    error_code="invalid_config",
                    retryable=False,
                    details={
                        "issue_key": "test_channel_BASE_URL",
                        "issue_code": "invalid_hermes_url",
                        "reason": "invalid_hermes_url",
                    },
                    resolved_protocol=None,
                    resolved_model=None,
                    latency_ms=None,
                    capability_results=self._build_skipped_capability_results(
                        requested_capabilities,
                        "base_test_failed",
                        "Skipped because the base channel test did not pass",
                        redaction_values=redaction_values,
                    ),
                    redaction_values=redaction_values,
                )
        validation_issues = self._validate_llm_channel_definition(
            channel_name=channel_name,
            protocol_value=protocol,
            api_surface_value=api_surface,
            base_url_value=base_url,
            api_key_value=api_key,
            model_values=raw_models,
            enabled=enabled,
            field_prefix="test_channel",
            require_complete=True,
        )
        errors = [issue for issue in validation_issues if issue["severity"] == "error"]
        if errors:
            return self._build_llm_channel_result(
                success=False,
                message="LLM channel configuration is invalid",
                error=errors[0]["message"],
                stage=generation_stage,
                error_code="invalid_config",
                retryable=False,
                details={
                    "issue_key": errors[0]["key"],
                    "issue_code": errors[0]["code"],
                    "reason": errors[0]["code"],
                },
                resolved_protocol=None,
                resolved_model=None,
                latency_ms=None,
                capability_results=self._build_skipped_capability_results(
                    requested_capabilities,
                    "base_test_failed",
                    "Skipped because the base channel test did not pass",
                    redaction_values=redaction_values,
                ),
                redaction_values=redaction_values,
            )

        resolved_protocol = resolve_llm_channel_protocol(protocol, base_url=base_url, models=raw_models, channel_name=name)
        resolved_models = [normalize_llm_channel_model(model, resolved_protocol, base_url) for model in raw_models]
        resolved_model = resolved_models[0]
        if is_reserved_hermes_name(channel_name):
            resolved_model = canonicalize_hermes_model_ref(raw_models[0]).wire_model
        wire_model = apply_litellm_api_surface(resolved_model, resolved_api_surface)
        api_keys = [segment.strip() for segment in api_key.split(",") if segment.strip()]
        selected_api_key = api_keys[0] if api_keys else ""
        redaction_values.update(self._build_redaction_values(selected_api_key))

        call_kwargs: Dict[str, Any] = {
            "model": wire_model,
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 256,  # Increased to allow MiniMax-M3 thinking process + response
            "timeout": max(5.0, float(timeout_seconds)),
        }
        if selected_api_key:
            call_kwargs["api_key"] = selected_api_key
        if base_url.strip():
            call_kwargs["api_base"] = base_url.strip()
        call_kwargs = apply_litellm_generation_params(
            call_kwargs,
            wire_model,
            self._get_runtime_llm_temperature(),
        )

        try:
            import litellm
            from src.agent.llm_adapter import (
                resolve_fallback_litellm_wire_models,
                register_fallback_model_pricing,
            )

            # Register fallback pricing for OpenAI-compatible models to prevent cost calculation errors
            config_model_list = None
            if getattr(self, "_config", None) is not None:
                config_model_list = getattr(self._config, "llm_model_list", None)
            register_fallback_model_pricing(
                resolve_fallback_litellm_wire_models(
                    resolved_model,
                    config_model_list,
                )
            )

            started_at = time.perf_counter()
            if is_reserved_hermes_name(channel_name):
                with open_hermes_no_proxy_client(
                    api_key=selected_api_key,
                    base_url=base_url,
                    timeout=max(5.0, float(timeout_seconds)),
                ) as client:
                    hermes_call_kwargs = dict(call_kwargs)
                    hermes_call_kwargs["stream"] = False
                    hermes_call_kwargs["client"] = client
                    hermes_call_kwargs.pop("api_key", None)
                    hermes_call_kwargs.pop("api_base", None)
                    response = call_litellm_with_param_recovery(
                        lambda kwargs: litellm.completion(**kwargs),
                        model=wire_model,
                        call_kwargs=hermes_call_kwargs,
                        logger=logger,
                        log_label="[Hermes channel test]",
                    )
            else:
                response = call_litellm_with_param_recovery(
                    lambda kwargs: litellm.completion(**kwargs),
                    model=wire_model,
                    call_kwargs=call_kwargs,
                    logger=logger,
                    log_label="[LLM channel test]",
                )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            content, parse_error_code, parse_error, parse_reason = self._extract_llm_completion_content(response)
            if parse_error_code:
                message = (
                    "LLM channel returned an empty response"
                    if parse_error_code == "empty_response"
                    else "LLM channel returned an unexpected response format"
                )
                return self._build_llm_channel_result(
                    success=False,
                    message=message,
                    error=parse_error,
                    stage="response_parse",
                    error_code=parse_error_code,
                    retryable=False,
                    details={"response_error": parse_error, "reason": parse_reason},
                    resolved_protocol=resolved_protocol or None,
                    resolved_model=resolved_model,
                    resolved_api_surface=resolved_api_surface,
                    latency_ms=latency_ms,
                    capability_results=self._build_skipped_capability_results(
                        requested_capabilities,
                        "base_test_failed",
                        "Skipped because the base channel test did not pass",
                        redaction_values=redaction_values,
                    ),
                    redaction_values=redaction_values,
                )

            capability_results: Dict[str, Any] = {}
            if requested_capabilities and is_reserved_hermes_name(channel_name):
                capability_results = self._run_hermes_capability_checks(
                    litellm_module=litellm,
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    capability_checks=requested_capabilities,
                    redaction_values=redaction_values,
                )
            elif requested_capabilities:
                capability_results = self._run_llm_capability_checks(
                    litellm_module=litellm,
                    resolved_model=wire_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    capability_checks=requested_capabilities,
                )
            return self._build_llm_channel_result(
                success=True,
                message="LLM channel test succeeded",
                error=None,
                stage=generation_stage,
                error_code=None,
                retryable=False,
                details={"response_preview": content[:80]},
                resolved_protocol=resolved_protocol or None,
                resolved_model=resolved_model,
                resolved_api_surface=resolved_api_surface,
                latency_ms=latency_ms,
                capability_results=capability_results,
                redaction_values=redaction_values,
            )
        except Exception as exc:
            logger.warning(
                "LLM channel test failed for %s: %s",
                channel_name,
                self._sanitize_llm_error_text(exc, redaction_values=redaction_values),
            )
            diagnostic = self._classify_llm_exception(exc)
            return self._build_llm_channel_result(
                success=False,
                message=diagnostic.message,
                error=str(exc),
                stage=generation_stage,
                error_code=diagnostic.error_code,
                retryable=diagnostic.retryable,
                details=self._merge_llm_diagnostic_details({"model": resolved_model}, diagnostic),
                resolved_protocol=resolved_protocol or None,
                resolved_model=resolved_model,
                resolved_api_surface=resolved_api_surface,
                latency_ms=None,
                redaction_values=redaction_values,
                capability_results=self._build_skipped_capability_results(
                    requested_capabilities,
                    "base_test_failed",
                    "Skipped because the base channel test did not pass",
                    redaction_values=redaction_values,
                ),
            )


    @classmethod
    def _normalize_llm_capability_checks(cls, capability_checks: Sequence[str]) -> List[str]:
        requested = {str(check).strip().lower() for check in capability_checks if str(check).strip()}
        return [check for check in cls._LLM_CAPABILITY_ORDER if check in requested]


    @classmethod
    def _build_skipped_capability_results(
        cls,
        capability_checks: Sequence[str],
        reason: str,
        message: str,
        *,
        redaction_values: Optional[Set[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            capability: cls._build_llm_capability_result(
                capability=capability,
                status="skipped",
                message=message,
                error_code="skipped",
                retryable=False,
                details={"reason": reason},
                redaction_values=redaction_values,
            )
            for capability in capability_checks
        }


    @classmethod
    def _run_hermes_capability_checks(
        cls,
        *,
        litellm_module: Any,
        resolved_model: str,
        selected_api_key: str,
        base_url: str,
        timeout_seconds: float,
        capability_checks: Sequence[str],
        redaction_values: Optional[Set[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for capability in capability_checks:
            if capability != "json":
                results[capability] = cls._build_llm_capability_result(
                    capability=capability,
                    status="skipped",
                    message="Hermes Phase 3 does not probe this capability",
                    error_code="not_probed",
                    retryable=False,
                    details={"reason": "not_probed"},
                    redaction_values=redaction_values,
                )
                continue
            try:
                started_at = time.perf_counter()
                with open_hermes_no_proxy_client(
                    api_key=selected_api_key,
                    base_url=base_url,
                    timeout=max(5.0, float(timeout_seconds)),
                ) as client:
                    call_kwargs = cls._build_llm_capability_completion_kwargs(
                        resolved_model=resolved_model,
                        selected_api_key=selected_api_key,
                        base_url=base_url,
                        timeout_seconds=timeout_seconds,
                        messages=[{"role": "user", "content": 'Return exactly this JSON object: {"status":"ok"}'}],
                        max_tokens=64,
                        extra={"response_format": {"type": "json_object"}, "client": client},
                    )
                    call_kwargs.pop("api_key", None)
                    call_kwargs.pop("api_base", None)
                    response = litellm_module.completion(**call_kwargs)
                    content, parse_error_code, parse_error, parse_reason = cls._extract_llm_completion_content(response)
                latency_ms = int((time.perf_counter() - started_at) * 1000)
                if parse_error_code:
                    results[capability] = cls._build_llm_capability_result(
                        capability="json",
                        status="failed",
                        message="JSON capability check returned no parseable content",
                        error_code=parse_error_code,
                        retryable=False,
                        latency_ms=latency_ms,
                        details={"reason": parse_reason, "response_error": parse_error},
                        redaction_values=redaction_values,
                    )
                    continue
                try:
                    payload = json.loads(content)
                except ValueError:
                    payload = None
                if not isinstance(payload, dict) or payload.get("status") != "ok":
                    results[capability] = cls._build_llm_capability_result(
                        capability="json",
                        status="failed",
                        message="JSON capability check returned non-JSON content",
                        error_code="format_error",
                        retryable=False,
                        latency_ms=latency_ms,
                        details={"reason": "non_json", "response_preview": content[:80]},
                        redaction_values=redaction_values,
                    )
                    continue
                results[capability] = cls._build_llm_capability_result(
                    capability="json",
                    status="passed",
                    message="JSON output capability check passed",
                    latency_ms=latency_ms,
                    details={"reason": "json_valid"},
                    redaction_values=redaction_values,
                )
            except Exception as exc:
                diagnostic = cls._classify_llm_capability_exception(exc, "json")
                results[capability] = cls._build_llm_capability_result_from_diagnostic(
                    "json",
                    diagnostic,
                    cls._sanitize_llm_error_text(exc, redaction_values=redaction_values),
                    redaction_values=redaction_values,
                )
        return results


    @classmethod
    def _run_llm_capability_checks(
        cls,
        *,
        litellm_module: Any,
        resolved_model: str,
        selected_api_key: str,
        base_url: str,
        timeout_seconds: float,
        capability_checks: Sequence[str],
    ) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for capability in capability_checks:
            if capability == "json":
                results[capability] = cls._run_json_capability_check(
                    litellm_module=litellm_module,
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
            elif capability == "tools":
                results[capability] = cls._run_tools_capability_check(
                    litellm_module=litellm_module,
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
            elif capability == "stream":
                results[capability] = cls._run_stream_capability_check(
                    litellm_module=litellm_module,
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
            elif capability == "vision":
                results[capability] = cls._run_vision_capability_check(
                    litellm_module=litellm_module,
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                )
        return results


    @classmethod
    def _run_json_capability_check(
        cls,
        *,
        litellm_module: Any,
        resolved_model: str,
        selected_api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        try:
            started_at = time.perf_counter()
            response = litellm_module.completion(
                **cls._build_llm_capability_completion_kwargs(
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    messages=[{"role": "user", "content": 'Return exactly this JSON object: {"status":"ok"}'}],
                    max_tokens=64,
                    extra={"response_format": {"type": "json_object"}},
                )
            )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            content, parse_error_code, parse_error, parse_reason = cls._extract_llm_completion_content(response)
            if parse_error_code:
                return cls._build_llm_capability_result(
                    capability="json",
                    status="failed",
                    message="JSON capability check returned no parseable content",
                    error_code=parse_error_code,
                    retryable=False,
                    latency_ms=latency_ms,
                    details={"reason": parse_reason, "response_error": parse_error},
                )
            try:
                payload = json.loads(content)
            except ValueError:
                return cls._build_llm_capability_result(
                    capability="json",
                    status="failed",
                    message="JSON capability check returned non-JSON content",
                    error_code="format_error",
                    retryable=False,
                    latency_ms=latency_ms,
                    details={"reason": "non_json", "response_preview": content[:80]},
                )
            if not isinstance(payload, dict) or payload.get("status") != "ok":
                return cls._build_llm_capability_result(
                    capability="json",
                    status="failed",
                    message="JSON capability check returned unexpected JSON",
                    error_code="format_error",
                    retryable=False,
                    latency_ms=latency_ms,
                    details={"reason": "non_json", "response_preview": content[:80]},
                )
            return cls._build_llm_capability_result(
                capability="json",
                status="passed",
                message="JSON output capability check passed",
                latency_ms=latency_ms,
                details={"reason": "json_valid"},
            )
        except Exception as exc:
            diagnostic = cls._classify_llm_capability_exception(exc, "json")
            return cls._build_llm_capability_result_from_diagnostic("json", diagnostic, str(exc))


    @classmethod
    def _run_tools_capability_check(
        cls,
        *,
        litellm_module: Any,
        resolved_model: str,
        selected_api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "dsa_probe_echo",
                    "description": "Return the provided text.",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ]
        try:
            started_at = time.perf_counter()
            response = litellm_module.completion(
                **cls._build_llm_capability_completion_kwargs(
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    messages=[{"role": "user", "content": "Call the dsa_probe_echo tool with text set to ok."}],
                    max_tokens=64,
                    extra={
                        "tools": tools,
                        "tool_choice": {"type": "function", "function": {"name": "dsa_probe_echo"}},
                    },
                )
            )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            tool_names = cls._extract_llm_tool_call_names(response)
            if "dsa_probe_echo" not in tool_names:
                return cls._build_llm_capability_result(
                    capability="tools",
                    status="failed",
                    message="Tool calling capability check did not return the probe tool call",
                    error_code="capability_unsupported",
                    retryable=False,
                    latency_ms=latency_ms,
                    details={"reason": "tool_calls_missing", "tool_calls": tool_names},
                )
            return cls._build_llm_capability_result(
                capability="tools",
                status="passed",
                message="Tool calling capability check passed",
                latency_ms=latency_ms,
                details={"reason": "tool_call_returned"},
            )
        except Exception as exc:
            diagnostic = cls._classify_llm_capability_exception(exc, "tools")
            return cls._build_llm_capability_result_from_diagnostic("tools", diagnostic, str(exc))


    @classmethod
    def _run_stream_capability_check(
        cls,
        *,
        litellm_module: Any,
        resolved_model: str,
        selected_api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        stream = None
        started_at = time.perf_counter()
        try:
            stream = litellm_module.completion(
                **cls._build_llm_capability_completion_kwargs(
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    messages=[{"role": "user", "content": "Reply with OK"}],
                    max_tokens=32,
                    extra={"stream": True},
                )
            )
            for index, chunk in enumerate(stream):
                content = cls._extract_llm_stream_chunk_content(chunk)
                if content:
                    latency_ms = int((time.perf_counter() - started_at) * 1000)
                    return cls._build_llm_capability_result(
                        capability="stream",
                        status="passed",
                        message="Streaming capability check passed",
                        latency_ms=latency_ms,
                        details={"reason": "stream_chunk_received"},
                    )
                if index + 1 >= cls._LLM_STREAM_CHUNK_LIMIT:
                    break
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            return cls._build_llm_capability_result(
                capability="stream",
                status="failed",
                message="Streaming capability check returned no content chunks",
                error_code="empty_response",
                retryable=False,
                latency_ms=latency_ms,
                details={"reason": "stream_no_content"},
            )
        except Exception as exc:
            diagnostic = cls._classify_llm_capability_exception(exc, "stream")
            return cls._build_llm_capability_result_from_diagnostic("stream", diagnostic, str(exc))
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                try:
                    close_stream()
                except Exception as exc:
                    logger.debug("Failed to close LLM stream capability probe: %s", exc)


    @classmethod
    def _run_vision_capability_check(
        cls,
        *,
        litellm_module: Any,
        resolved_model: str,
        selected_api_key: str,
        base_url: str,
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        try:
            started_at = time.perf_counter()
            response = litellm_module.completion(
                **cls._build_llm_capability_completion_kwargs(
                    resolved_model=resolved_model,
                    selected_api_key=selected_api_key,
                    base_url=base_url,
                    timeout_seconds=timeout_seconds,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Reply with OK if this image is visible."},
                                {"type": "image_url", "image_url": {"url": cls._LLM_CAPABILITY_PROBE_IMAGE}},
                            ],
                        }
                    ],
                    max_tokens=32,
                )
            )
            latency_ms = int((time.perf_counter() - started_at) * 1000)
            content, parse_error_code, parse_error, parse_reason = cls._extract_llm_completion_content(response)
            if parse_error_code:
                return cls._build_llm_capability_result(
                    capability="vision",
                    status="failed",
                    message="Vision capability check returned no parseable content",
                    error_code=parse_error_code,
                    retryable=False,
                    latency_ms=latency_ms,
                    details={"reason": parse_reason, "response_error": parse_error},
                )
            return cls._build_llm_capability_result(
                capability="vision",
                status="passed",
                message="Vision capability check passed",
                latency_ms=latency_ms,
                details={"reason": "vision_response_received", "response_preview": content[:80]},
            )
        except Exception as exc:
            diagnostic = cls._classify_llm_capability_exception(exc, "vision")
            return cls._build_llm_capability_result_from_diagnostic("vision", diagnostic, str(exc))


    @classmethod
    def _build_llm_capability_completion_kwargs(
        cls,
        *,
        resolved_model: str,
        selected_api_key: str,
        base_url: str,
        timeout_seconds: float,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError):
            timeout = 10.0
        call_kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "timeout": min(max(5.0, timeout), 10.0),
        }
        if selected_api_key:
            call_kwargs["api_key"] = selected_api_key
        if base_url.strip():
            call_kwargs["api_base"] = base_url.strip()
        if extra:
            call_kwargs.update(extra)
        call_kwargs = apply_litellm_generation_params(
            call_kwargs,
            resolved_model,
            0.0,
        )
        return call_kwargs


    @classmethod
    def _build_llm_capability_result(
        cls,
        *,
        capability: str,
        status: str,
        message: str,
        error_code: Optional[str] = None,
        retryable: bool = False,
        latency_ms: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
        redaction_values: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "message": cls._sanitize_llm_error_text(message, redaction_values=redaction_values),
            "error_code": error_code,
            "stage": f"capability_{capability}",
            "retryable": retryable,
            "latency_ms": latency_ms,
            "details": cls._sanitize_llm_details(
                {"capability": capability, **(details or {})},
                redaction_values=redaction_values,
            ),
        }


    @classmethod
    def _build_llm_capability_result_from_diagnostic(
        cls,
        capability: str,
        diagnostic: _LLMDiagnostic,
        error: str,
        *,
        redaction_values: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        details = cls._merge_llm_diagnostic_details({"error": error}, diagnostic)
        return cls._build_llm_capability_result(
            capability=capability,
            status="failed",
            message=diagnostic.message,
            error_code=diagnostic.error_code,
            retryable=diagnostic.retryable,
            details=details,
            redaction_values=redaction_values,
        )


    @staticmethod
    def _extract_llm_tool_call_names(response: Any) -> List[str]:
        choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
        if not choices:
            return []
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        else:
            tool_calls = getattr(message, "tool_calls", None) if message is not None else None
        names: List[str] = []
        for call in tool_calls or []:
            function = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
            if isinstance(function, dict):
                name = str(function.get("name") or "").strip()
            else:
                name = str(getattr(function, "name", "") or "").strip()
            if name:
                names.append(name)
        return names


    @staticmethod
    def _extract_llm_stream_chunk_content(chunk: Any) -> str:
        choices = chunk.get("choices") if isinstance(chunk, dict) else getattr(chunk, "choices", None)
        if not choices:
            return ""
        choice = choices[0]
        delta = choice.get("delta") if isinstance(choice, dict) else getattr(choice, "delta", None)
        message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
        for container in (delta, message):
            if not container:
                continue
            content = container.get("content") if isinstance(container, dict) else getattr(container, "content", None)
            if content:
                return str(content)
        content = choice.get("text") if isinstance(choice, dict) else getattr(choice, "text", None)
        return str(content or "")


    @classmethod
    def _classify_llm_capability_exception(cls, exc: Exception, capability: str) -> _LLMDiagnostic:
        text = str(exc).lower()
        capability_tokens = {
            "json": ("response_format", "json_object", "json mode"),
            "tools": ("tool_choice", "tools", "function calling", "tool call"),
            "stream": ("stream", "streaming"),
            "vision": ("image", "image_url", "vision", "multimodal", "multi-modal"),
        }
        unsupported_markers = (
            "unsupported",
            "not support",
            "not supported",
            "unknown parameter",
            "unrecognized parameter",
            "invalid parameter",
            "unexpected keyword",
            "not allowed",
        )
        has_unsupported_marker = any(marker in text for marker in unsupported_markers)
        has_capability_token = any(token in text for token in capability_tokens.get(capability, ()))
        if has_unsupported_marker and (has_capability_token or capability in text):
            return _LLMDiagnostic(
                "capability_unsupported",
                False,
                f"LLM channel does not support {capability} capability",
                "capability_unsupported",
                {"capability": capability},
            )
        return cls._classify_llm_exception(exc)


    @staticmethod
    def _is_safe_base_url(value: str) -> bool:
        """Block link-local and cloud metadata addresses to prevent SSRF.

        Allows localhost / private-LAN addresses (e.g. Ollama on 192.168.x.x)
        but blocks 169.254.x.x (AWS/Azure/GCP/Alibaba instance-metadata service)
        and other known metadata hostnames.
        """
        from src.services.system_config.validation import SystemConfigValidationMixin
        import ipaddress

        try:
            parsed = urlparse(value)
            raw_host = parsed.hostname or ""
        except ValueError:
            return False
        if not raw_host:
            return True
        host = SystemConfigValidationMixin._normalize_hostname_for_security(raw_host)
        if not host:
            return False
        # Known cloud metadata hostnames
        _BLOCKED_HOSTS = frozenset({
            "169.254.169.254",
            "metadata.google.internal",
            "100.100.100.200",
        })
        if host in _BLOCKED_HOSTS:
            return False
        if SystemConfigValidationMixin._is_noncanonical_ipv4_numeric_host(host):
            return False
        # Numeric IPs: block link-local range (169.254.0.0/16), including IPv4-mapped IPv6.
        try:
            addr = ipaddress.ip_address(host)
            candidate_addrs = [addr]
            mapped_addr = getattr(addr, "ipv4_mapped", None)
            if mapped_addr is not None:
                candidate_addrs.append(mapped_addr)
            for candidate_addr in candidate_addrs:
                if str(candidate_addr) in _BLOCKED_HOSTS or candidate_addr.is_link_local:
                    return False
        except ValueError:
            pass  # hostname, not an IP — already checked against blocklist above
        return True


    @staticmethod
    def _build_llm_models_url(base_url: str) -> str:
        """Convert a channel base URL into a `/models` endpoint."""
        from src.services.system_config.validation import SystemConfigValidationMixin
        if not SystemConfigValidationMixin._is_valid_llm_base_url(base_url):
            raise ValueError("LLM channel base URL must be a valid absolute URL")
        if not SystemConfigLLMMixin._is_safe_base_url(base_url):
            raise ValueError("LLM channel base URL points to a restricted address")

        parsed = urlparse(base_url)
        normalized = (parsed.path or "").rstrip("/")
        for suffix in ("/chat/completions", "/completions"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        if normalized.endswith("/models"):
            models_path = normalized or "/models"
        else:
            models_path = f"{normalized}/models" if normalized else "/models"
        models_url = urlunparse(parsed._replace(path=models_path, params="", query="", fragment=""))
        if not SystemConfigValidationMixin._is_valid_llm_base_url(models_url):
            raise ValueError("LLM channel models URL must be a valid absolute URL")
        if not SystemConfigLLMMixin._is_safe_base_url(models_url):
            raise ValueError("LLM channel models URL points to a restricted address")
        return models_url


    @staticmethod
    def _get_runtime_llm_temperature() -> float:
        """Return the current configured LLM temperature for ad-hoc channel tests."""
        config = Config._load_from_env()
        try:
            return float(getattr(config, "llm_temperature", 0.7))
        except (TypeError, ValueError):
            return 0.7


    @classmethod
    def _build_llm_channel_result(
        cls,
        *,
        success: bool,
        message: str,
        error: Optional[str],
        stage: Optional[str],
        error_code: Optional[str],
        retryable: Optional[bool],
        details: Optional[Dict[str, Any]] = None,
        resolved_protocol: Optional[str] = None,
        resolved_api_surface: Optional[str] = None,
        resolved_model: Optional[str] = None,
        models: Optional[List[str]] = None,
        latency_ms: Optional[int] = None,
        capability_results: Optional[Dict[str, Any]] = None,
        redaction_values: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "success": success,
            "message": cls._sanitize_llm_error_text(message, redaction_values=redaction_values),
            "error": cls._sanitize_llm_error_text(error, redaction_values=redaction_values) if error else None,
            "stage": stage,
            "error_code": error_code,
            "retryable": retryable,
            "details": cls._sanitize_llm_details(details, redaction_values=redaction_values),
            "resolved_protocol": cls._sanitize_llm_error_text(
                resolved_protocol,
                redaction_values=redaction_values,
            ) if resolved_protocol is not None else None,
            "resolved_api_surface": cls._sanitize_llm_error_text(
                resolved_api_surface,
                redaction_values=redaction_values,
            ) if resolved_api_surface is not None else None,
            "latency_ms": latency_ms,
        }
        if resolved_model is not None or models is None:
            payload["resolved_model"] = cls._sanitize_llm_error_text(
                resolved_model,
                redaction_values=redaction_values,
            ) if resolved_model is not None else resolved_model
        if models is not None:
            payload["models"] = cls._sanitize_llm_value(models, redaction_values=redaction_values)
        if capability_results is not None:
            payload["capability_results"] = cls._sanitize_llm_value(capability_results, redaction_values=redaction_values)
        return payload


    @staticmethod
    def _merge_llm_diagnostic_details(
        base_details: Optional[Dict[str, Any]],
        diagnostic: _LLMDiagnostic,
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = dict(base_details or {})
        if diagnostic.reason:
            details.setdefault("reason", diagnostic.reason)
        details.update(diagnostic.details)
        return details


    @staticmethod
    def _build_redaction_values(*values: Any) -> Set[str]:
        return build_hermes_redaction_values(*values)


    @staticmethod
    def _comma_flexible_secret_pattern(secret: str) -> Optional[re.Pattern[str]]:
        normalized = re.sub(r"(?i)^\s*authorization\s*[:=]\s*", "", str(secret or "").strip())
        normalized = re.sub(r"(?i)^\s*bearer\s+", "", normalized)
        parts = [part.strip() for part in normalized.split(",") if part.strip()]
        if len(parts) <= 1:
            return None
        return re.compile(
            r"(?i)(?:authorization\s*[:=]\s*)?(?:bearer\s+)?"
            + r"\s*,\s*".join(re.escape(part) for part in parts)
        )


    @classmethod
    def _sanitize_llm_error_text(cls, text: Any, *, redaction_values: Optional[Set[str]] = None) -> str:
        if text is None:
            return ""
        sanitized = str(text).strip()
        if not sanitized:
            return ""
        for secret in sorted((redaction_values or set()), key=len, reverse=True):
            pattern = cls._comma_flexible_secret_pattern(secret)
            if pattern is not None:
                sanitized = pattern.sub("[REDACTED]", sanitized)
        for secret in sorted((redaction_values or set()), key=len, reverse=True):
            if secret:
                sanitized = sanitized.replace(secret, "[REDACTED]")

        patterns = [
            (r"(?i)(authorization\s*[:=]\s*)(bearer\s+)?([^\s,;]+)", r"\1[REDACTED]"),
            (r"(?i)(api[_-]?key\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]"),
            (r"(?i)(cookie\s*[:=]\s*)([^\s,;]+)", r"\1[REDACTED]"),
            (r"(?i)bearer\s+[a-z0-9._\-]+", "Bearer [REDACTED]"),
            (r"(?i)sk-[a-z0-9_\-]+", "[REDACTED]"),
        ]
        for pattern, replacement in patterns:
            sanitized = re.sub(pattern, replacement, sanitized)
        sanitized = " ".join(sanitized.split())
        return sanitized[:300]


    @classmethod
    def _sanitize_llm_details(
        cls,
        details: Optional[Dict[str, Any]],
        *,
        redaction_values: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        if not details:
            return {}
        sanitized = cls._sanitize_llm_value(details, redaction_values=redaction_values)
        return sanitized if isinstance(sanitized, dict) else {}


    @classmethod
    def _sanitize_llm_value(cls, value: Any, *, redaction_values: Optional[Set[str]] = None) -> Any:
        if isinstance(value, str):
            return cls._sanitize_llm_error_text(value, redaction_values=redaction_values)
        if isinstance(value, dict):
            return {
                cls._sanitize_llm_error_text(key, redaction_values=redaction_values): cls._sanitize_llm_value(
                    item,
                    redaction_values=redaction_values,
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                cls._sanitize_llm_value(item, redaction_values=redaction_values)
                for item in value
            ]
        if isinstance(value, tuple):
            return [
                cls._sanitize_llm_value(item, redaction_values=redaction_values)
                for item in value
            ]
        return value


    @staticmethod
    def _classify_llm_http_error(status_code: int, error_text: str) -> _LLMDiagnostic:
        lowered = (error_text or "").lower()
        if SystemConfigLLMMixin._has_model_access_denied_signal(error_text or ""):
            return _LLMDiagnostic(
                "model_not_found",
                False,
                "Configured model is not available for this channel",
                "model_access_denied",
            )
        if "model" in lowered and any(token in lowered for token in ("not found", "does not exist", "unknown")):
            return _LLMDiagnostic(
                "model_not_found",
                False,
                "Configured model could not be found on this channel",
                "model_not_found",
            )
        if status_code == 402 or any(token in lowered for token in ("billing", "balance", "insufficient balance")):
            return _LLMDiagnostic(
                "quota",
                True,
                "LLM request was rejected by quota or billing limits",
                "insufficient_balance",
            )
        if any(token in lowered for token in ("quota", "insufficient_quota", "quota exceeded")):
            return _LLMDiagnostic(
                "quota",
                True,
                "LLM request was rejected by quota or rate limiting",
                "quota_exceeded",
            )
        if status_code == 429 or any(token in lowered for token in ("rate limit", "too many requests", "rpm", "tpm")):
            return _LLMDiagnostic(
                "quota",
                True,
                "LLM request was rejected by quota or rate limiting",
                "rate_limit",
            )
        if SystemConfigLLMMixin._has_transport_blocked_signal(error_text or ""):
            return _LLMDiagnostic(
                "network_error",
                True,
                "LLM request failed before a valid response was returned",
                "network_error",
            )
        if SystemConfigLLMMixin._has_request_blocked_signal(error_text or ""):
            return _LLMDiagnostic(
                "request_blocked",
                False,
                "LLM request was blocked by provider or gateway policy",
                "provider_blocked",
            )
        if status_code in {401, 403} or any(token in lowered for token in ("unauthorized", "forbidden", "invalid api key", "authentication")):
            return _LLMDiagnostic("auth", False, "LLM authentication failed", "api_key_rejected")
        if status_code == 404:
            return _LLMDiagnostic(
                "network_error",
                False,
                "LLM model discovery endpoint could not be found",
                "endpoint_not_found",
            )
        if any(token in lowered for token in ("timeout", "timed out")):
            return _LLMDiagnostic("timeout", True, "LLM request timed out", "timeout")
        return _LLMDiagnostic(
            "network_error",
            status_code >= 500,
            "LLM request failed before a valid response was returned",
            "http_error",
        )


    @staticmethod
    def _has_model_not_found_signal(text: str) -> bool:
        lowered = text.lower()

        model_candidates = [
            re.search(r"model\s+not\s+found\s*[:：]?\s*[`\"']?\s*([a-z0-9._/-]{2,})", lowered),
            re.search(r"model\s*[`\"']?\s*([a-z0-9._/-]{2,})\s*[`\"']?\s+does\s+not\s+exist", lowered),
            re.search(r"model\s+does\s+not\s+exist\s*[:：]?\s*[`\"']?\s*([a-z0-9._/-]{2,})", lowered),
            re.search(r"unknown\s+model\s*[:：]?\s*[`\"']?\s*([a-z0-9._/-]{2,})", lowered),
            re.search(r"no\s+such\s+model\s*[:：]?\s*[`\"']?\s*([a-z0-9._/-]{2,})", lowered),
        ]

        for match in model_candidates:
            if not match:
                continue
            model_id = match.group(1).strip()
            if model_id and not model_id.startswith("/") and "http" not in model_id:
                return True

        return False


    @staticmethod
    def _has_model_access_denied_signal(text: str) -> bool:
        lowered = text.lower()
        if "model" not in lowered:
            return False

        # Best-effort classifier for observed provider messages. Keep it gated by
        # an explicit "model" mention plus access/disabled/unavailable signals so
        # unrelated provider-specific failures continue to use the fallback path.
        access_denied_tokens = (
            "not authorized",
            "not allowed",
            "access denied",
            "permission denied",
            "model disabled",
            "model is disabled",
            "disabled model",
            "model has been disabled",
            "model not enabled",
            "model not available",
            "model is not available",
        )
        return any(token in lowered for token in access_denied_tokens)


    @staticmethod
    def _has_request_blocked_signal(text: str) -> bool:
        lowered = text.lower()
        if SystemConfigLLMMixin._has_transport_blocked_signal(lowered):
            return False
        blocked_tokens = (
            "your request was blocked",
            "the request was blocked",
            "request blocked by policy",
            "blocked by policy",
            "blocked due to policy",
            "moderation_blocked",
            "policy_blocked",
            "请求被拦截",
        )
        return any(token in lowered for token in blocked_tokens)


    @staticmethod
    def _has_transport_blocked_signal(text: str) -> bool:
        lowered = text.lower()
        transport_tokens = (
            "connection blocked",
            "connection request was blocked",
            "network blocked",
            "blocked by network policy",
            "blocked by firewall",
            "firewall blocked",
        )
        return any(token in lowered for token in transport_tokens)


    @staticmethod
    def _has_provider_prefix_mismatch_signal(text: str) -> bool:
        lowered = text.lower()
        mismatch_tokens = (
            "provider prefix",
            "llm provider not provided",
            "invalid provider",
            "unknown provider",
            "custom_llm_provider",
            "not a valid llm provider",
        )
        return any(token in lowered for token in mismatch_tokens)


    @staticmethod
    def _classify_llm_exception(exc: Exception) -> _LLMDiagnostic:
        exc_name = type(exc).__name__.lower()
        text = str(exc).lower()
        if isinstance(exc, TimeoutError) or "timeout" in exc_name or "timed out" in text:
            return _LLMDiagnostic("timeout", True, "LLM request timed out", "timeout")
        if any(token in text for token in ("billing", "balance", "insufficient balance")):
            return _LLMDiagnostic(
                "quota",
                True,
                "LLM request was rejected by quota or billing limits",
                "insufficient_balance",
            )
        if any(token in text for token in ("quota", "insufficient_quota", "quota exceeded")):
            return _LLMDiagnostic(
                "quota",
                True,
                "LLM request was rejected by quota or rate limiting",
                "quota_exceeded",
            )
        if "ratelimit" in exc_name or any(token in text for token in ("rate limit", "too many requests", "rpm", "tpm")):
            return _LLMDiagnostic(
                "quota",
                True,
                "LLM request was rejected by quota or rate limiting",
                "rate_limit",
            )
        if SystemConfigLLMMixin._has_provider_prefix_mismatch_signal(text):
            return _LLMDiagnostic(
                "model_not_found",
                False,
                "Configured model prefix does not match this channel",
                "provider_prefix_mismatch",
            )
        if SystemConfigLLMMixin._has_model_access_denied_signal(str(exc)):
            return _LLMDiagnostic(
                "model_not_found",
                False,
                "Configured model is not available for this channel",
                "model_access_denied",
            )
        if SystemConfigLLMMixin._has_request_blocked_signal(str(exc)):
            return _LLMDiagnostic(
                "request_blocked",
                False,
                "LLM request was blocked by provider or gateway policy",
                "provider_blocked",
            )
        if any(token in exc_name for token in ("auth", "permission")) or any(token in text for token in ("unauthorized", "forbidden", "invalid api key", "authentication")):
            return _LLMDiagnostic("auth", False, "LLM authentication failed", "api_key_rejected")
        if ("notfound" in exc_name or "model" in text) and (
            "not found" in text or "does not exist" in text or "unknown model" in text
        ) and SystemConfigLLMMixin._has_model_not_found_signal(text):
            return _LLMDiagnostic(
                "model_not_found",
                False,
                "Configured model could not be found on this channel",
                "model_not_found",
            )
        if "dns" in text or "name resolution" in text or "temporary failure in name resolution" in text:
            return _LLMDiagnostic("network_error", True, "LLM request failed before a valid response was returned", "dns_error")
        if "refused" in text or "connection refused" in text:
            return _LLMDiagnostic("network_error", True, "LLM request failed before a valid response was returned", "connection_refused")
        if "ssl" in text or "tls" in text or "certificate" in text:
            return _LLMDiagnostic("network_error", True, "LLM request failed before a valid response was returned", "tls_error")
        if any(token in exc_name for token in ("connection", "network")) or any(
            token in text for token in ("connection", "network", "firewall")
        ):
            return _LLMDiagnostic("network_error", True, "LLM request failed before a valid response was returned", "network_error")
        return _LLMDiagnostic("network_error", False, "LLM channel test failed", "unknown_error")


    @staticmethod
    def _extract_llm_completion_content(response: Any) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
        def _field(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _text_from_blocks(blocks: Any) -> str:
            if not isinstance(blocks, list):
                return ""
            text_parts: List[str] = []
            for block in blocks:
                if isinstance(block, str):
                    text_parts.append(block)
                    continue
                block_type = str(_field(block, "type") or "").strip().lower()
                if block_type and block_type not in {"text", "output_text"}:
                    continue
                text = _field(block, "text")
                if text is None:
                    text = _field(block, "content")
                if isinstance(text, str) and text:
                    text_parts.append(text)
            return strip_leading_think_wrapper("".join(text_parts))

        if response is None:
            return "", "empty_response", "Completion returned no response object", "null_response"

        choices = _field(response, "choices")
        if not choices:
            return "", "format_error", "Completion response did not include choices", "malformed_choices"

        choice = choices[0]
        content_blocks = _field(choice, "content_blocks")
        message = _field(choice, "message")
        if content_blocks is None:
            if message is not None:
                content_blocks = _field(message, "content_blocks")
        if content_blocks is not None:
            content = _text_from_blocks(content_blocks)
            if content:
                return content, None, None, None

        if message is None:
            return "", "format_error", "Completion response did not include a message object", "malformed_choices"
        if isinstance(message, dict):
            has_content = "content" in message
        else:
            has_content = hasattr(message, "content")
        if not has_content:
            return "", "format_error", "Completion message did not include a content field", "malformed_choices"
        raw_content = _field(message, "content")
        if raw_content is None:
            return "", "empty_response", "Completion returned null message content", "null_content"
        content = (
            _text_from_blocks(raw_content)
            if isinstance(raw_content, list)
            else strip_leading_think_wrapper(str(raw_content))
        )
        if not content:
            return "", "empty_response", "Completion returned an empty message content", "empty_content"
        return content, None, None, None


    @staticmethod
    def _extract_llm_discovery_error(response: requests.Response) -> str:
        """Extract a concise error message from a failed model discovery response."""
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            error_payload = payload.get("error")
            if isinstance(error_payload, dict):
                message = str(
                    error_payload.get("message")
                    or error_payload.get("code")
                    or ""
                ).strip()
                if message:
                    return message

            message = str(payload.get("message") or payload.get("detail") or "").strip()
            if message:
                return message

        text = response.text.strip()
        if text:
            return text[:200]
        return f"HTTP {response.status_code}"


    @staticmethod
    def _extract_discovered_llm_models(payload: Any) -> List[str]:
        """Normalize common `/models` response shapes into a unique model ID list."""
        raw_models: List[Any] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                raw_models = payload["data"]
            elif isinstance(payload.get("models"), list):
                raw_models = payload["models"]
        elif isinstance(payload, list):
            raw_models = payload

        models: List[str] = []
        seen: Set[str] = set()
        for entry in raw_models:
            if isinstance(entry, str):
                model_id = entry.strip()
            elif isinstance(entry, dict):
                model_id = str(
                    entry.get("id") or entry.get("model") or entry.get("name") or ""
                ).strip()
            else:
                model_id = ""

            if not model_id or model_id in seen:
                continue

            seen.add(model_id)
            models.append(model_id)

        return models


    @staticmethod
    def _validate_llm_channel_map(effective_map: Dict[str, str], updated_keys: Set[str]) -> List[Dict[str, Any]]:
        """Validate channel-style LLM configuration stored in `.env`."""
        issues: List[Dict[str, Any]] = []
        if SystemConfigLLMMixin._uses_litellm_yaml(effective_map):
            return issues

        raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
        if not raw_channels:
            return issues

        normalized_names: List[str] = []
        seen_names: Set[str] = set()
        for raw_name in raw_channels.split(","):
            name = raw_name.strip()
            if not name:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                issues.append(
                    {
                        "key": "LLM_CHANNELS",
                        "code": "invalid_channel_name",
                        "message": f"LLM channel name '{name}' may only contain letters, numbers, and underscores",
                        "severity": "error",
                        "expected": "letters/numbers/underscores",
                        "actual": name,
                    }
                )
                continue

            normalized_upper = name.upper()
            if normalized_upper in seen_names:
                issues.append(
                    {
                        "key": "LLM_CHANNELS",
                        "code": "duplicate_channel_name",
                        "message": f"LLM channel '{name}' is declared more than once",
                        "severity": "error",
                        "expected": "unique channel names",
                        "actual": raw_channels,
                    }
                )
                continue

            seen_names.add(normalized_upper)
            normalized_names.append(name)

        validated_channels: List[Dict[str, Any]] = []
        for name in normalized_names:
            prefix = f"LLM_{name.upper()}"
            protocol_value = (effective_map.get(f"{prefix}_PROTOCOL") or "").strip()
            api_surface_value = (effective_map.get(f"{prefix}_API_SURFACE") or "").strip()
            if name.lower() == "anspire" and not protocol_value:
                protocol_value = "openai"
            base_url_value = (effective_map.get(f"{prefix}_BASE_URL") or "").strip()
            if name.lower() == "anspire" and not base_url_value:
                base_url_value = (
                    effective_map.get("ANSPIRE_LLM_BASE_URL")
                    or ANSPIRE_LLM_BASE_URL_DEFAULT
                ).strip()
            api_key_value = (
                (effective_map.get(f"{prefix}_API_KEYS") or "").strip()
                or (effective_map.get(f"{prefix}_API_KEY") or "").strip()
            )
            if name.lower() == "anspire" and not api_key_value:
                api_key_value = (effective_map.get("ANSPIRE_API_KEYS") or "").strip()
            models_value = [
                model.strip()
                for model in (effective_map.get(f"{prefix}_MODELS") or "").split(",")
                if model.strip()
            ]
            if name.lower() == "anspire" and not models_value:
                models_value = [
                    (
                        effective_map.get("ANSPIRE_LLM_MODEL")
                        or ANSPIRE_LLM_MODEL_DEFAULT
                    ).strip()
                ]
            enabled_raw = effective_map.get(f"{prefix}_ENABLED")
            if name.lower() == "anspire" and not (enabled_raw or "").strip():
                enabled_raw = effective_map.get("ANSPIRE_LLM_ENABLED")
            enabled = parse_env_bool(enabled_raw, default=True)
            if not enabled:
                continue
            if is_reserved_hermes_name(name):
                if not is_supported_llm_channel_api_surface_value(api_surface_value):
                    issues.append(
                        {
                            "key": f"{prefix}_API_SURFACE",
                            "code": "invalid_api_surface",
                            "message": (
                                f"Unsupported LLM API surface '{api_surface_value}'. "
                                f"Supported: {', '.join(SUPPORTED_LLM_CHANNEL_API_SURFACES)}"
                            ),
                            "severity": "error",
                            "expected": ",".join(SUPPORTED_LLM_CHANNEL_API_SURFACES),
                            "actual": api_surface_value,
                        }
                    )
                    continue
                if normalize_llm_channel_api_surface(api_surface_value) == "responses":
                    issues.append(
                        {
                            "key": f"{prefix}_API_SURFACE",
                            "code": "hermes_responses_unsupported",
                            "message": "The reserved Hermes channel does not support the Responses API surface",
                            "severity": "error",
                            "expected": "chat_completions",
                            "actual": "responses",
                        }
                    )
                    continue
                result = parse_hermes_channel(
                    enabled=enabled,
                    protocol=protocol_value or HERMES_DEFAULT_PROTOCOL,
                    base_url=base_url_value or HERMES_DEFAULT_BASE_URL,
                    api_key=(effective_map.get(f"{prefix}_API_KEY") or "").strip(),
                    api_keys_raw=(effective_map.get(f"{prefix}_API_KEYS") or "").strip(),
                    extra_headers_raw=(effective_map.get(f"{prefix}_EXTRA_HEADERS") or "").strip(),
                    models=models_value or [HERMES_DEFAULT_MODEL],
                )
                for issue in result.issues:
                    issues.append(
                        {
                            "key": issue.field,
                            "code": issue.code,
                            "message": issue.message,
                            "severity": issue.severity,
                            "expected": "valid reserved Hermes channel",
                            "actual": "",
                        }
                    )
                if result.channel is not None and not result.issues:
                    validated_channels.append(result.channel)
                continue
            channel_issues = SystemConfigLLMMixin._validate_llm_channel_definition(
                channel_name=name,
                protocol_value=protocol_value,
                api_surface_value=api_surface_value,
                base_url_value=base_url_value,
                api_key_value=api_key_value,
                model_values=models_value,
                enabled=enabled,
                field_prefix=prefix,
                require_complete=enabled,
            )
            issues.extend(channel_issues)
            if not any(issue.get("severity") == "error" for issue in channel_issues):
                resolved_protocol = resolve_llm_channel_protocol(
                    protocol_value,
                    base_url=base_url_value,
                    models=models_value,
                    channel_name=name,
                )
                validated_channels.append(
                    {
                        "name": name.lower(),
                        "protocol": resolved_protocol,
                        "api_surface": normalize_llm_channel_api_surface(api_surface_value),
                        "base_url": base_url_value,
                        "models": models_value,
                        "enabled": True,
                    }
                )

        for model, surfaces in find_llm_channel_surface_conflicts(validated_channels).items():
            issues.append(
                {
                    "key": "LLM_CHANNELS",
                    "code": "mixed_api_surfaces_for_route",
                    "message": (
                        f"LLM route alias '{model}' is declared with multiple API surfaces: "
                        f"{', '.join(surfaces)}"
                    ),
                    "severity": "error",
                    "expected": "one API surface per normalized route alias",
                    "actual": ",".join(surfaces),
                }
            )

        return issues


    @staticmethod
    def _collect_llm_channel_models_from_map(effective_map: Dict[str, str]) -> List[str]:
        """Collect normalized model names from channel-style env values."""
        raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
        if not raw_channels:
            return []

        models: List[str] = []
        seen: Set[str] = set()
        for raw_name in raw_channels.split(","):
            name = raw_name.strip()
            if not name:
                continue

            prefix = f"LLM_{name.upper()}"
            enabled_raw = effective_map.get(f"{prefix}_ENABLED")
            if name.lower() == "anspire" and not (enabled_raw or "").strip():
                enabled_raw = effective_map.get("ANSPIRE_LLM_ENABLED")
            enabled = parse_env_bool(enabled_raw, default=True)
            if not enabled:
                continue

            base_url_value = (effective_map.get(f"{prefix}_BASE_URL") or "").strip()
            if name.lower() == "anspire" and not base_url_value:
                base_url_value = (
                    effective_map.get("ANSPIRE_LLM_BASE_URL")
                    or ANSPIRE_LLM_BASE_URL_DEFAULT
                ).strip()
            protocol_value = (effective_map.get(f"{prefix}_PROTOCOL") or "").strip()
            if name.lower() == "anspire" and not protocol_value:
                protocol_value = "openai"
            api_surface_value = (effective_map.get(f"{prefix}_API_SURFACE") or "").strip()
            raw_models = [
                model.strip()
                for model in (effective_map.get(f"{prefix}_MODELS") or "").split(",")
                if model.strip()
            ]
            if name.lower() == "anspire" and not raw_models:
                raw_models = [
                    (
                        effective_map.get("ANSPIRE_LLM_MODEL")
                        or ANSPIRE_LLM_MODEL_DEFAULT
                    ).strip()
                ]
            if is_reserved_hermes_name(name):
                if (
                    not is_supported_llm_channel_api_surface_value(api_surface_value)
                    or normalize_llm_channel_api_surface(api_surface_value) == "responses"
                ):
                    continue
                result = parse_hermes_channel(
                    enabled=True,
                    protocol=protocol_value or HERMES_DEFAULT_PROTOCOL,
                    base_url=base_url_value or HERMES_DEFAULT_BASE_URL,
                    api_key=(effective_map.get(f"{prefix}_API_KEY") or "").strip(),
                    api_keys_raw=(effective_map.get(f"{prefix}_API_KEYS") or "").strip(),
                    extra_headers_raw=(effective_map.get(f"{prefix}_EXTRA_HEADERS") or "").strip(),
                    models=raw_models or [HERMES_DEFAULT_MODEL],
                )
                channel = result.channel or {}
                for model in channel.get("models") or []:
                    if model and model not in seen:
                        seen.add(model)
                        models.append(model)
                continue
            resolved_protocol = resolve_llm_channel_protocol(protocol_value, base_url=base_url_value, models=raw_models, channel_name=name)
            if (
                not is_supported_llm_channel_api_surface_value(api_surface_value)
                or find_incompatible_llm_channel_models(
                    raw_models,
                    resolved_protocol,
                    api_surface_value,
                    base_url_value,
                )
            ):
                continue
            for model in raw_models:
                normalized_model = normalize_llm_channel_model(model, resolved_protocol, base_url_value)
                if not normalized_model or normalized_model in seen:
                    continue
                seen.add(normalized_model)
                models.append(normalized_model)

        return models


    @staticmethod
    def _uses_litellm_yaml(effective_map: Dict[str, str]) -> bool:
        """Return True when a valid LiteLLM YAML config takes precedence over channels."""
        config_path = (effective_map.get("LITELLM_CONFIG") or "").strip()
        if not config_path:
            return False
        return bool(Config._parse_litellm_yaml(config_path))


    @staticmethod
    def _collect_yaml_models_from_map(effective_map: Dict[str, str]) -> List[str]:
        """Collect declared router model names from LiteLLM YAML config."""
        config_path = (effective_map.get("LITELLM_CONFIG") or "").strip()
        if not config_path:
            return []
        return get_configured_llm_models(Config._parse_litellm_yaml(config_path))


    @staticmethod
    def _has_legacy_key_for_provider(provider: str, effective_map: Dict[str, str]) -> bool:
        """Return True when legacy env config can still back the provider."""
        from src.services.system_config.setup import SystemConfigSetupMixin
        normalized_provider = canonicalize_llm_channel_protocol(provider)
        if normalized_provider in {"gemini", "vertex_ai"}:
            return bool(
                (effective_map.get("GEMINI_API_KEYS") or "").strip()
                or (effective_map.get("GEMINI_API_KEY") or "").strip()
            )
        if normalized_provider == "anthropic":
            return bool(
                (effective_map.get("ANTHROPIC_API_KEYS") or "").strip()
                or (effective_map.get("ANTHROPIC_API_KEY") or "").strip()
            )
        if normalized_provider == "deepseek":
            return bool(
                (effective_map.get("DEEPSEEK_API_KEYS") or "").strip()
                or (effective_map.get("DEEPSEEK_API_KEY") or "").strip()
            )
        if normalized_provider == "openai":
            return bool(
                (effective_map.get("OPENAI_API_KEYS") or "").strip()
                or (effective_map.get("AIHUBMIX_KEY") or "").strip()
                or (effective_map.get("OPENAI_API_KEY") or "").strip()
                or (
                    SystemConfigSetupMixin._anspire_legacy_llm_enabled(effective_map)
                    and (effective_map.get("ANSPIRE_API_KEYS") or "").strip()
                )
            )
        return False


    @staticmethod
    def _has_runtime_source_for_model(model: str, effective_map: Dict[str, str]) -> bool:
        """Whether the selected model still has a backing runtime source."""
        if not model or _uses_direct_env_provider(model):
            return True
        provider = _get_litellm_provider(model)
        return SystemConfigLLMMixin._has_legacy_key_for_provider(provider, effective_map)


    @staticmethod
    def _validate_llm_runtime_selection(effective_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate selected primary/fallback/vision models against configured channels."""
        from src.services.system_config.hermes import SystemConfigHermesMixin
        issues: List[Dict[str, Any]] = []

        available_models = (
            SystemConfigLLMMixin._collect_yaml_models_from_map(effective_map)
            or SystemConfigLLMMixin._collect_llm_channel_models_from_map(effective_map)
        )
        available_model_set = set(available_models)
        hermes_route_set = set(SystemConfigHermesMixin._collect_hermes_channel_models_from_map(effective_map))
        mixed_hermes_routes = SystemConfigHermesMixin._collect_mixed_hermes_routes_from_map(effective_map)
        if not available_model_set:
            raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
            if not raw_channels:
                return issues

            configured_agent_model_raw = (effective_map.get("AGENT_LITELLM_MODEL") or "").strip()
            configured_agent_model = normalize_agent_litellm_model(
                configured_agent_model_raw,
                configured_models=available_model_set,
            )
            primary_model = (effective_map.get("LITELLM_MODEL") or "").strip()
            if primary_model and not SystemConfigLLMMixin._has_runtime_source_for_model(primary_model, effective_map):
                issues.append(
                    {
                        "key": "LITELLM_MODEL",
                        "code": "missing_runtime_source",
                        "message": (
                            "A primary model is selected, but no usable runtime source was found. "
                            "Enable at least one channel with available models, or provide the "
                            "matching provider API key so the model can be resolved."
                        ),
                        "severity": "error",
                        "expected": "enabled channel model or matching legacy API key",
                        "actual": primary_model,
                    }
                )

            if (
                configured_agent_model_raw
                and configured_agent_model
                and not SystemConfigLLMMixin._has_runtime_source_for_model(
                    configured_agent_model,
                    effective_map,
                )
            ):
                issues.append(
                    {
                        "key": "AGENT_LITELLM_MODEL",
                        "code": "missing_runtime_source",
                        "message": (
                            "An Agent primary model is selected, but no usable runtime source was found. "
                            "Enable at least one channel with available models, or provide the "
                            "matching provider API key so the model can be resolved."
                        ),
                        "severity": "error",
                        "expected": "enabled channel model or matching legacy API key",
                        "actual": configured_agent_model,
                    }
                )
            elif (
                configured_agent_model_raw
                and configured_agent_model
                and SystemConfigHermesMixin._matches_route_set(configured_agent_model, hermes_route_set)
                and not SystemConfigHermesMixin._matches_route_set(configured_agent_model, mixed_hermes_routes)
            ):
                issues.append(
                    {
                        "key": "AGENT_LITELLM_MODEL",
                        "code": "explicit_agent_model_no_safe_deployment",
                        "message": (
                            "Hermes-only routes are not valid Agent models in Phase 3. "
                            "Choose a route with at least one non-Hermes deployment."
                        ),
                        "severity": "error",
                        "expected": "Agent-safe route with non-Hermes deployment",
                        "actual": configured_agent_model,
                    }
                )

            fallback_models = [
                model.strip()
                for model in (effective_map.get("LITELLM_FALLBACK_MODELS") or "").split(",")
                if model.strip()
            ]
            invalid_fallbacks = [
                model for model in fallback_models
                if not SystemConfigLLMMixin._has_runtime_source_for_model(model, effective_map)
            ]
            if invalid_fallbacks:
                issues.append(
                    {
                        "key": "LITELLM_FALLBACK_MODELS",
                        "code": "missing_runtime_source",
                        "message": (
                            "Some fallback models do not have an enabled channel "
                            "or matching API key available"
                        ),
                        "severity": "error",
                        "expected": "enabled channel models or matching legacy API keys",
                        "actual": ", ".join(invalid_fallbacks[:3]),
                    }
                )

            vision_model = (effective_map.get("VISION_MODEL") or "").strip()
            if vision_model and SystemConfigHermesMixin._matches_route_set(vision_model, hermes_route_set):
                issues.append(
                    {
                        "key": "VISION_MODEL",
                        "code": "hermes_vision_unsupported",
                        "message": (
                            "Hermes routes are not valid Vision models in Phase 3. "
                            "Choose a pure non-Hermes Vision-capable route."
                        ),
                        "severity": "error",
                        "expected": "pure non-Hermes Vision route",
                        "actual": vision_model,
                    }
                )
            elif vision_model and not SystemConfigLLMMixin._has_runtime_source_for_model(vision_model, effective_map):
                issues.append(
                    {
                        "key": "VISION_MODEL",
                        "code": "missing_runtime_source",
                        "message": (
                            "A Vision model is selected, but there is no enabled channel "
                            "or matching API key available for it"
                        ),
                        "severity": "warning",
                        "expected": "enabled channel model or matching legacy API key",
                        "actual": vision_model,
                    }
                )

            return issues

        primary_model = (effective_map.get("LITELLM_MODEL") or "").strip()
        if SystemConfigHermesMixin._matches_route_set(primary_model, mixed_hermes_routes):
            issues.append(
                {
                    "key": "LITELLM_MODEL",
                    "code": "mixed_hermes_route_unsupported",
                    "message": (
                        "Mixed Hermes/non-Hermes generation routes are not supported in Phase 3. "
                        "Choose a pure Hermes or pure non-Hermes route."
                    ),
                    "severity": "error",
                    "expected": "pure generation route",
                    "actual": primary_model,
                }
            )
        if (
            primary_model
            and not SystemConfigHermesMixin._matches_exact_route(primary_model, available_model_set)
            and not _uses_direct_env_provider(primary_model)
        ):
            issues.append(
                {
                    "key": "LITELLM_MODEL",
                    "code": "unknown_model",
                    "message": (
                        "The selected primary model is not declared by the current enabled channels "
                        "or advanced model routing config. "
                        f"Available models: {', '.join(available_models[:6])}"
                    ),
                    "severity": "error",
                    "expected": "one configured channel model",
                    "actual": primary_model,
                }
            )

        configured_agent_model_raw = (effective_map.get("AGENT_LITELLM_MODEL") or "").strip()
        configured_agent_model = normalize_agent_litellm_model(
            configured_agent_model_raw,
            configured_models=available_model_set,
        )
        if (
            configured_agent_model_raw
            and configured_agent_model
            and not SystemConfigHermesMixin._matches_exact_route(configured_agent_model, available_model_set)
            and not _uses_direct_env_provider(configured_agent_model)
        ):
            issues.append(
                {
                    "key": "AGENT_LITELLM_MODEL",
                    "code": "unknown_model",
                    "message": (
                        "The selected Agent primary model is not declared by the current enabled channels "
                        "or advanced model routing config. "
                        f"Available models: {', '.join(available_models[:6])}"
                    ),
                    "severity": "error",
                    "expected": "one configured channel model",
                    "actual": configured_agent_model,
                }
            )
        elif (
                configured_agent_model_raw
                and configured_agent_model
                and SystemConfigHermesMixin._matches_route_set(configured_agent_model, hermes_route_set)
                and not SystemConfigHermesMixin._matches_route_set(configured_agent_model, mixed_hermes_routes)
            ):
            issues.append(
                {
                    "key": "AGENT_LITELLM_MODEL",
                    "code": "explicit_agent_model_no_safe_deployment",
                    "message": (
                        "Hermes-only routes are not valid Agent models in Phase 3. "
                        "Choose a route with at least one non-Hermes deployment."
                    ),
                    "severity": "error",
                    "expected": "Agent-safe route with non-Hermes deployment",
                    "actual": configured_agent_model,
                }
            )

        fallback_models = [
            model.strip()
            for model in (effective_map.get("LITELLM_FALLBACK_MODELS") or "").split(",")
            if model.strip()
        ]
        mixed_fallbacks = [
            model for model in fallback_models
            if SystemConfigHermesMixin._matches_route_set(model, mixed_hermes_routes)
        ]
        if mixed_fallbacks:
            issues.append(
                {
                    "key": "LITELLM_FALLBACK_MODELS",
                    "code": "mixed_hermes_route_unsupported",
                    "message": (
                        "Mixed Hermes/non-Hermes generation routes are not supported as fallback models in Phase 3."
                    ),
                    "severity": "error",
                    "expected": "pure generation fallback routes",
                    "actual": ", ".join(mixed_fallbacks[:3]),
                }
            )
        invalid_fallbacks = [
            model for model in fallback_models
            if not SystemConfigHermesMixin._matches_exact_route(model, available_model_set)
            and not _uses_direct_env_provider(model)
        ]
        if invalid_fallbacks:
            issues.append(
                {
                    "key": "LITELLM_FALLBACK_MODELS",
                    "code": "unknown_model",
                    "message": (
                        "Fallback models include entries that are not declared by the current enabled channels "
                        "or advanced model routing config"
                    ),
                    "severity": "error",
                    "expected": ",".join(available_models[:6]),
                    "actual": ", ".join(invalid_fallbacks[:3]),
                }
            )

        vision_model = (effective_map.get("VISION_MODEL") or "").strip()
        if vision_model and SystemConfigHermesMixin._matches_route_set(vision_model, hermes_route_set):
            issues.append(
                {
                    "key": "VISION_MODEL",
                    "code": "hermes_vision_unsupported",
                    "message": (
                        "Hermes routes are not valid Vision models in Phase 3. "
                        "Choose a pure non-Hermes Vision-capable route."
                    ),
                    "severity": "error",
                    "expected": "pure non-Hermes Vision route",
                    "actual": vision_model,
                }
            )
        elif (
            vision_model
            and not SystemConfigHermesMixin._matches_exact_route(vision_model, available_model_set)
            and not _uses_direct_env_provider(vision_model)
        ):
            issues.append(
                {
                    "key": "VISION_MODEL",
                    "code": "unknown_model",
                    "message": (
                        "The selected Vision model is not declared by the current enabled channels "
                        "or advanced model routing config"
                    ),
                    "severity": "warning",
                    "expected": ",".join(available_models[:6]),
                    "actual": vision_model,
                }
            )

        return issues


    @staticmethod
    def _validate_llm_channel_definition(
        *,
        channel_name: str,
        protocol_value: str,
        api_surface_value: str,
        base_url_value: str,
        api_key_value: str,
        model_values: Sequence[str],
        enabled: bool,
        field_prefix: str,
        require_complete: bool,
    ) -> List[Dict[str, Any]]:
        """Validate one normalized LLM channel definition."""
        if not require_complete:
            return []

        issues, resolved_protocol = SystemConfigLLMMixin._validate_llm_channel_connection(
            channel_name=channel_name,
            protocol_value=protocol_value,
            base_url_value=base_url_value,
            api_key_value=api_key_value,
            model_values=model_values,
            field_prefix=field_prefix,
            require_base_url=False,
        )
        models_key = f"{field_prefix}_MODELS" if field_prefix != "test_channel" else "models"
        api_surface_key = (
            f"{field_prefix}_API_SURFACE"
            if field_prefix != "test_channel"
            else "api_surface"
        )
        canonical_api_surface = canonicalize_llm_channel_api_surface(api_surface_value)
        resolved_api_surface = normalize_llm_channel_api_surface(api_surface_value)
        if (
            canonical_api_surface
            and canonical_api_surface not in SUPPORTED_LLM_CHANNEL_API_SURFACES
        ):
            issues.append(
                {
                    "key": api_surface_key,
                    "code": "invalid_api_surface",
                    "message": (
                        f"Unsupported LLM API surface '{api_surface_value}'. "
                        f"Supported: {', '.join(SUPPORTED_LLM_CHANNEL_API_SURFACES)}"
                    ),
                    "severity": "error",
                    "expected": ",".join(SUPPORTED_LLM_CHANNEL_API_SURFACES),
                    "actual": api_surface_value,
                }
            )
        elif resolved_api_surface == "responses" and resolved_protocol != "openai":
            issues.append(
                {
                    "key": api_surface_key,
                    "code": "responses_requires_openai_protocol",
                    "message": "Responses API surface currently requires the openai protocol",
                    "severity": "error",
                    "expected": "openai",
                    "actual": resolved_protocol or protocol_value,
                }
            )
        elif resolved_api_surface == "responses" and is_reserved_hermes_name(channel_name):
            issues.append(
                {
                    "key": api_surface_key,
                    "code": "hermes_responses_unsupported",
                    "message": "The reserved Hermes channel does not support the Responses API surface",
                    "severity": "error",
                    "expected": "chat_completions",
                    "actual": resolved_api_surface,
                }
            )
        elif resolved_api_surface == "responses":
            incompatible_models = find_incompatible_llm_channel_models(
                list(model_values),
                resolved_protocol,
                resolved_api_surface,
                base_url_value,
            )
            if incompatible_models:
                issues.append(
                    {
                        "key": models_key,
                        "code": "responses_requires_openai_model_provider",
                        "message": (
                            "Responses API surface requires every model to use the OpenAI "
                            f"provider route; incompatible: {', '.join(incompatible_models[:3])}"
                        ),
                        "severity": "error",
                        "expected": "openai/<model> or an unprefixed OpenAI-compatible model ID",
                        "actual": ", ".join(incompatible_models[:3]),
                    }
                )

        if not model_values:
            issues.append(
                {
                    "key": models_key,
                    "code": "missing_models",
                    "message": f"LLM channel '{channel_name}' requires at least one model",
                    "severity": "error",
                    "expected": "comma-separated model list",
                    "actual": "",
                }
            )
        elif not resolved_protocol:
            unresolved = [model for model in model_values if "/" not in model]
            if unresolved:
                issues.append(
                    {
                        "key": models_key,
                        "code": "missing_protocol",
                        "message": (
                            f"LLM channel '{channel_name}' uses bare model names. "
                            "Set PROTOCOL or add provider/model prefixes."
                        ),
                        "severity": "error",
                        "expected": "protocol or provider/model",
                        "actual": ", ".join(unresolved[:3]),
                    }
                )

        return issues


    @staticmethod
    def _validate_llm_channel_connection(
        *,
        channel_name: str,
        protocol_value: str,
        base_url_value: str,
        api_key_value: str,
        model_values: Sequence[str] = (),
        field_prefix: str,
        require_base_url: bool,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Validate connection-level fields shared by test and discovery flows."""
        from src.services.system_config.validation import SystemConfigValidationMixin
        issues: List[Dict[str, Any]] = []
        protocol_key = f"{field_prefix}_PROTOCOL" if field_prefix != "test_channel" else "protocol"
        base_url_key = f"{field_prefix}_BASE_URL" if field_prefix != "test_channel" else "base_url"
        api_key_key = f"{field_prefix}_API_KEY" if field_prefix != "test_channel" else "api_key"

        normalized_protocol = canonicalize_llm_channel_protocol(protocol_value)
        if normalized_protocol and normalized_protocol not in SUPPORTED_LLM_CHANNEL_PROTOCOLS:
            issues.append(
                {
                    "key": protocol_key,
                    "code": "invalid_protocol",
                    "message": (
                        f"Unsupported LLM channel protocol '{protocol_value}'. "
                        f"Supported: {', '.join(SUPPORTED_LLM_CHANNEL_PROTOCOLS)}"
                    ),
                    "severity": "error",
                    "expected": ",".join(SUPPORTED_LLM_CHANNEL_PROTOCOLS),
                    "actual": protocol_value,
                }
            )

        if require_base_url and not base_url_value.strip():
            issues.append(
                {
                    "key": base_url_key,
                    "code": "missing_base_url",
                    "message": f"LLM channel '{channel_name}' requires a base URL to discover models",
                    "severity": "error",
                    "expected": "http(s)://host/v1",
                    "actual": "",
                }
            )
        elif base_url_value and not SystemConfigValidationMixin._is_valid_llm_base_url(base_url_value):
            issues.append(
                {
                    "key": base_url_key,
                    "code": "invalid_url",
                    "message": "LLM channel base URL must be a valid absolute URL",
                    "severity": "error",
                    "expected": "http(s)://host",
                    "actual": base_url_value,
                }
            )
        elif base_url_value and not SystemConfigLLMMixin._is_safe_base_url(base_url_value):
            issues.append(
                {
                    "key": base_url_key,
                    "code": "ssrf_blocked",
                    "message": "LLM channel base URL points to a restricted address (cloud metadata services are not allowed)",
                    "severity": "error",
                    "expected": "publicly reachable or local LLM endpoint",
                    "actual": base_url_value,
                }
            )

        resolved_protocol = resolve_llm_channel_protocol(
            protocol_value,
            base_url=base_url_value,
            models=list(model_values) if model_values else None,
            channel_name=channel_name,
        )
        # Validate parsed key segments so that inputs like "," or " , " are
        # treated as empty (they produce zero usable keys after split+strip).
        _parsed_api_keys = [seg.strip() for seg in api_key_value.split(",") if seg.strip()]
        if not _parsed_api_keys and not channel_allows_empty_api_key(resolved_protocol, base_url_value):
            issues.append(
                {
                    "key": api_key_key,
                    "code": "missing_api_key",
                    "message": f"LLM channel '{channel_name}' requires an API key",
                    "severity": "error",
                    "expected": "non-empty API key",
                    "actual": api_key_value,
                }
            )
        return issues, resolved_protocol
