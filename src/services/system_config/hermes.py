# -*- coding: utf-8 -*-
"""Hermes 密钥解析与路由模型收集（自 system_config_service 拆出，v1.1.1 重构）。

公开入口 ``src.services.system_config_service.SystemConfigService`` 通过多继承
聚合各 Mixin，方法经 MRO 解析，行为与拆分前完全一致。
"""

from __future__ import annotations

from src.config import (
    ANSPIRE_LLM_BASE_URL_DEFAULT,
    ANSPIRE_LLM_MODEL_DEFAULT,
    find_incompatible_llm_channel_models,
    is_supported_llm_channel_api_surface_value,
    normalize_llm_channel_api_surface,
    normalize_llm_channel_model,
    parse_env_bool,
    resolve_llm_channel_protocol,
)
from src.llm.hermes import (
    HERMES_DEFAULT_BASE_URL,
    HERMES_DEFAULT_MODEL,
    HERMES_DEFAULT_PROTOCOL,
    canonicalize_hermes_base_url,
    is_masked_secret_placeholder,
    is_reserved_hermes_name,
    parse_hermes_channel,
    route_identity_candidates,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import os


class SystemConfigHermesMixin:
    """Mixin：见模块 docstring。"""

    def _resolve_hermes_saved_secret(
        self,
        *,
        channel_name: str,
        protocol: str,
        base_url: str,
        submitted_api_key: str,
        use_saved_secret: bool,
        stage: str,
    ) -> Tuple[Optional[str], Dict[str, Any], Set[str]]:
        """Resolve a saved Hermes key only when the submitted endpoint is unchanged."""

        redaction_values = self._build_redaction_values(submitted_api_key)
        if not use_saved_secret:
            return submitted_api_key, {}, redaction_values

        if not is_reserved_hermes_name(channel_name):
            return None, self._build_llm_channel_result(
                success=False,
                message="Saved secret scope mismatch",
                error="Saved Hermes secret can only be used with the reserved hermes channel",
                stage=stage,
                error_code="saved_secret_scope_mismatch",
                retryable=False,
                details={"reason": "channel_identity_mismatch"},
                resolved_protocol=None,
                models=[] if stage == "model_discovery" else None,
                latency_ms=None,
                redaction_values=redaction_values,
            ), redaction_values

        saved_map = self._manager.read_config_map()
        saved_key = (saved_map.get("LLM_HERMES_API_KEY") or "").strip()
        if not saved_key or is_masked_secret_placeholder(saved_key):
            error_code = (
                "runtime_secret_not_reusable"
                if is_masked_secret_placeholder(saved_key) or (os.environ.get("LLM_HERMES_API_KEY") or "").strip()
                else "missing_saved_secret"
            )
            return None, self._build_llm_channel_result(
                success=False,
                message=(
                    "Runtime Hermes secret is not reusable"
                    if error_code == "runtime_secret_not_reusable"
                    else "Missing saved Hermes secret"
                ),
                error=(
                    "Runtime-injected LLM_HERMES_API_KEY cannot be reused from the settings test flow"
                    if error_code == "runtime_secret_not_reusable"
                    else "No saved LLM_HERMES_API_KEY is available for this endpoint"
                ),
                stage=stage,
                error_code=error_code,
                retryable=False,
                details={"reason": error_code},
                resolved_protocol=None,
                models=[] if stage == "model_discovery" else None,
                latency_ms=None,
                redaction_values=redaction_values,
            ), redaction_values

        redaction_values.update(self._build_redaction_values(saved_key))
        saved_protocol = (saved_map.get("LLM_HERMES_PROTOCOL") or "openai").strip()
        saved_base_url = (saved_map.get("LLM_HERMES_BASE_URL") or "").strip()
        try:
            submitted_protocol = (protocol or "openai").strip().lower() or "openai"
            saved_protocol_canonical = (saved_protocol or "openai").strip().lower() or "openai"
            submitted_base = canonicalize_hermes_base_url(base_url)
            saved_base = canonicalize_hermes_base_url(saved_base_url)
        except ValueError as exc:
            return None, self._build_llm_channel_result(
                success=False,
                message="Saved secret scope mismatch",
                error=str(exc),
                stage=stage,
                error_code="saved_secret_scope_mismatch",
                retryable=False,
                details={"reason": "invalid_hermes_endpoint"},
                resolved_protocol=None,
                models=[] if stage == "model_discovery" else None,
                latency_ms=None,
                redaction_values=redaction_values,
            ), redaction_values

        if submitted_protocol != saved_protocol_canonical or submitted_base != saved_base:
            return None, self._build_llm_channel_result(
                success=False,
                message="Saved secret scope mismatch",
                error="Hermes endpoint changed; re-enter LLM_HERMES_API_KEY before testing",
                stage=stage,
                error_code="saved_secret_scope_mismatch",
                retryable=False,
                details={
                    "reason": "endpoint_mismatch",
                    "submitted_base_url": submitted_base,
                    "saved_base_url": saved_base,
                },
                resolved_protocol=submitted_protocol,
                models=[] if stage == "model_discovery" else None,
                latency_ms=None,
                redaction_values=redaction_values,
            ), redaction_values

        return saved_key, {}, redaction_values


    def _validate_hermes_submitted_secret(
        self,
        *,
        api_key: str,
        use_saved_secret: bool,
        stage: str,
        models: Optional[List[str]] = None,
        capability_checks: Sequence[str] = (),
        redaction_values: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Reject Hermes secret shapes that must not reach an outbound request."""

        secret = (api_key or "").strip()
        redactions = set(redaction_values or set())
        redactions.update(self._build_redaction_values(secret))
        if is_masked_secret_placeholder(secret):
            return self._build_llm_channel_result(
                success=False,
                message="Runtime Hermes secret is not reusable",
                error=(
                    "Runtime-injected Hermes secret is masked and cannot be reused by "
                    "test/discovery. Re-enter the key or save it to .env."
                ),
                stage=stage,
                error_code="runtime_secret_not_reusable",
                retryable=False,
                details={"reason": "runtime_secret_not_reusable"},
                resolved_protocol=None,
                models=models if stage == "model_discovery" else None,
                latency_ms=None,
                capability_results=(
                    self._build_skipped_capability_results(
                        capability_checks,
                        "base_test_failed",
                        "Skipped because the base channel test did not pass",
                        redaction_values=redactions,
                    )
                    if capability_checks
                    else None
                ),
                redaction_values=redactions,
            )
        if "," in secret:
            return self._build_llm_channel_result(
                success=False,
                message="Hermes API key is invalid",
                error="Hermes Phase 3 only supports a single LLM_HERMES_API_KEY",
                stage=stage,
                error_code="invalid_config",
                retryable=False,
                details={
                    "issue_key": "LLM_HERMES_API_KEY",
                    "issue_code": "multiple_api_keys",
                    "reason": "multiple_api_keys",
                },
                resolved_protocol=None,
                models=models if stage == "model_discovery" else None,
                latency_ms=None,
                capability_results=(
                    self._build_skipped_capability_results(
                        capability_checks,
                        "base_test_failed",
                        "Skipped because the base channel test did not pass",
                        redaction_values=redactions,
                    )
                    if capability_checks
                    else None
                ),
                redaction_values=redactions,
            )
        return None


    @staticmethod
    def _collect_hermes_channel_models_from_map(effective_map: Dict[str, str]) -> List[str]:
        """Collect valid reserved Hermes route aliases from channel-style env values."""
        from src.services.system_config.validation import SystemConfigValidationMixin
        raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
        if not raw_channels:
            return []

        models: List[str] = []
        seen: Set[str] = set()
        for raw_name in raw_channels.split(","):
            name = raw_name.strip()
            if not is_reserved_hermes_name(name):
                continue

            prefix = f"LLM_{name.upper()}"
            enabled = parse_env_bool(effective_map.get(f"{prefix}_ENABLED"), default=True)
            if not enabled:
                continue

            api_surface_value = (effective_map.get(f"{prefix}_API_SURFACE") or "").strip()
            if (
                not is_supported_llm_channel_api_surface_value(api_surface_value)
                or normalize_llm_channel_api_surface(api_surface_value) == "responses"
            ):
                continue
            raw_models = SystemConfigValidationMixin._split_csv(effective_map.get(f"{prefix}_MODELS") or "")
            result = parse_hermes_channel(
                enabled=True,
                protocol=(effective_map.get(f"{prefix}_PROTOCOL") or HERMES_DEFAULT_PROTOCOL).strip(),
                base_url=(effective_map.get(f"{prefix}_BASE_URL") or HERMES_DEFAULT_BASE_URL).strip(),
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
        return models


    @staticmethod
    def _collect_non_hermes_channel_models_from_map(effective_map: Dict[str, str]) -> List[str]:
        """Collect enabled non-Hermes channel route aliases from channel-style env values."""
        from src.services.system_config.validation import SystemConfigValidationMixin
        raw_channels = (effective_map.get("LLM_CHANNELS") or "").strip()
        if not raw_channels:
            return []
        models: List[str] = []
        seen: Set[str] = set()
        for raw_name in raw_channels.split(","):
            name = raw_name.strip()
            if not name or is_reserved_hermes_name(name):
                continue
            prefix = f"LLM_{name.upper()}"
            enabled_raw = effective_map.get(f"{prefix}_ENABLED")
            if name.lower() == "anspire" and not (enabled_raw or "").strip():
                enabled_raw = effective_map.get("ANSPIRE_LLM_ENABLED")
            if not parse_env_bool(enabled_raw, default=True):
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
            if not is_supported_llm_channel_api_surface_value(api_surface_value):
                continue
            raw_models = SystemConfigValidationMixin._split_csv(effective_map.get(f"{prefix}_MODELS") or "")
            if name.lower() == "anspire" and not raw_models:
                raw_models = [
                    (
                        effective_map.get("ANSPIRE_LLM_MODEL")
                        or ANSPIRE_LLM_MODEL_DEFAULT
                    ).strip()
                ]
            resolved_protocol = resolve_llm_channel_protocol(
                protocol_value,
                base_url=base_url_value,
                models=raw_models,
                channel_name=name,
            )
            if find_incompatible_llm_channel_models(
                raw_models,
                resolved_protocol,
                api_surface_value,
                base_url_value,
            ):
                continue
            for raw_model in raw_models:
                model = normalize_llm_channel_model(raw_model, resolved_protocol, base_url_value)
                if model and model not in seen:
                    seen.add(model)
                    models.append(model)
        return models


    @staticmethod
    def _collect_mixed_hermes_routes_from_map(effective_map: Dict[str, str]) -> Set[str]:
        hermes_routes = set(SystemConfigHermesMixin._collect_hermes_channel_models_from_map(effective_map))
        non_hermes_routes = set(SystemConfigHermesMixin._collect_non_hermes_channel_models_from_map(effective_map))
        return hermes_routes & non_hermes_routes


    @staticmethod
    def _matches_route_set(model: str, routes: Set[str]) -> bool:
        """Loose safety match for Hermes/provenance checks, not normal route availability."""
        return bool(route_identity_candidates(model) & set(routes or set()))


    @staticmethod
    def _matches_exact_route(model: str, routes: Set[str]) -> bool:
        """Match the Router's top-level model_name exactly for normal availability checks."""
        normalized_model = str(model or "").strip()
        return bool(normalized_model) and normalized_model in set(routes or set())
