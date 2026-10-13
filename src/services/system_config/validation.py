# -*- coding: utf-8 -*-
"""字段级与跨字段校验（自 system_config_service 拆出，v1.1.1 重构）。

公开入口 ``src.services.system_config_service.SystemConfigService`` 通过多继承
聚合各 Mixin，方法经 MRO 解析，行为与拆分前完全一致。
"""

from __future__ import annotations

from src.config import (
    parse_env_bool,
)
from src.core.config_registry import get_field_definition
from src.llm.backend_registry import (
    LITELLM_BACKEND_ID,
    normalize_backend_id,
)
from src.notification_contracts import (
    is_feishu_app_bot_env_configured,
    is_feishu_static_env_configured,
)
from src.notification_noise import validate_notification_timezone
from src.notification_sender.gotify_sender import resolve_gotify_message_endpoint
from src.notification_sender.ntfy_sender import resolve_ntfy_endpoint
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
from urllib.parse import urlparse
import json
import re


class SystemConfigValidationMixin:
    """Mixin：见模块 docstring。"""

    def _collect_issues(self, items: Sequence[Dict[str, str]], mask_token: str) -> List[Dict[str, Any]]:
        """Collect field-level and cross-field validation issues."""
        saved_config_map = self._manager.read_config_map()
        display_config_map = self._build_display_config_map(saved_config_map)
        runtime_config_map = self._build_runtime_display_config_map(display_config_map)
        effective_map = {
            **runtime_config_map,
            **display_config_map,
        }
        issues: List[Dict[str, Any]] = []
        updated_map: Dict[str, str] = {}

        for item in items:
            key = item["key"].upper()
            value = item["value"]
            field_schema = get_field_definition(key, value)
            is_sensitive = bool(field_schema.get("is_sensitive", False))

            if is_sensitive and value == mask_token and saved_config_map.get(key):
                continue

            updated_map[key] = value
            effective_map[key] = value
            issues.extend(self._validate_value(key=key, value=value, field_schema=field_schema))

        issues.extend(self._validate_cross_field(effective_map=effective_map, updated_keys=set(updated_map.keys())))
        return issues


    @classmethod
    def _is_generation_backend_status_key(cls, key: str) -> bool:
        normalized = str(key or "").strip().upper()
        return (
            normalized in cls._GENERATION_BACKEND_STATUS_EXACT_KEYS
            or normalized == "LLM_CHANNELS"
            or bool(cls._GENERATION_BACKEND_STATUS_LLM_CHANNEL_RE.fullmatch(normalized))
        )


    @classmethod
    def _filter_generation_backend_items(
        cls,
        items: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        filtered: List[Dict[str, str]] = []
        for item in items:
            key = str(item.get("key", "")).strip().upper()
            if not key or not cls._is_generation_backend_status_key(key):
                continue
            filtered.append({"key": key, "value": "" if item.get("value") is None else str(item.get("value"))})
        return filtered


    @classmethod
    def _filter_agent_backend_items(
        cls,
        items: Sequence[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        return [
            {
                "key": str(item.get("key", "")).strip().upper(),
                "value": "" if item.get("value") is None else str(item.get("value")),
            }
            for item in items
            if (
                str(item.get("key", "")).strip().upper() in cls._AGENT_BACKEND_STATUS_EXACT_KEYS
                or cls._is_generation_backend_status_key(
                    str(item.get("key", "")).strip().upper()
                )
            )
        ]


    def _collect_generation_backend_issues(
        self,
        *,
        items: Sequence[Dict[str, str]],
        mask_token: str,
    ) -> List[Dict[str, Any]]:
        """Collect only config issues that affect generation backend status/smoke."""
        issues = self._collect_issues(
            items=self._filter_generation_backend_items(items),
            mask_token=mask_token,
        )
        effective_map = self._build_generation_backend_effective_map(
            items=items,
            mask_token=mask_token,
        )
        issues.extend(self._validate_generation_backend_litellm_runtime_source(effective_map))
        return [
            issue for issue in issues
            if self._is_generation_backend_status_key(str(issue.get("key", "")))
        ]


    @staticmethod
    def _validate_generation_backend_litellm_runtime_source(effective_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """Validate explicit LiteLLM models when no route list can back them."""
        from src.services.system_config.llm import SystemConfigLLMMixin
        primary_backend = normalize_backend_id(
            effective_map.get("GENERATION_BACKEND"),
            default=LITELLM_BACKEND_ID,
        )
        fallback_backend = (
            LITELLM_BACKEND_ID
            if "GENERATION_FALLBACK_BACKEND" not in effective_map
            else (effective_map.get("GENERATION_FALLBACK_BACKEND") or "").strip().lower()
        )
        litellm_selected = (
            primary_backend == LITELLM_BACKEND_ID
            or (fallback_backend == LITELLM_BACKEND_ID and primary_backend != LITELLM_BACKEND_ID)
        )
        if not litellm_selected:
            return []
        if SystemConfigLLMMixin._uses_litellm_yaml(effective_map):
            return []
        if SystemConfigLLMMixin._collect_llm_channel_models_from_map(effective_map):
            return []
        if (effective_map.get("LLM_CHANNELS") or "").strip():
            return []

        issues: List[Dict[str, Any]] = []
        primary_model = (effective_map.get("LITELLM_MODEL") or "").strip()
        if primary_model and not SystemConfigLLMMixin._has_runtime_source_for_model(primary_model, effective_map):
            issues.append(
                {
                    "key": "LITELLM_MODEL",
                    "code": "missing_runtime_source",
                    "message": (
                        "A primary model is selected, but no usable runtime source was found. "
                        "Configure a matching provider API key, LLM channel, or LiteLLM YAML route."
                    ),
                    "severity": "error",
                    "expected": "matching provider API key, enabled channel model, or YAML model",
                    "actual": primary_model,
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
                        "Some fallback models do not have a matching provider API key, "
                        "enabled channel, or LiteLLM YAML route."
                    ),
                    "severity": "error",
                    "expected": "matching provider API key, enabled channel model, or YAML model",
                    "actual": ", ".join(invalid_fallbacks[:3]),
                }
            )
        return issues


    def _collect_generation_backend_issues_from_map(
        self,
        effective_map: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        items = [
            {"key": key, "value": value}
            for key, value in effective_map.items()
            if self._is_generation_backend_status_key(key)
        ]
        return self._collect_generation_backend_issues(items=items, mask_token="******")


    @staticmethod
    def _validate_value(key: str, value: str, field_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate a single field value against schema metadata."""
        issues: List[Dict[str, Any]] = []
        data_type = field_schema.get("data_type", "string")
        validation = field_schema.get("validation", {}) or {}
        is_required = field_schema.get("is_required", False)

        # Empty values are valid for non-required fields (skip type validation)
        if not value.strip() and not is_required:
            return issues

        if ("\n" in value or "\r" in value) and data_type != "json":
            issues.append(
                {
                    "key": key,
                    "code": "invalid_value",
                    "message": "Value cannot contain newline characters",
                    "severity": "error",
                    "expected": "single-line value",
                    "actual": "contains newline",
                }
            )
            return issues

        if data_type == "integer":
            try:
                numeric = int(value)
            except ValueError:
                return [
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be an integer",
                        "severity": "error",
                        "expected": "integer",
                        "actual": value,
                    }
                ]
            issues.extend(SystemConfigValidationMixin._validate_numeric_range(key, numeric, validation))

        elif data_type == "number":
            try:
                numeric = float(value)
            except ValueError:
                return [
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be a number",
                        "severity": "error",
                        "expected": "number",
                        "actual": value,
                    }
                ]
            issues.extend(SystemConfigValidationMixin._validate_numeric_range(key, numeric, validation))

        elif data_type == "boolean":
            if value.strip().lower() not in {"true", "false"}:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_type",
                        "message": "Value must be true or false",
                        "severity": "error",
                        "expected": "true|false",
                        "actual": value,
                    }
                )

        elif data_type == "time":
            pattern = validation.get("pattern") or r"^([01]\d|2[0-3]):[0-5]\d$"
            if not re.match(pattern, value.strip()):
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_format",
                        "message": "Value must be in HH:MM format",
                        "severity": "error",
                        "expected": "HH:MM",
                        "actual": value,
                    }
                )

        elif data_type == "json":
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_json",
                        "message": "Value must be valid JSON",
                        "severity": "error",
                        "expected": "valid JSON",
                        "actual": value[:120],
                    }
                )
            else:
                if key == "AGENT_EVENT_ALERT_RULES_JSON":
                    try:
                        from src.agent.events import parse_event_alert_rules, validate_event_alert_rule

                        rule_index = 0
                        for rule_index, rule in enumerate(parse_event_alert_rules(parsed), start=1):
                            validate_event_alert_rule(rule)
                    except ValueError as exc:
                        issues.append(
                            {
                                "key": key,
                                "code": "invalid_event_rule",
                                "message": f"Rule validation failed: {exc}",
                                "severity": "error",
                                "expected": "supported EventMonitor rule fields and enum values",
                                "actual": f"rule #{rule_index or 1}",
                            }
                        )

        elif validation.get("pattern"):
            pattern = validation["pattern"]
            if not re.match(pattern, value.strip()):
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_format",
                        "message": "Value does not match the required format",
                        "severity": "error",
                        "expected": pattern,
                        "actual": value,
                    }
                )

        if validation.get("timezone") and value:
            try:
                validate_notification_timezone(value)
            except ValueError as exc:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_timezone",
                        "message": str(exc),
                        "severity": "error",
                        "expected": "valid IANA timezone or empty",
                        "actual": value,
                    }
                )

        if "enum" in validation and value and value not in validation["enum"]:
            issues.append(
                {
                    "key": key,
                    "code": "invalid_enum",
                    "message": "Value is not in allowed options",
                    "severity": "error",
                    "expected": ",".join(validation["enum"]),
                    "actual": value,
                }
            )

        if "allowed_values" in validation and value:
            delimiter = validation.get("delimiter")
            raw_values = value.split(delimiter) if delimiter else [value]
            allowed_values = {str(item).strip().lower() for item in validation["allowed_values"]}
            invalid_values = []
            seen_invalid = set()
            for raw_item in raw_values:
                item = raw_item.strip().lower()
                if not item:
                    continue
                if item not in allowed_values and item not in seen_invalid:
                    invalid_values.append(item)
                    seen_invalid.add(item)
            if invalid_values:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_allowed_value",
                        "message": "Value contains unsupported item(s)",
                        "severity": "error",
                        "expected": ",".join(str(item) for item in validation["allowed_values"]),
                        "actual": ", ".join(invalid_values),
                    }
                )

        if validation.get("item_type") == "url":
            delimiter = validation.get("delimiter", ",")
            values = [item.strip() for item in value.split(delimiter)] if validation.get("multi_value") else [value.strip()]
            allowed_schemes = tuple(validation.get("allowed_schemes", ["http", "https"]))
            invalid_values = [
                item for item in values
                if item and not SystemConfigValidationMixin._is_valid_url(item, allowed_schemes=allowed_schemes)
            ]
            if invalid_values:
                issues.append(
                    {
                        "key": key,
                        "code": "invalid_url",
                        "message": "Value must contain valid URLs with scheme and host",
                        "severity": "error",
                        "expected": ",".join(allowed_schemes) + " URL(s)",
                        "actual": ", ".join(invalid_values[:3]),
                    }
                )

        if key == "NTFY_URL" and value.strip():
            allowed_schemes = tuple(validation.get("allowed_schemes", ["http", "https"]))
            if SystemConfigValidationMixin._is_valid_url(value.strip(), allowed_schemes=allowed_schemes):
                ntfy_server_url, ntfy_topic = resolve_ntfy_endpoint(value)
                if not ntfy_server_url or not ntfy_topic:
                    issues.append(
                        {
                            "key": key,
                            "code": "invalid_ntfy_url",
                            "message": "NTFY_URL must include a topic path, e.g. https://ntfy.sh/my-topic",
                            "severity": "error",
                            "expected": "ntfy publish endpoint with topic path",
                            "actual": value,
                        }
                    )

        if key == "GOTIFY_URL" and value.strip():
            allowed_schemes = tuple(validation.get("allowed_schemes", ["http", "https"]))
            if SystemConfigValidationMixin._is_valid_url(value.strip(), allowed_schemes=allowed_schemes):
                gotify_endpoint = resolve_gotify_message_endpoint(value)
                if not gotify_endpoint:
                    issues.append(
                        {
                            "key": key,
                            "code": "invalid_gotify_url",
                            "message": "GOTIFY_URL must be a Gotify server base URL and must not include /message",
                            "severity": "error",
                            "expected": "Gotify server base URL, e.g. https://gotify.example",
                            "actual": value,
                        }
                    )

        return issues


    @staticmethod
    def _normalize_value_for_storage(value: str, field_schema: Dict[str, Any]) -> str:
        """Normalize submitted values before persisting to the single-line .env file."""
        if field_schema.get("data_type", "string") != "json":
            return value

        if not value.strip():
            return value

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value

        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


    @staticmethod
    def _validate_numeric_range(key: str, numeric_value: float, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        min_value = validation.get("min")
        max_value = validation.get("max")

        if min_value is not None and numeric_value < min_value:
            issues.append(
                {
                    "key": key,
                    "code": "out_of_range",
                    "message": "Value is lower than minimum",
                    "severity": "error",
                    "expected": f">={min_value}",
                    "actual": str(numeric_value),
                }
            )
        if max_value is not None and numeric_value > max_value:
            issues.append(
                {
                    "key": key,
                    "code": "out_of_range",
                    "message": "Value is greater than maximum",
                    "severity": "error",
                    "expected": f"<={max_value}",
                    "actual": str(numeric_value),
                }
            )
        return issues


    @staticmethod
    def _is_valid_url(value: str, allowed_schemes: Tuple[str, ...]) -> bool:
        """Return True when *value* looks like a valid absolute URL."""
        parsed = urlparse(value)
        return parsed.scheme in allowed_schemes and bool(parsed.netloc)


    @staticmethod
    def _canonical_ipv4_numeric_host(host: str) -> Optional[str]:
        """Return canonical IPv4 for libc-style numeric host aliases."""
        import socket

        candidate = (host or "").lower()
        if not candidate or ":" in candidate:
            return None

        try:
            return socket.inet_ntoa(socket.inet_aton(candidate))
        except (OSError, ValueError):
            return None


    @staticmethod
    def _is_noncanonical_ipv4_numeric_host(host: str) -> bool:
        canonical = SystemConfigValidationMixin._canonical_ipv4_numeric_host(host)
        return canonical is not None and host.lower() != canonical


    @staticmethod
    def _normalize_hostname_for_security(host: str) -> Optional[str]:
        """Return a normalized ASCII host for URL safety checks."""
        import unicodedata

        candidate = (host or "").strip().lower().rstrip(".")
        if not candidate:
            return None
        if ":" in candidate:
            return candidate
        try:
            normalized = unicodedata.normalize("NFKC", candidate)
            ascii_host = normalized.encode("idna").decode("ascii").lower().rstrip(".")
        except UnicodeError:
            return None
        return ascii_host or None


    @staticmethod
    def _is_valid_llm_base_url(value: str, allowed_schemes: Tuple[str, ...] = ("http", "https")) -> bool:
        """Return True when an LLM base URL is safe to parse consistently."""
        if not value:
            return False
        if any(char == "\\" or char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value):
            return False

        try:
            parsed = urlparse(value)
            host = parsed.hostname
            _ = parsed.port
        except ValueError:
            return False

        if parsed.scheme not in allowed_schemes or not parsed.netloc or not host:
            return False
        if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None:
            return False
        if SystemConfigValidationMixin._is_noncanonical_ipv4_numeric_host(host):
            return False

        return True


    @staticmethod
    def _split_csv(value: str) -> List[str]:
        return [item.strip() for item in (value or "").split(",") if item.strip()]


    @staticmethod
    def _validate_cross_field(effective_map: Dict[str, str], updated_keys: Set[str]) -> List[Dict[str, Any]]:
        """Validate dependencies across multiple keys."""
        from src.services.system_config.llm import SystemConfigLLMMixin
        issues: List[Dict[str, Any]] = []

        agent_backend = (effective_map.get("AGENT_BACKEND") or "auto").strip().lower()
        agent_arch = (effective_map.get("AGENT_ARCH") or "single").strip().lower()
        if agent_backend == "codex_app_server" and agent_arch != "single" and (
            {"AGENT_BACKEND", "AGENT_ARCH"} & updated_keys
        ):
            issues.append(
                {
                    "key": "AGENT_ARCH",
                    "code": "unsupported_agent_arch",
                    "message": "Codex 本地 Agent 当前只支持单 Agent 问股，请切换为 single。",
                    "severity": "error",
                    "expected": "single",
                    "actual": agent_arch,
                }
            )

        timeout_raw = (effective_map.get("AGENT_ORCHESTRATOR_TIMEOUT_S") or "600").strip()
        try:
            agent_timeout = int(timeout_raw)
        except ValueError:
            agent_timeout = None
        if agent_backend == "codex_app_server" and agent_timeout is not None and agent_timeout <= 0 and (
            {"AGENT_BACKEND", "AGENT_ORCHESTRATOR_TIMEOUT_S"} & updated_keys
        ):
            issues.append(
                {
                    "key": "AGENT_ORCHESTRATOR_TIMEOUT_S",
                    "code": "codex_timeout_required",
                    "message": "Codex 本地 Agent 必须设置大于 0 的整体时限，确保每次问股都会结束。",
                    "severity": "error",
                    "expected": ">0 when AGENT_BACKEND=codex_app_server",
                    "actual": timeout_raw,
                }
            )

        token_value = (effective_map.get("TELEGRAM_BOT_TOKEN") or "").strip()
        chat_id_value = (effective_map.get("TELEGRAM_CHAT_ID") or "").strip()
        if token_value and not chat_id_value and (
            "TELEGRAM_BOT_TOKEN" in updated_keys or "TELEGRAM_CHAT_ID" in updated_keys
        ):
            issues.append(
                {
                    "key": "TELEGRAM_CHAT_ID",
                    "code": "missing_dependency",
                    "message": "TELEGRAM_CHAT_ID is required when TELEGRAM_BOT_TOKEN is set",
                    "severity": "error",
                    "expected": "non-empty TELEGRAM_CHAT_ID",
                    "actual": chat_id_value,
                }
            )

        feishu_relevant_keys = {
            "FEISHU_APP_ID",
            "FEISHU_APP_SECRET",
            "FEISHU_WEBHOOK_URL",
            "FEISHU_WEBHOOK_SECRET",
            "FEISHU_WEBHOOK_KEYWORD",
            "FEISHU_STREAM_ENABLED",
            "FEISHU_FOLDER_TOKEN",
            "FEISHU_CHAT_ID",
        }
        has_feishu_app_id = bool((effective_map.get("FEISHU_APP_ID") or "").strip())
        has_feishu_app_secret = bool((effective_map.get("FEISHU_APP_SECRET") or "").strip())
        has_feishu_app_credentials_complete = has_feishu_app_id and has_feishu_app_secret
        has_feishu_app_credentials = has_feishu_app_id or has_feishu_app_secret
        has_feishu_folder_token = bool((effective_map.get("FEISHU_FOLDER_TOKEN") or "").strip())
        has_feishu_full_cloud_doc_credentials = (
            has_feishu_app_credentials_complete
            and has_feishu_folder_token
        )
        # Match runtime semantics: Config.from_env only enables stream mode
        # when the value is exactly "true" (case-insensitive).
        feishu_stream_enabled = (
            (effective_map.get("FEISHU_STREAM_ENABLED") or "false")
            .strip()
            .lower()
            == "true"
        )
        has_feishu_stream_route = feishu_stream_enabled and has_feishu_app_credentials_complete
        has_feishu_app_bot_route = is_feishu_app_bot_env_configured(effective_map)
        if (
            has_feishu_app_credentials
            and not has_feishu_full_cloud_doc_credentials
            and not is_feishu_static_env_configured(effective_map)
            and not has_feishu_stream_route
            and not has_feishu_app_bot_route
            and (updated_keys & feishu_relevant_keys)
        ):
            issues.append(
                {
                    "key": "FEISHU_CHAT_ID",
                    "code": "feishu_mode_mismatch",
                    "message": (
                        "仅配置 FEISHU_APP_ID / FEISHU_APP_SECRET 不会开启飞书静态通知；"
                        "App Bot 主动推送需要同时配置 FEISHU_CHAT_ID，"
                        "Webhook 推送请填写 FEISHU_WEBHOOK_URL；"
                        "事件订阅请使用 FEISHU_STREAM_ENABLED=true 并完成应用发布与权限配置。"
                    ),
                    "severity": "warning",
                    "expected": (
                        "static notification: FEISHU_WEBHOOK_URL or "
                        "FEISHU_APP_ID + FEISHU_APP_SECRET + FEISHU_CHAT_ID; "
                        "event subscription: FEISHU_STREAM_ENABLED=true"
                    ),
                    "actual": "app credentials without notification target",
                }
            )

        issues.extend(
            SystemConfigLLMMixin._validate_llm_channel_map(
                effective_map=effective_map,
                updated_keys=updated_keys,
            )
        )
        issues.extend(SystemConfigLLMMixin._validate_llm_runtime_selection(effective_map=effective_map))

        if parse_env_bool(effective_map.get("NOTIFICATION_DAILY_DIGEST_ENABLED"), default=False):
            issues.append(
                {
                    "key": "NOTIFICATION_DAILY_DIGEST_ENABLED",
                    "code": "reserved_notification_daily_digest",
                    "message": (
                        "NOTIFICATION_DAILY_DIGEST_ENABLED is reserved; "
                        "the current P4 implementation does not send daily digests."
                    ),
                    "severity": "warning",
                    "expected": "reserved flag only",
                    "actual": effective_map.get("NOTIFICATION_DAILY_DIGEST_ENABLED", ""),
                }
            )

        return issues
