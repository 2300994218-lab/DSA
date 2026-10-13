# -*- coding: utf-8 -*-
"""System configuration service for `.env` based settings."""

from __future__ import annotations

from src.services.system_config import (
    SystemConfigSchemaMixin,
    SystemConfigImportExportMixin,
    SystemConfigHermesMixin,
    SystemConfigLLMMixin,
    SystemConfigUpdateMixin,
    SystemConfigValidationMixin,
    SystemConfigNotificationMixin,
    SystemConfigSetupMixin,
)
from src.services.system_config_errors import (  # re-export
    ConfigConflictError,
    ConfigImportError,
    ConfigValidationError,
)
import logging
from src.config import (
    Config,
    get_litellm_model_providers,
)
from src.core.config_manager import ConfigManager
from src.core.config_registry import (
    get_category_definitions,
    get_field_definition,
    get_registered_field_keys,
)
from src.notification_contracts import (
    FEISHU_APP_BOT_ENV_GROUP,
    FEISHU_WEBHOOK_ENV_GROUP,
)
from src.services.agent_backend_status_service import AgentBackendStatusService
from src.services.generation_backend_status_service import GenerationBackendStatusService
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import re

logger = logging.getLogger("src.services.system_config_service")


class SystemConfigService(SystemConfigSchemaMixin, SystemConfigImportExportMixin, SystemConfigHermesMixin, SystemConfigLLMMixin, SystemConfigUpdateMixin, SystemConfigValidationMixin, SystemConfigNotificationMixin, SystemConfigSetupMixin):
    """Service layer for reading, validating, and updating runtime configuration."""

    _GENERATION_BACKEND_STATUS_EXACT_KEYS = {
        "GENERATION_BACKEND",
        "GENERATION_FALLBACK_BACKEND",
        "GENERATION_BACKEND_TIMEOUT_SECONDS",
        "GENERATION_BACKEND_MAX_OUTPUT_BYTES",
        "GENERATION_BACKEND_MAX_CONCURRENCY",
        "LOCAL_CLI_BACKEND_MAX_CONCURRENCY",
        "OPENCODE_CLI_MODEL",
        "LITELLM_CONFIG",
        "LITELLM_MODEL",
        "LITELLM_FALLBACK_MODELS",
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "GEMINI_MODEL",
        "GEMINI_MODEL_FALLBACK",
        "GEMINI_TEMPERATURE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEYS",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_TEMPERATURE",
        "ANTHROPIC_MAX_TOKENS",
        "OPENAI_API_KEY",
        "OPENAI_API_KEYS",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_VISION_MODEL",
        "OPENAI_TEMPERATURE",
        "OLLAMA_API_BASE",
        "OLLAMA_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_API_KEYS",
        "AIHUBMIX_KEY",
        "ANSPIRE_LLM_ENABLED",
        "ANSPIRE_LLM_BASE_URL",
        "ANSPIRE_LLM_MODEL",
        "ANSPIRE_API_KEYS",
    }
    _GENERATION_BACKEND_STATUS_LLM_CHANNEL_RE = re.compile(
        r"^LLM_[A-Z0-9_]+_(PROTOCOL|API_SURFACE|BASE_URL|API_KEY|API_KEYS|MODELS|EXTRA_HEADERS|ENABLED)$"
    )
    _AGENT_BACKEND_STATUS_EXACT_KEYS = {
        "AGENT_BACKEND",
        "AGENT_GENERATION_BACKEND",
        "AGENT_LITELLM_MODEL",
        "AGENT_MODE",
        "AGENT_ARCH",
        "AGENT_ORCHESTRATOR_TIMEOUT_S",
    }

    _LLM_CAPABILITY_ORDER: Tuple[str, ...] = ("json", "tools", "stream", "vision")
    _LLM_STREAM_CHUNK_LIMIT = 8
    _WEB_SETTINGS_LLM_CHANNEL_SUPPORT_KEY_RE = re.compile(
        r"^LLM_([A-Z0-9_]+)_(PROTOCOL|API_SURFACE|BASE_URL|API_KEY|API_KEYS|MODELS|EXTRA_HEADERS|ENABLED)$"
    )
    _LLM_CAPABILITY_PROBE_IMAGE = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )

    _DISPLAY_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
        "AGENT_SKILL_DIR": ("AGENT_SKILL_DIR", "AGENT_STRATEGY_DIR"),
        "AGENT_SKILL_AUTOWEIGHT": ("AGENT_SKILL_AUTOWEIGHT", "AGENT_STRATEGY_AUTOWEIGHT"),
        "AGENT_SKILL_ROUTING": ("AGENT_SKILL_ROUTING", "AGENT_STRATEGY_ROUTING"),
    }
    _DISPLAY_VALUE_ALIASES: Dict[str, Dict[str, str]] = {
        "AGENT_ORCHESTRATOR_MODE": {
            "strategy": "specialist",
            "skill": "specialist",
        }
    }
    _SERVER_MASKED_CONFIG_KEYS: Set[str] = {
        "LLM_HERMES_API_KEY",
        "LLM_HERMES_API_KEYS",
        "LLM_HERMES_EXTRA_HEADERS",
        "LLM_USAGE_HMAC_SECRET",
    }
    _NOTIFICATION_TEST_CHANNELS: Tuple[str, ...] = (
        "wechat",
        "dingtalk",
        "feishu",
        "telegram",
        "email",
        "pushover",
        "ntfy",
        "gotify",
        "pushplus",
        "serverchan3",
        "custom",
        "discord",
        "slack",
        "astrbot",
    )
    _NOTIFICATION_TEST_KEY_MAP: Dict[str, Tuple[str, str]] = {
        "WECHAT_WEBHOOK_URL": ("wechat_webhook_url", "string"),
        "WECHAT_MSG_TYPE": ("wechat_msg_type", "string"),
        "WECHAT_MAX_BYTES": ("wechat_max_bytes", "int"),
        "FEISHU_WEBHOOK_URL": ("feishu_webhook_url", "string"),
        "FEISHU_WEBHOOK_SECRET": ("feishu_webhook_secret", "string"),
        "FEISHU_WEBHOOK_KEYWORD": ("feishu_webhook_keyword", "string"),
        "FEISHU_MAX_BYTES": ("feishu_max_bytes", "int"),
        "FEISHU_SEND_AS_FILE": ("feishu_send_as_file", "bool"),
        "DINGTALK_WEBHOOK_URL": ("dingtalk_webhook_url", "string"),
        "DINGTALK_SECRET": ("dingtalk_secret", "string"),
        "FEISHU_APP_ID": ("feishu_app_id", "string"),
        "FEISHU_APP_SECRET": ("feishu_app_secret", "string"),
        "FEISHU_CHAT_ID": ("feishu_chat_id", "string"),
        "FEISHU_RECEIVE_ID_TYPE": ("feishu_receive_id_type", "string"),
        "FEISHU_DOMAIN": ("feishu_domain", "string"),
        "TELEGRAM_BOT_TOKEN": ("telegram_bot_token", "string"),
        "TELEGRAM_CHAT_ID": ("telegram_chat_id", "string"),
        "TELEGRAM_MESSAGE_THREAD_ID": ("telegram_message_thread_id", "string"),
        "EMAIL_SENDER": ("email_sender", "string"),
        "EMAIL_SENDER_NAME": ("email_sender_name", "string"),
        "EMAIL_PASSWORD": ("email_password", "string"),
        "EMAIL_RECEIVERS": ("email_receivers", "csv"),
        "PUSHOVER_USER_KEY": ("pushover_user_key", "string"),
        "PUSHOVER_API_TOKEN": ("pushover_api_token", "string"),
        "NTFY_URL": ("ntfy_url", "string"),
        "NTFY_TOKEN": ("ntfy_token", "string"),
        "GOTIFY_URL": ("gotify_url", "string"),
        "GOTIFY_TOKEN": ("gotify_token", "string"),
        "PUSHPLUS_TOKEN": ("pushplus_token", "string"),
        "PUSHPLUS_TOPIC": ("pushplus_topic", "string"),
        "SERVERCHAN3_SENDKEY": ("serverchan3_sendkey", "string"),
        "CUSTOM_WEBHOOK_URLS": ("custom_webhook_urls", "csv"),
        "CUSTOM_WEBHOOK_BEARER_TOKEN": ("custom_webhook_bearer_token", "string"),
        "CUSTOM_WEBHOOK_BODY_TEMPLATE": ("custom_webhook_body_template", "string"),
        "WEBHOOK_VERIFY_SSL": ("webhook_verify_ssl", "bool"),
        "DISCORD_WEBHOOK_URL": ("discord_webhook_url", "string"),
        "DISCORD_BOT_TOKEN": ("discord_bot_token", "string"),
        "DISCORD_MAIN_CHANNEL_ID": ("discord_main_channel_id", "string"),
        "DISCORD_CHANNEL_ID": ("discord_main_channel_id", "string"),
        "DISCORD_MAX_WORDS": ("discord_max_words", "int"),
        "SLACK_WEBHOOK_URL": ("slack_webhook_url", "string"),
        "SLACK_BOT_TOKEN": ("slack_bot_token", "string"),
        "SLACK_CHANNEL_ID": ("slack_channel_id", "string"),
        "ASTRBOT_URL": ("astrbot_url", "string"),
        "ASTRBOT_TOKEN": ("astrbot_token", "string"),
    }
    _NOTIFICATION_REQUIRED_KEY_GROUPS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
        "wechat": (("WECHAT_WEBHOOK_URL",),),
        "dingtalk": (("DINGTALK_WEBHOOK_URL",),),
        "feishu": (FEISHU_WEBHOOK_ENV_GROUP, FEISHU_APP_BOT_ENV_GROUP),
        "telegram": (("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"),),
        "email": (("EMAIL_SENDER", "EMAIL_PASSWORD"),),
        "pushover": (("PUSHOVER_USER_KEY", "PUSHOVER_API_TOKEN"),),
        "ntfy": (("NTFY_URL",),),
        "gotify": (("GOTIFY_URL", "GOTIFY_TOKEN"),),
        "pushplus": (("PUSHPLUS_TOKEN",),),
        "serverchan3": (("SERVERCHAN3_SENDKEY",),),
        "custom": (("CUSTOM_WEBHOOK_URLS",),),
        "discord": (("DISCORD_WEBHOOK_URL",), ("DISCORD_BOT_TOKEN", "DISCORD_MAIN_CHANNEL_ID"), ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID")),
        "slack": (("SLACK_WEBHOOK_URL",), ("SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID")),
        "astrbot": (("ASTRBOT_URL",),),
    }
    _NOTIFICATION_TEST_TARGET_KEYS: Dict[str, Tuple[str, ...]] = {
        "wechat": ("WECHAT_WEBHOOK_URL",),
        "dingtalk": ("DINGTALK_WEBHOOK_URL",),
        "feishu": FEISHU_WEBHOOK_ENV_GROUP + FEISHU_APP_BOT_ENV_GROUP,
        "telegram": ("TELEGRAM_BOT_TOKEN",),
        "email": ("EMAIL_RECEIVERS", "EMAIL_SENDER"),
        "pushover": ("PUSHOVER_USER_KEY",),
        "ntfy": ("NTFY_URL",),
        "gotify": ("GOTIFY_URL",),
        "pushplus": ("PUSHPLUS_TOPIC",),
        "serverchan3": ("SERVERCHAN3_SENDKEY",),
        "custom": ("CUSTOM_WEBHOOK_URLS",),
        "discord": ("DISCORD_WEBHOOK_URL", "DISCORD_MAIN_CHANNEL_ID", "DISCORD_CHANNEL_ID"),
        "slack": ("SLACK_WEBHOOK_URL", "SLACK_CHANNEL_ID"),
        "astrbot": ("ASTRBOT_URL",),
    }

    def __init__(self, manager: Optional[ConfigManager] = None, runtime_scheduler: Optional[Any] = None):
        self._manager = manager or ConfigManager()
        self._runtime_scheduler = runtime_scheduler

    def get_config(self, include_schema: bool = True, mask_token: str = "******") -> Dict[str, Any]:
        """Return display config values with mask metadata for server-masked fields."""
        saved_config_map = self._build_display_config_map(self._manager.read_config_map())
        runtime_config_map = self._build_runtime_display_config_map(saved_config_map)
        config_map = {
            **runtime_config_map,
            **saved_config_map,
        }
        registered_keys = set(get_registered_field_keys())
        all_keys = set(config_map.keys()) | registered_keys
        if include_schema:
            all_keys = self._get_schema_config_keys(config_map, registered_keys)

        category_orders = {
            item["category"]: item["display_order"]
            for item in get_category_definitions()
        }

        schema_by_key: Dict[str, Dict[str, Any]] = {
            key: get_field_definition(key, config_map.get(key, ""))
            for key in all_keys
        }

        items: List[Dict[str, Any]] = []
        for key in all_keys:
            raw_value_exists = key in saved_config_map
            raw_value = config_map.get(key, "")
            field_schema = schema_by_key[key]
            display_value = self._resolve_display_value(raw_value, field_schema, raw_value_exists)
            is_masked = False
            if key in self._SERVER_MASKED_CONFIG_KEYS and display_value:
                display_value = mask_token
                is_masked = True
            item: Dict[str, Any] = {
                "key": key,
                "value": display_value,
                "raw_value_exists": raw_value_exists,
                "is_masked": is_masked,
            }
            if include_schema:
                item["schema"] = field_schema
            items.append(item)

        items.sort(
            key=lambda item: (
                category_orders.get(schema_by_key[item["key"]].get("category", "uncategorized"), 999),
                schema_by_key[item["key"]].get("display_order", 9999),
                item["key"],
            )
        )

        return {
            "config_version": self._manager.get_config_version(),
            "mask_token": mask_token,
            "items": items,
            "llm_model_providers": sorted(get_litellm_model_providers()),
            "updated_at": self._manager.get_updated_at(),
        }

    def validate(self, items: Sequence[Dict[str, str]], mask_token: str = "******") -> Dict[str, Any]:
        """Validate submitted items without writing to `.env`."""
        issues = self._collect_issues(items=items, mask_token=mask_token)
        valid = not any(issue["severity"] == "error" for issue in issues)
        return {
            "valid": valid,
            "issues": issues,
        }

    def test_notification_channel(
        self,
        *,
        channel: str,
        items: Sequence[Dict[str, str]],
        mask_token: str = "******",
        title: str = "DSA 通知测试",
        content: str = "这是一条来自 DSA Web 设置页的通知测试消息。",
        timeout_seconds: float = 20.0,
    ) -> Dict[str, Any]:
        """Send one real notification test without persisting submitted values."""
        normalized_channel = (channel or "").strip().lower()
        if normalized_channel not in self._NOTIFICATION_TEST_CHANNELS:
            raise ValueError(f"Unsupported notification channel: {channel}")

        effective_map = self._build_notification_test_effective_map(
            items=items,
            mask_token=mask_token,
        )
        missing = self._get_missing_notification_test_keys(normalized_channel, effective_map)
        if missing:
            return self._build_notification_test_result(
                success=False,
                message=f"通知渠道配置不完整，缺少: {', '.join(missing)}",
                error_code="config_missing",
                stage="config_validation",
                retryable=False,
                latency_ms=None,
                attempts=[],
            )
        invalid_message = self._get_invalid_notification_test_config_message(
            normalized_channel,
            effective_map,
        )
        if invalid_message:
            return self._build_notification_test_result(
                success=False,
                message=invalid_message,
                error_code="config_invalid",
                stage="config_validation",
                retryable=False,
                latency_ms=None,
                attempts=[],
            )

        config = self._build_notification_test_config(effective_map)
        try:
            return self._dispatch_notification_test(
                channel=normalized_channel,
                config=config,
                effective_map=effective_map,
                title=title.strip(),
                content=content.strip(),
                timeout_seconds=float(timeout_seconds),
            )
        except Exception as exc:
            logger.warning("Notification channel test failed for %s: %s", normalized_channel, exc)
            error_code, retryable = self._classify_notification_exception(exc)
            return self._build_notification_test_result(
                success=False,
                message=f"通知测试异常: {exc}",
                error_code=error_code,
                stage="notification_send",
                retryable=retryable,
                latency_ms=None,
                attempts=[
                    {
                        "channel": normalized_channel,
                        "success": False,
                        "message": str(exc),
                        "target": self._resolve_notification_test_target(normalized_channel, effective_map),
                        "error_code": error_code,
                        "stage": "notification_send",
                        "retryable": retryable,
                        "latency_ms": None,
                    }
                ],
            )

    def get_setup_status(self) -> Dict[str, Any]:
        """Return read-only first-run setup status without mutating runtime state."""
        effective_map = self._build_setup_effective_config_map()
        llm_check = self._build_setup_primary_llm_check(effective_map)
        agent_check = self._build_setup_agent_llm_check(effective_map, llm_check)
        checks = [
            llm_check,
            agent_check,
            self._build_setup_stock_list_check(effective_map),
            self._build_setup_notification_check(effective_map),
            self._build_setup_storage_check(effective_map),
        ]

        required_missing = [
            check["key"]
            for check in checks
            if check["required"] and check["status"] == "needs_action"
        ]
        smoke_blocking_missing = [
            check["key"]
            for check in checks
            if check["key"] in {"llm_primary", "stock_list"}
            and check["status"] == "needs_action"
        ]
        return {
            "is_complete": not required_missing,
            "ready_for_smoke": not smoke_blocking_missing,
            "required_missing_keys": required_missing,
            "next_step_key": required_missing[0] if required_missing else None,
            "checks": checks,
        }

    def get_generation_backend_status(self) -> Dict[str, Any]:
        """Return cheap generation backend status for saved/runtime config only."""
        effective_map = self._build_generation_backend_base_map()
        service = GenerationBackendStatusService(
            effective_map=effective_map,
            validation_issues=self._collect_generation_backend_issues_from_map(effective_map),
        )
        return service.get_status()

    def preview_generation_backend_status(
        self,
        *,
        items: Sequence[Dict[str, str]],
        mask_token: str = "******",
    ) -> Dict[str, Any]:
        """Return cheap generation backend status for unsaved settings draft."""
        issues = self._collect_generation_backend_issues(items=items, mask_token=mask_token)
        errors = [issue for issue in issues if issue["severity"] == "error"]
        if errors:
            raise ConfigValidationError(issues=errors)
        effective_map = self._build_generation_backend_effective_map(
            items=items,
            mask_token=mask_token,
        )
        service = GenerationBackendStatusService(
            effective_map=effective_map,
            validation_issues=issues,
        )
        return service.get_status()

    def test_generation_backend(
        self,
        *,
        backend_id: Optional[str] = None,
        mode: str = "json",
        items: Sequence[Dict[str, str]] = (),
        mask_token: str = "******",
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run an explicit generation backend smoke test without persisting config."""
        issues = self._collect_generation_backend_issues(items=items, mask_token=mask_token)
        errors = [issue for issue in issues if issue["severity"] == "error"]
        if errors:
            raise ConfigValidationError(issues=errors)
        effective_map = self._build_generation_backend_effective_map(
            items=items,
            mask_token=mask_token,
        )
        service = GenerationBackendStatusService(
            effective_map=effective_map,
            validation_issues=issues,
        )
        return service.smoke_test(
            backend_id=backend_id,
            mode=mode,
            timeout_seconds=timeout_seconds,
        )

    def get_agent_backend_status(self) -> Dict[str, Any]:
        """Return cheap Agent Chat backend status for saved/runtime config."""
        return AgentBackendStatusService(
            config=Config.get_instance(),
        ).get_status()

    def preview_agent_backend_status(
        self,
        *,
        items: Sequence[Dict[str, str]],
        mask_token: str = "******",
    ) -> Dict[str, Any]:
        """Return Agent Chat backend status for an unsaved settings draft."""
        filtered = self._filter_agent_backend_items(items)
        return AgentBackendStatusService(
            effective_map=self._build_agent_backend_effective_map(
                items=filtered,
                mask_token=mask_token,
            ),
        ).get_status()


# ============================================================
# v1.1.1 重构说明：SystemConfigService 的方法按职责拆分为
# src/services/system_config/ 下的各 Mixin（schema/import_export/hermes/llm/
# update/validation/notification/setup），异常类型位于
# src/services.system_config_errors 并由本模块再导出。
# 公开 API 与导入路径均保持不变。
# ============================================================
