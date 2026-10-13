# -*- coding: utf-8 -*-
"""配置 schema 与展示值归一（自 system_config_service 拆出，v1.1.1 重构）。

公开入口 ``src.services.system_config_service.SystemConfigService`` 通过多继承
聚合各 Mixin，方法经 MRO 解析，行为与拆分前完全一致。
"""

from __future__ import annotations

from src.core.config_registry import (
    build_schema_response,
    get_registered_field_keys,
)
from typing import (
    Any,
    Dict,
    Optional,
    Set,
)
import os


class SystemConfigSchemaMixin:
    """Mixin：见模块 docstring。"""

    def get_schema(self) -> Dict[str, Any]:
        """Return grouped schema metadata for UI rendering."""
        return build_schema_response()


    @staticmethod
    def _reload_runtime_singletons() -> None:
        """Reset runtime singleton services after config reload."""
        from src.agent.factory import reset_tool_registry
        from src.agent.tools.data_tools import reset_fetcher_manager
        from src.search_service import reset_search_service

        # Drop the module-level ToolRegistry so the next
        # ``build_agent_executor`` / ``build_agent_chat_executor`` call
        # rebuilds ``_TOOL_REGISTRY`` against the fresh Config and picks up
        # new ``AGENT_*_TOOL_TIMEOUT_S`` overrides (Issue #1890).
        reset_tool_registry()
        reset_fetcher_manager()
        reset_search_service()


    @classmethod
    def _normalize_display_value(cls, key: str, value: str) -> str:
        alias_map = cls._DISPLAY_VALUE_ALIASES.get(key.upper())
        if not alias_map:
            return value
        return alias_map.get(value.strip().lower(), value)


    @classmethod
    def _build_display_config_map(cls, raw_config_map: Dict[str, str]) -> Dict[str, str]:
        raw_upper = {key.upper(): value for key, value in raw_config_map.items()}
        aliased_keys = {
            alias
            for candidates in cls._DISPLAY_KEY_ALIASES.values()
            for alias in candidates
        }
        display_map: Dict[str, str] = {}

        for key, value in raw_upper.items():
            if key in aliased_keys:
                continue
            display_map[key] = cls._normalize_display_value(key, value)

        for canonical_key, candidates in cls._DISPLAY_KEY_ALIASES.items():
            canonical_env_key = candidates[0]
            if canonical_env_key in raw_upper:
                display_map[canonical_key] = cls._normalize_display_value(
                    canonical_key,
                    raw_upper[canonical_env_key],
                )
                continue

            selected_value: Optional[str] = None
            candidate_seen = False
            for candidate_key in candidates[1:]:
                if candidate_key not in raw_upper:
                    continue
                candidate_seen = True
                candidate_value = raw_upper[candidate_key]
                if candidate_value:
                    selected_value = candidate_value
                    break
            if candidate_seen:
                if selected_value is None:
                    for candidate_key in candidates[1:]:
                        if candidate_key in raw_upper:
                            selected_value = raw_upper[candidate_key]
                            break
                if selected_value is None:
                    selected_value = ""
                display_map[canonical_key] = cls._normalize_display_value(
                    canonical_key,
                    selected_value,
                )

        return display_map


    @staticmethod
    def _resolve_display_value(raw_value: str, field_schema: Dict[str, Any], raw_value_exists: bool) -> str:
        if raw_value_exists:
            return raw_value

        if field_schema.get("ui_control") == "switch" and raw_value:
            return raw_value

        if field_schema.get("ui_control") == "switch":
            default_value = field_schema.get("default_value")
            if isinstance(default_value, str) and default_value:
                return default_value

        return raw_value


    @classmethod
    def _get_schema_config_keys(cls, config_map: Dict[str, str], registered_keys: Set[str]) -> Set[str]:
        """Return keys needed by the Web schema payload.

        Ordinary settings must be registry-backed. LLM channel detail keys are
        kept only as editor support data for channels declared in LLM_CHANNELS.
        """
        keys = set(registered_keys)
        channel_names = {
            segment.strip().upper()
            for segment in config_map.get("LLM_CHANNELS", "").split(",")
            if segment.strip()
        }
        if not channel_names:
            return keys

        for key in config_map:
            match = cls._WEB_SETTINGS_LLM_CHANNEL_SUPPORT_KEY_RE.match(key)
            if match and match.group(1) in channel_names:
                keys.add(key)

        return keys


    @classmethod
    def _build_runtime_display_config_map(cls, saved_config_map: Dict[str, str]) -> Dict[str, str]:
        """Return Web settings values injected through the process environment.

        Docker ``env_file`` / ``--env-file`` only populate process environment
        variables; they do not create an active ``.env`` file inside the
        container. Use these values as display fallbacks so Settings can show
        startup-injected config without letting it override later WebUI saves.
        """
        registered_keys = {key.upper() for key in get_registered_field_keys()}
        channel_names = {
            segment.strip().upper()
            for raw_channels in (
                saved_config_map.get("LLM_CHANNELS", ""),
                os.environ.get("LLM_CHANNELS", ""),
            )
            for segment in raw_channels.split(",")
            if segment.strip()
        }
        runtime_map: Dict[str, str] = {}

        for raw_key, raw_value in os.environ.items():
            key = str(raw_key).upper()
            llm_channel_match = cls._WEB_SETTINGS_LLM_CHANNEL_SUPPORT_KEY_RE.match(key)
            if (
                key in registered_keys
                or (llm_channel_match and llm_channel_match.group(1) in channel_names)
            ):
                runtime_map[key] = "" if raw_value is None else str(raw_value)

        return cls._build_display_config_map(runtime_map)
